use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use anyhow::Result;
use async_stream::try_stream;
use futures::{stream, Stream, StreamExt, TryStreamExt};
use tokio::sync::Mutex;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;
use tracing::{error, info, warn};
use uuid::Uuid;

use crate::agents::agent::{Agent, AgentEvent, ToolStreamItem};
use crate::agents::final_output_tool::FINAL_OUTPUT_CONTINUATION_MESSAGE;
use crate::agents::platform_extensions::MANAGE_EXTENSIONS_TOOL_NAME_COMPLETE;
use crate::agents::tool_execution::CHAT_MODE_TOOL_SKIPPED_RESPONSE;
use crate::agents::types::SessionConfig;
use crate::config::GooseMode;
use crate::context_mgmt::compact_messages;
use crate::conversation::message::{
    Message, MessageContent, ProviderMetadata, SystemNotificationType, ToolRequest,
};
use crate::conversation::Conversation;
use crate::permission::permission_judge::PermissionCheckResult;
use crate::providers::errors::ProviderError;
use crate::session::{Session, SessionManager};
use crate::utils::is_token_cancelled;
use rmcp::model::Role;
use rmcp::model::{CallToolResult, Content, CustomNotification, ServerNotification, Tool};

pub(crate) enum TurnGuard {
    Proceed,
    Emit(AgentEvent),
    Stop,
}

pub(crate) enum ProviderResponse {
    TextOnly {
        response: Message,
    },
    WithToolCalls {
        filtered_response: Message,
        plan: ToolExecutionPlan,
    },
}

pub(crate) struct ToolExecutionPlan {
    pub frontend_requests: Vec<ToolRequest>,
    pub backend_requests: Vec<ToolRequest>,
    pub thinking_content: Vec<MessageContent>,
    pub reasoning_content: Vec<MessageContent>,
    pub response_role: Role,
    pub response_created: i64,
}

pub(crate) enum ToolEvent {
    Passthrough(AgentEvent),
    Done(ToolExecutionOutcome),
}

pub(crate) struct ToolExecutionOutcome {
    pub tool_messages: Vec<(Message, Message)>,
    pub thinking_msg: Option<Message>,
    pub extension_installed: bool,
    pub had_parse_error: bool,
}

pub(crate) enum ErrorAction {
    Compacted { events: Vec<AgentEvent> },
    Break(AgentEvent),
}

pub(crate) enum TurnResult {
    TextOnly,
    ToolsExecuted { extension_installed: bool },
    Compacted,
}

pub(crate) enum TurnOutcome {
    Continue,
    Retry,
    Exit,
}

pub(crate) struct TurnContext<'a> {
    pub session_config: &'a SessionConfig,
    pub session: &'a Session,
    pub cancel_token: &'a Option<CancellationToken>,
    pub goose_mode: GooseMode,
    pub initial_messages: &'a [Message],
}

pub(crate) struct ReplyState {
    pub tools: Vec<Tool>,
    pub toolshim_tools: Vec<Tool>,
    pub system_prompt: String,
}

impl ReplyState {
    pub async fn refresh(
        &mut self,
        agent: &Agent,
        session_id: &str,
        working_dir: &std::path::Path,
    ) -> Result<()> {
        let (tools, toolshim_tools, system_prompt) = agent
            .prepare_tools_and_prompt(session_id, working_dir)
            .await?;
        self.tools = tools;
        self.toolshim_tools = toolshim_tools;
        self.system_prompt = system_prompt;
        Ok(())
    }
}

pub(crate) const COMPACTION_THINKING_TEXT: &str = "goose is compacting the conversation...";

fn extract_platform_notification(call_result: &CallToolResult) -> Option<ServerNotification> {
    let meta = call_result.meta.as_ref()?;
    let notification_data = meta.0.get("platform_notification")?;
    let method = notification_data.get("method").and_then(|v| v.as_str())?;
    let params = notification_data.get("params").cloned();
    Some(ServerNotification::CustomNotification(
        CustomNotification::new(method.to_string(), params),
    ))
}

impl Agent {
    pub(crate) async fn check_turn_guard(
        &self,
        turns_taken: u32,
        max_turns: u32,
        ctx: &TurnContext<'_>,
    ) -> TurnGuard {
        if is_token_cancelled(ctx.cancel_token) {
            return TurnGuard::Stop;
        }

        {
            let guard = self.final_output_tool.lock().await;
            if let Some(output) = guard.as_ref().and_then(|fot| fot.final_output.clone()) {
                return TurnGuard::Emit(AgentEvent::Message(
                    Message::assistant().with_text(output),
                ));
            }
        }

        if turns_taken > max_turns {
            return TurnGuard::Emit(AgentEvent::Message(
                Message::assistant().with_text(
                    "I've reached the maximum number of actions I can do without user input. Would you like me to continue?"
                ),
            ));
        }

        TurnGuard::Proceed
    }

    pub(crate) async fn handle_provider_error(
        &self,
        error: &ProviderError,
        conversation: &mut Conversation,
        ctx: &TurnContext<'_>,
        compaction_attempts: &mut u32,
    ) -> Result<ErrorAction> {
        crate::posthog::emit_error(error.telemetry_type(), &error.to_string());

        match error {
            ProviderError::ContextLengthExceeded(_) => {
                *compaction_attempts += 1;

                if *compaction_attempts >= 2 {
                    error!("Context limit exceeded after compaction - prompt too large");
                    return Ok(ErrorAction::Break(AgentEvent::Message(
                        Message::assistant().with_system_notification(
                            SystemNotificationType::InlineMessage,
                            "Unable to continue: Context limit still exceeded after compaction. Try using a shorter message, a model with a larger context window, or start a new session."
                        ),
                    )));
                }

                let mut events = vec![
                    AgentEvent::Message(Message::assistant().with_system_notification(
                        SystemNotificationType::InlineMessage,
                        "Context limit reached. Compacting to continue conversation...",
                    )),
                    AgentEvent::Message(Message::assistant().with_system_notification(
                        SystemNotificationType::ThinkingMessage,
                        COMPACTION_THINKING_TEXT,
                    )),
                ];

                match compact_messages(
                    self.provider().await?.as_ref(),
                    &ctx.session_config.id,
                    conversation,
                    false,
                )
                .await
                {
                    Ok((compacted_conversation, usage)) => {
                        let session_manager = self.config.session_manager.clone();
                        session_manager
                            .replace_conversation(&ctx.session_config.id, &compacted_conversation)
                            .await?;
                        self.update_session_metrics(
                            &ctx.session_config.id,
                            ctx.session_config.schedule_id.clone(),
                            &usage,
                            true,
                        )
                        .await?;
                        *conversation = compacted_conversation;
                        events.push(AgentEvent::HistoryReplaced(conversation.clone()));
                        Ok(ErrorAction::Compacted { events })
                    }
                    Err(e) => {
                        crate::posthog::emit_error("compaction_failed", &e.to_string());
                        error!("Compaction failed: {}", e);
                        Ok(ErrorAction::Break(AgentEvent::Message(
                            Message::assistant().with_text(
                                format!("Ran into this error trying to compact: {e}.\n\nPlease try again or create a new session"),
                            ),
                        )))
                    }
                }
            }
            ProviderError::CreditsExhausted {
                details: _,
                ref top_up_url,
            } => {
                error!("Error: {}", error);
                let user_msg = if top_up_url.is_some() {
                    "Please add credits to your account, then resend your message to continue."
                        .to_string()
                } else {
                    "Please check your account with your provider to add more credits, then resend your message to continue."
                        .to_string()
                };
                let notification_data = serde_json::json!({ "top_up_url": top_up_url });
                Ok(ErrorAction::Break(AgentEvent::Message(
                    Message::assistant().with_system_notification_with_data(
                        SystemNotificationType::CreditsExhausted,
                        user_msg,
                        notification_data,
                    ),
                )))
            }
            ProviderError::NetworkError(_) => {
                error!("Error: {}", error);
                Ok(ErrorAction::Break(AgentEvent::Message(
                    Message::assistant().with_text(format!(
                        "{error}\n\nPlease resend your message to try again."
                    )),
                )))
            }
            other => {
                error!("Error: {}", other);
                Ok(ErrorAction::Break(AgentEvent::Message(
                    Message::assistant().with_text(format!(
                        "Ran into this error: {other}.\n\nPlease retry if you think this is a transient or recoverable error."
                    )),
                )))
            }
        }
    }

    pub(crate) async fn turn_epilogue(
        &self,
        conversation: &mut Conversation,
        reply_state: &mut ReplyState,
        ctx: &TurnContext<'_>,
        turn_result: &TurnResult,
        summarization_task: JoinHandle<Option<(Message, String)>>,
    ) -> Result<(TurnOutcome, Vec<AgentEvent>)> {
        let session_manager = self.config.session_manager.clone();
        let session_config = ctx.session_config;
        let working_dir = &ctx.session.working_dir;
        let mut events = Vec::new();

        let has_new_hints = self
            .prompt_manager
            .lock()
            .await
            .load_subdirectory_hints(working_dir);
        let tools_updated = matches!(
            turn_result,
            TurnResult::ToolsExecuted {
                extension_installed: true
            }
        );
        if tools_updated || has_new_hints {
            reply_state
                .refresh(self, &session_config.id, working_dir)
                .await?;
        }

        match turn_result {
            TurnResult::ToolsExecuted { .. } => {
                if let Ok(Some((summary_msg, tool_id))) = summarization_task.await {
                    let mut updated_messages = conversation.messages().clone();

                    let matching: Vec<&mut Message> = updated_messages
                        .iter_mut()
                        .filter(|msg| {
                            msg.id.is_some()
                                && msg.content.iter().any(|c| match c {
                                    MessageContent::ToolRequest(req) => req.id == tool_id,
                                    MessageContent::ToolResponse(resp) => resp.id == tool_id,
                                    _ => false,
                                })
                        })
                        .collect();

                    if matching.len() == 2 {
                        for msg in matching {
                            let id = msg.id.as_ref().unwrap();
                            msg.metadata = msg.metadata.with_agent_invisible();
                            SessionManager::update_message_metadata(
                                &session_config.id,
                                id,
                                |metadata| metadata.with_agent_invisible(),
                            )
                            .await?;
                        }
                        *conversation = Conversation::new_unvalidated(updated_messages);
                        session_manager
                            .add_message(&session_config.id, &summary_msg)
                            .await?;
                        conversation.push(summary_msg);
                    } else {
                        warn!(
                            "Expected a tool request/reply pair, but found {} matching messages",
                            matching.len()
                        );
                    }
                }

                Ok((TurnOutcome::Continue, events))
            }
            TurnResult::Compacted => {
                summarization_task.abort();
                Ok((TurnOutcome::Continue, events))
            }
            TurnResult::TextOnly => {
                summarization_task.abort();

                // Lock, extract state, drop guard before branching -- handle_retry_logic
                // also locks final_output_tool and tokio::sync::Mutex is not reentrant.
                let final_output = {
                    let guard = self.final_output_tool.lock().await;
                    guard.as_ref().map(|fot| fot.final_output.clone())
                };

                match final_output {
                    Some(None) => {
                        warn!(
                            "Final output tool has not been called yet. Continuing agent loop."
                        );
                        let message =
                            Message::user().with_text(FINAL_OUTPUT_CONTINUATION_MESSAGE);
                        session_manager
                            .add_message(&session_config.id, &message)
                            .await?;
                        conversation.push(message.clone());
                        events.push(AgentEvent::Message(message));
                        Ok((TurnOutcome::Continue, events))
                    }
                    Some(Some(output)) => {
                        let message = Message::assistant().with_text(output);
                        session_manager
                            .add_message(&session_config.id, &message)
                            .await?;
                        conversation.push(message.clone());
                        events.push(AgentEvent::Message(message));
                        Ok((TurnOutcome::Exit, events))
                    }
                    None => {
                        match self
                            .handle_retry_logic(
                                conversation,
                                session_config,
                                ctx.initial_messages,
                            )
                            .await
                        {
                            Ok(should_retry) => {
                                if should_retry {
                                    info!("Retry logic triggered, restarting agent loop");
                                    session_manager
                                        .replace_conversation(
                                            &session_config.id,
                                            conversation,
                                        )
                                        .await?;
                                    events.push(AgentEvent::HistoryReplaced(
                                        conversation.clone(),
                                    ));
                                    Ok((TurnOutcome::Retry, events))
                                } else {
                                    Ok((TurnOutcome::Exit, events))
                                }
                            }
                            Err(e) => {
                                error!("Retry logic failed: {}", e);
                                events.push(AgentEvent::Message(
                                    Message::assistant().with_text(format!(
                                        "Retry logic encountered an error: {}", e
                                    )),
                                ));
                                Ok((TurnOutcome::Exit, events))
                            }
                        }
                    }
                }
            }
        }
    }

    pub(crate) async fn classify_provider_response(
        &self,
        response: Message,
        tools: &[Tool],
    ) -> ProviderResponse {
        let categorized = self.categorize_tools(&response, tools).await;
        let has_tool_calls =
            !categorized.frontend_requests.is_empty() || !categorized.remaining_requests.is_empty();

        if has_tool_calls {
            let thinking_content: Vec<MessageContent> = response
                .content
                .iter()
                .filter(|c| matches!(c, MessageContent::Thinking(_)))
                .cloned()
                .collect();
            let reasoning_content: Vec<MessageContent> = response
                .content
                .iter()
                .filter(|c| matches!(c, MessageContent::Reasoning(_)))
                .cloned()
                .collect();
            ProviderResponse::WithToolCalls {
                filtered_response: categorized.filtered_response,
                plan: ToolExecutionPlan {
                    frontend_requests: categorized.frontend_requests,
                    backend_requests: categorized.remaining_requests,
                    thinking_content,
                    reasoning_content,
                    response_role: response.role,
                    response_created: response.created,
                },
            }
        } else {
            ProviderResponse::TextOnly { response }
        }
    }

    pub(crate) fn execute_tools<'a>(
        &'a self,
        plan: ToolExecutionPlan,
        conversation_messages: &'a [Message],
        ctx: &'a TurnContext<'a>,
    ) -> impl Stream<Item = Result<ToolEvent>> + 'a {
        let ToolExecutionPlan {
            frontend_requests,
            backend_requests,
            thinking_content,
            reasoning_content,
            response_role,
            response_created,
        } = plan;

        let num_tool_requests = frontend_requests.len() + backend_requests.len();

        try_stream! {
            // HACK: async_stream needs a type hint to infer the error type
            let _: Result<()> = Ok(());
            let tool_response_messages: Vec<Arc<Mutex<Message>>> = (0..num_tool_requests)
                .map(|_| Arc::new(Mutex::new(Message::user().with_generated_id())))
                .collect();

            let mut request_to_response_map: HashMap<String, Arc<Mutex<Message>>> = HashMap::new();
            let mut request_metadata: HashMap<String, Option<ProviderMetadata>> = HashMap::new();
            for (idx, request) in frontend_requests.iter().chain(backend_requests.iter()).enumerate() {
                request_to_response_map.insert(request.id.clone(), tool_response_messages[idx].clone());
                request_metadata.insert(request.id.clone(), request.metadata.clone());
            }

            for (idx, request) in frontend_requests.iter().enumerate() {
                let mut frontend_tool_stream = self.handle_frontend_tool_request(
                    request,
                    tool_response_messages[idx].clone(),
                );
                while let Some(msg) = frontend_tool_stream.try_next().await? {
                    yield ToolEvent::Passthrough(AgentEvent::Message(msg));
                }
            }

            let mut extension_installed = false;

            if ctx.goose_mode == GooseMode::Chat {
                for request in &backend_requests {
                    if let Some(response_msg) = request_to_response_map.get(&request.id) {
                        let mut resp = response_msg.lock().await;
                        *resp = resp.clone().with_tool_response_with_metadata(
                            request.id.clone(),
                            Ok(CallToolResult::success(vec![Content::text(CHAT_MODE_TOOL_SKIPPED_RESPONSE)])),
                            request.metadata.as_ref(),
                        );
                    }
                }
            } else {
                let inspection_results = self.tool_inspection_manager
                    .inspect_tools(
                        &ctx.session_config.id,
                        &backend_requests,
                        conversation_messages,
                        ctx.goose_mode,
                    )
                    .await?;

                let permission_check_result = self.tool_inspection_manager
                    .process_inspection_results_with_permission_inspector(
                        &backend_requests,
                        &inspection_results,
                    )
                    .unwrap_or_else(|| {
                        let mut result = PermissionCheckResult {
                            approved: vec![],
                            needs_approval: vec![],
                            denied: vec![],
                        };
                        result.needs_approval.extend(backend_requests.iter().cloned());
                        result
                    });

                let mut enable_extension_request_ids = HashSet::new();
                for request in &backend_requests {
                    if let Ok(tool_call) = &request.tool_call {
                        if tool_call.name == MANAGE_EXTENSIONS_TOOL_NAME_COMPLETE {
                            enable_extension_request_ids.insert(request.id.clone());
                        }
                    }
                }

                let mut tool_futures = self.handle_approved_and_denied_tools(
                    &permission_check_result,
                    &request_to_response_map,
                    ctx.cancel_token.clone(),
                    ctx.session,
                ).await?;

                let tool_futures_arc = Arc::new(Mutex::new(tool_futures));

                let mut tool_approval_stream = self.handle_approval_tool_requests(
                    &permission_check_result.needs_approval,
                    tool_futures_arc.clone(),
                    &request_to_response_map,
                    ctx.cancel_token.clone(),
                    ctx.session,
                    &inspection_results,
                );

                while let Some(msg) = tool_approval_stream.try_next().await? {
                    yield ToolEvent::Passthrough(AgentEvent::Message(msg));
                }

                tool_futures = {
                    let mut futures_lock = tool_futures_arc.lock().await;
                    futures_lock.drain(..).collect::<Vec<_>>()
                };

                let with_id = tool_futures
                    .into_iter()
                    .map(|(request_id, s)| s.map(move |item| (request_id.clone(), item)))
                    .collect::<Vec<_>>();

                let mut combined = stream::select_all(with_id);
                let mut all_install_successful = true;

                loop {
                    if is_token_cancelled(ctx.cancel_token) {
                        break;
                    }

                    for msg in self.drain_elicitation_messages(&ctx.session_config.id).await {
                        yield ToolEvent::Passthrough(AgentEvent::Message(msg));
                    }

                    tokio::select! {
                        biased;

                        tool_item = combined.next() => {
                            match tool_item {
                                Some((request_id, item)) => {
                                    match item {
                                        ToolStreamItem::Result(output) => {
                                            if let Ok(ref call_result) = output {
                                                if let Some(notification) = extract_platform_notification(call_result) {
                                                    yield ToolEvent::Passthrough(AgentEvent::McpNotification((request_id.clone(), notification)));
                                                }
                                            }

                                            if enable_extension_request_ids.contains(&request_id)
                                                && output.is_err()
                                            {
                                                all_install_successful = false;
                                            }
                                            if let Some(response_msg) = request_to_response_map.get(&request_id) {
                                                let metadata = request_metadata.get(&request_id).and_then(|m| m.as_ref());
                                                let mut resp = response_msg.lock().await;
                                                *resp = resp.clone().with_tool_response_with_metadata(request_id, output, metadata);
                                            }
                                        }
                                        ToolStreamItem::Message(msg) => {
                                            yield ToolEvent::Passthrough(AgentEvent::McpNotification((request_id, msg)));
                                        }
                                    }
                                }
                                None => break,
                            }
                        }

                        _ = tokio::time::sleep(std::time::Duration::from_millis(100)) => {}
                    }
                }

                for msg in self.drain_elicitation_messages(&ctx.session_config.id).await {
                    yield ToolEvent::Passthrough(AgentEvent::Message(msg));
                }

                if all_install_successful && !enable_extension_request_ids.is_empty() {
                    if let Err(e) = self.save_extension_state(ctx.session_config).await {
                        warn!("Failed to save extension state after runtime changes: {}", e);
                    }
                    extension_installed = true;
                }
            }

            let mut tool_messages = Vec::new();
            let mut had_parse_error = false;

            let thinking_msg = if !thinking_content.is_empty() {
                Some(Message::new(
                    response_role,
                    response_created,
                    thinking_content,
                ).with_id(format!("msg_{}", Uuid::new_v4())))
            } else {
                None
            };

            for (idx, request) in frontend_requests.iter().chain(backend_requests.iter()).enumerate() {
                if request.tool_call.is_ok() {
                    let mut request_msg = Message::assistant()
                        .with_id(format!("msg_{}", Uuid::new_v4()));
                    for rc in &reasoning_content {
                        request_msg = request_msg.with_content(rc.clone());
                    }
                    request_msg = request_msg.with_tool_request_with_metadata(
                        request.id.clone(),
                        request.tool_call.clone(),
                        request.metadata.as_ref(),
                        request.tool_meta.clone(),
                    );
                    let final_response = tool_response_messages[idx].lock().await.clone();
                    tool_messages.push((request_msg, final_response));
                } else {
                    error!(
                        "Tool call could not be parsed: {}",
                        request.tool_call.as_ref().unwrap_err(),
                    );
                    had_parse_error = true;
                    break;
                }
            }

            yield ToolEvent::Done(ToolExecutionOutcome {
                tool_messages,
                thinking_msg,
                extension_installed,
                had_parse_error,
            });
        }
    }
}
