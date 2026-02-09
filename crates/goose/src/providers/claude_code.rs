use anyhow::Result;
use async_stream::try_stream;
use async_trait::async_trait;
use futures::future::BoxFuture;
use rmcp::model::{Role, Tool};
use serde_json::{json, Value};
use std::path::PathBuf;
use std::process::Stdio;
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::Command;

use super::base::{
    stream_from_single_message, ConfigKey, MessageStream, Provider, ProviderDef, ProviderMetadata,
    ProviderUsage, Usage,
};
use super::errors::ProviderError;
use super::utils::{filter_extensions_from_system_prompt, RequestLog};
use crate::config::base::ClaudeCodeCommand;
use crate::config::search_path::SearchPaths;
use crate::config::{Config, GooseMode};
use crate::conversation::message::{Message, MessageContent};
use crate::model::ModelConfig;
use crate::subprocess::configure_subprocess;

fn extract_usage_tokens(usage_info: &Value) -> (Option<i32>, Option<i32>) {
    let input = usage_info
        .get("input_tokens")
        .and_then(|v| v.as_i64())
        .and_then(|v| i32::try_from(v).ok());
    let output = usage_info
        .get("output_tokens")
        .and_then(|v| v.as_i64())
        .and_then(|v| i32::try_from(v).ok());
    (input, output)
}

const CLAUDE_CODE_PROVIDER_NAME: &str = "claude-code";
pub const CLAUDE_CODE_DEFAULT_MODEL: &str = "claude-sonnet-4-20250514";
pub const CLAUDE_CODE_KNOWN_MODELS: &[&str] = &["sonnet", "opus"];
pub const CLAUDE_CODE_DOC_URL: &str = "https://code.claude.com/docs/en/setup";

#[derive(Debug)]
struct CliProcess {
    child: tokio::process::Child,
    stdin: tokio::process::ChildStdin,
    reader: BufReader<tokio::process::ChildStdout>,
    stderr_handle: tokio::task::JoinHandle<String>,
    messages_sent: usize,
    needs_drain: bool,
}

impl CliProcess {
    async fn drain_pending_response(&mut self) {
        if !self.needs_drain {
            return;
        }
        tracing::debug!("Draining cancelled response from CLI process");
        let mut line = String::new();
        loop {
            line.clear();
            match self.reader.read_line(&mut line).await {
                Ok(0) => break,
                Ok(_) => {
                    let trimmed = line.trim();
                    if trimmed.is_empty() {
                        continue;
                    }
                    if let Ok(parsed) = serde_json::from_str::<Value>(trimmed) {
                        match parsed.get("type").and_then(|t| t.as_str()) {
                            Some("result") | Some("error") => break,
                            _ => continue,
                        }
                    }
                }
                Err(_) => break,
            }
        }
        self.needs_drain = false;
        tracing::debug!("Drain complete, protocol re-synced");
    }
}

impl Drop for CliProcess {
    fn drop(&mut self) {
        self.stderr_handle.abort();
        let _ = self.child.start_kill();
    }
}

#[derive(Debug, serde::Serialize)]
pub struct ClaudeCodeProvider {
    command: PathBuf,
    model: ModelConfig,
    #[serde(skip)]
    name: String,
    #[serde(skip)]
    cli_process: Arc<tokio::sync::Mutex<Option<CliProcess>>>,
}

impl ClaudeCodeProvider {
    pub async fn from_env(model: ModelConfig) -> Result<Self> {
        let config = crate::config::Config::global();
        let command: String = config.get_claude_code_command().unwrap_or_default().into();
        let resolved_command = SearchPaths::builder().with_npm().resolve(&command)?;

        Ok(Self {
            command: resolved_command,
            model,
            name: CLAUDE_CODE_PROVIDER_NAME.to_string(),
            cli_process: Arc::new(tokio::sync::Mutex::new(None)),
        })
    }

    fn messages_to_content_blocks(&self, messages: &[Message]) -> Vec<Value> {
        let mut blocks: Vec<Value> = Vec::new();
        for message in messages.iter().filter(|m| m.is_agent_visible()) {
            let prefix = match message.role {
                Role::User => "Human: ",
                Role::Assistant => "Assistant: ",
            };
            let mut text_parts = Vec::new();
            for content in &message.content {
                match content {
                    MessageContent::Text(t) => text_parts.push(t.text.clone()),
                    MessageContent::Image(img) => {
                        if !text_parts.is_empty() {
                            blocks.push(json!({"type":"text","text":format!("{}{}", prefix, text_parts.join("\n"))}));
                            text_parts.clear();
                        }
                        blocks.push(json!({"type":"image","source":{"type":"base64","media_type":img.mime_type,"data":img.data}}));
                    }
                    MessageContent::ToolRequest(req) => {
                        if let Ok(call) = &req.tool_call {
                            text_parts.push(format!("[tool_use: {} id={}]", call.name, req.id));
                        }
                    }
                    MessageContent::ToolResponse(resp) => {
                        if let Ok(result) = &resp.tool_result {
                            let text: String = result
                                .content
                                .iter()
                                .filter_map(|c| match &c.raw {
                                    rmcp::model::RawContent::Text(t) => Some(t.text.as_str()),
                                    _ => None,
                                })
                                .collect::<Vec<&str>>()
                                .join("\n");
                            text_parts.push(format!("[tool_result id={}] {}", resp.id, text));
                        }
                    }
                    _ => {}
                }
            }
            if !text_parts.is_empty() {
                blocks.push(
                    json!({"type":"text","text":format!("{}{}", prefix, text_parts.join("\n"))}),
                );
            }
        }
        blocks
    }

    fn apply_permission_flags(cmd: &mut Command) -> Result<(), ProviderError> {
        let config = Config::global();
        let goose_mode = config.get_goose_mode().unwrap_or(GooseMode::Auto);

        match goose_mode {
            GooseMode::Auto => {
                cmd.arg("--dangerously-skip-permissions");
            }
            GooseMode::SmartApprove => {
                cmd.arg("--permission-mode").arg("acceptEdits");
            }
            GooseMode::Approve => {
                return Err(ProviderError::RequestFailed(
                    "\n\n\n### NOTE\n\n\n \
                    Claude Code CLI provider does not support Approve mode.\n \
                    Please use Auto (which will run anything it needs to) or \
                    SmartApprove (most things will run or Chat Mode)\n\n\n"
                        .to_string(),
                ));
            }
            GooseMode::Chat => {}
        }
        Ok(())
    }

    fn parse_claude_response(
        &self,
        json_lines: &[String],
    ) -> Result<(Message, Usage), ProviderError> {
        let mut all_text_content = Vec::new();
        let mut usage = Usage::default();

        for line in json_lines {
            if let Ok(parsed) = serde_json::from_str::<Value>(line) {
                match parsed.get("type").and_then(|t| t.as_str()) {
                    Some("assistant") => {
                        if let Some(message) = parsed.get("message") {
                            if let Some(content) = message.get("content").and_then(|c| c.as_array())
                            {
                                for item in content {
                                    if item.get("type").and_then(|t| t.as_str()) == Some("text") {
                                        if let Some(text) =
                                            item.get("text").and_then(|t| t.as_str())
                                        {
                                            all_text_content.push(text.to_string());
                                        }
                                    }
                                }
                            }

                            if let Some(usage_info) = message.get("usage") {
                                let (input, output) = extract_usage_tokens(usage_info);
                                usage = Usage::new(input, output, None);
                            }
                        }
                    }
                    Some("result") => {
                        if let Some(result_usage) = parsed.get("usage") {
                            let (input, output) = extract_usage_tokens(result_usage);
                            usage = Usage::new(
                                input.or(usage.input_tokens),
                                output.or(usage.output_tokens),
                                None,
                            );
                        }
                    }
                    Some("error") => {
                        return Err(error_from_event(&parsed));
                    }
                    Some("system") => {}
                    _ => {}
                }
            }
        }

        let combined_text = all_text_content.join("\n\n");
        if combined_text.is_empty() {
            return Err(ProviderError::RequestFailed(
                "No text content found in response".to_string(),
            ));
        }

        let message_content = vec![MessageContent::text(combined_text)];

        let response_message = Message::new(
            Role::Assistant,
            chrono::Utc::now().timestamp(),
            message_content,
        );

        Ok((response_message, usage))
    }

    fn spawn_process(&self, filtered_system: &str) -> Result<CliProcess, ProviderError> {
        let mut cmd = Command::new(&self.command);
        configure_subprocess(&mut cmd);
        cmd.arg("--input-format")
            .arg("stream-json")
            .arg("--output-format")
            .arg("stream-json")
            .arg("--verbose")
            .arg("--include-partial-messages")
            .arg("--system-prompt")
            .arg(filtered_system);

        if CLAUDE_CODE_KNOWN_MODELS.contains(&self.model.model_name.as_str()) {
            cmd.arg("--model").arg(&self.model.model_name);
        }

        Self::apply_permission_flags(&mut cmd)?;

        cmd.stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        let mut child = cmd.spawn().map_err(|e| {
            ProviderError::RequestFailed(format!(
                "Failed to spawn Claude CLI command '{:?}': {}.",
                self.command, e
            ))
        })?;

        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| ProviderError::RequestFailed("Failed to capture stdin".to_string()))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| ProviderError::RequestFailed("Failed to capture stdout".to_string()))?;

        // Drain stderr concurrently to prevent pipe buffer deadlock
        let stderr = child.stderr.take();
        let stderr_handle = tokio::spawn(async move {
            let mut output = String::new();
            if let Some(mut stderr) = stderr {
                use tokio::io::AsyncReadExt;
                let _ = stderr.read_to_string(&mut output).await;
            }
            output
        });

        Ok(CliProcess {
            child,
            stdin,
            reader: BufReader::new(stdout),
            stderr_handle,
            messages_sent: 0,
            needs_drain: false,
        })
    }

    async fn ensure_process(
        &self,
        guard: &mut tokio::sync::MutexGuard<'_, Option<CliProcess>>,
        filtered_system: &str,
    ) -> Result<(), ProviderError> {
        if guard.is_none() {
            **guard = Some(self.spawn_process(filtered_system)?);
        }
        Ok(())
    }

    async fn execute_command(
        &self,
        system: &str,
        messages: &[Message],
        _tools: &[Tool],
    ) -> Result<Vec<String>, ProviderError> {
        let filtered_system = filter_extensions_from_system_prompt(system);

        tracing::debug!(
            command = ?self.command,
            system_prompt_len = system.len(),
            "Executing Claude CLI command"
        );

        let mut guard = self.cli_process.lock().await;
        self.ensure_process(&mut guard, &filtered_system).await?;

        let process = guard.as_mut().unwrap();
        process.drain_pending_response().await;
        let already_sent = process.messages_sent;
        self.send_turn(&mut process.stdin, messages, already_sent)
            .await?;

        let process = guard.as_mut().unwrap();
        let mut lines = Vec::new();
        let mut line = String::new();

        loop {
            line.clear();
            match process.reader.read_line(&mut line).await {
                Ok(0) => {
                    return Err(ProviderError::RequestFailed(
                        "Claude CLI process terminated unexpectedly".to_string(),
                    ));
                }
                Ok(_) => {
                    let trimmed = line.trim();
                    if trimmed.is_empty() {
                        continue;
                    }

                    if let Ok(parsed) = serde_json::from_str::<Value>(trimmed) {
                        match parsed.get("type").and_then(|t| t.as_str()) {
                            Some("stream_event") => continue,
                            Some("result") | Some("error") => {
                                lines.push(trimmed.to_string());
                                break;
                            }
                            _ => {}
                        }
                    }
                    lines.push(trimmed.to_string());
                }
                Err(e) => {
                    return Err(ProviderError::RequestFailed(format!(
                        "Failed to read output: {}",
                        e
                    )));
                }
            }
        }

        process.messages_sent = messages.len();
        tracing::debug!("Command executed successfully, got {} lines", lines.len());
        Ok(lines)
    }

    async fn send_turn(
        &self,
        stdin: &mut tokio::process::ChildStdin,
        messages: &[Message],
        messages_sent: usize,
    ) -> Result<(), ProviderError> {
        let new_messages = if messages_sent > 0 && messages_sent < messages.len() {
            &messages[messages_sent..]
        } else {
            messages
        };
        let content_blocks = self.messages_to_content_blocks(new_messages);
        let mut payload = build_stream_json_input(&content_blocks).into_bytes();
        payload.push(b'\n');
        stdin
            .write_all(&payload)
            .await
            .map_err(|e| ProviderError::RequestFailed(format!("Failed to write to stdin: {e}")))?;
        stdin
            .flush()
            .await
            .map_err(|e| ProviderError::RequestFailed(format!("Failed to flush stdin: {e}")))?;
        Ok(())
    }

    fn is_session_description_request(system: &str) -> bool {
        system.contains("four words or less") || system.contains("4 words or less")
    }

    fn generate_simple_session_description(
        &self,
        messages: &[Message],
    ) -> Result<(Message, ProviderUsage), ProviderError> {
        let description = messages
            .iter()
            .find(|m| m.role == Role::User)
            .and_then(|m| {
                m.content.iter().find_map(|c| match c {
                    MessageContent::Text(text_content) => Some(&text_content.text),
                    _ => None,
                })
            })
            .map(|text| {
                text.split_whitespace()
                    .take(4)
                    .collect::<Vec<_>>()
                    .join(" ")
            })
            .unwrap_or_else(|| "Simple task".to_string());

        tracing::debug!(
            description = %description,
            "Generated simple session description, skipped subprocess"
        );

        let message = Message::new(
            Role::Assistant,
            chrono::Utc::now().timestamp(),
            vec![MessageContent::text(description)],
        );

        let usage = Usage::default();

        Ok((
            message,
            ProviderUsage::new(self.model.model_name.clone(), usage),
        ))
    }
}

fn build_stream_json_input(content_blocks: &[Value]) -> String {
    let msg = json!({"type":"user","message":{"role":"user","content":content_blocks}});
    serde_json::to_string(&msg).expect("serializing JSON content blocks cannot fail")
}

fn error_from_event(parsed: &Value) -> ProviderError {
    let error_msg = parsed
        .get("error")
        .and_then(|e| e.as_str())
        .unwrap_or("Unknown error");
    if error_msg.contains("context") && error_msg.contains("exceeded") {
        ProviderError::ContextLengthExceeded(error_msg.to_string())
    } else {
        ProviderError::RequestFailed(format!("Claude CLI error: {}", error_msg))
    }
}

#[async_trait]
impl ProviderDef for ClaudeCodeProvider {
    type Provider = Self;

    fn metadata() -> ProviderMetadata {
        ProviderMetadata::new(
            CLAUDE_CODE_PROVIDER_NAME,
            "Claude Code CLI",
            "Requires claude CLI installed, no MCPs. Use Anthropic provider for full features.",
            CLAUDE_CODE_DEFAULT_MODEL,
            CLAUDE_CODE_KNOWN_MODELS.to_vec(),
            CLAUDE_CODE_DOC_URL,
            vec![ConfigKey::from_value_type::<ClaudeCodeCommand>(true, false)],
        )
    }

    fn from_env(model: ModelConfig) -> BoxFuture<'static, Result<Self::Provider>> {
        Box::pin(Self::from_env(model))
    }
}

#[async_trait]
impl Provider for ClaudeCodeProvider {
    fn get_name(&self) -> &str {
        &self.name
    }

    fn get_model_config(&self) -> ModelConfig {
        self.model.clone()
    }

    async fn fetch_supported_models(&self) -> Result<Vec<String>, ProviderError> {
        Ok(CLAUDE_CODE_KNOWN_MODELS
            .iter()
            .map(|s| s.to_string())
            .collect())
    }

    #[tracing::instrument(
        skip(self, model_config, system, messages, tools),
        fields(model_config, input, output, input_tokens, output_tokens, total_tokens)
    )]
    async fn complete_with_model(
        &self,
        _session_id: Option<&str>,
        model_config: &ModelConfig,
        system: &str,
        messages: &[Message],
        tools: &[Tool],
    ) -> Result<(Message, ProviderUsage), ProviderError> {
        if Self::is_session_description_request(system) {
            return self.generate_simple_session_description(messages);
        }

        let json_lines = self.execute_command(system, messages, tools).await?;

        let (message, usage) = self.parse_claude_response(&json_lines)?;

        let payload = json!({
            "command": self.command,
            "model": model_config.model_name,
            "system": system,
            "messages": messages.len()
        });
        let mut log = RequestLog::start(model_config, &payload)?;

        let response = json!({
            "lines": json_lines.len(),
            "usage": usage
        });

        log.write(&response, Some(&usage))?;

        Ok((
            message,
            ProviderUsage::new(model_config.model_name.clone(), usage),
        ))
    }

    fn supports_streaming(&self) -> bool {
        true
    }

    async fn stream(
        &self,
        _session_id: &str,
        system: &str,
        messages: &[Message],
        _tools: &[Tool],
    ) -> Result<MessageStream, ProviderError> {
        if Self::is_session_description_request(system) {
            let (message, usage) = self.generate_simple_session_description(messages)?;
            return Ok(stream_from_single_message(message, usage));
        }

        let filtered_system = filter_extensions_from_system_prompt(system);
        let process_arc = Arc::clone(&self.cli_process);

        {
            let mut guard = process_arc.lock().await;
            self.ensure_process(&mut guard, &filtered_system).await?;
            let process = guard.as_mut().unwrap();
            process.drain_pending_response().await;
            let already_sent = process.messages_sent;
            self.send_turn(&mut process.stdin, messages, already_sent)
                .await?;
        }

        let total_messages = messages.len();
        let model_name = self.model.model_name.clone();
        let message_id = uuid::Uuid::new_v4().to_string();

        Ok(Box::pin(try_stream! {
            let mut guard = Arc::clone(&process_arc).lock_owned().await;

            if guard.is_none() {
                Err(ProviderError::RequestFailed(
                    "Claude CLI process not available".to_string(),
                ))?;
            }

            let process = guard.as_mut().unwrap();
            process.needs_drain = true;
            let mut line = String::new();
            let mut accumulated_usage = Usage::default();
            let mut stream_error: Option<ProviderError> = None;
            let stream_timestamp = chrono::Utc::now().timestamp();

            loop {
                line.clear();
                match process.reader.read_line(&mut line).await {
                    Ok(0) => {
                        process.needs_drain = false;
                        stream_error = Some(ProviderError::RequestFailed(
                            "Claude CLI process terminated unexpectedly".to_string(),
                        ));
                        break;
                    }
                    Ok(_) => {
                        let trimmed = line.trim();
                        if trimmed.is_empty() {
                            continue;
                        }

                        if let Ok(parsed) = serde_json::from_str::<Value>(trimmed) {
                            match parsed.get("type").and_then(|t| t.as_str()) {
                                Some("stream_event") => {
                                    if let Some(event) = parsed.get("event") {
                                        match event.get("type").and_then(|t| t.as_str()) {
                                            Some("content_block_delta") => {
                                                if let Some(text) = event
                                                    .get("delta")
                                                    .filter(|d| {
                                                        d.get("type").and_then(|t| t.as_str())
                                                            == Some("text_delta")
                                                    })
                                                    .and_then(|d| d.get("text"))
                                                    .and_then(|t| t.as_str())
                                                {
                                                    let mut partial_message = Message::new(
                                                        Role::Assistant,
                                                        stream_timestamp,
                                                        vec![MessageContent::text(text)],
                                                    );
                                                    partial_message.id =
                                                        Some(message_id.clone());
                                                    yield (Some(partial_message), None);
                                                }
                                            }
                                            Some("message_start") => {
                                                if let Some(usage_info) = event
                                                    .get("message")
                                                    .and_then(|m| m.get("usage"))
                                                {
                                                    let (input, _) =
                                                        extract_usage_tokens(usage_info);
                                                    if let Some(i) = input {
                                                        accumulated_usage.input_tokens = Some(i);
                                                    }
                                                }
                                            }
                                            Some("message_delta") => {
                                                if let Some(usage_info) = event.get("usage") {
                                                    let (_, output) =
                                                        extract_usage_tokens(usage_info);
                                                    if let Some(o) = output {
                                                        accumulated_usage.output_tokens = Some(o);
                                                    }
                                                }
                                            }
                                            _ => {}
                                        }
                                    }
                                }
                                Some("result") => {
                                    process.needs_drain = false;
                                    if let Some(usage_info) = parsed.get("usage") {
                                        let (input, output) = extract_usage_tokens(usage_info);
                                        accumulated_usage = Usage::new(
                                            input.or(accumulated_usage.input_tokens),
                                            output.or(accumulated_usage.output_tokens),
                                            None,
                                        );
                                    }
                                    break;
                                }
                                Some("error") => {
                                    process.needs_drain = false;
                                    stream_error = Some(error_from_event(&parsed));
                                    break;
                                }
                                _ => {}
                            }
                        }
                    }
                    Err(e) => {
                        process.needs_drain = false;
                        stream_error = Some(ProviderError::RequestFailed(format!(
                            "Failed to read streaming output: {e}"
                        )));
                        break;
                    }
                }
            }

            guard.as_mut().unwrap().messages_sent = total_messages;

            if let Some(err) = stream_error {
                Err(err)?;
            }

            let provider_usage = ProviderUsage::new(model_name, accumulated_usage);
            yield (None, Some(provider_usage));
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use test_case::test_case;

    #[test_case(
        json!({"input_tokens": 100, "output_tokens": 50}),
        Some(100), Some(50)
        ; "both_tokens"
    )]
    #[test_case(json!({"input_tokens": 100}), Some(100), None ; "input_only")]
    #[test_case(json!({}), None, None ; "empty_usage")]
    fn test_extract_usage_tokens(
        usage_json: Value,
        expected_input: Option<i32>,
        expected_output: Option<i32>,
    ) {
        let (input, output) = extract_usage_tokens(&usage_json);
        assert_eq!(input, expected_input);
        assert_eq!(output, expected_output);
    }

    #[test_case(
        r#"{"type":"error","error":"context window exceeded"}"#,
        true
        ; "context_exceeded"
    )]
    #[test_case(
        r#"{"type":"error","error":"Model not supported"}"#,
        false
        ; "generic_error_from_event"
    )]
    #[test_case(r#"{"type":"error"}"#, false ; "missing_error_field")]
    fn test_error_from_event(line: &str, is_context_exceeded: bool) {
        let parsed: Value = serde_json::from_str(line).unwrap();
        let err = error_from_event(&parsed);
        if is_context_exceeded {
            assert!(matches!(err, ProviderError::ContextLengthExceeded(_)));
        } else {
            assert!(matches!(err, ProviderError::RequestFailed(_)));
        }
    }

    /// (role, text, optional (image_data, mime_type))
    type MsgSpec<'a> = (&'a str, &'a str, Option<(&'a str, &'a str)>);

    fn build_messages(specs: &[MsgSpec]) -> Vec<Message> {
        specs
            .iter()
            .map(|(role, text, image)| {
                let role = if *role == "user" {
                    Role::User
                } else {
                    Role::Assistant
                };
                let mut msg = Message::new(role, 0, vec![]);
                if !text.is_empty() {
                    msg = Message::new(msg.role.clone(), 0, vec![MessageContent::text(*text)]);
                }
                if let Some((data, mime)) = image {
                    msg.content.push(MessageContent::image(*data, *mime));
                }
                msg
            })
            .collect()
    }

    #[test_case(
        &[],
        &[]
        ; "empty"
    )]
    #[test_case(
        &[("user", "Hello", None)],
        &[json!({"type":"text","text":"Human: Hello"})]
        ; "single_user"
    )]
    #[test_case(
        &[("user", "Hello", None), ("assistant", "Hi there!", None)],
        &[json!({"type":"text","text":"Human: Hello"}), json!({"type":"text","text":"Assistant: Hi there!"})]
        ; "user_and_assistant"
    )]
    #[test_case(
        &[("user", "Describe this", Some(("base64data", "image/png")))],
        &[json!({"type":"text","text":"Human: Describe this"}),
          json!({"type":"image","source":{"type":"base64","media_type":"image/png","data":"base64data"}})]
        ; "user_with_image"
    )]
    #[test_case(
        &[("user", "", Some(("iVBORw0KGgo", "image/png")))],
        &[json!({"type":"image","source":{"type":"base64","media_type":"image/png","data":"iVBORw0KGgo"}})]
        ; "image_only"
    )]
    fn test_messages_to_content_blocks(pairs: &[MsgSpec], expected: &[Value]) {
        let provider = make_provider();
        let messages = build_messages(pairs);
        let blocks = provider.messages_to_content_blocks(&messages);
        assert_eq!(blocks, expected);
    }

    #[test]
    fn test_messages_to_content_blocks_tool_request() {
        use rmcp::model::CallToolRequestParams;
        let provider = make_provider();
        let tool_call = Ok(CallToolRequestParams {
            name: "developer__shell".into(),
            arguments: Some(serde_json::from_value(json!({"cmd": "ls"})).unwrap()),
            meta: None,
            task: None,
        });
        let msg = Message::new(
            Role::Assistant,
            0,
            vec![MessageContent::tool_request("call_123", tool_call)],
        );
        let blocks = provider.messages_to_content_blocks(&[msg]);
        assert_eq!(
            blocks,
            vec![
                json!({"type":"text","text":"Assistant: [tool_use: developer__shell id=call_123]"})
            ]
        );
    }

    #[test]
    fn test_messages_to_content_blocks_tool_response() {
        use rmcp::model::{CallToolResult, Content};
        let provider = make_provider();
        let result = CallToolResult {
            content: vec![Content::text("file1.txt\nfile2.txt")],
            is_error: None,
            structured_content: None,
            meta: None,
        };
        let msg = Message::new(
            Role::User,
            0,
            vec![MessageContent::tool_response("call_123", Ok(result))],
        );
        let blocks = provider.messages_to_content_blocks(&[msg]);
        assert_eq!(
            blocks,
            vec![
                json!({"type":"text","text":"Human: [tool_result id=call_123] file1.txt\nfile2.txt"})
            ]
        );
    }

    #[test_case(
        &[json!({"type":"text","text":"Hello"})],
        json!({"type":"user","message":{"role":"user","content":[{"type":"text","text":"Hello"}]}})
        ; "text_block"
    )]
    #[test_case(
        &[json!({"type":"text","text":"Look"}), json!({"type":"image","source":{"type":"base64","media_type":"image/png","data":"abc"}})],
        json!({"type":"user","message":{"role":"user","content":[{"type":"text","text":"Look"},{"type":"image","source":{"type":"base64","media_type":"image/png","data":"abc"}}]}})
        ; "text_and_image_blocks"
    )]
    fn test_build_stream_json_input(blocks: &[Value], expected: Value) {
        let line = build_stream_json_input(blocks);
        let parsed: Value = serde_json::from_str(&line).unwrap();
        assert_eq!(parsed, expected);
    }

    #[test_case(
        &[
            r#"{"type":"assistant","message":{"role":"assistant","content":[{"type":"text","text":"The answer is 2."}],"usage":{"input_tokens":100,"output_tokens":20}}}"#,
            r#"{"type":"result","subtype":"success","result":"The answer is 2.","session_id":"abc"}"#,
        ],
        "The answer is 2.",
        Some(100), Some(20)
        ; "assistant_with_usage"
    )]
    #[test_case(
        &[
            r#"{"type":"assistant","message":{"role":"assistant","content":[{"type":"text","text":"First"},{"type":"text","text":"Second"}]}}"#,
        ],
        "First\n\nSecond",
        None, None
        ; "multiple_text_blocks"
    )]
    #[test_case(
        &[
            r#"{"type":"system","subtype":"init","session_id":"abc"}"#,
            r#"{"type":"assistant","message":{"role":"assistant","content":[{"type":"text","text":"Hello"}]}}"#,
            r#"{"type":"result","subtype":"success","result":"Hello","session_id":"abc"}"#,
        ],
        "Hello",
        None, None
        ; "system_init_filtered"
    )]
    #[test_case(
        &[
            r#"{"type":"stream_event","event":{"type":"content_block_delta","delta":{"type":"text_delta","text":"He"}}}"#,
            r#"{"type":"stream_event","event":{"type":"content_block_delta","delta":{"type":"text_delta","text":"llo"}}}"#,
            r#"{"type":"assistant","message":{"role":"assistant","content":[{"type":"text","text":"Hello"}],"usage":{"input_tokens":50,"output_tokens":10}}}"#,
            r#"{"type":"result","subtype":"success","result":"Hello","session_id":"abc"}"#,
        ],
        "Hello",
        Some(50), Some(10)
        ; "streaming_events_ignored_by_parse"
    )]
    fn test_parse_claude_response_ok(
        lines: &[&str],
        expected_text: &str,
        expected_input: Option<i32>,
        expected_output: Option<i32>,
    ) {
        let provider = make_provider();
        let lines: Vec<String> = lines.iter().map(|s| s.to_string()).collect();
        let (message, usage) = provider.parse_claude_response(&lines).unwrap();
        assert_eq!(message.role, Role::Assistant);
        if let MessageContent::Text(t) = &message.content[0] {
            assert_eq!(t.text, expected_text);
        } else {
            panic!("expected text content");
        }
        assert_eq!(usage.input_tokens, expected_input);
        assert_eq!(usage.output_tokens, expected_output);
    }

    #[test_case(
        &[],
        ProviderError::RequestFailed("No text content found in response".into())
        ; "empty_lines"
    )]
    #[test_case(
        &[r#"{"type":"error","error":"context window exceeded"}"#],
        ProviderError::ContextLengthExceeded("context window exceeded".into())
        ; "context_length"
    )]
    #[test_case(
        &[r#"{"type":"error","error":"Model not supported"}"#],
        ProviderError::RequestFailed("Claude CLI error: Model not supported".into())
        ; "generic_error"
    )]
    fn test_parse_claude_response_err(lines: &[&str], expected: ProviderError) {
        let provider = make_provider();
        let lines: Vec<String> = lines.iter().map(|s| s.to_string()).collect();
        assert_eq!(
            provider.parse_claude_response(&lines).unwrap_err(),
            expected
        );
    }

    fn make_provider() -> ClaudeCodeProvider {
        ClaudeCodeProvider {
            command: PathBuf::from("claude"),
            model: ModelConfig::new("sonnet").unwrap(),
            name: "claude-code".to_string(),
            cli_process: Arc::new(tokio::sync::Mutex::new(None)),
        }
    }
}
