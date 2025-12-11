use reqwest::{Client, Method};
use rmcp::{
    handler::server::{router::tool::ToolRouter, wrapper::Parameters},
    model::{
        CallToolResult, Content, ErrorCode, ErrorData, Implementation, ServerCapabilities,
        ServerInfo,
    },
    schemars::JsonSchema,
    tool, tool_handler, tool_router, ServerHandler,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct GetIssueParams {
    pub issue_key: String,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct GetDevStatusParams {
    pub issue_id: u64,
    pub application_type: Option<String>,
    pub data_type: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct EditIssueParams {
    pub issue_key: String,
    pub fields: HashMap<String, Value>,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct AddCommentParams {
    pub issue_key: String,
    pub body: String,
}

#[derive(Clone)]
pub struct JiraServer {
    tool_router: ToolRouter<Self>,
    client: Client,
    instance_url: String,
    token: Option<String>,
}

impl Default for JiraServer {
    fn default() -> Self {
        Self::new()
    }
}

#[tool_router(router = tool_router)]
impl JiraServer {
    pub fn new() -> Self {
        let token = std::env::var("JIRA_PAT")
            .or_else(|_| std::env::var("JIRA_API_TOKEN"))
            .ok();

        let instance_url = std::env::var("JIRA_INSTANCE_URL")
            .unwrap_or_else(|_| "https://jira.atlassian.com".to_string())
            .trim_end_matches('/')
            .to_string();

        Self {
            tool_router: Self::tool_router(),
            client: Client::builder()
                .user_agent("goose-jira-mcp/1.0")
                .build()
                .unwrap(),
            instance_url,
            token,
        }
    }

    async fn request(
        &self,
        method: Method,
        path: &str,
        body: Option<Value>,
    ) -> Result<Value, ErrorData> {
        let url = format!("{}/{}", self.instance_url, path);
        let mut req = self
            .client
            .request(method, &url)
            .header("Accept", "application/json")
            .header("Content-Type", "application/json");

        if let Some(token) = &self.token {
            req = req.header("Authorization", format!("Bearer {}", token));
        }

        if let Some(b) = body {
            req = req.json(&b);
        }

        let resp = req.send().await.map_err(|e| {
            ErrorData::new(
                ErrorCode::INTERNAL_ERROR,
                format!("Failed to send request: {}", e),
                None,
            )
        })?;

        let status = resp.status();
        let text = resp.text().await.map_err(|e| {
            ErrorData::new(
                ErrorCode::INTERNAL_ERROR,
                format!("Failed to read response body: {}", e),
                None,
            )
        })?;

        if !status.is_success() {
            return Err(ErrorData::new(
                ErrorCode::INTERNAL_ERROR,
                format!("Jira API Error {}: {}", status, text),
                None,
            ));
        }
        
        if text.is_empty() {
            return Ok(Value::Null);
        }

        serde_json::from_str(&text).map_err(|e| {
             ErrorData::new(
                ErrorCode::INTERNAL_ERROR,
                format!("Failed to parse JSON response: {}", e),
                None,
            )
        })
    }

    #[tool(
        name = "jira_get_issue",
        description = "Get details about a Jira issue, including comments and status."
    )]
    pub async fn get_issue(
        &self,
        params: Parameters<GetIssueParams>,
    ) -> Result<CallToolResult, ErrorData> {
        // Jira REST API v2
        let path = format!(
            "rest/api/2/issue/{}?expand=names,renderedFields",
            params.0.issue_key
        );
        let json = self.request(Method::GET, &path, None).await?;

        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string_pretty(&json).unwrap(),
        )]))
    }

    #[tool(
        name = "jira_get_dev_status",
        description = "Get development status (PRs, commits) for an issue. Requires numeric issue ID."
    )]
    pub async fn get_dev_status(
        &self,
        params: Parameters<GetDevStatusParams>,
    ) -> Result<CallToolResult, ErrorData> {
        let app_type = params.0.application_type.as_deref().unwrap_or("github");
        let data_type = params.0.data_type.as_deref().unwrap_or("pullrequest");

        let path = format!(
            "rest/dev-status/1.0/issue/detail?issueId={}&applicationType={}&dataType={}",
            params.0.issue_id, app_type, data_type
        );
        let json = self.request(Method::GET, &path, None).await?;

        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string_pretty(&json).unwrap(),
        )]))
    }

    #[tool(
        name = "jira_edit_issue",
        description = "Edit a Jira issue. 'fields' should be a JSON object mapping field IDs/names to values."
    )]
    pub async fn edit_issue(
        &self,
        params: Parameters<EditIssueParams>,
    ) -> Result<CallToolResult, ErrorData> {
        let path = format!("rest/api/2/issue/{}", params.0.issue_key);
        let body = serde_json::json!({
            "fields": params.0.fields
        });

        // PUT usually returns 204 No Content on success, or JSON on error
        let _ = self.request(Method::PUT, &path, Some(body)).await?;

        Ok(CallToolResult::success(vec![Content::text(
            "Issue updated successfully.",
        )]))
    }

    #[tool(
        name = "jira_add_comment",
        description = "Add a comment to a Jira issue."
    )]
    pub async fn add_comment(
        &self,
        params: Parameters<AddCommentParams>,
    ) -> Result<CallToolResult, ErrorData> {
        let path = format!("rest/api/2/issue/{}/comment", params.0.issue_key);
        let body = serde_json::json!({ "body": params.0.body });

        let json = self.request(Method::POST, &path, Some(body)).await?;

        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string_pretty(&json).unwrap(),
        )]))
    }
}

#[tool_handler(router = self.tool_router)]
impl ServerHandler for JiraServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo {
            server_info: Implementation {
                name: "goose-jira".to_string(),
                version: env!("CARGO_PKG_VERSION").to_owned(),
                title: None,
                icons: None,
                website_url: None,
            },
            capabilities: ServerCapabilities::builder().enable_tools().build(),
            instructions: Some("Tools for interacting with Jira.".to_string()),
            ..Default::default()
        }
    }
}
