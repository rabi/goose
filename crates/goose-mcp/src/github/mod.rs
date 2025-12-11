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

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct GetPrParams {
    pub owner: String,
    pub repo: String,
    pub pull_number: u64,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct GetIssueParams {
    pub owner: String,
    pub repo: String,
    pub issue_number: u64,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct SearchIssuesParams {
    pub query: String,
}
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct GetCommentsParams {
    pub owner: String,
    pub repo: String,
    pub number: u64,
}

#[derive(Clone)]
pub struct GithubServer {
    tool_router: ToolRouter<Self>,
    client: Client,
    token: Option<String>,
}

impl Default for GithubServer {
    fn default() -> Self {
        Self::new()
    }
}

#[tool_router(router = tool_router)]
impl GithubServer {
    pub fn new() -> Self {
        // Attempt to find a GitHub token from the environment
        let token = std::env::var("GITHUB_TOKEN")
            .or_else(|_| std::env::var("GH_TOKEN"))
            .ok();

        Self {
            tool_router: Self::tool_router(),
            client: Client::builder()
                .user_agent("goose-github-mcp/1.0")
                .build()
                .unwrap(),
            token,
        }
    }

    async fn request(&self, method: Method, url: &str) -> Result<Value, ErrorData> {
        let mut req = self
            .client
            .request(method, url)
            .header("Accept", "application/vnd.github.v3+json");

        if let Some(token) = &self.token {
            req = req.header("Authorization", format!("Bearer {}", token));
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
                format!("GitHub API Error {}: {}", status, text),
                None,
            ));
        }

        serde_json::from_str(&text).map_err(|e| {
            ErrorData::new(
                ErrorCode::INTERNAL_ERROR,
                format!("Failed to parse JSON response: {}", e),
                None,
            )
        })
    }

    // Using raw string url for now, but we could use url crate
    // https://api.github.com

    #[tool(
        name = "github_get_pr",
        description = "Get details about a pull request, including title, body, and state."
    )]
    pub async fn get_pr(
        &self,
        params: Parameters<GetPrParams>,
    ) -> Result<CallToolResult, ErrorData> {
        let url = format!(
            "https://api.github.com/repos/{}/{}/pulls/{}",
            params.0.owner, params.0.repo, params.0.pull_number
        );
        let json = self.request(Method::GET, &url).await?;

        // We might want to filter this json to be more concise for the LLM
        // For now, returning the full JSON is a safe bet, though large

        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string_pretty(&json).unwrap(),
        )]))
    }

    #[tool(
        name = "github_get_pr_diff",
        description = "Get the diff content of a pull request."
    )]
    pub async fn get_pr_diff(
        &self,
        params: Parameters<GetPrParams>,
    ) -> Result<CallToolResult, ErrorData> {
        let url = format!(
            "https://api.github.com/repos/{}/{}/pulls/{}",
            params.0.owner, params.0.repo, params.0.pull_number
        );

        let mut req = self
            .client
            .get(&url)
            .header("Accept", "application/vnd.github.v3.diff");

        if let Some(token) = &self.token {
            req = req.header("Authorization", format!("Bearer {}", token));
        }

        let resp = req.send().await.map_err(|e| {
            ErrorData::new(
                ErrorCode::INTERNAL_ERROR,
                format!("Failed to send request: {}", e),
                None,
            )
        })?;

        if !resp.status().is_success() {
            return Err(ErrorData::new(
                ErrorCode::INTERNAL_ERROR,
                format!("GitHub API Error {}", resp.status()),
                None,
            ));
        }

        let diff = resp.text().await.map_err(|e| {
            ErrorData::new(
                ErrorCode::INTERNAL_ERROR,
                format!("Failed to read response body: {}", e),
                None,
            )
        })?;

        Ok(CallToolResult::success(vec![Content::text(diff)]))
    }

    #[tool(name = "github_get_issue", description = "Get details about an issue.")]
    pub async fn get_issue(
        &self,
        params: Parameters<GetIssueParams>,
    ) -> Result<CallToolResult, ErrorData> {
        let url = format!(
            "https://api.github.com/repos/{}/{}/issues/{}",
            params.0.owner, params.0.repo, params.0.issue_number
        );
        let json = self.request(Method::GET, &url).await?;

        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string_pretty(&json).unwrap(),
        )]))
    }

    #[tool(
        name = "github_get_comments",
        description = "Get comments on an issue or pull request, filtering out bot comments."
    )]
    pub async fn get_comments(
        &self,
        params: Parameters<GetCommentsParams>,
    ) -> Result<CallToolResult, ErrorData> {
        let url = format!(
            "https://api.github.com/repos/{}/{}/issues/{}/comments",
            params.0.owner, params.0.repo, params.0.number
        );
        let json = self.request(Method::GET, &url).await?;

        // Filter out bot comments
        let comments = if let Value::Array(items) = json {
            let filtered: Vec<Value> = items
                .into_iter()
                .filter(|comment| {
                    comment
                        .get("user")
                        .and_then(|u| u.get("type"))
                        .and_then(|t| t.as_str())
                        .map(|t| t != "Bot")
                        .unwrap_or(true)
                })
                .collect();
            Value::Array(filtered)
        } else {
            json
        };

        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string_pretty(&comments).unwrap(),
        )]))
    }
}

#[tool_handler(router = self.tool_router)]
impl ServerHandler for GithubServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo {
            server_info: Implementation {
                name: "goose-github".to_string(),
                version: env!("CARGO_PKG_VERSION").to_owned(),
                title: None,
                icons: None,
                website_url: None,
            },
            capabilities: ServerCapabilities::builder().enable_tools().build(),
            instructions: Some(
                "Tools for interacting with GitHub, including viewing PRs and Issues.".to_string(),
            ),
            ..Default::default()
        }
    }
}
