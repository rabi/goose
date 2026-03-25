use goose::config::GooseMode;
use goose::conversation::message::{Message, MessageContent, ToolRequest};
use goose::security::adversary_inspector::AdversaryInspector;
use goose::tool_inspection::{InspectionContext, ToolInspector};
use rmcp::model::CallToolRequestParams;
use rmcp::object;
use std::sync::Arc;
use tokio::sync::Mutex;

fn make_request(
    id: &str,
    tool: &str,
    args: serde_json::Map<String, serde_json::Value>,
) -> ToolRequest {
    ToolRequest {
        id: id.into(),
        tool_call: Ok(CallToolRequestParams::new(tool.to_string()).with_arguments(args)),
        metadata: None,
        tool_meta: None,
    }
}

fn write_adversary_md(dir: &std::path::Path, content: &str) {
    std::fs::create_dir_all(dir).unwrap();
    std::fs::write(dir.join("adversary.md"), content).unwrap();
}

fn make_context<'a>(requests: &'a [ToolRequest], messages: &'a [Message]) -> InspectionContext<'a> {
    InspectionContext::new("test-session", requests, messages, GooseMode::SmartApprove)
}

#[tokio::test]
async fn test_adversary_disabled_without_config_file() {
    let tmp = tempfile::tempdir().unwrap();

    let provider = Arc::new(Mutex::new(None));
    let inspector = AdversaryInspector::with_config_dir(provider, tmp.path().to_path_buf());

    assert_eq!(inspector.name(), "adversary");
    assert!(!inspector.is_enabled());

    let requests = [make_request(
        "r1",
        "shell",
        object!({"command": "rm -rf /"}),
    )];
    let results = inspector
        .inspect(&make_context(&requests, &[]))
        .await
        .unwrap();

    assert!(results.is_empty());
}

#[tokio::test]
async fn test_adversary_enabled_default_tools() {
    let tmp = tempfile::tempdir().unwrap();
    write_adversary_md(tmp.path(), "BLOCK everything for testing");

    let provider = Arc::new(Mutex::new(None));
    let inspector = AdversaryInspector::with_config_dir(provider, tmp.path().to_path_buf());

    assert!(inspector.is_enabled());

    let messages = vec![Message::new(
        rmcp::model::Role::User,
        chrono::Utc::now().timestamp(),
        vec![MessageContent::text("build the project")],
    )];

    // shell is reviewed by default — no provider means fail-open (Allow)
    let requests = [make_request(
        "r1",
        "shell",
        object!({"command": "cargo build"}),
    )];
    let results = inspector
        .inspect(&make_context(&requests, &messages))
        .await
        .unwrap();

    assert_eq!(results.len(), 1);
    assert!(matches!(
        results[0].action,
        goose::tool_inspection::InspectionAction::Allow
    ));

    // write is NOT reviewed by default — skipped entirely
    let requests = [make_request(
        "r1",
        "write",
        object!({"path": "foo.txt", "content": "hi"}),
    )];
    let results = inspector
        .inspect(&make_context(&requests, &messages))
        .await
        .unwrap();

    assert!(results.is_empty());
}

#[tokio::test]
async fn test_adversary_custom_tool_filter() {
    let tmp = tempfile::tempdir().unwrap();
    write_adversary_md(
        tmp.path(),
        "tools: shell, computercontroller__automation_script\n---\nBLOCK bad stuff",
    );

    let provider = Arc::new(Mutex::new(None));
    let inspector = AdversaryInspector::with_config_dir(provider, tmp.path().to_path_buf());

    assert!(inspector.is_enabled());

    let messages = vec![Message::new(
        rmcp::model::Role::User,
        chrono::Utc::now().timestamp(),
        vec![MessageContent::text("do something")],
    )];

    // shell — reviewed
    let requests = [make_request("r1", "shell", object!({"command": "ls"}))];
    let c = InspectionContext::new("test", &requests, &messages, GooseMode::Auto);
    let results = inspector.inspect(&c).await.unwrap();
    assert_eq!(results.len(), 1);

    // automation_script — reviewed
    let requests = [make_request(
        "r2",
        "computercontroller__automation_script",
        object!({"script": "echo hi", "language": "shell"}),
    )];
    let c = InspectionContext::new("test", &requests, &messages, GooseMode::Auto);
    let results = inspector.inspect(&c).await.unwrap();
    assert_eq!(results.len(), 1);

    // write — NOT reviewed
    let requests = [make_request(
        "r3",
        "write",
        object!({"path": "x.txt", "content": "y"}),
    )];
    let c = InspectionContext::new("test", &requests, &messages, GooseMode::Auto);
    let results = inspector.inspect(&c).await.unwrap();
    assert!(results.is_empty());
}
