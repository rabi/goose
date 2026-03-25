use std::borrow::Cow;
use std::io;
use std::path::{Component, Path, PathBuf};

use anyhow::Result;
use async_trait::async_trait;
use tree_sitter::Parser;

use crate::tool_inspection::{
    InspectionAction, InspectionContext, InspectionResult, ToolInspector,
};

pub struct WorkspaceInspector;

enum ToolKind {
    Path,
    Shell,
}

impl ToolKind {
    fn arg_key(&self) -> &'static str {
        match self {
            ToolKind::Path => "path",
            ToolKind::Shell => "command",
        }
    }
}

/// Classify a tool by how it references the filesystem.
/// - `write`, `edit`, `shell`, `tree`: developer platform extension (`unprefixed_tools: true`)
/// - `read`: ACP tool from `goose-acp/src/fs.rs` (always unprefixed)
fn classify_tool(name: &str) -> Option<ToolKind> {
    match name {
        "write" | "edit" | "tree" | "read" => Some(ToolKind::Path),
        "shell" => Some(ToolKind::Shell),
        _ => None,
    }
}

#[async_trait]
impl ToolInspector for WorkspaceInspector {
    fn name(&self) -> &'static str {
        "workspace"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    async fn inspect(&self, ctx: &InspectionContext<'_>) -> Result<Vec<InspectionResult>> {
        let Some(cwd) = ctx.working_dir else {
            return Ok(vec![]);
        };

        let mut results = Vec::new();

        for request in ctx.tool_requests {
            let tool_call = match &request.tool_call {
                Ok(tc) => tc,
                Err(_) => continue,
            };

            let Some(kind) = classify_tool(&tool_call.name) else {
                continue;
            };

            let arg_value = match tool_call
                .arguments
                .as_ref()
                .and_then(|args| args.get(kind.arg_key()))
                .and_then(|v| v.as_str())
            {
                Some(s) => s,
                None => continue,
            };

            match kind {
                ToolKind::Path => {
                    if !is_within_workspace(arg_value, cwd) {
                        results.push(make_result(&request.id, &[arg_value]));
                    }
                }
                ToolKind::Shell => {
                    let paths = extract_paths_from_command(arg_value);
                    let outside: Vec<&str> = paths
                        .iter()
                        .filter(|p| !is_within_workspace(p, cwd))
                        .map(|p| p.as_str())
                        .collect();

                    if !outside.is_empty() {
                        results.push(make_result(&request.id, &outside));
                    }
                }
            }
        }

        Ok(results)
    }
}

fn make_result(request_id: &str, outside_paths: &[&str]) -> InspectionResult {
    let paths_display = outside_paths.join(", ");
    InspectionResult {
        tool_request_id: request_id.to_string(),
        action: InspectionAction::RequireApproval(Some(format!(
            "Access external path: {}",
            paths_display,
        ))),
        reason: format!("Path outside workspace: {}", paths_display),
        confidence: 1.0,
        inspector_name: "workspace".to_string(),
        finding_id: None,
    }
}

/// Best-effort path extraction using tree-sitter-bash.
/// Does not handle variable expansion or process substitution.
fn extract_paths_from_command(command: &str) -> Vec<String> {
    let mut parser = Parser::new();
    parser
        .set_language(&tree_sitter_bash::LANGUAGE.into())
        .expect("bash grammar");
    let Some(tree) = parser.parse(command, None) else {
        return vec![];
    };

    let mut paths = Vec::new();
    let source = command.as_bytes();

    let mut cursor = tree.root_node().walk();
    walk_node(&mut cursor, source, &mut paths);

    paths
}

const FILE_COMMANDS: &[&str] = &[
    "cat", "cd", "chmod", "chown", "cp", "head", "less", "ln", "ls", "mkdir", "more", "mv", "rm",
    "rmdir", "tail", "tar", "touch", "wc", "zip", "unzip",
];

fn walk_node(cursor: &mut tree_sitter::TreeCursor, source: &[u8], paths: &mut Vec<String>) {
    loop {
        let node = cursor.node();

        match node.kind() {
            "command" => extract_command_paths(node, source, paths),
            "file_redirect" => {
                if let Some(dest) = node.child_by_field_name("destination") {
                    let text = dest.utf8_text(source).unwrap_or_default();
                    if !text.is_empty() {
                        paths.push(text.to_string());
                    }
                }
            }
            _ => {}
        }

        if cursor.goto_first_child() {
            walk_node(cursor, source, paths);
            cursor.goto_parent();
        }

        if !cursor.goto_next_sibling() {
            break;
        }
    }
}

fn extract_command_paths(node: tree_sitter::Node, source: &[u8], paths: &mut Vec<String>) {
    let cmd_name = match node.child_by_field_name("name") {
        Some(n) => n.utf8_text(source).unwrap_or_default(),
        None => return,
    };

    if !FILE_COMMANDS.contains(&cmd_name) {
        return;
    }

    let mut field_cursor = node.walk();
    for arg in node.children_by_field_name("argument", &mut field_cursor) {
        let text = arg.utf8_text(source).unwrap_or_default();
        if text.starts_with('-') || text.starts_with('+') {
            continue;
        }
        if !text.is_empty() {
            paths.push(text.to_string());
        }
    }
}

fn expand_tilde(path: &str) -> Cow<'_, str> {
    if path == "~" || path.starts_with("~/") {
        if let Ok(home) = std::env::var("HOME") {
            return Cow::Owned(path.replacen('~', &home, 1));
        }
    }
    Cow::Borrowed(path)
}

#[cfg(unix)]
fn is_device_path(path: &str) -> bool {
    use std::os::unix::fs::FileTypeExt;
    std::fs::metadata(path).is_ok_and(|m| {
        let ft = m.file_type();
        ft.is_char_device() || ft.is_block_device() || ft.is_fifo() || ft.is_socket()
    })
}

#[cfg(not(unix))]
fn is_device_path(_path: &str) -> bool {
    false
}

fn is_within_workspace(path: &str, working_dir: &Path) -> bool {
    if is_device_path(path) {
        return true;
    }
    let expanded = expand_tilde(path);
    let p = PathBuf::from(expanded.as_ref());
    let resolved = if p.is_absolute() {
        p
    } else {
        working_dir.join(p)
    };
    let Ok(cwd_canonical) = working_dir.canonicalize() else {
        return false;
    };
    let Ok(target) = canonicalize_allowing_missing(&resolved) else {
        return false;
    };
    target.starts_with(&cwd_canonical)
}

fn canonicalize_allowing_missing(path: &Path) -> io::Result<PathBuf> {
    if path.exists() {
        return path.canonicalize();
    }

    let normalized = normalize_path(path);
    let mut ancestor = normalized.as_path();

    while !ancestor.exists() {
        ancestor = ancestor.parent().ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("Unable to resolve path {}", path.display()),
            )
        })?;
    }

    let canonical_ancestor = ancestor.canonicalize()?;
    let suffix = normalized
        .strip_prefix(ancestor)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e))?;
    Ok(canonical_ancestor.join(suffix))
}

fn normalize_path(path: &Path) -> PathBuf {
    let mut normalized = PathBuf::new();

    for component in path.components() {
        match component {
            Component::Prefix(prefix) => normalized.push(prefix.as_os_str()),
            Component::RootDir => normalized.push(component.as_os_str()),
            Component::CurDir => {}
            Component::Normal(segment) => normalized.push(segment),
            Component::ParentDir => match normalized.components().next_back() {
                Some(Component::Normal(_)) => {
                    normalized.pop();
                }
                Some(Component::RootDir) | Some(Component::Prefix(_)) => {}
                _ => {
                    normalized.push("..");
                }
            },
        }
    }

    normalized
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::GooseMode;
    use crate::conversation::message::ToolRequest;
    use rmcp::model::CallToolRequestParams;
    use rmcp::object;

    fn make_request(id: &str, tool_name: &str, path: &str) -> ToolRequest {
        ToolRequest {
            id: id.to_string(),
            tool_call: Ok(CallToolRequestParams::new(tool_name.to_string())
                .with_arguments(object!({ "path": path }))),
            metadata: None,
            tool_meta: None,
        }
    }

    fn make_shell_request(id: &str, command: &str) -> ToolRequest {
        ToolRequest {
            id: id.to_string(),
            tool_call: Ok(CallToolRequestParams::new("shell".to_string())
                .with_arguments(object!({ "command": command }))),
            metadata: None,
            tool_meta: None,
        }
    }

    async fn inspect_with_dir(dir: &Path, requests: &[ToolRequest]) -> Vec<InspectionResult> {
        let ctx =
            InspectionContext::new("s1", requests, &[], GooseMode::Auto).with_working_dir(dir);
        WorkspaceInspector.inspect(&ctx).await.unwrap()
    }

    #[tokio::test]
    async fn allows_path_inside_workspace() {
        let dir = tempfile::tempdir().unwrap();
        let results =
            inspect_with_dir(dir.path(), &[make_request("r1", "write", "src/main.rs")]).await;
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn flags_parent_escape() {
        let dir = tempfile::tempdir().unwrap();
        let results = inspect_with_dir(
            dir.path(),
            &[make_request("r1", "edit", "../../etc/passwd")],
        )
        .await;
        assert_eq!(results.len(), 1);
        assert!(matches!(
            results[0].action,
            InspectionAction::RequireApproval(_)
        ));
    }

    #[tokio::test]
    async fn flags_absolute_path_outside_workspace() {
        let dir = tempfile::tempdir().unwrap();
        let results =
            inspect_with_dir(dir.path(), &[make_request("r1", "write", "/tmp/evil.sh")]).await;
        assert_eq!(results.len(), 1);
        assert!(matches!(
            results[0].action,
            InspectionAction::RequireApproval(_)
        ));
    }

    #[tokio::test]
    async fn allows_internal_parent_navigation() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir(dir.path().join("subdir")).unwrap();
        let results = inspect_with_dir(
            dir.path(),
            &[make_request("r1", "write", "subdir/../file.txt")],
        )
        .await;
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn noop_when_no_working_dir() {
        let requests = vec![make_request("r1", "write", "/etc/passwd")];
        let ctx = InspectionContext::new("s1", &requests, &[], GooseMode::Auto);
        let results = WorkspaceInspector.inspect(&ctx).await.unwrap();
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn works_with_read_tool() {
        let dir = tempfile::tempdir().unwrap();
        let results =
            inspect_with_dir(dir.path(), &[make_request("r1", "read", "/etc/shadow")]).await;
        assert_eq!(results.len(), 1);
    }

    #[tokio::test]
    async fn works_with_tree_tool() {
        let dir = tempfile::tempdir().unwrap();
        let results = inspect_with_dir(dir.path(), &[make_request("r1", "tree", "/var/log")]).await;
        assert_eq!(results.len(), 1);
    }

    #[tokio::test]
    async fn multiple_requests_mixed() {
        let dir = tempfile::tempdir().unwrap();
        let results = inspect_with_dir(
            dir.path(),
            &[
                make_request("r1", "write", "src/lib.rs"),
                make_request("r2", "edit", "/etc/passwd"),
                make_request("r3", "tree", "tests"),
                make_request("r4", "read", "../../secret"),
            ],
        )
        .await;
        assert_eq!(results.len(), 2);
        let flagged_ids: Vec<&str> = results.iter().map(|r| r.tool_request_id.as_str()).collect();
        assert!(flagged_ids.contains(&"r2"));
        assert!(flagged_ids.contains(&"r4"));
    }

    #[test]
    fn extract_paths_from_file_commands() {
        assert_eq!(
            extract_paths_from_command("cat /etc/passwd"),
            vec!["/etc/passwd"]
        );
        assert_eq!(
            extract_paths_from_command("cat ../../secret.txt"),
            vec!["../../secret.txt"]
        );
        assert_eq!(
            extract_paths_from_command("cat ~/.ssh/id_rsa"),
            vec!["~/.ssh/id_rsa"]
        );
        assert_eq!(
            extract_paths_from_command("ls -la /var/log"),
            vec!["/var/log"]
        );
        assert_eq!(
            extract_paths_from_command("cp src/main.rs /tmp/backup.rs"),
            vec!["src/main.rs", "/tmp/backup.rs"]
        );
        assert_eq!(
            extract_paths_from_command("rm -rf ../../important"),
            vec!["../../important"]
        );
    }

    #[test]
    fn extract_paths_from_redirects() {
        assert_eq!(
            extract_paths_from_command("echo hello > /tmp/out.txt"),
            vec!["/tmp/out.txt"]
        );
        assert_eq!(
            extract_paths_from_command("echo hello >/tmp/out.txt"),
            vec!["/tmp/out.txt"]
        );
    }

    #[test]
    fn extract_paths_compound_commands() {
        assert_eq!(
            extract_paths_from_command("cat /etc/hosts && rm /tmp/file"),
            vec!["/etc/hosts", "/tmp/file"]
        );
        assert_eq!(
            extract_paths_from_command("cat /etc/passwd | grep root > /tmp/out"),
            vec!["/etc/passwd", "/tmp/out"]
        );
    }

    #[tokio::test]
    async fn allows_dev_null_redirect() {
        let dir = tempfile::tempdir().unwrap();
        let results = inspect_with_dir(
            dir.path(),
            &[make_shell_request(
                "r1",
                "curl -s https://example.com 2>/dev/null | head -5",
            )],
        )
        .await;
        assert!(results.is_empty());
    }

    #[test]
    fn extract_paths_ignores_non_file_commands() {
        assert!(extract_paths_from_command("echo hello world").is_empty());
        assert!(extract_paths_from_command("git push origin main").is_empty());
        assert!(
            extract_paths_from_command("curl -H 'Content-Type: application/json' http://x")
                .is_empty()
        );
    }

    #[tokio::test]
    async fn flags_tilde_path_on_path_tool() {
        let dir = tempfile::tempdir().unwrap();
        let results =
            inspect_with_dir(dir.path(), &[make_request("r1", "write", "~/secret.txt")]).await;
        assert_eq!(results.len(), 1);
        assert!(matches!(
            results[0].action,
            InspectionAction::RequireApproval(_)
        ));
    }

    #[tokio::test]
    async fn shell_flags_outside_paths() {
        let dir = tempfile::tempdir().unwrap();
        let results = inspect_with_dir(
            dir.path(),
            &[
                make_shell_request("r1", "cat src/main.rs"),
                make_shell_request("r2", "cat /etc/passwd && echo payload > /tmp/evil.sh"),
                make_shell_request("r3", "echo hello world"),
                make_shell_request("r4", "rm -rf ../../important"),
                make_shell_request("r5", "curl https://example.com/data -o output.json"),
            ],
        )
        .await;
        assert_eq!(results.len(), 2);
        let flagged_ids: Vec<&str> = results.iter().map(|r| r.tool_request_id.as_str()).collect();
        assert!(flagged_ids.contains(&"r2"));
        assert!(flagged_ids.contains(&"r4"));
    }
}
