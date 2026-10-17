//! End-to-end `awp run`: the process must reach exit after the turn —
//! a lane that keeps its event sender alive once held the drainer open
//! forever. `try_wait` polling bounds the assertion so a regression fails
//! instead of hanging the suite.

use std::io::{Read, Write};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

#[test]
#[cfg(unix)]
fn run_exits_with_turn_output() {
    let dir = std::env::temp_dir().join(format!("awp-cli-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let config = dir.join("registry.json");
    let mut file = std::fs::File::create(&config).unwrap();
    write!(
        file,
        r#"{{
  "agents": {{
    "echo": {{
      "id": "echo",
      "lane": "cli",
      "args": ["{{prompt}}"],
      "promptChannel": "arg-only",
      "command": {{"program": "echo", "args": []}}
    }}
  }},
  "profiles": {{"p": {{"agent": "echo"}}}},
  "bindings": {{"run": "p"}}
}}"#
    )
    .unwrap();

    let mut child = Command::new(env!("CARGO_BIN_EXE_awp"))
        .args([
            "run",
            "--config",
            config.to_str().unwrap(),
            "--profile",
            "p",
            "--prompt",
            "hello-run",
        ])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();

    let deadline = Instant::now() + Duration::from_secs(30);
    let status = loop {
        if let Some(status) = child.try_wait().unwrap() {
            break status;
        }
        if Instant::now() > deadline {
            let _ = child.kill();
            panic!("awp run did not exit — event drainer still open");
        }
        std::thread::sleep(Duration::from_millis(100));
    };
    let mut stdout = String::new();
    child
        .stdout
        .take()
        .unwrap()
        .read_to_string(&mut stdout)
        .unwrap();
    std::fs::remove_dir_all(&dir).ok();

    assert!(status.success(), "exit status: {status}");
    assert_eq!(stdout.trim_end(), "hello-run");
}
