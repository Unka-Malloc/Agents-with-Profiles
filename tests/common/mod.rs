//! Shared constructors for integration tests — one authority for spec shapes.

use awp::{AgentSpec, CliLaneSpec, CommandSpec, LaneSpec, PromptChannel, StreamMode};

pub fn cli_spec(id: &str, program: &str, args: &[&str], channel: PromptChannel) -> AgentSpec {
    AgentSpec {
        id: id.to_owned(),
        client_name: None,
        lane: LaneSpec::Cli(Box::new(CliLaneSpec {
            args: args.iter().map(|a| a.to_string()).collect(),
            prompt_channel: channel,
            stream: StreamMode::Stdio,
            ..CliLaneSpec::default()
        })),
        command: CommandSpec {
            program: program.to_owned(),
            args: vec![],
            cwd: None,
            env: Default::default(),
        },
        max_output_bytes: None,
    }
}
