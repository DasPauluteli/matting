use clap::{CommandFactory, Parser};
use matting::cli::{Cli, Command};

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Prepare(args) => matting::prepare::run(&args),
        Command::Run(args) => matting::pipeline::run(&args),
        Command::Completions(args) => {
            let mut cmd = Cli::command();
            let name = cmd.get_name().to_string();
            clap_complete::generate(args.shell, &mut cmd, name, &mut std::io::stdout());
            Ok(())
        }
    }
}
