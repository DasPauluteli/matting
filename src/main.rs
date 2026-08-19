use clap::Parser;
use matting::cli::{Cli, Command};

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Prepare(args) => matting::prepare::run(&args),
        Command::Run(args) => matting::pipeline::run(&args),
    }
}
