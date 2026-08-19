use clap::Parser;
use matting::cli::{Cli, Command};

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Prepare(_) => anyhow::bail!("prepare not implemented yet"),
        Command::Run(args) => {
            args.validate().map_err(anyhow::Error::msg)?;
            anyhow::bail!("run not implemented yet")
        }
    }
}
