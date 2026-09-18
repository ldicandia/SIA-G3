use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::{Args, Parser, Subcommand};
use tp3_rust::{
    config::AppConfig,
    data::load_fraud_dataset,
    experiment::{run_generalization, run_inspection, run_learning_comparison},
};

#[derive(Debug, Parser)]
#[command(name = "tp3-rust")]
#[command(about = "Perceptron experiments for fraud knowledge distillation")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Validate and summarize the fraud dataset.
    Inspect(DataArgs),
    /// Compare linear and sigmoid single-layer perceptrons on all samples.
    Compare(ExperimentArgs),
    /// Run the train/validation/test generalization study.
    Generalize(ExperimentArgs),
    /// Run inspection, learning comparison, and generalization.
    RunAll(ExperimentArgs),
}

#[derive(Debug, Args)]
struct DataArgs {
    /// Path to fraud_dataset.csv.
    #[arg(long)]
    data: PathBuf,
    /// Directory where CSV metrics and PNG plots are written.
    #[arg(long, default_value = "output")]
    output: PathBuf,
}

#[derive(Debug, Args)]
struct ExperimentArgs {
    /// Path to fraud_dataset.csv.
    #[arg(long)]
    data: PathBuf,
    /// Experiment configuration in TOML format.
    #[arg(long, default_value = "configs/default.toml")]
    config: PathBuf,
    /// Directory where CSV metrics and PNG plots are written.
    #[arg(long, default_value = "output")]
    output: PathBuf,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Inspect(args) => {
            let dataset = load_fraud_dataset(&args.data)
                .with_context(|| format!("failed to load {}", args.data.display()))?;
            run_inspection(&dataset, &args.output)?;
        }
        Command::Compare(args) => {
            let (dataset, config) = load_inputs(&args)?;
            run_learning_comparison(&dataset, &config, &args.output)?;
        }
        Command::Generalize(args) => {
            let (dataset, config) = load_inputs(&args)?;
            run_generalization(&dataset, &config, &args.output)?;
        }
        Command::RunAll(args) => {
            let (dataset, config) = load_inputs(&args)?;
            run_inspection(&dataset, &args.output)?;
            run_learning_comparison(&dataset, &config, &args.output)?;
            run_generalization(&dataset, &config, &args.output)?;
        }
    }
    Ok(())
}

fn load_inputs(args: &ExperimentArgs) -> Result<(tp3_rust::data::FraudDataset, AppConfig)> {
    let dataset = load_fraud_dataset(&args.data)
        .with_context(|| format!("failed to load {}", args.data.display()))?;
    let config = AppConfig::from_path(&args.config)
        .with_context(|| format!("failed to load {}", args.config.display()))?;
    Ok((dataset, config))
}
