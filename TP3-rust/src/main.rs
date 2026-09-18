use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::{Args, Parser, Subcommand};
use tp3_rust::{
    config::AppConfig,
    data::load_fraud_dataset,
    exercise2, exercise3,
    experiment::{run_generalization, run_inspection, run_learning_comparison},
};

#[derive(Debug, Parser)]
#[command(name = "tp3-rust")]
#[command(about = "Educational perceptron and multilayer-network experiments")]
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
    /// Train or evaluate the handwritten-digit study from Exercise 2.
    Exercise2 {
        #[command(subcommand)]
        command: Exercise2Command,
    },
    /// Train or evaluate the expanded handwritten-digit study from Exercise 3.
    Exercise3 {
        #[command(subcommand)]
        command: Exercise3Command,
    },
}

#[derive(Debug, Subcommand)]
enum Exercise2Command {
    /// Tune candidates on digits.csv and persist the selected refit model.
    Train(Exercise2TrainArgs),
    /// Evaluate the persisted model on digits_test.csv.
    Evaluate(DigitEvaluateArgs),
}

#[derive(Debug, Subcommand)]
enum Exercise3Command {
    /// Establish the inherited baseline, tune new candidates, and persist the winner.
    Train(Exercise3TrainArgs),
    /// Evaluate the persisted model on digits_test.csv.
    Evaluate(DigitEvaluateArgs),
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

#[derive(Debug, Args)]
struct Exercise2TrainArgs {
    /// Path to the learning dataset.
    #[arg(long, default_value = "../TP3/data/data and documentation/digits.csv")]
    data: PathBuf,
    /// Candidate definitions and split settings.
    #[arg(long, default_value = "configs/exercise2.toml")]
    config: PathBuf,
    /// Directory for models, metrics, histories, and plots.
    #[arg(long, default_value = "output/exercise2")]
    output: PathBuf,
}

#[derive(Debug, Args)]
struct Exercise3TrainArgs {
    /// Path to the expanded learning dataset.
    #[arg(
        long,
        default_value = "../TP3/data/data and documentation/more_digits.csv"
    )]
    data: PathBuf,
    /// Selected model from Exercise 2, used to define the controlled baseline.
    #[arg(long, default_value = "output/exercise2/selected_model.toml")]
    baseline_model: PathBuf,
    /// Candidate definitions and split settings.
    #[arg(long, default_value = "configs/exercise3.toml")]
    config: PathBuf,
    /// Directory for models, metrics, histories, and plots.
    #[arg(long, default_value = "output/exercise3")]
    output: PathBuf,
}

#[derive(Debug, Args)]
struct DigitEvaluateArgs {
    /// Path to the production-like digit dataset.
    #[arg(
        long,
        default_value = "../TP3/data/data and documentation/digits_test.csv"
    )]
    data: PathBuf,
    /// Persisted selected_model.toml file.
    #[arg(long)]
    model: PathBuf,
    /// Directory for evaluation metrics, predictions, and plots.
    #[arg(long)]
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
        Command::Exercise2 { command } => match command {
            Exercise2Command::Train(args) => {
                exercise2::train(&args.data, &args.config, &args.output)?;
            }
            Exercise2Command::Evaluate(args) => {
                exercise2::evaluate(&args.data, &args.model, &args.output)?;
            }
        },
        Command::Exercise3 { command } => match command {
            Exercise3Command::Train(args) => {
                exercise3::train(&args.data, &args.baseline_model, &args.config, &args.output)?;
            }
            Exercise3Command::Evaluate(args) => {
                exercise3::evaluate(&args.data, &args.model, &args.output)?;
            }
        },
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
