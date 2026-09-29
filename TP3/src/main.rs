use std::{net::SocketAddr, path::PathBuf};

use anyhow::{Context, Result};
use clap::{Args, Parser, Subcommand};
use tp3_rust::{
    config::AppConfig,
    data::load_fraud_dataset,
    digits::live_dashboard::{run_monitor, spawn_monitor_process},
    exercises::{exercise2, exercise3},
    experiment::{
        run_generalization, run_inspection, run_learning_comparison, run_scoring,
        FraudModelArtifact,
    },
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
    /// Validate and summarize the fraud dataset, and justify feature preprocessing.
    Inspect(ExperimentArgs),
    /// Compare linear and sigmoid single-layer perceptrons on all samples.
    Compare(ExperimentArgs),
    /// Run the train/validation/test generalization study.
    Generalize(ExperimentArgs),
    /// Run inspection, learning comparison, and generalization.
    RunAll(ExperimentArgs),
    /// Score a fraud CSV with a saved TinyModel (fraud_model.toml).
    Score(ScoreArgs),
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
    /// Run the separate live-training dashboard process.
    Monitor(MonitorArgs),
}

#[derive(Debug, Subcommand)]
enum Exercise2Command {
    /// Tune candidates on digits.csv and persist the selected refit model.
    Train(Exercise2TrainArgs),
    /// Evaluate the persisted model on digits_test.csv.
    Evaluate(DigitEvaluateArgs),
    /// Resume training of a saved model for more epochs.
    Continue(DigitContinueArgs),
}

#[derive(Debug, Subcommand)]
enum Exercise3Command {
    /// Establish the inherited baseline, tune new candidates, and persist the winner.
    Train(Exercise3TrainArgs),
    /// Evaluate the persisted model on digits_test.csv.
    Evaluate(DigitEvaluateArgs),
    /// Resume training of a saved model for more epochs.
    Continue(DigitContinueArgs),
}

#[derive(Debug, Args)]
struct ScoreArgs {
    /// Path to a fraud CSV with the documented columns.
    #[arg(long)]
    data: PathBuf,
    /// Persisted fraud_model.toml produced by `generalize`.
    #[arg(long, default_value = "output/fraud_model.toml")]
    model: PathBuf,
    /// Directory where scores and metrics are written.
    #[arg(long, default_value = "output/scoring")]
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
    /// Start the live dashboard in a separate process.
    #[arg(long, default_value_t = false)]
    live: bool,
    /// UDP destination used by the non-blocking live-metrics publisher.
    #[arg(long, default_value = "127.0.0.1:7879")]
    live_address: SocketAddr,
    /// HTTP address on which the automatically started dashboard is served.
    #[arg(long, default_value = "127.0.0.1:7878")]
    live_http_address: SocketAddr,
}

#[derive(Debug, Args)]
struct Exercise3TrainArgs {
    /// Path to the expanded learning dataset.
    #[arg(
        long,
        default_value = "../TP3/data/data and documentation/more_digits.csv"
    )]
    data: PathBuf,
    /// Exercise 2 dataset, used to document what changed in the new data.
    #[arg(long, default_value = "../TP3/data/data and documentation/digits.csv")]
    previous_data: PathBuf,
    /// Selected model from Exercise 2, used to define the controlled baseline.
    #[arg(long, default_value = "output/exercise2/selected_model.toml")]
    baseline_model: PathBuf,
    /// Candidate definitions and split settings.
    #[arg(long, default_value = "configs/exercise3.toml")]
    config: PathBuf,
    /// Directory for models, metrics, histories, and plots.
    #[arg(long, default_value = "output/exercise3")]
    output: PathBuf,
    /// Start the live dashboard in a separate process.
    #[arg(long, default_value_t = false)]
    live: bool,
    /// UDP destination used by the non-blocking live-metrics publisher.
    #[arg(long, default_value = "127.0.0.1:7879")]
    live_address: SocketAddr,
    /// HTTP address on which the automatically started dashboard is served.
    #[arg(long, default_value = "127.0.0.1:7878")]
    live_http_address: SocketAddr,
}

#[derive(Debug, Args)]
struct MonitorArgs {
    /// UDP address on which training metrics are received.
    #[arg(long, default_value = "127.0.0.1:7879")]
    udp_address: SocketAddr,
    /// HTTP address on which the live dashboard is served.
    #[arg(long, default_value = "127.0.0.1:7878")]
    http_address: SocketAddr,
}

#[derive(Debug, Args)]
struct DigitContinueArgs {
    /// Dataset to keep training on (the same one used by `train`).
    #[arg(long)]
    data: PathBuf,
    /// Persisted selected_model.toml to resume from.
    #[arg(long)]
    model: PathBuf,
    /// Additional epochs to train.
    #[arg(long)]
    epochs: usize,
    /// Directory for the continued model and its history.
    #[arg(long)]
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
            let (dataset, config) = load_inputs(&args)?;
            run_inspection(&dataset, &config.features, &args.output)?;
        }
        Command::Score(args) => {
            let dataset = load_fraud_dataset(&args.data)
                .with_context(|| format!("failed to load {}", args.data.display()))?;
            let artifact = FraudModelArtifact::load(&args.model)
                .with_context(|| format!("failed to load {}", args.model.display()))?;
            run_scoring(&dataset, &artifact, &args.output)?;
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
            run_inspection(&dataset, &config.features, &args.output)?;
            run_learning_comparison(&dataset, &config, &args.output)?;
            run_generalization(&dataset, &config, &args.output)?;
        }
        Command::Exercise2 { command } => match command {
            Exercise2Command::Train(args) => {
                let live_target =
                    start_live_monitor(args.live, args.live_address, args.live_http_address)?;
                exercise2::train(&args.data, &args.config, &args.output, live_target)?;
            }
            Exercise2Command::Evaluate(args) => {
                exercise2::evaluate(&args.data, &args.model, &args.output)?;
            }
            Exercise2Command::Continue(args) => {
                exercise2::resume(&args.data, &args.model, args.epochs, &args.output)?;
            }
        },
        Command::Exercise3 { command } => match command {
            Exercise3Command::Train(args) => {
                let live_target =
                    start_live_monitor(args.live, args.live_address, args.live_http_address)?;
                exercise3::train(
                    &args.data,
                    &args.previous_data,
                    &args.baseline_model,
                    &args.config,
                    &args.output,
                    live_target,
                )?;
            }
            Exercise3Command::Evaluate(args) => {
                exercise3::evaluate(&args.data, &args.model, &args.output)?;
            }
            Exercise3Command::Continue(args) => {
                exercise3::resume(&args.data, &args.model, args.epochs, &args.output)?;
            }
        },
        Command::Monitor(args) => run_monitor(args.udp_address, args.http_address)?,
    }
    Ok(())
}

fn start_live_monitor(
    enabled: bool,
    udp_address: SocketAddr,
    http_address: SocketAddr,
) -> Result<Option<SocketAddr>> {
    if !enabled {
        return Ok(None);
    }

    let process_id = spawn_monitor_process(udp_address, http_address)?;
    eprintln!("live dashboard process started (pid={process_id}): http://{http_address}");
    Ok(Some(udp_address))
}

fn load_inputs(args: &ExperimentArgs) -> Result<(tp3_rust::data::FraudDataset, AppConfig)> {
    let dataset = load_fraud_dataset(&args.data)
        .with_context(|| format!("failed to load {}", args.data.display()))?;
    let config = AppConfig::from_path(&args.config)
        .with_context(|| format!("failed to load {}", args.config.display()))?;
    Ok((dataset, config))
}
