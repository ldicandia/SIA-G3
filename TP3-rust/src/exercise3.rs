use std::{net::SocketAddr, path::Path};

use anyhow::{Context, Result};

use crate::digits::{
    artifact::DigitModelArtifact,
    config::DigitStudyConfig,
    data::load_digit_dataset,
    study::{run_digit_evaluation, run_digit_training, TrainingOutcome},
};

const BASELINE_NAME: &str = "exercise2_winner_on_more_digits";

pub fn train(
    data: &Path,
    baseline_model: &Path,
    config: &Path,
    output: &Path,
    live_target: Option<SocketAddr>,
) -> Result<()> {
    let dataset = load_digit_dataset(data)
        .with_context(|| format!("failed to load digit dataset {}", data.display()))?;
    let baseline_artifact = DigitModelArtifact::load(baseline_model).with_context(|| {
        format!(
            "failed to load Exercise 2 baseline {}",
            baseline_model.display()
        )
    })?;
    let config = DigitStudyConfig::from_path(config)
        .with_context(|| format!("failed to load configuration {}", config.display()))?;
    let mut baseline_candidate = baseline_artifact.candidate.clone();
    baseline_candidate.name = BASELINE_NAME.to_owned();
    let outcome = run_digit_training(
        "exercise3",
        &dataset,
        &config,
        &[baseline_candidate],
        output,
        live_target,
    )?;
    write_comparison(&baseline_artifact, &outcome, output)?;
    Ok(())
}

pub fn evaluate(data: &Path, model: &Path, output: &Path) -> Result<()> {
    let dataset = load_digit_dataset(data)
        .with_context(|| format!("failed to load digit dataset {}", data.display()))?;
    let artifact = DigitModelArtifact::load(model)
        .with_context(|| format!("failed to load model {}", model.display()))?;
    run_digit_evaluation(&dataset, &artifact, output)?;
    Ok(())
}

fn write_comparison(
    exercise2: &DigitModelArtifact,
    outcome: &TrainingOutcome,
    output: &Path,
) -> Result<()> {
    let baseline = outcome
        .candidates
        .iter()
        .find(|candidate| candidate.candidate.name == BASELINE_NAME)
        .context("Exercise 3 baseline result is missing")?;
    let mut writer = csv::Writer::from_path(output.join("exercise3_comparison.csv"))?;
    writer.write_record(["measurement", "accuracy", "delta"])?;
    writer.serialize((
        "exercise2_winner_on_digits",
        exercise2.selection_validation_accuracy,
        0.0,
    ))?;
    writer.serialize((
        "same_hyperparameters_on_more_digits",
        baseline.validation_accuracy,
        baseline.validation_accuracy - exercise2.selection_validation_accuracy,
    ))?;
    writer.serialize((
        "exercise3_selected_candidate",
        outcome.artifact.selection_validation_accuracy,
        outcome.artifact.selection_validation_accuracy - baseline.validation_accuracy,
    ))?;
    writer.flush()?;
    Ok(())
}
