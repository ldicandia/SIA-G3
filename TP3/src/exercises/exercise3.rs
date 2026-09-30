use std::{
    net::SocketAddr,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result};

use crate::digits::{
    analyze::{
        run_digit_continue, run_digit_evaluation, run_digit_training, write_data_shift,
        TrainingOutcome,
    },
    artifact::DigitModelArtifact,
    attribution::run_attribution_study,
    config::DigitStudyConfig,
    data::load_digit_dataset,
    noise::{model_label, run_noise_study},
};

const BASELINE_NAME: &str = "exercise2_winner_on_more_digits";

pub fn train(
    data: &Path,
    previous_data: &Path,
    baseline_model: &Path,
    config: &Path,
    output: &Path,
    live_target: Option<SocketAddr>,
) -> Result<()> {
    let dataset = load_digit_dataset(data)
        .with_context(|| format!("failed to load digit dataset {}", data.display()))?;
    // Documents what changed in the data itself (Exercise 3 question c),
    // independently of any modelling technique.
    let previous = load_digit_dataset(previous_data).with_context(|| {
        format!(
            "failed to load previous dataset {}",
            previous_data.display()
        )
    })?;
    std::fs::create_dir_all(output)?;
    write_data_shift(&previous, &dataset, output)?;
    drop(previous);
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

/// Resumes training of a saved model for `epochs` more epochs on `data`.
pub fn resume(data: &Path, model: &Path, epochs: usize, output: &Path) -> Result<()> {
    let dataset = load_digit_dataset(data)
        .with_context(|| format!("failed to load digit dataset {}", data.display()))?;
    let artifact = DigitModelArtifact::load(model)
        .with_context(|| format!("failed to load model {}", model.display()))?;
    run_digit_continue(&dataset, &artifact, epochs, output)?;
    Ok(())
}

/// Evaluates saved models on `data` under Gaussian and salt-and-pepper
/// noise. The first model is the primary one shown in the example grid.
pub fn noise(
    data: &Path,
    models: &[PathBuf],
    gaussian_sigmas: &[f64],
    salt_pepper_fractions: &[f64],
    seed: u64,
    output: &Path,
) -> Result<()> {
    let dataset = load_digit_dataset(data)
        .with_context(|| format!("failed to load digit dataset {}", data.display()))?;
    let artifacts = models
        .iter()
        .map(|path| {
            let artifact = DigitModelArtifact::load(path)
                .with_context(|| format!("failed to load model {}", path.display()))?;
            Ok((model_label(path), artifact))
        })
        .collect::<Result<Vec<_>>>()?;
    std::fs::create_dir_all(output)?;
    run_noise_study(
        &dataset,
        &artifacts,
        gaussian_sigmas,
        salt_pepper_fractions,
        seed,
        output,
    )?;
    Ok(())
}

/// Saliency, gradient x input and Integrated Gradients for a saved model.
pub fn attribution(
    data: &Path,
    model: &Path,
    steps: usize,
    per_class: usize,
    examples: usize,
    output: &Path,
) -> Result<()> {
    let dataset = load_digit_dataset(data)
        .with_context(|| format!("failed to load digit dataset {}", data.display()))?;
    let artifact = DigitModelArtifact::load(model)
        .with_context(|| format!("failed to load model {}", model.display()))?;
    std::fs::create_dir_all(output)?;
    run_attribution_study(&dataset, &artifact, steps, per_class, examples, output)?;
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
