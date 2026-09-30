use std::{
    net::SocketAddr,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result};

use crate::digits::{
    analyze::{run_digit_continue, run_digit_evaluation, run_digit_training},
    artifact::DigitModelArtifact,
    attribution::run_attribution_study,
    config::DigitStudyConfig,
    data::load_digit_dataset,
    noise::{model_label, run_noise_study},
};

pub fn train(
    data: &Path,
    config: &Path,
    output: &Path,
    live_target: Option<SocketAddr>,
) -> Result<()> {
    let dataset = load_digit_dataset(data)
        .with_context(|| format!("failed to load digit dataset {}", data.display()))?;
    let config = DigitStudyConfig::from_path(config)
        .with_context(|| format!("failed to load configuration {}", config.display()))?;
    run_digit_training("exercise2", &dataset, &config, &[], output, live_target)?;
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
