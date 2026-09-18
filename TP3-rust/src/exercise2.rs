use std::{net::SocketAddr, path::Path};

use anyhow::{Context, Result};

use crate::digits::{
    artifact::DigitModelArtifact,
    config::DigitStudyConfig,
    data::load_digit_dataset,
    study::{run_digit_evaluation, run_digit_training},
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
