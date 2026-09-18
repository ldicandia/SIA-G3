mod candidates;
mod plots;
mod report;

use std::{fs, net::SocketAddr, path::Path};

use anyhow::{bail, Context, Result};

use super::{
    artifact::DigitModelArtifact,
    config::{CandidateConfig, DigitStudyConfig},
    data::{stratified_digit_split, DigitDataset, IMAGE_PIXELS},
    live_dashboard::LiveMetricsPublisher,
    metrics::{classification_metrics, ClassificationMetrics},
    training::{evaluate_digit_model, refit_digit_model, DigitTrainingReport},
};

use candidates::*;
use plots::*;
use report::*;

#[derive(Clone, Debug)]
pub struct CandidateResult {
    pub candidate: CandidateConfig,
    pub best_epoch: usize,
    pub validation_loss: f64,
    pub validation_accuracy: f64,
    pub stopped_early: bool,
}

pub struct TrainingOutcome {
    pub artifact: DigitModelArtifact,
    pub candidates: Vec<CandidateResult>,
}

struct CandidateRun {
    result: CandidateResult,
    report: DigitTrainingReport,
}

pub fn run_digit_training(
    exercise: &str,
    dataset: &DigitDataset,
    config: &DigitStudyConfig,
    additional_candidates: &[CandidateConfig],
    output: &Path,
    live_target: Option<SocketAddr>,
) -> Result<TrainingOutcome> {
    fs::create_dir_all(output)?;
    let publisher = match live_target {
        Some(target) => LiveMetricsPublisher::new(exercise, output, target)?,
        None => LiveMetricsPublisher::disabled(exercise),
    };
    let mut candidates = additional_candidates.to_vec();
    candidates.extend(config.candidates.clone());
    validate_candidates(&candidates, dataset.features.cols())?;

    let split = stratified_digit_split(&dataset.labels, config.validation_ratio, config.split_seed);
    write_dataset_summary(dataset, &split.train, &split.validation, output)?;

    let runs = train_candidates(candidates, dataset, &split, &publisher)?;

    let best_index = select_best_index(&runs).context("candidate list cannot be empty")?;
    write_candidate_summary(&runs, output)?;
    write_learning_history(&runs, output)?;
    plot_candidate_losses(&runs, &output.join("candidate_loss_curves.png"))?;
    plot_selected_loss(
        &runs[best_index],
        &output.join("selected_learning_curve.png"),
    )?;

    let best = &runs[best_index];
    let all_indices = (0..dataset.features.rows()).collect::<Vec<_>>();
    let mut final_model = create_model(&best.result.candidate)?;
    refit_digit_model(
        &mut final_model,
        &dataset.features,
        &dataset.labels,
        &all_indices,
        &best.result.candidate,
        best.result.best_epoch,
        Some(&publisher),
    )?;
    let artifact = DigitModelArtifact {
        format_version: 1,
        exercise: exercise.to_owned(),
        candidate: best.result.candidate.clone(),
        selected_epoch: best.result.best_epoch,
        selection_validation_loss: best.result.validation_loss,
        selection_validation_accuracy: best.result.validation_accuracy,
        model: final_model,
    };
    artifact.save(&output.join("selected_model.toml"))?;
    write_selected_model(&artifact, output)?;
    eprintln!(
        "selected candidate '{}' (validation_accuracy={:.6}, validation_loss={:.6}, epoch={})",
        artifact.candidate.name,
        artifact.selection_validation_accuracy,
        artifact.selection_validation_loss,
        artifact.selected_epoch
    );
    let dropped_events = publisher.dropped_events();
    if dropped_events > 0 {
        eprintln!("live metrics dropped {dropped_events} events to avoid blocking training");
    }

    Ok(TrainingOutcome {
        artifact,
        candidates: runs.into_iter().map(|run| run.result).collect(),
    })
}

pub fn run_digit_evaluation(
    dataset: &DigitDataset,
    artifact: &DigitModelArtifact,
    output: &Path,
) -> Result<ClassificationMetrics> {
    fs::create_dir_all(output)?;
    if artifact.model.topology().first() != Some(&dataset.features.cols()) {
        bail!("model input size does not match evaluation dataset");
    }
    let indices = (0..dataset.features.rows()).collect::<Vec<_>>();
    let evaluation = evaluate_digit_model(
        &artifact.model,
        &dataset.features,
        &dataset.labels,
        &indices,
    )?;
    let metrics = classification_metrics(&evaluation.predictions, &dataset.labels)?;
    write_evaluation_metrics(artifact, evaluation.loss, &metrics, output)?;
    write_predictions(
        &dataset.labels,
        &evaluation.predictions,
        &output.join("predictions.csv"),
    )?;
    write_confusion_matrix(&metrics, &output.join("confusion_matrix.csv"))?;
    plot_confusion_matrix(&metrics, &output.join("confusion_matrix.png"))?;
    eprintln!(
        "evaluation accuracy={:.6} loss={:.6} samples={}",
        metrics.accuracy,
        evaluation.loss,
        dataset.labels.len()
    );
    Ok(metrics)
}

pub fn expected_image_pixels() -> usize {
    IMAGE_PIXELS
}
