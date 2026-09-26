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

#[derive(Clone)]
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
    plot_axis_losses(&runs, output)?;
    plot_validation_accuracy(&runs, &output.join("validation_accuracy.png"))?;
    plot_selected_loss(
        &runs[best_index],
        &output.join("selected_learning_curve.png"),
    )?;

    let best = &runs[best_index];
    let all_indices = (0..dataset.features.rows()).collect::<Vec<_>>();
    let mut final_model = create_model(&best.result.candidate)?;
    eprintln!(
        "refitting '{}' on all {} rows for {} epochs",
        best.result.candidate.name,
        all_indices.len(),
        best.result.best_epoch
    );
    refit_digit_model(
        &mut final_model,
        &dataset.features,
        &dataset.labels,
        &all_indices,
        &best.result.candidate,
        best.result.best_epoch,
        0,
        "refit",
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

/// Resumes training of a persisted model on `dataset` for `epochs` more
/// epochs with the same hyperparameters. The learning-rate schedule and random
/// streams continue from the saved epoch; optimizer moments restart from zero
/// because they are not persisted.
pub fn run_digit_continue(
    dataset: &DigitDataset,
    artifact: &DigitModelArtifact,
    epochs: usize,
    output: &Path,
) -> Result<DigitModelArtifact> {
    fs::create_dir_all(output)?;
    if artifact.model.topology().first() != Some(&dataset.features.cols()) {
        bail!("model input size does not match the dataset");
    }
    let mut model = artifact.model.clone();
    let indices = (0..dataset.features.rows()).collect::<Vec<_>>();
    let report = refit_digit_model(
        &mut model,
        &dataset.features,
        &dataset.labels,
        &indices,
        &artifact.candidate,
        epochs,
        artifact.selected_epoch,
        "continue",
        None,
    )?;
    let continued = DigitModelArtifact {
        selected_epoch: artifact.selected_epoch + epochs,
        model,
        ..artifact.clone()
    };
    continued.save(&output.join("selected_model.toml"))?;
    write_selected_model(&continued, output)?;
    let mut writer = csv::Writer::from_path(output.join("continue_history.csv"))?;
    writer.write_record(["epoch", "train_loss", "train_accuracy"])?;
    for row in &report.history {
        writer.serialize((row.epoch, row.train_loss, row.train_accuracy))?;
    }
    writer.flush()?;
    eprintln!(
        "continued '{}' from epoch {} to {} (train_accuracy={:.4})",
        continued.candidate.name,
        artifact.selected_epoch,
        continued.selected_epoch,
        report
            .history
            .last()
            .map_or(f64::NAN, |row| row.train_accuracy)
    );
    Ok(continued)
}

/// Class counts of two datasets and how many images of the new one already
/// appear (pixel-for-pixel) in the previous one.
pub fn write_data_shift(
    previous: &DigitDataset,
    current: &DigitDataset,
    output: &Path,
) -> Result<()> {
    use std::collections::HashSet;

    use super::data::{class_counts, DIGIT_CLASSES};

    let signature = |dataset: &DigitDataset, row: usize| {
        dataset
            .features
            .row(row)
            .unwrap()
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
    };
    let known = (0..previous.features.rows())
        .map(|row| signature(previous, row))
        .collect::<HashSet<_>>();
    let shared = (0..current.features.rows())
        .filter(|&row| known.contains(&signature(current, row)))
        .count();
    let all = |dataset: &DigitDataset| (0..dataset.labels.len()).collect::<Vec<_>>();
    let before = class_counts(&previous.labels, &all(previous));
    let after = class_counts(&current.labels, &all(current));
    let mut writer = csv::Writer::from_path(output.join("data_shift.csv"))?;
    writer.write_record(["class", "previous_rows", "current_rows", "added_rows"])?;
    for class in 0..DIGIT_CLASSES {
        writer.serialize((
            class.to_string(),
            before[class],
            after[class],
            after[class] as i64 - before[class] as i64,
        ))?;
    }
    writer.serialize((
        "total",
        previous.labels.len(),
        current.labels.len(),
        current.labels.len() as i64 - previous.labels.len() as i64,
    ))?;
    writer.serialize(("images_shared_with_previous", shared, shared, 0))?;
    writer.flush()?;
    Ok(())
}

pub fn expected_image_pixels() -> usize {
    IMAGE_PIXELS
}
