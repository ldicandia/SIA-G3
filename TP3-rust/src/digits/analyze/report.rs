//! CSV report writers for the digit-classifier study.

use std::path::Path;

use anyhow::Result;

use super::super::{
    artifact::DigitModelArtifact,
    data::{class_counts, DigitDataset, DIGIT_CLASSES},
    metrics::ClassificationMetrics,
};
use super::CandidateRun;

pub(super) fn write_dataset_summary(
    dataset: &DigitDataset,
    train: &[usize],
    validation: &[usize],
    output: &Path,
) -> Result<()> {
    let all = (0..dataset.labels.len()).collect::<Vec<_>>();
    let all_counts = class_counts(&dataset.labels, &all);
    let train_counts = class_counts(&dataset.labels, train);
    let validation_counts = class_counts(&dataset.labels, validation);
    let mut writer = csv::Writer::from_path(output.join("dataset_summary.csv"))?;
    writer.write_record(["class", "all", "train", "validation"])?;
    for class in 0..DIGIT_CLASSES {
        writer.serialize((
            class,
            all_counts[class],
            train_counts[class],
            validation_counts[class],
        ))?;
    }
    writer.flush()?;
    let mut profile = csv::Writer::from_path(output.join("dataset_profile.csv"))?;
    profile.write_record(["metric", "value"])?;
    profile.serialize(("rows", dataset.features.rows() as f64))?;
    profile.serialize(("pixels_per_image", dataset.features.cols() as f64))?;
    profile.serialize((
        "minimum_pixel",
        dataset
            .features
            .as_slice()
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min),
    ))?;
    profile.serialize((
        "maximum_pixel",
        dataset
            .features
            .as_slice()
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max),
    ))?;
    profile.serialize(("missing_or_non_finite_values", 0.0))?;
    profile.flush()?;
    Ok(())
}

pub(super) fn write_candidate_summary(runs: &[CandidateRun], output: &Path) -> Result<()> {
    let mut writer = csv::Writer::from_path(output.join("candidate_summary.csv"))?;
    writer.write_record([
        "name",
        "topology",
        "hidden_activation",
        "optimizer",
        "learning_rate",
        "batch_size",
        "parameter_count",
        "best_epoch",
        "validation_loss",
        "validation_accuracy",
        "stopped_early",
    ])?;
    for run in runs {
        let candidate = &run.result.candidate;
        writer.serialize((
            &candidate.name,
            topology_text(&candidate.topology),
            format!("{:?}", candidate.hidden_activation).to_lowercase(),
            format!("{:?}", candidate.optimizer).to_lowercase(),
            candidate.learning_rate,
            candidate.batch_size,
            candidate.parameter_count(),
            run.result.best_epoch,
            run.result.validation_loss,
            run.result.validation_accuracy,
            run.result.stopped_early,
        ))?;
    }
    writer.flush()?;
    Ok(())
}

pub(super) fn write_learning_history(runs: &[CandidateRun], output: &Path) -> Result<()> {
    let mut writer = csv::Writer::from_path(output.join("learning_history.csv"))?;
    writer.write_record([
        "candidate",
        "epoch",
        "train_loss",
        "validation_loss",
        "train_accuracy",
        "validation_accuracy",
    ])?;
    for run in runs {
        for row in &run.report.history {
            writer.serialize((
                &run.result.candidate.name,
                row.epoch,
                row.train_loss,
                row.validation_loss,
                row.train_accuracy,
                row.validation_accuracy,
            ))?;
        }
    }
    writer.flush()?;
    Ok(())
}

pub(super) fn write_selected_model(artifact: &DigitModelArtifact, output: &Path) -> Result<()> {
    let mut writer = csv::Writer::from_path(output.join("selected_model.csv"))?;
    writer.write_record(["field", "value"])?;
    for (field, value) in [
        ("candidate", artifact.candidate.name.clone()),
        ("topology", topology_text(&artifact.candidate.topology)),
        (
            "hidden_activation",
            format!("{:?}", artifact.candidate.hidden_activation).to_lowercase(),
        ),
        (
            "optimizer",
            format!("{:?}", artifact.candidate.optimizer).to_lowercase(),
        ),
        (
            "learning_rate",
            artifact.candidate.learning_rate.to_string(),
        ),
        ("batch_size", artifact.candidate.batch_size.to_string()),
        ("selected_epoch", artifact.selected_epoch.to_string()),
        (
            "validation_loss",
            artifact.selection_validation_loss.to_string(),
        ),
        (
            "validation_accuracy",
            artifact.selection_validation_accuracy.to_string(),
        ),
    ] {
        writer.write_record([field, &value])?;
    }
    writer.flush()?;
    Ok(())
}

pub(super) fn write_evaluation_metrics(
    artifact: &DigitModelArtifact,
    loss: f64,
    metrics: &ClassificationMetrics,
    output: &Path,
) -> Result<()> {
    let mut writer = csv::Writer::from_path(output.join("evaluation_metrics.csv"))?;
    writer.write_record(["metric", "value"])?;
    writer.serialize(("loss", loss))?;
    writer.serialize(("accuracy", metrics.accuracy))?;
    if artifact.exercise == "exercise3" {
        writer.serialize(("meets_98_percent_target", metrics.accuracy >= 0.98))?;
    }
    writer.serialize(("selected_epoch", artifact.selected_epoch as f64))?;
    for class in 0..DIGIT_CLASSES {
        if let Some(recall) = metrics.per_class_recall[class] {
            writer.serialize((format!("class_{class}_recall"), recall))?;
        } else {
            writer.write_record([format!("class_{class}_recall"), "undefined".into()])?;
        }
    }
    writer.flush()?;
    Ok(())
}

pub(super) fn write_predictions(
    labels: &[usize],
    predictions: &[usize],
    path: &Path,
) -> Result<()> {
    let mut writer = csv::Writer::from_path(path)?;
    writer.write_record(["row", "label", "prediction", "correct"])?;
    for (row, (&label, &prediction)) in labels.iter().zip(predictions).enumerate() {
        writer.serialize((row, label, prediction, label == prediction))?;
    }
    writer.flush()?;
    Ok(())
}

pub(super) fn write_confusion_matrix(metrics: &ClassificationMetrics, path: &Path) -> Result<()> {
    let mut writer = csv::Writer::from_path(path)?;
    let mut header = vec!["actual\\predicted".to_owned()];
    header.extend((0..DIGIT_CLASSES).map(|class| class.to_string()));
    writer.write_record(header)?;
    for actual in 0..DIGIT_CLASSES {
        let mut row = vec![actual.to_string()];
        row.extend(metrics.confusion[actual].iter().map(ToString::to_string));
        writer.write_record(row)?;
    }
    writer.flush()?;
    Ok(())
}

fn topology_text(topology: &[usize]) -> String {
    topology
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join("-")
}
