//! CSV report writers for the fraud-distillation experiments.

use std::{collections::HashSet, path::Path};

use anyhow::Result;

use crate::{
    data::{FraudDataset, FEATURE_NAMES},
    metrics::{ConfusionMatrix, RegressionMetrics, ThresholdMetrics},
    split::DatasetSplit,
};

use super::trials::LearningRun;

pub(super) fn write_dataset_summary(dataset: &FraudDataset, path: &Path) -> Result<()> {
    let fraud_count = dataset.fraud_labels.iter().filter(|&&label| label).count();
    let mut writer = csv::Writer::from_path(path)?;
    writer.write_record(["metric", "value"])?;
    writer.serialize(("rows", dataset.len()))?;
    writer.serialize(("features", dataset.features.cols()))?;
    writer.serialize(("missing_values", 0))?;
    writer.serialize(("exact_duplicate_rows", dataset.exact_duplicate_count()))?;
    writer.serialize(("fraud_count", fraud_count))?;
    writer.serialize(("fraud_rate", fraud_count as f64 / dataset.len() as f64))?;
    writer.flush()?;
    Ok(())
}

pub(super) fn write_data_profile(dataset: &FraudDataset, path: &Path) -> Result<()> {
    let mut writer = csv::Writer::from_path(path)?;
    writer.write_record([
        "column", "count", "missing", "unique", "min", "q25", "median", "q75", "max", "mean",
        "std_dev",
    ])?;
    for (column, name) in FEATURE_NAMES.iter().enumerate() {
        let values = (0..dataset.len())
            .map(|row| dataset.features.row_unchecked(row)[column])
            .collect::<Vec<_>>();
        writer.serialize(profile_row(name, &values))?;
    }
    writer.serialize(profile_row(
        "big_model_fraud_probability",
        &dataset.teacher_targets,
    ))?;
    let labels = dataset
        .fraud_labels
        .iter()
        .map(|&label| u8::from(label) as f64)
        .collect::<Vec<_>>();
    writer.serialize(profile_row("flagged_fraud", &labels))?;
    writer.flush()?;
    Ok(())
}

fn profile_row<'a>(
    name: &'a str,
    values: &[f64],
) -> (
    &'a str,
    usize,
    usize,
    usize,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
) {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let std_dev = (values
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / values.len() as f64)
        .sqrt();
    let unique = values
        .iter()
        .map(|value| value.to_bits())
        .collect::<HashSet<_>>()
        .len();
    (
        name,
        values.len(),
        0,
        unique,
        sorted[0],
        quantile(&sorted, 0.25),
        quantile(&sorted, 0.5),
        quantile(&sorted, 0.75),
        *sorted.last().unwrap(),
        mean,
        std_dev,
    )
}

fn quantile(sorted: &[f64], probability: f64) -> f64 {
    let position = probability * (sorted.len() - 1) as f64;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    if lower == upper {
        sorted[lower]
    } else {
        let fraction = position - lower as f64;
        sorted[lower] * (1.0 - fraction) + sorted[upper] * fraction
    }
}

pub(super) fn write_learning_summary(runs: &[LearningRun], path: &Path) -> Result<()> {
    let mut writer = csv::Writer::from_path(path)?;
    writer.write_record([
        "model",
        "learning_rate",
        "best_epoch",
        "mse",
        "rmse",
        "r2",
        "prediction_min",
        "prediction_max",
        "boundary_rate",
    ])?;
    for run in runs {
        let minimum = run
            .predictions
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min);
        let maximum = run
            .predictions
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        let boundary_count = if run.name == "sigmoid" {
            run.predictions
                .iter()
                .filter(|&&prediction| prediction <= 0.01 || prediction >= 0.99)
                .count()
        } else {
            run.predictions
                .iter()
                .filter(|&&prediction| !(0.0..=1.0).contains(&prediction))
                .count()
        };
        writer.serialize((
            run.name,
            run.learning_rate,
            run.report.best_epoch,
            run.metrics.mse,
            run.metrics.rmse,
            run.metrics.r2,
            minimum,
            maximum,
            boundary_count as f64 / run.predictions.len() as f64,
        ))?;
    }
    writer.flush()?;
    Ok(())
}

pub(super) fn write_learning_history(runs: &[LearningRun], path: &Path) -> Result<()> {
    let mut writer = csv::Writer::from_path(path)?;
    writer.write_record(["model", "learning_rate", "epoch", "train_mse"])?;
    for run in runs {
        for epoch in &run.report.history {
            writer.serialize((run.name, run.learning_rate, epoch.epoch, epoch.train_loss))?;
        }
    }
    writer.flush()?;
    Ok(())
}

pub(super) fn write_split_summary(
    dataset: &FraudDataset,
    split: &DatasetSplit,
    path: &Path,
) -> Result<()> {
    let mut writer = csv::Writer::from_path(path)?;
    writer.write_record(["split", "rows", "teacher_mean", "fraud_count", "fraud_rate"])?;
    for (name, indices) in [
        ("train", &split.train),
        ("validation", &split.validation),
        ("test", &split.test),
    ] {
        let teacher_mean = indices
            .iter()
            .map(|&index| dataset.teacher_targets[index])
            .sum::<f64>()
            / indices.len() as f64;
        let fraud_count = indices
            .iter()
            .filter(|&&index| dataset.fraud_labels[index])
            .count();
        writer.serialize((
            name,
            indices.len(),
            teacher_mean,
            fraud_count,
            fraud_count as f64 / indices.len() as f64,
        ))?;
    }
    writer.flush()?;
    Ok(())
}

pub(super) fn write_threshold_sweep(sweep: &[ThresholdMetrics], path: &Path) -> Result<()> {
    let mut writer = csv::Writer::from_path(path)?;
    writer.write_record([
        "threshold",
        "precision",
        "recall",
        "accuracy",
        "true_positives",
        "false_positives",
        "true_negatives",
        "false_negatives",
    ])?;
    for row in sweep {
        writer.serialize((
            row.threshold,
            row.precision(),
            row.recall(),
            row.accuracy(),
            row.confusion.true_positives,
            row.confusion.false_positives,
            row.confusion.true_negatives,
            row.confusion.false_negatives,
        ))?;
    }
    writer.flush()?;
    Ok(())
}

pub(super) fn write_selected_model(
    learning_rate: f64,
    epochs: usize,
    threshold: f64,
    path: &Path,
) -> Result<()> {
    let mut writer = csv::Writer::from_path(path)?;
    writer.write_record([
        "model",
        "activation",
        "loss",
        "learning_rate",
        "epochs",
        "threshold",
    ])?;
    writer.serialize((
        "single_layer_perceptron",
        "sigmoid",
        "mse",
        learning_rate,
        epochs,
        threshold,
    ))?;
    writer.flush()?;
    Ok(())
}

pub(super) fn write_test_metrics(
    regression: RegressionMetrics,
    confusion: ConfusionMatrix,
    path: &Path,
) -> Result<()> {
    let mut writer = csv::Writer::from_path(path)?;
    writer.write_record(["category", "metric", "value"])?;
    for (category, metric, value) in [
        ("distillation", "mse", regression.mse),
        ("distillation", "rmse", regression.rmse),
        ("distillation", "r2", regression.r2),
        ("fraud", "precision", confusion.precision()),
        ("fraud", "recall", confusion.recall()),
        ("fraud", "specificity", confusion.specificity()),
        ("fraud", "accuracy", confusion.accuracy()),
        ("fraud", "true_positives", confusion.true_positives as f64),
        ("fraud", "false_positives", confusion.false_positives as f64),
        ("fraud", "true_negatives", confusion.true_negatives as f64),
        ("fraud", "false_negatives", confusion.false_negatives as f64),
    ] {
        writer.serialize((category, metric, value))?;
    }
    writer.flush()?;
    Ok(())
}

pub(super) fn write_test_predictions(
    indices: &[usize],
    targets: &[f64],
    predictions: &[f64],
    labels: &[bool],
    threshold: f64,
    path: &Path,
) -> Result<()> {
    let mut writer = csv::Writer::from_path(path)?;
    writer.write_record([
        "row_index",
        "big_model_probability",
        "tiny_model_probability",
        "flagged_fraud",
        "predicted_fraud",
    ])?;
    for (((&index, &target), &prediction), &label) in
        indices.iter().zip(targets).zip(predictions).zip(labels)
    {
        writer.serialize((
            index,
            target,
            prediction,
            u8::from(label),
            u8::from(prediction >= threshold),
        ))?;
    }
    writer.flush()?;
    Ok(())
}
