//! CSV report writers for the fraud-distillation experiments.

use std::{collections::HashSet, path::Path};

use anyhow::Result;

use crate::{
    config::FeatureConfig,
    data::FraudDataset,
    metrics::{mean_std, ConfusionMatrix, RegressionMetrics, ThresholdMetrics},
    split::{Fold, HoldoutSplit},
};

use super::{
    trials::{FoldTrial, LearningRun},
    FoldClassification, LearningRateSummary,
};

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
    // How well the teacher itself separates the ground truth: if the largest
    // legitimate score is below the smallest fraud score, BigModel ranks perfectly.
    let teacher_extreme = |fraud: bool, pick: fn(f64, f64) -> f64, start: f64| {
        dataset
            .teacher_targets
            .iter()
            .zip(&dataset.fraud_labels)
            .filter(|&(_, &label)| label == fraud)
            .fold(start, |acc, (&score, _)| pick(acc, score))
    };
    writer.serialize((
        "teacher_max_legitimate",
        teacher_extreme(false, f64::max, f64::NEG_INFINITY),
    ))?;
    writer.serialize((
        "teacher_min_fraud",
        teacher_extreme(true, f64::min, f64::INFINITY),
    ))?;
    writer.flush()?;
    Ok(())
}

pub(super) fn write_data_profile(dataset: &FraudDataset, path: &Path) -> Result<()> {
    let mut writer = csv::Writer::from_path(path)?;
    writer.write_record([
        "column", "count", "missing", "unique", "min", "q25", "median", "q75", "max", "mean",
        "std_dev",
    ])?;
    for (column, name) in dataset.feature_names.iter().enumerate() {
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
        // Sigmoid: share of saturated outputs (flat gradient). Linear/ReLU:
        // share of outputs that are not valid probabilities.
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
    holdout: &HoldoutSplit,
    folds: &[Fold],
    path: &Path,
) -> Result<()> {
    let mut writer = csv::Writer::from_path(path)?;
    writer.write_record(["split", "rows", "teacher_mean", "fraud_count", "fraud_rate"])?;
    let mut groups = vec![
        ("development".to_owned(), &holdout.development),
        ("test".to_owned(), &holdout.test),
    ];
    for (index, fold) in folds.iter().enumerate() {
        groups.push((format!("fold_{index}_validation"), &fold.validation));
    }
    for (name, indices) in groups {
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

pub(super) fn write_cv_trials(trials: &[FoldTrial], path: &Path) -> Result<()> {
    let mut writer = csv::Writer::from_path(path)?;
    writer.write_record([
        "learning_rate",
        "fold",
        "best_epoch",
        "train_mse",
        "validation_mse",
        "validation_rmse",
        "validation_r2",
        "stopped_early",
    ])?;
    for trial in trials {
        writer.serialize((
            trial.learning_rate,
            trial.fold,
            trial.report.best_epoch,
            trial.train_mse,
            trial.validation_metrics.mse,
            trial.validation_metrics.rmse,
            trial.validation_metrics.r2,
            trial.report.stopped_early,
        ))?;
    }
    writer.flush()?;
    Ok(())
}

pub(super) fn write_cv_summary(summaries: &[LearningRateSummary], path: &Path) -> Result<()> {
    let mut writer = csv::Writer::from_path(path)?;
    writer.write_record([
        "learning_rate",
        "train_mse_mean",
        "train_mse_std",
        "validation_mse_mean",
        "validation_mse_std",
        "validation_r2_mean",
        "validation_r2_std",
        "median_best_epoch",
    ])?;
    for row in summaries {
        writer.serialize((
            row.learning_rate,
            row.train_mse.0,
            row.train_mse.1,
            row.validation_mse.0,
            row.validation_mse.1,
            row.validation_r2.0,
            row.validation_r2.1,
            row.median_epoch,
        ))?;
    }
    writer.flush()?;
    Ok(())
}

pub(super) fn write_fold_metrics(rows: &[FoldClassification], path: &Path) -> Result<()> {
    let mut writer = csv::Writer::from_path(path)?;
    writer.write_record([
        "fold",
        "mse",
        "r2",
        "precision",
        "recall",
        "f1",
        "accuracy",
        "average_precision",
    ])?;
    let columns: [fn(&FoldClassification) -> f64; 7] = [
        |row| row.mse,
        |row| row.r2,
        |row| row.precision,
        |row| row.recall,
        |row| row.f1,
        |row| row.accuracy,
        |row| row.average_precision,
    ];
    for row in rows {
        let mut record = vec![row.fold.to_string()];
        record.extend(columns.iter().map(|column| column(row).to_string()));
        writer.write_record(record)?;
    }
    let stats = columns
        .iter()
        .map(|column| mean_std(&rows.iter().map(column).collect::<Vec<_>>()))
        .collect::<Vec<_>>();
    let mut mean_row = vec!["mean".to_owned()];
    mean_row.extend(stats.iter().map(|(mean, _)| mean.to_string()));
    writer.write_record(mean_row)?;
    let mut std_row = vec!["std".to_owned()];
    std_row.extend(stats.iter().map(|(_, std)| std.to_string()));
    writer.write_record(std_row)?;
    writer.flush()?;
    Ok(())
}

pub(super) fn write_feature_analysis(
    dataset: &FraudDataset,
    features: &FeatureConfig,
    path: &Path,
) -> Result<()> {
    let labels = dataset
        .fraud_labels
        .iter()
        .map(|&label| f64::from(u8::from(label)))
        .collect::<Vec<_>>();
    let mut writer = csv::Writer::from_path(path)?;
    writer.write_record([
        "column",
        "action",
        "skewness",
        "pearson_teacher",
        "pearson_teacher_log1p",
        "pearson_flagged_fraud",
    ])?;
    for (column, name) in dataset.feature_names.iter().enumerate() {
        let values = (0..dataset.len())
            .map(|row| dataset.features.row_unchecked(row)[column])
            .collect::<Vec<_>>();
        let logged = values.iter().map(|value| value.ln_1p()).collect::<Vec<_>>();
        let action = if features.drop.contains(name) {
            "drop"
        } else if features.log1p.contains(name) {
            "log1p"
        } else {
            "keep"
        };
        writer.serialize((
            name,
            action,
            skewness(&values),
            pearson(&values, &dataset.teacher_targets),
            pearson(&logged, &dataset.teacher_targets),
            pearson(&values, &labels),
        ))?;
    }
    writer.flush()?;
    Ok(())
}

/// BigModel mean and fraud rate by hour of day and weekday, used to decide
/// whether the raw timestamp carries a usable signal.
pub(super) fn write_time_profile(dataset: &FraudDataset, path: &Path) -> Result<()> {
    let Some(column) = dataset
        .feature_names
        .iter()
        .position(|name| name == "timestamp")
    else {
        return Ok(());
    };
    let mut hours = vec![(0usize, 0.0, 0usize); 24];
    let mut weekdays = vec![(0usize, 0.0, 0usize); 7];
    for row in 0..dataset.len() {
        let seconds = dataset.features.row_unchecked(row)[column] as i64;
        let days = seconds.div_euclid(86_400);
        let hour = (seconds.rem_euclid(86_400) / 3_600) as usize;
        // 1970-01-01 was a Thursday; 0 = Monday.
        let weekday = (days + 3).rem_euclid(7) as usize;
        for (bucket, index) in [(&mut hours, hour), (&mut weekdays, weekday)] {
            bucket[index].0 += 1;
            bucket[index].1 += dataset.teacher_targets[row];
            bucket[index].2 += usize::from(dataset.fraud_labels[row]);
        }
    }
    let mut writer = csv::Writer::from_path(path)?;
    writer.write_record(["kind", "value", "rows", "teacher_mean", "fraud_rate"])?;
    for (kind, bucket) in [("hour", &hours), ("weekday", &weekdays)] {
        for (value, &(rows, teacher, frauds)) in bucket.iter().enumerate() {
            let count = rows.max(1) as f64;
            writer.serialize((kind, value, rows, teacher / count, frauds as f64 / count))?;
        }
    }
    writer.flush()?;
    Ok(())
}

fn pearson(left: &[f64], right: &[f64]) -> f64 {
    let (left_mean, left_std) = mean_std(left);
    let (right_mean, right_std) = mean_std(right);
    if left_std <= f64::EPSILON || right_std <= f64::EPSILON {
        return 0.0;
    }
    left.iter()
        .zip(right)
        .map(|(a, b)| (a - left_mean) * (b - right_mean))
        .sum::<f64>()
        / (left.len() as f64 * left_std * right_std)
}

fn skewness(values: &[f64]) -> f64 {
    let (mean, std) = mean_std(values);
    if std <= f64::EPSILON {
        return 0.0;
    }
    values
        .iter()
        .map(|value| ((value - mean) / std).powi(3))
        .sum::<f64>()
        / values.len() as f64
}

pub(super) fn write_threshold_sweep(sweep: &[ThresholdMetrics], path: &Path) -> Result<()> {
    let mut writer = csv::Writer::from_path(path)?;
    writer.write_record([
        "threshold",
        "precision",
        "recall",
        "f1",
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
            row.f1(),
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
        "threshold_criterion",
    ])?;
    writer.serialize((
        "single_layer_perceptron",
        "sigmoid",
        "mse",
        learning_rate,
        epochs,
        threshold,
        "max_f1_out_of_fold",
    ))?;
    writer.flush()?;
    Ok(())
}

pub(super) fn write_test_metrics(
    regression: RegressionMetrics,
    confusion: ConfusionMatrix,
    average_precision: f64,
    teacher_average_precision: f64,
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
        ("fraud", "f1", confusion.f1()),
        ("fraud", "average_precision", average_precision),
        (
            "fraud",
            "bigmodel_average_precision",
            teacher_average_precision,
        ),
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
