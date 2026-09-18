//! Reproducible experiment orchestration and artifact generation.

use std::{collections::HashSet, fs, path::Path};

use anyhow::{bail, Context, Result};
use plotters::prelude::*;

use crate::{
    config::AppConfig,
    data::{FraudDataset, StandardScaler, FEATURE_NAMES},
    loss::MeanSquaredError,
    matrix::DenseMatrix,
    metrics::{
        best_accuracy_threshold, confusion_matrix, regression_metrics, threshold_sweep,
        RegressionMetrics, ThresholdMetrics,
    },
    model::{Activation, SingleLayerPerceptron},
    split::stratified_split,
    training::{
        predict_single_indices, train_single_layer, EarlyStopping, EpochMetrics, TrainingConfig,
        TrainingReport,
    },
};

const BLUE: RGBColor = RGBColor(41, 98, 255);
const ORANGE: RGBColor = RGBColor(245, 124, 0);
const GREEN: RGBColor = RGBColor(0, 137, 123);
const RED: RGBColor = RGBColor(211, 47, 47);

pub fn run_inspection(dataset: &FraudDataset, output: &Path) -> Result<()> {
    fs::create_dir_all(output)?;
    write_dataset_summary(dataset, &output.join("dataset_summary.csv"))?;
    write_data_profile(dataset, &output.join("data_profile.csv"))?;
    plot_feature_distributions(dataset, &output.join("feature_distributions.png"))?;
    plot_targets(dataset, &output.join("target_distributions.png"))?;
    Ok(())
}

pub fn run_learning_comparison(
    dataset: &FraudDataset,
    config: &AppConfig,
    output: &Path,
) -> Result<()> {
    fs::create_dir_all(output)?;
    let all_indices = (0..dataset.len()).collect::<Vec<_>>();
    let scaler = StandardScaler::fit(&dataset.features, &all_indices)?;
    let features = scaler.transform(&dataset.features)?;
    let mut runs = Vec::new();
    let mut trial_writer = csv::Writer::from_path(output.join("learning_trials.csv"))?;
    trial_writer.write_record([
        "model",
        "learning_rate",
        "best_epoch",
        "mse",
        "rmse",
        "r2",
        "stopped_early",
    ])?;

    for (name, activation) in [
        ("linear", Activation::Linear),
        ("sigmoid", Activation::Sigmoid),
    ] {
        let trials = run_learning_trials(
            name,
            activation,
            &config.search.learning_rates,
            &features,
            &all_indices,
            &dataset.teacher_targets,
            config,
        )?;
        let mut best: Option<LearningRun> = None;
        for run in trials {
            trial_writer.serialize((
                run.name,
                run.learning_rate,
                run.report.best_epoch,
                run.metrics.mse,
                run.metrics.rmse,
                run.metrics.r2,
                run.report.stopped_early,
            ))?;
            if best
                .as_ref()
                .is_none_or(|current| run.metrics.mse < current.metrics.mse)
            {
                best = Some(run);
            }
        }
        runs.push(best.context("no learning-rate candidates were configured")?);
    }
    trial_writer.flush()?;

    write_learning_summary(&runs, &output.join("learning_summary.csv"))?;
    write_learning_history(&runs, &output.join("learning_history.csv"))?;
    plot_learning_history(&runs, &output.join("learning_curves.png"))?;
    plot_prediction_comparison(
        &runs,
        &dataset.teacher_targets,
        &output.join("prediction_comparison.png"),
    )?;
    Ok(())
}

pub fn run_generalization(dataset: &FraudDataset, config: &AppConfig, output: &Path) -> Result<()> {
    fs::create_dir_all(output)?;
    let split = stratified_split(
        &dataset.teacher_targets,
        config.split.train_ratio,
        config.split.validation_ratio,
        config.split.stratification_bins,
        config.split.seed,
    )?;
    write_split_summary(dataset, &split, &output.join("split_summary.csv"))?;

    let train_scaler = StandardScaler::fit(&dataset.features, &split.train)?;
    let train_scaled = train_scaler.transform(&dataset.features)?;
    let mut trial_writer = csv::Writer::from_path(output.join("generalization_trials.csv"))?;
    trial_writer.write_record([
        "learning_rate",
        "best_epoch",
        "train_mse",
        "validation_mse",
        "validation_rmse",
        "validation_r2",
        "stopped_early",
    ])?;

    let trials = run_generalization_trials(
        &config.search.learning_rates,
        &train_scaled,
        &split.train,
        &split.validation,
        &dataset.teacher_targets,
        config,
    )?;
    let mut best: Option<GeneralizationRun> = None;
    for trial in trials {
        trial_writer.serialize((
            trial.run.learning_rate,
            trial.run.report.best_epoch,
            trial.train_mse,
            trial.run.validation_metrics.mse,
            trial.run.validation_metrics.rmse,
            trial.run.validation_metrics.r2,
            trial.run.report.stopped_early,
        ))?;
        if best
            .as_ref()
            .is_none_or(|current| trial.run.validation_metrics.mse < current.validation_metrics.mse)
        {
            best = Some(trial.run);
        }
    }
    trial_writer.flush()?;
    let best = best.context("no learning-rate candidates were configured")?;

    let validation_labels = select_values(&dataset.fraud_labels, &split.validation);
    let sweep = threshold_sweep(&best.validation_predictions, &validation_labels)?;
    let chosen = best_accuracy_threshold(&sweep).context("threshold sweep was empty")?;
    let chosen_threshold = chosen.threshold;
    write_threshold_sweep(&sweep, &output.join("threshold_sweep.csv"))?;

    let mut development_indices = split.train.clone();
    development_indices.extend_from_slice(&split.validation);
    let final_scaler = StandardScaler::fit(&dataset.features, &development_indices)?;
    let final_features = final_scaler.transform(&dataset.features)?;
    let mut final_model = SingleLayerPerceptron::new(
        final_features.cols(),
        Activation::Sigmoid,
        config.search.initialization_seed,
    )?;
    train_single_layer(
        &mut final_model,
        &final_features,
        &dataset.teacher_targets,
        &development_indices,
        None,
        &MeanSquaredError,
        TrainingConfig::fixed(
            best.learning_rate,
            best.report.best_epoch,
            config.search.initialization_seed,
        ),
    )?;

    let test_predictions = predict_single_indices(&final_model, &final_features, &split.test)?;
    let test_targets = select_values(&dataset.teacher_targets, &split.test);
    let test_labels = select_values(&dataset.fraud_labels, &split.test);
    let regression = regression_metrics(&test_predictions, &test_targets)?;
    let confusion = confusion_matrix(&test_predictions, &test_labels, chosen_threshold)?;
    write_selected_model(
        best.learning_rate,
        best.report.best_epoch,
        chosen_threshold,
        &output.join("selected_model.csv"),
    )?;
    write_test_metrics(regression, confusion, &output.join("test_metrics.csv"))?;
    write_test_predictions(
        &split.test,
        &test_targets,
        &test_predictions,
        &test_labels,
        chosen_threshold,
        &output.join("test_predictions.csv"),
    )?;
    plot_generalization_history(
        &best.report.history,
        &output.join("generalization_learning_curve.png"),
    )?;
    plot_threshold_metrics(&sweep, &output.join("threshold_metrics.png"))?;
    Ok(())
}

fn run_learning_trial(
    name: &'static str,
    activation: Activation,
    learning_rate: f64,
    features: &DenseMatrix,
    all_indices: &[usize],
    teacher_targets: &[f64],
    config: &AppConfig,
) -> Result<LearningRun> {
    let mut model = SingleLayerPerceptron::new(
        features.cols(),
        activation,
        config.search.initialization_seed,
    )?;
    let report = train_single_layer(
        &mut model,
        features,
        teacher_targets,
        all_indices,
        None,
        &MeanSquaredError,
        search_training_config(config, learning_rate),
    )?;
    let predictions = predict_single_indices(&model, features, all_indices)?;
    let metrics = regression_metrics(&predictions, teacher_targets)?;
    Ok(LearningRun {
        name,
        learning_rate,
        report,
        predictions,
        metrics,
    })
}

#[allow(clippy::too_many_arguments)]
fn run_learning_trials(
    name: &'static str,
    activation: Activation,
    learning_rates: &[f64],
    features: &DenseMatrix,
    all_indices: &[usize],
    teacher_targets: &[f64],
    config: &AppConfig,
) -> Result<Vec<LearningRun>> {
    use rayon::prelude::*;

    learning_rates
        .par_iter()
        .map(|&learning_rate| {
            run_learning_trial(
                name,
                activation,
                learning_rate,
                features,
                all_indices,
                teacher_targets,
                config,
            )
        })
        .collect()
}

struct LearningRun {
    name: &'static str,
    learning_rate: f64,
    report: TrainingReport,
    predictions: Vec<f64>,
    metrics: RegressionMetrics,
}

struct GeneralizationRun {
    learning_rate: f64,
    report: TrainingReport,
    validation_predictions: Vec<f64>,
    validation_metrics: RegressionMetrics,
}

struct GeneralizationTrial {
    run: GeneralizationRun,
    train_mse: f64,
}

#[allow(clippy::too_many_arguments)]
fn run_generalization_trial(
    learning_rate: f64,
    train_scaled: &DenseMatrix,
    train_indices: &[usize],
    validation_indices: &[usize],
    teacher_targets: &[f64],
    config: &AppConfig,
) -> Result<GeneralizationTrial> {
    let mut model = SingleLayerPerceptron::new(
        train_scaled.cols(),
        Activation::Sigmoid,
        config.search.initialization_seed,
    )?;
    let report = train_single_layer(
        &mut model,
        train_scaled,
        teacher_targets,
        train_indices,
        Some(validation_indices),
        &MeanSquaredError,
        search_training_config(config, learning_rate),
    )?;
    let train_predictions = predict_single_indices(&model, train_scaled, train_indices)?;
    let validation_predictions = predict_single_indices(&model, train_scaled, validation_indices)?;
    let train_targets = select_values(teacher_targets, train_indices);
    let validation_targets = select_values(teacher_targets, validation_indices);
    let train_mse = regression_metrics(&train_predictions, &train_targets)?.mse;
    let validation_metrics = regression_metrics(&validation_predictions, &validation_targets)?;
    Ok(GeneralizationTrial {
        run: GeneralizationRun {
            learning_rate,
            report,
            validation_predictions,
            validation_metrics,
        },
        train_mse,
    })
}

#[allow(clippy::too_many_arguments)]
fn run_generalization_trials(
    learning_rates: &[f64],
    train_scaled: &DenseMatrix,
    train_indices: &[usize],
    validation_indices: &[usize],
    teacher_targets: &[f64],
    config: &AppConfig,
) -> Result<Vec<GeneralizationTrial>> {
    use rayon::prelude::*;

    learning_rates
        .par_iter()
        .map(|&learning_rate| {
            run_generalization_trial(
                learning_rate,
                train_scaled,
                train_indices,
                validation_indices,
                teacher_targets,
                config,
            )
        })
        .collect()
}

fn search_training_config(config: &AppConfig, learning_rate: f64) -> TrainingConfig {
    TrainingConfig {
        learning_rate,
        max_epochs: config.search.max_epochs,
        shuffle: true,
        seed: config.search.initialization_seed,
        early_stopping: Some(EarlyStopping {
            patience: config.search.patience,
            min_delta: config.search.min_delta,
        }),
    }
}

fn select_values<T: Copy>(values: &[T], indices: &[usize]) -> Vec<T> {
    indices.iter().map(|&index| values[index]).collect()
}

fn write_dataset_summary(dataset: &FraudDataset, path: &Path) -> Result<()> {
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

fn write_data_profile(dataset: &FraudDataset, path: &Path) -> Result<()> {
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

fn write_learning_summary(runs: &[LearningRun], path: &Path) -> Result<()> {
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

fn write_learning_history(runs: &[LearningRun], path: &Path) -> Result<()> {
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

fn write_split_summary(
    dataset: &FraudDataset,
    split: &crate::split::DatasetSplit,
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

fn write_threshold_sweep(sweep: &[ThresholdMetrics], path: &Path) -> Result<()> {
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

fn write_selected_model(
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

fn write_test_metrics(
    regression: RegressionMetrics,
    confusion: crate::metrics::ConfusionMatrix,
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

fn write_test_predictions(
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

fn plot_feature_distributions(dataset: &FraudDataset, path: &Path) -> Result<()> {
    let root = BitMapBackend::new(path, (1500, 1100)).into_drawing_area();
    root.fill(&WHITE)?;
    for (column, area) in root.split_evenly((3, 3)).into_iter().enumerate() {
        let values = (0..dataset.len())
            .map(|row| dataset.features.row_unchecked(row)[column])
            .collect::<Vec<_>>();
        draw_histogram(&area, FEATURE_NAMES[column], &values, BLUE)?;
    }
    root.present()?;
    Ok(())
}

fn plot_targets(dataset: &FraudDataset, path: &Path) -> Result<()> {
    let root = BitMapBackend::new(path, (1200, 500)).into_drawing_area();
    root.fill(&WHITE)?;
    let areas = root.split_evenly((1, 2));
    draw_histogram(
        &areas[0],
        "BigModel fraud probability",
        &dataset.teacher_targets,
        ORANGE,
    )?;
    let frauds = dataset.fraud_labels.iter().filter(|&&label| label).count() as i32;
    let legitimate = dataset.len() as i32 - frauds;
    let mut chart = ChartBuilder::on(&areas[1])
        .caption("Ground-truth class balance", ("sans-serif", 22))
        .margin(15)
        .x_label_area_size(35)
        .y_label_area_size(50)
        .build_cartesian_2d(0..2, 0..legitimate.max(frauds) + 300)?;
    chart
        .configure_mesh()
        .disable_mesh()
        .x_labels(2)
        .x_label_formatter(&|value| match *value {
            0 => "legitimate".into(),
            1 => "fraud".into(),
            _ => String::new(),
        })
        .draw()?;
    chart.draw_series([
        Rectangle::new([(0, 0), (1, legitimate)], BLUE.filled()),
        Rectangle::new([(1, 0), (2, frauds)], RED.filled()),
    ])?;
    root.present()?;
    Ok(())
}

fn draw_histogram(
    area: &DrawingArea<BitMapBackend<'_>, plotters::coord::Shift>,
    title: &str,
    values: &[f64],
    color: RGBColor,
) -> Result<()> {
    let minimum = values.iter().copied().fold(f64::INFINITY, f64::min);
    let maximum = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let bins = 30usize;
    let width = if maximum > minimum {
        (maximum - minimum) / bins as f64
    } else {
        1.0
    };
    let mut counts = vec![0i32; bins];
    for &value in values {
        let index = (((value - minimum) / width).floor() as usize).min(bins - 1);
        counts[index] += 1;
    }
    let max_count = counts.iter().copied().max().unwrap_or(1);
    let x_end = if maximum > minimum {
        maximum
    } else {
        minimum + 1.0
    };
    let mut chart = ChartBuilder::on(area)
        .caption(title, ("sans-serif", 18))
        .margin(10)
        .x_label_area_size(35)
        .y_label_area_size(45)
        .build_cartesian_2d(minimum..x_end, 0..max_count + max_count / 10 + 1)?;
    chart.configure_mesh().max_light_lines(3).draw()?;
    chart.draw_series(counts.iter().enumerate().map(|(index, &count)| {
        let start = minimum + index as f64 * width;
        Rectangle::new(
            [(start, 0), (start + width, count)],
            color.mix(0.75).filled(),
        )
    }))?;
    Ok(())
}

fn plot_learning_history(runs: &[LearningRun], path: &Path) -> Result<()> {
    let root = BitMapBackend::new(path, (1000, 650)).into_drawing_area();
    root.fill(&WHITE)?;
    let max_epoch = runs
        .iter()
        .flat_map(|run| run.report.history.iter().map(|row| row.epoch))
        .max()
        .unwrap_or(1);
    let values = runs
        .iter()
        .flat_map(|run| run.report.history.iter().map(|row| row.train_loss.log10()))
        .collect::<Vec<_>>();
    let (min_y, max_y) = padded_range(&values);
    let mut chart = ChartBuilder::on(&root)
        .caption("Learning comparison (log10 MSE)", ("sans-serif", 28))
        .margin(20)
        .x_label_area_size(45)
        .y_label_area_size(60)
        .build_cartesian_2d(1usize..max_epoch.max(2), min_y..max_y)?;
    chart
        .configure_mesh()
        .x_desc("epoch")
        .y_desc("log10 MSE")
        .draw()?;
    for (index, run) in runs.iter().enumerate() {
        let color = if index == 0 { BLUE } else { ORANGE };
        chart
            .draw_series(LineSeries::new(
                run.report
                    .history
                    .iter()
                    .map(|row| (row.epoch, row.train_loss.log10())),
                color.stroke_width(2),
            ))?
            .label(run.name)
            .legend(move |(x, y)| PathElement::new([(x, y), (x + 20, y)], color));
    }
    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.85))
        .border_style(BLACK)
        .draw()?;
    root.present()?;
    Ok(())
}

fn plot_prediction_comparison(runs: &[LearningRun], targets: &[f64], path: &Path) -> Result<()> {
    let root = BitMapBackend::new(path, (1200, 550)).into_drawing_area();
    root.fill(&WHITE)?;
    for (run, area) in runs.iter().zip(root.split_evenly((1, 2))) {
        let min_prediction = run.predictions.iter().copied().fold(0.0, f64::min);
        let max_prediction = run.predictions.iter().copied().fold(1.0, f64::max);
        let mut chart = ChartBuilder::on(&area)
            .caption(format!("{} predictions", run.name), ("sans-serif", 22))
            .margin(15)
            .x_label_area_size(45)
            .y_label_area_size(50)
            .build_cartesian_2d(0.0..1.0, min_prediction..max_prediction)?;
        chart
            .configure_mesh()
            .x_desc("BigModel probability")
            .y_desc("TinyModel prediction")
            .draw()?;
        chart.draw_series(
            targets
                .iter()
                .zip(&run.predictions)
                .map(|(&target, &prediction)| {
                    Circle::new((target, prediction), 1, BLUE.mix(0.3).filled())
                }),
        )?;
        chart.draw_series(LineSeries::new(vec![(0.0, 0.0), (1.0, 1.0)], RED))?;
    }
    root.present()?;
    Ok(())
}

fn plot_generalization_history(history: &[EpochMetrics], path: &Path) -> Result<()> {
    if history.is_empty() {
        bail!("cannot plot an empty history");
    }
    let root = BitMapBackend::new(path, (1000, 650)).into_drawing_area();
    root.fill(&WHITE)?;
    let values = history
        .iter()
        .flat_map(|row| [row.train_loss.log10(), row.validation_loss.unwrap().log10()])
        .collect::<Vec<_>>();
    let (min_y, max_y) = padded_range(&values);
    let mut chart = ChartBuilder::on(&root)
        .caption(
            "Generalization learning curve (log10 MSE)",
            ("sans-serif", 28),
        )
        .margin(20)
        .x_label_area_size(45)
        .y_label_area_size(60)
        .build_cartesian_2d(1usize..history.len().max(2), min_y..max_y)?;
    chart
        .configure_mesh()
        .x_desc("epoch")
        .y_desc("log10 MSE")
        .draw()?;
    for (label, color, values) in [
        (
            "train",
            BLUE,
            history
                .iter()
                .map(|row| (row.epoch, row.train_loss.log10()))
                .collect::<Vec<_>>(),
        ),
        (
            "validation",
            ORANGE,
            history
                .iter()
                .map(|row| (row.epoch, row.validation_loss.unwrap().log10()))
                .collect::<Vec<_>>(),
        ),
    ] {
        chart
            .draw_series(LineSeries::new(values, color.stroke_width(2)))?
            .label(label)
            .legend(move |(x, y)| PathElement::new([(x, y), (x + 20, y)], color));
    }
    chart.configure_series_labels().border_style(BLACK).draw()?;
    root.present()?;
    Ok(())
}

fn plot_threshold_metrics(sweep: &[ThresholdMetrics], path: &Path) -> Result<()> {
    if sweep.is_empty() {
        bail!("cannot plot an empty threshold sweep");
    }
    let root = BitMapBackend::new(path, (1000, 650)).into_drawing_area();
    root.fill(&WHITE)?;
    let min_x = sweep.iter().map(|row| row.threshold).fold(1.0, f64::min);
    let max_x = sweep.iter().map(|row| row.threshold).fold(0.0, f64::max);
    let mut chart = ChartBuilder::on(&root)
        .caption("Validation threshold trade-off", ("sans-serif", 28))
        .margin(20)
        .x_label_area_size(45)
        .y_label_area_size(55)
        .build_cartesian_2d(min_x..max_x, 0.0..1.02)?;
    chart
        .configure_mesh()
        .x_desc("threshold")
        .y_desc("metric")
        .draw()?;
    for (label, color, values) in [
        (
            "precision",
            BLUE,
            sweep
                .iter()
                .map(|row| (row.threshold, row.precision()))
                .collect::<Vec<_>>(),
        ),
        (
            "recall",
            ORANGE,
            sweep
                .iter()
                .map(|row| (row.threshold, row.recall()))
                .collect::<Vec<_>>(),
        ),
        (
            "accuracy",
            GREEN,
            sweep
                .iter()
                .map(|row| (row.threshold, row.accuracy()))
                .collect::<Vec<_>>(),
        ),
    ] {
        chart
            .draw_series(LineSeries::new(values, color.stroke_width(2)))?
            .label(label)
            .legend(move |(x, y)| PathElement::new([(x, y), (x + 20, y)], color));
    }
    chart.configure_series_labels().border_style(BLACK).draw()?;
    root.present()?;
    Ok(())
}

fn padded_range(values: &[f64]) -> (f64, f64) {
    let minimum = values.iter().copied().fold(f64::INFINITY, f64::min);
    let maximum = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if (maximum - minimum).abs() <= f64::EPSILON {
        (minimum - 1.0, maximum + 1.0)
    } else {
        let padding = (maximum - minimum) * 0.05;
        (minimum - padding, maximum + padding)
    }
}
