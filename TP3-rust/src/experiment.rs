//! Reproducible experiment orchestration and artifact generation.

mod plots;
mod report;
mod trials;

use std::{fs, path::Path};

use anyhow::{Context, Result};

use crate::{
    config::AppConfig,
    data::{FraudDataset, StandardScaler},
    loss::MeanSquaredError,
    metrics::{best_accuracy_threshold, confusion_matrix, regression_metrics, threshold_sweep},
    model::{Activation, SingleLayerPerceptron},
    split::stratified_split,
    training::{predict_single_indices, train_single_layer, TrainingConfig},
};

use plots::*;
use report::*;
use trials::*;

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
