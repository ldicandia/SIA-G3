//! Single learning-rate trials for the fraud-distillation experiments.

use anyhow::Result;

use crate::{
    config::AppConfig,
    loss::MeanSquaredError,
    matrix::DenseMatrix,
    metrics::{regression_metrics, RegressionMetrics},
    model::{Activation, SingleLayerPerceptron},
    training::{
        predict_single_indices, train_single_layer, EarlyStopping, TrainingConfig, TrainingReport,
    },
};

pub(super) struct LearningRun {
    pub(super) name: &'static str,
    pub(super) learning_rate: f64,
    pub(super) report: TrainingReport,
    pub(super) predictions: Vec<f64>,
    pub(super) metrics: RegressionMetrics,
}

pub(super) struct GeneralizationRun {
    pub(super) learning_rate: f64,
    pub(super) report: TrainingReport,
    pub(super) validation_predictions: Vec<f64>,
    pub(super) validation_metrics: RegressionMetrics,
}

pub(super) struct GeneralizationTrial {
    pub(super) run: GeneralizationRun,
    pub(super) train_mse: f64,
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
pub(super) fn run_learning_trials(
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
pub(super) fn run_generalization_trials(
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

pub(super) fn select_values<T: Copy>(values: &[T], indices: &[usize]) -> Vec<T> {
    indices.iter().map(|&index| values[index]).collect()
}
