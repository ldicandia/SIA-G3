//! Single training trials for the fraud-distillation experiments.

use anyhow::Result;

use crate::{
    config::AppConfig,
    loss::MeanSquaredError,
    matrix::DenseMatrix,
    metrics::{regression_metrics, RegressionMetrics},
    model::{Activation, SingleLayerPerceptron},
    split::Fold,
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

/// One (learning rate, fold) cell of the cross-validation grid.
pub(super) struct FoldTrial {
    pub(super) learning_rate: f64,
    pub(super) fold: usize,
    pub(super) report: TrainingReport,
    pub(super) train_mse: f64,
    pub(super) validation_predictions: Vec<f64>,
    pub(super) validation_metrics: RegressionMetrics,
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
    eprintln!(
        "compare model={name} learning_rate={learning_rate} epochs={} mse={:.6}",
        report.history.len(),
        metrics.mse
    );
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

fn run_fold_trial(
    learning_rate: f64,
    fold_index: usize,
    fold: &Fold,
    features: &DenseMatrix,
    teacher_targets: &[f64],
    config: &AppConfig,
) -> Result<FoldTrial> {
    let mut model = SingleLayerPerceptron::new(
        features.cols(),
        Activation::Sigmoid,
        config.search.initialization_seed,
    )?;
    let report = train_single_layer(
        &mut model,
        features,
        teacher_targets,
        &fold.train,
        Some(&fold.validation),
        &MeanSquaredError,
        search_training_config(config, learning_rate),
    )?;
    let train_predictions = predict_single_indices(&model, features, &fold.train)?;
    let validation_predictions = predict_single_indices(&model, features, &fold.validation)?;
    let train_mse = regression_metrics(
        &train_predictions,
        &select_values(teacher_targets, &fold.train),
    )?
    .mse;
    let validation_metrics = regression_metrics(
        &validation_predictions,
        &select_values(teacher_targets, &fold.validation),
    )?;
    eprintln!(
        "generalize learning_rate={learning_rate} fold={fold_index} best_epoch={} validation_mse={:.6}",
        report.best_epoch, validation_metrics.mse
    );
    Ok(FoldTrial {
        learning_rate,
        fold: fold_index,
        report,
        train_mse,
        validation_predictions,
        validation_metrics,
    })
}

/// Trains every (learning rate, fold) pair in parallel. `fold_features[k]` is
/// the full matrix scaled with a scaler fitted only on fold `k`'s train rows.
pub(super) fn run_cv_trials(
    learning_rates: &[f64],
    folds: &[Fold],
    fold_features: &[DenseMatrix],
    teacher_targets: &[f64],
    config: &AppConfig,
) -> Result<Vec<FoldTrial>> {
    use rayon::prelude::*;

    let cells = learning_rates
        .iter()
        .flat_map(|&rate| (0..folds.len()).map(move |fold| (rate, fold)))
        .collect::<Vec<_>>();
    cells
        .into_par_iter()
        .map(|(learning_rate, fold)| {
            run_fold_trial(
                learning_rate,
                fold,
                &folds[fold],
                &fold_features[fold],
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
