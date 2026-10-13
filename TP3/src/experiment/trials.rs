//! Single training trials for the fraud-distillation experiments.

use anyhow::Result;

use crate::{
    config::AppConfig,
    loss::MeanSquaredError,
    matrix::DenseMatrix,
    metrics::{mean_std, regression_metrics, RegressionMetrics},
    model::{Activation, SingleLayerPerceptron},
    split::Fold,
    training::{
        predict_single_indices, train_single_layer, EarlyStopping, TrainingConfig, TrainingReport,
    },
};

pub(super) struct LearningRun {
    pub(super) name: &'static str,
    pub(super) learning_rate: f64,
    pub(super) seed: u64,
    pub(super) report: TrainingReport,
    pub(super) predictions: Vec<f64>,
    pub(super) metrics: RegressionMetrics,
}

/// Every repeat of one (activation, learning rate) pair. The spread across
/// repeats is the noise from the initial weights and the shuffle order.
pub(super) struct LearningRateResult {
    pub(super) learning_rate: f64,
    pub(super) mse: (f64, f64),
    pub(super) r2: (f64, f64),
    pub(super) median_best_epoch: usize,
    pub(super) runs: Vec<LearningRun>,
}

impl LearningRateResult {
    fn new(learning_rate: f64, runs: Vec<LearningRun>) -> Self {
        let mse = runs.iter().map(|run| run.metrics.mse).collect::<Vec<_>>();
        let r2 = runs.iter().map(|run| run.metrics.r2).collect::<Vec<_>>();
        let mut epochs = runs
            .iter()
            .map(|run| run.report.best_epoch)
            .collect::<Vec<_>>();
        epochs.sort_unstable();
        Self {
            learning_rate,
            mse: mean_std(&mse),
            r2: mean_std(&r2),
            median_best_epoch: epochs[epochs.len() / 2],
            runs,
        }
    }

    pub(super) fn name(&self) -> &'static str {
        self.runs[0].name
    }

    /// The repeat with the median MSE: a typical run, not the luckiest seed.
    pub(super) fn representative(&self) -> &LearningRun {
        let mut order = self.runs.iter().collect::<Vec<_>>();
        order.sort_by(|left, right| left.metrics.mse.total_cmp(&right.metrics.mse));
        order[order.len() / 2]
    }
}

/// One (learning rate, fold, seed) cell of the cross-validation grid.
pub(super) struct FoldTrial {
    pub(super) learning_rate: f64,
    pub(super) fold: usize,
    pub(super) seed: u64,
    pub(super) report: TrainingReport,
    pub(super) train_mse: f64,
    pub(super) validation_predictions: Vec<f64>,
    pub(super) validation_metrics: RegressionMetrics,
}

#[allow(clippy::too_many_arguments)]
fn run_learning_trial(
    name: &'static str,
    activation: Activation,
    learning_rate: f64,
    seed: u64,
    features: &DenseMatrix,
    all_indices: &[usize],
    teacher_targets: &[f64],
    config: &AppConfig,
) -> Result<LearningRun> {
    let mut model = SingleLayerPerceptron::new(features.cols(), activation, seed)?;
    let report = train_single_layer(
        &mut model,
        features,
        teacher_targets,
        all_indices,
        None,
        &MeanSquaredError,
        search_training_config(config, learning_rate, seed),
    )?;
    let predictions = predict_single_indices(&model, features, all_indices)?;
    let metrics = regression_metrics(&predictions, teacher_targets)?;
    eprintln!(
        "compare model={name} learning_rate={learning_rate} seed={seed} epochs={} mse={:.6}",
        report.history.len(),
        metrics.mse
    );
    Ok(LearningRun {
        name,
        learning_rate,
        seed,
        report,
        predictions,
        metrics,
    })
}

/// Trains `config.search.repeats` seeds per learning rate, all in parallel,
/// and groups the runs by learning rate in the configured order.
#[allow(clippy::too_many_arguments)]
pub(super) fn run_learning_trials(
    name: &'static str,
    activation: Activation,
    learning_rates: &[f64],
    features: &DenseMatrix,
    all_indices: &[usize],
    teacher_targets: &[f64],
    config: &AppConfig,
) -> Result<Vec<LearningRateResult>> {
    use rayon::prelude::*;

    let seeds = repeat_seeds(config);
    let cells = learning_rates
        .iter()
        .flat_map(|&rate| seeds.iter().map(move |&seed| (rate, seed)))
        .collect::<Vec<_>>();
    let mut runs = cells
        .into_par_iter()
        .map(|(learning_rate, seed)| {
            run_learning_trial(
                name,
                activation,
                learning_rate,
                seed,
                features,
                all_indices,
                teacher_targets,
                config,
            )
        })
        .collect::<Result<Vec<_>>>()?
        .into_iter();
    // `cells` is rate-major, so each learning rate owns the next `repeats` runs.
    Ok(learning_rates
        .iter()
        .map(|&rate| {
            LearningRateResult::new(rate, runs.by_ref().take(config.search.repeats).collect())
        })
        .collect())
}

#[allow(clippy::too_many_arguments)]
fn run_fold_trial(
    learning_rate: f64,
    fold_index: usize,
    seed: u64,
    fold: &Fold,
    features: &DenseMatrix,
    teacher_targets: &[f64],
    config: &AppConfig,
) -> Result<FoldTrial> {
    let mut model = SingleLayerPerceptron::new(features.cols(), Activation::Sigmoid, seed)?;
    let report = train_single_layer(
        &mut model,
        features,
        teacher_targets,
        &fold.train,
        Some(&fold.validation),
        &MeanSquaredError,
        search_training_config(config, learning_rate, seed),
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
        "generalize learning_rate={learning_rate} fold={fold_index} seed={seed} best_epoch={} validation_mse={:.6}",
        report.best_epoch, validation_metrics.mse
    );
    Ok(FoldTrial {
        learning_rate,
        fold: fold_index,
        seed,
        report,
        train_mse,
        validation_predictions,
        validation_metrics,
    })
}

/// Trains every (learning rate, fold, seed) cell in parallel, with the same
/// `config.search.repeats` seeds as the activation comparison. `fold_features[k]`
/// is the full matrix scaled with a scaler fitted only on fold `k`'s train rows.
pub(super) fn run_cv_trials(
    learning_rates: &[f64],
    folds: &[Fold],
    fold_features: &[DenseMatrix],
    teacher_targets: &[f64],
    config: &AppConfig,
) -> Result<Vec<FoldTrial>> {
    use rayon::prelude::*;

    let seeds = repeat_seeds(config);
    let cells = learning_rates
        .iter()
        .flat_map(|&rate| {
            let seeds = &seeds;
            (0..folds.len()).flat_map(move |fold| seeds.iter().map(move |&seed| (rate, fold, seed)))
        })
        .collect::<Vec<_>>();
    cells
        .into_par_iter()
        .map(|(learning_rate, fold, seed)| {
            run_fold_trial(
                learning_rate,
                fold,
                seed,
                &folds[fold],
                &fold_features[fold],
                teacher_targets,
                config,
            )
        })
        .collect()
}

/// Repeat `r` seeds both the initial weights and the shuffle order with
/// `initialization_seed + r`.
pub(super) fn repeat_seeds(config: &AppConfig) -> Vec<u64> {
    (0..config.search.repeats)
        .map(|repeat| {
            config
                .search
                .initialization_seed
                .wrapping_add(repeat as u64)
        })
        .collect()
}

fn search_training_config(config: &AppConfig, learning_rate: f64, seed: u64) -> TrainingConfig {
    TrainingConfig {
        learning_rate,
        max_epochs: config.search.max_epochs,
        shuffle: true,
        seed,
        early_stopping: Some(EarlyStopping {
            patience: config.search.patience,
            min_delta: config.search.min_delta,
        }),
    }
}

pub(super) fn select_values<T: Copy>(values: &[T], indices: &[usize]) -> Vec<T> {
    indices.iter().map(|&index| values[index]).collect()
}
