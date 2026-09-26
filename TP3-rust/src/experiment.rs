//! Reproducible experiment orchestration and artifact generation.

mod artifact;
mod plots;
mod report;
mod trials;

use std::{fs, path::Path};

use anyhow::{Context, Result};

use crate::{
    config::{AppConfig, FeatureConfig},
    data::{apply_feature_config, FraudDataset, StandardScaler},
    loss::MeanSquaredError,
    matrix::DenseMatrix,
    metrics::{
        average_precision, best_f1_threshold, confusion_matrix, mean_std, regression_metrics,
        threshold_sweep,
    },
    model::{Activation, SingleLayerPerceptron},
    split::{stratified_folds, stratified_holdout},
    training::{predict_single_indices, train_single_layer, TrainingConfig},
};

pub use artifact::*;
use plots::*;
use report::*;
use trials::*;

pub fn run_inspection(
    dataset: &FraudDataset,
    features: &FeatureConfig,
    output: &Path,
) -> Result<()> {
    fs::create_dir_all(output)?;
    write_dataset_summary(dataset, &output.join("dataset_summary.csv"))?;
    write_data_profile(dataset, &output.join("data_profile.csv"))?;
    write_feature_analysis(dataset, features, &output.join("feature_analysis.csv"))?;
    write_time_profile(dataset, &output.join("timestamp_profile.csv"))?;
    plot_feature_distributions(dataset, &output.join("feature_distributions.png"))?;
    let transformed = apply_feature_config(dataset, features)?;
    plot_feature_distributions(
        &transformed,
        &output.join("model_feature_distributions.png"),
    )?;
    plot_targets(dataset, &output.join("target_distributions.png"))?;
    Ok(())
}

pub fn run_learning_comparison(
    raw: &FraudDataset,
    config: &AppConfig,
    output: &Path,
) -> Result<()> {
    fs::create_dir_all(output)?;
    let dataset = apply_feature_config(raw, &config.features)?;
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

    // ReLU is the optional activation suggested by the assignment; it is
    // trained exactly like the other two so its conclusions are comparable.
    for (name, activation) in [
        ("linear", Activation::Linear),
        ("sigmoid", Activation::Sigmoid),
        ("relu", Activation::Relu),
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

/// Generalization study for the sigmoid TinyModel:
///
/// 1. A stratified test set is held out and consulted exactly once.
/// 2. Stratified k-fold cross-validation over the remaining rows selects the
///    learning rate (lowest mean validation MSE) and the refit epoch count
///    (median early-stopping epoch). Each fold fits its own scaler on its
///    own train rows.
/// 3. The fraud threshold maximizes F1 on the pooled out-of-fold predictions,
///    so every score used to pick it comes from a model that never saw it.
/// 4. The model is refit on all development rows and evaluated on test.
pub fn run_generalization(raw: &FraudDataset, config: &AppConfig, output: &Path) -> Result<()> {
    fs::create_dir_all(output)?;
    let dataset = apply_feature_config(raw, &config.features)?;
    let split = &config.split;
    let holdout = stratified_holdout(
        &dataset.teacher_targets,
        split.test_ratio,
        split.stratification_bins,
        split.seed,
    )?;
    let folds = stratified_folds(
        &dataset.teacher_targets,
        &holdout.development,
        split.folds,
        split.stratification_bins,
        split.seed.wrapping_add(1),
    )?;
    write_split_summary(
        &dataset,
        &holdout,
        &folds,
        &output.join("split_summary.csv"),
    )?;

    let fold_features = folds
        .iter()
        .map(|fold| {
            let scaler = StandardScaler::fit(&dataset.features, &fold.train)?;
            Ok(scaler.transform(&dataset.features)?)
        })
        .collect::<Result<Vec<DenseMatrix>>>()?;
    let mut trials = run_cv_trials(
        &config.search.learning_rates,
        &folds,
        &fold_features,
        &dataset.teacher_targets,
        config,
    )?;
    trials.sort_by(|left, right| {
        left.learning_rate
            .total_cmp(&right.learning_rate)
            .then(left.fold.cmp(&right.fold))
    });
    write_cv_trials(&trials, &output.join("generalization_trials.csv"))?;

    let summaries = summarize_learning_rates(&config.search.learning_rates, &trials);
    write_cv_summary(&summaries, &output.join("cv_summary.csv"))?;
    let chosen = summaries
        .iter()
        .min_by(|left, right| left.validation_mse.0.total_cmp(&right.validation_mse.0))
        .context("no learning-rate candidates were configured")?
        .clone();
    let chosen_trials = trials
        .iter()
        .filter(|trial| trial.learning_rate == chosen.learning_rate)
        .collect::<Vec<_>>();

    // Out-of-fold scores cover every development row exactly once.
    let mut pooled_scores = Vec::with_capacity(holdout.development.len());
    let mut pooled_labels = Vec::with_capacity(holdout.development.len());
    for trial in &chosen_trials {
        pooled_scores.extend_from_slice(&trial.validation_predictions);
        pooled_labels.extend(select_values(
            &dataset.fraud_labels,
            &folds[trial.fold].validation,
        ));
    }
    let sweep = threshold_sweep(&pooled_scores, &pooled_labels)?;
    let threshold = best_f1_threshold(&sweep)
        .context("threshold sweep was empty")?
        .threshold;
    write_threshold_sweep(&sweep, &output.join("threshold_sweep.csv"))?;

    let mut fold_rows = Vec::new();
    for trial in &chosen_trials {
        let labels = select_values(&dataset.fraud_labels, &folds[trial.fold].validation);
        let confusion = confusion_matrix(&trial.validation_predictions, &labels, threshold)?;
        fold_rows.push(FoldClassification {
            fold: trial.fold,
            mse: trial.validation_metrics.mse,
            r2: trial.validation_metrics.r2,
            precision: confusion.precision(),
            recall: confusion.recall(),
            f1: confusion.f1(),
            accuracy: confusion.accuracy(),
            average_precision: average_precision(&trial.validation_predictions, &labels)?,
        });
    }
    write_fold_metrics(&fold_rows, &output.join("fold_metrics.csv"))?;

    let development = &holdout.development;
    let final_scaler = StandardScaler::fit(&dataset.features, development)?;
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
        development,
        None,
        &MeanSquaredError,
        TrainingConfig::fixed(
            chosen.learning_rate,
            chosen.median_epoch,
            config.search.initialization_seed,
        ),
    )?;

    let test_predictions = predict_single_indices(&final_model, &final_features, &holdout.test)?;
    let test_targets = select_values(&dataset.teacher_targets, &holdout.test);
    let test_labels = select_values(&dataset.fraud_labels, &holdout.test);
    let regression = regression_metrics(&test_predictions, &test_targets)?;
    let confusion = confusion_matrix(&test_predictions, &test_labels, threshold)?;
    let test_average_precision = average_precision(&test_predictions, &test_labels)?;
    let teacher_average_precision = average_precision(&test_targets, &test_labels)?;

    write_selected_model(
        chosen.learning_rate,
        chosen.median_epoch,
        threshold,
        &output.join("selected_model.csv"),
    )?;
    write_test_metrics(
        regression,
        confusion,
        test_average_precision,
        teacher_average_precision,
        &output.join("test_metrics.csv"),
    )?;
    write_test_predictions(
        &holdout.test,
        &test_targets,
        &test_predictions,
        &test_labels,
        threshold,
        &output.join("test_predictions.csv"),
    )?;
    FraudModelArtifact {
        format_version: 1,
        raw_feature_names: raw.feature_names.clone(),
        features: config.features.clone(),
        scaler: final_scaler,
        model: final_model,
        learning_rate: chosen.learning_rate,
        epochs: chosen.median_epoch,
        threshold,
    }
    .save(&output.join("fraud_model.toml"))?;

    let histories = chosen_trials
        .iter()
        .map(|trial| (trial.fold, trial.report.history.as_slice()))
        .collect::<Vec<_>>();
    plot_cv_histories(
        &histories,
        &output.join("generalization_learning_curve.png"),
    )?;
    plot_threshold_metrics(&sweep, threshold, &output.join("threshold_metrics.png"))?;
    eprintln!(
        "generalize selected learning_rate={} epochs={} threshold={threshold:.5} test_f1={:.4}",
        chosen.learning_rate,
        chosen.median_epoch,
        confusion.f1()
    );
    Ok(())
}

/// Scores a fraud CSV with a persisted TinyModel and reports its metrics.
pub fn run_scoring(raw: &FraudDataset, artifact: &FraudModelArtifact, output: &Path) -> Result<()> {
    fs::create_dir_all(output)?;
    let scores = (0..raw.len())
        .map(|row| artifact.score_raw(raw.features.row_unchecked(row)))
        .collect::<Result<Vec<_>, _>>()?;
    let rows = (0..raw.len()).collect::<Vec<_>>();
    let regression = regression_metrics(&scores, &raw.teacher_targets)?;
    let confusion = confusion_matrix(&scores, &raw.fraud_labels, artifact.threshold)?;
    write_test_metrics(
        regression,
        confusion,
        average_precision(&scores, &raw.fraud_labels)?,
        average_precision(&raw.teacher_targets, &raw.fraud_labels)?,
        &output.join("scoring_metrics.csv"),
    )?;
    write_test_predictions(
        &rows,
        &raw.teacher_targets,
        &scores,
        &raw.fraud_labels,
        artifact.threshold,
        &output.join("scoring_predictions.csv"),
    )?;
    eprintln!(
        "scored {} rows: mse={:.6} precision={:.4} recall={:.4} f1={:.4}",
        raw.len(),
        regression.mse,
        confusion.precision(),
        confusion.recall(),
        confusion.f1()
    );
    Ok(())
}

#[derive(Clone, Debug)]
pub(crate) struct LearningRateSummary {
    pub(crate) learning_rate: f64,
    pub(crate) train_mse: (f64, f64),
    pub(crate) validation_mse: (f64, f64),
    pub(crate) validation_r2: (f64, f64),
    pub(crate) median_epoch: usize,
}

pub(crate) struct FoldClassification {
    pub(crate) fold: usize,
    pub(crate) mse: f64,
    pub(crate) r2: f64,
    pub(crate) precision: f64,
    pub(crate) recall: f64,
    pub(crate) f1: f64,
    pub(crate) accuracy: f64,
    pub(crate) average_precision: f64,
}

fn summarize_learning_rates(rates: &[f64], trials: &[FoldTrial]) -> Vec<LearningRateSummary> {
    rates
        .iter()
        .map(|&rate| {
            let cells = trials
                .iter()
                .filter(|trial| trial.learning_rate == rate)
                .collect::<Vec<_>>();
            let collect =
                |f: &dyn Fn(&FoldTrial) -> f64| cells.iter().map(|t| f(t)).collect::<Vec<_>>();
            let mut epochs = cells
                .iter()
                .map(|trial| trial.report.best_epoch.max(1))
                .collect::<Vec<_>>();
            epochs.sort_unstable();
            LearningRateSummary {
                learning_rate: rate,
                train_mse: mean_std(&collect(&|t| t.train_mse)),
                validation_mse: mean_std(&collect(&|t| t.validation_metrics.mse)),
                validation_r2: mean_std(&collect(&|t| t.validation_metrics.r2)),
                median_epoch: epochs[epochs.len() / 2],
            }
        })
        .collect()
}
