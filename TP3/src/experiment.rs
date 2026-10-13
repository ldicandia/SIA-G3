//! Reproducible experiment orchestration and artifact generation.

mod artifact;
mod calibration;
mod feature_study;
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
        threshold_sweep, ConfusionMatrix, RegressionMetrics, ThresholdMetrics,
    },
    model::{Activation, SingleLayerPerceptron},
    split::{stratified_folds, stratified_holdout, Fold, HoldoutSplit},
    training::{predict_single_indices, train_single_layer, TrainingConfig},
};

pub use artifact::*;
pub use calibration::run_calibration;
pub use feature_study::run_feature_study;
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

/// `plot_relu: false` leaves ReLU out of the PNGs only; it is still trained
/// and written to every CSV.
pub fn run_learning_comparison(
    raw: &FraudDataset,
    config: &AppConfig,
    output: &Path,
    plot_relu: bool,
) -> Result<()> {
    fs::create_dir_all(output)?;
    let dataset = apply_feature_config(raw, &config.features)?;
    let all_indices = (0..dataset.len()).collect::<Vec<_>>();
    let scaler = StandardScaler::fit(&dataset.features, &all_indices)?;
    let features = scaler.transform(&dataset.features)?;
    let mut trial_writer = csv::Writer::from_path(output.join("learning_trials.csv"))?;
    trial_writer.write_record([
        "model",
        "learning_rate",
        "seed",
        "best_epoch",
        "mse",
        "rmse",
        "r2",
        "stopped_early",
    ])?;
    let mut results = Vec::new();
    let mut chosen = Vec::new();

    // ReLU is the optional activation suggested by the assignment; it is
    // trained exactly like the other two so its conclusions are comparable.
    for (name, activation) in [
        ("linear", Activation::Linear),
        ("sigmoid", Activation::Sigmoid),
        ("relu", Activation::Relu),
    ] {
        let rates = run_learning_trials(
            name,
            activation,
            &config.search.learning_rates,
            &features,
            &all_indices,
            &dataset.teacher_targets,
            config,
        )?;
        for run in rates.iter().flat_map(|rate| &rate.runs) {
            trial_writer.serialize((
                run.name,
                run.learning_rate,
                run.seed,
                run.report.best_epoch,
                run.metrics.mse,
                run.metrics.rmse,
                run.metrics.r2,
                run.report.stopped_early,
            ))?;
        }
        // Each activation keeps the learning rate with the lowest mean MSE
        // over the repeats, so one lucky seed cannot pick the rate.
        let best = rates
            .iter()
            .enumerate()
            .min_by(|(_, left), (_, right)| left.mse.0.total_cmp(&right.mse.0))
            .context("no learning-rate candidates were configured")?
            .0;
        chosen.push(results.len() + best);
        results.extend(rates);
    }
    trial_writer.flush()?;

    let chosen = chosen
        .iter()
        .map(|&index| &results[index])
        .collect::<Vec<_>>();
    let runs = chosen
        .iter()
        .map(|result| result.representative())
        .collect::<Vec<_>>();
    write_learning_rates(&results, &output.join("learning_rates.csv"))?;
    write_learning_summary(&chosen, &output.join("learning_summary.csv"))?;
    write_learning_history(&runs, &output.join("learning_history.csv"))?;
    let plotted = runs
        .iter()
        .copied()
        .filter(|run| plot_relu || run.name != "relu")
        .collect::<Vec<_>>();
    plot_learning_history(&plotted, &output.join("learning_curves.png"))?;
    plot_prediction_comparison(
        &plotted,
        &dataset.teacher_targets,
        &output.join("prediction_comparison.png"),
    )?;
    Ok(())
}

/// Generalization study for the sigmoid TinyModel:
///
/// 1. A stratified test set is held out and consulted exactly once.
/// 2. Stratified k-fold cross-validation over the remaining rows, with
///    `repeats` seeds per (learning rate, fold), selects the learning rate
///    (lowest mean validation MSE) and the refit epoch count (median
///    early-stopping epoch). Each fold fits its own scaler on its own train rows.
/// 3. Each seed's threshold maximizes F1 on its pooled out-of-fold predictions,
///    so every score used to pick it comes from a model that never saw it; the
///    chosen threshold is their mean.
/// 4. The model is refit on all development rows with every seed and evaluated
///    on test; the saved model is the representative seed's.
pub fn run_generalization(raw: &FraudDataset, config: &AppConfig, output: &Path) -> Result<()> {
    fs::create_dir_all(output)?;
    let dataset = apply_feature_config(raw, &config.features)?;
    let cv = cross_validate(&dataset, config)?;
    write_split_summary(
        &dataset,
        &cv.holdout,
        &cv.folds,
        &output.join("split_summary.csv"),
    )?;
    write_cv_trials(&cv.trials, &output.join("generalization_trials.csv"))?;
    write_cv_summary(&cv.summaries, &output.join("cv_summary.csv"))?;
    write_threshold_sweep(&cv.sweep, &output.join("threshold_sweep.csv"))?;
    write_fold_metrics(&cv.fold_rows, &output.join("fold_metrics.csv"))?;
    write_seed_thresholds(
        &cv.seed_rows,
        cv.representative_seed,
        &output.join("cv_seeds.csv"),
    )?;

    let (final_scaler, final_model) = refit_tinymodel(&dataset, &cv)?;
    let final_features = final_scaler.transform(&dataset.features)?;
    let holdout = &cv.holdout;
    let chosen = &cv.chosen;
    let threshold = cv.threshold;

    let test_predictions = predict_single_indices(&final_model, &final_features, &holdout.test)?;
    let test_targets = select_values(&dataset.teacher_targets, &holdout.test);
    let test_labels = select_values(&dataset.fraud_labels, &holdout.test);
    let regression = regression_metrics(&test_predictions, &test_targets)?;
    let confusion = confusion_matrix(&test_predictions, &test_labels, threshold)?;
    let test_average_precision = average_precision(&test_predictions, &test_labels)?;
    let teacher_average_precision = average_precision(&test_targets, &test_labels)?;

    // Refit once per seed to show how much the test result depends on it.
    let seed_tests = {
        use rayon::prelude::*;
        repeat_seeds(config)
            .into_par_iter()
            .map(|seed| {
                let (scaler, model) = refit_with_seed(&dataset, &cv, seed)?;
                let features = scaler.transform(&dataset.features)?;
                let predictions = predict_single_indices(&model, &features, &holdout.test)?;
                Ok(SeedTest {
                    seed,
                    regression: regression_metrics(&predictions, &test_targets)?,
                    confusion: confusion_matrix(&predictions, &test_labels, threshold)?,
                    average_precision: average_precision(&predictions, &test_labels)?,
                })
            })
            .collect::<Result<Vec<_>>>()?
    };
    write_seed_tests(
        &seed_tests,
        cv.representative_seed,
        &output.join("test_seeds.csv"),
    )?;

    write_selected_model(
        chosen.learning_rate,
        chosen.median_epoch,
        threshold,
        cv.representative_seed,
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

    let histories = cv
        .chosen_trials()
        .map(|trial| (trial.fold, trial.report.history.as_slice()))
        .collect::<Vec<_>>();
    plot_cv_histories(
        &histories,
        &output.join("generalization_learning_curve.png"),
    )?;
    plot_threshold_metrics(&cv.sweep, threshold, &output.join("threshold_metrics.png"))?;
    let (f1_mean, f1_std) = mean_std(
        &seed_tests
            .iter()
            .map(|test| test.confusion.f1())
            .collect::<Vec<_>>(),
    );
    eprintln!(
        "generalize selected learning_rate={} epochs={} threshold={threshold:.5} seed={} test_f1={:.4} (seeds {f1_mean:.4}±{f1_std:.4})",
        chosen.learning_rate,
        chosen.median_epoch,
        cv.representative_seed,
        confusion.f1()
    );
    Ok(())
}

/// Result of the Exercise 1 cross-validation protocol on an already
/// feature-transformed dataset. Shared by `generalize`, `calibrate` and
/// `feature-study` so every study uses exactly the same splits, search and
/// threshold rule.
struct CrossValidation {
    holdout: HoldoutSplit,
    folds: Vec<Fold>,
    /// Every (learning rate, fold, seed) trial, sorted in that order.
    trials: Vec<FoldTrial>,
    summaries: Vec<LearningRateSummary>,
    chosen: LearningRateSummary,
    /// One row per seed at the chosen learning rate.
    seed_rows: Vec<SeedThreshold>,
    /// Seed with the median out-of-fold MSE at the chosen learning rate: a
    /// typical run, not the luckiest one. It supplies the saved model, the
    /// threshold sweep and the plots.
    representative_seed: u64,
    /// Pooled out-of-fold scores of every seed, aligned with `seed_rows`.
    /// Folds are visited in the same order for every seed, so they all
    /// share `oof_rows` and `oof_labels`.
    seed_oof_scores: Vec<Vec<f64>>,
    /// Development row index of each pooled out-of-fold score.
    oof_rows: Vec<usize>,
    oof_labels: Vec<bool>,
    sweep: Vec<ThresholdMetrics>,
    /// Mean of the per-seed F1-maximizing thresholds.
    threshold: f64,
    /// Per fold, classification metrics averaged over the seeds.
    fold_rows: Vec<FoldClassification>,
}

impl CrossValidation {
    /// Trials of the chosen learning rate and representative seed, in fold order.
    fn chosen_trials(&self) -> impl Iterator<Item = &FoldTrial> {
        self.trials.iter().filter(|trial| {
            trial.learning_rate == self.chosen.learning_rate
                && trial.seed == self.representative_seed
        })
    }
}

/// Out-of-fold result of one seed at the chosen learning rate.
pub(crate) struct SeedThreshold {
    pub(crate) seed: u64,
    /// Mean validation MSE over the folds.
    pub(crate) validation_mse: f64,
    /// Threshold that maximizes this seed's pooled out-of-fold F1.
    pub(crate) threshold: f64,
    pub(crate) f1: f64,
}

/// Test result of the model refit with one seed.
pub(crate) struct SeedTest {
    pub(crate) seed: u64,
    pub(crate) regression: RegressionMetrics,
    pub(crate) confusion: ConfusionMatrix,
    pub(crate) average_precision: f64,
}

/// Holdout, stratified k-fold CV over the development rows (one scaler per
/// fold), learning-rate selection by mean validation MSE, pooled out-of-fold
/// scores and the F1-maximizing threshold. The test rows are never touched.
fn cross_validate(dataset: &FraudDataset, config: &AppConfig) -> Result<CrossValidation> {
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
            .then(left.seed.cmp(&right.seed))
    });

    let summaries = summarize_learning_rates(&config.search.learning_rates, folds.len(), &trials);
    let chosen = summaries
        .iter()
        .min_by(|left, right| left.validation_mse.0.total_cmp(&right.validation_mse.0))
        .context("no learning-rate candidates were configured")?
        .clone();
    let chosen_trials = trials
        .iter()
        .filter(|trial| trial.learning_rate == chosen.learning_rate)
        .collect::<Vec<_>>();

    // Each seed's out-of-fold scores cover every development row exactly once,
    // and each seed gets its own F1-maximizing threshold.
    let mut seed_rows = Vec::new();
    let mut pooled = Vec::new();
    for seed in repeat_seeds(config) {
        let mut rows = Vec::with_capacity(holdout.development.len());
        let mut scores = Vec::with_capacity(holdout.development.len());
        let mut labels = Vec::with_capacity(holdout.development.len());
        let mut mse = Vec::new();
        for trial in chosen_trials.iter().filter(|trial| trial.seed == seed) {
            let validation = &folds[trial.fold].validation;
            rows.extend_from_slice(validation);
            scores.extend_from_slice(&trial.validation_predictions);
            labels.extend(select_values(&dataset.fraud_labels, validation));
            mse.push(trial.validation_metrics.mse);
        }
        let sweep = threshold_sweep(&scores, &labels)?;
        let best = best_f1_threshold(&sweep).context("threshold sweep was empty")?;
        seed_rows.push(SeedThreshold {
            seed,
            validation_mse: mean_std(&mse).0,
            threshold: best.threshold,
            f1: best.f1(),
        });
        pooled.push((rows, scores, labels, sweep));
    }
    // The threshold is a decision, so like the learning rate it comes from
    // every seed: the mean of the per-seed thresholds.
    let threshold = mean_std(
        &seed_rows
            .iter()
            .map(|row| row.threshold)
            .collect::<Vec<_>>(),
    )
    .0;
    let mut order = (0..seed_rows.len()).collect::<Vec<_>>();
    order.sort_by(|&left, &right| {
        seed_rows[left]
            .validation_mse
            .total_cmp(&seed_rows[right].validation_mse)
    });
    let representative = order[order.len() / 2];
    let representative_seed = seed_rows[representative].seed;
    let seed_oof_scores = pooled
        .iter()
        .map(|(_, scores, _, _)| scores.clone())
        .collect();
    let (oof_rows, _, oof_labels, sweep) = pooled.swap_remove(representative);

    let mut fold_rows = Vec::with_capacity(folds.len());
    for (fold, split) in folds.iter().enumerate() {
        let labels = select_values(&dataset.fraud_labels, &split.validation);
        let mut per_seed = Vec::new();
        for trial in chosen_trials.iter().filter(|trial| trial.fold == fold) {
            let confusion = confusion_matrix(&trial.validation_predictions, &labels, threshold)?;
            per_seed.push(FoldClassification {
                fold,
                mse: trial.validation_metrics.mse,
                r2: trial.validation_metrics.r2,
                precision: confusion.precision(),
                recall: confusion.recall(),
                f1: confusion.f1(),
                accuracy: confusion.accuracy(),
                average_precision: average_precision(&trial.validation_predictions, &labels)?,
            });
        }
        fold_rows.push(FoldClassification::mean(fold, &per_seed));
    }

    Ok(CrossValidation {
        holdout,
        folds,
        trials,
        summaries,
        chosen,
        seed_rows,
        representative_seed,
        seed_oof_scores,
        oof_rows,
        oof_labels,
        sweep,
        threshold,
        fold_rows,
    })
}

/// Final TinyModel: the representative seed's refit.
fn refit_tinymodel(
    dataset: &FraudDataset,
    cv: &CrossValidation,
) -> Result<(StandardScaler, SingleLayerPerceptron)> {
    refit_with_seed(dataset, cv, cv.representative_seed)
}

/// A scaler fitted on all development rows and a sigmoid neuron trained with
/// `seed` for the chosen learning rate and median epoch count.
fn refit_with_seed(
    dataset: &FraudDataset,
    cv: &CrossValidation,
    seed: u64,
) -> Result<(StandardScaler, SingleLayerPerceptron)> {
    let development = &cv.holdout.development;
    let final_scaler = StandardScaler::fit(&dataset.features, development)?;
    let final_features = final_scaler.transform(&dataset.features)?;
    let mut final_model =
        SingleLayerPerceptron::new(final_features.cols(), Activation::Sigmoid, seed)?;
    train_single_layer(
        &mut final_model,
        &final_features,
        &dataset.teacher_targets,
        development,
        None,
        &MeanSquaredError,
        TrainingConfig::fixed(cv.chosen.learning_rate, cv.chosen.median_epoch, seed),
    )?;
    Ok((final_scaler, final_model))
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
    /// Mean and std across folds of the per-fold mean over seeds.
    pub(crate) validation_mse: (f64, f64),
    /// Std across seeds inside each fold, averaged over the folds.
    pub(crate) validation_mse_seed_std: f64,
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

impl FoldClassification {
    /// Average over the seeds of one fold.
    fn mean(fold: usize, rows: &[Self]) -> Self {
        let mean =
            |metric: fn(&Self) -> f64| rows.iter().map(metric).sum::<f64>() / rows.len() as f64;
        Self {
            fold,
            mse: mean(|row| row.mse),
            r2: mean(|row| row.r2),
            precision: mean(|row| row.precision),
            recall: mean(|row| row.recall),
            f1: mean(|row| row.f1),
            accuracy: mean(|row| row.accuracy),
            average_precision: mean(|row| row.average_precision),
        }
    }
}

fn summarize_learning_rates(
    rates: &[f64],
    folds: usize,
    trials: &[FoldTrial],
) -> Vec<LearningRateSummary> {
    rates
        .iter()
        .map(|&rate| {
            let cells = trials
                .iter()
                .filter(|trial| trial.learning_rate == rate)
                .collect::<Vec<_>>();
            // Fold spread: std of the per-fold means over seeds. Seed spread:
            // std over seeds inside each fold, averaged over the folds.
            let spread = |metric: fn(&FoldTrial) -> f64| {
                let per_fold = (0..folds)
                    .map(|fold| {
                        let values = cells
                            .iter()
                            .filter(|trial| trial.fold == fold)
                            .map(|trial| metric(trial))
                            .collect::<Vec<_>>();
                        mean_std(&values)
                    })
                    .collect::<Vec<_>>();
                let means = per_fold.iter().map(|(mean, _)| *mean).collect::<Vec<_>>();
                let seed_std = per_fold.iter().map(|(_, std)| std).sum::<f64>() / folds as f64;
                (mean_std(&means), seed_std)
            };
            let (validation_mse, validation_mse_seed_std) =
                spread(|trial| trial.validation_metrics.mse);
            let mut epochs = cells
                .iter()
                .map(|trial| trial.report.best_epoch.max(1))
                .collect::<Vec<_>>();
            epochs.sort_unstable();
            LearningRateSummary {
                learning_rate: rate,
                train_mse: spread(|trial| trial.train_mse).0,
                validation_mse,
                validation_mse_seed_std,
                validation_r2: spread(|trial| trial.validation_metrics.r2).0,
                median_epoch: epochs[epochs.len() / 2],
            }
        })
        .collect()
}
