//! Calibration study for the sigmoid TinyModel against `flagged_fraud`.
//!
//! Platt scaling and isotonic regression are fitted only on the pooled
//! out-of-fold development scores (the same scores that choose the F1
//! threshold), so the test split stays unseen until the final evaluation.
//!
//! Like `generalize`, the study is repeated for every seed: each seed's
//! calibrators are fitted on its own out-of-fold scores and applied to its
//! own refit model, and the reported metrics are the mean and std over seeds.

use std::{fs, path::Path};

use anyhow::{bail, Result};

use crate::{
    calibration::{
        brier_score, expected_calibration_error, reliability_bins, IsotonicRegression,
        PlattScaling, ReliabilityBin,
    },
    config::AppConfig,
    data::{apply_feature_config, FraudDataset},
    metrics::{average_precision, confusion_matrix, mean_std, ConfusionMatrix},
    training::predict_single_indices,
};

use super::{
    cross_validate, plots::plot_reliability_diagram, refit_with_seed, repeat_seeds,
    select_values,
};

/// Equal-width bins used for ECE and the reliability diagram.
const CALIBRATION_BINS: usize = 10;
/// BigModel separates both classes perfectly at this score (see README).
const BIGMODEL_THRESHOLD: f64 = 0.85;
/// Probability sources, in the order they are written and plotted.
const MODELS: [&str; 4] = [
    "tinymodel_raw",
    "tinymodel_platt",
    "tinymodel_isotonic",
    "bigmodel",
];
const SPLITS: [&str; 2] = ["development_oof", "test"];

/// Probabilities of every model on one split for one seed, in `MODELS` order.
type SplitScores = [Vec<f64>; 4];

/// Calibration study of one seed.
struct SeedCalibration {
    seed: u64,
    platt: PlattScaling,
    isotonic_blocks: usize,
    /// Indexed like `SPLITS`.
    scores: [SplitScores; 2],
}

/// One decision rule applied to the test scores of every seed.
struct Policy {
    name: &'static str,
    model: usize,
    threshold: f64,
}

/// Runs the calibration study and writes its CSVs and reliability diagram
/// to `output/calibration`. `cost_ratio` is the cost of a missed fraud in
/// units of one manual review.
pub fn run_calibration(
    raw: &FraudDataset,
    config: &AppConfig,
    cost_ratio: f64,
    output: &Path,
) -> Result<()> {
    if !cost_ratio.is_finite() || cost_ratio <= 0.0 {
        bail!("cost ratio must be finite and positive, got {cost_ratio}");
    }
    let output = output.join("calibration");
    fs::create_dir_all(&output)?;
    let dataset = apply_feature_config(raw, &config.features)?;
    let cv = cross_validate(&dataset, config)?;
    let test_rows = &cv.holdout.test;
    let test_labels = select_values(&dataset.fraud_labels, test_rows);
    let labels: [&[bool]; 2] = [&cv.oof_labels, &test_labels];

    let seeds = {
        use rayon::prelude::*;
        repeat_seeds(config)
            .into_par_iter()
            .zip(cv.seed_oof_scores.par_iter())
            .map(|(seed, oof_scores)| {
                let (scaler, model) = refit_with_seed(&dataset, &cv, seed)?;
                let features = scaler.transform(&dataset.features)?;
                let test_scores = predict_single_indices(&model, &features, test_rows)?;
                // Calibrators see only this seed's out-of-fold development
                // scores and labels.
                let platt = PlattScaling::fit(oof_scores, &cv.oof_labels)?;
                let isotonic = IsotonicRegression::fit(oof_scores, &cv.oof_labels)?;
                let sources = |scores: &[f64], rows: &[usize]| -> SplitScores {
                    [
                        scores.to_vec(),
                        scores.iter().map(|&s| platt.apply(s)).collect(),
                        scores.iter().map(|&s| isotonic.apply(s)).collect(),
                        select_values(&dataset.teacher_targets, rows),
                    ]
                };
                Ok(SeedCalibration {
                    seed,
                    scores: [
                        sources(oof_scores, &cv.oof_rows),
                        sources(&test_scores, test_rows),
                    ],
                    isotonic_blocks: isotonic.blocks.len(),
                    platt,
                })
            })
            .collect::<Result<Vec<_>>>()?
    };

    // Per seed, split and model: brier, ece and average precision.
    let mut seed_metrics = csv::Writer::from_path(output.join("calibration_seeds.csv"))?;
    seed_metrics.write_record([
        "seed",
        "split",
        "model",
        "brier",
        "ece",
        "average_precision",
        "calibrator_in_sample",
        "representative",
    ])?;
    let mut metrics = csv::Writer::from_path(output.join("calibration_metrics.csv"))?;
    metrics.write_record([
        "split",
        "model",
        "brier_mean",
        "brier_std",
        "ece_mean",
        "ece_std",
        "average_precision_mean",
        "average_precision_std",
        "calibrator_in_sample",
    ])?;
    // Bins of every seed, plus the bins of all seeds pooled (`seed = all`):
    // the pooled bins weigh each seed equally and are the ones plotted.
    let mut bins_writer = csv::Writer::from_path(output.join("reliability_bins.csv"))?;
    bins_writer.write_record([
        "seed",
        "split",
        "model",
        "bin",
        "low",
        "high",
        "count",
        "mean_predicted",
        "observed_rate",
    ])?;
    let mut write_bins = |seed: &str, split: &str, model: &str, bins: &[ReliabilityBin]| {
        bins.iter().try_for_each(|bin| {
            bins_writer.serialize((
                seed,
                split,
                model,
                bin.index,
                bin.low,
                bin.high,
                bin.count,
                bin.mean_predicted,
                bin.observed_rate,
            ))
        })
    };
    let mut test_bins = Vec::new();
    let mut test_summary = Vec::new();
    for (split_index, split) in SPLITS.into_iter().enumerate() {
        let split_labels = labels[split_index];
        for (model_index, model) in MODELS.into_iter().enumerate() {
            let in_sample = split == "development_oof"
                && matches!(model, "tinymodel_platt" | "tinymodel_isotonic");
            let mut rows = Vec::with_capacity(seeds.len());
            for seed in &seeds {
                let probabilities = &seed.scores[split_index][model_index];
                let row = [
                    brier_score(probabilities, split_labels)?,
                    expected_calibration_error(probabilities, split_labels, CALIBRATION_BINS)?,
                    average_precision(probabilities, split_labels)?,
                ];
                seed_metrics.serialize((
                    seed.seed,
                    split,
                    model,
                    row[0],
                    row[1],
                    row[2],
                    in_sample,
                    seed.seed == cv.representative_seed,
                ))?;
                let bins = reliability_bins(probabilities, split_labels, CALIBRATION_BINS)?;
                write_bins(&seed.seed.to_string(), split, model, &bins)?;
                rows.push(row);
            }
            let stats = (0..3)
                .map(|column| mean_std(&rows.iter().map(|row| row[column]).collect::<Vec<_>>()))
                .collect::<Vec<_>>();
            metrics.serialize((
                split,
                model,
                stats[0].0,
                stats[0].1,
                stats[1].0,
                stats[1].1,
                stats[2].0,
                stats[2].1,
                in_sample,
            ))?;

            let pooled_scores = seeds
                .iter()
                .flat_map(|seed| seed.scores[split_index][model_index].iter().copied())
                .collect::<Vec<_>>();
            let pooled_labels = split_labels.repeat(seeds.len());
            let pooled = reliability_bins(&pooled_scores, &pooled_labels, CALIBRATION_BINS)?;
            write_bins("all", split, model, &pooled)?;
            if split == "test" {
                test_bins.push((model, pooled));
                test_summary.push((model, stats[0], stats[1]));
            }
        }
    }
    seed_metrics.flush()?;
    metrics.flush()?;
    drop(write_bins);
    bins_writer.flush()?;

    let mut parameters = csv::Writer::from_path(output.join("platt_parameters.csv"))?;
    parameters.write_record(["seed", "a", "b", "isotonic_blocks", "representative"])?;
    for seed in &seeds {
        parameters.serialize((
            seed.seed,
            seed.platt.a,
            seed.platt.b,
            seed.isotonic_blocks,
            seed.seed == cv.representative_seed,
        ))?;
    }
    parameters.flush()?;

    // Row-level predictions of the representative seed (the saved model).
    let representative = seeds
        .iter()
        .find(|seed| seed.seed == cv.representative_seed)
        .expect("the representative seed is one of the repeats");
    let test = &representative.scores[1];
    let mut predictions = csv::Writer::from_path(output.join("calibrated_test_predictions.csv"))?;
    predictions.write_record([
        "row",
        "flagged_fraud",
        "bigmodel",
        "raw",
        "platt",
        "isotonic",
    ])?;
    for (position, &row) in test_rows.iter().enumerate() {
        predictions.serialize((
            row,
            u8::from(test_labels[position]),
            test[3][position],
            test[0][position],
            test[1][position],
            test[2][position],
        ))?;
    }
    predictions.flush()?;

    // Expected-cost rule: flag when p * cost_ratio > (1 - p) * 1, i.e.
    // p > 1 / (1 + cost_ratio). Valid only if p is a calibrated probability.
    let cost_threshold = 1.0 / (1.0 + cost_ratio);
    let policies = [
        ("raw_f1_threshold", 0, cv.threshold),
        ("raw_cost_threshold", 0, cost_threshold),
        ("platt_cost_threshold", 1, cost_threshold),
        ("isotonic_cost_threshold", 2, cost_threshold),
        ("bigmodel_threshold", 3, BIGMODEL_THRESHOLD),
    ]
    .map(|(name, model, threshold)| Policy {
        name,
        model,
        threshold,
    });
    let mut seed_decisions =
        csv::Writer::from_path(output.join("calibration_decisions_seeds.csv"))?;
    seed_decisions.write_record([
        "seed",
        "policy",
        "model",
        "threshold",
        "tp",
        "fp",
        "tn",
        "fn",
        "precision",
        "recall",
        "cost",
        "representative",
    ])?;
    let mut decisions = csv::Writer::from_path(output.join("calibration_decisions.csv"))?;
    decisions.write_record([
        "policy",
        "model",
        "threshold",
        "tp_mean",
        "fp_mean",
        "tn_mean",
        "fn_mean",
        "precision_mean",
        "precision_std",
        "recall_mean",
        "recall_std",
        "cost_mean",
        "cost_std",
    ])?;
    let cost = |confusion: &ConfusionMatrix| {
        confusion.false_positives as f64 + confusion.false_negatives as f64 * cost_ratio
    };
    for policy in &policies {
        let model = MODELS[policy.model];
        let mut confusions = Vec::with_capacity(seeds.len());
        for seed in &seeds {
            let scores = &seed.scores[1][policy.model];
            let confusion = confusion_matrix(scores, &test_labels, policy.threshold)?;
            seed_decisions.serialize((
                seed.seed,
                policy.name,
                model,
                policy.threshold,
                confusion.true_positives,
                confusion.false_positives,
                confusion.true_negatives,
                confusion.false_negatives,
                confusion.precision(),
                confusion.recall(),
                cost(&confusion),
                seed.seed == cv.representative_seed,
            ))?;
            confusions.push(confusion);
        }
        let stat = |value: fn(&ConfusionMatrix) -> f64| {
            mean_std(&confusions.iter().map(value).collect::<Vec<_>>())
        };
        let precision = stat(|c| c.precision());
        let recall = stat(|c| c.recall());
        let costs = mean_std(&confusions.iter().map(&cost).collect::<Vec<_>>());
        decisions.serialize((
            policy.name,
            model,
            policy.threshold,
            stat(|c| c.true_positives as f64).0,
            stat(|c| c.false_positives as f64).0,
            stat(|c| c.true_negatives as f64).0,
            stat(|c| c.false_negatives as f64).0,
            precision.0,
            precision.1,
            recall.0,
            recall.1,
            costs.0,
            costs.1,
        ))?;
    }
    seed_decisions.flush()?;
    decisions.flush()?;

    let series = test_bins
        .iter()
        .map(|(name, bins)| (*name, bins.as_slice()))
        .collect::<Vec<_>>();
    plot_reliability_diagram(
        &series,
        &format!(
            "Reliability diagram on test, {} seeds pooled ({CALIBRATION_BINS} equal-width bins)",
            seeds.len()
        ),
        &output.join("reliability_diagram.png"),
    )?;

    let find = |name: &str| {
        test_summary
            .iter()
            .find(|(model, _, _)| *model == name)
            .copied()
            .expect("summary covers every model")
    };
    let (_, raw_brier, raw_ece) = find("tinymodel_raw");
    let (_, platt_brier, platt_ece) = find("tinymodel_platt");
    eprintln!(
        "calibrate test over {} seeds raw: ece={:.4}±{:.4} brier={:.4}±{:.4} | platt: ece={:.4}±{:.4} brier={:.4}±{:.4}",
        seeds.len(),
        raw_ece.0,
        raw_ece.1,
        raw_brier.0,
        raw_brier.1,
        platt_ece.0,
        platt_ece.1,
        platt_brier.0,
        platt_brier.1,
    );
    Ok(())
}
