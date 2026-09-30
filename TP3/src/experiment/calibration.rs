//! Calibration study for the sigmoid TinyModel against `flagged_fraud`.
//!
//! Platt scaling and isotonic regression are fitted only on the pooled
//! out-of-fold development scores (the same scores that choose the F1
//! threshold), so the test split stays unseen until the final evaluation.

use std::{fs, path::Path};

use anyhow::{bail, Result};

use crate::{
    calibration::{
        brier_score, expected_calibration_error, reliability_bins, IsotonicRegression,
        PlattScaling, ReliabilityBin,
    },
    config::AppConfig,
    data::{apply_feature_config, FraudDataset},
    metrics::{average_precision, confusion_matrix},
    training::predict_single_indices,
};

use super::{cross_validate, plots::plot_reliability_diagram, refit_tinymodel, select_values};

/// Equal-width bins used for ECE and the reliability diagram.
const CALIBRATION_BINS: usize = 10;
/// BigModel separates both classes perfectly at this score (see README).
const BIGMODEL_THRESHOLD: f64 = 0.85;

/// Probability source evaluated on one split.
struct ScoredSplit<'a> {
    split: &'static str,
    labels: &'a [bool],
    models: Vec<(&'static str, Vec<f64>)>,
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
    let (scaler, model) = refit_tinymodel(&dataset, config, &cv)?;
    let features = scaler.transform(&dataset.features)?;
    let test_rows = &cv.holdout.test;
    let test_scores = predict_single_indices(&model, &features, test_rows)?;
    let test_labels = select_values(&dataset.fraud_labels, test_rows);

    // Calibrators see only out-of-fold development scores and labels.
    let platt = PlattScaling::fit(&cv.oof_scores, &cv.oof_labels)?;
    let isotonic = IsotonicRegression::fit(&cv.oof_scores, &cv.oof_labels)?;

    let sources = |scores: &[f64], rows: &[usize]| {
        vec![
            ("tinymodel_raw", scores.to_vec()),
            (
                "tinymodel_platt",
                scores.iter().map(|&s| platt.apply(s)).collect(),
            ),
            (
                "tinymodel_isotonic",
                scores.iter().map(|&s| isotonic.apply(s)).collect(),
            ),
            ("bigmodel", select_values(&dataset.teacher_targets, rows)),
        ]
    };
    let splits = [
        ScoredSplit {
            split: "development_oof",
            labels: &cv.oof_labels,
            models: sources(&cv.oof_scores, &cv.oof_rows),
        },
        ScoredSplit {
            split: "test",
            labels: &test_labels,
            models: sources(&test_scores, test_rows),
        },
    ];

    let mut metrics = csv::Writer::from_path(output.join("calibration_metrics.csv"))?;
    metrics.write_record([
        "split",
        "model",
        "brier",
        "ece",
        "average_precision",
        "calibrator_in_sample",
    ])?;
    let mut bins_writer = csv::Writer::from_path(output.join("reliability_bins.csv"))?;
    bins_writer.write_record([
        "split",
        "model",
        "bin",
        "low",
        "high",
        "count",
        "mean_predicted",
        "observed_rate",
    ])?;
    let mut test_bins: Vec<(&'static str, Vec<ReliabilityBin>)> = Vec::new();
    let mut test_summary = Vec::new();
    for scored in &splits {
        for (model_name, probabilities) in &scored.models {
            let brier = brier_score(probabilities, scored.labels)?;
            let ece = expected_calibration_error(probabilities, scored.labels, CALIBRATION_BINS)?;
            let in_sample = scored.split == "development_oof"
                && matches!(*model_name, "tinymodel_platt" | "tinymodel_isotonic");
            metrics.serialize((
                scored.split,
                model_name,
                brier,
                ece,
                average_precision(probabilities, scored.labels)?,
                in_sample,
            ))?;
            let bins = reliability_bins(probabilities, scored.labels, CALIBRATION_BINS)?;
            for bin in &bins {
                bins_writer.serialize((
                    scored.split,
                    model_name,
                    bin.index,
                    bin.low,
                    bin.high,
                    bin.count,
                    bin.mean_predicted,
                    bin.observed_rate,
                ))?;
            }
            if scored.split == "test" {
                test_bins.push((model_name, bins));
                test_summary.push((*model_name, brier, ece));
            }
        }
    }
    metrics.flush()?;
    bins_writer.flush()?;

    let mut parameters = csv::Writer::from_path(output.join("platt_parameters.csv"))?;
    parameters.write_record(["a", "b", "isotonic_blocks"])?;
    parameters.serialize((platt.a, platt.b, isotonic.blocks.len()))?;
    parameters.flush()?;

    let test = &splits[1];
    let column = |name: &str| {
        &test
            .models
            .iter()
            .find(|(model_name, _)| *model_name == name)
            .expect("every model is scored on test")
            .1
    };
    let (raw_scores, platt_scores, isotonic_scores, bigmodel_scores) = (
        column("tinymodel_raw"),
        column("tinymodel_platt"),
        column("tinymodel_isotonic"),
        column("bigmodel"),
    );
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
            bigmodel_scores[position],
            raw_scores[position],
            platt_scores[position],
            isotonic_scores[position],
        ))?;
    }
    predictions.flush()?;

    // Expected-cost rule: flag when p * cost_ratio > (1 - p) * 1, i.e.
    // p > 1 / (1 + cost_ratio). Valid only if p is a calibrated probability.
    let cost_threshold = 1.0 / (1.0 + cost_ratio);
    let policies: [(&str, &str, &[f64], f64); 5] = [
        (
            "raw_f1_threshold",
            "tinymodel_raw",
            raw_scores,
            cv.threshold,
        ),
        (
            "raw_cost_threshold",
            "tinymodel_raw",
            raw_scores,
            cost_threshold,
        ),
        (
            "platt_cost_threshold",
            "tinymodel_platt",
            platt_scores,
            cost_threshold,
        ),
        (
            "isotonic_cost_threshold",
            "tinymodel_isotonic",
            isotonic_scores,
            cost_threshold,
        ),
        (
            "bigmodel_threshold",
            "bigmodel",
            bigmodel_scores,
            BIGMODEL_THRESHOLD,
        ),
    ];
    let mut decisions = csv::Writer::from_path(output.join("calibration_decisions.csv"))?;
    decisions.write_record([
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
    ])?;
    for (policy, model_name, scores, threshold) in policies {
        let confusion = confusion_matrix(scores, &test_labels, threshold)?;
        let cost = confusion.false_positives as f64 + confusion.false_negatives as f64 * cost_ratio;
        decisions.serialize((
            policy,
            model_name,
            threshold,
            confusion.true_positives,
            confusion.false_positives,
            confusion.true_negatives,
            confusion.false_negatives,
            confusion.precision(),
            confusion.recall(),
            cost,
        ))?;
    }
    decisions.flush()?;

    let series = test_bins
        .iter()
        .map(|(name, bins)| (*name, bins.as_slice()))
        .collect::<Vec<_>>();
    plot_reliability_diagram(&series, &output.join("reliability_diagram.png"))?;

    let find = |name: &str| {
        test_summary
            .iter()
            .find(|(model_name, _, _)| *model_name == name)
            .copied()
            .expect("summary covers every model")
    };
    let (_, raw_brier, raw_ece) = find("tinymodel_raw");
    let (_, platt_brier, platt_ece) = find("tinymodel_platt");
    eprintln!(
        "calibrate test raw: ece={raw_ece:.4} brier={raw_brier:.4} | platt (a={:.3}, b={:.3}): ece={platt_ece:.4} brier={platt_brier:.4}",
        platt.a, platt.b
    );
    Ok(())
}
