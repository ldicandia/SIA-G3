//! Feature construction and ablation study for the Exercise 1 TinyModel.
//!
//! Every variant (baseline, drop-one, add-derived) goes through exactly the
//! same cross-validation protocol as `generalize`: same holdout, same folds,
//! learning-rate search and out-of-fold F1 threshold. Only development rows
//! are used; the test split is never consulted.

use std::{fs, path::Path};

use anyhow::Result;

use crate::{
    config::{AppConfig, FeatureConfig},
    data::{apply_feature_config, DerivedFeature, FraudDataset, FEATURE_NAMES},
    metrics::mean_std,
};

use super::{
    cross_validate,
    plots::{plot_feature_study, FeatureStudyBar},
    report::{pearson, skewness},
};

/// One feature set evaluated by the study.
struct Variant {
    name: String,
    kind: &'static str,
    features: FeatureConfig,
}

/// Cross-validated metrics of one variant.
struct VariantResult {
    name: String,
    kind: &'static str,
    n_features: usize,
    learning_rate: f64,
    threshold: f64,
    r2: (f64, f64),
    precision_mean: f64,
    recall_mean: f64,
    f1: (f64, f64),
    average_precision: (f64, f64),
}

fn with_derived(base: &FeatureConfig, extra: &[DerivedFeature]) -> FeatureConfig {
    let mut features = base.clone();
    for &feature in extra {
        if !features.derived.contains(&feature) {
            features.derived.push(feature);
        }
    }
    features
}

fn evaluate_variant(
    raw: &FraudDataset,
    config: &AppConfig,
    variant: &Variant,
) -> Result<VariantResult> {
    variant.features.validate()?;
    let dataset = apply_feature_config(raw, &variant.features)?;
    let cv = cross_validate(&dataset, config)?;
    let collect = |metric: fn(&super::FoldClassification) -> f64| {
        cv.fold_rows.iter().map(metric).collect::<Vec<_>>()
    };
    let result = VariantResult {
        name: variant.name.clone(),
        kind: variant.kind,
        n_features: dataset.features.cols(),
        learning_rate: cv.chosen.learning_rate,
        threshold: cv.threshold,
        r2: mean_std(&collect(|row| row.r2)),
        precision_mean: mean_std(&collect(|row| row.precision)).0,
        recall_mean: mean_std(&collect(|row| row.recall)).0,
        f1: mean_std(&collect(|row| row.f1)),
        average_precision: mean_std(&collect(|row| row.average_precision)),
    };
    eprintln!(
        "feature-study {:<32} features={:>2} f1={:.4}±{:.4} ap={:.4}±{:.4}",
        result.name,
        result.n_features,
        result.f1.0,
        result.f1.1,
        result.average_precision.0,
        result.average_precision.1
    );
    Ok(result)
}

/// Runs the baseline, one drop-one ablation per kept raw column, one
/// addition per derived feature, all derived features together and (when
/// it differs) the subset whose individual addition improved both CV F1 and
/// CV AP. Writes `output/feature_study`.
pub fn run_feature_study(raw: &FraudDataset, config: &AppConfig, output: &Path) -> Result<()> {
    let output = output.join("feature_study");
    fs::create_dir_all(&output)?;
    let base = &config.features;

    let mut variants = vec![Variant {
        name: "baseline".into(),
        kind: "baseline",
        features: base.clone(),
    }];
    for column in FEATURE_NAMES
        .iter()
        .filter(|column| !base.drop.iter().any(|dropped| dropped == *column))
    {
        let mut features = base.clone();
        features.drop.push((*column).to_owned());
        features.log1p.retain(|name| name != column);
        variants.push(Variant {
            name: format!("drop:{column}"),
            kind: "drop",
            features,
        });
    }
    let candidates = DerivedFeature::ALL
        .into_iter()
        .filter(|feature| !base.derived.contains(feature))
        .collect::<Vec<_>>();
    for &feature in &candidates {
        variants.push(Variant {
            name: format!("add:{}", feature.name()),
            kind: "add",
            features: with_derived(base, &[feature]),
        });
    }
    variants.push(Variant {
        name: "add:all_derived".into(),
        kind: "add",
        features: with_derived(base, &candidates),
    });

    let mut results = variants
        .iter()
        .map(|variant| evaluate_variant(raw, config, variant))
        .collect::<Result<Vec<_>>>()?;
    let baseline_f1 = results[0].f1.0;
    let baseline_ap = results[0].average_precision.0;

    let selected = candidates
        .iter()
        .copied()
        .filter(|feature| {
            let name = format!("add:{}", feature.name());
            results.iter().any(|result| {
                result.name == name
                    && result.f1.0 - baseline_f1 > 0.0
                    && result.average_precision.0 - baseline_ap > 0.0
            })
        })
        .collect::<Vec<_>>();
    if !selected.is_empty() && selected.len() != candidates.len() {
        let names = selected
            .iter()
            .map(|feature| feature.name())
            .collect::<Vec<_>>();
        eprintln!("feature-study selected = [{}]", names.join(", "));
        results.push(evaluate_variant(
            raw,
            config,
            &Variant {
                name: "add:selected".into(),
                kind: "add",
                features: with_derived(base, &selected),
            },
        )?);
    } else {
        eprintln!(
            "feature-study selected set is {} (no add:selected variant)",
            if selected.is_empty() {
                "empty"
            } else {
                "identical to all_derived"
            }
        );
    }

    let mut writer = csv::Writer::from_path(output.join("feature_study.csv"))?;
    writer.write_record([
        "variant",
        "kind",
        "n_features",
        "learning_rate",
        "threshold",
        "r2_mean",
        "r2_std",
        "precision_mean",
        "recall_mean",
        "f1_mean",
        "f1_std",
        "ap_mean",
        "ap_std",
        "delta_f1",
        "delta_ap",
    ])?;
    for result in &results {
        writer.serialize((
            &result.name,
            result.kind,
            result.n_features,
            result.learning_rate,
            result.threshold,
            result.r2.0,
            result.r2.1,
            result.precision_mean,
            result.recall_mean,
            result.f1.0,
            result.f1.1,
            result.average_precision.0,
            result.average_precision.1,
            result.f1.0 - baseline_f1,
            result.average_precision.0 - baseline_ap,
        ))?;
    }
    writer.flush()?;

    write_derived_profile(raw, &output.join("derived_feature_profile.csv"))?;

    let bars = results
        .iter()
        .skip(1)
        .map(|result| FeatureStudyBar {
            label: result.name.clone(),
            kind: result.kind,
            delta_f1_pp: (result.f1.0 - baseline_f1) * 100.0,
            delta_ap_pp: (result.average_precision.0 - baseline_ap) * 100.0,
        })
        .collect::<Vec<_>>();
    plot_feature_study(&bars, &output.join("feature_study.png"))?;
    Ok(())
}

/// Exploratory profile of every derived column on all rows (like
/// `inspect`): skewness after and, for log-compressed ratios, before `log1p`,
/// and Pearson correlation with the BigModel score and with `flagged_fraud`.
fn write_derived_profile(raw: &FraudDataset, path: &Path) -> Result<()> {
    let features = FeatureConfig {
        drop: FEATURE_NAMES
            .iter()
            .map(|name| (*name).to_owned())
            .collect(),
        log1p: Vec::new(),
        derived: DerivedFeature::ALL.to_vec(),
    };
    let derived = apply_feature_config(raw, &features)?;
    let fraud = raw
        .fraud_labels
        .iter()
        .map(|&label| f64::from(u8::from(label)))
        .collect::<Vec<_>>();
    let mut writer = csv::Writer::from_path(path)?;
    writer.write_record([
        "column",
        "skewness",
        "skewness_before_log1p",
        "pearson_teacher",
        "pearson_flagged_fraud",
    ])?;
    for (column, name) in derived.feature_names.iter().enumerate() {
        let values = (0..derived.len())
            .map(|row| derived.features.row_unchecked(row)[column])
            .collect::<Vec<_>>();
        // The interaction is a product of logs, not a log-compressed ratio.
        let before_log = (name.starts_with("log1p_") && !name.contains("_x_")).then(|| {
            let unlogged = values
                .iter()
                .map(|value| value.exp_m1())
                .collect::<Vec<_>>();
            skewness(&unlogged)
        });
        writer.serialize((
            name,
            skewness(&values),
            before_log,
            pearson(&values, &raw.teacher_targets),
            pearson(&values, &fraud),
        ))?;
    }
    writer.flush()?;
    Ok(())
}
