//! Robustness of saved digit models to pixel noise.
//!
//! Noise is generated with a per-row ChaCha8 stream seeded by `(seed, row)`,
//! so for a given seed every model sees exactly the same corrupted images,
//! and the same random draws are reused at every noise level (common random
//! numbers): curves are smooth and differences between models are not
//! sampling noise.

use std::{f64::consts::TAU, path::Path};

use anyhow::{bail, Result};
use plotters::prelude::*;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

use crate::matrix::DenseMatrix;

use super::{
    artifact::DigitModelArtifact,
    data::{DigitDataset, DIGIT_CLASSES, IMAGE_PIXELS},
    image_grid::{draw_map_grid, MapCell, MapStyle},
    metrics::classification_metrics,
    training::evaluate_digit_model,
};

const GAUSSIAN_SALT: u64 = 0x6A09_E667_F3BC_C908;
const SALT_PEPPER_SALT: u64 = 0xBB67_AE85_84CA_A73B;
/// Rows of each kind shown in `noisy_examples.png`.
const GAUSSIAN_EXAMPLE_LEVELS: usize = 6;
const SALT_PEPPER_EXAMPLE_LEVELS: usize = 3;

fn row_rng(seed: u64, row: usize, salt: u64) -> ChaCha8Rng {
    ChaCha8Rng::seed_from_u64(seed ^ (row as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ salt)
}

/// `x' = clamp(x + sigma * z, 0, 1)` with `z ~ N(0, 1)` drawn by Box–Muller
/// from the row's stream. For a given `(seed, row)` the draws do not depend
/// on `sigma`.
pub fn add_gaussian_noise(features: &DenseMatrix, sigma: f64, seed: u64) -> DenseMatrix {
    let cols = features.cols();
    let mut values = Vec::with_capacity(features.rows() * cols);
    for row in 0..features.rows() {
        let source = features.row_unchecked(row);
        let mut rng = row_rng(seed, row, GAUSSIAN_SALT);
        for pair in source.chunks(2) {
            // u1 in (0, 1] keeps ln(u1) finite.
            let u1 = 1.0 - rng.gen::<f64>();
            let u2 = rng.gen::<f64>();
            let radius = (-2.0 * u1.ln()).sqrt();
            let normals = [radius * (TAU * u2).cos(), radius * (TAU * u2).sin()];
            for (&value, normal) in pair.iter().zip(normals) {
                values.push((value + sigma * normal).clamp(0.0, 1.0));
            }
        }
    }
    DenseMatrix::new(features.rows(), cols, values).expect("noise keeps the input shape")
}

/// Salt-and-pepper noise: each pixel draws `u1, u2` from the row's stream;
/// if `u1 < fraction` it becomes 1 (ink) when `u2 < 0.5` and 0 otherwise.
/// Because the draws do not depend on `fraction`, the pixels corrupted at a
/// smaller fraction are a subset of those corrupted at a larger one.
pub fn add_salt_pepper_noise(features: &DenseMatrix, fraction: f64, seed: u64) -> DenseMatrix {
    let cols = features.cols();
    let mut values = Vec::with_capacity(features.rows() * cols);
    for row in 0..features.rows() {
        let mut rng = row_rng(seed, row, SALT_PEPPER_SALT);
        for &value in features.row_unchecked(row) {
            let (u1, u2) = (rng.gen::<f64>(), rng.gen::<f64>());
            values.push(if u1 < fraction {
                if u2 < 0.5 {
                    1.0
                } else {
                    0.0
                }
            } else {
                value
            });
        }
    }
    DenseMatrix::new(features.rows(), cols, values).expect("noise keeps the input shape")
}

/// Label of a saved model in the study outputs: the name of its directory
/// (for example `exercise3` for `output/exercise3/selected_model.toml`).
pub fn model_label(path: &Path) -> String {
    path.parent()
        .and_then(Path::file_name)
        .or_else(|| path.file_stem())
        .map_or_else(
            || path.display().to_string(),
            |name| name.to_string_lossy().into_owned(),
        )
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum NoiseKind {
    Gaussian,
    SaltPepper,
}

impl NoiseKind {
    fn name(self) -> &'static str {
        match self {
            Self::Gaussian => "gaussian",
            Self::SaltPepper => "salt_pepper",
        }
    }

    fn apply(self, features: &DenseMatrix, level: f64, seed: u64) -> DenseMatrix {
        match self {
            Self::Gaussian => add_gaussian_noise(features, level, seed),
            Self::SaltPepper => add_salt_pepper_noise(features, level, seed),
        }
    }
}

/// Accuracy of one model at one noise level.
struct NoisePoint {
    model: usize,
    kind: NoiseKind,
    level: f64,
    accuracy: f64,
}

/// Evenly picks up to `count` levels from the sorted list, always keeping
/// the smallest and the largest.
fn pick_levels(levels: &[f64], count: usize) -> Vec<f64> {
    let mut sorted = levels.to_vec();
    sorted.sort_by(f64::total_cmp);
    sorted.dedup();
    if sorted.len() <= count {
        return sorted;
    }
    let mut picked = (0..count)
        .map(|index| sorted[(index * (sorted.len() - 1) + (count - 1) / 2) / (count - 1)])
        .collect::<Vec<_>>();
    picked.dedup();
    picked
}

/// Evaluates every model on `dataset` under each Gaussian sigma and each
/// salt-and-pepper fraction. Each noisy matrix is built once and shared by
/// all models. The first model is the primary one shown in the examples.
pub fn run_noise_study(
    dataset: &DigitDataset,
    models: &[(String, DigitModelArtifact)],
    gaussian_sigmas: &[f64],
    salt_pepper_fractions: &[f64],
    seed: u64,
    output: &Path,
) -> Result<()> {
    if models.is_empty() {
        bail!("at least one model is required");
    }
    if gaussian_sigmas.is_empty() && salt_pepper_fractions.is_empty() {
        bail!("at least one noise level is required");
    }
    if let Some(sigma) = gaussian_sigmas
        .iter()
        .find(|sigma| !sigma.is_finite() || **sigma < 0.0)
    {
        bail!("gaussian sigma must be finite and >= 0, got {sigma}");
    }
    if let Some(fraction) = salt_pepper_fractions
        .iter()
        .find(|fraction| !(0.0..=1.0).contains(*fraction))
    {
        bail!("salt-and-pepper fraction must be in [0, 1], got {fraction}");
    }
    for (name, artifact) in models {
        let topology = artifact.model.topology();
        if topology.first() != Some(&IMAGE_PIXELS) || topology.last() != Some(&DIGIT_CLASSES) {
            bail!("model '{name}' has topology {topology:?}; expected 784 -> ... -> 10");
        }
    }
    let indices = (0..dataset.labels.len()).collect::<Vec<_>>();
    let example_rows = (0..DIGIT_CLASSES)
        .filter_map(|digit| dataset.labels.iter().position(|&label| label == digit))
        .collect::<Vec<_>>();
    let example_levels = [
        (
            NoiseKind::Gaussian,
            pick_levels(gaussian_sigmas, GAUSSIAN_EXAMPLE_LEVELS),
        ),
        (
            NoiseKind::SaltPepper,
            pick_levels(salt_pepper_fractions, SALT_PEPPER_EXAMPLE_LEVELS),
        ),
    ];

    let mut writer = csv::Writer::from_path(output.join("noise_robustness.csv"))?;
    let mut header = vec![
        "model".to_owned(),
        "noise".into(),
        "level".into(),
        "accuracy".into(),
        "loss".into(),
    ];
    header.extend((0..DIGIT_CLASSES).map(|digit| format!("recall_{digit}")));
    writer.write_record(&header)?;

    let mut points = Vec::new();
    // (title, pixels, primary prediction is wrong) per example cell.
    let mut example_cells: Vec<(String, Vec<f64>, bool)> = Vec::new();
    for (kind, levels) in [
        (NoiseKind::Gaussian, gaussian_sigmas),
        (NoiseKind::SaltPepper, salt_pepper_fractions),
    ] {
        for &level in levels {
            let noisy = kind.apply(&dataset.features, level, seed);
            for (model_index, (name, artifact)) in models.iter().enumerate() {
                let evaluation =
                    evaluate_digit_model(&artifact.model, &noisy, &dataset.labels, &indices)?;
                let metrics = classification_metrics(&evaluation.predictions, &dataset.labels)?;
                let mut record = vec![
                    name.clone(),
                    kind.name().into(),
                    level.to_string(),
                    evaluation.accuracy.to_string(),
                    evaluation.loss.to_string(),
                ];
                record.extend(
                    metrics
                        .per_class_recall
                        .iter()
                        .map(|recall| recall.map_or_else(String::new, |value| value.to_string())),
                );
                writer.write_record(&record)?;
                points.push(NoisePoint {
                    model: model_index,
                    kind,
                    level,
                    accuracy: evaluation.accuracy,
                });
                let is_example_level = example_levels
                    .iter()
                    .any(|(example_kind, picked)| *example_kind == kind && picked.contains(&level));
                if model_index == 0 && is_example_level {
                    for &row in &example_rows {
                        let predicted = evaluation.predictions[row];
                        let prefix = match kind {
                            NoiseKind::Gaussian => "sigma",
                            NoiseKind::SaltPepper => "s&p",
                        };
                        example_cells.push((
                            format!("{prefix} {level} -> {predicted}"),
                            noisy.row_unchecked(row).to_vec(),
                            predicted != dataset.labels[row],
                        ));
                    }
                }
            }
        }
    }
    writer.flush()?;

    for (model_index, (name, _)) in models.iter().enumerate() {
        for kind in [NoiseKind::Gaussian, NoiseKind::SaltPepper] {
            let curve = points
                .iter()
                .filter(|point| point.model == model_index && point.kind == kind)
                .collect::<Vec<_>>();
            let lowest = curve
                .iter()
                .min_by(|left, right| left.level.total_cmp(&right.level));
            let highest = curve
                .iter()
                .max_by(|left, right| left.level.total_cmp(&right.level));
            if let (Some(lowest), Some(highest)) = (lowest, highest) {
                eprintln!(
                    "noise model={name} kind={} accuracy@{}={:.4} accuracy@{}={:.4}",
                    kind.name(),
                    lowest.level,
                    lowest.accuracy,
                    highest.level,
                    highest.accuracy
                );
            }
        }
    }

    let names = models
        .iter()
        .map(|(name, _)| name.as_str())
        .collect::<Vec<_>>();
    plot_noise_accuracy(&names, &points, &output.join("noise_accuracy.png"))?;
    let cells = example_cells
        .iter()
        .map(|(title, values, wrong)| {
            Some(MapCell {
                title: title.clone(),
                values,
                style: MapStyle::Grayscale,
                highlight: *wrong,
            })
        })
        .collect::<Vec<_>>();
    let rows = cells.len().div_ceil(example_rows.len().max(1));
    draw_map_grid(
        &output.join("noisy_examples.png"),
        &format!(
            "Noisy test digits (title: level -> prediction of {}; red = wrong)",
            names[0]
        ),
        rows.max(1),
        example_rows.len().max(1),
        &cells,
    )?;
    Ok(())
}

const PALETTE: [RGBColor; 5] = [
    RGBColor(41, 98, 255),
    RGBColor(245, 124, 0),
    RGBColor(0, 137, 123),
    RGBColor(123, 31, 162),
    RGBColor(93, 64, 55),
];

fn plot_noise_accuracy(models: &[&str], points: &[NoisePoint], path: &Path) -> Result<()> {
    let root = BitMapBackend::new(path, (1500, 650)).into_drawing_area();
    root.fill(&WHITE)?;
    let panels = root.split_evenly((1, 2));
    for (panel, (kind, title, description)) in panels.iter().zip([
        (
            NoiseKind::Gaussian,
            "Accuracy vs Gaussian noise",
            "sigma (pixels in [0, 1], clamped)",
        ),
        (
            NoiseKind::SaltPepper,
            "Accuracy vs salt-and-pepper noise",
            "fraction of corrupted pixels",
        ),
    ]) {
        let levels = points
            .iter()
            .filter(|point| point.kind == kind)
            .map(|point| point.level)
            .collect::<Vec<_>>();
        if levels.is_empty() {
            continue;
        }
        let max_level = levels.iter().copied().fold(0.0_f64, f64::max).max(1e-3);
        let mut chart = ChartBuilder::on(panel)
            .caption(title, ("sans-serif", 24))
            .margin(20)
            .x_label_area_size(45)
            .y_label_area_size(55)
            .build_cartesian_2d(0.0..max_level, 0.0..1.02)?;
        chart
            .configure_mesh()
            .x_desc(description)
            .y_desc("accuracy on digits_test.csv")
            .draw()?;
        chart
            .draw_series(DashedLineSeries::new(
                vec![(0.0, 0.98), (max_level, 0.98)],
                8,
                6,
                BLACK.stroke_width(1),
            ))?
            .label("98% target")
            .legend(|(x, y)| PathElement::new([(x, y), (x + 20, y)], BLACK));
        for (index, name) in models.iter().enumerate() {
            let color = PALETTE[index % PALETTE.len()];
            let mut curve = points
                .iter()
                .filter(|point| point.kind == kind && point.model == index)
                .map(|point| (point.level, point.accuracy))
                .collect::<Vec<_>>();
            curve.sort_by(|left, right| left.0.total_cmp(&right.0));
            chart
                .draw_series(LineSeries::new(curve.clone(), color.stroke_width(2)))?
                .label(*name)
                .legend(move |(x, y)| {
                    PathElement::new([(x, y), (x + 20, y)], color.stroke_width(3))
                });
            chart.draw_series(
                curve
                    .into_iter()
                    .map(|point| Circle::new(point, 4, color.filled())),
            )?;
        }
        chart
            .configure_series_labels()
            .position(SeriesLabelPosition::LowerLeft)
            .background_style(WHITE.mix(0.85))
            .border_style(BLACK)
            .draw()?;
    }
    root.present()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> DenseMatrix {
        DenseMatrix::from_rows(vec![
            vec![0.0, 0.25, 0.5, 1.0, 0.0],
            vec![1.0, 0.0, 0.75, 0.1, 0.9],
            vec![0.0; 5],
        ])
        .unwrap()
    }

    #[test]
    fn zero_noise_returns_the_input() {
        let features = sample();
        assert_eq!(add_gaussian_noise(&features, 0.0, 7), features);
        assert_eq!(add_salt_pepper_noise(&features, 0.0, 7), features);
    }

    #[test]
    fn noise_stays_in_range_and_is_reproducible() {
        let features = sample();
        for sigma in [0.1, 0.5, 3.0] {
            let noisy = add_gaussian_noise(&features, sigma, 11);
            assert!(noisy
                .as_slice()
                .iter()
                .all(|value| (0.0..=1.0).contains(value)));
            assert_eq!(noisy, add_gaussian_noise(&features, sigma, 11));
            assert_ne!(noisy, add_gaussian_noise(&features, sigma, 12));
        }
        let noisy = add_salt_pepper_noise(&features, 0.4, 11);
        assert_eq!(noisy, add_salt_pepper_noise(&features, 0.4, 11));
        assert!(noisy
            .as_slice()
            .iter()
            .all(|value| (0.0..=1.0).contains(value)));
    }

    #[test]
    fn full_salt_and_pepper_yields_only_binary_pixels() {
        let noisy = add_salt_pepper_noise(&sample(), 1.0, 5);
        assert!(noisy
            .as_slice()
            .iter()
            .all(|&value| value == 0.0 || value == 1.0));
    }

    #[test]
    fn salt_and_pepper_corruption_is_nested_across_fractions() {
        let features =
            DenseMatrix::new(20, 50, (0..1000).map(|i| (i % 7) as f64 / 10.0).collect()).unwrap();
        let low = add_salt_pepper_noise(&features, 0.1, 9);
        let high = add_salt_pepper_noise(&features, 0.2, 9);
        let corrupted = |noisy: &DenseMatrix| {
            features
                .as_slice()
                .iter()
                .zip(noisy.as_slice())
                .map(|(original, value)| original != value)
                .collect::<Vec<_>>()
        };
        let (low_mask, high_mask) = (corrupted(&low), corrupted(&high));
        assert!(low_mask.iter().any(|&changed| changed));
        for (index, (&in_low, &in_high)) in low_mask.iter().zip(&high_mask).enumerate() {
            if in_low {
                assert!(in_high, "pixel {index} corrupted at 0.1 but not at 0.2");
                assert_eq!(low.as_slice()[index], high.as_slice()[index]);
            }
        }
    }

    #[test]
    fn gaussian_draws_are_shared_across_sigmas() {
        let features = DenseMatrix::new(1, 4, vec![0.5; 4]).unwrap();
        let small = add_gaussian_noise(&features, 0.01, 3);
        let large = add_gaussian_noise(&features, 0.02, 3);
        for (a, b) in small.as_slice().iter().zip(large.as_slice()) {
            approx::assert_abs_diff_eq!((b - 0.5), 2.0 * (a - 0.5), epsilon = 1e-12);
        }
    }

    #[test]
    fn picks_evenly_spaced_levels_including_extremes() {
        let levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
        let picked = pick_levels(&levels, 6);
        assert_eq!(picked.len(), 6);
        assert_eq!(picked[0], 0.0);
        assert_eq!(*picked.last().unwrap(), 1.0);
        assert_eq!(pick_levels(&[0.3, 0.0], 6), vec![0.0, 0.3]);
    }
}
