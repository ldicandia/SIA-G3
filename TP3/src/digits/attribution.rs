//! Attribution methods for the digit MLP: which input pixels drive the
//! logit `z_c` of a class `c`.
//!
//! * Saliency: `|∂z_c / ∂x|`.
//! * Gradient × input: `x ⊙ ∂z_c / ∂x`.
//! * Integrated Gradients with a black baseline `x' = 0`:
//!   `IG_i = x_i · (1/m) Σ_k ∂z_c/∂x_i ((k + 0.5)/m · x)` (midpoint rule),
//!   which satisfies completeness: `Σ_i IG_i ≈ z_c(x) − z_c(0)`.
//!
//! The input gradient comes from the training backprop kernels
//! (`digit_input_gradients`): a one-hot delta on the logit propagated with the
//! same `D · W` products, one layer further down to the input.

use std::path::Path;

use anyhow::{bail, Result};

use crate::{matrix::DenseMatrix, model::MultilayerPerceptron};

use super::{
    artifact::DigitModelArtifact,
    data::{DigitDataset, DIGIT_CLASSES, IMAGE_PIXELS},
    image_grid::{draw_map_grid, MapCell, MapStyle},
    training::{digit_input_gradients, evaluate_digit_model, DigitTrainingError},
};

/// Samples per `digit_input_gradients` call in Integrated Gradients; bounds
/// the interpolation matrix to `64 * steps * 784` values.
const IG_CHUNK: usize = 64;

/// `|∂z_c / ∂x|` for every row of `inputs` and its target class.
pub fn saliency(
    model: &MultilayerPerceptron,
    inputs: &DenseMatrix,
    targets: &[usize],
) -> Result<DenseMatrix, DigitTrainingError> {
    let gradients = digit_input_gradients(model, inputs, targets)?;
    Ok(DenseMatrix::new(
        inputs.rows(),
        inputs.cols(),
        gradients
            .as_slice()
            .iter()
            .map(|value| value.abs())
            .collect(),
    )?)
}

/// `x ⊙ ∂z_c / ∂x` for every row of `inputs` and its target class.
pub fn gradient_times_input(
    model: &MultilayerPerceptron,
    inputs: &DenseMatrix,
    targets: &[usize],
) -> Result<DenseMatrix, DigitTrainingError> {
    let gradients = digit_input_gradients(model, inputs, targets)?;
    Ok(DenseMatrix::new(
        inputs.rows(),
        inputs.cols(),
        gradients
            .as_slice()
            .iter()
            .zip(inputs.as_slice())
            .map(|(gradient, input)| gradient * input)
            .collect(),
    )?)
}

/// Integrated Gradients of `inputs[rows[i]]` for class `targets[i]`, with a
/// black baseline and `steps` midpoint Riemann steps. Returns one row per
/// entry of `rows`.
pub fn integrated_gradients(
    model: &MultilayerPerceptron,
    inputs: &DenseMatrix,
    rows: &[usize],
    targets: &[usize],
    steps: usize,
) -> Result<DenseMatrix, DigitTrainingError> {
    if rows.is_empty() || steps == 0 {
        return Err(DigitTrainingError::EmptyTrainingSet);
    }
    if targets.len() != rows.len() {
        return Err(DigitTrainingError::LabelLength);
    }
    if let Some(&row) = rows.iter().find(|&&row| row >= inputs.rows()) {
        return Err(DigitTrainingError::InvalidIndex(row));
    }
    let cols = inputs.cols();
    let mut attributions = Vec::with_capacity(rows.len() * cols);
    for (chunk_rows, chunk_targets) in rows.chunks(IG_CHUNK).zip(targets.chunks(IG_CHUNK)) {
        let mut path = Vec::with_capacity(chunk_rows.len() * steps * cols);
        let mut path_targets = Vec::with_capacity(chunk_rows.len() * steps);
        for (&row, &target) in chunk_rows.iter().zip(chunk_targets) {
            let input = inputs.row_unchecked(row);
            for step in 0..steps {
                let alpha = (step as f64 + 0.5) / steps as f64;
                path.extend(input.iter().map(|value| alpha * value));
                path_targets.push(target);
            }
        }
        let path = DenseMatrix::new(chunk_rows.len() * steps, cols, path)?;
        let gradients = digit_input_gradients(model, &path, &path_targets)?;
        for (sample, &row) in chunk_rows.iter().enumerate() {
            let input = inputs.row_unchecked(row);
            let mut mean = vec![0.0; cols];
            for step in 0..steps {
                let gradient = gradients.row_unchecked(sample * steps + step);
                for (sum, value) in mean.iter_mut().zip(gradient) {
                    *sum += value;
                }
            }
            attributions.extend(
                mean.iter()
                    .zip(input)
                    .map(|(sum, value)| value * sum / steps as f64),
            );
        }
    }
    Ok(DenseMatrix::new(rows.len(), cols, attributions)?)
}

/// Completeness check of one attribution vector against the logit change
/// from the black baseline.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Completeness {
    pub logit_input: f64,
    pub logit_baseline: f64,
    pub attribution_sum: f64,
    /// `|Σ attributions − (z_c(x) − z_c(0))|`.
    pub absolute_gap: f64,
    /// `absolute_gap / max(|z_c(x) − z_c(0)|, 1e-12)`.
    pub relative_gap: f64,
}

pub fn completeness(
    model: &MultilayerPerceptron,
    input: &[f64],
    target: usize,
    attributions: &[f64],
) -> Result<Completeness, DigitTrainingError> {
    if target >= DIGIT_CLASSES {
        return Err(DigitTrainingError::InvalidClass(target));
    }
    let logit_input = model.predict(input)?[target];
    let logit_baseline = model.predict(&vec![0.0; input.len()])?[target];
    let attribution_sum = attributions.iter().sum::<f64>();
    let change = logit_input - logit_baseline;
    let absolute_gap = (attribution_sum - change).abs();
    Ok(Completeness {
        logit_input,
        logit_baseline,
        attribution_sum,
        absolute_gap,
        relative_gap: absolute_gap / change.abs().max(1e-12),
    })
}

fn select_rows(features: &DenseMatrix, rows: &[usize]) -> Result<DenseMatrix, DigitTrainingError> {
    let mut values = Vec::with_capacity(rows.len() * features.cols());
    for &row in rows {
        values.extend_from_slice(features.row_unchecked(row));
    }
    Ok(DenseMatrix::new(rows.len(), features.cols(), values)?)
}

/// Mean of the rows of `matrix` (a 784-pixel map).
fn column_means(matrix: &DenseMatrix) -> Vec<f64> {
    let mut means = vec![0.0; matrix.cols()];
    for row in 0..matrix.rows() {
        for (mean, value) in means.iter_mut().zip(matrix.row_unchecked(row)) {
            *mean += value;
        }
    }
    means
        .iter()
        .map(|sum| sum / matrix.rows().max(1) as f64)
        .collect()
}

/// One IG computation, kept for the completeness report.
struct CompletenessRow {
    row: usize,
    label: usize,
    target: usize,
    check: Completeness,
}

const METHODS: [&str; 4] = [
    "input_mean",
    "saliency",
    "gradient_x_input",
    "integrated_gradients",
];

/// Class-mean attribution maps, per-example maps for correct and
/// misclassified test digits, and the Integrated Gradients completeness
/// report. Writes CSVs and PNGs into `output`.
pub fn run_attribution_study(
    dataset: &DigitDataset,
    artifact: &DigitModelArtifact,
    steps: usize,
    per_class: usize,
    examples: usize,
    output: &Path,
) -> Result<()> {
    if steps == 0 || per_class == 0 || examples == 0 {
        bail!("steps, per-class and examples must all be at least 1");
    }
    let model = &artifact.model;
    let topology = model.topology();
    if topology.first() != Some(&IMAGE_PIXELS) || topology.last() != Some(&DIGIT_CLASSES) {
        bail!("model has topology {topology:?}; expected 784 -> ... -> 10");
    }
    let features = &dataset.features;
    let labels = &dataset.labels;
    let all_rows = (0..labels.len()).collect::<Vec<_>>();
    let predictions = evaluate_digit_model(model, features, labels, &all_rows)?.predictions;
    let mut checks = Vec::new();

    // (a) Class means over correctly classified digits, for the true class.
    let mut class_means: Vec<Option<[Vec<f64>; 4]>> = Vec::with_capacity(DIGIT_CLASSES);
    let mut writer = csv::Writer::from_path(output.join("attribution_class_means.csv"))?;
    writer.write_record(["method", "class", "pixel", "value"])?;
    for digit in 0..DIGIT_CLASSES {
        let rows = all_rows
            .iter()
            .copied()
            .filter(|&row| labels[row] == digit && predictions[row] == digit)
            .take(per_class)
            .collect::<Vec<_>>();
        if rows.is_empty() {
            eprintln!("attribution class={digit}: no correctly classified examples");
            class_means.push(None);
            continue;
        }
        let inputs = select_rows(features, &rows)?;
        let targets = vec![digit; rows.len()];
        let integrated = integrated_gradients(model, features, &rows, &targets, steps)?;
        for (position, &row) in rows.iter().enumerate() {
            checks.push(CompletenessRow {
                row,
                label: digit,
                target: digit,
                check: completeness(
                    model,
                    features.row_unchecked(row),
                    digit,
                    integrated.row_unchecked(position),
                )?,
            });
        }
        let maps = [
            column_means(&inputs),
            column_means(&saliency(model, &inputs, &targets)?),
            column_means(&gradient_times_input(model, &inputs, &targets)?),
            column_means(&integrated),
        ];
        for (method, map) in METHODS.iter().zip(&maps) {
            for (pixel, value) in map.iter().enumerate() {
                writer.serialize((method, digit, pixel, value))?;
            }
        }
        class_means.push(Some(maps));
    }
    writer.flush()?;

    let styles = [
        MapStyle::Grayscale,
        MapStyle::Sequential,
        MapStyle::Diverging,
        MapStyle::Diverging,
    ];
    let short = ["mean input", "saliency", "grad x input", "IG"];
    let mut cells = Vec::new();
    for (method, style) in styles.iter().enumerate() {
        for (digit, maps) in class_means.iter().enumerate() {
            cells.push(maps.as_ref().map(|maps| MapCell {
                title: format!("{digit}: {}", short[method]),
                values: &maps[method],
                style: *style,
                highlight: false,
            }));
        }
    }
    draw_map_grid(
        &output.join("attribution_class_means.png"),
        &format!(
            "Class means over up to {per_class} correct test digits (rows: input, saliency, gradient x input, IG; red = +, blue = -)"
        ),
        METHODS.len(),
        DIGIT_CLASSES,
        &cells,
    )?;

    // (b) Examples: the first correct and misclassified digits in file order.
    let correct = all_rows
        .iter()
        .copied()
        .filter(|&row| predictions[row] == labels[row])
        .take(examples);
    let wrong = all_rows
        .iter()
        .copied()
        .filter(|&row| predictions[row] != labels[row])
        .take(examples);
    let example_rows = correct.chain(wrong).collect::<Vec<_>>();
    let inputs = select_rows(features, &example_rows)?;
    let predicted = example_rows
        .iter()
        .map(|&row| predictions[row])
        .collect::<Vec<_>>();
    let truth = example_rows
        .iter()
        .map(|&row| labels[row])
        .collect::<Vec<_>>();
    let example_saliency = saliency(model, &inputs, &predicted)?;
    let example_gradient_input = gradient_times_input(model, &inputs, &predicted)?;
    let example_ig_predicted =
        integrated_gradients(model, features, &example_rows, &predicted, steps)?;
    let example_ig_true = integrated_gradients(model, features, &example_rows, &truth, steps)?;

    let mut writer = csv::Writer::from_path(output.join("attribution_examples.csv"))?;
    writer.write_record([
        "row",
        "label",
        "predicted",
        "method",
        "target_class",
        "pixel",
        "value",
    ])?;
    let example_maps = [
        ("saliency", &example_saliency, &predicted),
        ("gradient_x_input", &example_gradient_input, &predicted),
        ("integrated_gradients", &example_ig_predicted, &predicted),
        ("integrated_gradients", &example_ig_true, &truth),
    ];
    let mut cells = Vec::new();
    for (position, &row) in example_rows.iter().enumerate() {
        let (label, prediction) = (labels[row], predictions[row]);
        for (method, maps, targets) in example_maps {
            for (pixel, value) in maps.row_unchecked(position).iter().enumerate() {
                writer.serialize((
                    row,
                    label,
                    prediction,
                    method,
                    targets[position],
                    pixel,
                    value,
                ))?;
            }
        }
        for (maps, target) in [
            (&example_ig_predicted, prediction),
            (&example_ig_true, label),
        ] {
            checks.push(CompletenessRow {
                row,
                label,
                target,
                check: completeness(
                    model,
                    features.row_unchecked(row),
                    target,
                    maps.row_unchecked(position),
                )?,
            });
        }
        let wrong = label != prediction;
        cells.push(Some(MapCell {
            title: format!("row {row}: label {label} -> pred {prediction}"),
            values: inputs.row_unchecked(position),
            style: MapStyle::Grayscale,
            highlight: wrong,
        }));
        for (title, maps, style) in [
            (
                format!("saliency (pred {prediction})"),
                &example_saliency,
                MapStyle::Sequential,
            ),
            (
                format!("grad x input (pred {prediction})"),
                &example_gradient_input,
                MapStyle::Diverging,
            ),
            (
                format!("IG (pred {prediction})"),
                &example_ig_predicted,
                MapStyle::Diverging,
            ),
            (
                format!("IG (true {label})"),
                &example_ig_true,
                MapStyle::Diverging,
            ),
        ] {
            cells.push(Some(MapCell {
                title,
                values: maps.row_unchecked(position),
                style,
                highlight: wrong,
            }));
        }
    }
    writer.flush()?;
    draw_map_grid(
        &output.join("attribution_examples.png"),
        "Correct / misclassified digits (red title = wrong)",
        example_rows.len().max(1),
        5,
        &cells,
    )?;

    // (c) Completeness of every IG computation.
    let mut writer = csv::Writer::from_path(output.join("attribution_completeness.csv"))?;
    writer.write_record([
        "row",
        "label",
        "target_class",
        "logit_input",
        "logit_baseline",
        "attribution_sum",
        "absolute_gap",
        "relative_gap",
    ])?;
    for entry in &checks {
        writer.serialize((
            entry.row,
            entry.label,
            entry.target,
            entry.check.logit_input,
            entry.check.logit_baseline,
            entry.check.attribution_sum,
            entry.check.absolute_gap,
            entry.check.relative_gap,
        ))?;
    }
    writer.flush()?;
    let gaps = checks
        .iter()
        .map(|entry| entry.check.relative_gap)
        .collect::<Vec<_>>();
    let mean_gap = gaps.iter().sum::<f64>() / gaps.len().max(1) as f64;
    let max_gap = gaps.iter().copied().fold(0.0_f64, f64::max);
    eprintln!(
        "attribution: {} IG computations with {steps} steps, relative completeness gap mean={mean_gap:.5} max={max_gap:.5}",
        checks.len()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    use super::*;
    use crate::model::Activation;

    fn network(hidden: usize, activation: Activation, seed: u64) -> MultilayerPerceptron {
        MultilayerPerceptron::new(&[6, hidden, 10], &[activation, Activation::Linear], seed)
            .unwrap()
    }

    fn random_inputs(rows: usize, seed: u64) -> DenseMatrix {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        DenseMatrix::new(
            rows,
            6,
            (0..rows * 6).map(|_| rng.gen_range(-1.0..1.0)).collect(),
        )
        .unwrap()
    }

    #[test]
    fn input_gradients_match_central_finite_differences() {
        for activation in [Activation::Tanh, Activation::Sigmoid] {
            let model = network(5, activation, 4);
            let inputs = random_inputs(4, 9);
            for class in [0, 3, 9] {
                let targets = vec![class; inputs.rows()];
                let gradients = digit_input_gradients(&model, &inputs, &targets).unwrap();
                for row in 0..inputs.rows() {
                    let input = inputs.row(row).unwrap();
                    for pixel in 0..input.len() {
                        let eps = 1e-6;
                        let mut plus = input.to_vec();
                        let mut minus = input.to_vec();
                        plus[pixel] += eps;
                        minus[pixel] -= eps;
                        let numeric = (model.predict(&plus).unwrap()[class]
                            - model.predict(&minus).unwrap()[class])
                            / (2.0 * eps);
                        let analytic = gradients.row(row).unwrap()[pixel];
                        assert!(
                            (numeric - analytic).abs() < 1e-6,
                            "{activation:?} class {class} row {row} pixel {pixel}: {numeric} vs {analytic}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn integrated_gradients_satisfy_completeness() {
        let model = network(8, Activation::Tanh, 11);
        let inputs = random_inputs(3, 5);
        let rows = [0, 1, 2];
        for class in [1, 7] {
            let targets = [class; 3];
            let attributions = integrated_gradients(&model, &inputs, &rows, &targets, 300).unwrap();
            for row in rows {
                let check = completeness(
                    &model,
                    inputs.row(row).unwrap(),
                    class,
                    attributions.row(row).unwrap(),
                )
                .unwrap();
                let change = check.logit_input - check.logit_baseline;
                assert!(
                    check.absolute_gap < 1e-3 * change.abs().max(1.0),
                    "class {class} row {row}: gap {} for change {change}",
                    check.absolute_gap
                );
            }
        }
    }

    #[test]
    fn saliency_is_non_negative_and_gradient_times_input_vanishes_on_zero_pixels() {
        let model = network(5, Activation::Tanh, 2);
        let mut values = random_inputs(3, 1).as_slice().to_vec();
        for index in (0..values.len()).step_by(3) {
            values[index] = 0.0;
        }
        let inputs = DenseMatrix::new(3, 6, values).unwrap();
        let targets = [2, 5, 8];
        let saliency = saliency(&model, &inputs, &targets).unwrap();
        assert!(saliency.as_slice().iter().all(|&value| value >= 0.0));
        let product = gradient_times_input(&model, &inputs, &targets).unwrap();
        for (value, input) in product.as_slice().iter().zip(inputs.as_slice()) {
            if *input == 0.0 {
                assert_eq!(*value, 0.0);
            }
        }
    }

    #[test]
    fn rejects_invalid_targets() {
        let model = network(5, Activation::Tanh, 2);
        let inputs = random_inputs(2, 1);
        assert!(matches!(
            digit_input_gradients(&model, &inputs, &[0]),
            Err(DigitTrainingError::LabelLength)
        ));
        assert!(matches!(
            digit_input_gradients(&model, &inputs, &[0, 10]),
            Err(DigitTrainingError::InvalidClass(10))
        ));
        let wrong_input =
            MultilayerPerceptron::new(&[5, 3, 10], &[Activation::Tanh, Activation::Linear], 1)
                .unwrap();
        assert!(matches!(
            digit_input_gradients(&wrong_input, &inputs, &[0, 1]),
            Err(DigitTrainingError::InvalidTopology)
        ));
    }
}
