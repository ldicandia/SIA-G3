//! Exercise 1: step perceptron (AND, XOR) and single-layer/MLP regression checks.

use anyhow::{ensure, Result};

use crate::{
    matrix::DenseMatrix,
    metrics::regression_metrics,
    model::{Activation, MultilayerPerceptron, SingleLayerPerceptron, StepPerceptron},
    training::{train_mlp_scalar, train_single_layer, train_step_perceptron, TrainingConfig},
    MeanSquaredError,
};

pub fn run() -> Result<()> {
    validate_and()?;
    validate_linear_regression()?;
    validate_nonlinear_regression()?;
    validate_xor(&[2, 2, 1], 19)?;
    validate_xor(&[2, 3, 2, 1], 11)?;
    compare_xor_with_step()?;
    println!("All educational validations passed.");
    Ok(())
}

fn validate_and() -> Result<()> {
    let features = logic_inputs();
    let targets = [-1, -1, -1, 1];
    let mut model = StepPerceptron::new(2)?;
    train_step_perceptron(
        &mut model,
        &features,
        &targets,
        TrainingConfig::fixed(0.1, 100, 42),
    )?;
    let predictions = (0..features.rows())
        .map(|row| model.predict(features.row(row).unwrap()).unwrap())
        .collect::<Vec<_>>();
    ensure!(
        predictions == targets,
        "AND validation failed: {predictions:?}"
    );
    println!("AND step perceptron: {predictions:?}");
    Ok(())
}

fn validate_linear_regression() -> Result<()> {
    let (features, targets) = sampled_function(|x| x);
    let mut model = SingleLayerPerceptron::new(1, Activation::Linear, 42)?;
    let indices = (0..features.rows()).collect::<Vec<_>>();
    train_single_layer(
        &mut model,
        &features,
        &targets,
        &indices,
        None,
        &MeanSquaredError,
        TrainingConfig::fixed(0.01, 500, 42),
    )?;
    let predictions = indices
        .iter()
        .map(|&row| model.predict(features.row(row).unwrap()).unwrap())
        .collect::<Vec<_>>();
    let rmse = regression_metrics(&predictions, &targets)?.rmse;
    ensure!(rmse < 0.02, "linear validation RMSE was {rmse}");
    println!("Linear y=x RMSE: {rmse:.6}");
    Ok(())
}

fn validate_nonlinear_regression() -> Result<()> {
    let (features, targets) = sampled_function(f64::tanh);
    let mut model = SingleLayerPerceptron::new(1, Activation::Tanh, 42)?;
    let indices = (0..features.rows()).collect::<Vec<_>>();
    train_single_layer(
        &mut model,
        &features,
        &targets,
        &indices,
        None,
        &MeanSquaredError,
        TrainingConfig::fixed(0.01, 1000, 42),
    )?;
    let predictions = indices
        .iter()
        .map(|&row| model.predict(features.row(row).unwrap()).unwrap())
        .collect::<Vec<_>>();
    let rmse = regression_metrics(&predictions, &targets)?.rmse;
    ensure!(rmse < 0.02, "nonlinear validation RMSE was {rmse}");
    println!("Nonlinear y=tanh(x) RMSE: {rmse:.6}");
    Ok(())
}

fn validate_xor(topology: &[usize], seed: u64) -> Result<()> {
    let features = logic_inputs();
    let targets = [1.0, 1.0, -1.0, -1.0];
    let activations = vec![Activation::Tanh; topology.len() - 1];
    let mut model = MultilayerPerceptron::new(topology, &activations, seed)?;
    let indices = (0..features.rows()).collect::<Vec<_>>();
    train_mlp_scalar(
        &mut model,
        &features,
        &targets,
        &indices,
        None,
        &MeanSquaredError,
        TrainingConfig::fixed(0.03, 20_000, seed),
    )?;
    let predictions = indices
        .iter()
        .map(|&row| model.predict(features.row(row).unwrap()).unwrap()[0])
        .collect::<Vec<_>>();
    let signs = predictions
        .iter()
        .map(|&value| if value >= 0.0 { 1 } else { -1 })
        .collect::<Vec<_>>();
    ensure!(
        signs == [1, 1, -1, -1],
        "XOR {topology:?} failed: {predictions:?}"
    );
    println!("XOR {topology:?}: {predictions:?}");
    Ok(())
}

fn compare_xor_with_step() -> Result<()> {
    let features = logic_inputs();
    let targets = [1, 1, -1, -1];
    let mut model = StepPerceptron::new(2)?;
    train_step_perceptron(
        &mut model,
        &features,
        &targets,
        TrainingConfig::fixed(0.1, 100, 42),
    )?;
    let correct = (0..features.rows())
        .filter(|&row| model.predict(features.row(row).unwrap()).unwrap() == targets[row])
        .count();
    ensure!(correct < 4, "a step perceptron unexpectedly solved XOR");
    println!("XOR step perceptron: {correct}/4 correct (not linearly separable)");
    Ok(())
}

fn logic_inputs() -> DenseMatrix {
    DenseMatrix::from_rows(vec![
        vec![-1.0, 1.0],
        vec![1.0, -1.0],
        vec![-1.0, -1.0],
        vec![1.0, 1.0],
    ])
    .unwrap()
}

fn sampled_function(function: fn(f64) -> f64) -> (DenseMatrix, Vec<f64>) {
    let values = (0..50)
        .map(|index| -1.0 + 2.0 * index as f64 / 49.0)
        .collect::<Vec<_>>();
    let targets = values.iter().copied().map(function).collect();
    let features = DenseMatrix::new(values.len(), 1, values).unwrap();
    (features, targets)
}
