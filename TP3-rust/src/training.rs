use rand::{seq::SliceRandom, SeedableRng};
use rand_chacha::ChaCha8Rng;
use thiserror::Error;

use crate::{
    loss::Loss,
    matrix::DenseMatrix,
    model::{dot, ModelError, MultilayerPerceptron, SingleLayerPerceptron, StepPerceptron},
};

#[derive(Clone, Copy, Debug)]
pub struct EarlyStopping {
    pub patience: usize,
    pub min_delta: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct TrainingConfig {
    pub learning_rate: f64,
    pub max_epochs: usize,
    pub shuffle: bool,
    pub seed: u64,
    pub early_stopping: Option<EarlyStopping>,
}

impl TrainingConfig {
    pub fn fixed(learning_rate: f64, max_epochs: usize, seed: u64) -> Self {
        Self {
            learning_rate,
            max_epochs,
            shuffle: true,
            seed,
            early_stopping: None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct EpochMetrics {
    pub epoch: usize,
    pub train_loss: f64,
    pub validation_loss: Option<f64>,
}

#[derive(Clone, Debug)]
pub struct TrainingReport {
    pub history: Vec<EpochMetrics>,
    pub best_epoch: usize,
    pub best_monitored_loss: f64,
    pub stopped_early: bool,
}

#[derive(Debug, Error)]
pub enum TrainingError {
    #[error("training configuration values must be finite and positive")]
    InvalidConfig,
    #[error("training indices cannot be empty")]
    EmptyTrainingSet,
    #[error("target count {targets} does not match matrix rows {rows}")]
    TargetLength { targets: usize, rows: usize },
    #[error("sample index {index} is out of bounds")]
    InvalidIndex { index: usize },
    #[error("step perceptron targets must be either -1 or 1")]
    InvalidStepTarget,
    #[error("MLP scalar training requires exactly one output")]
    NonScalarOutput,
    #[error("training produced a non-finite loss")]
    NonFiniteLoss,
    #[error(transparent)]
    Model(#[from] ModelError),
}

pub fn train_step_perceptron(
    model: &mut StepPerceptron,
    features: &DenseMatrix,
    targets: &[i8],
    config: TrainingConfig,
) -> Result<Vec<usize>, TrainingError> {
    validate_common(
        features,
        targets.len(),
        &(0..features.rows()).collect::<Vec<_>>(),
        config,
    )?;
    if targets.iter().any(|target| !matches!(target, -1 | 1)) {
        return Err(TrainingError::InvalidStepTarget);
    }
    if features.cols() != model.weights.len() {
        return Err(ModelError::InputDimension {
            expected: model.weights.len(),
            actual: features.cols(),
        }
        .into());
    }

    let mut order = (0..features.rows()).collect::<Vec<_>>();
    let mut rng = ChaCha8Rng::seed_from_u64(config.seed);
    let mut mistakes_by_epoch = Vec::with_capacity(config.max_epochs);
    for _ in 0..config.max_epochs {
        if config.shuffle {
            order.shuffle(&mut rng);
        }
        let mut mistakes = 0;
        for &index in &order {
            let input = features.row_unchecked(index);
            let prediction = if dot(&model.weights, input) + model.bias >= 0.0 {
                1
            } else {
                -1
            };
            if prediction != targets[index] {
                let correction = config.learning_rate * (targets[index] - prediction) as f64;
                for (weight, value) in model.weights.iter_mut().zip(input) {
                    *weight += correction * value;
                }
                model.bias += correction;
                mistakes += 1;
            }
        }
        mistakes_by_epoch.push(mistakes);
        if mistakes == 0 {
            break;
        }
    }
    Ok(mistakes_by_epoch)
}

pub fn train_single_layer<L: Loss>(
    model: &mut SingleLayerPerceptron,
    features: &DenseMatrix,
    targets: &[f64],
    train_indices: &[usize],
    validation_indices: Option<&[usize]>,
    loss: &L,
    config: TrainingConfig,
) -> Result<TrainingReport, TrainingError> {
    validate_common(features, targets.len(), train_indices, config)?;
    validate_indices(features, validation_indices.unwrap_or_default())?;
    if features.cols() != model.input_size() {
        return Err(ModelError::InputDimension {
            expected: model.input_size(),
            actual: features.cols(),
        }
        .into());
    }

    let mut order = train_indices.to_vec();
    let mut rng = ChaCha8Rng::seed_from_u64(config.seed);
    let mut history = Vec::with_capacity(config.max_epochs);
    let mut best_model = model.clone();
    let mut best_epoch = 0;
    let mut best_loss = f64::INFINITY;
    let mut stale_epochs = 0;
    let mut stopped_early = false;

    for epoch in 1..=config.max_epochs {
        if config.shuffle {
            order.shuffle(&mut rng);
        }
        for &index in &order {
            let input = features.row_unchecked(index);
            let linear = dot(&model.weights, input) + model.bias;
            let prediction = model.activation().apply(linear);
            let delta = loss.derivative(prediction, targets[index])
                * model.activation().derivative_from_output(prediction);
            for (weight, value) in model.weights.iter_mut().zip(input) {
                *weight -= config.learning_rate * delta * value;
            }
            model.bias -= config.learning_rate * delta;
        }

        let train_loss = indexed_loss_single(model, features, targets, train_indices, loss)?;
        let validation_loss = validation_indices
            .filter(|indices| !indices.is_empty())
            .map(|indices| indexed_loss_single(model, features, targets, indices, loss))
            .transpose()?;
        let monitored = validation_loss.unwrap_or(train_loss);
        if !monitored.is_finite() {
            return Err(TrainingError::NonFiniteLoss);
        }
        history.push(EpochMetrics {
            epoch,
            train_loss,
            validation_loss,
        });
        #[cfg(feature = "training-logs")]
        print_epoch_loss(
            "single-layer",
            config.learning_rate,
            history.last().unwrap(),
        );

        let min_delta = config
            .early_stopping
            .map(|stopping| stopping.min_delta)
            .unwrap_or(0.0);
        if monitored + min_delta < best_loss {
            best_loss = monitored;
            best_epoch = epoch;
            best_model = model.clone();
            stale_epochs = 0;
        } else {
            stale_epochs += 1;
        }
        if let Some(stopping) = config.early_stopping {
            if stale_epochs >= stopping.patience {
                stopped_early = true;
                break;
            }
        }
    }
    if config.early_stopping.is_some() {
        *model = best_model;
    }

    Ok(TrainingReport {
        history,
        best_epoch,
        best_monitored_loss: best_loss,
        stopped_early,
    })
}

pub fn train_mlp_scalar<L: Loss>(
    model: &mut MultilayerPerceptron,
    features: &DenseMatrix,
    targets: &[f64],
    train_indices: &[usize],
    validation_indices: Option<&[usize]>,
    loss: &L,
    config: TrainingConfig,
) -> Result<TrainingReport, TrainingError> {
    validate_common(features, targets.len(), train_indices, config)?;
    validate_indices(features, validation_indices.unwrap_or_default())?;
    let topology = model.topology();
    if features.cols() != topology[0] {
        return Err(ModelError::InputDimension {
            expected: topology[0],
            actual: features.cols(),
        }
        .into());
    }
    if *topology.last().unwrap() != 1 {
        return Err(TrainingError::NonScalarOutput);
    }

    let mut activations = topology
        .iter()
        .map(|&size| vec![0.0; size])
        .collect::<Vec<_>>();
    let mut deltas = topology[1..]
        .iter()
        .map(|&size| vec![0.0; size])
        .collect::<Vec<_>>();
    let mut order = train_indices.to_vec();
    let mut rng = ChaCha8Rng::seed_from_u64(config.seed);
    let mut history = Vec::with_capacity(config.max_epochs);
    let mut best_model = model.clone();
    let mut best_epoch = 0;
    let mut best_loss = f64::INFINITY;
    let mut stale_epochs = 0;
    let mut stopped_early = false;

    for epoch in 1..=config.max_epochs {
        if config.shuffle {
            order.shuffle(&mut rng);
        }
        for &index in &order {
            activations[0].copy_from_slice(features.row_unchecked(index));
            forward(model, &mut activations);
            backward_deltas(model, &activations, &mut deltas, targets[index], loss);
            apply_mlp_update(model, &activations, &deltas, config.learning_rate);
        }

        let train_loss = indexed_loss_mlp(model, features, targets, train_indices, loss)?;
        let validation_loss = validation_indices
            .filter(|indices| !indices.is_empty())
            .map(|indices| indexed_loss_mlp(model, features, targets, indices, loss))
            .transpose()?;
        let monitored = validation_loss.unwrap_or(train_loss);
        if !monitored.is_finite() {
            return Err(TrainingError::NonFiniteLoss);
        }
        history.push(EpochMetrics {
            epoch,
            train_loss,
            validation_loss,
        });
        #[cfg(feature = "training-logs")]
        print_epoch_loss("mlp", config.learning_rate, history.last().unwrap());
        let min_delta = config
            .early_stopping
            .map(|stopping| stopping.min_delta)
            .unwrap_or(0.0);
        if monitored + min_delta < best_loss {
            best_loss = monitored;
            best_epoch = epoch;
            best_model = model.clone();
            stale_epochs = 0;
        } else {
            stale_epochs += 1;
        }
        if let Some(stopping) = config.early_stopping {
            if stale_epochs >= stopping.patience {
                stopped_early = true;
                break;
            }
        }
    }
    if config.early_stopping.is_some() {
        *model = best_model;
    }
    Ok(TrainingReport {
        history,
        best_epoch,
        best_monitored_loss: best_loss,
        stopped_early,
    })
}

fn forward(model: &MultilayerPerceptron, activations: &mut [Vec<f64>]) {
    for (layer_index, layer) in model.layers.iter().enumerate() {
        let (before, after) = activations.split_at_mut(layer_index + 1);
        let input = &before[layer_index];
        let output = &mut after[0];
        for (output_index, output_value) in output.iter_mut().enumerate() {
            let start = output_index * layer.input_size;
            *output_value = layer.activation.apply(
                dot(&layer.weights[start..start + layer.input_size], input)
                    + layer.biases[output_index],
            );
        }
    }
}

fn backward_deltas<L: Loss>(
    model: &MultilayerPerceptron,
    activations: &[Vec<f64>],
    deltas: &mut [Vec<f64>],
    target: f64,
    loss: &L,
) {
    let last = model.layers.len() - 1;
    let output = activations.last().unwrap()[0];
    deltas[last][0] = loss.derivative(output, target)
        * model.layers[last].activation.derivative_from_output(output);

    for layer_index in (0..last).rev() {
        let next_layer = &model.layers[layer_index + 1];
        let (current_deltas, next_deltas) = deltas.split_at_mut(layer_index + 1);
        for neuron in 0..model.layers[layer_index].output_size {
            let propagated = (0..next_layer.output_size)
                .map(|next| {
                    next_layer.weights[next * next_layer.input_size + neuron] * next_deltas[0][next]
                })
                .sum::<f64>();
            let output = activations[layer_index + 1][neuron];
            current_deltas[layer_index][neuron] = propagated
                * model.layers[layer_index]
                    .activation
                    .derivative_from_output(output);
        }
    }
}

fn apply_mlp_update(
    model: &mut MultilayerPerceptron,
    activations: &[Vec<f64>],
    deltas: &[Vec<f64>],
    learning_rate: f64,
) {
    for (layer_index, layer) in model.layers.iter_mut().enumerate() {
        for output in 0..layer.output_size {
            let delta = deltas[layer_index][output];
            let start = output * layer.input_size;
            for input in 0..layer.input_size {
                layer.weights[start + input] -=
                    learning_rate * delta * activations[layer_index][input];
            }
            layer.biases[output] -= learning_rate * delta;
        }
    }
}

fn indexed_loss_single<L: Loss>(
    model: &SingleLayerPerceptron,
    features: &DenseMatrix,
    targets: &[f64],
    indices: &[usize],
    loss: &L,
) -> Result<f64, TrainingError> {
    let total = indices.iter().try_fold(0.0, |total, &index| {
        Ok::<_, TrainingError>(
            total
                + loss.sample(
                    model.predict(features.row_unchecked(index))?,
                    targets[index],
                ),
        )
    })?;
    Ok(total / indices.len() as f64)
}

fn indexed_loss_mlp<L: Loss>(
    model: &MultilayerPerceptron,
    features: &DenseMatrix,
    targets: &[f64],
    indices: &[usize],
    loss: &L,
) -> Result<f64, TrainingError> {
    let total = indices.iter().try_fold(0.0, |total, &index| {
        Ok::<_, TrainingError>(
            total
                + loss.sample(
                    model.predict(features.row_unchecked(index))?[0],
                    targets[index],
                ),
        )
    })?;
    Ok(total / indices.len() as f64)
}

fn validate_common(
    features: &DenseMatrix,
    target_count: usize,
    train_indices: &[usize],
    config: TrainingConfig,
) -> Result<(), TrainingError> {
    if !config.learning_rate.is_finite()
        || config.learning_rate <= 0.0
        || config.max_epochs == 0
        || config.early_stopping.is_some_and(|stopping| {
            stopping.patience == 0 || !stopping.min_delta.is_finite() || stopping.min_delta < 0.0
        })
    {
        return Err(TrainingError::InvalidConfig);
    }
    if target_count != features.rows() {
        return Err(TrainingError::TargetLength {
            targets: target_count,
            rows: features.rows(),
        });
    }
    if train_indices.is_empty() {
        return Err(TrainingError::EmptyTrainingSet);
    }
    validate_indices(features, train_indices)
}

fn validate_indices(features: &DenseMatrix, indices: &[usize]) -> Result<(), TrainingError> {
    if let Some(&index) = indices.iter().find(|&&index| index >= features.rows()) {
        return Err(TrainingError::InvalidIndex { index });
    }
    Ok(())
}

#[cfg(feature = "training-logs")]
fn print_epoch_loss(model: &str, learning_rate: f64, metrics: &EpochMetrics) {
    if let Some(validation_loss) = metrics.validation_loss {
        eprintln!(
            "model={model} learning_rate={learning_rate} epoch={} train_loss={:.10} validation_loss={:.10}",
            metrics.epoch, metrics.train_loss, validation_loss
        );
    } else {
        eprintln!(
            "model={model} learning_rate={learning_rate} epoch={} train_loss={:.10}",
            metrics.epoch, metrics.train_loss
        );
    }
}

pub fn predict_single_indices(
    model: &SingleLayerPerceptron,
    features: &DenseMatrix,
    indices: &[usize],
) -> Result<Vec<f64>, TrainingError> {
    validate_indices(features, indices)?;
    indices
        .iter()
        .map(|&index| Ok(model.predict(features.row_unchecked(index))?))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::loss::MeanSquaredError;
    use crate::model::Activation;

    #[test]
    fn one_linear_update_reduces_single_sample_loss() {
        let features = DenseMatrix::from_rows(vec![vec![1.0]]).unwrap();
        let targets = [0.5];
        let mut model = SingleLayerPerceptron::new(1, Activation::Linear, 7).unwrap();
        let before = MeanSquaredError.sample(model.predict(&[1.0]).unwrap(), targets[0]);
        train_single_layer(
            &mut model,
            &features,
            &targets,
            &[0],
            None,
            &MeanSquaredError,
            TrainingConfig::fixed(0.1, 1, 1),
        )
        .unwrap();
        let after = MeanSquaredError.sample(model.predict(&[1.0]).unwrap(), targets[0]);
        assert!(after < before);
    }

    #[test]
    fn one_mlp_update_reduces_single_sample_loss() {
        let features = DenseMatrix::from_rows(vec![vec![0.2, -0.4]]).unwrap();
        let targets = [0.7];
        let mut model =
            MultilayerPerceptron::new(&[2, 2, 1], &[Activation::Tanh, Activation::Linear], 3)
                .unwrap();
        let before = MeanSquaredError.sample(model.predict(&[0.2, -0.4]).unwrap()[0], 0.7);
        train_mlp_scalar(
            &mut model,
            &features,
            &targets,
            &[0],
            None,
            &MeanSquaredError,
            TrainingConfig::fixed(0.01, 1, 2),
        )
        .unwrap();
        let after = MeanSquaredError.sample(model.predict(&[0.2, -0.4]).unwrap()[0], 0.7);
        assert!(after < before);
    }
}
