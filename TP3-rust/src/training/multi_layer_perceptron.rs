use rand::{seq::SliceRandom, SeedableRng};
use rand_chacha::ChaCha8Rng;

use crate::{
    loss::Loss,
    matrix::DenseMatrix,
    model::{dot, ModelError, MultilayerPerceptron},
};

use super::{
    validate_common, validate_indices, EpochMetrics, TrainingConfig, TrainingError, TrainingReport,
};

#[cfg(feature = "training-logs")]
use super::print_epoch_loss;

pub fn train_mlp_scalar<L: Loss + Sync>(
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

fn indexed_loss_mlp<L: Loss + Sync>(
    model: &MultilayerPerceptron,
    features: &DenseMatrix,
    targets: &[f64],
    indices: &[usize],
    loss: &L,
) -> Result<f64, TrainingError> {
    use rayon::prelude::*;

    let total = indices
        .par_iter()
        .map(|&index| {
            Ok::<_, TrainingError>(loss.sample(
                model.predict(features.row_unchecked(index))?[0],
                targets[index],
            ))
        })
        .try_reduce(|| 0.0, |left, right| Ok(left + right))?;
    Ok(total / indices.len() as f64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::loss::MeanSquaredError;
    use crate::model::Activation;

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
