use crate::model::{softmax_in_place, MultilayerPerceptron};

use super::optimizer::LayerValues;

pub(super) fn forward_softmax(model: &MultilayerPerceptron, activations: &mut [Vec<f64>]) {
    for (layer_index, layer) in model.layers.iter().enumerate() {
        let (before, after) = activations.split_at_mut(layer_index + 1);
        let input = &before[layer_index];
        let output = &mut after[0];
        for (output_index, output_value) in output.iter_mut().enumerate() {
            let start = output_index * layer.input_size;
            let value = layer.weights[start..start + layer.input_size]
                .iter()
                .zip(input)
                .map(|(weight, input)| weight * input)
                .sum::<f64>()
                + layer.biases[output_index];
            *output_value = layer.activation.apply(value);
        }
    }
    softmax_in_place(activations.last_mut().unwrap());
}

pub(super) fn backward(
    model: &MultilayerPerceptron,
    activations: &[Vec<f64>],
    deltas: &mut [Vec<f64>],
    target: usize,
) {
    let last = model.layers.len() - 1;
    deltas[last].copy_from_slice(activations.last().unwrap());
    deltas[last][target] -= 1.0;

    for layer_index in (0..last).rev() {
        let next_layer = &model.layers[layer_index + 1];
        let (current, next) = deltas.split_at_mut(layer_index + 1);
        for neuron in 0..model.layers[layer_index].output_size {
            let propagated = (0..next_layer.output_size)
                .map(|output| {
                    next_layer.weights[output * next_layer.input_size + neuron] * next[0][output]
                })
                .sum::<f64>();
            let output = activations[layer_index + 1][neuron];
            current[layer_index][neuron] = propagated
                * model.layers[layer_index]
                    .activation
                    .derivative_from_output(output);
        }
    }
}

pub(super) fn accumulate_gradients(
    activations: &[Vec<f64>],
    deltas: &[Vec<f64>],
    gradients: &mut [LayerValues],
) {
    for (layer_index, gradient) in gradients.iter_mut().enumerate() {
        let input = &activations[layer_index];
        for (output, &delta) in deltas[layer_index].iter().enumerate() {
            let start = output * input.len();
            for (weight_gradient, &value) in gradient.weights[start..start + input.len()]
                .iter_mut()
                .zip(input)
            {
                *weight_gradient += delta * value;
            }
            gradient.biases[output] += delta;
        }
    }
}

pub(super) fn argmax(values: &[f64]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|left, right| left.1.total_cmp(right.1))
        .map_or(0, |(index, _)| index)
}
