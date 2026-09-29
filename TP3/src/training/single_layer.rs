use rand::{seq::SliceRandom, SeedableRng};
use rand_chacha::ChaCha8Rng;

use crate::{
    loss::Loss,
    matrix::DenseMatrix,
    model::{dot, ModelError, SingleLayerPerceptron},
};

use super::{
    validate_common, validate_indices, EpochMetrics, TrainingConfig, TrainingError, TrainingReport,
};

#[cfg(feature = "training-logs")]
use super::print_epoch_loss;

pub fn train_single_layer<L: Loss + Sync>(
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

fn indexed_loss_single<L: Loss + Sync>(
    model: &SingleLayerPerceptron,
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
                model.predict(features.row_unchecked(index))?,
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
}
