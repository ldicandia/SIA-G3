use rand::{seq::SliceRandom, SeedableRng};
use rand_chacha::ChaCha8Rng;

use crate::{
    matrix::DenseMatrix,
    model::{dot, ModelError, StepPerceptron},
};

use super::{validate_common, TrainingConfig, TrainingError};

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
