//! Candidate model construction, validation, and parallel training.

use anyhow::{bail, Result};

use crate::model::{Activation, MultilayerPerceptron};

use super::super::{
    config::CandidateConfig,
    data::{DigitDataset, DigitSplit},
    live_dashboard::LiveMetricsPublisher,
    training::train_digit_candidate,
};
use super::{evaluate_digit_model, CandidateResult, CandidateRun};

pub(super) fn validate_candidates(candidates: &[CandidateConfig], input_size: usize) -> Result<()> {
    if candidates.is_empty() {
        bail!("at least one candidate is required");
    }
    let mut names = std::collections::HashSet::new();
    for candidate in candidates {
        candidate.validate()?;
        if candidate.topology.first() != Some(&input_size) {
            bail!(
                "candidate '{}' expects {} inputs, but the dataset has {}",
                candidate.name,
                candidate.topology.first().copied().unwrap_or_default(),
                input_size
            );
        }
        if !names.insert(&candidate.name) {
            bail!("duplicate candidate name '{}'", candidate.name);
        }
    }
    Ok(())
}

fn run_candidate(
    candidate: CandidateConfig,
    dataset: &DigitDataset,
    split: &DigitSplit,
    publisher: &LiveMetricsPublisher,
) -> Result<CandidateRun> {
    eprintln!("training candidate '{}'", candidate.name);
    let mut model = create_model(&candidate)?;
    let report = train_digit_candidate(
        &mut model,
        &dataset.features,
        &dataset.labels,
        &split.train,
        &split.validation,
        &candidate,
        Some(publisher),
    )?;
    let validation = evaluate_digit_model(
        &model,
        &dataset.features,
        &dataset.labels,
        &split.validation,
    )?;
    Ok(CandidateRun {
        result: CandidateResult {
            candidate,
            best_epoch: report.best_epoch,
            validation_loss: validation.loss,
            validation_accuracy: validation.accuracy,
            stopped_early: report.stopped_early,
        },
        report,
    })
}

pub(super) fn train_candidates(
    candidates: Vec<CandidateConfig>,
    dataset: &DigitDataset,
    split: &DigitSplit,
    publisher: &LiveMetricsPublisher,
) -> Result<Vec<CandidateRun>> {
    use rayon::prelude::*;

    candidates
        .into_par_iter()
        .map(|candidate| run_candidate(candidate, dataset, split, publisher))
        .collect()
}

pub(super) fn create_model(candidate: &CandidateConfig) -> Result<MultilayerPerceptron> {
    let mut activations = vec![candidate.hidden_activation; candidate.topology.len() - 2];
    activations.push(Activation::Linear);
    Ok(MultilayerPerceptron::new(
        &candidate.topology,
        &activations,
        candidate.seed,
    )?)
}

pub(super) fn select_best_index(runs: &[CandidateRun]) -> Option<usize> {
    runs.iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| {
            left.result
                .validation_accuracy
                .total_cmp(&right.result.validation_accuracy)
                .then_with(|| {
                    right
                        .result
                        .validation_loss
                        .total_cmp(&left.result.validation_loss)
                })
                .then_with(|| {
                    right
                        .result
                        .candidate
                        .parameter_count()
                        .cmp(&left.result.candidate.parameter_count())
                })
                .then_with(|| right.result.candidate.name.cmp(&left.result.candidate.name))
        })
        .map(|(index, _)| index)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::digits::{config::OptimizerKind, training::DigitTrainingReport};

    fn candidate(name: &str, params: usize) -> CandidateConfig {
        CandidateConfig {
            name: name.into(),
            topology: vec![2, params, 10],
            hidden_activation: Activation::Relu,
            optimizer: OptimizerKind::Sgd,
            learning_rate: 0.1,
            batch_size: 1,
            max_epochs: 1,
            patience: 1,
            min_delta: 0.0,
            seed: 1,
            momentum: 0.9,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
        }
    }

    fn run(name: &str, params: usize, accuracy: f64, loss: f64) -> CandidateRun {
        CandidateRun {
            result: CandidateResult {
                candidate: candidate(name, params),
                best_epoch: 1,
                validation_loss: loss,
                validation_accuracy: accuracy,
                stopped_early: false,
            },
            report: DigitTrainingReport {
                history: vec![crate::digits::training::DigitEpochMetrics {
                    epoch: 1,
                    train_loss: loss,
                    validation_loss: Some(loss),
                    train_accuracy: accuracy,
                    validation_accuracy: Some(accuracy),
                }],
                best_epoch: 1,
                best_validation_loss: loss,
                best_validation_accuracy: accuracy,
                stopped_early: false,
            },
        }
    }

    #[test]
    fn selection_prefers_accuracy_then_loss_then_smaller_model() {
        let runs = vec![
            run("low-accuracy", 2, 0.8, 0.1),
            run("high-loss", 2, 0.9, 0.3),
            run("large", 5, 0.9, 0.2),
            run("winner", 3, 0.9, 0.2),
        ];
        assert_eq!(select_best_index(&runs), Some(3));
    }
}
