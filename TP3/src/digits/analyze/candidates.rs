//! Candidate model construction, validation, and parallel training.

use anyhow::{bail, Result};

use crate::{
    metrics::mean_std,
    model::{Activation, MultilayerPerceptron},
};

use super::super::{
    config::CandidateConfig,
    data::{DigitDataset, DigitSplit},
    live_dashboard::LiveMetricsPublisher,
    training::train_digit_candidate,
};
use super::{evaluate_digit_model, CandidateResult, CandidateRun, SeedRun};

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

/// Trains one repeat of `candidate` with `seed`. With several repeats the
/// live dashboard sees each one as `name@seed<seed>`.
fn run_repeat(
    candidate: &CandidateConfig,
    seed: u64,
    repeats: usize,
    dataset: &DigitDataset,
    split: &DigitSplit,
    publisher: &LiveMetricsPublisher,
) -> Result<SeedRun> {
    let mut repeat = candidate.clone();
    repeat.seed = seed;
    if repeats > 1 {
        repeat.name = format!("{}@seed{seed}", candidate.name);
    }
    eprintln!(
        "training candidate '{}' ({})",
        repeat.name,
        candidate.axis_label()
    );
    let mut model = create_model(&repeat)?;
    let report = train_digit_candidate(
        &mut model,
        &dataset.features,
        &dataset.labels,
        &split.train,
        &split.validation,
        &repeat,
        Some(publisher),
    )?;
    let validation = evaluate_digit_model(
        &model,
        &dataset.features,
        &dataset.labels,
        &split.validation,
    )?;
    eprintln!(
        "finished candidate '{}': epochs={} best_epoch={} validation_accuracy={:.4} ({:.1}s)",
        repeat.name,
        report.history.len(),
        report.best_epoch,
        validation.accuracy,
        report.seconds
    );
    Ok(SeedRun {
        seed,
        best_epoch: report.best_epoch,
        validation_loss: validation.loss,
        validation_accuracy: validation.accuracy,
        report,
    })
}

/// Summarizes the repeats of one candidate: mean validation accuracy and
/// loss, median best epoch, and the repeat with the median accuracy as the
/// representative (a typical run, not the luckiest seed).
fn summarize_repeats(mut candidate: CandidateConfig, mut repeats: Vec<SeedRun>) -> CandidateRun {
    repeats.sort_by_key(|repeat| repeat.seed);
    let accuracies = repeats
        .iter()
        .map(|repeat| repeat.validation_accuracy)
        .collect::<Vec<_>>();
    let (accuracy, accuracy_std) = mean_std(&accuracies);
    let (loss, _) = mean_std(
        &repeats
            .iter()
            .map(|repeat| repeat.validation_loss)
            .collect::<Vec<_>>(),
    );
    let mut epochs = repeats
        .iter()
        .map(|repeat| repeat.best_epoch)
        .collect::<Vec<_>>();
    epochs.sort_unstable();
    let mut order = (0..repeats.len()).collect::<Vec<_>>();
    order.sort_by(|&left, &right| accuracies[left].total_cmp(&accuracies[right]));
    let representative = &repeats[order[order.len() / 2]];
    candidate.seed = representative.seed;
    CandidateRun {
        result: CandidateResult {
            candidate,
            best_epoch: epochs[epochs.len() / 2],
            validation_loss: loss,
            validation_accuracy: accuracy,
            validation_accuracy_std: accuracy_std,
            stopped_early: representative.report.stopped_early,
        },
        report: representative.report.clone(),
        repeats,
    }
}

/// Trains every (candidate, repeat) pair in parallel. Repeat `r` uses seed
/// `candidate.seed + r`.
pub(super) fn train_candidates(
    candidates: Vec<CandidateConfig>,
    repeats: usize,
    dataset: &DigitDataset,
    split: &DigitSplit,
    publisher: &LiveMetricsPublisher,
) -> Result<Vec<CandidateRun>> {
    use rayon::prelude::*;

    let jobs = candidates
        .iter()
        .enumerate()
        .flat_map(|(index, candidate)| {
            (0..repeats as u64).map(move |repeat| (index, candidate.seed.wrapping_add(repeat)))
        })
        .collect::<Vec<_>>();
    let mut trained = jobs
        .into_par_iter()
        .map(|(index, seed)| {
            run_repeat(&candidates[index], seed, repeats, dataset, split, publisher)
                .map(|run| (index, run))
        })
        .collect::<Result<Vec<_>>>()?;
    let mut grouped = candidates
        .iter()
        .map(|_| Vec::with_capacity(repeats))
        .collect::<Vec<_>>();
    for (index, run) in trained.drain(..) {
        grouped[index].push(run);
    }
    Ok(candidates
        .into_iter()
        .zip(grouped)
        .map(|(candidate, runs)| summarize_repeats(candidate, runs))
        .collect())
}

pub(super) fn create_model(candidate: &CandidateConfig) -> Result<MultilayerPerceptron> {
    let mut activations = vec![candidate.hidden_activation; candidate.topology.len() - 2];
    activations.push(Activation::Linear);
    Ok(MultilayerPerceptron::with_initialization(
        &candidate.topology,
        &activations,
        candidate.seed,
        candidate.initialization,
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
            ..CandidateConfig::default()
        }
    }

    fn run(name: &str, params: usize, accuracy: f64, loss: f64) -> CandidateRun {
        CandidateRun {
            result: CandidateResult {
                candidate: candidate(name, params),
                best_epoch: 1,
                validation_loss: loss,
                validation_accuracy: accuracy,
                validation_accuracy_std: 0.0,
                stopped_early: false,
            },
            repeats: Vec::new(),
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
                seconds: 0.0,
            },
        }
    }

    #[test]
    fn repeats_are_averaged_and_the_median_seed_represents_them() {
        let template = run("x", 2, 0.0, 0.0);
        let repeat = |seed: u64, accuracy: f64, epoch: usize| SeedRun {
            seed,
            best_epoch: epoch,
            validation_loss: 1.0 - accuracy,
            validation_accuracy: accuracy,
            report: template.report.clone(),
        };
        let summary = summarize_repeats(
            candidate("x", 2),
            vec![repeat(9, 0.90, 7), repeat(7, 0.96, 3), repeat(8, 0.93, 5)],
        );
        assert!((summary.result.validation_accuracy - 0.93).abs() < 1e-12);
        assert!(summary.result.validation_accuracy_std > 0.0);
        assert_eq!(summary.result.best_epoch, 5);
        assert_eq!(summary.result.candidate.seed, 8);
        let seeds = summary.repeats.iter().map(|r| r.seed).collect::<Vec<_>>();
        assert_eq!(seeds, [7, 8, 9]);
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
