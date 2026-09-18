use std::{fs, path::Path};

use anyhow::{bail, Context, Result};
use plotters::prelude::*;

use crate::model::{Activation, MultilayerPerceptron};

use super::{
    artifact::DigitModelArtifact,
    config::{CandidateConfig, DigitStudyConfig},
    data::{class_counts, stratified_digit_split, DigitDataset, DIGIT_CLASSES, IMAGE_PIXELS},
    metrics::{classification_metrics, ClassificationMetrics},
    training::{
        evaluate_digit_model, refit_digit_model, train_digit_candidate, DigitTrainingReport,
    },
};

#[derive(Clone, Debug)]
pub struct CandidateResult {
    pub candidate: CandidateConfig,
    pub best_epoch: usize,
    pub validation_loss: f64,
    pub validation_accuracy: f64,
    pub stopped_early: bool,
}

pub struct TrainingOutcome {
    pub artifact: DigitModelArtifact,
    pub candidates: Vec<CandidateResult>,
}

struct CandidateRun {
    result: CandidateResult,
    report: DigitTrainingReport,
}

pub fn run_digit_training(
    exercise: &str,
    dataset: &DigitDataset,
    config: &DigitStudyConfig,
    additional_candidates: &[CandidateConfig],
    output: &Path,
) -> Result<TrainingOutcome> {
    fs::create_dir_all(output)?;
    let mut candidates = additional_candidates.to_vec();
    candidates.extend(config.candidates.clone());
    validate_candidates(&candidates, dataset.features.cols())?;

    let split = stratified_digit_split(&dataset.labels, config.validation_ratio, config.split_seed);
    write_dataset_summary(dataset, &split.train, &split.validation, output)?;

    let mut runs = Vec::with_capacity(candidates.len());
    for candidate in candidates {
        eprintln!("training candidate '{}'", candidate.name);
        let mut model = create_model(&candidate)?;
        let report = train_digit_candidate(
            &mut model,
            &dataset.features,
            &dataset.labels,
            &split.train,
            &split.validation,
            &candidate,
        )?;
        let validation = evaluate_digit_model(
            &model,
            &dataset.features,
            &dataset.labels,
            &split.validation,
        )?;
        runs.push(CandidateRun {
            result: CandidateResult {
                candidate,
                best_epoch: report.best_epoch,
                validation_loss: validation.loss,
                validation_accuracy: validation.accuracy,
                stopped_early: report.stopped_early,
            },
            report,
        });
    }

    let best_index = select_best_index(&runs).context("candidate list cannot be empty")?;
    write_candidate_summary(&runs, output)?;
    write_learning_history(&runs, output)?;
    plot_candidate_losses(&runs, &output.join("candidate_loss_curves.png"))?;
    plot_selected_loss(
        &runs[best_index],
        &output.join("selected_learning_curve.png"),
    )?;

    let best = &runs[best_index];
    let all_indices = (0..dataset.features.rows()).collect::<Vec<_>>();
    let mut final_model = create_model(&best.result.candidate)?;
    refit_digit_model(
        &mut final_model,
        &dataset.features,
        &dataset.labels,
        &all_indices,
        &best.result.candidate,
        best.result.best_epoch,
    )?;
    let artifact = DigitModelArtifact {
        format_version: 1,
        exercise: exercise.to_owned(),
        candidate: best.result.candidate.clone(),
        selected_epoch: best.result.best_epoch,
        selection_validation_loss: best.result.validation_loss,
        selection_validation_accuracy: best.result.validation_accuracy,
        model: final_model,
    };
    artifact.save(&output.join("selected_model.toml"))?;
    write_selected_model(&artifact, output)?;
    eprintln!(
        "selected candidate '{}' (validation_accuracy={:.6}, validation_loss={:.6}, epoch={})",
        artifact.candidate.name,
        artifact.selection_validation_accuracy,
        artifact.selection_validation_loss,
        artifact.selected_epoch
    );

    Ok(TrainingOutcome {
        artifact,
        candidates: runs.into_iter().map(|run| run.result).collect(),
    })
}

pub fn run_digit_evaluation(
    dataset: &DigitDataset,
    artifact: &DigitModelArtifact,
    output: &Path,
) -> Result<ClassificationMetrics> {
    fs::create_dir_all(output)?;
    if artifact.model.topology().first() != Some(&dataset.features.cols()) {
        bail!("model input size does not match evaluation dataset");
    }
    let indices = (0..dataset.features.rows()).collect::<Vec<_>>();
    let evaluation = evaluate_digit_model(
        &artifact.model,
        &dataset.features,
        &dataset.labels,
        &indices,
    )?;
    let metrics = classification_metrics(&evaluation.predictions, &dataset.labels)?;
    write_evaluation_metrics(artifact, evaluation.loss, &metrics, output)?;
    write_predictions(
        &dataset.labels,
        &evaluation.predictions,
        &output.join("predictions.csv"),
    )?;
    write_confusion_matrix(&metrics, &output.join("confusion_matrix.csv"))?;
    plot_confusion_matrix(&metrics, &output.join("confusion_matrix.png"))?;
    eprintln!(
        "evaluation accuracy={:.6} loss={:.6} samples={}",
        metrics.accuracy,
        evaluation.loss,
        dataset.labels.len()
    );
    Ok(metrics)
}

fn validate_candidates(candidates: &[CandidateConfig], input_size: usize) -> Result<()> {
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

fn create_model(candidate: &CandidateConfig) -> Result<MultilayerPerceptron> {
    let mut activations = vec![candidate.hidden_activation; candidate.topology.len() - 2];
    activations.push(Activation::Linear);
    Ok(MultilayerPerceptron::new(
        &candidate.topology,
        &activations,
        candidate.seed,
    )?)
}

fn select_best_index(runs: &[CandidateRun]) -> Option<usize> {
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

fn write_dataset_summary(
    dataset: &DigitDataset,
    train: &[usize],
    validation: &[usize],
    output: &Path,
) -> Result<()> {
    let all = (0..dataset.labels.len()).collect::<Vec<_>>();
    let all_counts = class_counts(&dataset.labels, &all);
    let train_counts = class_counts(&dataset.labels, train);
    let validation_counts = class_counts(&dataset.labels, validation);
    let mut writer = csv::Writer::from_path(output.join("dataset_summary.csv"))?;
    writer.write_record(["class", "all", "train", "validation"])?;
    for class in 0..DIGIT_CLASSES {
        writer.serialize((
            class,
            all_counts[class],
            train_counts[class],
            validation_counts[class],
        ))?;
    }
    writer.flush()?;
    let mut profile = csv::Writer::from_path(output.join("dataset_profile.csv"))?;
    profile.write_record(["metric", "value"])?;
    profile.serialize(("rows", dataset.features.rows() as f64))?;
    profile.serialize(("pixels_per_image", dataset.features.cols() as f64))?;
    profile.serialize((
        "minimum_pixel",
        dataset
            .features
            .as_slice()
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min),
    ))?;
    profile.serialize((
        "maximum_pixel",
        dataset
            .features
            .as_slice()
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max),
    ))?;
    profile.serialize(("missing_or_non_finite_values", 0.0))?;
    profile.flush()?;
    Ok(())
}

fn write_candidate_summary(runs: &[CandidateRun], output: &Path) -> Result<()> {
    let mut writer = csv::Writer::from_path(output.join("candidate_summary.csv"))?;
    writer.write_record([
        "name",
        "topology",
        "hidden_activation",
        "optimizer",
        "learning_rate",
        "batch_size",
        "parameter_count",
        "best_epoch",
        "validation_loss",
        "validation_accuracy",
        "stopped_early",
    ])?;
    for run in runs {
        let candidate = &run.result.candidate;
        writer.serialize((
            &candidate.name,
            topology_text(&candidate.topology),
            format!("{:?}", candidate.hidden_activation).to_lowercase(),
            format!("{:?}", candidate.optimizer).to_lowercase(),
            candidate.learning_rate,
            candidate.batch_size,
            candidate.parameter_count(),
            run.result.best_epoch,
            run.result.validation_loss,
            run.result.validation_accuracy,
            run.result.stopped_early,
        ))?;
    }
    writer.flush()?;
    Ok(())
}

fn write_learning_history(runs: &[CandidateRun], output: &Path) -> Result<()> {
    let mut writer = csv::Writer::from_path(output.join("learning_history.csv"))?;
    writer.write_record([
        "candidate",
        "epoch",
        "train_loss",
        "validation_loss",
        "train_accuracy",
        "validation_accuracy",
    ])?;
    for run in runs {
        for row in &run.report.history {
            writer.serialize((
                &run.result.candidate.name,
                row.epoch,
                row.train_loss,
                row.validation_loss,
                row.train_accuracy,
                row.validation_accuracy,
            ))?;
        }
    }
    writer.flush()?;
    Ok(())
}

fn write_selected_model(artifact: &DigitModelArtifact, output: &Path) -> Result<()> {
    let mut writer = csv::Writer::from_path(output.join("selected_model.csv"))?;
    writer.write_record(["field", "value"])?;
    for (field, value) in [
        ("candidate", artifact.candidate.name.clone()),
        ("topology", topology_text(&artifact.candidate.topology)),
        (
            "hidden_activation",
            format!("{:?}", artifact.candidate.hidden_activation).to_lowercase(),
        ),
        (
            "optimizer",
            format!("{:?}", artifact.candidate.optimizer).to_lowercase(),
        ),
        (
            "learning_rate",
            artifact.candidate.learning_rate.to_string(),
        ),
        ("batch_size", artifact.candidate.batch_size.to_string()),
        ("selected_epoch", artifact.selected_epoch.to_string()),
        (
            "validation_loss",
            artifact.selection_validation_loss.to_string(),
        ),
        (
            "validation_accuracy",
            artifact.selection_validation_accuracy.to_string(),
        ),
    ] {
        writer.write_record([field, &value])?;
    }
    writer.flush()?;
    Ok(())
}

fn write_evaluation_metrics(
    artifact: &DigitModelArtifact,
    loss: f64,
    metrics: &ClassificationMetrics,
    output: &Path,
) -> Result<()> {
    let mut writer = csv::Writer::from_path(output.join("evaluation_metrics.csv"))?;
    writer.write_record(["metric", "value"])?;
    writer.serialize(("loss", loss))?;
    writer.serialize(("accuracy", metrics.accuracy))?;
    if artifact.exercise == "exercise3" {
        writer.serialize(("meets_98_percent_target", metrics.accuracy >= 0.98))?;
    }
    writer.serialize(("selected_epoch", artifact.selected_epoch as f64))?;
    for class in 0..DIGIT_CLASSES {
        if let Some(recall) = metrics.per_class_recall[class] {
            writer.serialize((format!("class_{class}_recall"), recall))?;
        } else {
            writer.write_record([format!("class_{class}_recall"), "undefined".into()])?;
        }
    }
    writer.flush()?;
    Ok(())
}

fn write_predictions(labels: &[usize], predictions: &[usize], path: &Path) -> Result<()> {
    let mut writer = csv::Writer::from_path(path)?;
    writer.write_record(["row", "label", "prediction", "correct"])?;
    for (row, (&label, &prediction)) in labels.iter().zip(predictions).enumerate() {
        writer.serialize((row, label, prediction, label == prediction))?;
    }
    writer.flush()?;
    Ok(())
}

fn write_confusion_matrix(metrics: &ClassificationMetrics, path: &Path) -> Result<()> {
    let mut writer = csv::Writer::from_path(path)?;
    let mut header = vec!["actual\\predicted".to_owned()];
    header.extend((0..DIGIT_CLASSES).map(|class| class.to_string()));
    writer.write_record(header)?;
    for actual in 0..DIGIT_CLASSES {
        let mut row = vec![actual.to_string()];
        row.extend(metrics.confusion[actual].iter().map(ToString::to_string));
        writer.write_record(row)?;
    }
    writer.flush()?;
    Ok(())
}

fn plot_candidate_losses(runs: &[CandidateRun], path: &Path) -> Result<()> {
    let max_epoch = runs
        .iter()
        .flat_map(|run| run.report.history.iter().map(|row| row.epoch))
        .max()
        .unwrap_or(1);
    let max_loss = runs
        .iter()
        .flat_map(|run| {
            run.report
                .history
                .iter()
                .filter_map(|row| row.validation_loss)
        })
        .fold(0.0_f64, f64::max)
        .max(1e-6);
    let root = BitMapBackend::new(path, (1280, 720)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption("Validation loss by candidate", ("sans-serif", 30))
        .margin(20)
        .x_label_area_size(45)
        .y_label_area_size(70)
        .build_cartesian_2d(1usize..max_epoch.max(2), 0.0..max_loss * 1.05)?;
    chart
        .configure_mesh()
        .x_desc("Epoch")
        .y_desc("Cross-entropy loss")
        .draw()?;
    for (index, run) in runs.iter().enumerate() {
        let color = Palette99::pick(index).mix(0.9);
        chart
            .draw_series(LineSeries::new(
                run.report
                    .history
                    .iter()
                    .filter_map(|row| row.validation_loss.map(|loss| (row.epoch, loss))),
                color,
            ))?
            .label(run.result.candidate.name.clone())
            .legend(move |(x, y)| PathElement::new([(x, y), (x + 20, y)], color));
    }
    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.85))
        .border_style(BLACK)
        .draw()?;
    root.present()?;
    Ok(())
}

fn plot_selected_loss(run: &CandidateRun, path: &Path) -> Result<()> {
    let max_epoch = run.report.history.last().map_or(1, |row| row.epoch);
    let max_loss = run
        .report
        .history
        .iter()
        .flat_map(|row| [Some(row.train_loss), row.validation_loss])
        .flatten()
        .fold(0.0_f64, f64::max)
        .max(1e-6);
    let root = BitMapBackend::new(path, (1100, 680)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption(
            format!("Selected candidate: {}", run.result.candidate.name),
            ("sans-serif", 30),
        )
        .margin(20)
        .x_label_area_size(45)
        .y_label_area_size(70)
        .build_cartesian_2d(1usize..max_epoch.max(2), 0.0..max_loss * 1.05)?;
    chart
        .configure_mesh()
        .x_desc("Epoch")
        .y_desc("Cross-entropy loss")
        .draw()?;
    chart
        .draw_series(LineSeries::new(
            run.report
                .history
                .iter()
                .map(|row| (row.epoch, row.train_loss)),
            BLUE,
        ))?
        .label("train")
        .legend(|(x, y)| PathElement::new([(x, y), (x + 20, y)], BLUE));
    chart
        .draw_series(LineSeries::new(
            run.report
                .history
                .iter()
                .filter_map(|row| row.validation_loss.map(|loss| (row.epoch, loss))),
            RED,
        ))?
        .label("validation")
        .legend(|(x, y)| PathElement::new([(x, y), (x + 20, y)], RED));
    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.85))
        .border_style(BLACK)
        .draw()?;
    root.present()?;
    Ok(())
}

fn plot_confusion_matrix(metrics: &ClassificationMetrics, path: &Path) -> Result<()> {
    let root = BitMapBackend::new(path, (850, 850)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption("Confusion matrix", ("sans-serif", 30))
        .margin(30)
        .x_label_area_size(50)
        .y_label_area_size(50)
        .build_cartesian_2d(0usize..DIGIT_CLASSES, 0usize..DIGIT_CLASSES)?;
    chart
        .configure_mesh()
        .x_desc("Predicted")
        .y_desc("Actual")
        .x_label_formatter(&|value| {
            if *value < DIGIT_CLASSES {
                value.to_string()
            } else {
                String::new()
            }
        })
        .y_label_formatter(&|value| {
            if *value < DIGIT_CLASSES {
                (DIGIT_CLASSES - value - 1).to_string()
            } else {
                String::new()
            }
        })
        .disable_mesh()
        .draw()?;
    let maximum = metrics
        .confusion
        .iter()
        .flatten()
        .copied()
        .max()
        .unwrap_or(1)
        .max(1) as f64;
    for actual in 0..DIGIT_CLASSES {
        for predicted in 0..DIGIT_CLASSES {
            let count = metrics.confusion[actual][predicted];
            let intensity = count as f64 / maximum;
            let color = RGBColor(
                (245.0 * (1.0 - intensity)) as u8,
                (248.0 * (1.0 - intensity)) as u8,
                255,
            );
            chart.draw_series(std::iter::once(Rectangle::new(
                [
                    (predicted, DIGIT_CLASSES - actual - 1),
                    (predicted + 1, DIGIT_CLASSES - actual),
                ],
                color.filled(),
            )))?;
        }
    }
    root.present()?;
    Ok(())
}

fn topology_text(topology: &[usize]) -> String {
    topology
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join("-")
}

pub fn expected_image_pixels() -> usize {
    IMAGE_PIXELS
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::digits::config::OptimizerKind;

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
