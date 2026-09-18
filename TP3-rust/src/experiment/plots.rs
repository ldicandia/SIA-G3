//! PNG plots for the fraud-distillation experiments.

use std::path::Path;

use anyhow::{bail, Result};
use plotters::prelude::*;

use crate::{
    data::{FraudDataset, FEATURE_NAMES},
    metrics::ThresholdMetrics,
    training::EpochMetrics,
};

use super::trials::LearningRun;

const BLUE: RGBColor = RGBColor(41, 98, 255);
const ORANGE: RGBColor = RGBColor(245, 124, 0);
const GREEN: RGBColor = RGBColor(0, 137, 123);
const RED: RGBColor = RGBColor(211, 47, 47);

pub(super) fn plot_feature_distributions(dataset: &FraudDataset, path: &Path) -> Result<()> {
    let root = BitMapBackend::new(path, (1500, 1100)).into_drawing_area();
    root.fill(&WHITE)?;
    for (column, area) in root.split_evenly((3, 3)).into_iter().enumerate() {
        let values = (0..dataset.len())
            .map(|row| dataset.features.row_unchecked(row)[column])
            .collect::<Vec<_>>();
        draw_histogram(&area, FEATURE_NAMES[column], &values, BLUE)?;
    }
    root.present()?;
    Ok(())
}

pub(super) fn plot_targets(dataset: &FraudDataset, path: &Path) -> Result<()> {
    let root = BitMapBackend::new(path, (1200, 500)).into_drawing_area();
    root.fill(&WHITE)?;
    let areas = root.split_evenly((1, 2));
    draw_histogram(
        &areas[0],
        "BigModel fraud probability",
        &dataset.teacher_targets,
        ORANGE,
    )?;
    let frauds = dataset.fraud_labels.iter().filter(|&&label| label).count() as i32;
    let legitimate = dataset.len() as i32 - frauds;
    let mut chart = ChartBuilder::on(&areas[1])
        .caption("Ground-truth class balance", ("sans-serif", 22))
        .margin(15)
        .x_label_area_size(35)
        .y_label_area_size(50)
        .build_cartesian_2d(0..2, 0..legitimate.max(frauds) + 300)?;
    chart
        .configure_mesh()
        .disable_mesh()
        .x_labels(2)
        .x_label_formatter(&|value| match *value {
            0 => "legitimate".into(),
            1 => "fraud".into(),
            _ => String::new(),
        })
        .draw()?;
    chart.draw_series([
        Rectangle::new([(0, 0), (1, legitimate)], BLUE.filled()),
        Rectangle::new([(1, 0), (2, frauds)], RED.filled()),
    ])?;
    root.present()?;
    Ok(())
}

fn draw_histogram(
    area: &DrawingArea<BitMapBackend<'_>, plotters::coord::Shift>,
    title: &str,
    values: &[f64],
    color: RGBColor,
) -> Result<()> {
    let minimum = values.iter().copied().fold(f64::INFINITY, f64::min);
    let maximum = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let bins = 30usize;
    let width = if maximum > minimum {
        (maximum - minimum) / bins as f64
    } else {
        1.0
    };
    let mut counts = vec![0i32; bins];
    for &value in values {
        let index = (((value - minimum) / width).floor() as usize).min(bins - 1);
        counts[index] += 1;
    }
    let max_count = counts.iter().copied().max().unwrap_or(1);
    let x_end = if maximum > minimum {
        maximum
    } else {
        minimum + 1.0
    };
    let mut chart = ChartBuilder::on(area)
        .caption(title, ("sans-serif", 18))
        .margin(10)
        .x_label_area_size(35)
        .y_label_area_size(45)
        .build_cartesian_2d(minimum..x_end, 0..max_count + max_count / 10 + 1)?;
    chart.configure_mesh().max_light_lines(3).draw()?;
    chart.draw_series(counts.iter().enumerate().map(|(index, &count)| {
        let start = minimum + index as f64 * width;
        Rectangle::new(
            [(start, 0), (start + width, count)],
            color.mix(0.75).filled(),
        )
    }))?;
    Ok(())
}

pub(super) fn plot_learning_history(runs: &[LearningRun], path: &Path) -> Result<()> {
    let root = BitMapBackend::new(path, (1000, 650)).into_drawing_area();
    root.fill(&WHITE)?;
    let max_epoch = runs
        .iter()
        .flat_map(|run| run.report.history.iter().map(|row| row.epoch))
        .max()
        .unwrap_or(1);
    let values = runs
        .iter()
        .flat_map(|run| run.report.history.iter().map(|row| row.train_loss.log10()))
        .collect::<Vec<_>>();
    let (min_y, max_y) = padded_range(&values);
    let mut chart = ChartBuilder::on(&root)
        .caption("Learning comparison (log10 MSE)", ("sans-serif", 28))
        .margin(20)
        .x_label_area_size(45)
        .y_label_area_size(60)
        .build_cartesian_2d(1usize..max_epoch.max(2), min_y..max_y)?;
    chart
        .configure_mesh()
        .x_desc("epoch")
        .y_desc("log10 MSE")
        .draw()?;
    for (index, run) in runs.iter().enumerate() {
        let color = if index == 0 { BLUE } else { ORANGE };
        chart
            .draw_series(LineSeries::new(
                run.report
                    .history
                    .iter()
                    .map(|row| (row.epoch, row.train_loss.log10())),
                color.stroke_width(2),
            ))?
            .label(run.name)
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

pub(super) fn plot_prediction_comparison(
    runs: &[LearningRun],
    targets: &[f64],
    path: &Path,
) -> Result<()> {
    let root = BitMapBackend::new(path, (1200, 550)).into_drawing_area();
    root.fill(&WHITE)?;
    for (run, area) in runs.iter().zip(root.split_evenly((1, 2))) {
        let min_prediction = run.predictions.iter().copied().fold(0.0, f64::min);
        let max_prediction = run.predictions.iter().copied().fold(1.0, f64::max);
        let mut chart = ChartBuilder::on(&area)
            .caption(format!("{} predictions", run.name), ("sans-serif", 22))
            .margin(15)
            .x_label_area_size(45)
            .y_label_area_size(50)
            .build_cartesian_2d(0.0..1.0, min_prediction..max_prediction)?;
        chart
            .configure_mesh()
            .x_desc("BigModel probability")
            .y_desc("TinyModel prediction")
            .draw()?;
        chart.draw_series(
            targets
                .iter()
                .zip(&run.predictions)
                .map(|(&target, &prediction)| {
                    Circle::new((target, prediction), 1, BLUE.mix(0.3).filled())
                }),
        )?;
        chart.draw_series(LineSeries::new(vec![(0.0, 0.0), (1.0, 1.0)], RED))?;
    }
    root.present()?;
    Ok(())
}

pub(super) fn plot_generalization_history(history: &[EpochMetrics], path: &Path) -> Result<()> {
    if history.is_empty() {
        bail!("cannot plot an empty history");
    }
    let root = BitMapBackend::new(path, (1000, 650)).into_drawing_area();
    root.fill(&WHITE)?;
    let values = history
        .iter()
        .flat_map(|row| [row.train_loss.log10(), row.validation_loss.unwrap().log10()])
        .collect::<Vec<_>>();
    let (min_y, max_y) = padded_range(&values);
    let mut chart = ChartBuilder::on(&root)
        .caption(
            "Generalization learning curve (log10 MSE)",
            ("sans-serif", 28),
        )
        .margin(20)
        .x_label_area_size(45)
        .y_label_area_size(60)
        .build_cartesian_2d(1usize..history.len().max(2), min_y..max_y)?;
    chart
        .configure_mesh()
        .x_desc("epoch")
        .y_desc("log10 MSE")
        .draw()?;
    for (label, color, values) in [
        (
            "train",
            BLUE,
            history
                .iter()
                .map(|row| (row.epoch, row.train_loss.log10()))
                .collect::<Vec<_>>(),
        ),
        (
            "validation",
            ORANGE,
            history
                .iter()
                .map(|row| (row.epoch, row.validation_loss.unwrap().log10()))
                .collect::<Vec<_>>(),
        ),
    ] {
        chart
            .draw_series(LineSeries::new(values, color.stroke_width(2)))?
            .label(label)
            .legend(move |(x, y)| PathElement::new([(x, y), (x + 20, y)], color));
    }
    chart.configure_series_labels().border_style(BLACK).draw()?;
    root.present()?;
    Ok(())
}

pub(super) fn plot_threshold_metrics(sweep: &[ThresholdMetrics], path: &Path) -> Result<()> {
    if sweep.is_empty() {
        bail!("cannot plot an empty threshold sweep");
    }
    let root = BitMapBackend::new(path, (1000, 650)).into_drawing_area();
    root.fill(&WHITE)?;
    let min_x = sweep.iter().map(|row| row.threshold).fold(1.0, f64::min);
    let max_x = sweep.iter().map(|row| row.threshold).fold(0.0, f64::max);
    let mut chart = ChartBuilder::on(&root)
        .caption("Validation threshold trade-off", ("sans-serif", 28))
        .margin(20)
        .x_label_area_size(45)
        .y_label_area_size(55)
        .build_cartesian_2d(min_x..max_x, 0.0..1.02)?;
    chart
        .configure_mesh()
        .x_desc("threshold")
        .y_desc("metric")
        .draw()?;
    for (label, color, values) in [
        (
            "precision",
            BLUE,
            sweep
                .iter()
                .map(|row| (row.threshold, row.precision()))
                .collect::<Vec<_>>(),
        ),
        (
            "recall",
            ORANGE,
            sweep
                .iter()
                .map(|row| (row.threshold, row.recall()))
                .collect::<Vec<_>>(),
        ),
        (
            "accuracy",
            GREEN,
            sweep
                .iter()
                .map(|row| (row.threshold, row.accuracy()))
                .collect::<Vec<_>>(),
        ),
    ] {
        chart
            .draw_series(LineSeries::new(values, color.stroke_width(2)))?
            .label(label)
            .legend(move |(x, y)| PathElement::new([(x, y), (x + 20, y)], color));
    }
    chart.configure_series_labels().border_style(BLACK).draw()?;
    root.present()?;
    Ok(())
}

fn padded_range(values: &[f64]) -> (f64, f64) {
    let minimum = values.iter().copied().fold(f64::INFINITY, f64::min);
    let maximum = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if (maximum - minimum).abs() <= f64::EPSILON {
        (minimum - 1.0, maximum + 1.0)
    } else {
        let padding = (maximum - minimum) * 0.05;
        (minimum - padding, maximum + padding)
    }
}
