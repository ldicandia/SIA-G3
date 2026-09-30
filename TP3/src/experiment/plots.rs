//! PNG plots for the fraud-distillation experiments.

use std::path::Path;

use anyhow::{bail, Result};
use plotters::prelude::*;

use crate::{
    calibration::ReliabilityBin, data::FraudDataset, metrics::ThresholdMetrics,
    training::EpochMetrics,
};

use super::trials::LearningRun;

const BLUE: RGBColor = RGBColor(41, 98, 255);
const ORANGE: RGBColor = RGBColor(245, 124, 0);
const GREEN: RGBColor = RGBColor(0, 137, 123);
const RED: RGBColor = RGBColor(211, 47, 47);
/// A named metric drawn as one line of the threshold plot.
type MetricSeries = (&'static str, RGBColor, fn(&ThresholdMetrics) -> f64);

const PALETTE: [RGBColor; 6] = [
    BLUE,
    ORANGE,
    GREEN,
    RED,
    RGBColor(123, 31, 162),
    RGBColor(93, 64, 55),
];

pub(super) fn plot_feature_distributions(dataset: &FraudDataset, path: &Path) -> Result<()> {
    let rows = dataset.feature_names.len().div_ceil(3);
    let height = 370 * rows as u32;
    let root = BitMapBackend::new(path, (1500, height)).into_drawing_area();
    root.fill(&WHITE)?;
    for (column, area) in root
        .split_evenly((rows, 3))
        .into_iter()
        .enumerate()
        .take(dataset.feature_names.len())
    {
        let values = (0..dataset.len())
            .map(|row| dataset.features.row_unchecked(row)[column])
            .collect::<Vec<_>>();
        draw_histogram(&area, &dataset.feature_names[column], &values, BLUE)?;
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
        let color = PALETTE[index % PALETTE.len()];
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
    let root = BitMapBackend::new(path, (600 * runs.len() as u32, 550)).into_drawing_area();
    root.fill(&WHITE)?;
    for (run, area) in runs.iter().zip(root.split_evenly((1, runs.len()))) {
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

/// Train (light) and validation (solid) MSE per epoch for every CV fold.
pub(super) fn plot_cv_histories(histories: &[(usize, &[EpochMetrics])], path: &Path) -> Result<()> {
    if histories.iter().all(|(_, history)| history.is_empty()) {
        bail!("cannot plot an empty history");
    }
    let root = BitMapBackend::new(path, (1000, 650)).into_drawing_area();
    root.fill(&WHITE)?;
    let values = histories
        .iter()
        .flat_map(|(_, history)| {
            history.iter().flat_map(|row| {
                [
                    row.train_loss,
                    row.validation_loss.unwrap_or(row.train_loss),
                ]
            })
        })
        .map(f64::log10)
        .collect::<Vec<_>>();
    let max_epoch = histories
        .iter()
        .map(|(_, history)| history.len())
        .max()
        .unwrap_or(2);
    let (min_y, max_y) = padded_range(&values);
    let mut chart = ChartBuilder::on(&root)
        .caption(
            "Cross-validation learning curves (log10 MSE)",
            ("sans-serif", 28),
        )
        .margin(20)
        .x_label_area_size(45)
        .y_label_area_size(60)
        .build_cartesian_2d(1usize..max_epoch.max(2), min_y..max_y)?;
    chart
        .configure_mesh()
        .x_desc("epoch")
        .y_desc("log10 MSE")
        .draw()?;
    for (index, (fold, history)) in histories.iter().enumerate() {
        let color = PALETTE[index % PALETTE.len()];
        chart.draw_series(LineSeries::new(
            history
                .iter()
                .map(|row| (row.epoch, row.train_loss.log10())),
            color.mix(0.35).stroke_width(1),
        ))?;
        chart
            .draw_series(LineSeries::new(
                history
                    .iter()
                    .filter_map(|row| row.validation_loss.map(|loss| (row.epoch, loss.log10()))),
                color.stroke_width(2),
            ))?
            .label(format!("fold {fold} validation"))
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

pub(super) fn plot_threshold_metrics(
    sweep: &[ThresholdMetrics],
    chosen: f64,
    path: &Path,
) -> Result<()> {
    if sweep.is_empty() {
        bail!("cannot plot an empty threshold sweep");
    }
    let root = BitMapBackend::new(path, (1000, 650)).into_drawing_area();
    root.fill(&WHITE)?;
    let min_x = sweep.iter().map(|row| row.threshold).fold(1.0, f64::min);
    let max_x = sweep.iter().map(|row| row.threshold).fold(0.0, f64::max);
    let mut chart = ChartBuilder::on(&root)
        .caption(
            "Out-of-fold threshold trade-off (development rows)",
            ("sans-serif", 28),
        )
        .margin(20)
        .x_label_area_size(45)
        .y_label_area_size(55)
        .build_cartesian_2d(min_x..max_x, 0.0..1.02)?;
    chart
        .configure_mesh()
        .x_desc("threshold")
        .y_desc("metric")
        .draw()?;
    let series: [MetricSeries; 4] = [
        ("precision", BLUE, ThresholdMetrics::precision),
        ("recall", ORANGE, ThresholdMetrics::recall),
        ("f1", RED, ThresholdMetrics::f1),
        ("accuracy", GREEN, ThresholdMetrics::accuracy),
    ];
    for (label, color, metric) in series {
        chart
            .draw_series(LineSeries::new(
                sweep.iter().map(|row| (row.threshold, metric(row))),
                color.stroke_width(2),
            ))?
            .label(label)
            .legend(move |(x, y)| PathElement::new([(x, y), (x + 20, y)], color));
    }
    chart
        .draw_series(LineSeries::new(
            vec![(chosen, 0.0), (chosen, 1.02)],
            BLACK.stroke_width(1),
        ))?
        .label(format!("chosen = {chosen:.3}"))
        .legend(|(x, y)| PathElement::new([(x, y), (x + 20, y)], BLACK));
    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.85))
        .border_style(BLACK)
        .draw()?;
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

/// Reliability diagram: observed fraud rate against mean predicted
/// probability per non-empty bin, one series per model, plus the y = x
/// diagonal of perfect calibration.
pub(super) fn plot_reliability_diagram(
    series: &[(&str, &[ReliabilityBin])],
    path: &Path,
) -> Result<()> {
    if series.is_empty() {
        bail!("cannot plot an empty reliability diagram");
    }
    let root = BitMapBackend::new(path, (1000, 800)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption(
            "Reliability diagram on test (10 equal-width bins)",
            ("sans-serif", 28),
        )
        .margin(20)
        .x_label_area_size(50)
        .y_label_area_size(60)
        .build_cartesian_2d(0.0..1.0, 0.0..1.0)?;
    chart
        .configure_mesh()
        .x_desc("mean predicted probability")
        .y_desc("observed fraud rate (flagged_fraud)")
        .draw()?;
    chart
        .draw_series(DashedLineSeries::new(
            vec![(0.0, 0.0), (1.0, 1.0)],
            8,
            6,
            BLACK.stroke_width(1),
        ))?
        .label("perfect calibration")
        .legend(|(x, y)| PathElement::new([(x, y), (x + 20, y)], BLACK));
    for (index, (name, bins)) in series.iter().enumerate() {
        let color = PALETTE[index % PALETTE.len()];
        let points = bins
            .iter()
            .filter_map(|bin| Some((bin.mean_predicted?, bin.observed_rate?)))
            .collect::<Vec<_>>();
        chart
            .draw_series(LineSeries::new(points.clone(), color.stroke_width(2)))?
            .label(*name)
            .legend(move |(x, y)| PathElement::new([(x, y), (x + 20, y)], color.stroke_width(3)));
        chart.draw_series(
            points
                .into_iter()
                .map(|point| Circle::new(point, 5, color.filled())),
        )?;
    }
    chart
        .configure_series_labels()
        .position(SeriesLabelPosition::UpperLeft)
        .background_style(WHITE.mix(0.85))
        .border_style(BLACK)
        .draw()?;
    root.present()?;
    Ok(())
}

/// A labelled metric drawn as one panel of the feature-study plot.
type BarMetric = (&'static str, fn(&FeatureStudyBar) -> f64);

/// One variant of the feature study, as a change against the baseline.
pub(super) struct FeatureStudyBar {
    pub(super) label: String,
    pub(super) kind: &'static str,
    pub(super) delta_f1_pp: f64,
    pub(super) delta_ap_pp: f64,
}

/// Horizontal bars of the CV-mean F1 and AP change (percentage points)
/// of every variant against the baseline, colored by kind.
pub(super) fn plot_feature_study(bars: &[FeatureStudyBar], path: &Path) -> Result<()> {
    if bars.is_empty() {
        bail!("cannot plot an empty feature study");
    }
    let count = bars.len();
    let height = (80 + 34 * count as u32).max(400);
    let root = BitMapBackend::new(path, (1500, height)).into_drawing_area();
    root.fill(&WHITE)?;
    let (title, body) = root.split_vertically(50);
    title.titled(
        "Feature study: change vs baseline (5-fold CV mean, percentage points)",
        ("sans-serif", 26),
    )?;
    let panels = body.split_evenly((1, 2));
    // The first variant is drawn at the top.
    let flip = |index: usize| count - index - 1;
    let metrics: [BarMetric; 2] = [
        ("delta F1 (pp)", |bar| bar.delta_f1_pp),
        ("delta average precision (pp)", |bar| bar.delta_ap_pp),
    ];
    for (panel, (description, metric)) in panels.iter().zip(metrics) {
        let values = bars.iter().map(metric).collect::<Vec<_>>();
        let limit = values
            .iter()
            .fold(0.0_f64, |acc, value| acc.max(value.abs()))
            .max(0.05)
            * 1.1;
        let mut chart = ChartBuilder::on(panel)
            .margin(15)
            .x_label_area_size(45)
            .y_label_area_size(260)
            .build_cartesian_2d(-limit..limit, (0usize..count - 1).into_segmented())?;
        chart
            .configure_mesh()
            .disable_y_mesh()
            .x_desc(description)
            .y_labels(count)
            .label_style(("sans-serif", 15))
            .y_label_formatter(&|value| match value {
                SegmentValue::CenterOf(row) if *row < count => bars[flip(*row)].label.clone(),
                _ => String::new(),
            })
            .draw()?;
        for (index, (bar, value)) in bars.iter().zip(&values).enumerate() {
            let color = match bar.kind {
                "drop" => ORANGE,
                "add" => BLUE,
                _ => GREEN,
            };
            let row = flip(index);
            chart.draw_series(std::iter::once(Rectangle::new(
                [
                    (0.0, SegmentValue::Exact(row)),
                    (*value, SegmentValue::Exact(row + 1)),
                ],
                color.mix(0.8).filled(),
            )))?;
        }
        chart.draw_series(std::iter::once(PathElement::new(
            vec![
                (0.0, SegmentValue::Exact(0)),
                (0.0, SegmentValue::Exact(count)),
            ],
            BLACK.stroke_width(2),
        )))?;
    }
    root.present()?;
    Ok(())
}
