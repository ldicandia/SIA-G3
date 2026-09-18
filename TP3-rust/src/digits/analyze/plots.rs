//! PNG plots for the digit-classifier study.

use std::path::Path;

use anyhow::Result;
use plotters::prelude::*;

use super::super::data::DIGIT_CLASSES;
use super::super::metrics::ClassificationMetrics;
use super::CandidateRun;

pub(super) fn plot_candidate_losses(runs: &[CandidateRun], path: &Path) -> Result<()> {
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

pub(super) fn plot_selected_loss(run: &CandidateRun, path: &Path) -> Result<()> {
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

pub(super) fn plot_confusion_matrix(metrics: &ClassificationMetrics, path: &Path) -> Result<()> {
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
