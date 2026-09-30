//! PNG plots for the digit-classifier study.

use std::path::Path;

use anyhow::Result;
use plotters::prelude::*;
use plotters::style::text_anchor::{HPos, Pos, VPos};

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

/// Heatmap of the confusion matrix: rows are the true digit, columns the
/// prediction. Each cell shows its count and is shaded by its share of the
/// row, so the diagonal reads as per-class recall regardless of class size.
pub(super) fn plot_confusion_matrix(metrics: &ClassificationMetrics, path: &Path) -> Result<()> {
    let root = BitMapBackend::new(path, (900, 900)).into_drawing_area();
    root.fill(&WHITE)?;
    let caption = format!(
        "Confusion matrix (accuracy {:.2}%)",
        metrics.accuracy * 100.0
    );
    // Actual digit 0 is drawn at the top, so the y coordinate is flipped.
    let flip = |actual: usize| DIGIT_CLASSES - actual - 1;
    let mut chart = ChartBuilder::on(&root)
        .caption(caption, ("sans-serif", 30))
        .margin(30)
        .x_label_area_size(60)
        .y_label_area_size(130)
        // A segmented `a..b` range holds one segment per value in `a..=b`.
        .build_cartesian_2d(
            (0usize..DIGIT_CLASSES - 1).into_segmented(),
            (0usize..DIGIT_CLASSES - 1).into_segmented(),
        )?;
    chart
        .configure_mesh()
        .disable_mesh()
        .x_desc("Predicted digit")
        .y_desc("Actual digit (recall)")
        .x_labels(DIGIT_CLASSES)
        .y_labels(DIGIT_CLASSES)
        .label_style(("sans-serif", 18))
        .axis_desc_style(("sans-serif", 20))
        .x_label_formatter(&|value| match value {
            SegmentValue::CenterOf(digit) if *digit < DIGIT_CLASSES => digit.to_string(),
            _ => String::new(),
        })
        .y_label_formatter(&|value| match value {
            SegmentValue::CenterOf(row) if *row < DIGIT_CLASSES => {
                let actual = flip(*row);
                match metrics.per_class_recall[actual] {
                    Some(recall) => format!("{actual} ({:.1}%)", recall * 100.0),
                    None => format!("{actual} (n/a)"),
                }
            }
            _ => String::new(),
        })
        .draw()?;
    for actual in 0..DIGIT_CLASSES {
        let support = metrics.confusion[actual].iter().sum::<usize>().max(1) as f64;
        for predicted in 0..DIGIT_CLASSES {
            let count = metrics.confusion[actual][predicted];
            let share = count as f64 / support;
            let color = RGBColor(
                (245.0 * (1.0 - share)) as u8,
                (248.0 * (1.0 - share)) as u8,
                255,
            );
            let (x, y) = (predicted, flip(actual));
            chart.draw_series(std::iter::once(Rectangle::new(
                [
                    (SegmentValue::Exact(x), SegmentValue::Exact(y)),
                    (SegmentValue::Exact(x + 1), SegmentValue::Exact(y + 1)),
                ],
                color.filled(),
            )))?;
            if count > 0 {
                let text_color = if share > 0.5 { WHITE } else { BLACK };
                let style = ("sans-serif", 18)
                    .into_font()
                    .color(&text_color)
                    .pos(Pos::new(HPos::Center, VPos::Center));
                chart.draw_series(std::iter::once(Text::new(
                    count.to_string(),
                    (SegmentValue::CenterOf(x), SegmentValue::CenterOf(y)),
                    style,
                )))?;
            }
        }
    }
    root.present()?;
    Ok(())
}

/// Horizontal bars of validation accuracy, grouped by study axis.
pub(super) fn plot_validation_accuracy(runs: &[CandidateRun], path: &Path) -> Result<()> {
    let mut order = (0..runs.len()).collect::<Vec<_>>();
    order.sort_by(|&left, &right| {
        let (left, right) = (&runs[left].result, &runs[right].result);
        left.candidate
            .axis_label()
            .cmp(right.candidate.axis_label())
            .then(
                left.validation_accuracy
                    .total_cmp(&right.validation_accuracy),
            )
    });
    let labels = order
        .iter()
        .map(|&index| {
            let candidate = &runs[index].result.candidate;
            format!("[{}] {}", candidate.axis_label(), candidate.name)
        })
        .collect::<Vec<_>>();
    let minimum = runs
        .iter()
        .map(|run| run.result.validation_accuracy)
        .fold(1.0_f64, f64::min);
    let lower = ((minimum - 0.01) * 100.0).floor() / 100.0;
    let height = (140 + 28 * runs.len()) as u32;
    let root = BitMapBackend::new(path, (1400, height)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption("Validation accuracy by candidate", ("sans-serif", 30))
        .margin(20)
        .x_label_area_size(45)
        .y_label_area_size(430)
        .build_cartesian_2d(lower..1.0, 0.0..runs.len() as f64)?;
    let label_text = labels.clone();
    chart
        .configure_mesh()
        .disable_y_mesh()
        .y_labels(runs.len() * 2)
        .y_label_formatter(&move |value| {
            let index = (*value - 0.5).round();
            if (value - 0.5 - index).abs() < 1e-6 && index >= 0.0 {
                label_text.get(index as usize).cloned().unwrap_or_default()
            } else {
                String::new()
            }
        })
        .x_desc("Validation accuracy")
        .draw()?;
    let axes = order
        .iter()
        .map(|&index| runs[index].result.candidate.axis_label().to_owned())
        .collect::<Vec<_>>();
    let mut distinct = axes.clone();
    distinct.dedup();
    for (position, &index) in order.iter().enumerate() {
        let accuracy = runs[index].result.validation_accuracy;
        let color = Palette99::pick(
            distinct
                .iter()
                .position(|axis| *axis == axes[position])
                .unwrap_or(0),
        );
        let y = position as f64;
        chart.draw_series([Rectangle::new(
            [(lower, y + 0.15), (accuracy, y + 0.85)],
            color.filled(),
        )])?;
        chart.draw_series([Text::new(
            format!("{:.2}%", accuracy * 100.0),
            (accuracy + 0.001, y + 0.3),
            ("sans-serif", 14),
        )])?;
    }
    root.present()?;
    Ok(())
}

/// One validation-loss chart per study axis, so each comparison stays legible.
pub(super) fn plot_axis_losses(runs: &[CandidateRun], output: &Path) -> Result<()> {
    let mut axes = runs
        .iter()
        .map(|run| run.result.candidate.axis_label().to_owned())
        .collect::<Vec<_>>();
    axes.sort();
    axes.dedup();
    for axis in axes {
        let subset = runs
            .iter()
            .filter(|run| run.result.candidate.axis_label() == axis)
            .cloned()
            .collect::<Vec<_>>();
        plot_candidate_losses(&subset, &output.join(format!("loss_curves_{axis}.png")))?;
    }
    Ok(())
}
