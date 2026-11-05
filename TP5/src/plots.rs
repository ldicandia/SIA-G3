//! Graficos PNG de las curvas del sweep (plotters). Ningun modulo numerico importa este.

use std::path::Path;

use anyhow::{bail, Context, Result};
use plotters::prelude::*;
use plotters::style::text_anchor::{HPos, Pos, VPos};

use crate::experiments::{variants_of, ExperimentResult, STUDIES};

const PALETTE: [RGBColor; 8] = [
    RGBColor(41, 98, 255),
    RGBColor(245, 124, 0),
    RGBColor(0, 137, 123),
    RGBColor(211, 47, 47),
    RGBColor(123, 31, 162),
    RGBColor(93, 64, 55),
    RGBColor(0, 151, 167),
    RGBColor(120, 144, 156),
];

/// Piso aplicado a la perdida en ejes logaritmicos.
const LOG_FLOOR: f64 = 1e-12;

#[derive(Debug, Clone)]
pub struct Series {
    pub name: String,
    pub points: Vec<(f64, f64)>,
}

/// Dibuja un PNG de 1000 x 600 con una linea por serie. Con `log_y` el eje y es logaritmico y los
/// valores se acotan inferiormente en 1e-12.
pub fn plot_series(
    path: &Path,
    title: &str,
    x_label: &str,
    y_label: &str,
    series: &[Series],
    log_y: bool,
) -> Result<()> {
    if series.is_empty() || series.iter().all(|s| s.points.is_empty()) {
        bail!("no hay series con puntos para graficar");
    }
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).with_context(|| format!("no se pudo crear {}", parent.display()))?;
    }

    let pts: Vec<Vec<(f64, f64)>> = series
        .iter()
        .map(|s| {
            s.points
                .iter()
                .map(|&(x, y)| (x, if log_y { y.max(LOG_FLOOR) } else { y }))
                .collect()
        })
        .collect();
    let all = || pts.iter().flatten();
    let x_min = all().map(|p| p.0).fold(f64::INFINITY, f64::min);
    let mut x_max = all().map(|p| p.0).fold(f64::NEG_INFINITY, f64::max);
    if x_max <= x_min {
        x_max = x_min + 1.0;
    }
    let y_min = all().map(|p| p.1).fold(f64::INFINITY, f64::min);
    let mut y_max = all().map(|p| p.1).fold(f64::NEG_INFINITY, f64::max);

    let root = BitMapBackend::new(path, (1000, 600)).into_drawing_area();
    root.fill(&WHITE).map_err(|e| anyhow::anyhow!("{e}"))?;

    macro_rules! draw {
        ($chart:ident) => {{
            $chart
                .configure_mesh()
                .x_desc(x_label)
                .y_desc(y_label)
                .draw()
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            for (i, (s, p)) in series.iter().zip(&pts).enumerate() {
                let color = PALETTE[i % PALETTE.len()];
                $chart
                    .draw_series(LineSeries::new(p.iter().copied(), color.stroke_width(2)))
                    .map_err(|e| anyhow::anyhow!("{e}"))?
                    .label(s.name.clone())
                    .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], color.stroke_width(2)));
            }
            $chart
                .configure_series_labels()
                .background_style(WHITE.mix(0.85))
                .border_style(BLACK)
                .position(SeriesLabelPosition::UpperRight)
                .draw()
                .map_err(|e| anyhow::anyhow!("{e}"))?;
        }};
    }

    if log_y {
        let lo = y_min.max(LOG_FLOOR);
        if y_max <= lo {
            y_max = lo * 10.0;
        }
        let mut chart = ChartBuilder::on(&root)
            .caption(title, ("sans-serif", 24))
            .margin(15)
            .x_label_area_size(40)
            .y_label_area_size(70)
            .build_cartesian_2d(x_min..x_max, (lo..y_max * 1.2).log_scale())
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        draw!(chart);
    } else {
        let lo = y_min.min(0.0);
        if y_max <= lo {
            y_max = lo + 1.0;
        }
        let mut chart = ChartBuilder::on(&root)
            .caption(title, ("sans-serif", 24))
            .margin(15)
            .x_label_area_size(40)
            .y_label_area_size(70)
            .build_cartesian_2d(x_min..x_max, lo..y_max * 1.05)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        draw!(chart);
    }
    root.present().map_err(|e| anyhow::anyhow!("{e}"))?;
    Ok(())
}

/// Rango de un eje: minimo y maximo de los datos con 8 % del rango de relleno (+-1 si el rango es 0).
fn padded_range(values: impl Iterator<Item = f64> + Clone) -> (f64, f64) {
    let lo = values.clone().fold(f64::INFINITY, f64::min);
    let hi = values.fold(f64::NEG_INFINITY, f64::max);
    let span = hi - lo;
    if span <= 0.0 {
        (lo - 1.0, hi + 1.0)
    } else {
        (lo - 0.08 * span, hi + 0.08 * span)
    }
}

/// AE-05: dibuja los puntos `(etiqueta, z1, z2)` en el plano latente (PNG de 1000 x 1000), cada
/// uno como un circulo relleno con su caracter al lado.
pub fn plot_latent_scatter(path: &Path, points: &[(String, f64, f64)]) -> Result<()> {
    if points.is_empty() {
        bail!("no hay puntos latentes para graficar");
    }
    if points.iter().any(|(_, x, y)| !x.is_finite() || !y.is_finite()) {
        bail!("hay coordenadas latentes no finitas");
    }
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).with_context(|| format!("no se pudo crear {}", parent.display()))?;
    }
    let (x0, x1) = padded_range(points.iter().map(|p| p.1));
    let (y0, y1) = padded_range(points.iter().map(|p| p.2));

    let root = BitMapBackend::new(path, (1000, 1000)).into_drawing_area();
    root.fill(&WHITE).map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut chart = ChartBuilder::on(&root)
        .caption("Espacio latente 2D - 32 patrones de font.h", ("sans-serif", 28))
        .margin(20)
        .x_label_area_size(50)
        .y_label_area_size(70)
        .build_cartesian_2d(x0..x1, y0..y1)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    chart
        .configure_mesh()
        .x_desc("z1")
        .y_desc("z2")
        .draw()
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    chart
        .draw_series(points.iter().map(|(_, x, y)| Circle::new((*x, *y), 5, PALETTE[0].filled())))
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let label_style = ("sans-serif", 20)
        .into_font()
        .color(&RGBColor(180, 30, 30))
        .pos(Pos::new(HPos::Left, VPos::Bottom));
    chart
        .draw_series(points.iter().map(|(l, x, y)| Text::new(l.clone(), (*x, *y), label_style.clone())))
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    root.present().map_err(|e| anyhow::anyhow!("{e}"))?;
    Ok(())
}

/// Escribe `curves_{estudio}_loss.png` y `curves_{estudio}_wrongpx.png` para cada estudio con
/// resultados, usando la corrida de semilla 0 de cada variante. Devuelve cuantas imagenes escribio.
pub fn write_study_curves(out_dir: &Path, results: &[ExperimentResult]) -> Result<usize> {
    let mut written = 0;
    for study in STUDIES {
        let mut loss_series = Vec::new();
        let mut wrong_series = Vec::new();
        for key in variants_of(results, study) {
            let Some(r) = results.iter().find(|r| r.config.study == study && r.config.variant_key() == key && r.config.seed == 0)
            else {
                continue;
            };
            let n = r.history.loss.len();
            let step = (n / 500).max(1);
            let loss_pts: Vec<(f64, f64)> = r
                .history
                .loss
                .iter()
                .enumerate()
                .filter(|(i, _)| i % step == 0 || *i + 1 == n)
                .map(|(i, &l)| ((i + 1) as f64, l))
                .collect();
            loss_series.push(Series { name: key.clone(), points: loss_pts });
            let wrong_pts: Vec<(f64, f64)> =
                r.history.evals.iter().map(|e| (e.epoch as f64, e.max_wrong as f64)).collect();
            wrong_series.push(Series { name: key, points: wrong_pts });
        }
        if loss_series.is_empty() {
            continue;
        }
        plot_series(
            &out_dir.join(format!("curves_{study}_loss.png")),
            &format!("Perdida de entrenamiento - estudio {study}"),
            "epoca",
            "perdida",
            &loss_series,
            true,
        )?;
        plot_series(
            &out_dir.join(format!("curves_{study}_wrongpx.png")),
            &format!("Pixeles incorrectos - estudio {study}"),
            "epoca",
            "maximo de pixeles incorrectos",
            &wrong_series,
            false,
        )?;
        written += 2;
    }
    Ok(written)
}
