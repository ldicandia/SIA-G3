//! Graficos PNG de las curvas del sweep (plotters). Ningun modulo numerico importa este.

use std::path::Path;

use anyhow::{bail, Context, Result};
use ndarray::Array1;
use plotters::prelude::*;
use plotters::style::text_anchor::{HPos, Pos, VPos};

use crate::dae_experiments::{dae_variants_of, DaeResult, DAE_STUDIES};
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

/// Lado en pixeles de cada celda de `plot_pattern_grid` y margen exterior.
const CELL: i32 = 40;
const MARGIN: i32 = 40;

/// Dibuja los items lado a lado: cada uno es una grilla de 7 filas x 5 columnas de celdas de 40 px
/// (negra si el valor es al menos 0.5, blanca con borde gris claro si no) con su titulo centrado
/// debajo. Se reutiliza en la Fase 2 para las figuras del denoising.
pub fn plot_pattern_grid(path: &Path, items: &[(String, Array1<f64>)]) -> Result<()> {
    use crate::data::{COLS, N_PIXELS, ROWS};
    if items.is_empty() {
        bail!("no hay patrones para graficar");
    }
    if let Some((t, _)) = items.iter().find(|(_, p)| p.len() != N_PIXELS) {
        bail!("el patron '{t}' no tiene {N_PIXELS} pixeles");
    }
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).with_context(|| format!("no se pudo crear {}", parent.display()))?;
    }
    let grid_w = COLS as i32 * CELL;
    let grid_h = ROWS as i32 * CELL;
    let width = MARGIN + items.len() as i32 * (grid_w + MARGIN);
    let title_h = 40;
    let height = MARGIN + grid_h + title_h + MARGIN;

    let root = BitMapBackend::new(path, (width as u32, height as u32)).into_drawing_area();
    root.fill(&WHITE).map_err(|e| anyhow::anyhow!("{e}"))?;
    let outline = RGBColor(200, 200, 200);
    let title_style = ("sans-serif", 22).into_font().color(&BLACK).pos(Pos::new(HPos::Center, VPos::Top));
    for (k, (title, pattern)) in items.iter().enumerate() {
        let x0 = MARGIN + k as i32 * (grid_w + MARGIN);
        let y0 = MARGIN;
        for r in 0..ROWS {
            for c in 0..COLS {
                let (px, py) = (x0 + c as i32 * CELL, y0 + r as i32 * CELL);
                let on = pattern[r * COLS + c] >= 0.5;
                let corners = [(px, py), (px + CELL, py + CELL)];
                if on {
                    root.draw(&Rectangle::new(corners, BLACK.filled())).map_err(|e| anyhow::anyhow!("{e}"))?;
                } else {
                    root.draw(&Rectangle::new(corners, WHITE.filled())).map_err(|e| anyhow::anyhow!("{e}"))?;
                    root.draw(&Rectangle::new(corners, outline.stroke_width(1))).map_err(|e| anyhow::anyhow!("{e}"))?;
                }
            }
        }
        root.draw(&Text::new(title.clone(), (x0 + grid_w / 2, y0 + grid_h + 8), title_style.clone()))
            .map_err(|e| anyhow::anyhow!("{e}"))?;
    }
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

/// Una fila de `plot_pattern_rows`: etiqueta a la izquierda y celdas `(pie, patron)` lado a lado.
#[derive(Debug, Clone)]
pub struct PatternRow {
    pub label: String,
    pub cells: Vec<(String, Array1<f64>)>,
}

/// Geometria de `plot_pattern_rows` (en pixeles).
const ROWS_SQUARE: i32 = 28;
const ROWS_TITLE_H: i32 = 50;
const ROWS_LABEL_W: i32 = 260;
const ROWS_CAPTION_H: i32 = 26;
const ROWS_GAP_X: i32 = 24;
const ROWS_GAP_Y: i32 = 14;
const ROWS_MARGIN: i32 = 20;

/// Dibuja un PNG con un titulo centrado arriba y una banda por fila: la etiqueta de la fila en una
/// columna de 260 px a la izquierda y sus patrones de 7 x 5 (cuadrados de 28 px, negros si el valor
/// es al menos 0.5) lado a lado, cada uno con su pie centrado debajo.
pub fn plot_pattern_rows(path: &Path, title: &str, rows: &[PatternRow]) -> Result<()> {
    use crate::data::{COLS, N_PIXELS, ROWS};
    if rows.is_empty() {
        bail!("no hay filas de patrones para graficar");
    }
    let n_cells = rows[0].cells.len();
    if n_cells == 0 {
        bail!("la fila '{}' no tiene patrones", rows[0].label);
    }
    for r in rows {
        if r.cells.is_empty() {
            bail!("la fila '{}' no tiene patrones", r.label);
        }
        if r.cells.len() != n_cells {
            bail!("la fila '{}' tiene {} patrones y se esperaban {n_cells}", r.label, r.cells.len());
        }
        if let Some((cap, _)) = r.cells.iter().find(|(_, p)| p.len() != N_PIXELS) {
            bail!("el patron '{cap}' de la fila '{}' no tiene {N_PIXELS} pixeles", r.label);
        }
    }
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).with_context(|| format!("no se pudo crear {}", parent.display()))?;
    }

    let grid_w = COLS as i32 * ROWS_SQUARE;
    let grid_h = ROWS as i32 * ROWS_SQUARE;
    let band_h = grid_h + ROWS_CAPTION_H;
    let n_rows = rows.len() as i32;
    let n = n_cells as i32;
    let width = ROWS_MARGIN + ROWS_LABEL_W + n * grid_w + (n - 1) * ROWS_GAP_X + ROWS_MARGIN;
    let height = ROWS_TITLE_H + n_rows * band_h + (n_rows - 1) * ROWS_GAP_Y + ROWS_MARGIN;

    let root = BitMapBackend::new(path, (width as u32, height as u32)).into_drawing_area();
    root.fill(&WHITE).map_err(|e| anyhow::anyhow!("{e}"))?;
    let outline = RGBColor(200, 200, 200);
    let title_style = ("sans-serif", 28).into_font().color(&BLACK).pos(Pos::new(HPos::Center, VPos::Center));
    let label_style = ("sans-serif", 20).into_font().color(&BLACK).pos(Pos::new(HPos::Left, VPos::Center));
    let caption_style = ("sans-serif", 16).into_font().color(&BLACK).pos(Pos::new(HPos::Center, VPos::Top));
    root.draw(&Text::new(title.to_string(), (width / 2, ROWS_TITLE_H / 2), title_style))
        .map_err(|e| anyhow::anyhow!("{e}"))?;

    for (ri, row) in rows.iter().enumerate() {
        let y0 = ROWS_TITLE_H + ri as i32 * (band_h + ROWS_GAP_Y);
        root.draw(&Text::new(row.label.clone(), (ROWS_MARGIN, y0 + grid_h / 2), label_style.clone()))
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        for (ci, (caption, pattern)) in row.cells.iter().enumerate() {
            let x0 = ROWS_MARGIN + ROWS_LABEL_W + ci as i32 * (grid_w + ROWS_GAP_X);
            for r in 0..ROWS {
                for c in 0..COLS {
                    let (px, py) = (x0 + c as i32 * ROWS_SQUARE, y0 + r as i32 * ROWS_SQUARE);
                    let corners = [(px, py), (px + ROWS_SQUARE, py + ROWS_SQUARE)];
                    if pattern[r * COLS + c] >= 0.5 {
                        root.draw(&Rectangle::new(corners, BLACK.filled())).map_err(|e| anyhow::anyhow!("{e}"))?;
                    } else {
                        root.draw(&Rectangle::new(corners, WHITE.filled())).map_err(|e| anyhow::anyhow!("{e}"))?;
                        root.draw(&Rectangle::new(corners, outline.stroke_width(1)))
                            .map_err(|e| anyhow::anyhow!("{e}"))?;
                    }
                }
            }
            root.draw(&Text::new(caption.clone(), (x0 + grid_w / 2, y0 + grid_h + 4), caption_style.clone()))
                .map_err(|e| anyhow::anyhow!("{e}"))?;
        }
    }
    root.present().map_err(|e| anyhow::anyhow!("{e}"))?;
    Ok(())
}

/// Curvas de perdida de entrenamiento del estudio del DAE: una imagen `dae_curves_{estudio}_loss.png`
/// por estudio con resultados, con la corrida de semilla 0 de cada variante. Devuelve cuantas
/// imagenes escribio.
pub fn write_dae_study_curves(out_dir: &Path, results: &[DaeResult]) -> Result<usize> {
    let mut written = 0;
    for study in DAE_STUDIES {
        let mut loss_series = Vec::new();
        for key in dae_variants_of(results, study) {
            let Some(r) = results
                .iter()
                .find(|r| r.config.study == study && r.config.variant_key() == key && r.config.seed == 0)
            else {
                continue;
            };
            let n = r.history.loss.len();
            let step = (n / 500).max(1);
            let pts: Vec<(f64, f64)> = r
                .history
                .loss
                .iter()
                .enumerate()
                .filter(|(i, _)| i % step == 0 || *i + 1 == n)
                .map(|(i, &l)| ((i + 1) as f64, l))
                .collect();
            loss_series.push(Series { name: key, points: pts });
        }
        if loss_series.is_empty() {
            continue;
        }
        plot_series(
            &out_dir.join(format!("dae_curves_{study}_loss.png")),
            &format!("Perdida de entrenamiento (lote ruidoso) - estudio {study}"),
            "epoca",
            "perdida",
            &loss_series,
            true,
        )?;
        written += 1;
    }
    Ok(written)
}

/// Geometria de `plot_gray_tiles` (en pixeles de pantalla).
const GRAY_TITLE_H: i64 = 44;
const GRAY_CAPTION_H: i64 = 18;
const GRAY_GAP_CAPTIONED: i64 = 8;
const GRAY_GAP_DENSE: i64 = 2;
const GRAY_MARGIN: i64 = 16;
/// Tope de ancho y alto de la imagen, para que un argumento erroneo no reserve un bitmap enorme.
const GRAY_MAX_SIDE_PX: i64 = 6000;

/// Dibuja un PNG con un titulo centrado en una banda de 44 px y debajo los `tiles` en orden fila
/// mayor, `cols` por fila (la ultima fila puede quedar incompleta). Cada tile es una imagen de
/// `side x side` valores en escala de grises ampliada `scale` veces (un cuadrado de `scale x scale`
/// por valor; nivel de gris 255 - round(255 * valor acotado a [0, 1]), asi 1.0 es tinta negra sobre
/// blanco), con un marco gris claro de 1 px y su pie centrado debajo (banda de 18 px). Si todos los
/// pies son cadenas vacias se omite la banda de pies y el hueco entre tiles baja a 2 px, de modo que
/// una grilla densa se lee como una sola imagen continua.
pub fn plot_gray_tiles(
    path: &Path,
    title: &str,
    tiles: &[(String, Array1<f64>)],
    side: usize,
    cols: usize,
    scale: usize,
) -> Result<()> {
    if tiles.is_empty() {
        bail!("no hay imagenes para graficar");
    }
    if cols == 0 || scale == 0 || side == 0 {
        bail!("el lado ({side}), las columnas ({cols}) y la escala ({scale}) deben ser positivos");
    }
    for (caption, tile) in tiles {
        if tile.len() != side * side {
            bail!("la imagen '{caption}' tiene {} valores y se esperaban {}", tile.len(), side * side);
        }
        if tile.iter().any(|v| !v.is_finite()) {
            bail!("la imagen '{caption}' tiene valores no finitos");
        }
    }
    let dense = tiles.iter().all(|(c, _)| c.is_empty());
    let gap = if dense { GRAY_GAP_DENSE } else { GRAY_GAP_CAPTIONED };
    let caption_h = if dense { 0 } else { GRAY_CAPTION_H };
    let tile_px = (side as i64).saturating_mul(scale as i64);
    let n_cols = cols.min(tiles.len()) as i64;
    let n_rows = tiles.len().div_ceil(cols) as i64;
    let width = (2 * GRAY_MARGIN).saturating_add(n_cols.saturating_mul(tile_px)).saturating_add((n_cols - 1) * gap);
    let height = GRAY_MARGIN
        .saturating_add(GRAY_TITLE_H)
        .saturating_add(n_rows.saturating_mul(tile_px + caption_h))
        .saturating_add((n_rows - 1) * gap)
        .saturating_add(GRAY_MARGIN);
    if width > GRAY_MAX_SIDE_PX || height > GRAY_MAX_SIDE_PX {
        bail!("la imagen de {width} x {height} px supera el maximo de {GRAY_MAX_SIDE_PX} px por lado");
    }
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).with_context(|| format!("no se pudo crear {}", parent.display()))?;
    }

    let root = BitMapBackend::new(path, (width as u32, height as u32)).into_drawing_area();
    root.fill(&WHITE).map_err(|e| anyhow::anyhow!("{e}"))?;
    let title_style = ("sans-serif", 18).into_font().color(&BLACK).pos(Pos::new(HPos::Center, VPos::Center));
    root.draw(&Text::new(title.to_string(), ((width / 2) as i32, (GRAY_MARGIN / 2 + GRAY_TITLE_H / 2) as i32), title_style))
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let caption_style = ("sans-serif", 12).into_font().color(&BLACK).pos(Pos::new(HPos::Center, VPos::Top));
    let frame = RGBColor(200, 200, 200);

    for (k, (caption, tile)) in tiles.iter().enumerate() {
        let (row, col) = ((k / cols) as i64, (k % cols) as i64);
        let x0 = GRAY_MARGIN + col * (tile_px + gap);
        let y0 = GRAY_MARGIN + GRAY_TITLE_H + row * (tile_px + caption_h + gap);
        for r in 0..side {
            for c in 0..side {
                let level = 255 - (255.0 * tile[r * side + c].clamp(0.0, 1.0)).round() as u8;
                let (px, py) = (x0 + c as i64 * scale as i64, y0 + r as i64 * scale as i64);
                root.draw(&Rectangle::new(
                    [(px as i32, py as i32), ((px + scale as i64) as i32, (py + scale as i64) as i32)],
                    RGBColor(level, level, level).filled(),
                ))
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            }
        }
        root.draw(&Rectangle::new(
            [(x0 as i32, y0 as i32), ((x0 + tile_px) as i32, (y0 + tile_px) as i32)],
            frame.stroke_width(1),
        ))
        .map_err(|e| anyhow::anyhow!("{e}"))?;
        if !dense {
            root.draw(&Text::new(
                caption.clone(),
                ((x0 + tile_px / 2) as i32, (y0 + tile_px + 3) as i32),
                caption_style.clone(),
            ))
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        }
    }
    root.present().map_err(|e| anyhow::anyhow!("{e}"))?;
    Ok(())
}
