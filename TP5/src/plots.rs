//! Graficos PNG de las curvas del sweep (plotters). Ningun modulo numerico importa este.
//!
//! Legibilidad: toda figura se piensa proyectada sola en el area de contenido de una diapositiva
//! (`SLIDE_CONTENT_W_IN` x `SLIDE_CONTENT_H_IN`). A esa escala ningun texto queda por debajo de
//! `MIN_FIGURE_TEXT_PT`; si no se puede, la funcion devuelve un error en vez de achicar el texto.

use std::path::Path;

use anyhow::{bail, Context, Result};
use ndarray::Array1;
use plotters::coord::Shift;
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

/// Area de contenido de una diapositiva (presentacion/pptx_minimo.py): ancho menos dos margenes,
/// alto desde el inicio del contenido hasta el margen inferior. tests/plots.rs las compara.
pub const SLIDE_CONTENT_W_IN: f64 = 12.333;
pub const SLIDE_CONTENT_H_IN: f64 = 5.35;
/// Minimo de cualquier texto de una figura, en puntos, con la figura sola en la diapositiva.
pub const MIN_FIGURE_TEXT_PT: f64 = 16.0;

const FONT: &str = "sans-serif";
const FRAME: RGBColor = RGBColor(150, 150, 150);
/// Tope de ancho y alto de una imagen, para que un argumento erroneo no reserve un bitmap enorme.
const MAX_SIDE_PX: i64 = 6000;
const MAX_FONT_PX: u32 = 400;

type Area<'a> = DrawingArea<BitMapBackend<'a>, Shift>;

fn de<E: std::fmt::Display>(e: E) -> anyhow::Error {
    anyhow::anyhow!("{e}")
}

/// Tamano en puntos de un texto de `font_px` pixeles cuando la figura de `width_px` x `height_px`
/// ocupa sola el area de contenido de la diapositiva conservando su proporcion.
pub fn effective_pt(font_px: u32, width_px: u32, height_px: u32) -> f64 {
    let scale = (SLIDE_CONTENT_W_IN / width_px as f64).min(SLIDE_CONTENT_H_IN / height_px as f64);
    font_px as f64 * scale * 72.0
}

/// Menor tamano de letra en pixeles que llega a `MIN_FIGURE_TEXT_PT` en una figura de ese tamano.
pub fn min_font_px(width_px: u32, height_px: u32) -> u32 {
    (MIN_FIGURE_TEXT_PT / effective_pt(1, width_px, height_px) - 1e-9).ceil().max(1.0) as u32
}

/// Error si un texto de `font_px` queda por debajo de `MIN_FIGURE_TEXT_PT` en la diapositiva.
pub fn check_legible(what: &str, font_px: u32, width_px: u32, height_px: u32) -> Result<()> {
    let pt = effective_pt(font_px, width_px, height_px);
    if pt + 1e-9 < MIN_FIGURE_TEXT_PT {
        bail!(
            "{what}: el texto de {font_px} px queda en {pt:.1} pt en una figura de {width_px} x {height_px} px \
             y el minimo es {MIN_FIGURE_TEXT_PT} pt; quite texto o reduzca la figura"
        );
    }
    Ok(())
}

/// Menor letra, desde `base`, con la que el lienzo `canvas(letra)` deja el texto legible. El lienzo
/// puede crecer con la letra (bandas de pies, columnas de etiquetas): si no converge, error.
fn fit_font(what: &str, base: u32, canvas: impl Fn(u32) -> (i64, i64)) -> Result<u32> {
    let mut font = base;
    loop {
        let (w, h) = canvas(font);
        if w > MAX_SIDE_PX || h > MAX_SIDE_PX {
            bail!("{what}: la imagen de {w} x {h} px supera el maximo de {MAX_SIDE_PX} px por lado");
        }
        let need = min_font_px(w as u32, h as u32);
        if need <= font {
            return Ok(font);
        }
        if need > MAX_FONT_PX {
            bail!(
                "{what}: el texto no llega a {MIN_FIGURE_TEXT_PT} pt en la diapositiva; \
                 quite pies o etiquetas, o reparta la figura"
            );
        }
        font = need;
    }
}

fn text_w(text: &str, px: u32) -> i64 {
    if text.is_empty() {
        return 0;
    }
    (FONT, px)
        .into_font()
        .box_size(text)
        .map(|(w, _)| w as i64)
        .unwrap_or_else(|_| (text.chars().count() as f64 * 0.62 * px as f64).ceil() as i64)
}

fn style(px: u32, h: HPos, v: VPos) -> TextStyle<'static> {
    (FONT, px).into_font().color(&BLACK).pos(Pos::new(h, v))
}

fn ensure_parent(path: &Path) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).with_context(|| format!("no se pudo crear {}", parent.display()))?;
    }
    Ok(())
}

#[derive(Debug, Clone)]
pub struct Series {
    pub name: String,
    pub points: Vec<(f64, f64)>,
}

/// Lienzo de las curvas: a lo ancho de la diapositiva deja lugar para una linea de texto debajo.
const CHART_W: u32 = 1380;
const CHART_H: u32 = 530;
const CHART_TITLE_PX: u32 = 36;
const CHART_AXIS_PX: u32 = 28;
const CHART_TICK_PX: u32 = 26;
const CHART_LEGEND_PX: u32 = 28;
const CHART_STROKE: u32 = 3;
const LEGEND_ROW_H: i32 = 40;
const LEGEND_SWATCH: i32 = 48;
const LEGEND_PAD: i32 = 14;
const X_AREA: i32 = 82;
const Y_AREA: i32 = 128;

/// Numero de un eje lineal sin decimales de mas: enteros si el rango es amplio.
fn tick_label(v: f64, span: f64) -> String {
    if span >= 8.0 {
        format!("{v:.0}")
    } else if span >= 0.8 {
        format!("{v:.1}")
    } else {
        format!("{v:.2}")
    }
}

/// Potencia de diez de un eje logaritmico: 100, 1, 0.01, 1e-5.
fn log_tick_label(v: f64) -> String {
    let e = v.log10().round();
    if (v / 10f64.powf(e) - 1.0).abs() > 1e-6 {
        return format!("{v:.1e}");
    }
    let e = e as i32;
    if e >= 0 {
        format!("{v:.0}")
    } else if e >= -3 {
        format!("{:.*}", (-e) as usize, v)
    } else {
        format!("1e{e}")
    }
}

/// Dibuja un PNG de 1380 x 530 con una linea por serie, sin grilla, con la leyenda en un panel a la
/// derecha del grafico (nunca sobre las curvas). Con `log_y` el eje y es logaritmico y los valores se
/// acotan inferiormente en 1e-12.
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
    let smallest = CHART_TICK_PX.min(CHART_AXIS_PX).min(CHART_LEGEND_PX).min(CHART_TITLE_PX);
    check_legible("grafico de curvas", smallest, CHART_W, CHART_H)?;
    let legend_w =
        LEGEND_SWATCH + 2 * LEGEND_PAD + series.iter().map(|s| text_w(&s.name, CHART_LEGEND_PX)).max().unwrap_or(0) as i32 + 10;
    let plot_w = CHART_W as i32 - legend_w;
    if plot_w < CHART_W as i32 / 2 {
        bail!("los nombres de las series ocupan {legend_w} px y no dejan lugar al grafico; acortelos");
    }
    let legend_h = series.len() as i32 * LEGEND_ROW_H;
    if legend_h > CHART_H as i32 - CHART_TITLE_PX as i32 - X_AREA - 30 {
        bail!("{} series no entran en la leyenda con letra legible", series.len());
    }
    ensure_parent(path)?;

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
    let x_span = x_max - x_min;

    let root = BitMapBackend::new(path, (CHART_W, CHART_H)).into_drawing_area();
    root.fill(&WHITE).map_err(de)?;
    let body = root.titled(title, (FONT, CHART_TITLE_PX)).map_err(de)?;
    let (left, right) = body.split_horizontally(plot_w);

    macro_rules! draw {
        ($chart:ident, $yfmt:expr) => {{
            $chart
                .configure_mesh()
                .disable_mesh()
                .x_desc(x_label)
                .y_desc(y_label)
                .axis_desc_style((FONT, CHART_AXIS_PX))
                .label_style((FONT, CHART_TICK_PX))
                .x_labels(6)
                .y_labels(6)
                .x_label_formatter(&|v| tick_label(*v, x_span))
                .y_label_formatter(&$yfmt)
                .draw()
                .map_err(de)?;
            for (i, p) in pts.iter().enumerate() {
                let color = PALETTE[i % PALETTE.len()];
                $chart.draw_series(LineSeries::new(p.iter().copied(), color.stroke_width(CHART_STROKE))).map_err(de)?;
            }
        }};
    }
    macro_rules! builder {
        () => {
            ChartBuilder::on(&left)
                .margin_top(10)
                .margin_bottom(6)
                .margin_left(8)
                .margin_right(48)
                .x_label_area_size(X_AREA)
                .y_label_area_size(Y_AREA)
        };
    }

    if log_y {
        let lo = y_min.max(LOG_FLOOR);
        if y_max <= lo {
            y_max = lo * 10.0;
        }
        let mut chart = builder!().build_cartesian_2d(x_min..x_max, (lo..y_max * 1.2).log_scale()).map_err(de)?;
        draw!(chart, |v: &f64| log_tick_label(*v));
    } else {
        let lo = y_min.min(0.0);
        if y_max <= lo {
            y_max = lo + 1.0;
        }
        let y_span = y_max * 1.05 - lo;
        let mut chart = builder!().build_cartesian_2d(x_min..x_max, lo..y_max * 1.05).map_err(de)?;
        draw!(chart, |v: &f64| tick_label(*v, y_span));
    }

    let legend_style = style(CHART_LEGEND_PX, HPos::Left, VPos::Center);
    for (i, s) in series.iter().enumerate() {
        let color = PALETTE[i % PALETTE.len()];
        let y = 30 + i as i32 * LEGEND_ROW_H + LEGEND_ROW_H / 2;
        right
            .draw(&PathElement::new(vec![(LEGEND_PAD, y), (LEGEND_PAD + LEGEND_SWATCH, y)], color.stroke_width(5)))
            .map_err(de)?;
        right
            .draw(&Text::new(s.name.clone(), (2 * LEGEND_PAD + LEGEND_SWATCH, y), legend_style.clone()))
            .map_err(de)?;
    }
    root.present().map_err(de)?;
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

const LATENT_W: u32 = 1200;
const LATENT_H: u32 = 720;
const LATENT_TITLE_PX: u32 = 36;
const LATENT_TEXT_PX: u32 = 30;
const LATENT_DOT: i32 = 5;

fn overlaps(a: &[i32; 4], b: &[i32; 4]) -> bool {
    a[0] < b[2] && b[0] < a[2] && a[1] < b[3] && b[1] < a[3]
}

/// AE-05: dibuja los puntos `(etiqueta, z1, z2)` en el plano latente (PNG de 1200 x 720, sin
/// grilla), cada uno como un circulo relleno con su caracter a la derecha. Si ese lugar esta
/// ocupado la etiqueta se corre a la primera posicion libre alrededor del punto y se une a el con
/// una linea fina.
pub fn plot_latent_scatter(path: &Path, points: &[(String, f64, f64)]) -> Result<()> {
    if points.is_empty() {
        bail!("no hay puntos latentes para graficar");
    }
    if points.iter().any(|(_, x, y)| !x.is_finite() || !y.is_finite()) {
        bail!("hay coordenadas latentes no finitas");
    }
    check_legible("plano latente", LATENT_TEXT_PX.min(LATENT_TITLE_PX), LATENT_W, LATENT_H)?;
    ensure_parent(path)?;
    let (x0, x1) = padded_range(points.iter().map(|p| p.1));
    let (y0, y1) = padded_range(points.iter().map(|p| p.2));
    let (x_span, y_span) = (x1 - x0, y1 - y0);

    let root = BitMapBackend::new(path, (LATENT_W, LATENT_H)).into_drawing_area();
    root.fill(&WHITE).map_err(de)?;
    let title = format!("Espacio latente de 2 dimensiones: {} patrones", points.len());
    let mut chart = ChartBuilder::on(&root)
        .caption(title, (FONT, LATENT_TITLE_PX))
        .margin_top(8)
        .margin_bottom(6)
        .margin_left(8)
        .margin_right(40)
        .x_label_area_size(86)
        .y_label_area_size(118)
        .build_cartesian_2d(x0..x1, y0..y1)
        .map_err(de)?;
    chart
        .configure_mesh()
        .disable_mesh()
        .x_desc("z1")
        .y_desc("z2")
        .axis_desc_style((FONT, LATENT_TEXT_PX))
        .label_style((FONT, LATENT_TEXT_PX))
        .x_labels(6)
        .y_labels(6)
        .x_label_formatter(&|v| tick_label(*v, x_span))
        .y_label_formatter(&|v| tick_label(*v, y_span))
        .draw()
        .map_err(de)?;

    let px: Vec<(i32, i32)> = points.iter().map(|(_, x, y)| chart.backend_coord(&(*x, *y))).collect();
    let (xr, yr) = chart.plotting_area().get_pixel_range();
    let dots: Vec<[i32; 4]> =
        px.iter().map(|&(x, y)| [x - LATENT_DOT - 2, y - LATENT_DOT - 2, x + LATENT_DOT + 2, y + LATENT_DOT + 2]).collect();
    const DIRS: [(i32, i32); 8] = [(1, 0), (1, -1), (1, 1), (-1, 0), (-1, -1), (-1, 1), (0, -1), (0, 1)];
    let h = (LATENT_TEXT_PX as f64 * 0.9) as i32;
    let mut placed: Vec<[i32; 4]> = Vec::with_capacity(points.len());
    let mut centers: Vec<(i32, i32, bool)> = Vec::with_capacity(points.len());
    for (i, (label, _, _)) in points.iter().enumerate() {
        let w = text_w(label, LATENT_TEXT_PX) as i32;
        let (cx, cy) = px[i];
        let boxed = |ring: i32, (ux, uy): (i32, i32)| {
            let step = ring * (h / 2 + 4);
            let x = cx + ux * (w / 2 + LATENT_DOT + 6 + step);
            let y = cy + uy * (h / 2 + LATENT_DOT + 4 + step);
            [x - w / 2 - 2, y - h / 2 - 2, x + w / 2 + 2, y + h / 2 + 2]
        };
        let free = |b: &[i32; 4]| {
            b[0] >= xr.start
                && b[2] <= xr.end
                && b[1] >= yr.start
                && b[3] <= yr.end
                && !placed.iter().any(|p| overlaps(p, b))
                && !dots.iter().any(|d| overlaps(d, b))
        };
        let mut chosen = (boxed(0, DIRS[0]), false);
        'search: for ring in 0..8 {
            for dir in DIRS {
                let b = boxed(ring, dir);
                if free(&b) {
                    chosen = (b, ring > 0 || dir != DIRS[0]);
                    break 'search;
                }
            }
        }
        placed.push(chosen.0);
        centers.push(((chosen.0[0] + chosen.0[2]) / 2, (chosen.0[1] + chosen.0[3]) / 2, chosen.1));
    }
    for (i, b) in placed.iter().enumerate() {
        if centers[i].2 {
            let (cx, cy) = px[i];
            let end = (cx.clamp(b[0], b[2]), cy.clamp(b[1], b[3]));
            root.draw(&PathElement::new(vec![(cx, cy), end], FRAME.stroke_width(1))).map_err(de)?;
        }
    }
    for &(x, y) in &px {
        root.draw(&Circle::new((x, y), LATENT_DOT, PALETTE[0].filled())).map_err(de)?;
    }
    let label_style =
        (FONT, LATENT_TEXT_PX).into_font().color(&RGBColor(180, 30, 30)).pos(Pos::new(HPos::Center, VPos::Center));
    for (i, (label, _, _)) in points.iter().enumerate() {
        root.draw(&Text::new(label.clone(), (centers[i].0, centers[i].1), label_style.clone())).map_err(de)?;
    }
    root.present().map_err(de)?;
    Ok(())
}

/// Un patron de 7 x 5: cuadrados negros de `sq` px donde el valor es al menos 0.5 y un solo marco
/// fino alrededor del patron (sin contorno por celda).
fn draw_pattern(root: &Area, x0: i32, y0: i32, sq: i32, pattern: &Array1<f64>) -> Result<()> {
    use crate::data::{COLS, ROWS};
    for r in 0..ROWS {
        for c in 0..COLS {
            if pattern[r * COLS + c] >= 0.5 {
                let (px, py) = (x0 + c as i32 * sq, y0 + r as i32 * sq);
                root.draw(&Rectangle::new([(px, py), (px + sq, py + sq)], BLACK.filled())).map_err(de)?;
            }
        }
    }
    root.draw(&Rectangle::new(
        [(x0 - 1, y0 - 1), (x0 + COLS as i32 * sq + 1, y0 + ROWS as i32 * sq + 1)],
        FRAME.stroke_width(1),
    ))
    .map_err(de)?;
    Ok(())
}

/// Lado en pixeles de cada celda de `plot_pattern_grid` y margen exterior.
const CELL: i32 = 40;
const MARGIN: i32 = 40;

/// Dibuja los items lado a lado: cada uno es un patron de 7 filas x 5 columnas de celdas de 40 px
/// (negra si el valor es al menos 0.5) con un marco fino y su titulo centrado debajo.
pub fn plot_pattern_grid(path: &Path, items: &[(String, Array1<f64>)]) -> Result<()> {
    use crate::data::{COLS, N_PIXELS, ROWS};
    if items.is_empty() {
        bail!("no hay patrones para graficar");
    }
    if let Some((t, _)) = items.iter().find(|(_, p)| p.len() != N_PIXELS) {
        bail!("el patron '{t}' no tiene {N_PIXELS} pixeles");
    }
    let grid_w = COLS as i64 * CELL as i64;
    let grid_h = ROWS as i64 * CELL as i64;
    let n = items.len() as i64;
    let pitch = |f: u32| (grid_w + MARGIN as i64).max(items.iter().map(|(t, _)| text_w(t, f)).max().unwrap_or(0) + 24);
    let canvas = |f: u32| (MARGIN as i64 + n * pitch(f), MARGIN as i64 + grid_h + 12 + (1.5 * f as f64) as i64 + MARGIN as i64 / 2);
    let font = fit_font("patrones lado a lado", 24, canvas)?;
    let (width, height) = canvas(font);
    check_legible("patrones lado a lado", font, width as u32, height as u32)?;
    ensure_parent(path)?;

    let root = BitMapBackend::new(path, (width as u32, height as u32)).into_drawing_area();
    root.fill(&WHITE).map_err(de)?;
    let title_style = style(font, HPos::Center, VPos::Top);
    let p = pitch(font) as i32;
    for (k, (title, pattern)) in items.iter().enumerate() {
        let x0 = MARGIN / 2 + k as i32 * p + (p - grid_w as i32) / 2;
        draw_pattern(&root, x0, MARGIN, CELL, pattern)?;
        root.draw(&Text::new(title.clone(), (x0 + grid_w as i32 / 2, MARGIN + grid_h as i32 + 12), title_style.clone()))
            .map_err(de)?;
    }
    root.present().map_err(de)?;
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
            &format!("Pérdida de entrenamiento, semilla 0 (estudio {study})"),
            "época",
            "pérdida",
            &loss_series,
            true,
        )?;
        plot_series(
            &out_dir.join(format!("curves_{study}_wrongpx.png")),
            &format!("Máximo de píxeles incorrectos, semilla 0 (estudio {study})"),
            "época",
            "píxeles incorrectos",
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
const ROWS_SQUARE: i64 = 28;
const ROWS_GAP_X: i64 = 24;
const ROWS_GAP_Y: i64 = 16;
const ROWS_MARGIN: i64 = 24;
const ROWS_BLOCK_GAP: i64 = 70;

/// Como `plot_pattern_rows_in_blocks` con un solo bloque.
pub fn plot_pattern_rows(path: &Path, title: &str, rows: &[PatternRow]) -> Result<()> {
    plot_pattern_rows_in_blocks(path, title, rows, 1)
}

/// Dibuja un PNG con un titulo centrado arriba y una banda por fila: la etiqueta de la fila a la
/// izquierda y sus patrones de 7 x 5 (cuadrados de 28 px, negros si el valor es al menos 0.5, con un
/// marco fino por patron) lado a lado, cada uno con su pie centrado debajo si alguno lo tiene. Las
/// filas se reparten en `blocks` bloques lado a lado, en orden. El tamano de letra es el menor que
/// queda legible en la diapositiva; si los pies o las etiquetas no lo permiten, devuelve un error.
pub fn plot_pattern_rows_in_blocks(path: &Path, title: &str, rows: &[PatternRow], blocks: usize) -> Result<()> {
    use crate::data::{COLS, N_PIXELS, ROWS};
    if rows.is_empty() {
        bail!("no hay filas de patrones para graficar");
    }
    if blocks == 0 {
        bail!("la cantidad de bloques debe ser positiva");
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

    let grid_w = COLS as i64 * ROWS_SQUARE;
    let grid_h = ROWS as i64 * ROWS_SQUARE;
    let n = n_cells as i64;
    let blocks = blocks.min(rows.len());
    let per_block = rows.len().div_ceil(blocks) as i64;
    let n_blocks = blocks as i64;
    let captioned = rows.iter().any(|r| r.cells.iter().any(|(c, _)| !c.is_empty()));
    let label_w = |f: u32| match rows.iter().map(|r| text_w(&r.label, f)).max().unwrap_or(0) {
        0 => 0,
        w => w + 24,
    };
    let caption_h = |f: u32| if captioned { (1.35 * f as f64) as i64 } else { 0 };
    let pitch = |f: u32| {
        let widest = rows.iter().flat_map(|r| r.cells.iter()).map(|(c, _)| text_w(c, f)).max().unwrap_or(0);
        (grid_w + ROWS_GAP_X).max(widest + 16)
    };
    let title_h = |f: u32| (1.7 * f as f64) as i64;
    let block_w = |f: u32| label_w(f) + n * pitch(f);
    let canvas = |f: u32| {
        let content = n_blocks * block_w(f) + (n_blocks - 1) * ROWS_BLOCK_GAP;
        let w = 2 * ROWS_MARGIN + content.max(text_w(title, f));
        let h = ROWS_MARGIN / 2 + title_h(f) + per_block * (grid_h + caption_h(f)) + (per_block - 1) * ROWS_GAP_Y + ROWS_MARGIN;
        (w, h)
    };
    let font = fit_font("filas de patrones", 22, canvas)?;
    if font as i64 > grid_h {
        bail!("filas de patrones: la letra legible ({font} px) es mas alta que un patron; quite pies o etiquetas");
    }
    let (width, height) = canvas(font);
    check_legible("filas de patrones", font, width as u32, height as u32)?;
    ensure_parent(path)?;

    let root = BitMapBackend::new(path, (width as u32, height as u32)).into_drawing_area();
    root.fill(&WHITE).map_err(de)?;
    root.draw(&Text::new(
        title.to_string(),
        ((width / 2) as i32, (ROWS_MARGIN / 2 + title_h(font) / 2) as i32),
        style(font, HPos::Center, VPos::Center),
    ))
    .map_err(de)?;
    let label_style = style(font, HPos::Left, VPos::Center);
    let caption_style = style(font, HPos::Center, VPos::Top);
    let content = n_blocks * block_w(font) + (n_blocks - 1) * ROWS_BLOCK_GAP;
    let left = (width - content) / 2;
    let (p, lw, band) = (pitch(font), label_w(font), grid_h + caption_h(font));

    for (ri, row) in rows.iter().enumerate() {
        let (block, in_block) = ((ri as i64) / per_block, (ri as i64) % per_block);
        let bx = left + block * (block_w(font) + ROWS_BLOCK_GAP);
        let y0 = ROWS_MARGIN / 2 + title_h(font) + in_block * (band + ROWS_GAP_Y);
        root.draw(&Text::new(row.label.clone(), (bx as i32, (y0 + grid_h / 2) as i32), label_style.clone()))
            .map_err(de)?;
        for (ci, (caption, pattern)) in row.cells.iter().enumerate() {
            let x0 = bx + lw + ci as i64 * p + (p - grid_w) / 2;
            draw_pattern(&root, x0 as i32, y0 as i32, ROWS_SQUARE as i32, pattern)?;
            if !caption.is_empty() {
                root.draw(&Text::new(
                    caption.clone(),
                    ((x0 + grid_w / 2) as i32, (y0 + grid_h + 6) as i32),
                    caption_style.clone(),
                ))
                .map_err(de)?;
            }
        }
    }
    root.present().map_err(de)?;
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
            &format!("Pérdida de entrenamiento sobre el lote ruidoso, semilla 0 (estudio {study})"),
            "época",
            "pérdida",
            &loss_series,
            true,
        )?;
        written += 1;
    }
    Ok(written)
}

/// Geometria de `plot_gray_tiles` (en pixeles de pantalla).
const GRAY_GAP_CAPTIONED: i64 = 8;
const GRAY_MARGIN: i64 = 16;

/// Dibuja un PNG con un titulo centrado arriba y debajo los `tiles` en orden fila mayor, `cols` por
/// fila (la ultima fila puede quedar incompleta). Cada tile es una imagen de `side x side` valores en
/// escala de grises ampliada `scale` veces (un cuadrado de `scale x scale` por valor; nivel de gris
/// 255 - round(255 * valor acotado a [0, 1]), asi 1.0 es tinta negra sobre blanco). Con pies, cada
/// tile lleva un marco fino y su pie centrado debajo (solo las filas que tienen algun pie reservan
/// banda). Si todos los pies son cadenas vacias los tiles van pegados, sin marcos internos y con un
/// solo marco exterior, de modo que una grilla densa se lee como una sola imagen continua. El tamano
/// de letra es el menor que queda legible en la diapositiva; si no se puede, devuelve un error.
pub fn plot_gray_tiles(
    path: &Path,
    title: &str,
    tiles: &[(String, Array1<f64>)],
    side: usize,
    cols: usize,
    scale: usize,
) -> Result<()> {
    plot_tiles(path, title, tiles, side, cols, scale, 1)
}

/// Como `plot_gray_tiles`, pero cada tile es una imagen RGB de `side x side` pixeles con sus 3 canales
/// en [0, 1] seguidos (orden fila, columna, canal), dibujada con sus propios colores.
pub fn plot_color_tiles(
    path: &Path,
    title: &str,
    tiles: &[(String, Array1<f64>)],
    side: usize,
    cols: usize,
    scale: usize,
) -> Result<()> {
    plot_tiles(path, title, tiles, side, cols, scale, 3)
}

fn plot_tiles(
    path: &Path,
    title: &str,
    tiles: &[(String, Array1<f64>)],
    side: usize,
    cols: usize,
    scale: usize,
    channels: usize,
) -> Result<()> {
    if tiles.is_empty() {
        bail!("no hay imagenes para graficar");
    }
    if cols == 0 || scale == 0 || side == 0 {
        bail!("el lado ({side}), las columnas ({cols}) y la escala ({scale}) deben ser positivos");
    }
    let values = side * side * channels;
    for (caption, tile) in tiles {
        if tile.len() != values {
            bail!("la imagen '{caption}' tiene {} valores y se esperaban {values}", tile.len());
        }
        if tile.iter().any(|v| !v.is_finite()) {
            bail!("la imagen '{caption}' tiene valores no finitos");
        }
    }
    let dense = tiles.iter().all(|(c, _)| c.is_empty());
    let gap = if dense { 0 } else { GRAY_GAP_CAPTIONED };
    let tile_px = (side as i64).saturating_mul(scale as i64);
    let n_cols = cols.min(tiles.len()) as i64;
    let n_rows = tiles.len().div_ceil(cols);
    if tile_px.saturating_mul(n_cols.max(n_rows as i64)) > MAX_SIDE_PX {
        bail!("la imagen de {n_cols} x {n_rows} tiles de {tile_px} px supera el maximo de {MAX_SIDE_PX} px por lado");
    }
    let row_has_caption: Vec<bool> =
        (0..n_rows).map(|r| tiles.iter().skip(r * cols).take(cols).any(|(c, _)| !c.is_empty())).collect();
    let captioned_rows = row_has_caption.iter().filter(|&&b| b).count() as i64;
    let caption_h = |f: u32| (1.35 * f as f64) as i64;
    let title_h = |f: u32| (1.7 * f as f64) as i64;
    let pitch = |f: u32| {
        let widest = tiles.iter().map(|(c, _)| text_w(c, f)).max().unwrap_or(0);
        (tile_px + gap).max(widest + 14)
    };
    let canvas = |f: u32| {
        let w = 2 * GRAY_MARGIN + (n_cols * pitch(f)).max(text_w(title, f));
        let h = GRAY_MARGIN / 2
            + title_h(f)
            + n_rows as i64 * tile_px
            + captioned_rows * caption_h(f)
            + (n_rows as i64 - 1) * gap
            + GRAY_MARGIN;
        (w, h)
    };
    let font = fit_font("mosaico de imagenes", 14, canvas)?;
    let (width, height) = canvas(font);
    check_legible("mosaico de imagenes", font, width as u32, height as u32)?;
    ensure_parent(path)?;

    let root = BitMapBackend::new(path, (width as u32, height as u32)).into_drawing_area();
    root.fill(&WHITE).map_err(de)?;
    root.draw(&Text::new(
        title.to_string(),
        ((width / 2) as i32, (GRAY_MARGIN / 2 + title_h(font) / 2) as i32),
        style(font, HPos::Center, VPos::Center),
    ))
    .map_err(de)?;
    let caption_style = style(font, HPos::Center, VPos::Top);
    let p = pitch(font);
    let left = (width - n_cols * p) / 2;
    let top = GRAY_MARGIN / 2 + title_h(font);
    let mut row_y = Vec::with_capacity(n_rows);
    let mut y = top;
    for has in &row_has_caption {
        row_y.push(y);
        y += tile_px + gap + if *has { caption_h(font) } else { 0 };
    }

    for (k, (caption, tile)) in tiles.iter().enumerate() {
        let (row, col) = (k / cols, (k % cols) as i64);
        let x0 = left + col * p + (p - tile_px) / 2;
        let y0 = row_y[row];
        for r in 0..side {
            for c in 0..side {
                let at = (r * side + c) * channels;
                let byte = |v: f64| (255.0 * v.clamp(0.0, 1.0)).round() as u8;
                let color = if channels == 1 {
                    let level = 255 - byte(tile[at]);
                    RGBColor(level, level, level)
                } else {
                    RGBColor(byte(tile[at]), byte(tile[at + 1]), byte(tile[at + 2]))
                };
                let (px, py) = (x0 + c as i64 * scale as i64, y0 + r as i64 * scale as i64);
                root.draw(&Rectangle::new(
                    [(px as i32, py as i32), ((px + scale as i64) as i32, (py + scale as i64) as i32)],
                    color.filled(),
                ))
                .map_err(de)?;
            }
        }
        if !dense {
            root.draw(&Rectangle::new(
                [(x0 as i32, y0 as i32), ((x0 + tile_px) as i32, (y0 + tile_px) as i32)],
                FRAME.stroke_width(1),
            ))
            .map_err(de)?;
            if !caption.is_empty() {
                root.draw(&Text::new(
                    caption.clone(),
                    ((x0 + tile_px / 2) as i32, (y0 + tile_px + 4) as i32),
                    caption_style.clone(),
                ))
                .map_err(de)?;
            }
        }
    }
    if dense {
        let last_row_y = *row_y.last().unwrap_or(&top);
        root.draw(&Rectangle::new(
            [(left as i32 - 1, top as i32 - 1), ((left + n_cols * p) as i32 + 1, (last_row_y + tile_px) as i32 + 1)],
            FRAME.stroke_width(1),
        ))
        .map_err(de)?;
    }
    root.present().map_err(de)?;
    Ok(())
}
