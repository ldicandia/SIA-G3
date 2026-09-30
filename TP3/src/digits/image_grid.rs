//! Grids of titled 28x28 maps (digit images and attribution maps) drawn
//! with plotters, shared by the noise and attribution studies.

use std::path::Path;

use anyhow::{bail, Result};
use plotters::prelude::*;

use super::data::IMAGE_PIXELS;

/// Side of a digit image in pixels.
const SIDE: usize = 28;
/// On-screen pixels per image pixel.
const SCALE: u32 = 5;
const CELL_WIDTH: u32 = SIDE as u32 * SCALE + 20;
const CELL_HEIGHT: u32 = SIDE as u32 * SCALE + 34;
const HEADER: u32 = 50;

/// How the values of one map are turned into colors.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum MapStyle {
    /// Input images in [0, 1]: ink dark on white.
    Grayscale,
    /// Non-negative maps: white to red, scaled by the cell maximum.
    Sequential,
    /// Signed maps: blue (negative), white (zero), red (positive), scaled
    /// symmetrically by the cell's maximum absolute value.
    Diverging,
}

/// One titled map of the grid; `values` holds 784 pixels in row-major order.
pub(crate) struct MapCell<'a> {
    pub(crate) title: String,
    pub(crate) values: &'a [f64],
    pub(crate) style: MapStyle,
    /// Draws the title in red (for example, a misclassified digit).
    pub(crate) highlight: bool,
}

/// Draws `cells` row by row into a `rows x cols` grid. Missing cells (when
/// `cells.len() < rows * cols`) are left blank.
pub(crate) fn draw_map_grid(
    path: &Path,
    title: &str,
    rows: usize,
    cols: usize,
    cells: &[Option<MapCell<'_>>],
) -> Result<()> {
    if rows == 0 || cols == 0 || cells.len() > rows * cols {
        bail!("invalid map grid: {rows}x{cols} for {} cells", cells.len());
    }
    let width = CELL_WIDTH * cols as u32;
    let height = HEADER + CELL_HEIGHT * rows as u32;
    let root = BitMapBackend::new(path, (width.max(400), height)).into_drawing_area();
    root.fill(&WHITE)?;
    let body = root.titled(title, ("sans-serif", 24))?;
    let areas = body.split_evenly((rows, cols));
    for (area, cell) in areas.iter().zip(cells) {
        let Some(cell) = cell else { continue };
        if cell.values.len() != IMAGE_PIXELS {
            bail!("map '{}' has {} values", cell.title, cell.values.len());
        }
        draw_cell(area, cell)?;
    }
    root.present()?;
    Ok(())
}

fn draw_cell(
    area: &DrawingArea<BitMapBackend<'_>, plotters::coord::Shift>,
    cell: &MapCell<'_>,
) -> Result<()> {
    let color = if cell.highlight {
        RGBColor(211, 47, 47)
    } else {
        BLACK
    };
    area.draw(&Text::new(
        cell.title.clone(),
        (8, 6),
        ("sans-serif", 14).into_font().color(&color),
    ))?;
    let scale = match cell.style {
        MapStyle::Grayscale => 1.0,
        MapStyle::Sequential => cell.values.iter().copied().fold(0.0_f64, f64::max),
        MapStyle::Diverging => cell
            .values
            .iter()
            .fold(0.0_f64, |acc, value| acc.max(value.abs())),
    };
    let (left, top) = (10_i32, 26_i32);
    let step = SCALE as i32;
    for (index, &value) in cell.values.iter().enumerate() {
        let (row, col) = ((index / SIDE) as i32, (index % SIDE) as i32);
        let x = left + col * step;
        let y = top + row * step;
        area.draw(&Rectangle::new(
            [(x, y), (x + step, y + step)],
            pixel_color(value, scale, cell.style).filled(),
        ))?;
    }
    let size = SIDE as i32 * step;
    area.draw(&Rectangle::new(
        [(left - 1, top - 1), (left + size, top + size)],
        RGBColor(160, 160, 160).stroke_width(1),
    ))?;
    Ok(())
}

fn pixel_color(value: f64, scale: f64, style: MapStyle) -> RGBColor {
    let channel = |fraction: f64| (255.0 * fraction.clamp(0.0, 1.0)).round() as u8;
    match style {
        MapStyle::Grayscale => {
            let gray = channel(1.0 - value);
            RGBColor(gray, gray, gray)
        }
        MapStyle::Sequential => {
            let t = if scale > 0.0 { value / scale } else { 0.0 };
            let fade = channel(1.0 - t);
            RGBColor(255, fade, fade)
        }
        MapStyle::Diverging => {
            let t = if scale > 0.0 { value / scale } else { 0.0 };
            if t >= 0.0 {
                let fade = channel(1.0 - t);
                RGBColor(255, fade, fade)
            } else {
                let fade = channel(1.0 + t);
                RGBColor(fade, fade, 255)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn colors_follow_the_documented_scales() {
        assert_eq!(
            pixel_color(1.0, 1.0, MapStyle::Grayscale),
            RGBColor(0, 0, 0)
        );
        assert_eq!(
            pixel_color(0.0, 1.0, MapStyle::Grayscale),
            RGBColor(255, 255, 255)
        );
        assert_eq!(
            pixel_color(2.0, 2.0, MapStyle::Sequential),
            RGBColor(255, 0, 0)
        );
        assert_eq!(
            pixel_color(-3.0, 3.0, MapStyle::Diverging),
            RGBColor(0, 0, 255)
        );
        assert_eq!(
            pixel_color(0.0, 0.0, MapStyle::Diverging),
            RGBColor(255, 255, 255)
        );
    }
}
