//! Exportadores de las figuras del Ej 2 (VAE): montaje del dataset y reconstrucciones. Paralelo a
//! `dae_figures`; el dibujo lo hace `plots`. No contiene codigo del modelo.

use std::path::{Path, PathBuf};

use anyhow::{anyhow, Result};
use ndarray::{Array1, Array2};

use crate::ores::{wrong_pixel_counts, OreDataset, ORE_COUNT, ORE_MINERALS, ORE_SIDE};
use crate::plots::{plot_color_tiles, plot_series, Series};
use crate::vae::{Vae, VaeHistory};
use crate::vae_eval::SampleSearch;

/// Nombres fijos de los archivos (no se interpola texto del usuario en la ruta).
pub const ORES_PNG: &str = "ores.png";
pub const RECON_PNG: &str = "vae_reconstruction.png";
pub const LOSS_CSV: &str = "vae_loss.csv";
pub const LOSS_PNG: &str = "vae_loss.png";

pub const NEW_ORE_PNG: &str = "vae_new_ore.png";

/// Bloques por banda de `build_reconstruction_tiles` (columnas de la figura).
const RECON_COLS: usize = 9;

/// Columnas de `vae_new_ore.png`.
pub const NEW_ORE_COLS: usize = 4;
/// Columnas de `ores.png`.
const MONTAGE_COLS: usize = 6;

/// Una imagen por muestra aceptada, en el orden de la galeria, con el pie `{i}: {minerales}` (la roca
/// se ve en el fondo de la propia imagen).
pub fn build_new_ore_tiles(search: &SampleSearch) -> Vec<(String, Array1<f64>)> {
    search
        .accepted
        .iter()
        .enumerate()
        .map(|(i, s)| {
            let minerals: Vec<&str> = s.verdict.minerals.iter().map(|&m| ORE_MINERALS[m]).collect();
            (format!("{}: {}", i + 1, minerals.join("+")), s.image.clone())
        })
        .collect()
}

/// Escribe `vae_new_ore.png`: los bloques nuevos de la galeria con sus minerales.
pub fn export_new_ore(search: &SampleSearch, out_dir: &Path) -> Result<PathBuf> {
    if search.accepted.is_empty() {
        return Err(anyhow!(
            "no se encontro ningun bloque nuevo y plausible en {} sorteos; no se escribe {NEW_ORE_PNG}",
            search.examined
        ));
    }
    let path = out_dir.join(NEW_ORE_PNG);
    plot_color_tiles(
        &path,
        "Bloques nuevos generados por el VAE y sus minerales",
        &build_new_ore_tiles(search),
        ORE_SIDE,
        search.accepted.len().min(NEW_ORE_COLS),
        6,
    )?;
    Ok(path)
}

/// Escribe `ores.png`: los 18 bloques en el orden del dataset (piedra, pizarra y Nether), cada uno con
/// su nombre `{mineral}-{roca}` como pie.
pub fn export_ore_montage(ds: &OreDataset, out_dir: &Path) -> Result<PathBuf> {
    let tiles: Vec<(String, Array1<f64>)> =
        ds.x.rows().into_iter().zip(&ds.names).map(|(r, n)| (n.clone(), r.to_owned())).collect();
    let path = out_dir.join(ORES_PNG);
    plot_color_tiles(
        &path,
        &format!("Dataset: {ORE_COUNT} bloques de mena de Minecraft, {ORE_SIDE} x {ORE_SIDE} RGB"),
        &tiles,
        ORE_SIDE,
        MONTAGE_COLS,
        5,
    )?;
    Ok(path)
}

/// 36 imagenes en bandas de 9 originales seguidas de 9 reconstrucciones. Las reconstrucciones son
/// la decodificacion de la media del codificador (RGB en [0, 1]) y su pie es
/// `{k} px`, con k los pixeles con algun canal a mas de 32 niveles del original.
pub fn build_reconstruction_tiles(vae: &Vae, ds: &OreDataset) -> Vec<(String, Array1<f64>)> {
    let recon = vae.reconstruct_mean(&ds.x);
    let counts = wrong_pixel_counts(&ds.x, &recon);
    let mut tiles = Vec::with_capacity(2 * ORE_COUNT);
    for band in (0..ORE_COUNT).step_by(RECON_COLS) {
        let end = (band + RECON_COLS).min(ORE_COUNT);
        for i in band..end {
            tiles.push((ds.names[i].clone(), ds.x.row(i).to_owned()));
        }
        for i in band..end {
            tiles.push((format!("{} px", counts[i]), recon.row(i).to_owned()));
        }
    }
    tiles
}

/// Escribe `vae_reconstruction.png` con originales (filas impares, sin pie) y reconstrucciones (filas
/// pares, con sus pixeles incorrectos como pie).
pub fn export_reconstruction(vae: &Vae, ds: &OreDataset, out_dir: &Path) -> Result<PathBuf> {
    let path = out_dir.join(RECON_PNG);
    let tiles: Vec<(String, Array1<f64>)> = build_reconstruction_tiles(vae, ds)
        .into_iter()
        .enumerate()
        .map(|(k, (caption, tile))| (if (k / RECON_COLS) % 2 == 0 { String::new() } else { caption }, tile))
        .collect();
    plot_color_tiles(
        &path,
        "Originales y, debajo, reconstrucciones con sus píxeles incorrectos",
        &tiles,
        ORE_SIDE,
        RECON_COLS,
        4,
    )?;
    Ok(path)
}

/// Escribe `epoch,recon,kl,total` con una fila por epoca (desde 1) y tres valores a 6 decimales.
pub fn write_loss_csv(path: &Path, history: &VaeHistory) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| anyhow!("no se pudo crear {}: {e}", parent.display()))?;
    }
    let mut out = String::from("epoch,recon,kl,total\n");
    for i in 0..history.recon.len() {
        out.push_str(&format!("{},{:.6},{:.6},{:.6}\n", i + 1, history.recon[i], history.kl[i], history.total(i)));
    }
    std::fs::write(path, out).map_err(|e| anyhow!("no se pudo escribir {}: {e}", path.display()))
}

/// Escribe `vae_loss.png`: reconstruccion, KL y total contra la epoca en escala logaritmica, con a lo
/// sumo unos 500 puntos por serie (uno de cada n/500 mas el ultimo, como `write_study_curves`).
pub fn export_loss_chart(history: &VaeHistory, out_dir: &Path) -> Result<PathBuf> {
    let n = history.recon.len();
    let step = (n / 500).max(1);
    let points = |value: &dyn Fn(usize) -> f64| -> Vec<(f64, f64)> {
        (0..n).filter(|i| i % step == 0 || i + 1 == n).map(|i| ((i + 1) as f64, value(i))).collect()
    };
    let series = vec![
        Series { name: "reconstrucción (BCE)".to_string(), points: points(&|i| history.recon[i]) },
        Series { name: "divergencia KL".to_string(), points: points(&|i| history.kl[i]) },
        Series { name: "total".to_string(), points: points(&|i| history.total(i)) },
    ];
    let path = out_dir.join(LOSS_PNG);
    plot_series(
        &path,
        "Pérdida del VAE por época",
        "época",
        "pérdida",
        &series,
        true,
    )?;
    Ok(path)
}

pub const MANIFOLD_PNG: &str = "vae_manifold.png";

/// Una imagen por celda de la grilla, en el orden de las celdas y con el pie vacio.
pub fn build_manifold_tiles(decoded: &Array2<f64>) -> Vec<(String, Array1<f64>)> {
    decoded.rows().into_iter().map(|r| (String::new(), r.to_owned())).collect()
}

fn write_grid_png(decoded: &Array2<f64>, side: usize, path: &Path, title: &str) -> Result<()> {
    if decoded.nrows() != side * side {
        return Err(anyhow!(
            "la grilla de {side}x{side} necesita {} imagenes y se recibieron {}",
            side * side,
            decoded.nrows()
        ));
    }
    plot_color_tiles(path, title, &build_manifold_tiles(decoded), ORE_SIDE, side, 3)
}

/// Escribe `vae_manifold.png`: una sola imagen continua con la grilla decodificada del plano latente.
pub fn export_manifold(decoded: &Array2<f64>, side: usize, out_dir: &Path) -> Result<PathBuf> {
    let path = out_dir.join(MANIFOLD_PNG);
    let title = format!("VAE: grilla {side}x{side} del plano latente");
    write_grid_png(decoded, side, &path, &title)?;
    Ok(path)
}

pub const PLAIN_MANIFOLD_PNG: &str = "plain_manifold.png";

/// Escribe `plain_manifold.png`: igual que `export_manifold` para el autoencoder sin esquema variacional.
pub fn export_plain_manifold(decoded: &Array2<f64>, side: usize, out_dir: &Path) -> Result<PathBuf> {
    let path = out_dir.join(PLAIN_MANIFOLD_PNG);
    let title = format!("Autoencoder simple: grilla {side}x{side}");
    write_grid_png(decoded, side, &path, &title)?;
    Ok(path)
}
