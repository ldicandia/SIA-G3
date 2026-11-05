//! Exportadores de las figuras del Ej 2 (VAE): montaje del dataset y reconstrucciones. Paralelo a
//! `dae_figures`; el dibujo lo hace `plots`. No contiene codigo del modelo.

use std::path::{Path, PathBuf};

use anyhow::{anyhow, Result};
use ndarray::{Array1, Array2};

use crate::emoji::{EmojiDataset, EMOJI_COUNT, EMOJI_SIDE};
use crate::plots::{plot_gray_tiles, plot_series, Series};
use crate::report::wrong_pixel_counts;
use crate::vae::{Vae, VaeHistory};
use crate::vae_eval::SampleSearch;

/// Nombres fijos de los archivos (no se interpola texto del usuario en la ruta).
pub const EMOJIS_PNG: &str = "emojis.png";
pub const RECON_PNG: &str = "vae_reconstruction.png";
pub const LOSS_CSV: &str = "vae_loss.csv";
pub const LOSS_PNG: &str = "vae_loss.png";

pub const NEW_EMOJI_PNG: &str = "vae_new_emoji.png";

/// Emojis por banda de `build_reconstruction_tiles` (columnas de la figura).
const RECON_COLS: usize = 12;

/// Con k muestras aceptadas: 2k imagenes en dos bandas de k. Primero las imagenes decodificadas con el
/// pie `nuevo {i}: {d} px` y despues el emoji del dataset mas cercano de cada una con su nombre.
pub fn build_new_emoji_tiles(search: &SampleSearch, ds: &EmojiDataset) -> Vec<(String, Array1<f64>)> {
    let mut tiles: Vec<(String, Array1<f64>)> = search
        .accepted
        .iter()
        .enumerate()
        .map(|(i, s)| (format!("nuevo {}: {} px", i + 1, s.judgement.distance), s.image.clone()))
        .collect();
    tiles.extend(
        search
            .accepted
            .iter()
            .map(|s| (ds.names[s.judgement.nearest_index].clone(), ds.x.row(s.judgement.nearest_index).to_owned())),
    );
    tiles
}

/// Escribe `vae_new_emoji.png`: los emojis nuevos arriba y el emoji del dataset mas cercano abajo.
pub fn export_new_emoji(search: &SampleSearch, ds: &EmojiDataset, out_dir: &Path) -> Result<PathBuf> {
    if search.accepted.is_empty() {
        return Err(anyhow!(
            "no se encontro ningun emoji nuevo y plausible en {} sorteos; no se escribe {NEW_EMOJI_PNG}",
            search.examined
        ));
    }
    let path = out_dir.join(NEW_EMOJI_PNG);
    plot_gray_tiles(
        &path,
        "Ej 2 - emojis nuevos muestreados del prior (arriba) y el emoji del dataset mas cercano (abajo)",
        &build_new_emoji_tiles(search, ds),
        EMOJI_SIDE,
        search.accepted.len(),
        4,
    )?;
    Ok(path)
}

/// Escribe `emojis.png`: tabla de 6 x 6 con un estilo de ojos por fila y un estilo de boca por columna.
pub fn export_emoji_montage(ds: &EmojiDataset, out_dir: &Path) -> Result<PathBuf> {
    let tiles: Vec<(String, Array1<f64>)> =
        ds.names.iter().cloned().zip(ds.x.rows().into_iter().map(|r| r.to_owned())).collect();
    let path = out_dir.join(EMOJIS_PNG);
    plot_gray_tiles(
        &path,
        "Ej 2 - dataset de 36 emojis (filas: ojos, columnas: boca)",
        &tiles,
        EMOJI_SIDE,
        6,
        4,
    )?;
    Ok(path)
}

/// 72 imagenes en bandas de 12 originales seguidas de 12 reconstrucciones. Las reconstrucciones son
/// la decodificacion de la media del codificador (probabilidades en gris, sin binarizar) y su pie es
/// `{k} px`, con k los pixeles que difieren del original tras binarizar en 0.5.
pub fn build_reconstruction_tiles(vae: &Vae, ds: &EmojiDataset) -> Vec<(String, Array1<f64>)> {
    let recon = vae.reconstruct_mean(&ds.x);
    let counts = wrong_pixel_counts(&ds.x, &recon);
    let mut tiles = Vec::with_capacity(2 * EMOJI_COUNT);
    for band in (0..EMOJI_COUNT).step_by(RECON_COLS) {
        let end = (band + RECON_COLS).min(EMOJI_COUNT);
        for i in band..end {
            tiles.push((ds.names[i].clone(), ds.x.row(i).to_owned()));
        }
        for i in band..end {
            tiles.push((format!("{} px", counts[i]), recon.row(i).to_owned()));
        }
    }
    tiles
}

/// Escribe `vae_reconstruction.png` con originales (filas impares) y reconstrucciones (filas pares).
pub fn export_reconstruction(vae: &Vae, ds: &EmojiDataset, out_dir: &Path) -> Result<PathBuf> {
    let path = out_dir.join(RECON_PNG);
    plot_gray_tiles(
        &path,
        "Ej 2 - originales (filas impares) y reconstrucciones desde la media del codificador (filas pares)",
        &build_reconstruction_tiles(vae, ds),
        EMOJI_SIDE,
        RECON_COLS,
        3,
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
        Series { name: "reconstruccion (BCE)".to_string(), points: points(&|i| history.recon[i]) },
        Series { name: "divergencia KL".to_string(), points: points(&|i| history.kl[i]) },
        Series { name: "total".to_string(), points: points(&|i| history.total(i)) },
    ];
    let path = out_dir.join(LOSS_PNG);
    plot_series(
        &path,
        "Ej 2 - evolucion de la perdida del VAE (reconstruccion + KL)",
        "epoca",
        "perdida (escala logaritmica)",
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
    plot_gray_tiles(path, title, &build_manifold_tiles(decoded), EMOJI_SIDE, side, 2)
}

/// Escribe `vae_manifold.png`: una sola imagen continua con la grilla decodificada del plano latente.
pub fn export_manifold(decoded: &Array2<f64>, side: usize, out_dir: &Path) -> Result<PathBuf> {
    let path = out_dir.join(MANIFOLD_PNG);
    let title = format!(
        "Ej 2 - grilla {side}x{side} del plano latente: z1 de -2.5 a 2.5 (derecha), z2 de 2.5 a -2.5 (abajo)"
    );
    write_grid_png(decoded, side, &path, &title)?;
    Ok(path)
}
