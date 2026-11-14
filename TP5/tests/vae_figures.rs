//! Figuras del Ej 2: mosaicos de grises y de color, dataset de bloques y reconstrucciones del VAE.

use std::path::PathBuf;
use std::sync::OnceLock;

use ndarray::Array1;

use tp5::ores::{load_ores, OreDataset};
use tp5::plots::{plot_color_tiles, plot_gray_tiles};
use tp5::vae::{chosen_vae_config, train_vae, Vae};
use tp5::vae_figures::{
    build_reconstruction_tiles, export_ore_montage, export_reconstruction, ORES_PNG, RECON_PNG,
};

const PNG_SIGNATURE: [u8; 8] = [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];

fn unique_tmp_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("tp5-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    dir
}

fn quick_model() -> &'static (OreDataset, Vae) {
    static MODEL: OnceLock<(OreDataset, Vae)> = OnceLock::new();
    MODEL.get_or_init(|| {
        let ds = ores();
        let mut cfg = chosen_vae_config();
        cfg.epochs = 100;
        let (vae, _) = train_vae(&ds, &cfg).unwrap();
        (ds, vae)
    })
}

fn gradient_tile(side: usize) -> Array1<f64> {
    Array1::from_shape_fn(side * side, |i| (i % 7) as f64 / 6.0)
}

fn png_size(bytes: &[u8]) -> (u32, u32) {
    let w = u32::from_be_bytes(bytes[16..20].try_into().unwrap());
    let h = u32::from_be_bytes(bytes[20..24].try_into().unwrap());
    (w, h)
}

#[test]
fn plot_gray_tiles_writes_png() {
    let dir = unique_tmp_dir("gray-tiles");
    let path = dir.join("tiles.png");
    let tiles: Vec<(String, Array1<f64>)> = (0..4).map(|k| (format!("t{k}"), gradient_tile(8))).collect();
    plot_gray_tiles(&path, "titulo", &tiles, 8, 2, 5).unwrap();
    let bytes = std::fs::read(&path).unwrap();
    assert_eq!(&bytes[..8], &PNG_SIGNATURE);
    assert!(bytes.len() > 3000, "{} bytes", bytes.len());
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn plot_gray_tiles_rejects_bad_input() {
    let dir = unique_tmp_dir("gray-tiles-bad");
    let path = dir.join("bad.png");
    let ok = vec![("a".to_string(), gradient_tile(4))];
    assert!(plot_gray_tiles(&path, "t", &[], 4, 1, 1).is_err(), "sin imagenes");
    assert!(plot_gray_tiles(&path, "t", &ok, 4, 0, 1).is_err(), "cols 0");
    assert!(plot_gray_tiles(&path, "t", &ok, 4, 1, 0).is_err(), "escala 0");
    assert!(plot_gray_tiles(&path, "t", &ok, 0, 1, 1).is_err(), "lado 0");
    assert!(plot_gray_tiles(&path, "t", &ok, 5, 1, 1).is_err(), "largo equivocado");
    let mut nan = gradient_tile(4);
    nan[3] = f64::NAN;
    assert!(plot_gray_tiles(&path, "t", &[("n".to_string(), nan)], 4, 1, 1).is_err(), "NaN");
    let big = vec![("b".to_string(), gradient_tile(100))];
    assert!(plot_gray_tiles(&path, "t", &big, 100, 1, 100).is_err(), "demasiado grande");
    assert!(!path.exists(), "no debe escribirse nada con entrada invalida");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn plot_gray_tiles_without_captions_is_denser() {
    let dir = unique_tmp_dir("gray-tiles-dense");
    let with = dir.join("with.png");
    let without = dir.join("without.png");
    let named: Vec<(String, Array1<f64>)> = (0..6).map(|k| (format!("t{k}"), gradient_tile(8))).collect();
    let bare: Vec<(String, Array1<f64>)> = (0..6).map(|_| (String::new(), gradient_tile(8))).collect();
    plot_gray_tiles(&with, "t", &named, 8, 3, 4).unwrap();
    plot_gray_tiles(&without, "t", &bare, 8, 3, 4).unwrap();
    let (w1, h1) = png_size(&std::fs::read(&with).unwrap());
    let (w2, h2) = png_size(&std::fs::read(&without).unwrap());
    assert!(w2 < w1, "ancho {w2} vs {w1}");
    assert!(h2 < h1, "alto {h2} vs {h1}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn ore_montage_writes_png_with_the_fixed_name() {
    let dir = unique_tmp_dir("bloque-montage");
    let ds = ores();
    let path = export_ore_montage(&ds, &dir).unwrap();
    assert_eq!(path, dir.join(ORES_PNG));
    let bytes = std::fs::read(&path).unwrap();
    assert_eq!(&bytes[..8], &PNG_SIGNATURE);
    assert!(bytes.len() > 3000);
    let _ = std::fs::remove_dir_all(&dir);
}


#[test]
fn plot_color_tiles_draws_the_rgb_values_and_checks_the_length() {
    let dir = unique_tmp_dir("color-tiles");
    let path = dir.join("color.png");
    let red = Array1::from_shape_fn(4 * 4 * 3, |k| if k % 3 == 0 { 1.0 } else { 0.0 });
    plot_color_tiles(&path, "titulo", &[("rojo".to_string(), red.clone())], 4, 1, 20).unwrap();
    assert_eq!(&std::fs::read(&path).unwrap()[..8], &PNG_SIGNATURE);
    // Un tile gris (un canal) no tiene el largo de un tile RGB.
    assert!(plot_color_tiles(&path, "t", &[("g".to_string(), gradient_tile(4))], 4, 1, 1).is_err());
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn reconstruction_tiles_are_36_in_band_order() {
    let (ds, vae) = quick_model();
    let tiles = build_reconstruction_tiles(vae, ds);
    assert_eq!(tiles.len(), 36);
    let recon = vae.reconstruct_mean(&ds.x);
    for band in 0..2 {
        for k in 0..9 {
            let (name, original) = &tiles[band * 18 + k];
            assert_eq!(name, &ds.names[band * 9 + k]);
            assert_eq!(original, &ds.x.row(band * 9 + k).to_owned());
            let (caption, rec) = &tiles[band * 18 + 9 + k];
            assert!(caption.ends_with(" px"), "{caption}");
            assert_eq!(rec, &recon.row(band * 9 + k).to_owned());
        }
    }
}

#[test]
fn reconstruction_captions_match_an_independent_pixel_count() {
    let (ds, vae) = quick_model();
    let tiles = build_reconstruction_tiles(vae, ds);
    let recon = vae.reconstruct_mean(&ds.x);
    for i in 0..18 {
        let mut wrong = 0;
        for p in 0..256 {
            if (0..3).any(|c| (ds.x[[i, 3 * p + c]] - recon[[i, 3 * p + c]]).abs() > 32.0 / 255.0) {
                wrong += 1;
            }
        }
        let (band, k) = (i / 9, i % 9);
        assert_eq!(tiles[band * 18 + 9 + k].0, format!("{wrong} px"), "bloque {i}");
    }
}

#[test]
fn reconstruction_export_writes_png_with_the_fixed_name() {
    let dir = unique_tmp_dir("recon-export");
    let (ds, vae) = quick_model();
    let path = export_reconstruction(vae, ds, &dir).unwrap();
    assert_eq!(path, dir.join(RECON_PNG));
    let bytes = std::fs::read(&path).unwrap();
    assert_eq!(&bytes[..8], &PNG_SIGNATURE);
    assert!(bytes.len() > 3000);
    let _ = std::fs::remove_dir_all(&dir);
}

/// El dataset de menas; falta si no se corrio `bash scripts/descargar_menas.sh`.
fn ores() -> OreDataset {
    load_ores().expect("dataset de menas: corra `bash scripts/descargar_menas.sh`")
}
