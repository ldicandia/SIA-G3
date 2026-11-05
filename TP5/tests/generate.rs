//! AE-06: una letra nueva generada desde un punto latente que no es un patron de entrenamiento.

use std::sync::OnceLock;

use tp5::autoencoder::Autoencoder;
use tp5::data::{load_font, Dataset};
use tp5::experiments::train_chosen;
use tp5::generate::{
    decode_point, export_new_letter, pick_new_letter, scan_novel, NovelLetter, NEW_LETTER_PNG, NOVELTY_MIN_HAMMING,
};

struct Trained {
    ds: Dataset,
    ae: Autoencoder,
    /// Resultado de scan_novel con 41 pasos.
    cands: Vec<NovelLetter>,
}

fn trained() -> &'static Trained {
    static CELL: OnceLock<Trained> = OnceLock::new();
    CELL.get_or_init(|| {
        let ds = load_font().unwrap();
        let (ae, _) = train_chosen(&ds);
        let cands = scan_novel(&ae, &ds, 41, 0.1);
        Trained { ds, ae, cands }
    })
}

fn picked() -> &'static NovelLetter {
    pick_new_letter(&trained().cands).expect("ninguna letra nueva en el barrido")
}

#[test]
fn scan_finds_novel_letters() {
    assert!(!trained().cands.is_empty());
}

#[test]
fn picked_letter_is_not_a_training_pattern() {
    let t = trained();
    let l = picked();
    let mut min_h = usize::MAX;
    for row in t.ds.x.rows() {
        let h = row.iter().zip(l.pattern.iter()).filter(|(&a, &b)| (a >= 0.5) != (b >= 0.5)).count();
        assert!(h > 0, "la letra elegida coincide con un patron de entrenamiento");
        min_h = min_h.min(h);
    }
    assert!(l.min_hamming >= NOVELTY_MIN_HAMMING);
    assert_eq!(l.min_hamming, min_h, "min_hamming almacenado distinto del recalculado");
}

#[test]
fn picked_point_is_not_a_training_code() {
    let t = trained();
    let l = picked();
    let codes = t.ae.encode(&t.ds.x);
    for r in codes.rows() {
        let d = ((r[0] - l.z[0]).powi(2) + (r[1] - l.z[1]).powi(2)).sqrt();
        assert!(d > 1e-9, "el punto elegido es un codigo de entrenamiento");
    }
}

#[test]
fn decoding_the_picked_point_reproduces_the_pattern() {
    let t = trained();
    let l = picked();
    assert_eq!(decode_point(&t.ae, l.z), l.pattern);
}

#[test]
fn explicit_point_is_validated() {
    let t = trained();
    let dir = std::env::temp_dir().join(format!("tp5-gen-val-{}", std::process::id()));
    assert!(export_new_letter(&t.ae, &t.ds, &dir, Some([f64::NAN, 0.0]), 41).is_err());
    assert!(export_new_letter(&t.ae, &t.ds, &dir, Some([0.0, f64::INFINITY]), 41).is_err());
    assert!(export_new_letter(&t.ae, &t.ds, &dir, Some([1e9, 0.0]), 41).is_err());
    let ok = export_new_letter(&t.ae, &t.ds, &dir, Some([-1.5, 2.0]), 41).unwrap();
    assert_eq!(ok.z, [-1.5, 2.0]);
    assert_eq!(ok.pattern.len(), 35);
    assert!(ok.latent_distance.is_finite());
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn new_letter_png_is_written() {
    let t = trained();
    let dir = std::env::temp_dir().join(format!("tp5-gen-png-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    export_new_letter(&t.ae, &t.ds, &dir, None, 41).unwrap();
    let png = std::fs::read(dir.join(NEW_LETTER_PNG)).unwrap();
    assert_eq!(&png[..8], &[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]);
    assert!(png.len() > 1000, "png de {} bytes", png.len());
    let _ = std::fs::remove_dir_all(&dir);
}
