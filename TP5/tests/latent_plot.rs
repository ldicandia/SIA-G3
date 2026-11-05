//! AE-05: los 32 patrones en el plano latente de 2 dimensiones.

use std::sync::OnceLock;

use ndarray::Array2;
use tp5::analysis::{export_latent, LATENT_CSV, LATENT_PNG};
use tp5::autoencoder::Autoencoder;
use tp5::data::{load_font, Dataset};
use tp5::experiments::train_chosen;
use tp5::plots::plot_latent_scatter;

struct Trained {
    ds: Dataset,
    ae: Autoencoder,
}

fn trained() -> &'static Trained {
    static CELL: OnceLock<Trained> = OnceLock::new();
    CELL.get_or_init(|| {
        let ds = load_font().unwrap();
        let (ae, _) = train_chosen(&ds);
        Trained { ds, ae }
    })
}

fn codes() -> Array2<f64> {
    let t = trained();
    t.ae.encode(&t.ds.x)
}

#[test]
fn codes_have_shape_32_by_2_and_are_finite() {
    let c = codes();
    assert_eq!(c.dim(), (32, 2));
    assert!(c.iter().all(|v| v.is_finite()));
}

#[test]
fn codes_are_pairwise_distinct() {
    let c = codes();
    let mut min_d = f64::INFINITY;
    for i in 0..32 {
        for j in (i + 1)..32 {
            let d = ((c[[i, 0]] - c[[j, 0]]).powi(2) + (c[[i, 1]] - c[[j, 1]]).powi(2)).sqrt();
            min_d = min_d.min(d);
        }
    }
    assert!(min_d > 1e-6, "distancia minima entre codigos {min_d}");
}

#[test]
fn export_latent_writes_png_and_csv() {
    let t = trained();
    let dir = std::env::temp_dir().join(format!("tp5-latent-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    export_latent(&t.ae, &t.ds, &dir).unwrap();

    let png = std::fs::read(dir.join(LATENT_PNG)).unwrap();
    assert_eq!(&png[..8], &[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]);
    assert!(png.len() > 5000, "png de {} bytes", png.len());

    let csv = std::fs::read_to_string(dir.join(LATENT_CSV)).unwrap();
    let lines: Vec<&str> = csv.lines().collect();
    assert_eq!(lines.len(), 33);
    assert_eq!(lines[0], "label,z1,z2");
    for (line, label) in lines[1..].iter().zip(&t.ds.labels) {
        let first = if line.starts_with('"') {
            // Etiqueta entrecomillada: hasta la comilla de cierre.
            let rest = &line[1..];
            let end = rest.rfind("\",").unwrap();
            rest[..end].replace("\"\"", "\"")
        } else {
            line.split(',').next().unwrap().to_string()
        };
        assert_eq!(&first, label);
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn scatter_rejects_non_finite_points() {
    let dir = std::env::temp_dir().join(format!("tp5-latent-nan-{}", std::process::id()));
    let pts = vec![("a".to_string(), 0.0, 1.0), ("b".to_string(), f64::NAN, 2.0)];
    assert!(plot_latent_scatter(&dir.join("x.png"), &pts).is_err());
    assert!(plot_latent_scatter(&dir.join("x.png"), &[]).is_err());
    let _ = std::fs::remove_dir_all(&dir);
}
