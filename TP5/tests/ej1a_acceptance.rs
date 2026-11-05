//! Criterio de aceptacion del Ej 1a: latente de 2 dimensiones y a lo sumo 1 pixel incorrecto
//! en cada uno de los 32 patrones, verificado con una reevaluacion independiente.

use tp5::data::load_font;
use tp5::experiments::{chosen_config, sweep_grid, train_chosen};
use tp5::report::{wrong_pixel_counts, TOLERANCE_PX};

#[test]
fn chosen_config_uses_two_dimensional_latent() {
    let ds = load_font().unwrap();
    let cfg = chosen_config();
    assert_eq!(cfg.spec.latent_dim, 2);
    let (ae, _) = train_chosen(&ds);
    assert_eq!(ae.encode(&ds.x).ncols(), 2);
}

#[test]
fn chosen_config_matches_a_sweep_variant() {
    let c = chosen_config();
    let found = sweep_grid(false).into_iter().any(|g| {
        g.name == c.name
            && g.spec == c.spec
            && g.loss == c.loss
            && g.optimizer == c.optimizer
            && g.epochs == c.epochs
            && g.seed == c.seed
    });
    assert!(found, "{} no esta en sweep_grid(false)", c.name);
}

#[test]
fn chosen_config_learns_all_32_patterns_within_one_pixel() {
    let ds = load_font().unwrap();
    let (ae, res) = train_chosen(&ds);
    // Reevaluacion independiente del historial y del reporte de la corrida.
    let counts = wrong_pixel_counts(&ds.x, &ae.reconstruct(&ds.x));
    assert_eq!(counts.len(), 32);
    for (label, c) in ds.labels.iter().zip(&counts) {
        assert!(*c <= TOLERANCE_PX, "el patron '{label}' tiene {c} pixeles incorrectos");
    }
    assert_eq!(res.final_max_wrong, *counts.iter().max().unwrap());
}

#[test]
fn training_is_deterministic() {
    let ds = load_font().unwrap();
    let (_, a) = train_chosen(&ds);
    let (_, b) = train_chosen(&ds);
    assert_eq!(a.final_loss.to_bits(), b.final_loss.to_bits());
    let ca: Vec<usize> = a.report.patterns.iter().map(|p| p.wrong_pixels).collect();
    let cb: Vec<usize> = b.report.patterns.iter().map(|p| p.wrong_pixels).collect();
    assert_eq!(ca, cb);
}
