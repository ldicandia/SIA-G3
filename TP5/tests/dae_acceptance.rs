//! Aceptacion del DAE elegido: la constante congelada coincide con la regla sobre el sweep
//! completo y la red elegida elimina ruido sobre distorsiones que no vio ni al entrenar ni al
//! seleccionar.

use std::sync::OnceLock;

use tp5::autoencoder::Autoencoder;
use tp5::dae_experiments::{
    chosen_dae_config, dae_sweep_grid, run_dae_sweep, select_dae_variant, train_chosen_dae, DaeResult,
};
use tp5::data::{load_font, Dataset};
use tp5::denoising::{evaluate_levels, LevelStats, DEFAULT_LEVELS, N_EVAL_SAMPLES, TEST_NOISE_SEED};
use tp5::mlp::Mlp;

fn dataset() -> &'static Dataset {
    static DS: OnceLock<Dataset> = OnceLock::new();
    DS.get_or_init(|| load_font().unwrap())
}

fn full_sweep() -> &'static Vec<DaeResult> {
    static SWEEP: OnceLock<Vec<DaeResult>> = OnceLock::new();
    SWEEP.get_or_init(|| run_dae_sweep(dataset(), &dae_sweep_grid(false), 8))
}

fn chosen_net() -> &'static Autoencoder {
    static NET: OnceLock<Autoencoder> = OnceLock::new();
    NET.get_or_init(|| train_chosen_dae(dataset()).0)
}

fn test_stats() -> &'static Vec<LevelStats> {
    static STATS: OnceLock<Vec<LevelStats>> = OnceLock::new();
    STATS.get_or_init(|| {
        evaluate_levels(chosen_net(), &dataset().x, &DEFAULT_LEVELS, N_EVAL_SAMPLES, TEST_NOISE_SEED)
    })
}

fn same_mlp(a: &Mlp, b: &Mlp) -> bool {
    a.layers.len() == b.layers.len()
        && a.layers.iter().zip(&b.layers).all(|(la, lb)| {
            la.w.iter().zip(lb.w.iter()).all(|(p, q)| p.to_bits() == q.to_bits())
                && la.b.iter().zip(lb.b.iter()).all(|(p, q)| p.to_bits() == q.to_bits())
        })
}

#[test]
fn chosen_dae_config_is_the_selection_rule_winner() {
    let sel = select_dae_variant(full_sweep()).expect("alguna variante debe ser elegible en la grilla completa");
    assert_eq!(sel.name, chosen_dae_config().name);
}

#[test]
fn chosen_dae_config_is_a_full_grid_entry() {
    let chosen = chosen_dae_config();
    let grid = dae_sweep_grid(false);
    let entry = grid.iter().find(|c| c.name == chosen.name).expect("la elegida no esta en la grilla");
    assert_eq!(entry.spec, chosen.spec);
    assert_eq!(entry.loss, chosen.loss);
    assert_eq!(entry.optimizer, chosen.optimizer);
    assert_eq!(entry.train_noise, chosen.train_noise);
    assert_eq!(entry.epochs, chosen.epochs);
    assert_eq!(entry.copies, chosen.copies);
    assert_eq!(entry.seed, chosen.seed);
}

#[test]
fn chosen_dae_removes_noise_on_unseen_draws() {
    for s in test_stats().iter().filter(|s| s.level > 0.0 && s.level <= 0.30) {
        assert!(
            s.output_mean_wrong < s.input_mean_wrong,
            "nivel {}: salida {} no es menor que entrada {}",
            s.level,
            s.output_mean_wrong,
            s.input_mean_wrong
        );
    }
    let at = test_stats().iter().find(|s| s.level == 0.10).expect("falta el nivel 0.10");
    assert!(
        at.output_mean_wrong < at.input_mean_wrong / 2.0,
        "a 0.10 la salida {} no es menor que la mitad de la entrada {}",
        at.output_mean_wrong,
        at.input_mean_wrong
    );
}

#[test]
fn chosen_dae_reconstructs_clean_patterns_at_level_zero() {
    let s = test_stats().iter().find(|s| s.level == 0.0).expect("falta el nivel 0.0");
    assert_eq!(s.frac_within, 1.0);
    assert!(s.output_max_wrong <= 1, "peor patron limpio con {} pixeles incorrectos", s.output_max_wrong);
}

#[test]
fn chosen_dae_training_is_deterministic() {
    let (again, _) = train_chosen_dae(dataset());
    let first = chosen_net();
    assert!(same_mlp(&first.encoder, &again.encoder), "pesos del encoder distintos");
    assert!(same_mlp(&first.decoder, &again.decoder), "pesos del decoder distintos");
}
