//! Denoising Autoencoder (Ej 1b): trazador, determinismo y garantias de la evaluacion.

use tp5::autoencoder::Autoencoder;
use tp5::data::load_font;
use tp5::denoising::{
    base_dae_config, evaluate_level, train_dae, DaeConfig, N_EVAL_SAMPLES, TEST_NOISE_SEED,
};
use tp5::mlp::Mlp;

fn assert_same_net(a: &Mlp, b: &Mlp) {
    assert_eq!(a.layers.len(), b.layers.len());
    for (la, lb) in a.layers.iter().zip(&b.layers) {
        assert!(la.w.iter().zip(lb.w.iter()).all(|(x, y)| x.to_bits() == y.to_bits()), "pesos distintos");
        assert!(la.b.iter().zip(lb.b.iter()).all(|(x, y)| x.to_bits() == y.to_bits()), "sesgos distintos");
    }
}

fn same_weights(a: &Autoencoder, b: &Autoencoder) -> bool {
    let eq = |x: &Mlp, y: &Mlp| {
        x.layers.iter().zip(&y.layers).all(|(la, lb)| {
            la.w.iter().zip(lb.w.iter()).all(|(p, q)| p.to_bits() == q.to_bits())
                && la.b.iter().zip(lb.b.iter()).all(|(p, q)| p.to_bits() == q.to_bits())
        })
    };
    eq(&a.encoder, &b.encoder) && eq(&a.decoder, &b.decoder)
}

fn short_cfg(epochs: usize) -> DaeConfig {
    let mut cfg = base_dae_config();
    cfg.epochs = epochs;
    cfg.copies = 2;
    cfg
}

#[test]
fn dae_removes_noise_at_level_0_1() {
    let ds = load_font().unwrap();
    let mut cfg = base_dae_config();
    cfg.epochs = 1000;
    cfg.copies = 4;
    let (ae, _) = train_dae(&ds, &cfg);
    let s = evaluate_level(&ae, &ds.x, 0.10, N_EVAL_SAMPLES, TEST_NOISE_SEED);
    assert!(
        s.output_mean_wrong < s.input_mean_wrong / 2.0,
        "salida {} no es menos de la mitad de la entrada {}",
        s.output_mean_wrong,
        s.input_mean_wrong
    );
    assert!(
        (s.input_mean_wrong - 3.5).abs() < 0.35,
        "la entrada deberia tener cerca de 35 * 0.10 pixeles incorrectos, tiene {}",
        s.input_mean_wrong
    );
}

#[test]
fn evaluation_is_deterministic() {
    let ds = load_font().unwrap();
    let (ae, _) = train_dae(&ds, &short_cfg(20));
    let a = evaluate_level(&ae, &ds.x, 0.20, 20, TEST_NOISE_SEED);
    let b = evaluate_level(&ae, &ds.x, 0.20, 20, TEST_NOISE_SEED);
    assert_eq!(a, b);
    assert_eq!(a.input_mean_wrong.to_bits(), b.input_mean_wrong.to_bits());
    let c = evaluate_level(&ae, &ds.x, 0.20, 20, TEST_NOISE_SEED ^ 1);
    assert_ne!(a.input_mean_wrong, c.input_mean_wrong);
}

#[test]
fn train_dae_is_deterministic_by_seed() {
    let ds = load_font().unwrap();
    let cfg = short_cfg(50);
    let (a, ha) = train_dae(&ds, &cfg);
    let (b, hb) = train_dae(&ds, &cfg);
    assert_same_net(&a.encoder, &b.encoder);
    assert_same_net(&a.decoder, &b.decoder);
    assert_eq!(ha.loss.len(), 50);
    assert!(ha.loss.iter().zip(&hb.loss).all(|(x, y)| x.to_bits() == y.to_bits()));

    let mut other = cfg.clone();
    other.seed = 1;
    let (c, _) = train_dae(&ds, &other);
    assert!(!same_weights(&a, &c), "otra semilla deberia dar otros pesos");
}
