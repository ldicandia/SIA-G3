//! Formulas de los optimizadores e historial de `train`.

use ndarray::{array, s, Array2};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use tp5::autoencoder::{train, ArchSpec, Autoencoder, TrainConfig};
use tp5::data::{load_font, N_PIXELS};
use tp5::mlp::{Activation, Grads, Loss, Mlp};
use tp5::optimizer::OptimizerKind;

fn single_layer() -> (Mlp, Grads) {
    let mut rng = ChaCha8Rng::seed_from_u64(7);
    let net = Mlp::new(&[3, 2], &[Activation::Identity], &mut rng);
    let grads = Grads {
        dw: vec![array![[0.5, -0.25], [1.0, 2.0], [-1.5, 0.125]]],
        db: vec![array![0.3, -0.7]],
    };
    (net, grads)
}

fn assert_close(a: &Array2<f64>, b: &Array2<f64>, tol: f64) {
    for (x, y) in a.iter().zip(b.iter()) {
        assert!((x - y).abs() < tol, "{x} vs {y}");
    }
}

#[test]
fn sgd_step_matches_formula() {
    let (mut net, g) = single_layer();
    let (w0, b0) = (net.layers[0].w.clone(), net.layers[0].b.clone());
    let lr = 0.37;
    let mut opt = OptimizerKind::Sgd { lr }.build(&net);
    opt.step(&mut net, &g);
    assert_close(&net.layers[0].w, &(&w0 - &(&g.dw[0] * lr)), 1e-12);
    let expected_b = &b0 - &(&g.db[0] * lr);
    for (x, y) in net.layers[0].b.iter().zip(expected_b.iter()) {
        assert!((x - y).abs() < 1e-12);
    }
}

#[test]
fn momentum_two_steps_with_constant_gradient() {
    let (mut net, g) = single_layer();
    let w0 = net.layers[0].w.clone();
    let (lr, beta) = (0.1, 0.9);
    let mut opt = OptimizerKind::Momentum { lr, beta }.build(&net);
    opt.step(&mut net, &g);
    opt.step(&mut net, &g);
    // w0 - lr * (g + (1 + beta) * g)
    let expected = &w0 - &(&g.dw[0] * (lr * (1.0 + (1.0 + beta))));
    assert_close(&net.layers[0].w, &expected, 1e-12);
}

#[test]
fn adam_first_step_is_lr_sized() {
    let (mut net, g) = single_layer();
    let w0 = net.layers[0].w.clone();
    let lr = 0.01;
    let mut opt = OptimizerKind::Adam { lr, beta1: 0.9, beta2: 0.999, eps: 1e-8 }.build(&net);
    opt.step(&mut net, &g);
    for ((w, w0), gv) in net.layers[0].w.iter().zip(w0.iter()).zip(g.dw[0].iter()) {
        assert!(gv.abs() >= 1e-3);
        assert!(((w0 - w).abs() - lr).abs() < 1e-6, "paso {} vs lr {lr}", (w0 - w).abs());
        assert_eq!((w0 - w).signum(), gv.signum());
    }
}

fn toy() -> (Autoencoder, Array2<f64>) {
    let ds = load_font().unwrap();
    let x = ds.x.slice(s![0..4, ..]).to_owned();
    let mut rng = ChaCha8Rng::seed_from_u64(0);
    let spec = ArchSpec {
        hidden: vec![8],
        latent_dim: 2,
        hidden_act: Activation::Tanh,
        latent_act: Activation::Identity,
        output_act: Activation::Sigmoid,
    };
    (Autoencoder::new(&spec, N_PIXELS, &mut rng), x)
}

fn adam() -> OptimizerKind {
    OptimizerKind::Adam { lr: 0.01, beta1: 0.9, beta2: 0.999, eps: 1e-8 }
}

#[test]
fn train_records_history() {
    let (mut ae, x) = toy();
    let cfg = TrainConfig { epochs: 120, eval_every: 20, tolerance_px: 1, early_stop: false };
    let h = train(&mut ae, &x, &x, Loss::Bce, &adam(), &cfg);
    assert_eq!(h.loss.len(), 120);
    assert_eq!(h.epochs_run, 120);
    let epochs: Vec<usize> = h.evals.iter().map(|e| e.epoch).collect();
    assert_eq!(epochs, vec![20, 40, 60, 80, 100, 120]);
    assert!(h.loss.last().unwrap() < h.loss.first().unwrap());
    assert!(h.final_loss.is_finite());
}

#[test]
fn early_stop_halts_at_first_eval() {
    let (mut ae, x) = toy();
    let cfg = TrainConfig { epochs: 120, eval_every: 20, tolerance_px: 35, early_stop: true };
    let h = train(&mut ae, &x, &x, Loss::Bce, &adam(), &cfg);
    assert_eq!(h.epochs_run, 20);
    assert_eq!(h.first_epoch_within, Some(20));
    assert_eq!(h.loss.len(), 20);
}

#[test]
fn each_optimizer_lowers_the_loss() {
    let kinds = [
        OptimizerKind::Sgd { lr: 0.5 },
        OptimizerKind::Momentum { lr: 0.1, beta: 0.9 },
        adam(),
    ];
    for kind in kinds {
        let (mut ae, x) = toy();
        let cfg = TrainConfig { epochs: 100, eval_every: 50, tolerance_px: 1, early_stop: false };
        let h = train(&mut ae, &x, &x, Loss::Mse, &kind, &cfg);
        assert!(h.final_loss < h.loss[0], "{}: {} !< {}", kind.label(), h.final_loss, h.loss[0]);
    }
}

#[test]
fn labels_have_expected_format() {
    assert!(OptimizerKind::Sgd { lr: 0.5 }.label().starts_with("SGD("));
    assert_eq!(OptimizerKind::Momentum { lr: 0.1, beta: 0.9 }.label(), "Momentum(lr=0.1,beta=0.9)");
    assert_eq!(adam().label(), "Adam(lr=0.01)");
}
