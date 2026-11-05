//! Autoencoder = encoder + decoder (dos `Mlp`) unidos en una capa latente.

use ndarray::Array2;
use rand_chacha::ChaCha8Rng;

use crate::mlp::{Activation, Grads, Loss, Mlp};
use crate::optimizer::{Optimizer, OptimizerKind};
use crate::report::wrong_pixel_counts;

#[derive(Debug, Clone, PartialEq)]
pub struct ArchSpec {
    /// Tamanos ocultos del encoder; el decoder los refleja en orden inverso.
    pub hidden: Vec<usize>,
    pub latent_dim: usize,
    pub hidden_act: Activation,
    pub latent_act: Activation,
    pub output_act: Activation,
}

#[derive(Debug, Clone)]
pub struct Autoencoder {
    pub encoder: Mlp,
    pub decoder: Mlp,
}

#[derive(Debug, Clone)]
pub struct AeGrads {
    pub enc: Grads,
    pub dec: Grads,
}

impl Autoencoder {
    pub fn new(spec: &ArchSpec, n_inputs: usize, rng: &mut ChaCha8Rng) -> Autoencoder {
        let mut enc_sizes = vec![n_inputs];
        enc_sizes.extend(&spec.hidden);
        enc_sizes.push(spec.latent_dim);
        let mut enc_acts = vec![spec.hidden_act; spec.hidden.len()];
        enc_acts.push(spec.latent_act);

        let mut dec_sizes = vec![spec.latent_dim];
        dec_sizes.extend(spec.hidden.iter().rev());
        dec_sizes.push(n_inputs);
        let mut dec_acts = vec![spec.hidden_act; spec.hidden.len()];
        dec_acts.push(spec.output_act);

        Autoencoder {
            encoder: Mlp::new(&enc_sizes, &enc_acts, rng),
            decoder: Mlp::new(&dec_sizes, &dec_acts, rng),
        }
    }

    pub fn encode(&self, x: &Array2<f64>) -> Array2<f64> {
        self.encoder.forward(x)
    }

    pub fn decode(&self, z: &Array2<f64>) -> Array2<f64> {
        self.decoder.forward(z)
    }

    pub fn reconstruct(&self, x: &Array2<f64>) -> Array2<f64> {
        self.decode(&self.encode(x))
    }

    /// Por ejemplo "35-16-2-16-35".
    pub fn sizes_label(&self) -> String {
        let mut sizes = vec![self.encoder.layers[0].w.nrows()];
        sizes.extend(self.encoder.layers.iter().map(|l| l.w.ncols()));
        sizes.extend(self.decoder.layers.iter().map(|l| l.w.ncols()));
        sizes.iter().map(usize::to_string).collect::<Vec<_>>().join("-")
    }

    /// Valor de la perdida y gradientes de todos los parametros (encoder y decoder).
    pub fn loss_and_grads(&self, x: &Array2<f64>, target: &Array2<f64>, loss: Loss) -> (f64, AeGrads) {
        let out_act = self.decoder.layers.last().expect("decoder vacio").act;
        if let Err(e) = loss.validate(out_act) {
            panic!("combinacion de perdida y activacion de salida invalida: {e}");
        }
        let enc_cache = self.encoder.forward_cached(x);
        let dec_cache = self.decoder.forward_cached(enc_cache.last());
        let pred = dec_cache.last();
        let value = loss.value(pred, target);

        let dec_delta = loss.output_delta(pred, target, out_act);
        let (dec_grads, grad_latent) = self.decoder.backward(&dec_cache, dec_delta);

        // dL/dz de la capa latente = dL/da_latente * f'(a_latente).
        let latent_act = self.encoder.layers.last().expect("encoder vacio").act;
        let enc_delta = grad_latent * latent_act.derivative_from_output(enc_cache.last());
        let (enc_grads, _) = self.encoder.backward(&enc_cache, enc_delta);

        (value, AeGrads { enc: enc_grads, dec: dec_grads })
    }

    /// Un paso de entrenamiento; devuelve la perdida del forward de este paso (antes de actualizar).
    pub fn train_step(
        &mut self,
        x: &Array2<f64>,
        target: &Array2<f64>,
        loss: Loss,
        opt_enc: &mut dyn Optimizer,
        opt_dec: &mut dyn Optimizer,
    ) -> f64 {
        let (value, grads) = self.loss_and_grads(x, target, loss);
        opt_enc.step(&mut self.encoder, &grads.enc);
        opt_dec.step(&mut self.decoder, &grads.dec);
        value
    }
}

#[derive(Debug, Clone)]
pub struct TrainConfig {
    pub epochs: usize,
    /// Se evalua (pixeles incorrectos) cada `eval_every` epocas y en la ultima.
    pub eval_every: usize,
    pub tolerance_px: usize,
    /// Corta en la primera evaluacion dentro de tolerancia.
    pub early_stop: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EvalPoint {
    pub epoch: usize,
    pub max_wrong: usize,
    pub n_within: usize,
}

#[derive(Debug, Clone)]
pub struct TrainHistory {
    pub loss: Vec<f64>,
    pub evals: Vec<EvalPoint>,
    pub first_epoch_within: Option<usize>,
    pub epochs_run: usize,
    pub final_loss: f64,
}

/// Entrenamiento full-batch. Las evaluaciones se toman con los parametros ya actualizados.
pub fn train(
    ae: &mut Autoencoder,
    input: &Array2<f64>,
    target: &Array2<f64>,
    loss: Loss,
    optimizer: &OptimizerKind,
    cfg: &TrainConfig,
) -> TrainHistory {
    let mut opt_enc = optimizer.build(&ae.encoder);
    let mut opt_dec = optimizer.build(&ae.decoder);
    let eval_every = cfg.eval_every.max(1);
    let mut hist = TrainHistory {
        loss: Vec::with_capacity(cfg.epochs),
        evals: Vec::new(),
        first_epoch_within: None,
        epochs_run: 0,
        final_loss: f64::NAN,
    };

    for epoch in 1..=cfg.epochs {
        let l = ae.train_step(input, target, loss, opt_enc.as_mut(), opt_dec.as_mut());
        hist.loss.push(l);
        hist.epochs_run = epoch;

        if epoch % eval_every == 0 || epoch == cfg.epochs {
            let counts = wrong_pixel_counts(target, &ae.reconstruct(input));
            let max_wrong = counts.iter().copied().max().unwrap_or(0);
            let n_within = counts.iter().filter(|&&c| c <= cfg.tolerance_px).count();
            hist.evals.push(EvalPoint { epoch, max_wrong, n_within });
            let within = max_wrong <= cfg.tolerance_px;
            if within && hist.first_epoch_within.is_none() {
                hist.first_epoch_within = Some(epoch);
            }
            if within && cfg.early_stop {
                break;
            }
        }
    }

    hist.final_loss = loss.value(&ae.reconstruct(input), target);
    hist
}
