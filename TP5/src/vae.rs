//! Autoencoder variacional (Ej 2, VAE-02): el codificador entrega la media y el log de la varianza
//! de q(z|x) = N(mu, diag(sigma^2)), z se muestrea con el truco de reparametrizacion
//! z = mu + eps * exp(0.5 * logvar), y la perdida es reconstruccion + KL(q(z|x) || N(0, I)).
//! Modulo numerico: no dibuja ni parsea argumentos.

use ndarray::{concatenate, s, Array2, Axis};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

use crate::autoencoder::ArchSpec;
use crate::emoji::{EmojiDataset, EMOJI_PIXELS};
use crate::mlp::{Activation, Grads, Loss, Mlp};
use crate::optimizer::{Optimizer, OptimizerKind};

/// Semilla por defecto de la muestra del prior.
pub const SAMPLE_SEED: u64 = 0x5a3f_0201;

/// Tope de epocas aceptado por `--epochs`.
pub const MAX_EPOCHS: usize = 20000;

/// Etiqueta que se aplica con XOR a la semilla del entrenamiento: el ruido `eps` sale de un flujo
/// independiente del de la inicializacion de los pesos.
const NOISE_STREAM_TAG: u64 = 0x9e37_79b9_7f4a_7c15;

#[derive(Debug, Clone)]
pub struct Vae {
    /// Salida: columnas `0..k` = media, `k..2k` = log-varianza (k = `latent_dim`).
    pub encoder: Mlp,
    pub decoder: Mlp,
    pub latent_dim: usize,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LossTerms {
    pub recon: f64,
    pub kl: f64,
}

impl LossTerms {
    pub fn total(&self) -> f64 {
        self.recon + self.kl
    }
}

#[derive(Debug, Clone)]
pub struct VaeGrads {
    pub enc: Grads,
    pub dec: Grads,
}

/// `mu + eps * exp(0.5 * logvar)` elemento a elemento.
pub fn reparameterize(mu: &Array2<f64>, logvar: &Array2<f64>, eps: &Array2<f64>) -> Array2<f64> {
    mu + &(eps * &logvar.mapv(|lv| (0.5 * lv).exp()))
}

/// Media sobre el batch de la suma por muestra de `0.5 * (exp(logvar) + mu^2 - 1 - logvar)`.
pub fn kl_divergence(mu: &Array2<f64>, logvar: &Array2<f64>) -> f64 {
    let total: f64 = mu.iter().zip(logvar.iter()).map(|(&m, &lv)| 0.5 * (lv.exp() + m * m - 1.0 - lv)).sum();
    total / mu.nrows() as f64
}

/// Normal estandar por Box-Muller a partir de dos uniformes.
pub fn standard_normal(rng: &mut ChaCha8Rng) -> f64 {
    let u1 = 1.0 - rng.gen::<f64>(); // en (0, 1]: el logaritmo nunca ve 0
    let u2: f64 = rng.gen();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

/// Matriz de normales estandar, llenada en orden fila mayor.
pub fn standard_normal_matrix(rng: &mut ChaCha8Rng, rows: usize, cols: usize) -> Array2<f64> {
    let data: Vec<f64> = (0..rows * cols).map(|_| standard_normal(rng)).collect();
    Array2::from_shape_vec((rows, cols), data).expect("forma rows x cols")
}

impl Vae {
    pub fn new(spec: &ArchSpec, n_inputs: usize, rng: &mut ChaCha8Rng) -> Vae {
        assert!(
            spec.latent_act == Activation::Identity,
            "el VAE necesita activacion Identity en la capa latente: la media y el log de la varianza no deben estar acotados"
        );
        let k = spec.latent_dim;
        let mut enc_sizes = vec![n_inputs];
        enc_sizes.extend(&spec.hidden);
        enc_sizes.push(2 * k);
        let mut enc_acts = vec![spec.hidden_act; spec.hidden.len()];
        enc_acts.push(Activation::Identity);

        let mut dec_sizes = vec![k];
        dec_sizes.extend(spec.hidden.iter().rev());
        dec_sizes.push(n_inputs);
        let mut dec_acts = vec![spec.hidden_act; spec.hidden.len()];
        dec_acts.push(spec.output_act);

        let encoder = Mlp::new(&enc_sizes, &enc_acts, rng);
        let decoder = Mlp::new(&dec_sizes, &dec_acts, rng);
        Vae { encoder, decoder, latent_dim: k }
    }

    /// (media, log-varianza) de q(z|x).
    pub fn encode_params(&self, x: &Array2<f64>) -> (Array2<f64>, Array2<f64>) {
        let out = self.encoder.forward(x);
        let k = self.latent_dim;
        (out.slice(s![.., ..k]).to_owned(), out.slice(s![.., k..2 * k]).to_owned())
    }

    pub fn decode(&self, z: &Array2<f64>) -> Array2<f64> {
        self.decoder.forward(z)
    }

    /// Decodifica la media del codificador (sin ruido).
    pub fn reconstruct_mean(&self, x: &Array2<f64>) -> Array2<f64> {
        let (mu, _) = self.encode_params(x);
        self.decode(&mu)
    }

    /// Por ejemplo `encoder 576-128-32-4 | decoder 2-32-128-576`.
    pub fn sizes_label(&self) -> String {
        let join = |sizes: Vec<usize>| sizes.iter().map(usize::to_string).collect::<Vec<_>>().join("-");
        let mut enc = vec![self.encoder.layers[0].w.nrows()];
        enc.extend(self.encoder.layers.iter().map(|l| l.w.ncols()));
        let mut dec = vec![self.decoder.layers[0].w.nrows()];
        dec.extend(self.decoder.layers.iter().map(|l| l.w.ncols()));
        format!("encoder {} | decoder {}", join(enc), join(dec))
    }

    /// Terminos de la perdida y gradientes de todos los parametros con el ruido `eps` dado.
    /// La entrada es tambien el objetivo de la reconstruccion.
    pub fn loss_and_grads(&self, x: &Array2<f64>, eps: &Array2<f64>, recon: Loss) -> (LossTerms, VaeGrads) {
        let out_act = self.decoder.layers.last().expect("decoder vacio").act;
        if let Err(e) = recon.validate(out_act) {
            panic!("combinacion de perdida y activacion de salida invalida: {e}");
        }
        let k = self.latent_dim;
        let batch = x.nrows() as f64;

        let enc_cache = self.encoder.forward_cached(x);
        let params = enc_cache.last();
        let mu = params.slice(s![.., ..k]).to_owned();
        let logvar = params.slice(s![.., k..2 * k]).to_owned();
        let z = reparameterize(&mu, &logvar, eps);
        let dec_cache = self.decoder.forward_cached(&z);
        let pred = dec_cache.last();

        let terms = LossTerms { recon: recon.value(pred, x), kl: kl_divergence(&mu, &logvar) };

        let dec_delta = recon.output_delta(pred, x, out_act);
        let (dec_grads, dz) = self.decoder.backward(&dec_cache, dec_delta);

        // dL/dmu = dz + mu / batch ; dL/dlogvar = dz * eps * 0.5 * exp(0.5 logvar) + 0.5 (exp(logvar) - 1) / batch.
        let d_mu = &dz + &(&mu / batch);
        let d_logvar = &dz * &(eps * &logvar.mapv(|lv| 0.5 * (0.5 * lv).exp()))
            + logvar.mapv(|lv| 0.5 * (lv.exp() - 1.0) / batch);
        // La ultima capa del codificador es Identity: dL/dz = dL/d(salida).
        let enc_delta = concatenate![Axis(1), d_mu, d_logvar];
        let (enc_grads, _) = self.encoder.backward(&enc_cache, enc_delta);

        (terms, VaeGrads { enc: enc_grads, dec: dec_grads })
    }

    /// Un paso de entrenamiento; devuelve los terminos del forward de este paso (antes de actualizar).
    pub fn train_step(
        &mut self,
        x: &Array2<f64>,
        eps: &Array2<f64>,
        recon: Loss,
        opt_enc: &mut dyn Optimizer,
        opt_dec: &mut dyn Optimizer,
    ) -> LossTerms {
        let (terms, grads) = self.loss_and_grads(x, eps, recon);
        opt_enc.step(&mut self.encoder, &grads.enc);
        opt_dec.step(&mut self.decoder, &grads.dec);
        terms
    }
}

#[derive(Debug, Clone)]
pub struct VaeConfig {
    pub name: String,
    pub spec: ArchSpec,
    pub recon: Loss,
    pub optimizer: OptimizerKind,
    pub epochs: usize,
    pub seed: u64,
}

/// Configuracion congelada del Ej 2.
pub fn chosen_vae_config() -> VaeConfig {
    VaeConfig {
        name: "vae-h128x32-l2-s0".to_string(),
        spec: ArchSpec {
            hidden: vec![128, 32],
            latent_dim: 2,
            hidden_act: Activation::Tanh,
            latent_act: Activation::Identity,
            output_act: Activation::Sigmoid,
        },
        recon: Loss::Bce,
        optimizer: OptimizerKind::Adam { lr: 0.003, beta1: 0.9, beta2: 0.999, eps: 1e-8 },
        epochs: 3000,
        seed: 0,
    }
}

#[derive(Debug, Clone)]
pub struct VaeHistory {
    pub recon: Vec<f64>,
    pub kl: Vec<f64>,
}

impl VaeHistory {
    pub fn total(&self, i: usize) -> f64 {
        self.recon[i] + self.kl[i]
    }
}

/// Entrenamiento full-batch sobre el dataset de emojis (entrada = objetivo). Devuelve Err si algun
/// termino de la perdida deja de ser finito.
pub fn train_vae(ds: &EmojiDataset, cfg: &VaeConfig) -> anyhow::Result<(Vae, VaeHistory)> {
    let mut rng = ChaCha8Rng::seed_from_u64(cfg.seed);
    let mut vae = Vae::new(&cfg.spec, EMOJI_PIXELS, &mut rng);
    let mut noise_rng = ChaCha8Rng::seed_from_u64(cfg.seed ^ NOISE_STREAM_TAG);
    let mut opt_enc = cfg.optimizer.build(&vae.encoder);
    let mut opt_dec = cfg.optimizer.build(&vae.decoder);
    let mut hist = VaeHistory { recon: Vec::with_capacity(cfg.epochs), kl: Vec::with_capacity(cfg.epochs) };
    for epoch in 1..=cfg.epochs {
        let eps = standard_normal_matrix(&mut noise_rng, ds.x.nrows(), cfg.spec.latent_dim);
        let t = vae.train_step(&ds.x, &eps, cfg.recon, opt_enc.as_mut(), opt_dec.as_mut());
        if !t.recon.is_finite() || !t.kl.is_finite() {
            anyhow::bail!(
                "la perdida dejo de ser finita en la epoca {epoch} (reconstruccion {}, KL {}); reduzca la tasa de aprendizaje",
                t.recon,
                t.kl
            );
        }
        hist.recon.push(t.recon);
        hist.kl.push(t.kl);
    }
    Ok((vae, hist))
}

/// Entrena `chosen_vae_config()`.
pub fn train_chosen_vae(ds: &EmojiDataset) -> (Vae, VaeHistory) {
    train_vae(ds, &chosen_vae_config()).expect("el entrenamiento de la configuracion elegida diverge")
}

/// Valida el argumento `--epochs`: entre 1 y `MAX_EPOCHS`.
pub fn validate_epochs(n: usize) -> anyhow::Result<usize> {
    if n == 0 || n > MAX_EPOCHS {
        anyhow::bail!("el numero de epocas debe estar entre 1 y {MAX_EPOCHS} (se recibio {n})");
    }
    Ok(n)
}

/// Linea de configuracion que imprime el subcomando `vae`.
pub fn config_line(cfg: &VaeConfig, vae: &Vae) -> String {
    format!(
        "configuracion: {} | arquitectura: {} | optimizador: {} | reconstruccion: {} + KL | epocas: {} | semilla de entrenamiento: {}",
        cfg.name,
        vae.sizes_label(),
        cfg.optimizer.label(),
        match cfg.recon {
            Loss::Bce => "BCE",
            Loss::Mse => "MSE",
        },
        cfg.epochs,
        cfg.seed
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn kl_is_zero_for_the_standard_normal_posterior() {
        let mu = Array2::zeros((3, 2));
        let logvar = Array2::zeros((3, 2));
        assert!(kl_divergence(&mu, &logvar).abs() < 1e-15);
    }

    #[test]
    fn kl_matches_the_closed_form_for_a_hand_example() {
        let mu = array![[1.0, 0.0]];
        let logvar = array![[0.0, 2.0_f64.ln()]];
        let expected = 0.5 * ((1.0 + 1.0 - 1.0 - 0.0) + (2.0 + 0.0 - 1.0 - 2.0_f64.ln()));
        assert!((kl_divergence(&mu, &logvar) - expected).abs() < 1e-12);
    }

    #[test]
    fn reparameterize_with_zero_noise_returns_the_mean() {
        let mu = array![[0.3, -1.2], [2.0, 0.5]];
        let logvar = array![[0.7, -0.4], [1.1, 0.0]];
        let z = reparameterize(&mu, &logvar, &Array2::zeros((2, 2)));
        assert_eq!(z, mu);
    }

    #[test]
    fn reparameterize_scales_noise_by_the_standard_deviation() {
        let mu = array![[1.0]];
        let logvar = array![[4.0_f64.ln()]];
        let eps = array![[1.5]];
        let z = reparameterize(&mu, &logvar, &eps);
        assert!((z[[0, 0]] - (1.0 + 1.5 * 2.0)).abs() < 1e-12);
    }

    #[test]
    fn standard_normal_has_mean_zero_and_unit_variance() {
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        let n = 100_000;
        let draws: Vec<f64> = (0..n).map(|_| standard_normal(&mut rng)).collect();
        let mean = draws.iter().sum::<f64>() / n as f64;
        let var = draws.iter().map(|d| (d - mean) * (d - mean)).sum::<f64>() / n as f64;
        assert!(mean.abs() < 0.02, "media {mean}");
        assert!((var - 1.0).abs() < 0.03, "varianza {var}");
    }

    #[test]
    fn standard_normal_is_deterministic_by_seed() {
        let mut a = ChaCha8Rng::seed_from_u64(3);
        let mut b = ChaCha8Rng::seed_from_u64(3);
        let ma = standard_normal_matrix(&mut a, 4, 3);
        let mb = standard_normal_matrix(&mut b, 4, 3);
        assert!(ma.iter().zip(mb.iter()).all(|(p, q)| p.to_bits() == q.to_bits()));
        let mut c = ChaCha8Rng::seed_from_u64(4);
        assert_ne!(ma, standard_normal_matrix(&mut c, 4, 3));
    }
}
