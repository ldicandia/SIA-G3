//! Denoising Autoencoder (Ej 1b): entrada ruidosa, objetivo limpio.
//!
//! Se alimenta a la red con la instancia distorsionada y se pide como salida la original sin
//! alterar (teoria de DAE del curso). Este modulo es numerico: no dibuja ni parsea la linea de
//! comandos.

use ndarray::Array2;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use crate::autoencoder::{ArchSpec, Autoencoder};
use crate::data::{Dataset, N_PIXELS};
use crate::mlp::{Activation, Loss};
use crate::noise::{distort, level_rng, tile_rows, NoiseSchedule, TrainNoiseStream};
use crate::optimizer::OptimizerKind;
use crate::report::{wrong_pixel_counts, TOLERANCE_PX};

/// Semilla del ruido de evaluacion, independiente del ruido de entrenamiento.
pub const TEST_NOISE_SEED: u64 = 0x7e57_0001;

/// Distorsiones sorteadas por patron y por nivel en la evaluacion.
pub const N_EVAL_SAMPLES: usize = 100;

/// Niveles evaluados por defecto.
pub const DEFAULT_LEVELS: [f64; 8] = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50];

/// Fraccion minima de distorsiones no vistas que debe quedar dentro de la tolerancia para decir
/// que un nivel se elimina.
pub const ROBUST_FRACTION: f64 = 0.90;

/// Constante que separa el flujo de ruido de entrenamiento de cualquier flujo de evaluacion.
const TRAIN_NOISE_STREAM_TAG: u64 = 0x7241_1100;

/// Semilla del flujo de ruido de entrenamiento derivada de la semilla de la configuracion.
pub fn train_noise_seed(seed: u64) -> u64 {
    seed ^ TRAIN_NOISE_STREAM_TAG
}

#[derive(Debug, Clone)]
pub struct DaeConfig {
    pub name: String,
    pub study: String,
    pub spec: ArchSpec,
    pub loss: Loss,
    pub optimizer: OptimizerKind,
    pub train_noise: NoiseSchedule,
    pub epochs: usize,
    /// Copias de los 32 patrones por epoca (cada copia recibe ruido distinto).
    pub copies: usize,
    pub seed: u64,
}

impl DaeConfig {
    /// El nombre sin el sufijo `-s{semilla}`.
    pub fn variant_key(&self) -> String {
        match self.name.rfind("-s") {
            Some(i) if self.name[i + 2..].chars().all(|c| c.is_ascii_digit()) && i + 2 < self.name.len() => {
                self.name[..i].to_string()
            }
            _ => self.name.clone(),
        }
    }
}

/// Configuracion base del denoising: la receta BCE + Adam que convergio en el Ej 1a, con un
/// cuello de 8 dimensiones (la consigna no impone 2 dimensiones al denoising).
pub fn base_dae_config() -> DaeConfig {
    DaeConfig {
        name: "base-dae-s0".to_string(),
        study: "base".to_string(),
        spec: ArchSpec {
            hidden: vec![64, 32],
            latent_dim: 8,
            hidden_act: Activation::Tanh,
            latent_act: Activation::Identity,
            output_act: Activation::Sigmoid,
        },
        loss: Loss::Bce,
        optimizer: OptimizerKind::Adam { lr: 0.005, beta1: 0.9, beta2: 0.999, eps: 1e-8 },
        train_noise: NoiseSchedule::Uniform { lo: 0.0, hi: 0.2 },
        epochs: 3000,
        copies: 8,
        seed: 0,
    }
}

#[derive(Debug, Clone)]
pub struct DaeHistory {
    /// Perdida de cada epoca sobre el lote ruidoso de esa epoca.
    pub loss: Vec<f64>,
    /// Promedio de las ultimas `min(100, epocas)` perdidas.
    pub tail_loss: f64,
}

/// Entrenamiento full-batch: cada epoca usa ruido nuevo como entrada y los patrones limpios como
/// objetivo.
pub fn train_denoising(ae: &mut Autoencoder, clean: &Array2<f64>, cfg: &DaeConfig) -> DaeHistory {
    let mut opt_enc = cfg.optimizer.build(&ae.encoder);
    let mut opt_dec = cfg.optimizer.build(&ae.decoder);
    let target = tile_rows(clean, cfg.copies);
    let mut stream = TrainNoiseStream::new(cfg.train_noise, train_noise_seed(cfg.seed));

    let mut loss = Vec::with_capacity(cfg.epochs);
    for _ in 0..cfg.epochs {
        let noisy = stream.next_batch(&target);
        loss.push(ae.train_step(&noisy, &target, cfg.loss, opt_enc.as_mut(), opt_dec.as_mut()));
    }
    let tail = &loss[loss.len().saturating_sub(100)..];
    let tail_loss = if tail.is_empty() { f64::NAN } else { tail.iter().sum::<f64>() / tail.len() as f64 };
    DaeHistory { loss, tail_loss }
}

/// Construye el autoencoder igual que `experiments::run_experiment` y lo entrena. El tamano
/// latente no se fuerza a 2.
pub fn train_dae(ds: &Dataset, cfg: &DaeConfig) -> (Autoencoder, DaeHistory) {
    let mut rng = ChaCha8Rng::seed_from_u64(cfg.seed);
    let mut ae = Autoencoder::new(&cfg.spec, N_PIXELS, &mut rng);
    let history = train_denoising(&mut ae, &ds.x, cfg);
    (ae, history)
}

/// Estadisticas de un nivel de ruido sobre distorsiones no vistas, medidas contra el patron limpio.
#[derive(Debug, Clone, PartialEq)]
pub struct LevelStats {
    pub level: f64,
    pub n_samples: usize,
    pub input_mean_wrong: f64,
    pub output_mean_wrong: f64,
    pub output_max_wrong: usize,
    pub input_frac_within: f64,
    pub frac_within: f64,
}

/// Evalua un nivel: `samples_per_pattern` distorsiones de cada patron con el generador
/// `level_rng(seed, level)`, reconstruccion y conteo de pixeles incorrectos contra los limpios.
pub fn evaluate_level(
    ae: &Autoencoder,
    clean: &Array2<f64>,
    level: f64,
    samples_per_pattern: usize,
    seed: u64,
) -> LevelStats {
    let target = tile_rows(clean, samples_per_pattern);
    let noisy = distort(&target, level, &mut level_rng(seed, level));
    let output = ae.reconstruct(&noisy);
    let in_counts = wrong_pixel_counts(&target, &noisy);
    let out_counts = wrong_pixel_counts(&target, &output);
    let n = out_counts.len();
    let mean = |c: &[usize]| c.iter().sum::<usize>() as f64 / n as f64;
    let frac = |c: &[usize]| c.iter().filter(|&&k| k <= TOLERANCE_PX).count() as f64 / n as f64;
    LevelStats {
        level,
        n_samples: n,
        input_mean_wrong: mean(&in_counts),
        output_mean_wrong: mean(&out_counts),
        output_max_wrong: out_counts.iter().copied().max().unwrap_or(0),
        input_frac_within: frac(&in_counts),
        frac_within: frac(&out_counts),
    }
}

/// Evalua cada nivel, en el orden dado.
pub fn evaluate_levels(
    ae: &Autoencoder,
    clean: &Array2<f64>,
    levels: &[f64],
    samples_per_pattern: usize,
    seed: u64,
) -> Vec<LevelStats> {
    levels.iter().map(|&l| evaluate_level(ae, clean, l, samples_per_pattern, seed)).collect()
}

/// Mayor nivel tal que ese y todos los menores tienen `frac_within >= ROBUST_FRACTION`.
/// Un nivel mayor que se recupera despues de una falla no cuenta.
pub fn max_removed_level(stats: &[LevelStats]) -> Option<f64> {
    let mut sorted: Vec<&LevelStats> = stats.iter().collect();
    sorted.sort_by(|a, b| a.level.total_cmp(&b.level));
    let mut best = None;
    for s in sorted {
        if s.frac_within >= ROBUST_FRACTION {
            best = Some(s.level);
        } else {
            break;
        }
    }
    best
}

/// Linea de titular `NIVEL MAXIMO ELIMINADO: ...`.
pub fn max_level_line(stats: &[LevelStats]) -> String {
    let pct = ROBUST_FRACTION * 100.0;
    match max_removed_level(stats) {
        Some(p) => format!(
            "NIVEL MAXIMO ELIMINADO: {p:.2} (a ese nivel y a todos los menores al menos el {pct:.0} % de las distorsiones no vistas queda con a lo sumo {TOLERANCE_PX} pixel incorrecto)"
        ),
        None => format!(
            "NIVEL MAXIMO ELIMINADO: ninguno (ni el primer nivel evaluado alcanza el {pct:.0} % de distorsiones no vistas con a lo sumo {TOLERANCE_PX} pixel incorrecto)"
        ),
    }
}

/// Tabla por nivel: entrada ruidosa vs salida del autoencoder, contra el patron limpio.
pub fn render_level_table(stats: &[LevelStats]) -> String {
    let mut out = String::from("nivel  entrada_px  salida_px  max_px  dentro_1px_%");
    for s in stats {
        out.push_str(&format!(
            "\n{:>5.2}  {:>10.3}  {:>9.3}  {:>6}  {:>12.1}",
            s.level,
            s.input_mean_wrong,
            s.output_mean_wrong,
            s.output_max_wrong,
            s.frac_within * 100.0
        ));
    }
    out
}
