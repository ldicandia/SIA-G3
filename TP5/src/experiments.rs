//! Experimentos del Ej 1a: configuraciones, grilla del sweep, corridas en paralelo y escritores
//! de CSV / markdown. Este modulo no dibuja nada (los graficos viven en `plots`).

use std::fmt::Write as _;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use anyhow::{Context, Result};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use crate::autoencoder::{train, ArchSpec, Autoencoder, TrainConfig, TrainHistory};
use crate::data::{Dataset, N_PIXELS};
use crate::mlp::{Activation, Loss};
use crate::optimizer::OptimizerKind;
use crate::report::{wrong_pixel_counts, Report, TOLERANCE_PX};

/// Dimension del espacio latente exigida por la consigna (AE-01).
pub const LATENT_DIM: usize = 2;

/// Orden fijo de los estudios en CSV, markdown y graficos.
pub const STUDIES: [&str; 4] = ["arch", "optim", "act", "loss"];

#[derive(Debug, Clone)]
pub struct ExperimentConfig {
    pub name: String,
    pub study: String,
    pub spec: ArchSpec,
    pub loss: Loss,
    pub optimizer: OptimizerKind,
    pub epochs: usize,
    pub eval_every: usize,
    pub seed: u64,
}

impl ExperimentConfig {
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

#[derive(Debug, Clone)]
pub struct ExperimentResult {
    pub config: ExperimentConfig,
    pub history: TrainHistory,
    pub report: Report,
    pub final_loss: f64,
    pub final_max_wrong: usize,
    pub final_n_within: usize,
    pub first_epoch_within: Option<usize>,
}

pub fn loss_label(loss: &Loss) -> &'static str {
    match loss {
        Loss::Mse => "MSE",
        Loss::Bce => "BCE",
    }
}

/// Entrena un autoencoder con la configuracion dada. La evaluacion final se hace sobre los
/// parametros finales (independiente del historial).
pub fn run_experiment(ds: &Dataset, cfg: &ExperimentConfig) -> (Autoencoder, ExperimentResult) {
    assert_eq!(cfg.spec.latent_dim, LATENT_DIM, "el espacio latente de este ejercicio es de 2 dimensiones");
    let mut rng = ChaCha8Rng::seed_from_u64(cfg.seed);
    let mut ae = Autoencoder::new(&cfg.spec, N_PIXELS, &mut rng);
    let tcfg = TrainConfig {
        epochs: cfg.epochs,
        eval_every: cfg.eval_every,
        tolerance_px: TOLERANCE_PX,
        early_stop: false,
    };
    let history = train(&mut ae, &ds.x, &ds.x, cfg.loss, &cfg.optimizer, &tcfg);
    let recon = ae.reconstruct(&ds.x);
    let report = Report::new(&ds.labels, &wrong_pixel_counts(&ds.x, &recon));
    let result = ExperimentResult {
        config: cfg.clone(),
        final_loss: history.final_loss,
        final_max_wrong: report.max_wrong(),
        final_n_within: report.n_within(TOLERANCE_PX),
        first_epoch_within: history.first_epoch_within,
        history,
        report,
    };
    (ae, result)
}

// ---------------------------------------------------------------------------------------------
// Grilla
// ---------------------------------------------------------------------------------------------

fn adam(lr: f64) -> OptimizerKind {
    OptimizerKind::Adam { lr, beta1: 0.9, beta2: 0.999, eps: 1e-8 }
}

fn spec(hidden: &[usize], hidden_act: Activation, latent_act: Activation) -> ArchSpec {
    ArchSpec {
        hidden: hidden.to_vec(),
        latent_dim: LATENT_DIM,
        hidden_act,
        latent_act,
        output_act: Activation::Sigmoid,
    }
}

fn make(
    variant: &str,
    spec: ArchSpec,
    loss: Loss,
    optimizer: OptimizerKind,
    epochs: usize,
    seeds: &[u64],
) -> Vec<ExperimentConfig> {
    let study = variant.split('-').next().unwrap_or(variant).to_string();
    seeds
        .iter()
        .map(|&seed| ExperimentConfig {
            name: format!("{variant}-s{seed}"),
            study: study.clone(),
            spec: spec.clone(),
            loss,
            optimizer: optimizer.clone(),
            epochs,
            eval_every: 25,
            seed,
        })
        .collect()
}

/// Grilla del sweep: 18 variantes (un factor por estudio respecto de la base) por semilla.
/// Completa: 10000 epocas, semillas 0, 1, 2 (54 corridas). Rapida: 300 epocas, semilla 0 (18).
pub fn sweep_grid(quick: bool) -> Vec<ExperimentConfig> {
    let (epochs, seeds): (usize, Vec<u64>) = if quick { (300, vec![0]) } else { (10_000, vec![0, 1, 2]) };
    let (t, id, tl) = (Activation::Tanh, Activation::Identity, Activation::Tanh);
    let r = Activation::Relu;
    let base = |hidden: &[usize]| spec(hidden, t, id);
    let mut g = Vec::new();

    // Estudio arquitectura: Adam 0.005, BCE, tanh, latente identidad.
    for (key, hidden) in [
        ("arch-h24", vec![24]),
        ("arch-h20x10", vec![20, 10]),
        ("arch-h32x16", vec![32, 16]),
        ("arch-h64x32", vec![64, 32]),
        ("arch-h48x24x12", vec![48, 24, 12]),
    ] {
        g.extend(make(key, base(&hidden), Loss::Bce, adam(0.005), epochs, &seeds));
    }

    // Estudio optimizador: hidden [20,10], BCE, tanh, latente identidad.
    for (key, opt) in [
        ("optim-sgd-lr0.5", OptimizerKind::Sgd { lr: 0.5 }),
        ("optim-sgd-lr0.1", OptimizerKind::Sgd { lr: 0.1 }),
        ("optim-mom-lr0.1-b0.9", OptimizerKind::Momentum { lr: 0.1, beta: 0.9 }),
        ("optim-mom-lr0.01-b0.9", OptimizerKind::Momentum { lr: 0.01, beta: 0.9 }),
        ("optim-adam-lr0.001", adam(0.001)),
        ("optim-adam-lr0.005", adam(0.005)),
        ("optim-adam-lr0.01", adam(0.01)),
    ] {
        g.extend(make(key, base(&[20, 10]), Loss::Bce, opt, epochs, &seeds));
    }

    // Estudio activaciones: hidden [20,10], Adam 0.005, BCE.
    for (key, hidden_act, latent_act) in [
        ("act-tanh-latid", t, id),
        ("act-tanh-lattanh", t, tl),
        ("act-relu-latid", r, id),
        ("act-relu-lattanh", r, tl),
    ] {
        g.extend(make(key, spec(&[20, 10], hidden_act, latent_act), Loss::Bce, adam(0.005), epochs, &seeds));
    }

    // Estudio perdida: hidden [20,10], Adam 0.005, tanh, latente identidad.
    for (key, loss) in [("loss-bce", Loss::Bce), ("loss-mse", Loss::Mse)] {
        g.extend(make(key, base(&[20, 10]), loss, adam(0.005), epochs, &seeds));
    }
    g
}

// ---------------------------------------------------------------------------------------------
// Corrida paralela
// ---------------------------------------------------------------------------------------------

/// Corre todas las configuraciones con `workers` hilos. Cada corrida es independiente y con
/// semilla, por lo que el resultado es identico al secuencial y respeta el orden de `configs`.
pub fn run_sweep(ds: &Dataset, configs: &[ExperimentConfig], workers: usize) -> Vec<ExperimentResult> {
    if configs.is_empty() {
        return Vec::new();
    }
    let workers = workers.clamp(1, configs.len());
    let next = AtomicUsize::new(0);
    let slots: Mutex<Vec<Option<ExperimentResult>>> = Mutex::new((0..configs.len()).map(|_| None).collect());
    std::thread::scope(|scope| {
        for _ in 0..workers {
            scope.spawn(|| loop {
                let i = next.fetch_add(1, Ordering::SeqCst);
                if i >= configs.len() {
                    break;
                }
                let (_, res) = run_experiment(ds, &configs[i]);
                slots.lock().expect("mutex envenenado")[i] = Some(res);
            });
        }
    });
    slots
        .into_inner()
        .expect("mutex envenenado")
        .into_iter()
        .map(|r| r.expect("corrida sin resultado"))
        .collect()
}

// ---------------------------------------------------------------------------------------------
// Escritores
// ---------------------------------------------------------------------------------------------

pub const SWEEP_CSV_HEADER: &str = "study,name,variant,hidden,hidden_act,latent_act,output_act,loss,optimizer,seed,epochs,final_loss,final_max_wrong,final_n_within_1px,first_epoch_within_1px";

fn hidden_label(hidden: &[usize]) -> String {
    hidden.iter().map(usize::to_string).collect::<Vec<_>>().join("x")
}

pub fn write_sweep_csv(path: &Path, results: &[ExperimentResult]) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).with_context(|| format!("no se pudo crear {}", parent.display()))?;
    }
    let mut out = String::from(SWEEP_CSV_HEADER);
    out.push('\n');
    for r in results {
        let c = &r.config;
        let first = r.first_epoch_within.map(|e| e.to_string()).unwrap_or_default();
        let _ = writeln!(
            out,
            "{},{},{},{},{:?},{:?},{:?},{},{},{},{},{:.8},{},{},{}",
            c.study,
            c.name,
            c.variant_key(),
            hidden_label(&c.spec.hidden),
            c.spec.hidden_act,
            c.spec.latent_act,
            c.spec.output_act,
            loss_label(&c.loss),
            c.optimizer.label().replace(',', ";"),
            c.seed,
            c.epochs,
            r.final_loss,
            r.final_max_wrong,
            r.final_n_within,
            first
        );
    }
    std::fs::write(path, out).with_context(|| format!("no se pudo escribir {}", path.display()))
}

fn median(mut v: Vec<f64>) -> Option<f64> {
    if v.is_empty() {
        return None;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = v.len();
    Some(if n % 2 == 1 { v[n / 2] } else { (v[n / 2 - 1] + v[n / 2]) / 2.0 })
}

/// Estudios presentes, primero los de `STUDIES` en orden y luego cualquier otro (extensiones).
fn studies_present(results: &[ExperimentResult]) -> Vec<String> {
    let mut studies: Vec<String> = STUDIES
        .iter()
        .filter(|s| results.iter().any(|r| r.config.study == **s))
        .map(|s| s.to_string())
        .collect();
    for r in results {
        if !studies.contains(&r.config.study) {
            studies.push(r.config.study.clone());
        }
    }
    studies
}

/// Claves de variante de un estudio en orden de aparicion.
pub fn variants_of(results: &[ExperimentResult], study: &str) -> Vec<String> {
    let mut keys: Vec<String> = Vec::new();
    for r in results.iter().filter(|r| r.config.study == study) {
        let k = r.config.variant_key();
        if !keys.contains(&k) {
            keys.push(k);
        }
    }
    keys
}

pub fn render_comparison_md(results: &[ExperimentResult]) -> String {
    let mut out = String::new();
    for study in studies_present(results) {
        let _ = writeln!(out, "## Estudio: {study}\n");
        let _ = writeln!(
            out,
            "| variante | semillas con los 32 patrones a lo sumo 1 pixel | mejor maximo de pixeles incorrectos | mediana de primera epoca dentro de tolerancia | mediana de perdida final |"
        );
        let _ = writeln!(out, "|---|---|---|---|---|");
        for key in variants_of(results, &study) {
            let rs: Vec<&ExperimentResult> =
                results.iter().filter(|r| r.config.study == study && r.config.variant_key() == key).collect();
            let ok = rs.iter().filter(|r| r.final_max_wrong <= TOLERANCE_PX).count();
            let best = rs.iter().map(|r| r.final_max_wrong).min().unwrap_or(0);
            let first = median(rs.iter().filter_map(|r| r.first_epoch_within.map(|e| e as f64)).collect())
                .map(|m| format!("{m:.0}"))
                .unwrap_or_else(|| "-".to_string());
            let loss = median(rs.iter().map(|r| r.final_loss).collect())
                .map(|m| format!("{m:.6}"))
                .unwrap_or_else(|| "-".to_string());
            let _ = writeln!(out, "| {key} | {ok}/{} | {best} | {first} | {loss} |", rs.len());
        }
        out.push('\n');
    }
    out
}

// ---------------------------------------------------------------------------------------------
// Configuracion elegida
// ---------------------------------------------------------------------------------------------

/// Configuracion elegida para el Ej 1a, tomada del sweep completo (`cargo run --release -- sweep`).
///
/// Regla de seleccion, paso (1): se consideran las variantes robustas (todas sus semillas terminan
/// con `final_max_wrong <= 1`); se ordenan por la menor mediana de `first_epoch_within_1px`, y
/// dentro de la mejor variante se toma la semilla con menor `final_max_wrong` y luego menor
/// `final_loss`.
///
/// Variantes robustas y su mediana: arch-h24 (1100), arch-h32x16 (575), arch-h64x32 (275),
/// arch-h48x24x12 (400), optim-adam-lr0.01 (450), act-tanh-lattanh (975), act-relu-latid (1500) y
/// act-relu-lattanh (1325). Gana `arch-h64x32` (hidden [64,32], tanh, latente identidad, salida
/// sigmoide, BCE, Adam lr 0.005, 10000 epocas); sus filas de `sweep.csv` son
/// `arch-h64x32-s0` (max 0, perdida 0.00013432), `-s1` (max 0, 0.00014268) y `-s2` (max 0,
/// 0.00015151), por lo que se elige la semilla 0.
pub fn chosen_config() -> ExperimentConfig {
    make(
        "arch-h64x32",
        spec(&[64, 32], Activation::Tanh, Activation::Identity),
        Loss::Bce,
        adam(0.005),
        10_000,
        &[0],
    )
    .remove(0)
}

/// Entrena `chosen_config()`; es la configuracion que usan las fases siguientes.
pub fn train_chosen(ds: &Dataset) -> (Autoencoder, ExperimentResult) {
    run_experiment(ds, &chosen_config())
}
