//! Estudio de arquitectura del Denoising Autoencoder (Ej 1b): grilla de variantes, corridas en
//! paralelo puntuadas sobre un ruido de seleccion y escritor de CSV. Este modulo no dibuja ni
//! parsea la linea de comandos (los graficos viven en `plots`, la CLI en `main`).

use std::fmt::Write as _;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use anyhow::{Context, Result};

use crate::autoencoder::{ArchSpec, Autoencoder};
use crate::data::Dataset;
use crate::denoising::{
    evaluate_levels, train_dae, DaeConfig, DaeHistory, LevelStats, N_EVAL_SAMPLES,
};
use crate::mlp::{Activation, Loss};
use crate::noise::NoiseSchedule;
use crate::optimizer::OptimizerKind;

/// Semilla del ruido de seleccion: distinta del ruido de entrenamiento (derivado de la semilla
/// de cada corrida) y del ruido de prueba (`TEST_NOISE_SEED`) con el que se informan las cifras.
pub const SELECTION_NOISE_SEED: u64 = 0x5e1e_0002;

/// Niveles con los que se puntua cada corrida. El 0.0 es el chequeo de elegibilidad.
pub const SWEEP_LEVELS: [f64; 6] = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30];

/// Distorsiones sorteadas por patron y por nivel en la puntuacion.
pub const SWEEP_SAMPLES: usize = N_EVAL_SAMPLES;

/// Orden fijo de los estudios en CSV, markdown y graficos.
pub const DAE_STUDIES: [&str; 3] = ["lat", "hid", "noise"];

pub const SWEEP_CSV_HEADER: &str = "study,name,variant,hidden,latent_dim,train_noise,seed,epochs,copies,final_loss,eligible,score,wrong_p0.00,wrong_p0.05,wrong_p0.10,wrong_p0.15,wrong_p0.20,wrong_p0.30,within_p0.00,within_p0.05,within_p0.10,within_p0.15,within_p0.20,within_p0.30";

/// Resultado de una corrida del estudio.
#[derive(Debug, Clone)]
pub struct DaeResult {
    pub config: DaeConfig,
    pub history: DaeHistory,
    /// Estadisticas por nivel de `SWEEP_LEVELS` sobre el ruido de seleccion.
    pub stats: Vec<LevelStats>,
    /// Media de los pixeles incorrectos de salida en los niveles mayores que 0.0 (menor es mejor).
    pub score: f64,
    /// Los 32 patrones limpios vuelven con a lo sumo 1 pixel incorrecto (criterio del Ej 1a).
    pub eligible: bool,
}

/// Media de `output_mean_wrong` sobre las entradas con nivel mayor que 0.0.
pub fn score_of(stats: &[LevelStats]) -> f64 {
    let v: Vec<f64> = stats.iter().filter(|s| s.level > 0.0).map(|s| s.output_mean_wrong).collect();
    if v.is_empty() {
        return f64::NAN;
    }
    v.iter().sum::<f64>() / v.len() as f64
}

/// Existe la entrada de nivel 0.0 y todos los patrones limpios quedan dentro de la tolerancia.
pub fn is_eligible(stats: &[LevelStats]) -> bool {
    stats.iter().find(|s| s.level == 0.0).is_some_and(|s| s.frac_within == 1.0)
}

/// Entrena la configuracion y la puntua sobre el ruido de seleccion.
pub fn run_dae_experiment(ds: &Dataset, cfg: &DaeConfig) -> (Autoencoder, DaeResult) {
    let (ae, history) = train_dae(ds, cfg);
    let stats = evaluate_levels(&ae, &ds.x, &SWEEP_LEVELS, SWEEP_SAMPLES, SELECTION_NOISE_SEED);
    let result = DaeResult {
        config: cfg.clone(),
        history,
        score: score_of(&stats),
        eligible: is_eligible(&stats),
        stats,
    };
    (ae, result)
}

// ---------------------------------------------------------------------------------------------
// Grilla
// ---------------------------------------------------------------------------------------------

fn make(
    variant: &str,
    hidden: &[usize],
    latent_dim: usize,
    train_noise: NoiseSchedule,
    epochs: usize,
    copies: usize,
    seeds: &[u64],
) -> Vec<DaeConfig> {
    let study = variant.split('-').next().unwrap_or(variant).to_string();
    seeds
        .iter()
        .map(|&seed| DaeConfig {
            name: format!("{variant}-s{seed}"),
            study: study.clone(),
            spec: ArchSpec {
                hidden: hidden.to_vec(),
                latent_dim,
                hidden_act: Activation::Tanh,
                latent_act: Activation::Identity,
                output_act: Activation::Sigmoid,
            },
            loss: Loss::Bce,
            optimizer: OptimizerKind::Adam { lr: 0.005, beta1: 0.9, beta2: 0.999, eps: 1e-8 },
            train_noise,
            epochs,
            copies,
            seed,
        })
        .collect()
}

/// Grilla del estudio: 13 variantes en tres estudios (un factor por vez respecto de la base
/// oculta [64, 32], latente 8, ruido de entrenamiento uniforme 0.0-0.2).
/// Completa: 3000 epocas, 8 copias, semillas 0, 1, 2 (39 corridas). Rapida: 200 epocas, 2 copias,
/// semilla 0 (13 corridas).
pub fn dae_sweep_grid(quick: bool) -> Vec<DaeConfig> {
    let (epochs, copies, seeds): (usize, usize, Vec<u64>) =
        if quick { (200, 2, vec![0]) } else { (3000, 8, vec![0, 1, 2]) };
    let base_noise = NoiseSchedule::Uniform { lo: 0.0, hi: 0.2 };
    let mut g = Vec::new();

    // Estudio lat: tamano del espacio latente.
    for (key, latent) in [("lat-l2", 2), ("lat-l4", 4), ("lat-l8", 8), ("lat-l16", 16)] {
        g.extend(make(key, &[64, 32], latent, base_noise, epochs, copies, &seeds));
    }

    // Estudio hid: capacidad del encoder con latente 8.
    for (key, hidden) in [
        ("hid-h24", vec![24]),
        ("hid-h32x16", vec![32, 16]),
        ("hid-h64x32", vec![64, 32]),
        ("hid-h48x24x12", vec![48, 24, 12]),
    ] {
        g.extend(make(key, &hidden, 8, base_noise, epochs, copies, &seeds));
    }

    // Estudio noise: calendario de ruido de entrenamiento con [64, 32] y latente 8.
    for (key, noise) in [
        ("noise-fix0.10", NoiseSchedule::Fixed(0.10)),
        ("noise-fix0.20", NoiseSchedule::Fixed(0.20)),
        ("noise-u0-0.20", NoiseSchedule::Uniform { lo: 0.0, hi: 0.2 }),
        ("noise-u0-0.30", NoiseSchedule::Uniform { lo: 0.0, hi: 0.3 }),
        ("noise-u0-0.40", NoiseSchedule::Uniform { lo: 0.0, hi: 0.4 }),
    ] {
        g.extend(make(key, &[64, 32], 8, noise, epochs, copies, &seeds));
    }
    g
}

/// Claves de variante de un estudio en orden de aparicion.
pub fn dae_variants_of(results: &[DaeResult], study: &str) -> Vec<String> {
    let mut keys: Vec<String> = Vec::new();
    for r in results.iter().filter(|r| r.config.study == study) {
        let k = r.config.variant_key();
        if !keys.contains(&k) {
            keys.push(k);
        }
    }
    keys
}

// ---------------------------------------------------------------------------------------------
// Corrida paralela
// ---------------------------------------------------------------------------------------------

/// Corre todas las configuraciones con `workers` hilos. Cada corrida es independiente y con
/// semilla, por lo que el resultado es identico al secuencial y respeta el orden de `configs`.
pub fn run_dae_sweep(ds: &Dataset, configs: &[DaeConfig], workers: usize) -> Vec<DaeResult> {
    if configs.is_empty() {
        return Vec::new();
    }
    let workers = workers.clamp(1, configs.len());
    let next = AtomicUsize::new(0);
    let slots: Mutex<Vec<Option<DaeResult>>> = Mutex::new((0..configs.len()).map(|_| None).collect());
    std::thread::scope(|scope| {
        for _ in 0..workers {
            scope.spawn(|| loop {
                let i = next.fetch_add(1, Ordering::SeqCst);
                if i >= configs.len() {
                    break;
                }
                let (_, res) = run_dae_experiment(ds, &configs[i]);
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
// CSV
// ---------------------------------------------------------------------------------------------

fn hidden_label(hidden: &[usize]) -> String {
    hidden.iter().map(usize::to_string).collect::<Vec<_>>().join("x")
}

pub fn write_dae_sweep_csv(path: &Path, results: &[DaeResult]) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).with_context(|| format!("no se pudo crear {}", parent.display()))?;
    }
    let mut out = String::from(SWEEP_CSV_HEADER);
    out.push('\n');
    for r in results {
        let c = &r.config;
        let wrong: Vec<String> = r.stats.iter().map(|s| format!("{:.4}", s.output_mean_wrong)).collect();
        let within: Vec<String> = r.stats.iter().map(|s| format!("{:.4}", s.frac_within)).collect();
        let _ = writeln!(
            out,
            "{},{},{},{},{},{},{},{},{},{:.8},{},{:.6},{},{}",
            c.study,
            c.name,
            c.variant_key(),
            hidden_label(&c.spec.hidden),
            c.spec.latent_dim,
            c.train_noise.label(),
            c.seed,
            c.epochs,
            c.copies,
            r.history.tail_loss,
            r.eligible,
            r.score,
            wrong.join(","),
            within.join(",")
        );
    }
    std::fs::write(path, out).with_context(|| format!("no se pudo escribir {}", path.display()))
}

// ---------------------------------------------------------------------------------------------
// Comparacion y regla de seleccion
// ---------------------------------------------------------------------------------------------

fn median(mut v: Vec<f64>) -> Option<f64> {
    if v.is_empty() {
        return None;
    }
    v.sort_by(|a, b| a.total_cmp(b));
    let n = v.len();
    Some(if n % 2 == 1 { v[n / 2] } else { (v[n / 2 - 1] + v[n / 2]) / 2.0 })
}

/// Entrada de estadisticas del nivel pedido (comparacion exacta: los niveles vienen de `SWEEP_LEVELS`).
fn stat_at(r: &DaeResult, level: f64) -> Option<&LevelStats> {
    r.stats.iter().find(|s| s.level == level)
}

fn fmt_median(v: Vec<f64>, decimals: usize) -> String {
    median(v).map(|m| format!("{m:.decimals$}")).unwrap_or_else(|| "-".to_string())
}

/// Una tabla por estudio: por variante, semillas elegibles y medianas de puntaje, pixeles
/// incorrectos a 0.10 / 0.20 / 0.30, porcentaje dentro de 1 pixel a 0.10 y perdida final.
pub fn render_dae_comparison_md(results: &[DaeResult]) -> String {
    let mut out = String::new();
    for study in DAE_STUDIES {
        let keys = dae_variants_of(results, study);
        if keys.is_empty() {
            continue;
        }
        let _ = writeln!(out, "## Estudio: {study}\n");
        let _ = writeln!(
            out,
            "| variante | semillas elegibles | puntaje (mediana) | px a nivel 0.10 | px a nivel 0.20 | px a nivel 0.30 | dentro de 1 px a nivel 0.10 (%) | perdida final (mediana) |"
        );
        let _ = writeln!(out, "|---|---|---|---|---|---|---|---|");
        for key in keys {
            let rs: Vec<&DaeResult> =
                results.iter().filter(|r| r.config.study == study && r.config.variant_key() == key).collect();
            let ok = rs.iter().filter(|r| r.eligible).count();
            let wrong_at = |level: f64| -> Vec<f64> {
                rs.iter().filter_map(|r| stat_at(r, level).map(|s| s.output_mean_wrong)).collect()
            };
            let within: Vec<f64> =
                rs.iter().filter_map(|r| stat_at(r, 0.10).map(|s| s.frac_within * 100.0)).collect();
            let _ = writeln!(
                out,
                "| {key} | {ok}/{} | {} | {} | {} | {} | {} | {} |",
                rs.len(),
                fmt_median(rs.iter().map(|r| r.score).collect(), 4),
                fmt_median(wrong_at(0.10), 3),
                fmt_median(wrong_at(0.20), 3),
                fmt_median(wrong_at(0.30), 3),
                fmt_median(within, 1),
                fmt_median(rs.iter().map(|r| r.history.tail_loss).collect(), 6),
            );
        }
        out.push('\n');
    }
    out
}

/// Resultado de aplicar la regla de seleccion.
#[derive(Debug, Clone, PartialEq)]
pub struct Selection {
    pub variant: String,
    /// Nombre de la corrida ganadora (variante + semilla).
    pub name: String,
    pub seed: u64,
    pub median_score: f64,
    pub n_eligible: usize,
    pub n_runs: usize,
}

/// Regla de seleccion: se agrupan las corridas por variante en orden de aparicion; una variante es
/// elegible solo si todas sus corridas lo son; gana la de menor mediana de puntaje (empate: la
/// primera en el orden de la grilla) y dentro de ella la corrida de menor puntaje (empate: menor
/// semilla). `None` si ninguna variante es elegible.
pub fn select_dae_variant(results: &[DaeResult]) -> Option<Selection> {
    let mut keys: Vec<String> = Vec::new();
    for r in results {
        let k = r.config.variant_key();
        if !keys.contains(&k) {
            keys.push(k);
        }
    }
    let mut best: Option<(String, f64)> = None;
    for key in keys {
        let rs: Vec<&DaeResult> = results.iter().filter(|r| r.config.variant_key() == key).collect();
        if !rs.iter().all(|r| r.eligible) {
            continue;
        }
        let Some(m) = median(rs.iter().map(|r| r.score).collect()) else { continue };
        if best.as_ref().is_none_or(|(_, b)| m < *b) {
            best = Some((key, m));
        }
    }
    let (variant, median_score) = best?;
    let rs: Vec<&DaeResult> = results.iter().filter(|r| r.config.variant_key() == variant).collect();
    let winner = rs
        .iter()
        .min_by(|a, b| a.score.total_cmp(&b.score).then(a.config.seed.cmp(&b.config.seed)))
        .copied()?;
    Some(Selection {
        variant,
        name: winner.config.name.clone(),
        seed: winner.config.seed,
        median_score,
        n_eligible: rs.iter().filter(|r| r.eligible).count(),
        n_runs: rs.len(),
    })
}

/// Linea `SELECCION: ...` con el veredicto de la regla.
pub fn selection_line(sel: Option<&Selection>) -> String {
    match sel {
        Some(s) => format!(
            "SELECCION: {} (variante {}, puntaje mediano {:.4}, {}/{} semillas elegibles)",
            s.name, s.variant, s.median_score, s.n_eligible, s.n_runs
        ),
        None => "SELECCION: ninguna variante elegible (se necesita que todas las semillas dejen los 32 patrones limpios con a lo sumo 1 pixel incorrecto)".to_string(),
    }
}
