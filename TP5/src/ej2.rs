//! Orquestacion del Ej 2 completo como una sola corrida determinista: dataset de menas, VAE, evaluacion,
//! busqueda de un bloque nuevo, autoencoder simple de contraste y aceptacion. Solo compone funciones de
//! la biblioteca: no dibuja ni parsea argumentos, y las lineas de reporte salen de funciones
//! compartidas para que la linea de comandos y las pruebas no se separen.

use ndarray::Array2;

use crate::autoencoder::Autoencoder;
use crate::ores::{
    load_ores, separation, OreDataset, Separation, ORE_COUNT, ORE_MINERALS, ORE_ROCKS, ORE_SIDE, ORE_VALUES,
};
use crate::vae::{
    chosen_vae_config, reconstruction_summary, train_vae, validate_epochs, ReconSummary, Vae, VaeConfig, VaeHistory,
    SAMPLE_SEED,
};
use crate::vae_eval::{
    acceptance_checks, decode_grid, manifold_metrics, mean_wrong_pixels, negative_control, plain_grid_points,
    prior_stats, search_new_ore, train_plain_ae, validate_draws, validate_grid, AcceptanceCheck, Contrast,
    ControlStats, GridMetrics, OreJudge, PriorStats, SampleSearch, DEFAULT_DRAWS, GRID_HALF_RANGE, GRID_SIDE_DEFAULT,
};

/// Linea del dataset, construida con las constantes del modulo `ores`.
pub fn dataset_line() -> String {
    format!(
        "bloques de mena: {ORE_COUNT} texturas de Minecraft 1.21.4 de {ORE_SIDE}x{ORE_SIDE} RGB ({ORE_VALUES} valores en [0,1]) | {} minerales, {} rocas",
        ORE_MINERALS.len(),
        ORE_ROCKS.len()
    )
}

/// Linea de las estadisticas de separacion del dataset.
pub fn separation_line(sep: &Separation) -> String {
    format!(
        "separacion (error medio por canal al bloque mas cercano): minima {:.1} niveles, mediana {:.1} niveles",
        sep.min, sep.median
    )
}

/// Opciones de la corrida completa; todas se validan antes de entrenar.
#[derive(Debug, Clone)]
pub struct Ej2Options {
    /// Reemplazo de las epocas de la configuracion elegida.
    pub epochs: Option<usize>,
    pub grid: usize,
    /// Maximo de sorteos de la busqueda del bloque nuevo.
    pub draws: usize,
    pub seed: u64,
}

impl Default for Ej2Options {
    fn default() -> Self {
        Ej2Options { epochs: None, grid: GRID_SIDE_DEFAULT, draws: DEFAULT_DRAWS, seed: SAMPLE_SEED }
    }
}

/// Todo lo que produce el Ej 2 en una corrida.
#[derive(Debug, Clone)]
pub struct Ej2Run {
    pub ds: OreDataset,
    pub sep: Separation,
    pub cfg: VaeConfig,
    pub vae: Vae,
    pub history: VaeHistory,
    pub recon: ReconSummary,
    pub grid_decoded: Array2<f64>,
    pub grid: GridMetrics,
    pub search: SampleSearch,
    /// Juez con los pisos de las reconstrucciones del VAE.
    pub judge: OreJudge,
    pub prior: PriorStats,
    pub plain: Autoencoder,
    /// Juez con los pisos de las reconstrucciones del autoencoder simple.
    pub plain_judge: OreJudge,
    pub plain_prior: PriorStats,
    pub plain_grid_decoded: Array2<f64>,
    pub plain_grid: GridMetrics,
    pub contrast: Contrast,
    /// Control negativo: reconstrucciones de los bloques de entrenamiento no deben aceptarse como nuevas.
    pub control: ControlStats,
    pub checks: Vec<AcceptanceCheck>,
}

/// Entrena el VAE y el autoencoder simple una sola vez y calcula todos los resultados del ejercicio.
pub fn run_ej2(opts: &Ej2Options) -> anyhow::Result<Ej2Run> {
    let epochs = opts.epochs.map(validate_epochs).transpose()?;
    let grid_side = validate_grid(opts.grid)?;
    let draws = validate_draws(opts.draws)?;

    let ds = load_ores()?;
    let sep = separation(&ds);
    let mut cfg = chosen_vae_config();
    if let Some(n) = epochs {
        cfg.epochs = n;
    }
    let (vae, history) = train_vae(&ds, &cfg)?;
    let recon = reconstruction_summary(&vae, &ds);
    let judge = OreJudge::new(&ds, &vae.reconstruct_mean(&ds.x));
    let grid_decoded = decode_grid(&vae, grid_side, GRID_HALF_RANGE);
    let grid = manifold_metrics(&grid_decoded, grid_side, &judge);
    let search = search_new_ore(&vae, &ds, opts.seed, draws);

    let latent_dim = cfg.spec.latent_dim;
    let vae_decode = |z: &Array2<f64>| vae.decode(z);
    let prior = prior_stats(&vae_decode, latent_dim, &judge, opts.seed, DEFAULT_DRAWS);

    let plain = train_plain_ae(&ds, &cfg);
    let plain_recon = plain.reconstruct(&ds.x);
    let plain_judge = OreJudge::new(&ds, &plain_recon);
    let plain_decode = |z: &Array2<f64>| plain.decode(z);
    let plain_prior = prior_stats(&plain_decode, latent_dim, &plain_judge, opts.seed, DEFAULT_DRAWS);
    let plain_grid_decoded = plain.decode(&plain_grid_points(&plain, &ds, grid_side));
    let plain_grid = manifold_metrics(&plain_grid_decoded, grid_side, &plain_judge);

    let contrast = Contrast {
        vae_prior: prior.clone(),
        plain_prior: plain_prior.clone(),
        vae_grid: grid.clone(),
        plain_grid: plain_grid.clone(),
        vae_recon_mean: recon.mean_wrong,
        plain_recon_mean: mean_wrong_pixels(&ds, &plain_recon),
    };
    let control = negative_control(&vae, &judge);
    let checks = acceptance_checks(&recon, &history, &grid, &prior, &plain_prior, &control, &judge);

    Ok(Ej2Run {
        ds,
        sep,
        cfg,
        vae,
        history,
        recon,
        grid_decoded,
        grid,
        search,
        judge,
        prior,
        plain,
        plain_judge,
        plain_prior,
        plain_grid_decoded,
        plain_grid,
        contrast,
        control,
        checks,
    })
}

/// Veredicto final: `OK` solo si pasan todas las comprobaciones y se acepto al menos un bloque nuevo.
pub fn result_line(run: &Ej2Run) -> String {
    let total = run.checks.len();
    let passed = run.checks.iter().filter(|c| c.pass).count();
    let failed: Vec<&str> = run.checks.iter().filter(|c| !c.pass).map(|c| c.name).collect();
    match run.search.accepted.first() {
        Some(first) if failed.is_empty() => format!(
            "RESULTADO EJ2: OK ({passed}/{total} comprobaciones de aceptacion; bloque nuevo {})",
            first.verdict.combination()
        ),
        accepted => {
            let mut reasons: Vec<String> = Vec::new();
            if !failed.is_empty() {
                reasons.push(format!("fallan: {}", failed.join(", ")));
            }
            if accepted.is_none() {
                reasons.push(format!("no se encontro ningun bloque nuevo en {} sorteos", run.search.examined));
            }
            format!("RESULTADO EJ2: FALLA ({passed}/{total} comprobaciones de aceptacion; {})", reasons.join("; "))
        }
    }
}
