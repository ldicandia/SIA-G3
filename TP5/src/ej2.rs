//! Orquestacion del Ej 2 completo como una sola corrida determinista: dataset, VAE, evaluacion,
//! busqueda de un emoji nuevo, autoencoder simple de contraste y aceptacion. Solo compone funciones de
//! la biblioteca: no dibuja ni parsea argumentos, y las lineas de reporte salen de funciones
//! compartidas para que la linea de comandos y las pruebas no se separen.

use ndarray::Array2;

use crate::autoencoder::Autoencoder;
use crate::emoji::{build_emojis, separation, EmojiDataset, Separation, EMOJI_COUNT, EMOJI_PIXELS, EMOJI_SIDE};
use crate::report::wrong_pixel_counts;
use crate::vae::{
    chosen_vae_config, reconstruction_summary, train_vae, validate_epochs, ReconSummary, Vae, VaeConfig, VaeHistory,
    SAMPLE_SEED,
};
use crate::vae_eval::{
    acceptance_checks, decode_grid, manifold_metrics, plain_grid_points, prior_stats, search_new_emoji,
    train_plain_ae, validate_draws, validate_grid, AcceptanceCheck, Contrast, GridMetrics, PriorStats, SampleSearch,
    StyleJudge, DEFAULT_DRAWS, GRID_HALF_RANGE, GRID_SIDE_DEFAULT,
};

/// Linea del dataset, construida con las constantes del modulo `emoji`.
pub fn dataset_line() -> String {
    format!(
        "emojis: {EMOJI_COUNT} imagenes de {EMOJI_SIDE}x{EMOJI_SIDE} ({EMOJI_PIXELS} valores en [0,1], 1 = tinta)"
    )
}

/// Linea de las estadisticas de separacion del dataset.
pub fn separation_line(sep: &Separation) -> String {
    format!(
        "separacion: minima {} px, mediana {:.1} px, radio de cercania {} px",
        sep.min, sep.median, sep.near_radius
    )
}

/// Opciones de la corrida completa; todas se validan antes de entrenar.
#[derive(Debug, Clone)]
pub struct Ej2Options {
    /// Reemplazo de las epocas de la configuracion elegida.
    pub epochs: Option<usize>,
    pub grid: usize,
    /// Maximo de sorteos de la busqueda del emoji nuevo.
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
    pub ds: EmojiDataset,
    pub sep: Separation,
    pub cfg: VaeConfig,
    pub vae: Vae,
    pub history: VaeHistory,
    pub recon: ReconSummary,
    pub grid_decoded: Array2<f64>,
    pub grid: GridMetrics,
    pub search: SampleSearch,
    /// Juez por region con los pisos de ruido de las reconstrucciones del VAE.
    pub judge: StyleJudge,
    pub prior: PriorStats,
    pub plain: Autoencoder,
    /// Juez por region con los pisos de ruido de las reconstrucciones del autoencoder simple.
    pub plain_judge: StyleJudge,
    pub plain_prior: PriorStats,
    pub plain_grid_decoded: Array2<f64>,
    pub plain_grid: GridMetrics,
    pub contrast: Contrast,
    pub checks: Vec<AcceptanceCheck>,
}

/// Entrena el VAE y el autoencoder simple una sola vez y calcula todos los resultados del ejercicio.
pub fn run_ej2(opts: &Ej2Options) -> anyhow::Result<Ej2Run> {
    let epochs = opts.epochs.map(validate_epochs).transpose()?;
    let grid_side = validate_grid(opts.grid)?;
    let draws = validate_draws(opts.draws)?;

    let ds = build_emojis();
    let sep = separation(&ds);
    let mut cfg = chosen_vae_config();
    if let Some(n) = epochs {
        cfg.epochs = n;
    }
    let (vae, history) = train_vae(&ds, &cfg)?;
    let recon = reconstruction_summary(&vae, &ds);
    let grid_decoded = decode_grid(&vae, grid_side, GRID_HALF_RANGE);
    let grid = manifold_metrics(&grid_decoded, grid_side, &ds, &sep);
    let search = search_new_emoji(&vae, &ds, opts.seed, draws);

    let latent_dim = cfg.spec.latent_dim;
    let judge = StyleJudge::new(&ds, &vae.reconstruct_mean(&ds.x));
    let vae_decode = |z: &Array2<f64>| vae.decode(z);
    let prior = prior_stats(&vae_decode, latent_dim, &judge, opts.seed, DEFAULT_DRAWS);

    let plain = train_plain_ae(&ds, &cfg);
    let plain_judge = StyleJudge::new(&ds, &plain.reconstruct(&ds.x));
    let plain_decode = |z: &Array2<f64>| plain.decode(z);
    let plain_prior = prior_stats(&plain_decode, latent_dim, &plain_judge, opts.seed, DEFAULT_DRAWS);
    let plain_grid_decoded = plain.decode(&plain_grid_points(&plain, &ds, grid_side));
    let plain_grid = manifold_metrics(&plain_grid_decoded, grid_side, &ds, &sep);
    let plain_counts = wrong_pixel_counts(&ds.x, &plain.reconstruct(&ds.x));
    let plain_recon_mean = plain_counts.iter().sum::<usize>() as f64 / plain_counts.len() as f64;

    let contrast = Contrast {
        vae_prior: prior.clone(),
        plain_prior: plain_prior.clone(),
        vae_grid: grid.clone(),
        plain_grid: plain_grid.clone(),
        vae_recon_mean: recon.mean_wrong,
        plain_recon_mean,
    };
    let checks = acceptance_checks(&recon, &history, &grid, &prior, &plain_prior, &sep);

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
        checks,
    })
}

/// Veredicto final: `OK` solo si pasan todas las comprobaciones y se acepto al menos un emoji nuevo.
pub fn result_line(run: &Ej2Run) -> String {
    let total = run.checks.len();
    let passed = run.checks.iter().filter(|c| c.pass).count();
    let failed: Vec<&str> = run.checks.iter().filter(|c| !c.pass).map(|c| c.name).collect();
    match run.search.accepted.first() {
        Some(first) if failed.is_empty() => format!(
            "RESULTADO EJ2: OK ({passed}/{total} comprobaciones de aceptacion; emoji nuevo a {} px de {})",
            first.judgement.distance, run.ds.names[first.judgement.nearest_index]
        ),
        accepted => {
            let mut reasons: Vec<String> = Vec::new();
            if !failed.is_empty() {
                reasons.push(format!("fallan: {}", failed.join(", ")));
            }
            if accepted.is_none() {
                reasons.push(format!("no se encontro ningun emoji nuevo en {} sorteos", run.search.examined));
            }
            format!("RESULTADO EJ2: FALLA ({passed}/{total} comprobaciones de aceptacion; {})", reasons.join("; "))
        }
    }
}
