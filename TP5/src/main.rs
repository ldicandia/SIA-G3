use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{bail, Result};
use clap::{Parser, Subcommand};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use tp5::autoencoder::{ArchSpec, Autoencoder};
use tp5::analysis::{analyze, export_latent, LATENT_PNG};
use tp5::experiments::{loss_label, render_comparison_md, run_sweep, sweep_grid, train_chosen, write_sweep_csv};
use tp5::dae_experiments::{
    dae_sweep_grid, render_dae_comparison_md, run_dae_sweep, select_dae_variant, selection_line, train_chosen_dae,
    write_dae_sweep_csv,
};
use tp5::dae_figures::{
    build_noise_examples, export_dae_examples, export_level_charts, export_noise_examples, format_grids_side_by_side,
    write_levels_csv, EXAMPLES_NOISE_SEED, LEVELS_CSV, NOISE_EXAMPLES_PNG,
};
use tp5::data::{format_grid, load_font, N_PIXELS};
use tp5::ej2::{dataset_line, result_line, run_ej2, separation_line, Ej2Options};
use tp5::emoji::{build_emojis, format_emoji, format_emojis_side_by_side, nearest_emoji, separation, EMOJI_COUNT};
use tp5::denoising::{
    evaluate_levels, max_level_line, render_level_table, DEFAULT_LEVELS, N_EVAL_SAMPLES,
    TEST_NOISE_SEED,
};
use tp5::noise::validate_levels;
use tp5::generate::{export_new_letter, scan_novel, top_candidates, NEW_LETTER_PNG, NOVELTY_MIN_HAMMING};
use tp5::mlp::{Activation, Loss};
use tp5::plots::{write_dae_study_curves, write_study_curves};
use tp5::optimizer::OptimizerKind;
use tp5::report::{wrong_pixel_counts, write_report_csv, Report};
use tp5::vae::{
    chosen_vae_config, config_line, loss_summary_line, reconstruction_line, reconstruction_summary, render_loss_table,
    standard_normal_matrix, train_vae, validate_epochs, SAMPLE_SEED,
};
use tp5::vae_eval::{
    acceptance_checks, contrast_line, decode_grid, manifold_line, manifold_metrics, plain_grid_points, render_acceptance,
    train_plain_ae, validate_grid, Contrast, GRID_HALF_RANGE, GRID_SIDE_DEFAULT,
    new_emoji_line, prior_line, prior_stats, regions_line, search_new_emoji, validate_draws, StyleJudge,
    DEFAULT_DRAWS,
};
use tp5::vae_figures::{
    export_emoji_montage, export_loss_chart, export_manifold, export_new_emoji, export_reconstruction, write_loss_csv,
    export_plain_manifold, LOSS_CSV, MANIFOLD_PNG, PLAIN_MANIFOLD_PNG,
    NEW_EMOJI_PNG, RECON_PNG,
};

#[derive(Parser)]
#[command(name = "tp5", about = "SIA TP5: autoencoders con backprop propio sobre ndarray")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Imprime los 32 patrones de font.h como grillas de 7x5
    Show,
    /// Entrena de punta a punta un autoencoder 35-16-2-16-35 y muestra el reporte por patron
    Smoke,
    /// Barrido de arquitecturas, optimizadores, activaciones y perdidas con CSV y comparacion
    Sweep {
        /// Grilla reducida (300 epocas, semilla 0)
        #[arg(long)]
        quick: bool,
        /// Directorio de salida
        #[arg(long, default_value = DEFAULT_OUT)]
        out: PathBuf,
        /// Hilos de trabajo (por defecto el paralelismo disponible, maximo 16)
        #[arg(long)]
        workers: Option<usize>,
    },
    /// Entrena la configuracion elegida del Ej 1a y muestra el reporte y el analisis de subconjunto
    Ej1a {
        /// Directorio de salida
        #[arg(long, default_value = DEFAULT_OUT)]
        out: PathBuf,
    },
    /// Entrena el Denoising Autoencoder (entrada ruidosa, objetivo limpio) y mide el ruido que elimina
    Ej1b {
        /// Niveles de ruido a evaluar (probabilidad de invertir cada pixel, entre 0.0 y 0.5), separados por comas
        #[arg(long, value_delimiter = ',', allow_negative_numbers = true)]
        levels: Option<Vec<f64>>,
        /// Directorio de salida
        #[arg(long, default_value = EJ1B_OUT)]
        out: PathBuf,
    },
    /// Barrido de arquitecturas y calendarios de ruido del DAE (tamano latente, capacidad, ruido de entrenamiento)
    DaeSweep {
        /// Grilla reducida (200 epocas, 2 copias, semilla 0)
        #[arg(long)]
        quick: bool,
        /// Directorio de salida
        #[arg(long, default_value = EJ1B_OUT)]
        out: PathBuf,
        /// Hilos de trabajo (por defecto el paralelismo disponible, maximo 16)
        #[arg(long)]
        workers: Option<usize>,
    },
    /// Muestra letras distorsionadas a cada nivel de ruido (texto y PNG), sin entrenar nada
    Noise {
        /// Niveles de ruido (probabilidad de invertir cada pixel, entre 0.0 y 0.5), separados por comas
        #[arg(long, value_delimiter = ',', allow_negative_numbers = true)]
        levels: Option<Vec<f64>>,
        /// Semilla del ruido de los ejemplos
        #[arg(long, default_value_t = EXAMPLES_NOISE_SEED)]
        seed: u64,
        /// Directorio de salida
        #[arg(long, default_value = EJ1B_OUT)]
        out: PathBuf,
    },
    /// Grafica los 32 patrones en el plano latente de 2 dimensiones (PNG y CSV)
    Latent {
        /// Directorio de salida
        #[arg(long, default_value = DEFAULT_OUT)]
        out: PathBuf,
    },
    /// Decodifica un punto del plano latente que no es un patron de entrenamiento (letra nueva)
    Generate {
        /// Punto latente Z1 Z2 a decodificar (por defecto se busca uno automaticamente)
        #[arg(long, num_args = 2, value_names = ["Z1", "Z2"], allow_negative_numbers = true)]
        point: Option<Vec<f64>>,
        /// Lado de la grilla del barrido automatico
        #[arg(long, default_value_t = DEFAULT_STEPS)]
        steps: usize,
        /// Directorio de salida
        #[arg(long, default_value = DEFAULT_OUT)]
        out: PathBuf,
    },
    /// Muestra el dataset de 36 emojis de 24x24 construido por codigo (texto)
    Emojis {
        /// Directorio de salida
        #[arg(long, default_value = EJ2_OUT)]
        out: PathBuf,
    },
    /// Entrena el autoencoder variacional (VAE) sobre los emojis y decodifica una muestra del prior
    Vae {
        /// Epocas de entrenamiento (entre 1 y 20000; por defecto las de la configuracion elegida)
        #[arg(long)]
        epochs: Option<usize>,
        /// Semilla de la muestra del prior
        #[arg(long, default_value_t = SAMPLE_SEED)]
        seed: u64,
        /// Directorio de salida
        #[arg(long, default_value = EJ2_OUT)]
        out: PathBuf,
    },
    /// Muestrea el prior N(0, I) del VAE y busca un emoji nuevo que se juzga perteneciente al dataset
    VaeSample {
        /// Sorteos del prior a examinar (entre 1 y 100000)
        #[arg(long, default_value_t = DEFAULT_DRAWS)]
        draws: usize,
        /// Semilla del flujo de sorteos
        #[arg(long, default_value_t = SAMPLE_SEED)]
        seed: u64,
        /// Epocas de entrenamiento (entre 1 y 20000; por defecto las de la configuracion elegida)
        #[arg(long)]
        epochs: Option<usize>,
        /// Directorio de salida
        #[arg(long, default_value = EJ2_OUT)]
        out: PathBuf,
    },
    /// Decodifica una grilla del plano latente del VAE y mide cuanto es reconocible y cuan gradual es
    VaeManifold {
        /// Lado de la grilla (entre 3 y 41)
        #[arg(long, default_value_t = GRID_SIDE_DEFAULT)]
        grid: usize,
        /// Epocas de entrenamiento (entre 1 y 20000; por defecto las de la configuracion elegida)
        #[arg(long)]
        epochs: Option<usize>,
        /// Directorio de salida
        #[arg(long, default_value = EJ2_OUT)]
        out: PathBuf,
    },
    /// Contrasta el VAE con un autoencoder simple igual pero sin esquema variacional y verifica la aceptacion
    VaeBaseline {
        /// Lado de la grilla (entre 3 y 41)
        #[arg(long, default_value_t = GRID_SIDE_DEFAULT)]
        grid: usize,
        /// Epocas de entrenamiento (entre 1 y 20000; por defecto las de la configuracion elegida)
        #[arg(long)]
        epochs: Option<usize>,
        /// Semilla del flujo de sorteos del prior
        #[arg(long, default_value_t = SAMPLE_SEED)]
        seed: u64,
        /// Directorio de salida
        #[arg(long, default_value = EJ2_OUT)]
        out: PathBuf,
    },
    /// Corre el Ej 2 completo: dataset, VAE, evaluacion, emoji nuevo, contraste y aceptacion
    Ej2 {
        /// Epocas de entrenamiento (entre 1 y 20000; por defecto las de la configuracion elegida)
        #[arg(long)]
        epochs: Option<usize>,
        /// Lado de la grilla (entre 3 y 41)
        #[arg(long, default_value_t = GRID_SIDE_DEFAULT)]
        grid: usize,
        /// Sorteos maximos de la busqueda del emoji nuevo (entre 1 y 100000)
        #[arg(long, default_value_t = DEFAULT_DRAWS)]
        draws: usize,
        /// Semilla del flujo de sorteos del prior
        #[arg(long, default_value_t = SAMPLE_SEED)]
        seed: u64,
        /// Directorio de salida
        #[arg(long, default_value = EJ2_OUT)]
        out: PathBuf,
    },
}

/// Directorio de salida por defecto del Ej 2 (compartido por todos sus subcomandos).
const EJ2_OUT: &str = "output/ej2";

/// Lado por defecto de la grilla de barrido para buscar la letra nueva.
const DEFAULT_STEPS: usize = 81;

/// Directorio de salida por defecto del Ej 1a.
const DEFAULT_OUT: &str = "output/ej1a";

/// Directorio de salida por defecto del Ej 1b.
const EJ1B_OUT: &str = "output/ej1b";
const SWEEP_CSV: &str = "sweep.csv";
const COMPARISON_MD: &str = "comparison.md";
const REPORT_CSV: &str = "report.csv";
const DAE_SWEEP_CSV: &str = "dae_sweep.csv";
const DAE_COMPARISON_MD: &str = "dae_comparison.md";

fn sweep(quick: bool, out: &Path, workers: Option<usize>) -> Result<()> {
    let ds = load_font()?;
    let configs = sweep_grid(quick);
    let workers = workers
        .unwrap_or_else(|| std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1).min(16))
        .clamp(1, 16);
    let start = Instant::now();
    let results = run_sweep(&ds, &configs, workers);
    write_sweep_csv(&out.join(SWEEP_CSV), &results)?;
    let md = render_comparison_md(&results);
    std::fs::write(out.join(COMPARISON_MD), &md)?;
    println!("{md}");
    let n_img = write_study_curves(out, &results)?;
    println!("curvas: {n_img} imagenes -> {}", out.display());
    println!("sweep: {} corridas en {:.1} s -> {}", results.len(), start.elapsed().as_secs_f64(), out.display());
    Ok(())
}

fn dae_sweep(quick: bool, out: &Path, workers: Option<usize>) -> Result<()> {
    let ds = load_font()?;
    let configs = dae_sweep_grid(quick);
    let workers = workers
        .unwrap_or_else(|| std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1).min(16))
        .clamp(1, 16);
    let start = Instant::now();
    let results = run_dae_sweep(&ds, &configs, workers);
    write_dae_sweep_csv(&out.join(DAE_SWEEP_CSV), &results)?;
    for r in &results {
        println!(
            "{} | puntaje {:.4} | elegible {}",
            r.config.name,
            r.score,
            if r.eligible { "si" } else { "no" }
        );
    }
    let md = render_dae_comparison_md(&results);
    std::fs::write(out.join(DAE_COMPARISON_MD), &md)?;
    println!("{md}");
    let n_img = write_dae_study_curves(out, &results)?;
    println!("curvas: {n_img} imagenes -> {}", out.display());
    println!("{}", selection_line(select_dae_variant(&results).as_ref()));
    println!(
        "dae-sweep: {} corridas en {:.1} s -> {}",
        results.len(),
        start.elapsed().as_secs_f64(),
        out.display()
    );
    Ok(())
}

fn show() -> Result<()> {
    let ds = load_font()?;
    for (i, (label, code)) in ds.labels.iter().zip(&ds.codes).enumerate() {
        println!("[{i:02}] {label} (0x{code:02x})");
        println!("{}", format_grid(ds.x.row(i)));
    }
    Ok(())
}

fn smoke() -> Result<()> {
    const EPOCHS: usize = 300;
    let ds = load_font()?;
    println!("cargado: {} patrones x {} pixeles", ds.x.nrows(), N_PIXELS);

    let mut rng = ChaCha8Rng::seed_from_u64(0);
    let spec = ArchSpec {
        hidden: vec![16],
        latent_dim: 2,
        hidden_act: Activation::Tanh,
        latent_act: Activation::Identity,
        output_act: Activation::Sigmoid,
    };
    let mut ae = Autoencoder::new(&spec, N_PIXELS, &mut rng);
    let loss = Loss::Mse;
    let kind = OptimizerKind::Sgd { lr: 0.5 };
    let mut opt_enc = kind.build(&ae.encoder);
    let mut opt_dec = kind.build(&ae.decoder);
    println!(
        "arquitectura: {} | optimizador: {} | perdida: MSE | epocas: {EPOCHS}",
        ae.sizes_label(),
        kind.label()
    );

    let mut initial = f64::NAN;
    for epoch in 0..EPOCHS {
        let l = ae.train_step(&ds.x, &ds.x, loss, opt_enc.as_mut(), opt_dec.as_mut());
        if epoch == 0 {
            initial = l;
        }
    }
    let recon = ae.reconstruct(&ds.x);
    let final_loss = loss.value(&recon, &ds.x);
    println!("perdida inicial: {initial:.6} | perdida final: {final_loss:.6}");

    let report = Report::new(&ds.labels, &wrong_pixel_counts(&ds.x, &recon));
    println!("{}", report.render_table());
    println!("{}", report.summary_line());

    if !initial.is_finite() || !final_loss.is_finite() {
        bail!("SMOKE FAIL: la perdida no es finita (inicial {initial}, final {final_loss})");
    }
    if final_loss >= initial {
        bail!("SMOKE FAIL: la perdida final ({final_loss}) no es menor que la inicial ({initial})");
    }
    if report.patterns.len() != 32 {
        bail!("SMOKE FAIL: el reporte tiene {} filas en lugar de 32", report.patterns.len());
    }
    println!("SMOKE OK");
    Ok(())
}


fn ej1a(out: &Path) -> Result<()> {
    let ds = load_font()?;
    let (ae, res) = train_chosen(&ds);
    let c = &res.config;
    println!("== Ej 1a: autoencoder basico con latente de 2 dimensiones ==");
    println!(
        "configuracion elegida: {} | arquitectura: {} | optimizador: {} | perdida: {} | epocas: {} | semilla: {}",
        c.name,
        ae.sizes_label(),
        c.optimizer.label(),
        loss_label(&c.loss),
        c.epochs,
        c.seed
    );
    println!("{}", res.report.render_table());
    println!("{}", res.report.result_line());
    let train_codes = ae.encode(&ds.x);
    let analysis = analyze(&ds.labels, &ds.x, &train_codes, &res.report, &res.history);
    println!("{}", analysis.render());
    write_report_csv(&out.join(REPORT_CSV), &res.report)?;
    println!("reporte: 32 patrones -> {}/{}", out.display(), REPORT_CSV);
    let codes = export_latent(&ae, &ds, out)?;
    print_latent(&ds, &codes, out);
    new_letter_block(&ae, &ds, out, None, DEFAULT_STEPS)
}

/// Imprime la tabla de codigos latentes y la linea final `latente: ...`.
fn print_latent(ds: &tp5::data::Dataset, codes: &ndarray::Array2<f64>, out: &Path) {
    println!("idx  caracter  z1  z2");
    for (i, (l, r)) in ds.labels.iter().zip(codes.rows()).enumerate() {
        println!("{:>3}  {:<8}  {:.4}  {:.4}", i, l, r[0], r[1]);
    }
    println!("latente: {} patrones -> {}/{}", ds.labels.len(), out.display(), LATENT_PNG);
}

fn latent(out: &Path) -> Result<()> {
    let ds = load_font()?;
    let (ae, _) = train_chosen(&ds);
    let codes = export_latent(&ae, &ds, out)?;
    print_latent(&ds, &codes, out);
    Ok(())
}

/// Genera la letra nueva (automatica o del punto dado), escribe `new_letter.png` e imprime el
/// bloque con coordenadas, grilla, letra mas cercana y veredicto.
fn new_letter_block(
    ae: &Autoencoder,
    ds: &tp5::data::Dataset,
    out: &Path,
    point: Option<[f64; 2]>,
    steps: usize,
) -> Result<()> {
    let letter = export_new_letter(ae, ds, out, point, steps)?;
    println!("punto latente: ({:.4}, {:.4})", letter.z[0], letter.z[1]);
    println!("{}", format_grid(letter.pattern.view()));
    println!(
        "letra mas cercana del entrenamiento: {} (distancia Hamming {})",
        letter.nearest_label, letter.min_hamming
    );
    let verdict = if letter.min_hamming >= NOVELTY_MIN_HAMMING {
        "LETRA NUEVA"
    } else if letter.min_hamming == 1 {
        "CASI IGUAL"
    } else {
        "COINCIDE"
    };
    let detail = match verdict {
        "LETRA NUEVA" => "difiere de los 32 patrones de entrenamiento en al menos 2 pixeles".to_string(),
        "CASI IGUAL" => "difiere en 1 pixel de un patron (dentro de la tolerancia, no cuenta como nueva)".to_string(),
        _ => "coincide con un patron de entrenamiento".to_string(),
    };
    println!("{verdict}: {detail}; distancia latente al codigo de entrenamiento mas cercano {:.4}", letter.latent_distance);
    if point.is_none() {
        let cands = scan_novel(ae, ds, steps, 0.1);
        for c in top_candidates(&cands, 5).into_iter().filter(|c| c.z != letter.z).take(4) {
            println!(
                "otros candidatos: ({:.4}, {:.4}) | mas cercana: {} (Hamming {})\n{}",
                c.z[0],
                c.z[1],
                c.nearest_label,
                c.min_hamming,
                format_grid(c.pattern.view())
            );
        }
    }
    println!("letra nueva -> {}/{}", out.display(), NEW_LETTER_PNG);
    Ok(())
}

fn generate(point: Option<Vec<f64>>, steps: usize, out: &Path) -> Result<()> {
    let ds = load_font()?;
    let (ae, _) = train_chosen(&ds);
    let point = point.map(|p| [p[0], p[1]]);
    new_letter_block(&ae, &ds, out, point, steps)
}

fn ej1b(levels: Option<Vec<f64>>, out: &Path) -> Result<()> {
    let levels = validate_levels(&levels.unwrap_or_else(|| DEFAULT_LEVELS.to_vec()))?;
    let ds = load_font()?;
    let (ae, cfg) = train_chosen_dae(&ds);
    let stats = evaluate_levels(&ae, &ds.x, &levels, N_EVAL_SAMPLES, TEST_NOISE_SEED);
    println!("== Ej 1b: Denoising Autoencoder (entrada ruidosa -> objetivo limpio) ==");
    println!(
        "configuracion: {} | arquitectura: {} | optimizador: {} | perdida: {} | epocas: {} | copias por epoca: {} | ruido de entrenamiento: {} | semilla: {}",
        cfg.name,
        ae.sizes_label(),
        cfg.optimizer.label(),
        loss_label(&cfg.loss),
        cfg.epochs,
        cfg.copies,
        cfg.train_noise.label(),
        cfg.seed
    );
    println!(
        "evaluacion: {N_EVAL_SAMPLES} distorsiones no vistas por patron y por nivel (semilla de prueba {TEST_NOISE_SEED})"
    );
    println!("{}", render_level_table(&stats));
    println!("{}", max_level_line(&stats));

    write_levels_csv(&out.join(LEVELS_CSV), &stats)?;
    println!("datos: {} niveles -> {}/{}", stats.len(), out.display(), LEVELS_CSV);
    if stats.len() >= 2 {
        for path in export_level_charts(&stats, out)? {
            println!("grafico: {}", path.display());
        }
    } else {
        println!("graficos omitidos: se necesitan al menos 2 niveles");
    }

    let ex = build_noise_examples(&ds, &levels, EXAMPLES_NOISE_SEED)?;
    export_noise_examples(&ex, out)?;
    println!("ruido: {}/{}", out.display(), NOISE_EXAMPLES_PNG);
    let paths = export_dae_examples(&ae, &ex, out)?;
    println!("ejemplos: {} imagenes -> {}", paths.len(), out.display());
    Ok(())
}

fn noise(levels: Option<Vec<f64>>, seed: u64, out: &Path) -> Result<()> {
    let levels = validate_levels(&levels.unwrap_or_else(|| DEFAULT_LEVELS.to_vec()))?;
    let ds = load_font()?;
    let ex = build_noise_examples(&ds, &levels, seed)?;
    let rows = |m: &ndarray::Array2<f64>| m.rows().into_iter().map(|r| r.to_owned()).collect::<Vec<_>>();
    println!("originales:");
    println!("{}", format_grids_side_by_side(&rows(&ex.clean)));
    for (i, &level) in ex.levels.iter().enumerate() {
        let counts: Vec<String> =
            ex.labels.iter().zip(ex.inverted_pixels(i)).map(|(l, k)| format!("{l}={k}")).collect();
        println!("nivel {level:.2} | pixeles invertidos: {}", counts.join(" "));
        println!("{}", format_grids_side_by_side(&rows(&ex.noisy[i])));
    }
    export_noise_examples(&ex, out)?;
    println!("ejemplos de ruido -> {}/{}", out.display(), NOISE_EXAMPLES_PNG);
    Ok(())
}

fn emojis(out: &Path) -> Result<()> {
    let ds = build_emojis();
    println!("{}", dataset_line());
    let sep = separation(&ds);
    println!("{}", separation_line(&sep));
    let shown = ["dots-smile", "happy-grin", "wide-open", "x-frown"];
    println!("{}", shown.join(" "));
    let rows: Vec<_> = shown
        .iter()
        .map(|n| {
            let i = ds.names.iter().position(|m| m == n).expect("emoji del dataset");
            ds.x.row(i).to_owned()
        })
        .collect();
    println!("{}", format_emojis_side_by_side(&rows));
    let path = export_emoji_montage(&ds, out)?;
    println!("emojis.png: {}", path.display());
    Ok(())
}

fn vae(epochs: Option<usize>, seed: u64, out: &Path) -> Result<()> {
    let ds = build_emojis();
    let mut cfg = chosen_vae_config();
    if let Some(n) = epochs {
        cfg.epochs = validate_epochs(n)?;
    }
    let (model, hist) = train_vae(&ds, &cfg)?;
    println!("== Ej 2: autoencoder variacional (VAE) sobre emojis ==");
    println!("{}", config_line(&cfg, &model));
    for e in [1, cfg.epochs] {
        let i = e - 1;
        println!("epoca {e} recon={:.4} kl={:.4} total={:.4}", hist.recon[i], hist.kl[i], hist.total(i));
    }
    println!("{}", render_loss_table(&hist));
    println!("{}", loss_summary_line(&hist));
    export_reconstruction(&model, &ds, out)?;
    println!("reconstruccion: {EMOJI_COUNT} emojis -> {}/{RECON_PNG}", out.display());
    println!("{}", reconstruction_line(&reconstruction_summary(&model, &ds)));
    write_loss_csv(&out.join(LOSS_CSV), &hist)?;
    println!("datos: {} epocas -> {}/{LOSS_CSV}", cfg.epochs, out.display());
    let chart = export_loss_chart(&hist, out)?;
    println!("grafico: {}", chart.display());
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let z = standard_normal_matrix(&mut rng, 1, cfg.spec.latent_dim);
    let decoded = model.decode(&z);
    println!("muestra del prior (semilla {seed}): z=({:.4}, {:.4})", z[[0, 0]], z[[0, 1]]);
    println!("{}", format_emoji(decoded.row(0)));
    let (idx, dist) = nearest_emoji(&ds, decoded.row(0));
    println!("emoji mas cercano: {} (distancia {dist} px)", ds.names[idx]);
    Ok(())
}

fn vae_sample(draws: usize, seed: u64, epochs: Option<usize>, out: &Path) -> Result<()> {
    let draws = validate_draws(draws)?;
    let mut cfg = chosen_vae_config();
    if let Some(n) = epochs {
        cfg.epochs = validate_epochs(n)?;
    }
    let ds = build_emojis();
    let (model, _) = train_vae(&ds, &cfg)?;
    let sep = separation(&ds);
    println!("== Ej 2: emoji nuevo muestreado del prior del VAE ==");
    println!("{}", config_line(&cfg, &model));
    println!("{}", separation_line(&sep));
    let style_judge = StyleJudge::new(&ds, &model.reconstruct_mean(&ds.x));
    println!("{}", regions_line(&style_judge));
    let search = search_new_emoji(&model, &ds, seed, draws);
    println!("{}", new_emoji_line(&search, &ds, &sep));
    if let Some(first) = search.accepted.first() {
        let nearest = ds.x.row(first.judgement.nearest_index).to_owned();
        let binarized = first.image.mapv(|v| if v >= 0.5 { 1.0 } else { 0.0 });
        println!("nuevo (izquierda) y {} (derecha):", ds.names[first.judgement.nearest_index]);
        println!("{}", format_emojis_side_by_side(&[binarized, nearest]));
    }
    let decode = |z: &ndarray::Array2<f64>| model.decode(z);
    let stats = prior_stats(&decode, cfg.spec.latent_dim, &ds, &sep, seed, DEFAULT_DRAWS);
    println!("{}", prior_line(&stats, &sep));
    if search.accepted.is_empty() {
        bail!("no se encontro ningun emoji nuevo y plausible en {} sorteos", search.examined);
    }
    export_new_emoji(&search, &ds, out)?;
    println!("nuevo emoji -> {}/{NEW_EMOJI_PNG}", out.display());
    Ok(())
}

fn vae_manifold(grid: usize, epochs: Option<usize>, out: &Path) -> Result<()> {
    let grid = validate_grid(grid)?;
    let mut cfg = chosen_vae_config();
    if let Some(n) = epochs {
        cfg.epochs = validate_epochs(n)?;
    }
    let ds = build_emojis();
    let (model, _) = train_vae(&ds, &cfg)?;
    let sep = separation(&ds);
    println!("== Ej 2: plano latente del VAE decodificado en una grilla ==");
    println!("{}", config_line(&cfg, &model));
    println!("{}", separation_line(&sep));
    let decoded = decode_grid(&model, grid, GRID_HALF_RANGE);
    let metrics = manifold_metrics(&decoded, grid, &ds, &sep);
    println!("{}", manifold_line(&metrics, &sep));
    export_manifold(&decoded, grid, out)?;
    println!("manifold: {}/{MANIFOLD_PNG}", out.display());
    Ok(())
}

fn vae_baseline(grid: usize, epochs: Option<usize>, seed: u64, out: &Path) -> Result<()> {
    let grid = validate_grid(grid)?;
    let mut cfg = chosen_vae_config();
    if let Some(n) = epochs {
        cfg.epochs = validate_epochs(n)?;
    }
    let ds = build_emojis();
    let (model, hist) = train_vae(&ds, &cfg)?;
    let plain = train_plain_ae(&ds, &cfg);
    let sep = separation(&ds);
    println!("== Ej 2: contraste del VAE con un autoencoder sin esquema variacional ==");
    println!("{}", config_line(&cfg, &model));
    println!("autoencoder simple: arquitectura {} | mismos optimizador, perdida, epocas y semilla", plain.sizes_label());
    println!("{}", separation_line(&sep));

    let vae_decode = |z: &ndarray::Array2<f64>| model.decode(z);
    let plain_decode = |z: &ndarray::Array2<f64>| plain.decode(z);
    let latent_dim = cfg.spec.latent_dim;
    let vae_prior = prior_stats(&vae_decode, latent_dim, &ds, &sep, seed, DEFAULT_DRAWS);
    let plain_prior = prior_stats(&plain_decode, latent_dim, &ds, &sep, seed, DEFAULT_DRAWS);
    println!("VAE {}", prior_line(&vae_prior, &sep));
    println!("AE simple {}", prior_line(&plain_prior, &sep));

    let vae_grid_img = decode_grid(&model, grid, GRID_HALF_RANGE);
    let vae_grid = manifold_metrics(&vae_grid_img, grid, &ds, &sep);
    let plain_grid_img = plain.decode(&plain_grid_points(&plain, &ds, grid));
    let plain_grid = manifold_metrics(&plain_grid_img, grid, &ds, &sep);

    let recon = reconstruction_summary(&model, &ds);
    let plain_counts = wrong_pixel_counts(&ds.x, &plain.reconstruct(&ds.x));
    let plain_recon_mean = plain_counts.iter().sum::<usize>() as f64 / plain_counts.len() as f64;
    let contrast = Contrast {
        vae_prior: vae_prior.clone(),
        plain_prior: plain_prior.clone(),
        vae_grid: vae_grid.clone(),
        plain_grid,
        vae_recon_mean: recon.mean_wrong,
        plain_recon_mean,
    };
    println!("{}", contrast_line(&contrast));

    export_plain_manifold(&plain_grid_img, grid, out)?;
    println!("contraste: {}/{PLAIN_MANIFOLD_PNG}", out.display());

    let checks = acceptance_checks(&recon, &hist, &vae_grid, &vae_prior, &plain_prior, &sep);
    println!("{}", render_acceptance(&checks));
    if checks.iter().any(|c| !c.pass) {
        bail!("fallaron {} verificaciones de aceptacion", checks.iter().filter(|c| !c.pass).count());
    }
    Ok(())
}

fn ej2(opts: &Ej2Options, out: &Path) -> Result<()> {
    let run = run_ej2(opts)?;
    let ds = &run.ds;
    println!("== Ej 2: autoencoder variacional (VAE) sobre emojis ==");
    println!("{}", dataset_line());
    println!("{}", separation_line(&run.sep));
    let montage = export_emoji_montage(ds, out)?;
    println!("emojis.png: {}", montage.display());
    println!("{}", config_line(&run.cfg, &run.vae));

    println!("{}", render_loss_table(&run.history));
    println!("{}", loss_summary_line(&run.history));
    write_loss_csv(&out.join(LOSS_CSV), &run.history)?;
    println!("datos: {} epocas -> {}/{LOSS_CSV}", run.cfg.epochs, out.display());
    let chart = export_loss_chart(&run.history, out)?;
    println!("grafico: {}", chart.display());

    println!("{}", reconstruction_line(&run.recon));
    export_reconstruction(&run.vae, ds, out)?;
    println!("reconstruccion: {EMOJI_COUNT} emojis -> {}/{RECON_PNG}", out.display());

    println!("{}", manifold_line(&run.grid, &run.sep));
    export_manifold(&run.grid_decoded, run.grid.side, out)?;
    println!("manifold: {}/{MANIFOLD_PNG}", out.display());

    println!("{}", new_emoji_line(&run.search, ds, &run.sep));
    if let Some(first) = run.search.accepted.first() {
        let nearest = ds.x.row(first.judgement.nearest_index).to_owned();
        let binarized = first.image.mapv(|v| if v >= 0.5 { 1.0 } else { 0.0 });
        println!("nuevo (izquierda) y {} (derecha):", ds.names[first.judgement.nearest_index]);
        println!("{}", format_emojis_side_by_side(&[binarized, nearest]));
        export_new_emoji(&run.search, ds, out)?;
        println!("nuevo emoji -> {}/{NEW_EMOJI_PNG}", out.display());
    }

    println!("VAE {}", prior_line(&run.prior, &run.sep));
    println!("AE simple {}", prior_line(&run.plain_prior, &run.sep));
    println!("{}", contrast_line(&run.contrast));
    export_plain_manifold(&run.plain_grid_decoded, run.plain_grid.side, out)?;
    println!("contraste: {}/{PLAIN_MANIFOLD_PNG}", out.display());
    println!("{}", render_acceptance(&run.checks));

    let result = result_line(&run);
    println!("{result}");
    if result.starts_with("RESULTADO EJ2: FALLA") {
        bail!("el Ej 2 no cumplio la aceptacion");
    }
    Ok(())
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Show => show(),
        Command::Smoke => smoke(),
        Command::Sweep { quick, out, workers } => sweep(quick, &out, workers),
        Command::Ej1a { out } => ej1a(&out),
        Command::Ej1b { levels, out } => ej1b(levels, &out),
        Command::DaeSweep { quick, out, workers } => dae_sweep(quick, &out, workers),
        Command::Noise { levels, seed, out } => noise(levels, seed, &out),
        Command::Latent { out } => latent(&out),
        Command::Generate { point, steps, out } => generate(point, steps, &out),
        Command::Emojis { out } => emojis(&out),
        Command::Vae { epochs, seed, out } => vae(epochs, seed, &out),
        Command::VaeSample { draws, seed, epochs, out } => vae_sample(draws, seed, epochs, &out),
        Command::VaeManifold { grid, epochs, out } => vae_manifold(grid, epochs, &out),
        Command::VaeBaseline { grid, epochs, seed, out } => vae_baseline(grid, epochs, seed, &out),
        Command::Ej2 { epochs, grid, draws, seed, out } => ej2(&Ej2Options { epochs, grid, draws, seed }, &out),
    }
}
