use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{bail, Result};
use clap::{Parser, Subcommand};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use tp5::autoencoder::{ArchSpec, Autoencoder};
use tp5::analysis::{analyze, export_latent, LATENT_PNG};
use tp5::experiments::{loss_label, render_comparison_md, run_sweep, sweep_grid, train_chosen, write_sweep_csv};
use tp5::data::{format_grid, load_font, N_PIXELS};
use tp5::denoising::{
    base_dae_config, evaluate_levels, render_level_table, train_dae, DEFAULT_LEVELS, N_EVAL_SAMPLES, TEST_NOISE_SEED,
};
use tp5::generate::{export_new_letter, scan_novel, top_candidates, NEW_LETTER_PNG, NOVELTY_MIN_HAMMING};
use tp5::mlp::{Activation, Loss};
use tp5::plots::write_study_curves;
use tp5::optimizer::OptimizerKind;
use tp5::report::{wrong_pixel_counts, write_report_csv, Report};

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
    Ej1b,
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
}

/// Lado por defecto de la grilla de barrido para buscar la letra nueva.
const DEFAULT_STEPS: usize = 81;

/// Directorio de salida por defecto del Ej 1a.
const DEFAULT_OUT: &str = "output/ej1a";
const SWEEP_CSV: &str = "sweep.csv";
const COMPARISON_MD: &str = "comparison.md";
const REPORT_CSV: &str = "report.csv";

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

fn ej1b() -> Result<()> {
    let ds = load_font()?;
    let cfg = base_dae_config();
    let (ae, _history) = train_dae(&ds, &cfg);
    let stats = evaluate_levels(&ae, &ds.x, &DEFAULT_LEVELS, N_EVAL_SAMPLES, TEST_NOISE_SEED);
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
    Ok(())
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Show => show(),
        Command::Smoke => smoke(),
        Command::Sweep { quick, out, workers } => sweep(quick, &out, workers),
        Command::Ej1a { out } => ej1a(&out),
        Command::Ej1b => ej1b(),
        Command::Latent { out } => latent(&out),
        Command::Generate { point, steps, out } => generate(point, steps, &out),
    }
}
