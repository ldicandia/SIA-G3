use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{bail, Result};
use clap::{Parser, Subcommand};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use tp5::autoencoder::{ArchSpec, Autoencoder};
use tp5::analysis::analyze;
use tp5::experiments::{loss_label, render_comparison_md, run_sweep, sweep_grid, train_chosen, write_sweep_csv};
use tp5::data::{format_grid, load_font, N_PIXELS};
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
}

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
    let codes = ae.encode(&ds.x);
    let analysis = analyze(&ds.labels, &ds.x, &codes, &res.report, &res.history);
    println!("{}", analysis.render());
    write_report_csv(&out.join(REPORT_CSV), &res.report)?;
    Ok(())
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Show => show(),
        Command::Smoke => smoke(),
        Command::Sweep { quick, out, workers } => sweep(quick, &out, workers),
        Command::Ej1a { out } => ej1a(&out),
    }
}
