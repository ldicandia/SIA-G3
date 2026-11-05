//! La corrida completa del Ej 2: determinismo, coherencia con los calculos individuales, validacion
//! de opciones y formato de las lineas de dataset y de resultado.

use std::sync::OnceLock;

use ndarray::Array2;

use tp5::ej2::{dataset_line, result_line, run_ej2, separation_line, Ej2Options, Ej2Run};
use tp5::emoji::{build_emojis, separation};
use tp5::vae::{
    chosen_vae_config, loss_summary_line, reconstruction_line, render_loss_table, train_vae,
};
use tp5::vae_eval::{
    contrast_line, decode_grid, manifold_line, manifold_metrics, new_emoji_line, prior_line, render_acceptance,
    search_new_emoji, GRID_HALF_RANGE,
};

const QUICK_EPOCHS: usize = 100;

fn quick_opts() -> Ej2Options {
    Ej2Options { epochs: Some(QUICK_EPOCHS), grid: 7, draws: 200, ..Ej2Options::default() }
}

fn quick_run() -> &'static Ej2Run {
    static R: OnceLock<Ej2Run> = OnceLock::new();
    R.get_or_init(|| run_ej2(&quick_opts()).expect("corrida rapida"))
}

fn report_lines(run: &Ej2Run) -> Vec<String> {
    vec![
        render_loss_table(&run.history),
        loss_summary_line(&run.history),
        reconstruction_line(&run.recon),
        manifold_line(&run.grid, &run.sep),
        new_emoji_line(&run.search, &run.ds, &run.sep),
        prior_line(&run.prior, &run.sep),
        prior_line(&run.plain_prior, &run.sep),
        contrast_line(&run.contrast),
        render_acceptance(&run.checks),
        result_line(run),
    ]
}

#[test]
fn ej2_run_is_deterministic() {
    let a = quick_run();
    let b = run_ej2(&quick_opts()).unwrap();
    assert_eq!(report_lines(a), report_lines(&b));
}

#[test]
fn ej2_run_fields_are_consistent() {
    let run = quick_run();
    assert_eq!(run.grid.n_cells, 49);
    assert_eq!(run.grid_decoded.nrows(), 49);
    assert_eq!(run.plain_grid_decoded.nrows(), 49);
    assert_eq!(run.checks.len(), 7);
    assert_eq!(run.contrast.vae_prior, run.prior);
    assert_eq!(run.contrast.plain_prior, run.plain_prior);
    assert_eq!(run.contrast.vae_grid, run.grid);
    assert_eq!(run.contrast.plain_grid, run.plain_grid);
}

#[test]
fn ej2_numbers_equal_the_individual_computations() {
    let run = quick_run();
    let opts = quick_opts();
    let ds = build_emojis();
    let sep = separation(&ds);
    let mut cfg = chosen_vae_config();
    cfg.epochs = QUICK_EPOCHS;
    let (vae, history) = train_vae(&ds, &cfg).unwrap();
    assert_eq!(run.cfg.epochs, QUICK_EPOCHS);
    assert_eq!(render_loss_table(&run.history), render_loss_table(&history));
    let decoded: Array2<f64> = decode_grid(&vae, 7, GRID_HALF_RANGE);
    assert_eq!(run.grid, manifold_metrics(&decoded, 7, &ds, &sep));
    let search = search_new_emoji(&vae, &ds, opts.seed, opts.draws);
    assert_eq!(new_emoji_line(&run.search, &run.ds, &run.sep), new_emoji_line(&search, &ds, &sep));
    assert_eq!(run.search.examined, search.examined);
    assert_eq!(run.search.accepted.len(), search.accepted.len());
}

#[test]
fn options_are_validated() {
    let bad_epochs = Ej2Options { epochs: Some(0), ..quick_opts() };
    let bad_grid = Ej2Options { grid: 2, ..quick_opts() };
    let bad_draws = Ej2Options { draws: 0, ..quick_opts() };
    assert!(run_ej2(&bad_epochs).is_err());
    assert!(run_ej2(&bad_grid).is_err());
    assert!(run_ej2(&bad_draws).is_err());
}

#[test]
fn dataset_and_separation_lines_have_the_exact_format() {
    assert_eq!(dataset_line(), "emojis: 36 imagenes de 24x24 (576 valores en [0,1], 1 = tinta)");
    let sep = separation(&build_emojis());
    assert_eq!(
        separation_line(&sep),
        format!(
            "separacion: minima {} px, mediana {:.1} px, radio de cercania {} px",
            sep.min, sep.median, sep.near_radius
        )
    );
}

#[test]
fn result_line_reports_ok_and_failure() {
    let run = quick_run();
    let all_pass = run.checks.iter().all(|c| c.pass) && !run.search.accepted.is_empty();
    let line = result_line(run);
    if all_pass {
        assert!(line.starts_with("RESULTADO EJ2: OK"), "{line}");
    } else {
        assert!(line.starts_with("RESULTADO EJ2: FALLA"), "{line}");
    }

    let mut failed = run.clone();
    failed.checks[0].pass = false;
    let line = result_line(&failed);
    assert!(line.starts_with("RESULTADO EJ2: FALLA"), "{line}");
    assert!(line.contains(failed.checks[0].name), "{line}");

    let mut empty = run.clone();
    empty.search.accepted.clear();
    let line = result_line(&empty);
    assert!(line.starts_with("RESULTADO EJ2: FALLA"), "{line}");
    assert!(line.contains("no se encontro ningun emoji nuevo"), "{line}");
}
