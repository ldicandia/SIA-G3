//! Escritura de PNG con plotters.

use tp5::plots::{plot_series, Series};

fn unique_tmp_dir(tag: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("tp5-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    dir
}

fn series(name: &str, f: impl Fn(f64) -> f64) -> Series {
    Series { name: name.to_string(), points: (1..=50).map(|i| (i as f64, f(i as f64))).collect() }
}

#[test]
fn plot_series_writes_png() {
    let dir = unique_tmp_dir("plots");
    let path = dir.join("a.png");
    let s = [series("uno", |x| 1.0 / x), series("dos", |x| 2.0 / (x + 1.0))];
    plot_series(&path, "titulo", "epoca", "perdida", &s, false).unwrap();
    let bytes = std::fs::read(&path).unwrap();
    assert_eq!(&bytes[..8], &[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]);
    assert!(bytes.len() > 1000);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn plot_series_rejects_empty_input() {
    let dir = unique_tmp_dir("plots-empty");
    assert!(plot_series(&dir.join("e.png"), "t", "x", "y", &[], false).is_err());
    let vacia = [Series { name: "v".into(), points: vec![] }];
    assert!(plot_series(&dir.join("e.png"), "t", "x", "y", &vacia, false).is_err());
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn log_axis_tolerates_zero_values() {
    let dir = unique_tmp_dir("plots-log");
    let s = [series("con cero", |x| if x < 10.0 { 0.0 } else { 1.0 / x })];
    plot_series(&dir.join("l.png"), "t", "x", "y", &s, true).unwrap();
    let _ = std::fs::remove_dir_all(&dir);
}
