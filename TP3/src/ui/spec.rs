//! Pure logic of the UI: run requests, hyperparameter bounds and the study TOML
//! written for each run. Hyperparameters are rejected, never clamped, because
//! the stored config is the audit record of the run.

use std::collections::HashSet;

use serde::{Deserialize, Serialize};

use crate::{
    digits::{
        config::{CandidateConfig, DigitStudyConfig, OptimizerKind},
        data::{DIGIT_CLASSES, IMAGE_PIXELS},
    },
    model::{Activation, Initialization},
};

/// Name of the inherited baseline candidate that Exercise 3 always retrains.
pub const RESERVED_BASELINE_NAME: &str = "exercise2_winner_on_more_digits";
pub const NAME_MAX_CHARS: usize = 32;
pub const AXIS_MAX_CHARS: usize = 32;
pub const TOPOLOGY_MIN_LAYERS: usize = 3;
pub const TOPOLOGY_MAX_LAYERS: usize = 8;
pub const HIDDEN_WIDTH_MAX: usize = 2048;
pub const PARAMETER_COUNT_MAX: usize = 5_000_000;
pub const LEARNING_RATE_MAX: f64 = 10.0;
pub const BATCH_SIZE_MAX: usize = 4096;
pub const MAX_EPOCHS_MAX: usize = 1000;
pub const PATIENCE_MAX: usize = 1000;
pub const LR_DECAY_EVERY_MAX: usize = 10_000;
pub const AUGMENT_SHIFT_MAX: f64 = 28.0;
pub const AUGMENT_ROTATION_MAX: f64 = 180.0;
pub const VALIDATION_RATIO_MIN: f64 = 0.05;
pub const VALIDATION_RATIO_MAX: f64 = 0.5;

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Exercise {
    #[serde(rename = "exercise2")]
    Exercise2,
    #[serde(rename = "exercise3")]
    Exercise3,
}

impl Exercise {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Exercise2 => "exercise2",
            Self::Exercise3 => "exercise3",
        }
    }

    pub fn parse(text: &str) -> Option<Self> {
        match text {
            "exercise2" => Some(Self::Exercise2),
            "exercise3" => Some(Self::Exercise3),
            _ => None,
        }
    }
}

/// Fields of `CandidateConfig` a sweep may vary (everything except the
/// bookkeeping fields `name` and `axis`).
pub const SWEEP_FIELDS: [&str; 20] = [
    "learning_rate",
    "batch_size",
    "max_epochs",
    "patience",
    "min_delta",
    "seed",
    "momentum",
    "beta1",
    "beta2",
    "epsilon",
    "weight_decay",
    "dropout",
    "lr_decay_factor",
    "lr_decay_every",
    "augment_shift",
    "augment_rotation",
    "hidden_activation",
    "optimizer",
    "initialization",
    "topology",
];
pub const SWEEP_VALUES_MAX: usize = 24;
/// Generated sweep names (`BASE_field_token`) may be longer than a typed name.
pub const SWEEP_NAME_MAX_CHARS: usize = 80;
pub const DEFAULT_SWEEP_PARALLEL: usize = 2;

#[derive(Clone, Debug, Deserialize)]
pub struct SweepSpec {
    pub field: String,
    pub values: Vec<String>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct RunRequest {
    pub exercise: Exercise,
    #[serde(default = "default_validation_ratio")]
    pub validation_ratio: f64,
    #[serde(default = "default_split_seed")]
    pub split_seed: u64,
    pub candidate: CandidateConfig,
    #[serde(default)]
    pub sweep: Option<SweepSpec>,
    #[serde(default)]
    pub parallel: Option<usize>,
}

pub fn default_validation_ratio() -> f64 {
    0.2
}

pub fn default_split_seed() -> u64 {
    42
}

/// Shortens user-provided text before it is echoed in an error message.
fn clip(text: &str) -> String {
    text.chars().take(40).collect()
}

fn valid_token(text: &str) -> bool {
    text.chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
}

fn check_range(
    field: &str,
    value: f64,
    low: f64,
    low_inclusive: bool,
    high: f64,
    high_inclusive: bool,
) -> Result<(), String> {
    let above = if low_inclusive {
        value >= low
    } else {
        value > low
    };
    let below = if high_inclusive {
        value <= high
    } else {
        value < high
    };
    if value.is_finite() && above && below {
        Ok(())
    } else {
        let open = if low_inclusive { '[' } else { '(' };
        let close = if high_inclusive { ']' } else { ')' };
        Err(format!("{field} must be in {open}{low}, {high}{close}"))
    }
}

fn check_usize(field: &str, value: usize, low: usize, high: usize) -> Result<(), String> {
    if (low..=high).contains(&value) {
        Ok(())
    } else {
        Err(format!("{field} must be between {low} and {high}"))
    }
}

pub fn validate_validation_ratio(ratio: f64) -> Result<(), String> {
    check_range(
        "validation_ratio",
        ratio,
        VALIDATION_RATIO_MIN,
        true,
        VALIDATION_RATIO_MAX,
        true,
    )
}

/// Validates a candidate as typed in the form (name up to 32 characters).
pub fn validate_candidate(candidate: &CandidateConfig) -> Result<(), String> {
    validate_candidate_with(candidate, NAME_MAX_CHARS)
}

/// Validates a candidate produced by a sweep, whose generated name may be longer.
pub fn validate_expanded(candidate: &CandidateConfig, name_max_chars: usize) -> Result<(), String> {
    validate_candidate_with(candidate, name_max_chars)
}

fn validate_candidate_with(
    candidate: &CandidateConfig,
    name_max_chars: usize,
) -> Result<(), String> {
    let name_chars = candidate.name.chars().count();
    if name_chars == 0 || name_chars > name_max_chars || !valid_token(&candidate.name) {
        return Err(format!(
            "name must be 1 to {name_max_chars} characters of letters, digits, '_' or '-'"
        ));
    }
    if candidate.name == RESERVED_BASELINE_NAME {
        return Err("name is reserved for the inherited Exercise 2 baseline".into());
    }
    if candidate.axis.chars().count() > AXIS_MAX_CHARS || !valid_token(&candidate.axis) {
        return Err(format!(
            "axis must be at most {AXIS_MAX_CHARS} characters of letters, digits, '_' or '-'"
        ));
    }
    let topology = &candidate.topology;
    if !(TOPOLOGY_MIN_LAYERS..=TOPOLOGY_MAX_LAYERS).contains(&topology.len()) {
        return Err(format!(
            "topology must have between {TOPOLOGY_MIN_LAYERS} and {TOPOLOGY_MAX_LAYERS} layers"
        ));
    }
    if topology.first() != Some(&IMAGE_PIXELS) {
        return Err(format!("topology must start with {IMAGE_PIXELS} inputs"));
    }
    if topology.last() != Some(&DIGIT_CLASSES) {
        return Err(format!("topology must end with {DIGIT_CLASSES} outputs"));
    }
    if topology[1..topology.len() - 1]
        .iter()
        .any(|&width| !(1..=HIDDEN_WIDTH_MAX).contains(&width))
    {
        return Err(format!(
            "hidden layer widths must be between 1 and {HIDDEN_WIDTH_MAX}"
        ));
    }
    if candidate.parameter_count() > PARAMETER_COUNT_MAX {
        return Err(format!(
            "topology has more than {PARAMETER_COUNT_MAX} parameters"
        ));
    }
    check_range(
        "learning_rate",
        candidate.learning_rate,
        0.0,
        false,
        LEARNING_RATE_MAX,
        true,
    )?;
    check_usize("batch_size", candidate.batch_size, 1, BATCH_SIZE_MAX)?;
    check_usize("max_epochs", candidate.max_epochs, 1, MAX_EPOCHS_MAX)?;
    check_usize("patience", candidate.patience, 1, PATIENCE_MAX)?;
    check_range("min_delta", candidate.min_delta, 0.0, true, 1.0, true)?;
    check_range("momentum", candidate.momentum, 0.0, true, 1.0, false)?;
    check_range("beta1", candidate.beta1, 0.0, true, 1.0, false)?;
    check_range("beta2", candidate.beta2, 0.0, true, 1.0, false)?;
    check_range("epsilon", candidate.epsilon, 0.0, false, 1.0, true)?;
    check_range("weight_decay", candidate.weight_decay, 0.0, true, 1.0, true)?;
    check_range("dropout", candidate.dropout, 0.0, true, 1.0, false)?;
    check_range(
        "lr_decay_factor",
        candidate.lr_decay_factor,
        0.0,
        false,
        1.0,
        true,
    )?;
    if candidate.lr_decay_every > LR_DECAY_EVERY_MAX {
        return Err(format!(
            "lr_decay_every must be at most {LR_DECAY_EVERY_MAX}"
        ));
    }
    check_range(
        "augment_shift",
        candidate.augment_shift,
        0.0,
        true,
        AUGMENT_SHIFT_MAX,
        true,
    )?;
    check_range(
        "augment_rotation",
        candidate.augment_rotation,
        0.0,
        true,
        AUGMENT_ROTATION_MAX,
        true,
    )?;
    // The engine's own check, so the UI can never accept what the CLI rejects.
    candidate
        .validate()
        .map_err(|error| clip(&error.to_string()))
}

/// Builds the `config.toml` of one run: a study with exactly one candidate.
pub fn study_toml(
    validation_ratio: f64,
    split_seed: u64,
    candidate: &CandidateConfig,
) -> Result<String, String> {
    let study = DigitStudyConfig {
        validation_ratio,
        split_seed,
        repeats: 1,
        candidates: vec![candidate.clone()],
    };
    study.validate().map_err(|error| error.to_string())?;
    toml::to_string_pretty(&study).map_err(|error| error.to_string())
}

fn parse_number<T: std::str::FromStr>(field: &str, raw: &str) -> Result<T, String> {
    raw.parse::<T>()
        .map_err(|_| format!("{field}: invalid value '{}'", clip(raw)))
}

fn parse_finite(field: &str, raw: &str) -> Result<f64, String> {
    let value = parse_number::<f64>(field, raw)?;
    if value.is_finite() {
        Ok(value)
    } else {
        Err(format!("{field}: invalid value '{}'", clip(raw)))
    }
}

fn parse_enum<T: serde::de::DeserializeOwned>(field: &str, raw: &str) -> Result<T, String> {
    serde_json::from_value(serde_json::Value::String(raw.to_lowercase()))
        .map_err(|_| format!("{field}: invalid value '{}'", clip(raw)))
}

/// Sets one sweepable field from its textual form. Accepted enum names track
/// the engine's serde names.
pub fn apply_field(candidate: &mut CandidateConfig, field: &str, raw: &str) -> Result<(), String> {
    let raw = raw.trim();
    match field {
        "learning_rate" => candidate.learning_rate = parse_finite(field, raw)?,
        "batch_size" => candidate.batch_size = parse_number(field, raw)?,
        "max_epochs" => candidate.max_epochs = parse_number(field, raw)?,
        "patience" => candidate.patience = parse_number(field, raw)?,
        "min_delta" => candidate.min_delta = parse_finite(field, raw)?,
        "seed" => candidate.seed = parse_number(field, raw)?,
        "momentum" => candidate.momentum = parse_finite(field, raw)?,
        "beta1" => candidate.beta1 = parse_finite(field, raw)?,
        "beta2" => candidate.beta2 = parse_finite(field, raw)?,
        "epsilon" => candidate.epsilon = parse_finite(field, raw)?,
        "weight_decay" => candidate.weight_decay = parse_finite(field, raw)?,
        "dropout" => candidate.dropout = parse_finite(field, raw)?,
        "lr_decay_factor" => candidate.lr_decay_factor = parse_finite(field, raw)?,
        "lr_decay_every" => candidate.lr_decay_every = parse_number(field, raw)?,
        "augment_shift" => candidate.augment_shift = parse_finite(field, raw)?,
        "augment_rotation" => candidate.augment_rotation = parse_finite(field, raw)?,
        "hidden_activation" => candidate.hidden_activation = parse_enum::<Activation>(field, raw)?,
        "optimizer" => candidate.optimizer = parse_enum::<OptimizerKind>(field, raw)?,
        "initialization" => candidate.initialization = parse_enum::<Initialization>(field, raw)?,
        "topology" => {
            candidate.topology = raw
                .split(',')
                .map(|part| parse_number::<usize>(field, part.trim()))
                .collect::<Result<Vec<_>, _>>()?;
        }
        other => return Err(format!("{} cannot be swept", clip(other))),
    }
    Ok(())
}

/// Expands a one-axis sweep into one candidate per value.
pub fn expand_sweep(
    base: &CandidateConfig,
    sweep: &SweepSpec,
) -> Result<Vec<CandidateConfig>, String> {
    if !SWEEP_FIELDS.contains(&sweep.field.as_str()) {
        return Err(format!("{} cannot be swept", clip(&sweep.field)));
    }
    if sweep.values.is_empty() || sweep.values.len() > SWEEP_VALUES_MAX {
        return Err(format!(
            "a sweep needs between 1 and {SWEEP_VALUES_MAX} values"
        ));
    }
    let mut names = HashSet::new();
    let mut candidates = Vec::with_capacity(sweep.values.len());
    for raw in &sweep.values {
        let value = raw.trim();
        if value.is_empty() {
            return Err("sweep values must not be empty".into());
        }
        let mut candidate = base.clone();
        apply_field(&mut candidate, &sweep.field, value)?;
        let token: String = value
            .chars()
            .map(|c| if c.is_ascii_alphanumeric() { c } else { '-' })
            .collect();
        candidate.axis = sweep.field.clone();
        candidate.name = format!("{}_{}_{}", base.name, sweep.field, token);
        validate_expanded(&candidate, SWEEP_NAME_MAX_CHARS)
            .map_err(|message| format!("sweep value '{}': {message}", clip(value)))?;
        if !names.insert(candidate.name.clone()) {
            return Err("duplicate sweep value".into());
        }
        candidates.push(candidate);
    }
    Ok(candidates)
}

#[derive(Clone, Debug)]
pub struct PlannedBatch {
    pub candidates: Vec<CandidateConfig>,
    /// `Some` only for sweeps: the per-sweep concurrency cap.
    pub parallel: Option<usize>,
}

/// Validates a request and expands its sweep. Parallelism is the one value
/// clamped rather than rejected: it is an operational knob, not part of the
/// run's recorded configuration.
pub fn plan_request(request: &RunRequest, max_parallel: usize) -> Result<PlannedBatch, String> {
    validate_validation_ratio(request.validation_ratio)?;
    validate_candidate(&request.candidate)?;
    match &request.sweep {
        None => Ok(PlannedBatch {
            candidates: vec![request.candidate.clone()],
            parallel: None,
        }),
        Some(sweep) => {
            let candidates = expand_sweep(&request.candidate, sweep)?;
            let parallel = request
                .parallel
                .unwrap_or(DEFAULT_SWEEP_PARALLEL)
                .clamp(1, max_parallel.max(1));
            Ok(PlannedBatch {
                candidates,
                parallel: Some(parallel),
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid() -> CandidateConfig {
        CandidateConfig {
            name: "run1".into(),
            ..CandidateConfig::default()
        }
    }

    fn with(edit: impl FnOnce(&mut CandidateConfig)) -> CandidateConfig {
        let mut candidate = valid();
        edit(&mut candidate);
        candidate
    }

    #[test]
    fn default_candidate_with_a_valid_name_passes() {
        assert!(validate_candidate(&valid()).is_ok());
        assert!(validate_candidate(&CandidateConfig::default()).is_ok());
    }

    #[test]
    fn every_bound_is_enforced_at_its_edge() {
        type Edit = fn(&mut CandidateConfig);
        let cases: Vec<(&str, Edit, Edit)> = vec![
            (
                "learning_rate",
                |c| c.learning_rate = LEARNING_RATE_MAX,
                |c| c.learning_rate = LEARNING_RATE_MAX + 0.1,
            ),
            (
                "learning_rate zero",
                |c| c.learning_rate = 1e-9,
                |c| c.learning_rate = 0.0,
            ),
            (
                "batch_size",
                |c| c.batch_size = BATCH_SIZE_MAX,
                |c| c.batch_size = BATCH_SIZE_MAX + 1,
            ),
            (
                "batch_size zero",
                |c| c.batch_size = 1,
                |c| c.batch_size = 0,
            ),
            (
                "max_epochs",
                |c| c.max_epochs = MAX_EPOCHS_MAX,
                |c| c.max_epochs = MAX_EPOCHS_MAX + 1,
            ),
            (
                "max_epochs zero",
                |c| c.max_epochs = 1,
                |c| c.max_epochs = 0,
            ),
            (
                "patience",
                |c| c.patience = PATIENCE_MAX,
                |c| c.patience = PATIENCE_MAX + 1,
            ),
            ("patience zero", |c| c.patience = 1, |c| c.patience = 0),
            ("min_delta", |c| c.min_delta = 1.0, |c| c.min_delta = 1.01),
            (
                "min_delta neg",
                |c| c.min_delta = 0.0,
                |c| c.min_delta = -0.1,
            ),
            ("momentum", |c| c.momentum = 0.999, |c| c.momentum = 1.0),
            ("beta1", |c| c.beta1 = 0.0, |c| c.beta1 = 1.0),
            ("beta2", |c| c.beta2 = 0.9999, |c| c.beta2 = 1.0),
            ("epsilon", |c| c.epsilon = 1.0, |c| c.epsilon = 1.5),
            ("epsilon zero", |c| c.epsilon = 1e-12, |c| c.epsilon = 0.0),
            (
                "weight_decay",
                |c| c.weight_decay = 1.0,
                |c| c.weight_decay = 1.1,
            ),
            ("dropout", |c| c.dropout = 0.99, |c| c.dropout = 1.0),
            (
                "lr_decay_factor",
                |c| c.lr_decay_factor = 1.0,
                |c| c.lr_decay_factor = 1.1,
            ),
            (
                "lr_decay_factor zero",
                |c| c.lr_decay_factor = 0.01,
                |c| c.lr_decay_factor = 0.0,
            ),
            (
                "lr_decay_every",
                |c| c.lr_decay_every = LR_DECAY_EVERY_MAX,
                |c| c.lr_decay_every = LR_DECAY_EVERY_MAX + 1,
            ),
            (
                "augment_shift",
                |c| c.augment_shift = AUGMENT_SHIFT_MAX,
                |c| c.augment_shift = AUGMENT_SHIFT_MAX + 0.5,
            ),
            (
                "augment_rotation",
                |c| c.augment_rotation = AUGMENT_ROTATION_MAX,
                |c| c.augment_rotation = AUGMENT_ROTATION_MAX + 1.0,
            ),
        ];
        for (label, inside, outside) in cases {
            assert!(validate_candidate(&with(inside)).is_ok(), "{label} inside");
            assert!(
                validate_candidate(&with(outside)).is_err(),
                "{label} outside"
            );
        }
    }

    #[test]
    fn non_finite_numbers_are_rejected() {
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(validate_candidate(&with(|c| c.learning_rate = value)).is_err());
            assert!(validate_candidate(&with(|c| c.dropout = value)).is_err());
            assert!(validate_candidate(&with(|c| c.augment_rotation = value)).is_err());
        }
    }

    #[test]
    fn names_are_restricted() {
        assert!(validate_candidate(&with(|c| c.name = "a".repeat(32))).is_ok());
        for bad in ["", "has space", "semi;colon", "../x", "naïve"] {
            assert!(
                validate_candidate(&with(|c| c.name = bad.into())).is_err(),
                "{bad:?}"
            );
        }
        assert!(validate_candidate(&with(|c| c.name = "a".repeat(33))).is_err());
        assert!(validate_expanded(&with(|c| c.name = "a".repeat(80)), 80).is_ok());
        assert!(validate_expanded(&with(|c| c.name = "a".repeat(81)), 80).is_err());
        assert!(validate_candidate(&with(|c| c.name = RESERVED_BASELINE_NAME.into())).is_err());
        assert!(validate_candidate(&with(|c| c.axis = "bad axis".into())).is_err());
        assert!(validate_candidate(&with(|c| c.axis = "learning_rate".into())).is_ok());
    }

    #[test]
    fn error_messages_never_echo_long_user_text() {
        let long = "x y".repeat(100);
        let message = validate_candidate(&with(|c| c.name = long.clone())).unwrap_err();
        assert!(!message.contains(&long));
        assert!(message.len() < 200);
    }

    #[test]
    fn topology_rules() {
        assert!(validate_candidate(&with(|c| c.topology = vec![100, 8, 10])).is_err());
        assert!(validate_candidate(&with(|c| c.topology = vec![784, 8, 9])).is_err());
        assert!(validate_candidate(&with(|c| c.topology = vec![784, 10])).is_err());
        assert!(
            validate_candidate(&with(|c| c.topology = vec![784, 1, 1, 1, 1, 1, 1, 1, 10])).is_err()
        );
        assert!(
            validate_candidate(&with(|c| c.topology = vec![784, 1, 1, 1, 1, 1, 1, 10])).is_ok()
        );
        assert!(validate_candidate(&with(|c| c.topology = vec![784, 0, 10])).is_err());
        assert!(validate_candidate(&with(|c| c.topology = vec![784, 2049, 10])).is_err());
        assert!(validate_candidate(&with(|c| c.topology = vec![784, 2048, 10])).is_ok());
        // 784*2048 + 2048*2048 * k exceeds five million parameters
        assert!(
            validate_candidate(&with(|c| c.topology = vec![784, 2048, 2048, 2048, 10])).is_err()
        );
    }

    #[test]
    fn validation_ratio_edges() {
        assert!(validate_validation_ratio(0.05).is_ok());
        assert!(validate_validation_ratio(0.5).is_ok());
        assert!(validate_validation_ratio(0.04).is_err());
        assert!(validate_validation_ratio(0.51).is_err());
        assert!(validate_validation_ratio(f64::NAN).is_err());
    }

    #[test]
    fn study_toml_round_trips_and_validates() {
        let text = study_toml(0.2, 7, &valid()).unwrap();
        let study: DigitStudyConfig = toml::from_str(&text).unwrap();
        study.validate().unwrap();
        assert_eq!(study.candidates.len(), 1);
        assert_eq!(study.candidates[0].name, "run1");
        assert_eq!(study.split_seed, 7);
        assert!(study_toml(1.5, 7, &valid()).is_err());
    }

    fn sweep(field: &str, values: &[&str]) -> SweepSpec {
        SweepSpec {
            field: field.into(),
            values: values.iter().map(|v| (*v).to_owned()).collect(),
        }
    }

    fn request(sweep: Option<SweepSpec>, parallel: Option<usize>) -> RunRequest {
        RunRequest {
            exercise: Exercise::Exercise2,
            validation_ratio: 0.2,
            split_seed: 42,
            candidate: valid(),
            sweep,
            parallel,
        }
    }

    #[test]
    fn apply_field_parses_and_normalizes() {
        let mut c = valid();
        apply_field(&mut c, "learning_rate", "0.05").unwrap();
        assert_eq!(c.learning_rate, 0.05);
        apply_field(&mut c, "optimizer", " SGD ").unwrap();
        assert_eq!(c.optimizer, OptimizerKind::Sgd);
        apply_field(&mut c, "topology", "784,128,10").unwrap();
        assert_eq!(c.topology, vec![784, 128, 10]);
        apply_field(&mut c, "hidden_activation", "Tanh").unwrap();
        assert_eq!(c.hidden_activation, Activation::Tanh);
        apply_field(&mut c, "initialization", "he").unwrap();
        assert_eq!(c.initialization, Initialization::He);
        apply_field(&mut c, "seed", "9").unwrap();
        assert_eq!(c.seed, 9);
    }

    #[test]
    fn apply_field_rejects_bad_input_naming_the_field() {
        let mut c = valid();
        for (field, raw) in [
            ("hidden_activation", "swish"),
            ("dropout", "abc"),
            ("topology", "784,x,10"),
            ("learning_rate", "NaN"),
            ("learning_rate", "inf"),
            ("batch_size", "-3"),
            ("name", "other"),
            ("axis", "other"),
            ("exercise", "x"),
        ] {
            let message = apply_field(&mut c, field, raw).unwrap_err();
            assert!(message.contains(field), "{message}");
            assert!(message.len() < 120);
        }
        let long = "9".repeat(500);
        let message = apply_field(&mut c, "dropout", &format!("{long}x")).unwrap_err();
        assert!(!message.contains(&long));
    }

    #[test]
    fn expand_sweep_names_and_copies_the_base() {
        let base = with(|c| c.batch_size = 128);
        let candidates =
            expand_sweep(&base, &sweep("learning_rate", &["0.1", "0.01", "0.001"])).unwrap();
        let names: Vec<_> = candidates.iter().map(|c| c.name.as_str()).collect();
        assert_eq!(
            names,
            [
                "run1_learning_rate_0-1",
                "run1_learning_rate_0-01",
                "run1_learning_rate_0-001"
            ]
        );
        assert!(candidates.iter().all(|c| c.axis == "learning_rate"));
        assert!(candidates.iter().all(|c| c.batch_size == 128));
        assert_eq!(candidates[1].learning_rate, 0.01);
        let topologies =
            expand_sweep(&valid(), &sweep("topology", &["784,64,10", "784,128,10"])).unwrap();
        assert_eq!(topologies[1].name, "run1_topology_784-128-10");
    }

    #[test]
    fn expand_sweep_rejects_bad_sweeps() {
        let base = valid();
        assert!(expand_sweep(&base, &sweep("learning_rate", &[])).is_err());
        let many: Vec<String> = (1..=25).map(|n| format!("0.{n}")).collect();
        let many_refs: Vec<&str> = many.iter().map(String::as_str).collect();
        assert!(expand_sweep(&base, &sweep("learning_rate", &many_refs)).is_err());
        let max: Vec<&str> = many_refs[..24].to_vec();
        assert_eq!(
            expand_sweep(&base, &sweep("learning_rate", &max))
                .unwrap()
                .len(),
            24
        );
        assert!(expand_sweep(&base, &sweep("dropout", &["0.1", " 0.1 "])).is_err());
        assert!(expand_sweep(&base, &sweep("name", &["a"])).is_err());
        assert!(expand_sweep(&base, &sweep("axis", &["a"])).is_err());
        assert!(expand_sweep(&base, &sweep("dropout", &["1.5"])).is_err());
        assert!(expand_sweep(&base, &sweep("topology", &["784,64,9"])).is_err());
        assert!(expand_sweep(&base, &sweep("dropout", &["0.1", "  "])).is_err());
        assert!(expand_sweep(&base, &sweep("dropout", &["x"])).is_err());
    }

    #[test]
    fn plan_request_handles_parallelism_and_validation() {
        let single = plan_request(&request(None, Some(3)), 4).unwrap();
        assert_eq!(single.candidates.len(), 1);
        assert_eq!(single.parallel, None);

        let swept = |parallel| {
            plan_request(
                &request(Some(sweep("dropout", &["0.1", "0.2"])), parallel),
                4,
            )
            .unwrap()
        };
        assert_eq!(swept(None).parallel, Some(2));
        assert_eq!(swept(Some(0)).parallel, Some(1));
        assert_eq!(swept(Some(99)).parallel, Some(4));
        assert_eq!(swept(Some(3)).parallel, Some(3));
        let clamped = plan_request(&request(Some(sweep("dropout", &["0.1"])), None), 1).unwrap();
        assert_eq!(clamped.parallel, Some(1));
        assert_eq!(swept(None).candidates.len(), 2);

        let mut bad = request(Some(sweep("dropout", &["0.1"])), None);
        bad.candidate.name = "bad name".into();
        assert!(plan_request(&bad, 4).is_err());
        let mut bad_ratio = request(None, None);
        bad_ratio.validation_ratio = 0.9;
        assert!(plan_request(&bad_ratio, 4).is_err());
    }

    #[test]
    fn exercise_serde_names() {
        let parsed: Exercise = serde_json::from_str("\"exercise3\"").unwrap();
        assert_eq!(parsed, Exercise::Exercise3);
        assert_eq!(
            serde_json::to_string(&Exercise::Exercise2).unwrap(),
            "\"exercise2\""
        );
        assert_eq!(Exercise::parse("exercise2"), Some(Exercise::Exercise2));
        assert_eq!(Exercise::parse("exercise4"), None);
    }
}
