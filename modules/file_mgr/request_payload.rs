//! Request-payload contracts for `POST .../train` and `POST .../convert`.
//!
//! Train body is the flat [`TrainingCfg`]. Convert body is internally tagged on
//! `converter_type`; file paths are converter-rooted (`tfjs/model.json` ->
//! `<workspace>/converters/...`), legacy leading `/` accepted+stripped for one
//! release. The validators add numeric range gates the type system can't express;
//! the manifest round-trip + SHA helpers are diagnostics/replay only (cfg unpersisted).

use crate::common::asset_path::{AssetPath, AssetPathError};
use crate::common::error::{Categorized, ErrorKind};
use crate::file_mgr::error::{FileError, metadata_parse_err};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

/// BC alias for callers still importing `TrainRequest`.
pub type TrainRequest = TrainingCfg;

/// Operator-tunable training hyperparameters; bounds enforced by
/// [`validate_training_cfg`]. Not `Eq` (f32 fields); equality/hash callers use
/// [`canonical_training_cfg_sha256`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrainingCfg {
    /// Bounds: `1..=1_000`.
    pub epochs: u32,
    /// Bounds: `1..=4_096`.
    pub batch_size: u32,
    /// Bounds: finite, `0.0 < lr <= 1.0`.
    pub learning_rate: f32,
    /// `None` lets the daemon pick per-job entropy; `Some(_)` pins replay.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    /// Per-class validation fraction, finite `0.0 <= split < 1.0`. `0.0` = full
    /// dataset + last-epoch head; `(0.0, 1.0)` = stratified deterministic
    /// per-class split + best-val-accuracy epoch, val-loss as tiebreaker
    /// (singleton classes rejected).
    #[serde(default)]
    pub validation_split: f32,
}

/// Failure shapes for [`ConverterPath::parse`]; all `UserInput` (400).
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum ConverterPathError {
    /// Empty input or `/` alone.
    #[error("converter path is empty")]
    Empty,
    #[error("converter path invalid: {0}")]
    Invalid(#[from] AssetPathError),
}

impl Categorized for ConverterPathError {
    fn kind(&self) -> ErrorKind {
        ErrorKind::UserInput
    }
}

/// Operator path to a file under `<workspace>/converters/`. Canonical wire form
/// is slashless `<sub>` (legacy leading `/` accepted, one release); stored as
/// [`AssetPath`] `converters/<sub>`, always serialized in canonical slashless
/// form regardless of input variant.
#[derive(Clone, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct ConverterPath {
    workspace_relative: AssetPath,
}

impl ConverterPath {
    pub fn parse(input: &str) -> Result<Self, ConverterPathError> {
        let stripped = input.strip_prefix('/').unwrap_or(input);
        if stripped.is_empty() {
            return Err(ConverterPathError::Empty);
        }
        let mut combined = String::with_capacity("converters/".len() + stripped.len());
        combined.push_str("converters/");
        combined.push_str(stripped);
        let workspace_relative = AssetPath::parse(&combined)?;
        Ok(Self { workspace_relative })
    }

    pub fn workspace_path(&self) -> &AssetPath {
        &self.workspace_relative
    }

    pub fn wire_form(&self) -> String {
        self.workspace_relative
            .as_str()
            .strip_prefix("converters/")
            .expect("ConverterPath invariant: workspace path starts with converters/")
            .to_owned()
    }
}

impl std::fmt::Display for ConverterPath {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.wire_form())
    }
}

impl TryFrom<String> for ConverterPath {
    type Error = ConverterPathError;
    fn try_from(value: String) -> Result<Self, Self::Error> {
        Self::parse(&value)
    }
}

impl TryFrom<&str> for ConverterPath {
    type Error = ConverterPathError;
    fn try_from(value: &str) -> Result<Self, Self::Error> {
        Self::parse(value)
    }
}

impl std::str::FromStr for ConverterPath {
    type Err = ConverterPathError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::parse(s)
    }
}

impl From<ConverterPath> for String {
    fn from(p: ConverterPath) -> Self {
        p.wire_form()
    }
}

/// Convert request body, internally tagged on `converter_type`; per-variant
/// `deny_unknown_fields` rejects stray keys after dispatch.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "converter_type", rename_all = "snake_case")]
pub enum ConvertRequest {
    Tfjs(TfjsConvertParams),
    /// `.alpkg` head import (operator-uploaded `.mpk`+`.json`, verified then
    /// published via the same rotation primitive training uses).
    Alpkg(AlpkgParams),
}

impl ConvertRequest {
    pub fn converter_type(&self) -> crate::common::workspace::ConverterType {
        match self {
            ConvertRequest::Tfjs(_) => crate::common::workspace::ConverterType::Tfjs,
            ConvertRequest::Alpkg(_) => crate::common::workspace::ConverterType::Alpkg,
        }
    }
}

/// TFJS convert payload. Operators name only the manifest + labels file; shards
/// are derived daemon-side by parsing `model.json`'s `weightsManifest[].paths`,
/// prepending the manifest dir, and re-validating each via [`AssetPath::parse`].
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TfjsConvertParams {
    pub model_json_path: ConverterPath,
    pub labels_path: ConverterPath,
    pub labels_format: LabelsFormat,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LabelsFormat {
    /// One label per line, blank lines stripped.
    Lines,
    /// TFJS `metadata.json`: labels under `wordLabels` (Teachable Machine) or
    /// `words` (Speech-Commands).
    TfjsMetadata,
}

/// `.alpkg`-import payload. Operator names only the `.json` manifest; the daemon
/// derives the sibling `.mpk` at `<parent>/<head_id>.mpk` (re-parsed through
/// [`AssetPath`]), stream-verifies it against the manifest's `size_bytes`/
/// `sha256`, then publishes via rotation. Re-import of a present head_id is
/// idempotent on matching `sha256`, else 409 `head_id_collision`.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AlpkgParams {
    pub manifest_path: ConverterPath,
}

/// Inclusive bounds on [`TrainingCfg`] numeric fields.
pub const MIN_EPOCHS: u32 = 1;
pub const MAX_EPOCHS: u32 = 1_000;
pub const MIN_BATCH_SIZE: u32 = 1;
pub const MAX_BATCH_SIZE: u32 = 4_096;
/// Inclusive upper bound on `learning_rate`; lower bound is the strict `> 0.0` finiteness check (no `MIN_LEARNING_RATE`).
pub const MAX_LEARNING_RATE: f32 = 1.0;

/// Validator failures; all `UserInput` (400). Not `Eq` (f32-carrying variants).
#[derive(Debug, Error, Clone, PartialEq)]
pub enum ValidationError {
    #[error("epochs out of range: got {got}, allowed {min}..={max}")]
    EpochsOutOfRange { got: u32, min: u32, max: u32 },
    #[error("batch_size out of range: got {got}, allowed {min}..={max}")]
    BatchSizeOutOfRange { got: u32, min: u32, max: u32 },
    #[error("learning_rate out of range: got {got}, allowed (0.0, {max}] and finite")]
    LearningRateOutOfRange { got: f32, max: f32 },
    #[error("validation_split out of range: got {got}, allowed [0.0, 1.0) and finite")]
    ValidationSplitOutOfRange { got: f32 },
    /// Rejected at the boundary since the sibling `.mpk` derivation needs a
    /// `.json` filename.
    #[error("alpkg manifest_path must end in `.json`: got {got}")]
    AlpkgManifestExtension { got: String },
}

impl Categorized for ValidationError {
    fn kind(&self) -> ErrorKind {
        ErrorKind::UserInput
    }
}

/// Numeric range validator for [`TrainingCfg`]. Run at the request boundary AND
/// at publish time so a hand-crafted manifest cannot smuggle out-of-range values.
pub fn validate_training_cfg(cfg: &TrainingCfg) -> Result<(), ValidationError> {
    if !(MIN_EPOCHS..=MAX_EPOCHS).contains(&cfg.epochs) {
        return Err(ValidationError::EpochsOutOfRange {
            got: cfg.epochs,
            min: MIN_EPOCHS,
            max: MAX_EPOCHS,
        });
    }
    if !(MIN_BATCH_SIZE..=MAX_BATCH_SIZE).contains(&cfg.batch_size) {
        return Err(ValidationError::BatchSizeOutOfRange {
            got: cfg.batch_size,
            min: MIN_BATCH_SIZE,
            max: MAX_BATCH_SIZE,
        });
    }
    if !cfg.learning_rate.is_finite()
        || cfg.learning_rate <= 0.0
        || cfg.learning_rate > MAX_LEARNING_RATE
    {
        return Err(ValidationError::LearningRateOutOfRange {
            got: cfg.learning_rate,
            max: MAX_LEARNING_RATE,
        });
    }
    // Half-open `[0.0, 1.0)`: 1.0 would leave no training data; `Range::contains`
    // rejects NaN/±∞, so finiteness is implicit. `seed` is unconstrained.
    if !(0.0..1.0).contains(&cfg.validation_split) {
        return Err(ValidationError::ValidationSplitOutOfRange {
            got: cfg.validation_split,
        });
    }
    Ok(())
}

/// Validates [`ConvertRequest`] beyond the path shape already enforced by
/// [`ConverterPath`] at deserialize. TFJS has no request-level checks (shards
/// derived daemon-side, DoS-gated by `ConvertLimits::max_shards`).
pub fn validate_convert_request(req: &ConvertRequest) -> Result<(), ValidationError> {
    match req {
        ConvertRequest::Tfjs(_params) => Ok(()),
        ConvertRequest::Alpkg(params) => validate_alpkg_params(params),
    }
}

/// Case-sensitive `.json` gate for [`AlpkgParams`] (daemon exports lowercase),
/// required for the sibling `.mpk` derivation to be well-defined.
fn validate_alpkg_params(params: &AlpkgParams) -> Result<(), ValidationError> {
    // Wire form for operator-facing 400s; strips only the leading prefix, so the
    // trailing `.json` test is unaffected.
    let got = params.manifest_path.wire_form();
    if !got.ends_with(".json") {
        return Err(ValidationError::AlpkgManifestExtension { got });
    }
    Ok(())
}

/// Hex-lowercase SHA-256 of `cfg`'s canonical JSON (serde field order is the
/// contract); diagnostic fingerprinting only, not persisted. `PartialEq`-equal
/// cfgs hash equal. Validates first so a caller skipping pre-validation gets a
/// typed error rather than silently hashing a NaN/inf field that serde_json
/// coerces to JSON `null` (it does not error on non-finite floats).
pub fn canonical_training_cfg_sha256(cfg: &TrainingCfg) -> Result<String, ValidationError> {
    validate_training_cfg(cfg)?;
    // Collapse -0.0 -> +0.0: serde renders them as distinct "-0.0"/"0.0" yet
    // they are `PartialEq`-equal, which would break equal-hash. Only
    // `validation_split` admits -0.0 (lr's `> 0.0` gate rejects it); clone only
    // for that rare case to keep the common path alloc-free.
    let bytes = if cfg.validation_split == 0.0 && cfg.validation_split.is_sign_negative() {
        let mut normalized = cfg.clone();
        normalized.validation_split = 0.0;
        serde_json::to_vec(&normalized)
    } else {
        serde_json::to_vec(cfg)
    }
    .expect("validated TrainingCfg serializes infallibly via serde_json::to_vec");
    let digest = Sha256::digest(&bytes);
    Ok(crate::common::hex::hex_lowercase(digest.as_slice()))
}

/// Typed [`TrainingCfg`] -> opaque [`serde_json::Value`]. Caller MUST pass a
/// validated cfg: non-finite floats do not error, they serialize silently to
/// JSON `null`, corrupting the manifest and breaking the f32 round-trip; the
/// `debug_assert` surfaces this in debug builds.
pub fn to_manifest_value(cfg: &TrainingCfg) -> serde_json::Value {
    debug_assert!(
        cfg.learning_rate.is_finite() && cfg.validation_split.is_finite(),
        "to_manifest_value requires a validate_training_cfg-clean (finite) TrainingCfg",
    );
    serde_json::to_value(cfg).expect("TrainingCfg serializes infallibly to serde_json::Value")
}

/// Opaque [`serde_json::Value`] -> typed [`TrainingCfg`]; failures become
/// [`FileError::MetadataParse`] so corruption is Internal (500), not operator-input.
pub fn from_manifest_value(value: &serde_json::Value) -> Result<TrainingCfg, FileError> {
    serde_json::from_value(value.clone())
        .map_err(|source| metadata_parse_err("<HeadManifest::training_cfg>", source))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::error::Categorized;

    fn good_cfg() -> TrainingCfg {
        TrainingCfg {
            epochs: 4,
            batch_size: 16,
            learning_rate: 1e-3,
            seed: Some(42),
            validation_split: 0.0,
        }
    }

    fn good_tfjs_params() -> TfjsConvertParams {
        TfjsConvertParams {
            model_json_path: ConverterPath::parse("/tfjs/model.json").unwrap(),
            labels_path: ConverterPath::parse("/tfjs/metadata.json").unwrap(),
            labels_format: LabelsFormat::TfjsMetadata,
        }
    }

    fn good_convert_request() -> ConvertRequest {
        ConvertRequest::Tfjs(good_tfjs_params())
    }

    #[test]
    fn train_request_round_trips_flat() {
        let body = r#"{"epochs":4,"batch_size":16,"learning_rate":0.001,"seed":42}"#;
        let req: TrainRequest = serde_json::from_str(body).unwrap();
        assert_eq!(req.epochs, 4);
        assert_eq!(req.batch_size, 16);
        assert!((req.learning_rate - 1e-3).abs() < 1e-9);
        assert_eq!(req.seed, Some(42));

        let back = serde_json::to_string(&req).unwrap();
        let v: serde_json::Value = serde_json::from_str(&back).unwrap();
        assert!(v.get("epochs").is_some());
        assert!(v.get("dataset_path").is_none(), "no dataset_path");
        assert!(v.get("training_cfg").is_none(), "no wrapper object");
    }

    #[test]
    fn train_request_seed_is_optional() {
        let body = r#"{"epochs":1,"batch_size":1,"learning_rate":0.5}"#;
        let req: TrainRequest = serde_json::from_str(body).unwrap();
        assert_eq!(req.seed, None);
    }

    #[test]
    fn train_request_rejects_round_1_wrapper_shape() {
        let body = r#"{"dataset_path":"audio","training_cfg":{"epochs":1,"batch_size":1,"learning_rate":0.001}}"#;
        let res: Result<TrainRequest, _> = serde_json::from_str(body);
        assert!(res.is_err(), "wrapper body must fail to parse");
    }

    #[test]
    fn train_request_rejects_unknown_fields() {
        let body = r#"{"epochs":1,"batch_size":1,"learning_rate":0.001,"momentum":0.9}"#;
        let res: Result<TrainRequest, _> = serde_json::from_str(body);
        assert!(res.is_err(), "stray field must be rejected");
    }

    #[test]
    fn training_cfg_validates_epoch_range() {
        let mut cfg = good_cfg();
        cfg.epochs = 0;
        assert!(matches!(
            validate_training_cfg(&cfg),
            Err(ValidationError::EpochsOutOfRange { .. })
        ));
        cfg.epochs = MAX_EPOCHS + 1;
        assert!(matches!(
            validate_training_cfg(&cfg),
            Err(ValidationError::EpochsOutOfRange { .. })
        ));
        cfg.epochs = MAX_EPOCHS;
        assert!(validate_training_cfg(&cfg).is_ok());
        cfg.epochs = MIN_EPOCHS;
        assert!(validate_training_cfg(&cfg).is_ok());
    }

    #[test]
    fn training_cfg_validates_batch_size_range() {
        let mut cfg = good_cfg();
        cfg.batch_size = 0;
        assert!(matches!(
            validate_training_cfg(&cfg),
            Err(ValidationError::BatchSizeOutOfRange { .. })
        ));
        cfg.batch_size = MAX_BATCH_SIZE + 1;
        assert!(matches!(
            validate_training_cfg(&cfg),
            Err(ValidationError::BatchSizeOutOfRange { .. })
        ));
        cfg.batch_size = MAX_BATCH_SIZE;
        assert!(validate_training_cfg(&cfg).is_ok());
    }

    #[test]
    fn training_cfg_validates_learning_rate_range() {
        let mut cfg = good_cfg();

        cfg.learning_rate = 0.0;
        assert!(matches!(
            validate_training_cfg(&cfg),
            Err(ValidationError::LearningRateOutOfRange { .. })
        ));

        cfg.learning_rate = -1e-3;
        assert!(matches!(
            validate_training_cfg(&cfg),
            Err(ValidationError::LearningRateOutOfRange { .. })
        ));

        cfg.learning_rate = f32::NAN;
        assert!(matches!(
            validate_training_cfg(&cfg),
            Err(ValidationError::LearningRateOutOfRange { .. })
        ));

        cfg.learning_rate = f32::INFINITY;
        assert!(matches!(
            validate_training_cfg(&cfg),
            Err(ValidationError::LearningRateOutOfRange { .. })
        ));

        cfg.learning_rate = MAX_LEARNING_RATE + 1.0;
        assert!(matches!(
            validate_training_cfg(&cfg),
            Err(ValidationError::LearningRateOutOfRange { .. })
        ));

        cfg.learning_rate = MAX_LEARNING_RATE;
        assert!(validate_training_cfg(&cfg).is_ok());

        cfg.learning_rate = 1e-6;
        assert!(validate_training_cfg(&cfg).is_ok());
    }

    #[test]
    fn training_cfg_seed_is_unconstrained() {
        let mut cfg = good_cfg();
        cfg.seed = None;
        assert!(validate_training_cfg(&cfg).is_ok());
        cfg.seed = Some(0);
        assert!(validate_training_cfg(&cfg).is_ok());
        cfg.seed = Some(u64::MAX);
        assert!(validate_training_cfg(&cfg).is_ok());
    }

    #[test]
    fn converter_path_round_trips_via_serde_string_canonical_form() {
        let p = ConverterPath::parse("tfjs/model.json").unwrap();
        let s = serde_json::to_string(&p).unwrap();
        assert_eq!(s, r#""tfjs/model.json""#);
        let back: ConverterPath = serde_json::from_str(&s).unwrap();
        assert_eq!(p, back);
        assert_eq!(p.workspace_path().as_str(), "converters/tfjs/model.json");
        assert_eq!(p.wire_form(), "tfjs/model.json");
    }

    #[test]
    fn converter_path_accepts_legacy_leading_slash_bc_shim() {
        let p_legacy = ConverterPath::parse("/tfjs/model.json").unwrap();
        let p_canonical = ConverterPath::parse("tfjs/model.json").unwrap();
        assert_eq!(p_legacy, p_canonical);
        assert_eq!(p_legacy.wire_form(), "tfjs/model.json");
        assert_eq!(
            p_legacy.workspace_path().as_str(),
            "converters/tfjs/model.json"
        );
    }

    #[test]
    fn converter_path_rejects_empty_input() {
        for bad in ["", "/"] {
            let err = ConverterPath::parse(bad).unwrap_err();
            assert!(
                matches!(err, ConverterPathError::Empty),
                "{bad:?} should reject as Empty; got {err:?}",
            );
            assert_eq!(err.kind(), ErrorKind::UserInput);
        }
    }

    #[test]
    fn converter_path_rejects_traversal_in_either_form() {
        for bad in [
            "..",
            "../etc",
            "a/../b",
            ".hidden/file",
            "/..",
            "/../etc",
            "/a/../b",
            "/.hidden/file",
        ] {
            let res = ConverterPath::parse(bad);
            assert!(res.is_err(), "{bad:?} must be rejected");
        }
    }

    #[test]
    fn converter_path_rejects_double_slash() {
        // `//...` strips one slash to `/...`, whose empty first component is rejected.
        let res = ConverterPath::parse("//tfjs/model.json");
        assert!(res.is_err());
    }

    #[test]
    fn converter_path_rejects_url_encoded_traversal() {
        for bad in ["%2E%2E/etc", "/%2E%2E/etc"] {
            let res = ConverterPath::parse(bad);
            assert!(
                res.is_err(),
                "URL-encoded traversal {bad:?} must be rejected"
            );
        }
    }

    #[test]
    fn convert_request_round_trips_tfjs() {
        let body = r#"{
            "converter_type": "tfjs",
            "model_json_path": "/tfjs/model.json",
            "labels_path": "/tfjs/metadata.json",
            "labels_format": "tfjs_metadata"
        }"#;
        let req: ConvertRequest = serde_json::from_str(body).unwrap();
        assert_eq!(
            req.converter_type(),
            crate::common::workspace::ConverterType::Tfjs
        );
        let ConvertRequest::Tfjs(p) = &req else {
            panic!("expected ConvertRequest::Tfjs");
        };
        assert_eq!(p.labels_format, LabelsFormat::TfjsMetadata);
        assert_eq!(
            p.model_json_path.workspace_path().as_str(),
            "converters/tfjs/model.json",
        );
        validate_convert_request(&req).expect("good shape");

        let serialized = serde_json::to_string(&req).unwrap();
        let v: serde_json::Value = serde_json::from_str(&serialized).unwrap();
        assert_eq!(v["converter_type"], "tfjs");
        // Serialize emits canonical slashless form despite legacy-slash input.
        assert_eq!(v["model_json_path"], "tfjs/model.json");
    }

    #[test]
    fn convert_request_rejects_unknown_converter_type() {
        let body = r#"{
            "converter_type": "onnx",
            "model_json_path": "/m",
            "labels_path": "/l",
            "labels_format": "lines"
        }"#;
        let res: Result<ConvertRequest, _> = serde_json::from_str(body);
        assert!(res.is_err(), "unknown converter_type must be rejected");
    }

    #[test]
    fn convert_request_rejects_missing_converter_type() {
        let body = r#"{
            "model_json_path": "/tfjs/model.json",
            "labels_path": "/tfjs/metadata.json",
            "labels_format": "tfjs_metadata"
        }"#;
        let res: Result<ConvertRequest, _> = serde_json::from_str(body);
        assert!(
            res.is_err(),
            "body without converter_type must fail to parse",
        );
    }

    #[test]
    fn convert_request_rejects_unknown_field_after_dispatch() {
        let body = r#"{
            "converter_type": "tfjs",
            "model_json_path": "/m",
            "labels_path": "/l",
            "labels_format": "lines",
            "stray": true
        }"#;
        let res: Result<ConvertRequest, _> = serde_json::from_str(body);
        assert!(
            res.is_err(),
            "stray field after converter dispatch must be rejected",
        );
    }

    #[test]
    fn convert_request_accepts_relative_paths_as_canonical_form() {
        for field in ["model_json_path", "labels_path"] {
            let mut v = serde_json::json!({
                "converter_type": "tfjs",
                "model_json_path": "/m",
                "labels_path": "/l",
                "labels_format": "lines",
            });
            v[field] = serde_json::Value::String("relative/path".into());
            let body = serde_json::to_string(&v).unwrap();
            let req: ConvertRequest =
                serde_json::from_str(&body).expect("slashless path is canonical");
            let ConvertRequest::Tfjs(p) = &req else {
                panic!("expected ConvertRequest::Tfjs");
            };
            let bound_field = match field {
                "model_json_path" => p.model_json_path.workspace_path().as_str(),
                "labels_path" => p.labels_path.workspace_path().as_str(),
                _ => unreachable!(),
            };
            assert_eq!(bound_field, "converters/relative/path");
        }
    }

    #[test]
    fn convert_request_round_trips_lines_format() {
        let body = r#"{
            "converter_type": "tfjs",
            "model_json_path": "/tfjs/model.json",
            "labels_path": "/tfjs/labels.txt",
            "labels_format": "lines"
        }"#;
        let req: ConvertRequest = serde_json::from_str(body).unwrap();
        let ConvertRequest::Tfjs(p) = &req else {
            panic!("expected ConvertRequest::Tfjs");
        };
        assert_eq!(p.labels_format, LabelsFormat::Lines);
    }

    #[test]
    fn convert_request_rejects_traversal_in_paths() {
        for field in ["model_json_path", "labels_path"] {
            let mut v = serde_json::json!({
                "converter_type": "tfjs",
                "model_json_path": "/m",
                "labels_path": "/l",
                "labels_format": "lines",
            });
            v[field] = serde_json::Value::String("/..".into());
            let body = serde_json::to_string(&v).unwrap();
            let res: Result<ConvertRequest, _> = serde_json::from_str(&body);
            assert!(res.is_err(), "{field}=/.. must be rejected");
        }
    }

    #[test]
    fn labels_format_serializes_snake_case() {
        let lines = serde_json::to_string(&LabelsFormat::Lines).unwrap();
        assert_eq!(lines, "\"lines\"");
        let tfjs = serde_json::to_string(&LabelsFormat::TfjsMetadata).unwrap();
        assert_eq!(tfjs, "\"tfjs_metadata\"");
        let parsed: LabelsFormat = serde_json::from_str("\"lines\"").unwrap();
        assert_eq!(parsed, LabelsFormat::Lines);
        let parsed: LabelsFormat = serde_json::from_str("\"tfjs_metadata\"").unwrap();
        assert_eq!(parsed, LabelsFormat::TfjsMetadata);
        let res: Result<LabelsFormat, _> = serde_json::from_str("\"Lines\"");
        assert!(res.is_err());
    }

    #[test]
    fn training_cfg_canonical_sha256_is_deterministic() {
        let cfg = good_cfg();
        let h1 = canonical_training_cfg_sha256(&cfg).expect("good cfg validates");
        let h2 = canonical_training_cfg_sha256(&cfg).expect("good cfg validates");
        assert_eq!(h1, h2);
        assert_eq!(h1.len(), 64);
        assert!(
            h1.bytes()
                .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
        );

        let reordered = r#"{"learning_rate":0.001,"seed":42,"epochs":4,"batch_size":16}"#;
        let parsed: TrainingCfg = serde_json::from_str(reordered).unwrap();
        let h_reordered = canonical_training_cfg_sha256(&parsed).expect("good cfg validates");
        assert_eq!(h1, h_reordered);
    }

    #[test]
    fn training_cfg_canonical_sha256_rejects_non_finite_learning_rate() {
        let mut cfg = good_cfg();
        cfg.learning_rate = f32::NAN;
        assert!(canonical_training_cfg_sha256(&cfg).is_err());
        cfg.learning_rate = f32::INFINITY;
        assert!(canonical_training_cfg_sha256(&cfg).is_err());
    }

    #[test]
    fn training_cfg_canonical_sha256_changes_with_value() {
        let cfg_a = good_cfg();
        let cfg_b = TrainingCfg {
            epochs: cfg_a.epochs + 1,
            ..cfg_a.clone()
        };
        assert_ne!(
            canonical_training_cfg_sha256(&cfg_a).expect("good cfg validates"),
            canonical_training_cfg_sha256(&cfg_b).expect("good cfg validates")
        );
    }

    #[test]
    fn training_cfg_round_trips_through_manifest_value() {
        let cfg = good_cfg();
        let v = to_manifest_value(&cfg);
        let back = from_manifest_value(&v).unwrap();
        assert_eq!(cfg, back);
        assert_eq!(
            canonical_training_cfg_sha256(&cfg).expect("good cfg validates"),
            canonical_training_cfg_sha256(&back).expect("good cfg validates")
        );
    }

    #[test]
    fn from_manifest_value_classifies_corruption_as_internal() {
        let bad = serde_json::json!({
            "epochs": "four",
            "batch_size": 16,
            "learning_rate": 1e-3,
        });
        let err = from_manifest_value(&bad).unwrap_err();
        assert!(matches!(err, FileError::MetadataParse { .. }));
        assert_eq!(err.kind(), ErrorKind::Internal);
    }

    #[test]
    fn from_manifest_value_rejects_unknown_fields() {
        let bad = serde_json::json!({
            "epochs": 1,
            "batch_size": 1,
            "learning_rate": 1e-3,
            "stray": true,
        });
        assert!(from_manifest_value(&bad).is_err());
    }

    #[test]
    fn validation_error_classifies_user_input() {
        let err = ValidationError::EpochsOutOfRange {
            got: 0,
            min: 1,
            max: 1_000,
        };
        assert_eq!(err.kind(), ErrorKind::UserInput);
        let err = ValidationError::AlpkgManifestExtension {
            got: "alpkg/foo.bin".to_string(),
        };
        assert_eq!(err.kind(), ErrorKind::UserInput);
    }

    #[test]
    fn good_convert_request_round_trips_through_json() {
        let req = good_convert_request();
        let s = serde_json::to_string(&req).unwrap();
        let back: ConvertRequest = serde_json::from_str(&s).unwrap();
        assert_eq!(req, back);
    }
}
