//! Integration tests for `file_mgr::recover_all`: seed an active generation,
//! mutate on-disk state to reproduce a crash mode, run the sweep, then assert
//! the active-result variant, the post-recovery on-disk shape, and the report
//! counters. The synthetic loader returns an empty `()` candidate so recovery
//! runs without the inference crate's runtime preload; the daemon's real boot
//! path uses `HotHead::load` against an identical on-disk shape.

#![allow(clippy::disallowed_methods)]

use std::path::{Path, PathBuf};
use std::sync::Arc;

use acousticslab::common::ids::{HeadId, JobId, WorkspaceId};
use acousticslab::common::workspace::{
    HeadIndex, HeadManifest, HeadRecord, WorkspaceCore, WorkspaceRevision,
};
use acousticslab::file_mgr::active_head_writer::{
    ActivationOriginInput, DefaultHeadSource, HeadInnerLoader, PendingActivation,
    publish_active_generation, stage_and_validate_activation, staging_path_for,
};
use acousticslab::file_mgr::schema::{
    ACTIVE_HEAD_FILENAME, active_current_path, active_generation_dir, head_artifact_path,
    head_manifest_path, heads_dir, read_active_current, read_workspace_core, workspace_dir_for,
    workspaces_dir, write_head_index, write_head_manifest, write_workspace_core,
};
use acousticslab::file_mgr::staging::{DeleteTombstone, stage_payload, write_tombstone};
use acousticslab::file_mgr::time_util::now_rfc3339;
use acousticslab::file_mgr::{RecoveryActiveResult, WorkspaceCacheCell, recover_all};
use sha2::{Digest, Sha256};

fn synth_loader() -> Box<HeadInnerLoader> {
    Box::new(|_mpk: &Path, _labels: &Path, _id: HeadId| {
        Ok(Box::new(()) as Box<dyn std::any::Any + Send>)
    })
}

fn ws_id(byte: u8) -> WorkspaceId {
    let s = format!("11111111-2222-4333-8444-5555555555{byte:02x}");
    WorkspaceId::parse(&s).unwrap()
}

fn head_id(byte: u8) -> HeadId {
    let s = format!("11111111-2222-4333-8444-5555555555{byte:02x}");
    HeadId::parse(&s).unwrap()
}

fn rev(id: u64) -> WorkspaceRevision {
    WorkspaceRevision {
        id,
        at: "2026-05-07T12:00:00Z".to_string(),
    }
}

fn hex_sha256(bytes: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(bytes);
    let d = h.finalize();
    static HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = vec![0u8; d.len() * 2];
    for (i, &b) in d.iter().enumerate() {
        out[2 * i] = HEX[(b >> 4) as usize];
        out[2 * i + 1] = HEX[(b & 0x0f) as usize];
    }
    String::from_utf8(out).unwrap()
}

/// Stage a bundled-default fixture under `<dir>/bundled_default/`. The mpk is a
/// synthetic blob: the synthetic loader doesn't parse it, and the activation
/// pipeline only needs the bytes plus the labels list.
fn fresh_bundled_default(dir: &Path, mpk: &[u8], labels_text: &str) -> (PathBuf, PathBuf) {
    let bundled = dir.join("bundled_default");
    std::fs::create_dir_all(&bundled).unwrap();
    let head = bundled.join("head.mpk");
    let labels = bundled.join("labels.txt");
    std::fs::write(&head, mpk).unwrap();
    std::fs::write(&labels, labels_text).unwrap();
    (head, labels)
}

fn default_source<'a>(path: &'a Path, labels_path: &'a Path) -> DefaultHeadSource<'a> {
    DefaultHeadSource { path, labels_path }
}

fn default_origin<'a>(path: &'a Path, labels_path: &'a Path) -> ActivationOriginInput<'a> {
    ActivationOriginInput::Default {
        source: default_source(path, labels_path),
    }
}

/// Activate a fresh bundled-default generation so the test starts with a
/// known-good current pointer.
fn seed_active_generation(
    root: &Path,
    mpk: &[u8],
    labels_text: &str,
) -> (PathBuf, PathBuf, String) {
    std::fs::create_dir_all(root).unwrap();
    let (head, labels) = fresh_bundled_default(root, mpk, labels_text);
    let pending = PendingActivation {
        root,
        origin_input: default_origin(&head, &labels),
        now_rfc3339: now_rfc3339(),
    };
    let result = stage_and_validate_activation(pending, &*synth_loader()).unwrap();
    let staging = staging_path_for(root, &result.activation_id);
    publish_active_generation(root, &staging, &result.manifest, &result.activation_id).unwrap();
    (head, labels, result.activation_id)
}

/// Seed a workspace dir with one trained head, mirroring the daemon's
/// `WorkspaceMgr::create` + `publish_trained_head` outcome.
fn fresh_workspace_with_head(root: &Path, ws: WorkspaceId, head: HeadId) -> PathBuf {
    let ws_dir = workspace_dir_for(root, &ws);
    std::fs::create_dir_all(heads_dir(&ws_dir)).unwrap();
    std::fs::create_dir_all(ws_dir.join(".tmp")).unwrap();
    let mpk = b"MPK-CONTENT";
    let manifest = HeadManifest {
        head_id: head,
        workspace_id: ws,
        workspace_revision: rev(5),
        sha256: hex_sha256(mpk),
        n_classes: 2,
        size_bytes: mpk.len() as u64,
        created_at: "2026-05-07T12:34:56Z".to_string(),
        labels: vec!["alpha".to_string(), "beta".to_string()],
    };
    let mut idx = HeadIndex::default();
    idx.heads.push(HeadRecord {
        head_id: head,
        workspace_revision: manifest.workspace_revision.clone(),
        sha256: manifest.sha256.clone(),
        n_classes: manifest.n_classes,
        size_bytes: manifest.size_bytes,
        created_at: manifest.created_at.clone(),
    });
    write_head_index(&ws_dir, &idx).unwrap();
    write_head_manifest(&ws_dir, &manifest).unwrap();
    std::fs::write(head_artifact_path(&ws_dir, head), mpk).unwrap();
    write_workspace_core(
        &ws_dir,
        &WorkspaceCore {
            id: ws,
            name: "main".to_string(),
            tags: Vec::new(),
            created_at: "2026-05-07T12:34:56Z".to_string(),
            workspace_revision: rev(5),
            head_count: 1,
        },
    )
    .unwrap();
    ws_dir
}

fn fresh_caches() -> Arc<dashmap::DashMap<WorkspaceId, Arc<WorkspaceCacheCell>>> {
    Arc::new(dashmap::DashMap::new())
}

#[test]
fn corrupt_active_head_falls_back_to_previous_generation() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    // Two generations so recovery has a previous to promote; the second publish's
    // write_active_current makes it the current pointer (last-writer-wins). The
    // sleep only gives the two generations distinct activated_at timestamps so
    // recovery's newest-first scan promotes gen1 deterministically.
    let (head1, labels1, gen1) = seed_active_generation(root, b"DEFAULT-MPK-A", "cat\ndog\n");
    std::thread::sleep(std::time::Duration::from_millis(50));
    let pending2 = PendingActivation {
        root,
        origin_input: default_origin(&head1, &labels1),
        now_rfc3339: now_rfc3339(),
    };
    let r2 = stage_and_validate_activation(pending2, &*synth_loader()).unwrap();
    publish_active_generation(
        root,
        &staging_path_for(root, &r2.activation_id),
        &r2.manifest,
        &r2.activation_id,
    )
    .unwrap();
    let current_id = r2.activation_id.clone();
    // Corrupt current's head.mpk so the streaming-hash verify fails, forcing a
    // promotion of the previous generation (gen1).
    let head_path = active_generation_dir(root, &current_id).join(ACTIVE_HEAD_FILENAME);
    std::fs::write(&head_path, b"TAMPERED").unwrap();
    let caches = fresh_caches();
    let report = recover_all(
        root,
        Some(default_source(&head1, &labels1)),
        &caches,
        &*synth_loader(),
    )
    .unwrap();
    match &report.active {
        RecoveryActiveResult::PromotedPrevious { activation_id, .. } => {
            assert_eq!(*activation_id, gen1);
        }
        other => panic!("expected PromotedPrevious, got {other:?}"),
    }
    let pointer = read_active_current(root).unwrap();
    assert_eq!(pointer.activation_id, gen1);
}

#[test]
fn workspace_delete_tombstone_resumes_at_boot() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    // A healthy active generation so recovery verifies it alongside the sweep.
    let (default_head, default_labels, _) = seed_active_generation(root, b"DEFAULT-MPK", "cat\n");
    // Half-completed workspace-delete (tombstone + victim tree renamed under root
    // `.tmp/`); recovery must drain the payload, finalize, and evict the cell.
    let ws = ws_id(0xAA);
    let staging = root.join(".tmp");
    std::fs::create_dir_all(&staging).unwrap();
    let tombstone = DeleteTombstone::Workspace {
        job_id: JobId::new(),
        workspace_id: ws,
        created_at: now_rfc3339(),
    };
    let staged = write_tombstone(&staging, &tombstone).unwrap();
    let victim_dir = root.join("victim");
    std::fs::create_dir_all(victim_dir.join("heads")).unwrap();
    std::fs::write(victim_dir.join("workspace.json"), b"{}").unwrap();
    std::fs::write(victim_dir.join("heads/inner"), b"x").unwrap();
    stage_payload(&victim_dir, &staged).unwrap();
    // Stale cache cell for the victim; recovery's eviction hook must drop it.
    let caches = fresh_caches();
    caches.insert(
        ws,
        Arc::new(WorkspaceCacheCell::new(
            WorkspaceCore {
                id: ws,
                name: "victim".to_string(),
                tags: Vec::new(),
                created_at: "2026-05-07T12:34:56Z".to_string(),
                workspace_revision: rev(0),
                head_count: 0,
            },
            HeadIndex::default(),
        )),
    );
    let report = recover_all(
        root,
        Some(default_source(&default_head, &default_labels)),
        &caches,
        &*synth_loader(),
    )
    .unwrap();
    assert_eq!(report.root_staging.workspace_tombstones_completed, 1);
    assert!(!staged.tombstone.exists());
    assert!(!staged.stage_dir.exists());
    assert!(caches.get(&ws).is_none(), "cache cell evicted post-resume");
}

#[test]
fn daemon_owned_head_orphans_swept_on_boot() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    let (head, labels, _) = seed_active_generation(root, b"DEFAULT-MPK", "cat\n");
    std::fs::create_dir_all(workspaces_dir(root)).unwrap();
    let ws = ws_id(0xBB);
    let real_head = head_id(0xCC);
    let ws_dir = fresh_workspace_with_head(root, ws, real_head);
    // Two unreferenced files in heads/: the sweep removes both because the
    // head_id is absent from `heads.json.heads[]`.
    let orphan = head_id(0xDD);
    let orphan_mpk = head_artifact_path(&ws_dir, orphan);
    let orphan_json = head_manifest_path(&ws_dir, orphan);
    std::fs::write(&orphan_mpk, b"ORPHAN-MPK").unwrap();
    std::fs::write(&orphan_json, b"{}").unwrap();
    let caches = fresh_caches();
    let report = recover_all(
        root,
        Some(default_source(&head, &labels)),
        &caches,
        &*synth_loader(),
    )
    .unwrap();
    assert_eq!(report.workspaces.workspaces_scanned, 1);
    assert_eq!(report.workspaces.head_orphans_swept, 2);
    assert!(!orphan_mpk.exists());
    assert!(!orphan_json.exists());
    // The in-index head must survive the orphan sweep (precision guard).
    assert!(head_artifact_path(&ws_dir, real_head).is_file());
    assert!(head_manifest_path(&ws_dir, real_head).is_file());
}

#[test]
fn head_count_drift_repaired_on_boot() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    let (head, labels, _) = seed_active_generation(root, b"DEFAULT-MPK", "cat\n");
    std::fs::create_dir_all(workspaces_dir(root)).unwrap();
    let ws = ws_id(0xEE);
    let real_head = head_id(0xFA);
    let ws_dir = fresh_workspace_with_head(root, ws, real_head);
    // Second head record (heads.json has 2), then tamper workspace.json to claim
    // head_count=0 -- the drift recovery must repair.
    let head2 = head_id(0xFB);
    let mpk2 = b"MPK-2";
    let manifest2 = HeadManifest {
        head_id: head2,
        workspace_id: ws,
        workspace_revision: rev(5),
        sha256: hex_sha256(mpk2),
        n_classes: 2,
        size_bytes: mpk2.len() as u64,
        created_at: "2026-05-07T12:34:56Z".to_string(),
        labels: vec!["alpha".to_string(), "beta".to_string()],
    };
    write_head_manifest(&ws_dir, &manifest2).unwrap();
    std::fs::write(head_artifact_path(&ws_dir, head2), mpk2).unwrap();
    let mut idx = HeadIndex::default();
    for hid in [real_head, head2] {
        idx.heads.push(HeadRecord {
            head_id: hid,
            workspace_revision: rev(5),
            sha256: hex_sha256(if hid == real_head {
                b"MPK-CONTENT"
            } else {
                mpk2
            }),
            n_classes: 2,
            size_bytes: 11,
            created_at: "2026-05-07T12:34:56Z".to_string(),
        });
    }
    write_head_index(&ws_dir, &idx).unwrap();
    let mut core = read_workspace_core(&ws_dir).unwrap();
    core.head_count = 0;
    write_workspace_core(&ws_dir, &core).unwrap();
    let caches = fresh_caches();
    let report = recover_all(
        root,
        Some(default_source(&head, &labels)),
        &caches,
        &*synth_loader(),
    )
    .unwrap();
    assert_eq!(report.workspaces.head_count_repaired, 1);
    let core = read_workspace_core(&ws_dir).unwrap();
    assert_eq!(core.head_count, 2);
    // Active stays Current: only workspace state was mutated.
    assert!(matches!(
        report.active,
        RecoveryActiveResult::Current { .. }
    ));
    assert!(active_current_path(root).is_file());
}

/// A manifest carrying legacy `source_dataset_revision` (instead of
/// `workspace_revision`) parse-fails at `read_active_manifest`; recovery treats
/// the current generation as corrupt and falls back to the bundled default.
#[test]
fn legacy_active_manifest_falls_back_to_bundled_default() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    let (head, labels) = fresh_bundled_default(root, b"DEFAULT-MPK", "alpha\n");
    let activation_id = "11111111-2222-4333-8444-555555555560".to_string();
    let gen_dir = active_generation_dir(root, &activation_id);
    std::fs::create_dir_all(&gen_dir).unwrap();
    // mpk + labels are arbitrary: verify fails at the manifest-parse step before
    // reaching the streaming-hash gates.
    std::fs::write(gen_dir.join(ACTIVE_HEAD_FILENAME), b"any").unwrap();
    std::fs::write(
        gen_dir.join(acousticslab::file_mgr::schema::ACTIVE_LABELS_FILENAME),
        "alpha\n",
    )
    .unwrap();
    let round_1_manifest = serde_json::json!({
        "origin": "head",
        "source_workspace_id": "11111111-2222-4333-8444-555555555548",
        "source_head_id": "11111111-2222-4333-8444-555555555540",
        // Legacy alias: schema now requires `workspace_revision`, so this shape parse-fails.
        "source_dataset_revision": { "id": 5, "at": "2026-05-07T13:00:00Z" },
        "runtime_head_id": "11111111-2222-4333-8444-555555555540",
        "sha256": "deadbeef",
        "labels_sha256": "cafef00d",
        "n_classes": 1,
        "labels": ["alpha"],
        "activated_at": "2026-05-07T12:34:56Z",
    });
    std::fs::write(
        gen_dir.join(acousticslab::file_mgr::schema::ACTIVE_MANIFEST_FILENAME),
        serde_json::to_vec(&round_1_manifest).unwrap(),
    )
    .unwrap();
    acousticslab::file_mgr::schema::write_active_current(
        root,
        &acousticslab::file_mgr::schema::ActiveCurrentPointer {
            activation_id: activation_id.clone(),
        },
    )
    .unwrap();

    let caches = fresh_caches();
    let report = recover_all(
        root,
        Some(default_source(&head, &labels)),
        &caches,
        &*synth_loader(),
    )
    .unwrap();
    match &report.active {
        RecoveryActiveResult::DefaultedFromBundle {
            activation_id: new_id,
            ..
        } => {
            assert_ne!(*new_id, activation_id, "fresh bundled default published");
        }
        other => panic!("expected DefaultedFromBundle on legacy manifest, got {other:?}"),
    }
    let pointer = read_active_current(root).unwrap();
    assert_ne!(pointer.activation_id, activation_id);
}

/// `recover_all` walks every sub-sweep without short-circuiting on partial
/// failure: workspace tombstone + head orphans + head_count drift + incomplete
/// create all clear in one pass.
#[test]
fn recover_all_aggregates_multi_failure_residue() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    let (head, labels, _) = seed_active_generation(root, b"DEFAULT-MPK", "cat\n");
    std::fs::create_dir_all(workspaces_dir(root)).unwrap();

    // (a) Healthy workspace + head_count drift + head orphan.
    let ws_a = ws_id(0xA0);
    let head_a = head_id(0xA1);
    let ws_dir_a = fresh_workspace_with_head(root, ws_a, head_a);
    let mut core_a = read_workspace_core(&ws_dir_a).unwrap();
    core_a.head_count = 0; // claim 0 even though heads.json has 1
    write_workspace_core(&ws_dir_a, &core_a).unwrap();
    let orphan_a = head_id(0xA2);
    std::fs::write(head_artifact_path(&ws_dir_a, orphan_a), b"orphan-mpk").unwrap();
    std::fs::write(head_manifest_path(&ws_dir_a, orphan_a), b"{}").unwrap();

    // (b) Incomplete-create workspace dir (no workspace.json).
    let ws_b = ws_id(0xB0);
    let ws_dir_b = workspace_dir_for(root, &ws_b);
    std::fs::create_dir_all(ws_dir_b.join("heads")).unwrap();
    std::fs::create_dir_all(ws_dir_b.join(".tmp")).unwrap();

    // (c) Root-level workspace tombstone awaiting drain.
    let ws_c = ws_id(0xC0);
    let staging = root.join(".tmp");
    std::fs::create_dir_all(&staging).unwrap();
    let tombstone = DeleteTombstone::Workspace {
        job_id: JobId::new(),
        workspace_id: ws_c,
        created_at: now_rfc3339(),
    };
    let staged = write_tombstone(&staging, &tombstone).unwrap();
    let victim = root.join("victim_c");
    std::fs::create_dir_all(victim.join("heads")).unwrap();
    std::fs::write(victim.join("workspace.json"), b"{}").unwrap();
    stage_payload(&victim, &staged).unwrap();

    // Stale cache cell for ws_c to exercise root-staging recovery's eviction hook.
    let caches = fresh_caches();
    caches.insert(
        ws_c,
        Arc::new(WorkspaceCacheCell::new(
            WorkspaceCore {
                id: ws_c,
                name: "victim".to_string(),
                tags: Vec::new(),
                created_at: "2026-05-07T12:34:56Z".to_string(),
                workspace_revision: rev(0),
                head_count: 0,
            },
            HeadIndex::default(),
        )),
    );

    let report = recover_all(
        root,
        Some(default_source(&head, &labels)),
        &caches,
        &*synth_loader(),
    )
    .unwrap();

    assert_eq!(
        report.workspaces.workspaces_scanned, 1,
        "exactly one valid workspace (ws_a); ws_b is incomplete-create",
    );
    assert_eq!(report.workspaces.head_count_repaired, 1, "ws_a repaired");
    assert_eq!(
        report.workspaces.head_orphans_swept, 2,
        "two orphan files removed (mpk + json)",
    );
    assert_eq!(
        report.workspaces.incomplete_creates_removed, 1,
        "ws_b incomplete-create removed",
    );
    assert!(!ws_dir_b.exists(), "ws_b directory removed");
    let core_a = read_workspace_core(&ws_dir_a).unwrap();
    assert_eq!(core_a.head_count, 1);
    assert!(!head_artifact_path(&ws_dir_a, orphan_a).exists());
    assert!(!head_manifest_path(&ws_dir_a, orphan_a).exists());

    assert_eq!(report.root_staging.workspace_tombstones_completed, 1);
    assert!(!staged.tombstone.exists());
    assert!(caches.get(&ws_c).is_none(), "ws_c cache cell evicted");

    // Active stays Current: only workspace + root state had residue.
    assert!(matches!(
        report.active,
        RecoveryActiveResult::Current { .. }
    ));
}

/// A corrupt `heads.json` skips the head-orphan/count-repair half (logged) but
/// the tombstone drain still runs, so an operator-committed delete is never
/// stranded by one bad byte. The workspace counts as `workspaces_scanned` and
/// does NOT bump `workspace_recovery_failures`. A non-NotFound `heads.json` read
/// failure is logged and made non-fatal (heads set to None) rather than
/// early-returning, so the tombstone drain that follows the head-half still runs.
/// Load-bearing: ws_b stages a
/// REAL dataset-delete tombstone + payload and the test asserts the drain
/// COMPLETED it (counter == 1, staged paths gone); asserting only the scalar
/// scanned/failures counters would pass even if the drain were re-buried.
#[test]
fn per_workspace_heads_corruption_skips_head_half_but_drains_tombstones() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    let (head, labels, _) = seed_active_generation(root, b"DEFAULT-MPK", "cat\n");
    std::fs::create_dir_all(workspaces_dir(root)).unwrap();

    let ws_a = ws_id(0xE0);
    let head_a = head_id(0xE1);
    let _ws_dir_a = fresh_workspace_with_head(root, ws_a, head_a);

    // ws_b: workspace.json present, heads.json corrupt -- head-half skipped,
    // tombstone drain still runs.
    let ws_b = ws_id(0xE2);
    let ws_dir_b = workspace_dir_for(root, &ws_b);
    std::fs::create_dir_all(ws_dir_b.join("heads")).unwrap();
    std::fs::create_dir_all(ws_dir_b.join(".tmp")).unwrap();
    write_workspace_core(
        &ws_dir_b,
        &WorkspaceCore {
            id: ws_b,
            name: "broken".to_string(),
            tags: Vec::new(),
            created_at: "2026-05-08T12:00:00Z".to_string(),
            workspace_revision: rev(0),
            head_count: 0,
        },
    )
    .unwrap();
    // Malformed JSON: fails parse outright.
    std::fs::write(ws_dir_b.join("heads.json"), b"{not json}").unwrap();

    // Real dataset-delete tombstone + payload in ws_b's `.tmp/`, to prove the
    // drain ran despite corrupt heads.json (not just that ws_b was scanned).
    use acousticslab::common::asset_path::AssetPath;
    let ws_b_staging = ws_dir_b.join(".tmp");
    let ws_b_tombstone = DeleteTombstone::Dataset {
        job_id: JobId::new(),
        workspace_id: ws_b,
        path: Some(AssetPath::parse("audio/dog").unwrap()),
        workspace_revision_id: 3,
        created_at: now_rfc3339(),
    };
    let ws_b_staged = write_tombstone(&ws_b_staging, &ws_b_tombstone).unwrap();
    let ws_b_payload = root.join("ws_b_dataset_payload");
    std::fs::write(&ws_b_payload, b"strand-me-not").unwrap();
    stage_payload(&ws_b_payload, &ws_b_staged).unwrap();

    let caches = fresh_caches();
    let report = recover_all(
        root,
        Some(default_source(&head, &labels)),
        &caches,
        &*synth_loader(),
    )
    .unwrap();

    assert_eq!(
        report.workspaces.workspace_recovery_failures, 0,
        "head-half failure no longer counts as a workspace recovery failure",
    );
    assert_eq!(
        report.workspaces.workspaces_scanned, 2,
        "both workspaces scanned; ws_b's head-half was skipped but the workspace was still processed",
    );
    // The drain ran despite the head-half skip: ws_b's committed delete was
    // finalized, not stranded. This assertion fails if the drain is ever
    // re-buried under the head-half short-circuit.
    assert_eq!(
        report.workspaces.dataset_tombstones_completed, 1,
        "tombstone drain must complete even when the head-half (corrupt heads.json) fails",
    );
    assert!(
        !ws_b_staged.tombstone.exists(),
        "the drained tombstone marker must be removed",
    );
    assert!(
        !ws_b_staged.stage_dir.exists(),
        "the drained payload stage dir must be removed",
    );
    assert!(matches!(
        report.active,
        RecoveryActiveResult::Current { .. }
    ));
    // ws_b stays on disk: the orchestrator never auto-deletes a broken
    // workspace; operator action is required.
    assert!(ws_dir_b.exists());
}

/// Converter- and dataset-tombstone resume both fire in a single sweep reachable
/// from `recover_all`; the variant is dispatched by tombstone filename prefix.
#[test]
fn recover_all_drains_dataset_and_converter_tombstones_together() {
    use acousticslab::common::asset_path::AssetPath;
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    let (default_head, default_labels, _) = seed_active_generation(root, b"DEFAULT-MPK", "cat\n");
    std::fs::create_dir_all(workspaces_dir(root)).unwrap();

    let ws = ws_id(0xD0);
    let head = head_id(0xD1);
    let ws_dir = fresh_workspace_with_head(root, ws, head);

    // Both tombstones in one workspace's .tmp/; prefix dispatches the variant:
    // `delete-assets-<job_id>/` -> Dataset, `delete-converters-<job_id>/` -> Converter.
    let staging = ws_dir.join(".tmp");
    let dataset_tombstone = DeleteTombstone::Dataset {
        job_id: JobId::new(),
        workspace_id: ws,
        path: Some(AssetPath::parse("audio/cat").unwrap()),
        workspace_revision_id: 7,
        created_at: now_rfc3339(),
    };
    let dataset_staged = write_tombstone(&staging, &dataset_tombstone).unwrap();
    let dataset_target = root.join("dataset_payload");
    std::fs::write(&dataset_target, b"data-bytes").unwrap();
    stage_payload(&dataset_target, &dataset_staged).unwrap();

    let converter_tombstone = DeleteTombstone::Converter {
        job_id: JobId::new(),
        workspace_id: ws,
        path: Some(AssetPath::parse("tfjs/model.json").unwrap()),
        workspace_revision_id: 8,
        created_at: now_rfc3339(),
    };
    let converter_staged = write_tombstone(&staging, &converter_tombstone).unwrap();
    let converter_target = root.join("converter_payload");
    std::fs::write(&converter_target, b"manifest-bytes").unwrap();
    stage_payload(&converter_target, &converter_staged).unwrap();

    let caches = fresh_caches();
    let report = recover_all(
        root,
        Some(default_source(&default_head, &default_labels)),
        &caches,
        &*synth_loader(),
    )
    .unwrap();

    assert_eq!(report.workspaces.dataset_tombstones_completed, 1);
    assert_eq!(report.workspaces.converter_tombstones_completed, 1);
    assert!(!dataset_staged.tombstone.exists());
    assert!(!dataset_staged.stage_dir.exists());
    assert!(!converter_staged.tombstone.exists());
    assert!(!converter_staged.stage_dir.exists());
}
