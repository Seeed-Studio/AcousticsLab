//! `file_mgr` unit + integration tests.

#![allow(clippy::disallowed_methods)]

use super::*;

fn fresh_root() -> (tempfile::TempDir, WorkspaceMgr) {
    let dir = tempfile::tempdir().expect("tempdir");
    let mgr = WorkspaceMgr::new(dir.path().to_path_buf());
    (dir, mgr)
}

fn fresh_fs_service() -> (tempfile::TempDir, FsServiceImpl) {
    let dir = tempfile::tempdir().expect("tempdir");
    let fs = FsServiceImpl::new(dir.path().to_path_buf());
    (dir, fs)
}

/// Rehydrates the legacy `weights/`/`labels/`/`metadata.json` shape (no longer
/// written on `create`) so legacy `AssetKind` paths keep round-trip coverage.
fn seed_legacy_layout(mgr: &WorkspaceMgr, id: &WorkspaceId) {
    let ws = mgr.workspace_dir(id);
    for sub in ["weights", "labels"] {
        std::fs::create_dir_all(ws.join(sub)).expect("seed legacy subdir");
    }
    let core = mgr.read_cached_core(id).expect("core");
    let metadata = WorkspaceMetadata::new(core.id, core.name.clone());
    mgr.write_metadata(id, &metadata)
        .expect("seed metadata.json");
}

fn create_legacy(mgr: &WorkspaceMgr, name: &str) -> WorkspaceId {
    let id = mgr.create(name).expect("create");
    seed_legacy_layout(mgr, &id);
    id
}

/// FsService variant of [`create_legacy`]; re-derives a `WorkspaceMgr` view on
/// the same dir since `FsServiceImpl::mgr` is private.
fn create_legacy_fs(fs: &FsServiceImpl, name: &str) -> WorkspaceId {
    let id = fs.create(name).expect("create");
    let mgr = WorkspaceMgr::new(fs.root().to_path_buf());
    seed_legacy_layout(&mgr, &id);
    id
}

#[test]
fn metadata_guard_commit_persists_changes() {
    let (_dir, fs) = fresh_fs_service();
    let id = create_legacy_fs(&fs, "ws");

    let mut g = fs.metadata_mut(&id).expect("guard");
    g.metadata_mut().assets.push(AssetRecord {
        kind: AssetKind::HeadMpk,
        name: AssetId::parse("added.mpk").expect("valid AssetId"),
        sha256: "0".repeat(64),
        size_bytes: 0,
    });
    g.commit().expect("commit");

    let after = fs.read_metadata(&id).expect("read");
    assert_eq!(after.assets.len(), 1);
    assert_eq!(after.assets[0].name, "added.mpk");
}

#[test]
fn metadata_guard_drop_without_commit_rolls_back() {
    let (_dir, fs) = fresh_fs_service();
    let id = create_legacy_fs(&fs, "ws");

    let before = fs.read_metadata(&id).expect("read");
    assert!(before.assets.is_empty());

    {
        let mut g = fs.metadata_mut(&id).expect("guard");
        g.metadata_mut().assets.push(AssetRecord {
            kind: AssetKind::HeadMpk,
            name: AssetId::parse("uncommitted.mpk").expect("valid AssetId"),
            sha256: "0".repeat(64),
            size_bytes: 0,
        });
    }

    let after = fs.read_metadata(&id).expect("read");
    assert!(
        after.assets.is_empty(),
        "uncommitted mutation persisted: {:?}",
        after.assets
    );
}

#[test]
fn install_from_path_renames_and_commits() {
    let (dir, fs) = fresh_fs_service();
    let id = create_legacy_fs(&fs, "ws");

    // Production uploader self-mkdirs `.tmp/` on first write; staging directly
    // here, so materialize the dir to mirror that contract.
    let tmp_dir = fs.workspace_tmpdir(&id);
    std::fs::create_dir_all(&tmp_dir).expect("mkdir workspace .tmp");
    let mut tmp = tempfile::NamedTempFile::new_in(&tmp_dir).expect("tempfile");
    use std::io::Write;
    tmp.write_all(b"hello").expect("write");
    tmp.as_file().sync_all().expect("sync");

    let receipt = fs
        .install_from_path(&id, AssetKind::HeadLabels, "greet.txt", tmp.path())
        .expect("install");

    assert_eq!(receipt.size_bytes, 5);
    let on_disk = std::fs::read(&receipt.path).expect("read installed");
    assert_eq!(on_disk, b"hello");
    let meta = fs.read_metadata(&id).expect("read meta");
    assert_eq!(meta.assets.len(), 1);
    assert_eq!(meta.assets[0].kind, AssetKind::HeadLabels);
    assert_eq!(meta.assets[0].name, "greet.txt");
    assert_eq!(meta.assets[0].size_bytes, 5);
    assert!(receipt.path.starts_with(dir.path()));
}

#[test]
fn create_workspace_writes_redesign_layout() {
    // `create` writes only `workspace.json` + empty `heads.json`; every leaf
    // subdir is created lazily by its first writer. Pins the empty-on-create
    // shape so an eager-mkdir regression surfaces here.
    let (_dir, mgr) = fresh_root();
    let id = mgr.create("first").expect("create");
    let ws = mgr.workspace_dir(&id);
    assert!(ws.is_dir(), "workspace dir itself must exist");
    assert!(ws.join("workspace.json").is_file());
    assert!(ws.join("heads.json").is_file());
    for sub in [
        "datasets",
        "converters",
        "training_logs",
        "converter_logs",
        ".tmp",
        "heads",
    ] {
        assert!(
            !ws.join(sub).exists(),
            "subdir {sub} must NOT exist before first writer; create_with_tags is lazy",
        );
    }
    assert!(
        !ws.join("metadata.json").exists(),
        "metadata.json should not be created",
    );
    assert!(
        !ws.join("weights").exists(),
        "weights/ should not be created",
    );
    assert!(!ws.join("labels").exists(), "labels/ should not be created",);
}

#[test]
fn create_writes_workspace_core_and_heads_index() {
    let (_dir, mgr) = fresh_root();
    let id = mgr.create("ws-core").expect("create");
    let ws = mgr.workspace_dir(&id);
    assert!(
        ws.join("workspace.json").is_file(),
        "workspace.json must be written"
    );
    assert!(
        ws.join("heads.json").is_file(),
        "heads.json must be written"
    );
    let summary = mgr.summary(&id).expect("summary");
    assert_eq!(summary.core.id, id);
    assert_eq!(summary.core.name, "ws-core");
    assert_eq!(summary.core.workspace_revision.id, 0);
    assert_eq!(summary.core.head_count, 0);
    assert!(summary.heads.heads.is_empty());
    assert!(summary.head_statuses.is_empty());
}

#[test]
fn create_rejects_duplicate_name() {
    let (_dir, mgr) = fresh_root();
    mgr.create("main").expect("first");
    let err = mgr.create("main").unwrap_err();
    assert!(matches!(err, FileError::NameConflict(_)));
}

#[test]
fn create_rejects_invalid_name() {
    let (_dir, mgr) = fresh_root();
    assert!(matches!(
        mgr.create("bad/name").unwrap_err(),
        FileError::InvalidName(_)
    ));
    assert!(matches!(
        mgr.create("").unwrap_err(),
        FileError::InvalidName(_)
    ));
}

#[test]
fn create_with_tags_persists_normalized_tags() {
    let (_dir, mgr) = fresh_root();
    // Surrounding ASCII whitespace trimmed; order preserved.
    let id = mgr
        .create_with_tags(
            "scoped",
            &["  field-recordings  ".to_string(), "pet-noises".to_string()],
        )
        .expect("create");
    let summary = mgr.summary(&id).expect("summary");
    assert_eq!(
        summary.core.tags,
        vec!["field-recordings".to_string(), "pet-noises".to_string()]
    );
}

#[test]
fn create_with_tags_rejects_empty_after_trim() {
    let (_dir, mgr) = fresh_root();
    let err = mgr
        .create_with_tags("scoped", &["   ".to_string()])
        .unwrap_err();
    assert!(matches!(err, FileError::InvalidName(_)));
}

#[test]
fn create_with_tags_rejects_path_separator() {
    let (_dir, mgr) = fresh_root();
    let err = mgr
        .create_with_tags("scoped", &["a/b".to_string()])
        .unwrap_err();
    assert!(matches!(err, FileError::InvalidName(_)));
}

#[test]
fn create_with_tags_rejects_case_insensitive_duplicates() {
    let (_dir, mgr) = fresh_root();
    let err = mgr
        .create_with_tags("scoped", &["Field".to_string(), "FIELD".to_string()])
        .unwrap_err();
    assert!(matches!(err, FileError::InvalidName(_)));
}

#[test]
fn create_with_tags_rejects_over_cap() {
    let (_dir, mgr) = fresh_root();
    let many: Vec<String> = (0..33).map(|i| format!("t{i}")).collect();
    let err = mgr.create_with_tags("scoped", &many).unwrap_err();
    assert!(matches!(err, FileError::InvalidName(_)));
}

#[test]
fn patch_workspace_renames_and_retags_atomically() {
    let (_dir, mgr) = fresh_root();
    let id = mgr.create_with_tags("orig", &["a".into()]).unwrap();
    let revision_before = mgr.summary(&id).unwrap().core.workspace_revision.id;
    let patched = mgr
        .patch_workspace(&id, Some("renamed"), Some(&["b".into(), "c".into()]))
        .expect("patch");
    assert_eq!(patched.name, "renamed");
    assert_eq!(patched.tags, vec!["b".to_string(), "c".to_string()]);
    let summary = mgr.summary(&id).unwrap();
    assert_eq!(summary.core.name, "renamed");
    assert_eq!(summary.core.tags, vec!["b".to_string(), "c".to_string()]);
    // Name + tag edits do NOT bump the workspace revision.
    assert_eq!(summary.core.workspace_revision.id, revision_before);
}

#[test]
fn patch_workspace_name_only_preserves_tags() {
    let (_dir, mgr) = fresh_root();
    let id = mgr.create_with_tags("orig", &["pinned".into()]).unwrap();
    let patched = mgr
        .patch_workspace(&id, Some("renamed"), None)
        .expect("patch");
    assert_eq!(patched.name, "renamed");
    assert_eq!(patched.tags, vec!["pinned".to_string()]);
}

#[test]
fn patch_workspace_tags_only_preserves_name() {
    let (_dir, mgr) = fresh_root();
    let id = mgr.create_with_tags("orig", &[]).unwrap();
    let patched = mgr
        .patch_workspace(&id, None, Some(&["new".into()]))
        .expect("patch");
    assert_eq!(patched.name, "orig");
    assert_eq!(patched.tags, vec!["new".to_string()]);
}

#[test]
fn patch_workspace_self_rename_is_idempotent() {
    let (_dir, mgr) = fresh_root();
    let id = mgr.create("solo").unwrap();
    // Self-rename succeeds: the uniqueness check excludes the workspace under edit.
    mgr.patch_workspace(&id, Some("solo"), None).expect("patch");
    mgr.patch_workspace(&id, Some("Solo"), None).expect("patch");
    assert_eq!(mgr.summary(&id).unwrap().core.name, "Solo");
}

#[test]
fn patch_workspace_rejects_name_collision_with_other() {
    let (_dir, mgr) = fresh_root();
    let _other = mgr.create("taken").unwrap();
    let id = mgr.create("free").unwrap();
    let err = mgr.patch_workspace(&id, Some("TAKEN"), None).unwrap_err();
    assert!(matches!(err, FileError::NameConflict(_)));
}

#[test]
fn patch_workspace_returns_not_found_for_missing_workspace() {
    let (_dir, mgr) = fresh_root();
    let phantom = WorkspaceId::new();
    let err = mgr
        .patch_workspace(&phantom, Some("ghost"), None)
        .unwrap_err();
    assert!(matches!(err, FileError::NotFound(_)));
}

#[test]
fn patch_workspace_rejects_invalid_name() {
    let (_dir, mgr) = fresh_root();
    let id = mgr.create("ok").unwrap();
    let err = mgr.patch_workspace(&id, Some(""), None).unwrap_err();
    assert!(matches!(err, FileError::InvalidName(_)));
}

#[test]
fn patch_workspace_rejects_invalid_tags() {
    let (_dir, mgr) = fresh_root();
    let id = mgr.create("ok").unwrap();
    let err = mgr
        .patch_workspace(&id, None, Some(&["a/b".into()]))
        .unwrap_err();
    assert!(matches!(err, FileError::InvalidName(_)));
}

#[test]
fn list_workspaces_returns_created() {
    let (_dir, mgr) = fresh_root();
    let a = mgr.create("a").unwrap();
    let b = mgr.create("b").unwrap();
    let mut ids = mgr.list_workspaces().unwrap();
    ids.sort_by_key(|i| i.to_string());
    let mut expected = vec![a.to_string(), b.to_string()];
    expected.sort();
    let got: Vec<String> = ids.into_iter().map(|i| i.to_string()).collect();
    assert_eq!(got, expected);
}

#[test]
fn delete_removes_workspace() {
    let (_dir, mgr) = fresh_root();
    let id = mgr.create("doomed").unwrap();
    let ws = mgr.workspace_dir(&id);
    assert!(ws.exists());
    mgr.delete(&id).unwrap();
    assert!(!ws.exists());
    assert!(matches!(
        mgr.delete(&id).unwrap_err(),
        FileError::NotFound(_)
    ));
}

#[tokio::test]
async fn upload_atomic_writes_with_sha256() {
    let (_dir, mgr) = fresh_root();
    let id = create_legacy(&mgr, "upload-test");
    let payload = b"hello world\nthis is a head.mpk-flavored file";
    let receipt = mgr
        .upload(&id, AssetKind::HeadMpk, "demo.mpk", &payload[..])
        .await
        .expect("upload");

    assert_eq!(receipt.size_bytes, payload.len() as u64);
    let expected = hex_lowercase(&Sha256::digest(payload));
    assert_eq!(receipt.sha256, expected);
    assert!(receipt.path.exists());
    let on_disk = std::fs::read(&receipt.path).unwrap();
    assert_eq!(on_disk, payload);

    let meta = mgr.read_metadata(&id).unwrap();
    assert_eq!(meta.assets.len(), 1);
    assert_eq!(meta.assets[0].sha256, expected);
    assert_eq!(meta.assets[0].kind, AssetKind::HeadMpk);
}

#[tokio::test]
async fn upload_rejects_bad_extension() {
    let (_dir, mgr) = fresh_root();
    let id = create_legacy(&mgr, "x");
    let err = mgr
        .upload(&id, AssetKind::HeadMpk, "demo.txt", &b"x"[..])
        .await
        .unwrap_err();
    assert!(matches!(err, FileError::InvalidExtension { .. }));
}

#[tokio::test]
async fn upload_rejects_path_traversal_name() {
    let (_dir, mgr) = fresh_root();
    let id = create_legacy(&mgr, "x");
    let err = mgr
        .upload(&id, AssetKind::HeadMpk, "../escape.mpk", &b"x"[..])
        .await
        .unwrap_err();
    assert!(matches!(err, FileError::InvalidName(_)));
}

/// Guards [`validate_extension`] against a naive `&str` tail-slice refactor: a
/// suffix-match byte offset inside a multi-byte codepoint would panic on `&str`
/// slicing but the byte-level compare handles it. Dead as a live panic path
/// (callers gate on `validate_asset_name`'s ASCII allowlist first).
#[test]
fn validate_extension_does_not_panic_on_multibyte_utf8() {
    let r = validate_extension("Aä.mp", &["mpk"]);
    assert!(
        matches!(r, Err(FileError::InvalidExtension { .. })),
        "got {r:?}"
    );

    for bad in ["foo.äb", "foo.t\u{00e4}r", "naïve.bar", "Bä.x"] {
        let r = validate_extension(bad, &["mpk"]);
        assert!(
            matches!(r, Err(FileError::InvalidExtension { .. })),
            "expected InvalidExtension for {bad:?}, got {r:?}",
        );
    }
    // Multibyte stem with a recognised ASCII tail extension must succeed.
    validate_extension("naïve.mpk", &["mpk"]).expect("naïve.mpk");
    validate_extension("dataset-é.tar.gz", &["tar.gz", "zip"]).expect("dataset-é.tar.gz");
    validate_extension("ARCHIVE.TAR.GZ", &["tar.gz"]).expect("upper-case");
}

/// Asset names with embedded ASCII control chars are rejected (they corrupt
/// log lines / JSON metadata / receipt fields; `\n` splits log tokenisers).
#[test]
fn validate_asset_name_rejects_control_chars() {
    for bad in [
        "foo\nbar.mpk",
        "foo\tbar.mpk",
        "foo\x01bar.mpk",
        "x.mpk\x7f",
    ] {
        let err = validate_asset_name(bad).unwrap_err();
        assert!(
            matches!(err, FileError::InvalidName(_)),
            "control char {bad:?} not rejected"
        );
    }
    validate_asset_name("foo.mpk").unwrap();
    validate_asset_name("trained-09109000-3acb.labels.txt").unwrap();
}

/// Oversize upload rejects mid-stream with `PayloadTooLarge`, the tempfile drops
/// without committing (no orphan file or metadata row), and the same name+kind
/// re-uploads cleanly afterward.
#[tokio::test]
async fn admission_rejects_oversize_upload() {
    let dir = tempfile::tempdir().expect("tempdir");
    let cfg = AdmissionCfg {
        max_upload_bytes: 100,
        max_concurrent_uploads: 4,
    };
    let mgr = WorkspaceMgr::with_admission(dir.path().to_path_buf(), cfg);
    let id = create_legacy(&mgr, "admit");

    let payload = [0u8; 200];
    let err = mgr
        .upload(&id, AssetKind::HeadMpk, "big.mpk", &payload[..])
        .await
        .expect_err("oversize upload must reject");
    match err {
        FileError::PayloadTooLarge { observed, max } => {
            assert!(observed > 100, "observed must exceed cap: {observed}");
            assert_eq!(max, 100);
        }
        other => panic!("expected PayloadTooLarge, got {other:?}"),
    }

    let meta = mgr.read_metadata(&id).expect("read meta");
    assert!(
        meta.assets.is_empty(),
        "rejected upload must not commit metadata: {:?}",
        meta.assets
    );

    mgr.upload(&id, AssetKind::HeadMpk, "big.mpk", &b"under cap"[..])
        .await
        .expect("re-upload under cap must succeed");
}

/// Concurrency cap rejects the (max+1)th in-flight upload with
/// `TooManyConcurrentUploads` without blocking (`try_acquire_owned`, fail-fast).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn admission_rejects_too_many_concurrent_uploads() {
    let dir = tempfile::tempdir().expect("tempdir");
    let cfg = AdmissionCfg {
        max_upload_bytes: 1024 * 1024,
        max_concurrent_uploads: 1,
    };
    let mgr = WorkspaceMgr::with_admission(dir.path().to_path_buf(), cfg);
    let id = create_legacy(&mgr, "conc");

    let state = mgr
        .admission
        .as_ref()
        .expect("admission configured")
        .clone();
    let _hold = state
        .semaphore
        .clone()
        .try_acquire_owned()
        .expect("hold the only permit");

    let err = mgr
        .upload(&id, AssetKind::HeadMpk, "x.mpk", &b"any"[..])
        .await
        .expect_err("upload must reject when no permits");
    match err {
        FileError::TooManyConcurrentUploads { active, max } => {
            assert_eq!(active, 1, "1 active permit (the test's hold)");
            assert_eq!(max, 1);
        }
        other => panic!("expected TooManyConcurrentUploads, got {other:?}"),
    }
}

/// [`WorkspaceMgr::new`] (no admission) accepts uploads of any size: the cap is opt-in.
#[tokio::test]
async fn no_admission_accepts_any_size() {
    let (_dir, mgr) = fresh_root();
    let id = create_legacy(&mgr, "permissive");
    let payload = [0u8; 100 * 1024];
    mgr.upload(&id, AssetKind::HeadMpk, "big.mpk", &payload[..])
        .await
        .expect("no-admission ctor must accept any size");
}

/// `read_metadata` refuses a `schema_version` newer than this build, so a
/// downgraded daemon never silently drops future fields by reserialising the
/// older shape.
#[test]
fn read_metadata_rejects_schema_too_new() {
    let (_dir, mgr) = fresh_root();
    let id = create_legacy(&mgr, "future-schema");

    let path = mgr.workspace_dir(&id).join("metadata.json");
    let mut meta: WorkspaceMetadata =
        serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    meta.schema_version = WorkspaceMetadata::CURRENT + 1;
    std::fs::write(&path, serde_json::to_vec_pretty(&meta).unwrap()).unwrap();

    let err = mgr
        .read_metadata(&id)
        .expect_err("future schema must reject");
    match err {
        FileError::SchemaTooNew { found, max, .. } => {
            assert_eq!(found, WorkspaceMetadata::CURRENT + 1);
            assert_eq!(max, WorkspaceMetadata::CURRENT);
        }
        other => panic!("expected SchemaTooNew, got {other:?}"),
    }
}

#[test]
fn read_metadata_rejects_schema_too_old() {
    let (_dir, mgr) = fresh_root();
    let id = create_legacy(&mgr, "ancient-schema");

    let path = mgr.workspace_dir(&id).join("metadata.json");
    let mut meta: WorkspaceMetadata =
        serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    meta.schema_version = 0;
    std::fs::write(&path, serde_json::to_vec_pretty(&meta).unwrap()).unwrap();

    let err = mgr.read_metadata(&id).expect_err("schema 0 must reject");
    match err {
        FileError::SchemaTooOld { found, min, .. } => {
            assert_eq!(found, 0);
            assert_eq!(min, WorkspaceMetadata::MIN_COMPATIBLE);
        }
        other => panic!("expected SchemaTooOld, got {other:?}"),
    }
}

/// `foo.mpk` after `Foo.mpk` rejects with `NameConflict` while same-case
/// re-upload overwrites: guards the case-insensitive policy that on macOS HFS+
/// would otherwise let `tmp.persist` silently clobber a differing-case asset.
#[tokio::test]
async fn upload_rejects_case_insensitive_collision() {
    let (_dir, mgr) = fresh_root();
    let id = create_legacy(&mgr, "case-test");

    mgr.upload(&id, AssetKind::HeadMpk, "Foo.mpk", &b"first"[..])
        .await
        .expect("Foo.mpk");

    // Same-case re-upload overwrites: single metadata row, fresh sha.
    mgr.upload(&id, AssetKind::HeadMpk, "Foo.mpk", &b"second"[..])
        .await
        .expect("Foo.mpk re-upload");
    let meta = mgr.read_metadata(&id).unwrap();
    let foos: Vec<_> = meta
        .assets
        .iter()
        .filter(|a| a.kind == AssetKind::HeadMpk)
        .collect();
    assert_eq!(
        foos.len(),
        1,
        "same-case re-upload must overwrite, not duplicate: {foos:?}"
    );

    let err = mgr
        .upload(&id, AssetKind::HeadMpk, "foo.mpk", &b"different-case"[..])
        .await
        .expect_err("foo.mpk must collide with Foo.mpk");
    assert!(
        matches!(err, FileError::NameConflict(_)),
        "expected NameConflict, got {err:?}",
    );

    // Collision checked before rename: no orphan file, no half-committed row.
    let meta = mgr.read_metadata(&id).unwrap();
    let foos: Vec<_> = meta
        .assets
        .iter()
        .filter(|a| a.kind == AssetKind::HeadMpk)
        .collect();
    assert_eq!(
        foos.len(),
        1,
        "rejected upload must not commit metadata: {foos:?}"
    );
    assert_eq!(foos[0].name, "Foo.mpk");
}

#[tokio::test]
async fn upload_overwrite_updates_sha_in_metadata() {
    let (_dir, mgr) = fresh_root();
    let id = create_legacy(&mgr, "ov");
    let _ = mgr
        .upload(&id, AssetKind::HeadMpk, "h.mpk", &b"first"[..])
        .await
        .unwrap();
    let r2 = mgr
        .upload(&id, AssetKind::HeadMpk, "h.mpk", &b"second-revision"[..])
        .await
        .unwrap();
    let meta = mgr.read_metadata(&id).unwrap();
    assert_eq!(
        meta.assets.len(),
        1,
        "duplicate asset rows: {:?}",
        meta.assets
    );
    assert_eq!(meta.assets[0].sha256, r2.sha256);
    let on_disk = std::fs::read(&r2.path).unwrap();
    assert_eq!(on_disk, b"second-revision");
}

#[tokio::test]
async fn validate_detects_corruption_and_missing() {
    let (_dir, mgr) = fresh_root();
    let id = create_legacy(&mgr, "v");
    let r1 = mgr
        .upload(&id, AssetKind::HeadMpk, "good.mpk", &b"abc"[..])
        .await
        .unwrap();
    let r2 = mgr
        .upload(&id, AssetKind::HeadMpk, "missing.mpk", &b"def"[..])
        .await
        .unwrap();
    std::fs::write(&r1.path, b"tampered").unwrap();
    std::fs::remove_file(&r2.path).unwrap();
    // Orphan file present on disk but absent from metadata -> reported as extra.
    std::fs::write(mgr.workspace_dir(&id).join("weights/orphan.mpk"), b"orphan").unwrap();

    let report = mgr.validate(&id).unwrap();
    assert!(!report.ok);
    assert_eq!(report.corrupt.len(), 1);
    assert_eq!(report.corrupt[0].1, "good.mpk");
    assert_eq!(report.missing.len(), 1);
    assert_eq!(report.missing[0].1, "missing.mpk");
    assert_eq!(report.extra.len(), 1);
    assert_eq!(report.extra[0].1, "orphan.mpk");
}

/// Concurrent uploads must not lose metadata records: without the per-workspace
/// lock, racing read-modify-write tasks each see the stale `assets[]` and
/// clobber each other's appends.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn concurrent_uploads_preserve_all_records() {
    let (dir, mgr) = fresh_root();
    let id = create_legacy(&mgr, "conc");
    let mgr = std::sync::Arc::new(mgr);

    const N: usize = 32;
    let mut handles = Vec::with_capacity(N);
    for i in 0..N {
        let mgr = mgr.clone();
        handles.push(tokio::spawn(async move {
            let name = format!("file-{i:02}.mpk");
            let bytes = format!("payload-for-{i}").into_bytes();
            mgr.upload(&id, AssetKind::HeadMpk, &name, &bytes[..]).await
        }));
    }
    let mut ok = 0;
    for h in handles {
        h.await.expect("join").expect("upload");
        ok += 1;
    }
    assert_eq!(ok, N);

    let meta = mgr.read_metadata(&id).unwrap();
    let head_records: std::collections::HashSet<String> = meta
        .assets
        .iter()
        .filter(|a| a.kind == AssetKind::HeadMpk)
        .map(|a| a.name.as_str().to_string())
        .collect();
    assert_eq!(
        head_records.len(),
        N,
        "expected {N} unique head records, got {} (records were lost to a race)",
        head_records.len(),
    );
    for i in 0..N {
        let expected = format!("file-{i:02}.mpk");
        assert!(
            head_records.contains(&expected),
            "record {expected} missing from metadata",
        );
    }

    // File writes are atomic via tempfile + rename, so they need no metadata lock.
    for i in 0..N {
        let p = mgr
            .workspace_dir(&id)
            .join("weights")
            .join(format!("file-{i:02}.mpk"));
        assert!(p.exists(), "missing on-disk file: {}", p.display());
    }
    let _ = dir; // keep alive
}

/// Unicode filenames must reject pre-rename via `validate_asset_name`'s ASCII
/// allowlist; rejecting post-rename would orphan a file on disk, since the
/// metadata-record `AssetId::parse(name)?` rejects non-ASCII (via its ASCII
/// allowlist) only AFTER the rename has already landed.
#[tokio::test]
async fn upload_rejects_unicode_filename_pre_rename() {
    let (_dir, mgr) = fresh_root();
    let id = create_legacy(&mgr, "uni");
    let err = mgr
        .upload(&id, AssetKind::HeadMpk, "naïve.mpk", &b"x"[..])
        .await
        .expect_err("Unicode filename must reject");
    assert!(
        matches!(err, FileError::InvalidName(_)),
        "expected InvalidName, got {err:?}"
    );
    let weights = mgr.workspace_dir(&id).join("weights");
    let entries: Vec<_> = std::fs::read_dir(&weights)
        .unwrap()
        .filter_map(Result::ok)
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .collect();
    assert!(
        entries.is_empty(),
        "rejected upload must not leave files: {entries:?}"
    );
    let meta = mgr.read_metadata(&id).unwrap();
    assert!(meta.assets.is_empty());
}

/// Staging-then-install path also rejects Unicode filenames before the rename,
/// so no orphan lands in the asset subdir.
#[test]
fn install_from_path_rejects_unicode_filename() {
    let (_dir, mgr) = fresh_root();
    let id = create_legacy(&mgr, "uni-install");
    // `.tmp/` is created lazily by production writers; mkdir to mirror that.
    let staging = mgr
        .root()
        .join("workspaces")
        .join(id.to_string())
        .join(".tmp");
    std::fs::create_dir_all(&staging).expect("mkdir workspace .tmp");
    let tmp = tempfile::NamedTempFile::new_in(&staging).expect("tempfile");
    let err = mgr
        .install_from_path(&id, AssetKind::HeadLabels, "naïve.txt", tmp.path())
        .expect_err("Unicode filename must reject");
    assert!(
        matches!(err, FileError::InvalidName(_)),
        "expected InvalidName, got {err:?}"
    );
}

/// Concurrent `create("main")` must not both win: the registry mutex serializes
/// list-check-create, else both observe an empty registry and commit distinct
/// UUIDs sharing one name.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn concurrent_create_serializes_name_uniqueness() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mgr = std::sync::Arc::new(WorkspaceMgr::new(dir.path().to_path_buf()));

    const N: usize = 16;
    let mut handles = Vec::with_capacity(N);
    for _ in 0..N {
        let mgr = mgr.clone();
        handles.push(tokio::task::spawn_blocking(move || mgr.create("main")));
    }
    let mut ok_count = 0;
    let mut conflict_count = 0;
    for h in handles {
        match h.await.expect("join") {
            Ok(_) => ok_count += 1,
            Err(FileError::NameConflict(_)) => conflict_count += 1,
            Err(other) => panic!("unexpected error: {other:?}"),
        }
    }
    assert_eq!(ok_count, 1, "exactly one creator should win");
    assert_eq!(conflict_count, N - 1, "the rest must see NameConflict");

    let ids = mgr.list_workspaces().unwrap();
    assert_eq!(ids.len(), 1);
    let summary = mgr.summary(&ids[0]).unwrap();
    assert_eq!(summary.core.name, "main");
}

/// `asset_path` must `panic!` on every build (NOT a compiled-out
/// `debug_assert!`) for an unvalidated name, else a `..` name escapes the
/// workspace in release builds.
#[test]
#[should_panic(expected = "asset_path called with unvalidated name")]
fn asset_path_panics_on_unvalidated_name_in_release() {
    let (_dir, mgr) = fresh_root();
    let id = create_legacy(&mgr, "guard");
    let _ = mgr.asset_path(&id, AssetKind::HeadMpk, "../escape.mpk");
}

#[test]
fn asset_path_typed_skips_validation() {
    let (_dir, mgr) = fresh_root();
    let id = create_legacy(&mgr, "typed");
    let asset = AssetId::parse("foo.mpk").expect("valid");
    let p = mgr.asset_path_typed(&id, AssetKind::HeadMpk, &asset);
    let expected = mgr.workspace_dir(&id).join("weights").join("foo.mpk");
    assert_eq!(p, expected);
}

#[tokio::test]
async fn list_assets_filters_by_kind() {
    let (_dir, mgr) = fresh_root();
    let id = create_legacy(&mgr, "list");
    mgr.upload(&id, AssetKind::HeadMpk, "h.mpk", &b"x"[..])
        .await
        .unwrap();
    mgr.upload(&id, AssetKind::HeadLabels, "h.txt", &b"a\nb\n"[..])
        .await
        .unwrap();
    let heads = mgr.list_assets(&id, AssetKind::HeadMpk).unwrap();
    let labels = mgr.list_assets(&id, AssetKind::HeadLabels).unwrap();
    assert_eq!(heads.len(), 1);
    assert_eq!(heads[0].name, "h.mpk");
    assert_eq!(labels.len(), 1);
    assert_eq!(labels[0].name, "h.txt");
}

fn dataset_cat(name: &str) -> crate::common::asset_path::AssetPath {
    crate::common::asset_path::AssetPath::parse(&format!("datasets/{name}")).expect("asset path")
}

/// Seed `<ws>/datasets/<name>/` with slices by writing straight to disk,
/// bypassing `upload_workspace_file` so `workspace_revision` is NOT bumped.
fn seed_category(mgr: &WorkspaceMgr, id: &WorkspaceId, name: &str, files: &[(&str, &[u8])]) {
    let dir = mgr.workspace_dir(id).join("datasets").join(name);
    std::fs::create_dir_all(&dir).expect("mkdir category");
    for (fname, bytes) in files {
        std::fs::write(dir.join(fname), bytes).expect("write slice");
    }
}

fn revision_id(mgr: &WorkspaceMgr, id: &WorkspaceId) -> u64 {
    mgr.read_cached_core(id)
        .expect("core")
        .workspace_revision
        .id
}

#[test]
fn rename_dataset_category_moves_dir_and_bumps_revision() {
    let (_dir, mgr) = fresh_root();
    let mgr = std::sync::Arc::new(mgr);
    let id = mgr.create("ws").expect("create");
    seed_category(
        &mgr,
        &id,
        "dog",
        &[("a.wav", b"riff-a"), ("b.wav", b"riff-b")],
    );
    let rev_before = revision_id(&mgr, &id);
    let created_before = mgr.read_cached_core(&id).expect("core").created_at.clone();

    let receipt = mgr
        .rename_dataset_category(&id, &dataset_cat("dog"), &dataset_cat("puppy"))
        .expect("rename");

    let datasets = mgr.workspace_dir(&id).join("datasets");
    assert!(!datasets.join("dog").exists(), "old dir must be gone");
    assert!(datasets.join("puppy").is_dir(), "new dir must exist");
    // Content-addressed slices move verbatim (zero bytes copied).
    assert_eq!(
        std::fs::read(datasets.join("puppy").join("a.wav")).unwrap(),
        b"riff-a"
    );
    assert_eq!(
        std::fs::read(datasets.join("puppy").join("b.wav")).unwrap(),
        b"riff-b"
    );
    // Revision bumped exactly once (class label changed -> heads stale).
    assert_eq!(receipt.workspace_revision_id, rev_before + 1);
    assert_eq!(revision_id(&mgr, &id), rev_before + 1);
    // `created_at` must survive the workspace.json rewrite that bumps the revision.
    assert_eq!(
        mgr.read_cached_core(&id).unwrap().created_at,
        created_before,
        "rename must preserve created_at"
    );
}

#[test]
fn rename_dataset_category_self_rename_is_idempotent() {
    let (_dir, mgr) = fresh_root();
    let mgr = std::sync::Arc::new(mgr);
    let id = mgr.create("ws").expect("create");
    seed_category(&mgr, &id, "dog", &[("a.wav", b"x")]);
    let rev_before = revision_id(&mgr, &id);

    let receipt = mgr
        .rename_dataset_category(&id, &dataset_cat("dog"), &dataset_cat("dog"))
        .expect("self-rename ok");

    // No-op: dir untouched, revision NOT bumped (would needlessly stale heads).
    assert!(mgr.workspace_dir(&id).join("datasets").join("dog").is_dir());
    assert_eq!(receipt.workspace_revision_id, rev_before);
    assert_eq!(revision_id(&mgr, &id), rev_before);
}

#[test]
fn rename_dataset_category_missing_source_is_io_not_found() {
    let (_dir, mgr) = fresh_root();
    let mgr = std::sync::Arc::new(mgr);
    let id = mgr.create("ws").expect("create");
    // Missing category dir: pin `source.kind() == NotFound`, since the route's
    // 404 hinges on it -- a bare `Io { .. }` of another kind would silently
    // degrade the route to a 500.
    let err = mgr
        .rename_dataset_category(&id, &dataset_cat("ghost"), &dataset_cat("x"))
        .unwrap_err();
    assert!(
        matches!(&err, FileError::Io { source, .. } if source.kind() == std::io::ErrorKind::NotFound),
        "got {err:?}"
    );
}

#[test]
fn rename_dataset_category_rejects_existing_target() {
    let (_dir, mgr) = fresh_root();
    let mgr = std::sync::Arc::new(mgr);
    let id = mgr.create("ws").expect("create");
    seed_category(&mgr, &id, "dog", &[("a.wav", b"x")]);
    seed_category(&mgr, &id, "cat", &[("b.wav", b"y")]);
    let rev_before = revision_id(&mgr, &id);

    let err = mgr
        .rename_dataset_category(&id, &dataset_cat("dog"), &dataset_cat("cat"))
        .unwrap_err();
    assert!(matches!(err, FileError::NameConflict(_)), "got {err:?}");
    // Both dirs intact, no revision bump on the rejected rename.
    assert!(mgr.workspace_dir(&id).join("datasets").join("dog").is_dir());
    assert!(mgr.workspace_dir(&id).join("datasets").join("cat").is_dir());
    assert_eq!(revision_id(&mgr, &id), rev_before);
}

#[test]
fn rename_dataset_category_rejects_case_insensitive_sibling() {
    let (_dir, mgr) = fresh_root();
    let mgr = std::sync::Arc::new(mgr);
    let id = mgr.create("ws").expect("create");
    seed_category(&mgr, &id, "dog", &[("a.wav", b"x")]);
    seed_category(&mgr, &id, "Cat", &[("b.wav", b"y")]);
    let rev_before = revision_id(&mgr, &id);

    // dog -> cat collides case-insensitively with existing `Cat`. Which branch
    // rejects is host-FS-dependent (inode block on case-insensitive macOS,
    // sibling scan on case-sensitive Linux/ext4), but both yield NameConflict.
    let err = mgr
        .rename_dataset_category(&id, &dataset_cat("dog"), &dataset_cat("cat"))
        .unwrap_err();
    assert!(matches!(err, FileError::NameConflict(_)), "got {err:?}");
    // Both dirs intact, no revision bump on the rejected rename.
    assert!(mgr.workspace_dir(&id).join("datasets").join("dog").is_dir());
    assert!(mgr.workspace_dir(&id).join("datasets").join("Cat").is_dir());
    assert_eq!(revision_id(&mgr, &id), rev_before);
}

#[test]
fn rename_dataset_category_allows_case_only_self_rename() {
    let (_dir, mgr) = fresh_root();
    let mgr = std::sync::Arc::new(mgr);
    let id = mgr.create("ws").expect("create");
    seed_category(&mgr, &id, "Dog", &[("a.wav", b"x")]);

    // Dog -> dog is a legit case fix; the source dir must NOT be flagged as its
    // own case-insensitive collision. The allowing branch is host-FS-dependent
    // (`same_dir` dev+inode skip on macOS, `from_name` continue in the sibling
    // scan on case-sensitive Linux/ext4).
    mgr.rename_dataset_category(&id, &dataset_cat("Dog"), &dataset_cat("dog"))
        .expect("case-only rename ok");
    let datasets = mgr.workspace_dir(&id).join("datasets");
    // `dog` is addressable either way (old casing gone on case-sensitive FS,
    // both names resolve to one dir on case-insensitive FS).
    assert!(datasets.join("dog").is_dir());
    assert_eq!(
        std::fs::read(datasets.join("dog").join("a.wav")).unwrap(),
        b"x"
    );
}

#[test]
fn rename_dataset_category_rejects_multi_component_and_non_datasets() {
    let (_dir, mgr) = fresh_root();
    let mgr = std::sync::Arc::new(mgr);
    let id = mgr.create("ws").expect("create");

    // Multi-component source (smuggled `/`) is not a single class component.
    let nested = crate::common::asset_path::AssetPath::parse("datasets/dog/extra").unwrap();
    let err = mgr
        .rename_dataset_category(&id, &nested, &dataset_cat("x"))
        .unwrap_err();
    assert!(matches!(err, FileError::InvalidName(_)), "got {err:?}");

    // Non-`datasets` tree is rejected.
    let converters = crate::common::asset_path::AssetPath::parse("converters/dog").unwrap();
    let err = mgr
        .rename_dataset_category(&id, &converters, &dataset_cat("x"))
        .unwrap_err();
    assert!(matches!(err, FileError::InvalidName(_)), "got {err:?}");
}

#[test]
fn rename_dataset_category_rejected_while_train_active() {
    use crate::common::workspace::{JobReference, JobType};
    let (_dir, mgr) = fresh_root();
    let mgr = std::sync::Arc::new(mgr);
    let id = mgr.create("ws").expect("create");
    seed_category(&mgr, &id, "dog", &[("a.wav", b"x")]);
    let rev_before = revision_id(&mgr, &id);

    // Hold an active Train job: `scan_dataset` snapshots absolute paths
    // one-shot, so a mid-train rename would abort the run.
    let _train = mgr
        .jobs
        .try_acquire(
            JobType::Train,
            vec![JobReference::Workspace { workspace_id: id }],
            None,
        )
        .expect("acquire train job");

    let err = mgr
        .rename_dataset_category(&id, &dataset_cat("dog"), &dataset_cat("puppy"))
        .unwrap_err();
    assert!(matches!(err, FileError::JobConflict { .. }), "got {err:?}");
    // Untouched: dir stays, revision unchanged.
    assert!(mgr.workspace_dir(&id).join("datasets").join("dog").is_dir());
    assert_eq!(revision_id(&mgr, &id), rev_before);
}
