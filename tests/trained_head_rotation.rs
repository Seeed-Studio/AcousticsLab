//! Crash-safety integration tests for the trained-head rotation
//! primitive [`acousticslab::file_mgr::publish_trained_head`].
//!
//! The index commit (step 7) is the source of truth: it atomically
//! moves the publish point, so a later best-effort cleanup failure
//! (step 9) still leaves the rotation succeeded. The `.mpk` bytes
//! are opaque here (only `inference::head::load_inner` parses them).

#![allow(clippy::disallowed_methods)]

use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use acousticslab::common::ids::{HeadId, WorkspaceId};
use acousticslab::common::workspace::{
    HeadIndex, HeadManifest, HeadRecord, MAX_HEADS_PER_WORKSPACE, WorkspaceCore, WorkspaceRevision,
};
use acousticslab::file_mgr::{
    HeadRotationResult, PendingHead, WorkspaceCacheCell, head_artifact_path, head_index_path,
    head_manifest_path, heads_dir, publish_trained_head, read_head_index, read_head_manifest,
    read_workspace_core, write_head_index, write_workspace_core,
};

fn ws_id() -> WorkspaceId {
    WorkspaceId::parse("11111111-2222-4333-8444-555555555540").unwrap()
}

fn rev(id: u64) -> WorkspaceRevision {
    WorkspaceRevision {
        id,
        at: "2026-05-07T12:00:00Z".to_string(),
    }
}

fn sample_core(rev_id: u64, head_count: u8) -> WorkspaceCore {
    WorkspaceCore {
        id: ws_id(),
        name: "main".to_string(),
        tags: Vec::new(),
        created_at: "2026-05-07T12:34:56Z".to_string(),
        workspace_revision: rev(rev_id),
        head_count,
    }
}

fn sample_manifest(head_id: HeadId, rev_id: u64) -> HeadManifest {
    HeadManifest {
        head_id,
        workspace_id: ws_id(),
        workspace_revision: rev(rev_id),
        sha256: format!("sha-of-{head_id}"),
        n_classes: 3,
        size_bytes: 1024,
        created_at: "2026-05-07T12:34:56Z".to_string(),
        labels: vec!["cat".to_string(), "dog".to_string(), "bird".to_string()],
    }
}

/// Stage a fake `.mpk` tempfile under `<workspace>/.tmp/` with
/// distinct bytes per head id so an assertion can prove the right
/// file landed.
fn stage_mpk_tempfile(workspace_dir: &Path, head_id: HeadId) -> PathBuf {
    let tmp_dir = workspace_dir.join(".tmp");
    std::fs::create_dir_all(&tmp_dir).unwrap();
    let path = tmp_dir.join(format!("staged-{head_id}.mpk"));
    let mut f = std::fs::File::create(&path).unwrap();
    f.write_all(format!("MPK-{head_id}").as_bytes()).unwrap();
    f.sync_all().unwrap();
    path
}

fn fresh_workspace() -> (tempfile::TempDir, WorkspaceCacheCell) {
    let tmp = tempfile::tempdir().unwrap();
    let core = sample_core(0, 0);
    write_workspace_core(tmp.path(), &core).unwrap();
    write_head_index(tmp.path(), &HeadIndex::default()).unwrap();
    std::fs::create_dir_all(heads_dir(tmp.path())).unwrap();
    let cache = WorkspaceCacheCell::new(core, HeadIndex::default());
    (tmp, cache)
}

#[test]
fn publish_one_head_lands_index_atomic() {
    let (tmp, cache) = fresh_workspace();
    let head_id = HeadId::new();
    let mpk = stage_mpk_tempfile(tmp.path(), head_id);
    let manifest = sample_manifest(head_id, 5);

    let HeadRotationResult { displaced_head_id } = publish_trained_head(
        tmp.path(),
        &cache,
        PendingHead {
            head_id,
            mpk_tempfile: mpk.clone(),
            manifest: manifest.clone(),
        },
        None,
    )
    .unwrap();

    assert!(displaced_head_id.is_none(), "first publish never displaces");
    assert!(head_artifact_path(tmp.path(), head_id).is_file());
    assert!(head_manifest_path(tmp.path(), head_id).is_file());
    assert!(!mpk.exists(), "tempfile renamed away");
    let on_disk = read_head_index(tmp.path()).unwrap();
    assert_eq!(on_disk.heads.len(), 1);
    assert_eq!(on_disk.heads[0].head_id, head_id);
    let core = read_workspace_core(tmp.path()).unwrap();
    assert_eq!(core.head_count, 1);
    assert_eq!(*cache.heads(), on_disk);
    assert_eq!(*cache.core(), core);
}

/// Sliding window: the `MAX_HEADS_PER_WORKSPACE + 1`th publish
/// displaces the oldest and removes its `.mpk`/`.json` bytes; the
/// most-recent `MAX_HEADS_PER_WORKSPACE` heads remain in the index.
#[test]
fn publish_overflowing_heads_drops_oldest() {
    let (tmp, cache) = fresh_workspace();
    let oldest = HeadId::new();
    let mut filler: Vec<HeadId> = Vec::with_capacity(MAX_HEADS_PER_WORKSPACE);
    filler.push(oldest);
    for _ in 1..MAX_HEADS_PER_WORKSPACE {
        filler.push(HeadId::new());
    }
    for &h in &filler {
        let mpk = stage_mpk_tempfile(tmp.path(), h);
        publish_trained_head(
            tmp.path(),
            &cache,
            PendingHead {
                head_id: h,
                mpk_tempfile: mpk,
                manifest: sample_manifest(h, 1),
            },
            None,
        )
        .unwrap();
    }
    assert_eq!(cache.heads().heads.len(), MAX_HEADS_PER_WORKSPACE);
    let newest = HeadId::new();
    let mpk = stage_mpk_tempfile(tmp.path(), newest);
    let result = publish_trained_head(
        tmp.path(),
        &cache,
        PendingHead {
            head_id: newest,
            mpk_tempfile: mpk,
            manifest: sample_manifest(newest, 3),
        },
        None,
    )
    .unwrap();
    assert_eq!(result.displaced_head_id, Some(oldest));
    let on_disk = read_head_index(tmp.path()).unwrap();
    assert_eq!(on_disk.heads.len(), MAX_HEADS_PER_WORKSPACE);
    assert_eq!(on_disk.heads[0].head_id, newest);
    // Survivors are the non-oldest filler in newest-first order.
    for (slot, &h) in filler.iter().skip(1).rev().enumerate() {
        assert_eq!(on_disk.heads[slot + 1].head_id, h);
    }
    assert!(!head_artifact_path(tmp.path(), oldest).exists());
    assert!(!head_manifest_path(tmp.path(), oldest).exists());
    assert!(head_artifact_path(tmp.path(), newest).is_file());
    for &h in filler.iter().skip(1) {
        assert!(head_artifact_path(tmp.path(), h).is_file());
    }
}

/// Rotation must not touch unrelated `<random>.{mpk,json}` files
/// under `heads/`; those are residue that boot recovery sweeps.
#[test]
fn publish_does_not_disturb_orphans() {
    let (tmp, cache) = fresh_workspace();
    let orphan_id = HeadId::new();
    let orphan_mpk = head_artifact_path(tmp.path(), orphan_id);
    let orphan_json = head_manifest_path(tmp.path(), orphan_id);
    std::fs::write(&orphan_mpk, b"orphan-mpk").unwrap();
    std::fs::write(&orphan_json, b"{}").unwrap();

    let h1 = HeadId::new();
    let mpk = stage_mpk_tempfile(tmp.path(), h1);
    publish_trained_head(
        tmp.path(),
        &cache,
        PendingHead {
            head_id: h1,
            mpk_tempfile: mpk,
            manifest: sample_manifest(h1, 1),
        },
        None,
    )
    .unwrap();
    assert!(orphan_mpk.exists(), "rotation must not touch orphans");
    assert!(orphan_json.exists());
    let on_disk = read_head_index(tmp.path()).unwrap();
    assert!(!on_disk.heads.iter().any(|r| r.head_id == orphan_id));
}

/// Cache load tolerates `heads.json` referencing a missing file
/// because it does not stat `heads/` (boot recovery is the sweep
/// surface); pins this so a future load cannot silently start statting.
#[test]
fn cache_load_tolerates_phantom_index_entry() {
    let tmp = tempfile::tempdir().unwrap();
    let core = sample_core(0, 1);
    write_workspace_core(tmp.path(), &core).unwrap();
    let phantom = HeadId::new();
    let phantom_record = HeadRecord {
        head_id: phantom,
        workspace_revision: rev(0),
        sha256: "y".into(),
        n_classes: 1,
        size_bytes: 0,
        created_at: "2026-05-07T12:00:00Z".to_string(),
    };
    let bad_index = HeadIndex {
        heads: vec![phantom_record],
    };
    write_head_index(tmp.path(), &bad_index).unwrap();
    let cell = WorkspaceCacheCell::load_from_disk(tmp.path()).unwrap();
    assert_eq!(cell.heads().heads.len(), 1);
    assert!(!head_artifact_path(tmp.path(), phantom).exists());
    assert!(!head_manifest_path(tmp.path(), phantom).exists());
}

/// Displaced-file cleanup (step 9) failing because the file is
/// already gone (fs race / concurrent orphan sweep) still leaves
/// the rotation succeeded: the index commit (step 7) already moved
/// the publish point.
#[test]
fn publish_succeeds_when_displaced_files_are_already_gone() {
    let (tmp, cache) = fresh_workspace();
    let oldest = HeadId::new();
    let mut filler: Vec<HeadId> = Vec::with_capacity(MAX_HEADS_PER_WORKSPACE);
    filler.push(oldest);
    for _ in 1..MAX_HEADS_PER_WORKSPACE {
        filler.push(HeadId::new());
    }
    for &h in &filler {
        let mpk = stage_mpk_tempfile(tmp.path(), h);
        publish_trained_head(
            tmp.path(),
            &cache,
            PendingHead {
                head_id: h,
                mpk_tempfile: mpk,
                manifest: sample_manifest(h, 1),
            },
            None,
        )
        .unwrap();
    }
    // Race: remove the soon-to-be-displaced files before the overflow publish.
    std::fs::remove_file(head_artifact_path(tmp.path(), oldest)).unwrap();
    std::fs::remove_file(head_manifest_path(tmp.path(), oldest)).unwrap();

    let newest = HeadId::new();
    let mpk = stage_mpk_tempfile(tmp.path(), newest);
    let result = publish_trained_head(
        tmp.path(),
        &cache,
        PendingHead {
            head_id: newest,
            mpk_tempfile: mpk,
            manifest: sample_manifest(newest, 3),
        },
        None,
    )
    .unwrap();
    assert_eq!(result.displaced_head_id, Some(oldest));
    let on_disk = read_head_index(tmp.path()).unwrap();
    assert_eq!(on_disk.heads.len(), MAX_HEADS_PER_WORKSPACE);
    assert_eq!(on_disk.heads[0].head_id, newest);
    for (slot, &h) in filler.iter().skip(1).rev().enumerate() {
        assert_eq!(on_disk.heads[slot + 1].head_id, h);
    }
}

/// Crash between steps 6 and 7 (files land under `heads/` but
/// `heads.json` is unchanged): the index view treats the new head
/// as invisible, leaving the unreferenced files for boot recovery.
#[test]
fn crash_between_steps_6_and_7_leaves_orphan_files_invisible_to_index() {
    let (tmp, cache) = fresh_workspace();
    let h1 = HeadId::new();
    let mpk = stage_mpk_tempfile(tmp.path(), h1);
    publish_trained_head(
        tmp.path(),
        &cache,
        PendingHead {
            head_id: h1,
            mpk_tempfile: mpk,
            manifest: sample_manifest(h1, 1),
        },
        None,
    )
    .unwrap();
    let pre_crash_index = read_head_index(tmp.path()).unwrap();

    // Crash between steps 6 and 7 of a follow-up h2 publish: write
    // the new files into heads/ but do NOT rewrite heads.json.
    let h2 = HeadId::new();
    let mpk2_final = head_artifact_path(tmp.path(), h2);
    let json2_final = head_manifest_path(tmp.path(), h2);
    std::fs::write(&mpk2_final, b"orphan-h2-mpk").unwrap();
    std::fs::write(
        &json2_final,
        serde_json::to_vec(&sample_manifest(h2, 2)).unwrap(),
    )
    .unwrap();

    let post_crash_index = read_head_index(tmp.path()).unwrap();
    assert_eq!(
        post_crash_index, pre_crash_index,
        "heads.json must not reflect the partial publish",
    );
    assert!(post_crash_index.heads.iter().all(|r| r.head_id != h2));
    // Orphan files present but invisible to the index; boot recovery sweeps.
    assert!(mpk2_final.is_file());
    assert!(json2_final.is_file());

    assert!(head_index_path(tmp.path()).is_file());
}

/// A follow-up publish converges despite orphans from a prior
/// crashed publish: the rotation primitive is not derailed and
/// leaves them untouched for boot recovery.
#[test]
fn rotation_converges_after_simulated_partial_crash() {
    let (tmp, cache) = fresh_workspace();
    // Orphan simulating a prior crashed publish.
    let h_orphan = HeadId::new();
    std::fs::write(head_artifact_path(tmp.path(), h_orphan), b"residue").unwrap();
    std::fs::write(head_manifest_path(tmp.path(), h_orphan), b"{}").unwrap();
    let h1 = HeadId::new();
    let mpk = stage_mpk_tempfile(tmp.path(), h1);
    let result = publish_trained_head(
        tmp.path(),
        &cache,
        PendingHead {
            head_id: h1,
            mpk_tempfile: mpk,
            manifest: sample_manifest(h1, 1),
        },
        None,
    )
    .unwrap();
    assert!(result.displaced_head_id.is_none());
    assert!(head_artifact_path(tmp.path(), h1).is_file());
    assert!(head_artifact_path(tmp.path(), h_orphan).is_file());
    assert!(head_manifest_path(tmp.path(), h_orphan).is_file());
    let idx = read_head_index(tmp.path()).unwrap();
    assert_eq!(idx.heads.len(), 1);
    assert_eq!(idx.heads[0].head_id, h1);
}

/// `delete_head` round-trip via the `WorkspaceMgr` surface. A
/// fresh `FsServiceImpl` is built AFTER the disk writes so its
/// cache lazy-loads the post-publish state rather than being
/// shadowed by an empty create-side cache.
#[tokio::test(flavor = "current_thread")]
async fn delete_head_via_workspace_mgr_round_trip() {
    use acousticslab::file_mgr::{FsService, FsServiceImpl};
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().to_path_buf();
    // Create via the production lifecycle path for a canonical layout.
    let create_fs: Arc<dyn FsService> = Arc::new(FsServiceImpl::new(root.clone()));
    create_fs.ensure_root_layout().unwrap();
    let id = create_fs.create("rotation-mgr-test").unwrap();
    let workspace_dir = root.join("workspaces").join(id.to_string());
    drop(create_fs);
    // Stage two heads directly, mirroring publish_trained_head output;
    // heads/ is normally created lazily by the publisher, so pre-create it.
    std::fs::create_dir_all(workspace_dir.join("heads")).unwrap();
    let h1 = HeadId::new();
    let h2 = HeadId::new();
    for &h in &[h1, h2] {
        let mpk = stage_mpk_tempfile(&workspace_dir, h);
        std::fs::rename(&mpk, head_artifact_path(&workspace_dir, h)).unwrap();
        let manifest = sample_manifest(h, 1);
        std::fs::write(
            head_manifest_path(&workspace_dir, h),
            serde_json::to_vec(&manifest).unwrap(),
        )
        .unwrap();
    }
    let mut idx = HeadIndex::default();
    idx.heads.push(HeadRecord {
        head_id: h2,
        workspace_revision: rev(1),
        sha256: format!("sha-of-{h2}"),
        n_classes: 3,
        size_bytes: 1024,
        created_at: "2026-05-07T12:34:56Z".to_string(),
    });
    idx.heads.push(HeadRecord {
        head_id: h1,
        workspace_revision: rev(1),
        sha256: format!("sha-of-{h1}"),
        n_classes: 3,
        size_bytes: 1024,
        created_at: "2026-05-07T12:34:56Z".to_string(),
    });
    write_head_index(&workspace_dir, &idx).unwrap();
    let mut core = read_workspace_core(&workspace_dir).unwrap();
    core.head_count = 2;
    write_workspace_core(&workspace_dir, &core).unwrap();

    // Fresh FsServiceImpl lazy-loads post-publish disk state, as the
    // daemon does on first touch of a workspace recovered from a prior run.
    let fs: Arc<dyn FsService> = Arc::new(FsServiceImpl::new(root.clone()));
    let summary = fs.summary(&id).unwrap();
    assert_eq!(summary.heads.heads.len(), 2);

    fs.delete_head(&id, h1).unwrap();
    let summary = fs.summary(&id).unwrap();
    assert_eq!(summary.heads.heads.len(), 1);
    assert_eq!(summary.heads.heads[0].head_id, h2);
    assert!(!head_artifact_path(&workspace_dir, h1).exists());
    assert!(!head_manifest_path(&workspace_dir, h1).exists());
    assert_eq!(summary.core.head_count, 1);

    // Deleting a phantom head_id surfaces 404.
    let phantom = HeadId::new();
    let err = fs.delete_head(&id, phantom).unwrap_err();
    use acousticslab::common::error::{Categorized, ErrorKind};
    assert_eq!(err.kind(), ErrorKind::NotFound);
}

// Active-source pin contract: a workspace's active source head is
// never auto-removed -- delete_head returns ActiveSourcePinned (409)
// and overflow publish evicts a non-pinned tail entry instead. Live
// inference is unaffected either way: the active generation owns a
// separate byte copy under active/generations/<id>/.

/// Synth an active generation pointing at `(workspace_id, head_id)`,
/// mirroring `publish_active_generation`'s on-disk shape. Head bytes
/// are not copied; the pin check reads only the manifest origin + ids.
fn write_synth_active_generation(
    root: &std::path::Path,
    workspace_id: WorkspaceId,
    source_head_id: HeadId,
    workspace_revision: WorkspaceRevision,
) -> String {
    use acousticslab::common::workspace::{ActiveHeadManifest, ActiveOrigin};
    use acousticslab::file_mgr::{
        ActiveCurrentPointer, active_generation_dir, write_active_current, write_active_manifest,
    };
    let activation_id = "11111111-2222-4333-8444-555555555ace".to_string();
    let gen_dir = active_generation_dir(root, &activation_id);
    std::fs::create_dir_all(&gen_dir).unwrap();
    let manifest = ActiveHeadManifest {
        origin: ActiveOrigin::Head {
            source_workspace_id: workspace_id,
            source_head_id,
            workspace_revision,
        },
        runtime_head_id: source_head_id,
        sha256: format!("sha-of-{source_head_id}"),
        labels_sha256: "labels-sha".to_string(),
        n_classes: 3,
        labels: vec!["cat".to_string(), "dog".to_string(), "bird".to_string()],
        activated_at: "2026-05-07T12:34:56Z".to_string(),
    };
    write_active_manifest(root, &activation_id, &manifest).unwrap();
    write_active_current(
        root,
        &ActiveCurrentPointer {
            activation_id: activation_id.clone(),
        },
    )
    .unwrap();
    activation_id
}

/// `delete_head` refuses the active source with 409
/// `ActiveSourcePinned`; the pin clears once the active manifest
/// points elsewhere and the same delete then succeeds.
#[tokio::test(flavor = "current_thread")]
async fn delete_head_refuses_active_source_pin() {
    use acousticslab::common::error::{Categorized, ErrorKind};
    use acousticslab::file_mgr::{FileError, FsService, FsServiceImpl};
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().to_path_buf();
    let create_fs: Arc<dyn FsService> = Arc::new(FsServiceImpl::new(root.clone()));
    create_fs.ensure_root_layout().unwrap();
    let id = create_fs.create("active-pin-test").unwrap();
    let workspace_dir = root.join("workspaces").join(id.to_string());
    drop(create_fs);

    // Stage h1 + h2 directly, in isolation from the rotation primitive.
    std::fs::create_dir_all(workspace_dir.join("heads")).unwrap();
    let h1 = HeadId::new();
    let h2 = HeadId::new();
    for &h in &[h1, h2] {
        let mpk = stage_mpk_tempfile(&workspace_dir, h);
        std::fs::rename(&mpk, head_artifact_path(&workspace_dir, h)).unwrap();
        let manifest = sample_manifest(h, 1);
        std::fs::write(
            head_manifest_path(&workspace_dir, h),
            serde_json::to_vec(&manifest).unwrap(),
        )
        .unwrap();
    }
    let mut idx = HeadIndex::default();
    idx.heads.push(HeadRecord {
        head_id: h2,
        workspace_revision: rev(1),
        sha256: format!("sha-of-{h2}"),
        n_classes: 3,
        size_bytes: 1024,
        created_at: "2026-05-07T12:34:56Z".to_string(),
    });
    idx.heads.push(HeadRecord {
        head_id: h1,
        workspace_revision: rev(1),
        sha256: format!("sha-of-{h1}"),
        n_classes: 3,
        size_bytes: 1024,
        created_at: "2026-05-07T12:34:56Z".to_string(),
    });
    write_head_index(&workspace_dir, &idx).unwrap();
    let mut core = read_workspace_core(&workspace_dir).unwrap();
    core.head_count = 2;
    write_workspace_core(&workspace_dir, &core).unwrap();

    let activation_id = write_synth_active_generation(&root, id, h1, rev(1));

    let fs: Arc<dyn FsService> = Arc::new(FsServiceImpl::new(root.clone()));

    // Deleting the active source h1 is refused with 409. Downcast
    // pins the exact variant so a different Conflict discriminant
    // (e.g. JobConflict) cannot pass this test.
    let err = fs.delete_head(&id, h1).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::Conflict);
    let inner = std::error::Error::source(&err)
        .and_then(|e| e.downcast_ref::<FileError>())
        .expect("FsError wraps a FileError for this surface");
    assert!(
        matches!(inner, FileError::ActiveSourcePinned { .. }),
        "expected ActiveSourcePinned, got {inner:?}",
    );
    assert!(head_artifact_path(&workspace_dir, h1).is_file());
    assert!(head_manifest_path(&workspace_dir, h1).is_file());
    // Non-pinned head deletes normally.
    fs.delete_head(&id, h2).unwrap();
    assert!(!head_artifact_path(&workspace_dir, h2).exists());

    // Pointing the active manifest elsewhere releases the pin; h1 deletes.
    let other = HeadId::new();
    use acousticslab::common::workspace::{ActiveHeadManifest, ActiveOrigin};
    use acousticslab::file_mgr::write_active_manifest;
    let new_manifest = ActiveHeadManifest {
        origin: ActiveOrigin::Head {
            source_workspace_id: id,
            source_head_id: other,
            workspace_revision: rev(1),
        },
        runtime_head_id: other,
        sha256: format!("sha-of-{other}"),
        labels_sha256: "labels-sha".to_string(),
        n_classes: 3,
        labels: vec!["cat".to_string(), "dog".to_string(), "bird".to_string()],
        activated_at: "2026-05-07T12:34:56Z".to_string(),
    };
    write_active_manifest(&root, &activation_id, &new_manifest).unwrap();
    fs.delete_head(&id, h1).unwrap();
    assert!(!head_artifact_path(&workspace_dir, h1).exists());
}

/// The `WorkspaceMgr` publish wrapper resolves the active-source
/// pin and forwards it to the primitive: with the oldest
/// (chronological tail) head pinned, the overflow publish displaces
/// the second-oldest non-pinned head instead.
#[tokio::test(flavor = "current_thread")]
async fn publish_through_workspace_mgr_respects_active_pin() {
    use acousticslab::file_mgr::{FsService, FsServiceImpl};
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().to_path_buf();
    let create_fs: Arc<dyn FsService> = Arc::new(FsServiceImpl::new(root.clone()));
    create_fs.ensure_root_layout().unwrap();
    let id = create_fs.create("active-pin-rotation").unwrap();
    let workspace_dir = root.join("workspaces").join(id.to_string());
    drop(create_fs);

    // Fill to cap: `pinned` is the oldest (chronological tail) and
    // gets activated; `expected_evicted` (second-oldest, non-pinned)
    // becomes the eviction target for the later overflow publish.
    std::fs::create_dir_all(workspace_dir.join("heads")).unwrap();
    let pinned = HeadId::new();
    let expected_evicted = HeadId::new();
    let mut filler: Vec<HeadId> = Vec::with_capacity(MAX_HEADS_PER_WORKSPACE);
    filler.push(pinned);
    filler.push(expected_evicted);
    for _ in 2..MAX_HEADS_PER_WORKSPACE {
        filler.push(HeadId::new());
    }
    let fs: Arc<dyn FsService> = Arc::new(FsServiceImpl::new(root.clone()));
    for (rev_id, &h) in filler.iter().enumerate() {
        let mpk = stage_mpk_tempfile(&workspace_dir, h);
        fs.publish_trained_head(
            &id,
            PendingHead {
                head_id: h,
                mpk_tempfile: mpk,
                manifest: HeadManifest {
                    head_id: h,
                    workspace_id: id,
                    workspace_revision: rev(rev_id as u64 + 1),
                    sha256: format!("sha-of-{h}"),
                    n_classes: 3,
                    size_bytes: 1024,
                    created_at: "2026-05-07T12:34:56Z".to_string(),
                    labels: vec!["cat".to_string(), "dog".to_string(), "bird".to_string()],
                },
            },
        )
        .unwrap();
    }
    let summary = fs.summary(&id).unwrap();
    assert_eq!(summary.heads.heads.len(), MAX_HEADS_PER_WORKSPACE);

    let _activation_id = write_synth_active_generation(&root, id, pinned, rev(1));

    // Overflow publish: unprotected this would displace the
    // chronological tail (`pinned`); the pin redirects eviction to
    // `expected_evicted` instead.
    let overflow = HeadId::new();
    let mpk_overflow = stage_mpk_tempfile(&workspace_dir, overflow);
    let result = fs
        .publish_trained_head(
            &id,
            PendingHead {
                head_id: overflow,
                mpk_tempfile: mpk_overflow,
                manifest: HeadManifest {
                    head_id: overflow,
                    workspace_id: id,
                    workspace_revision: rev(filler.len() as u64 + 1),
                    sha256: format!("sha-of-{overflow}"),
                    n_classes: 3,
                    size_bytes: 1024,
                    created_at: "2026-05-07T12:34:56Z".to_string(),
                    labels: vec!["cat".to_string(), "dog".to_string(), "bird".to_string()],
                },
            },
        )
        .unwrap();
    assert_eq!(
        result.displaced_head_id,
        Some(expected_evicted),
        "pinned head survives; chronological tail just above it gets evicted",
    );

    let summary = fs.summary(&id).unwrap();
    let head_ids: Vec<HeadId> = summary.heads.heads.iter().map(|r| r.head_id).collect();
    assert!(
        head_ids.contains(&pinned),
        "active source survives rotation",
    );
    assert!(head_ids.contains(&overflow), "newest head lands");
    assert!(
        !head_ids.contains(&expected_evicted),
        "second-oldest non-pinned head evicted",
    );
    assert!(head_artifact_path(&workspace_dir, pinned).is_file());
    assert!(!head_artifact_path(&workspace_dir, expected_evicted).exists());
    assert!(head_artifact_path(&workspace_dir, overflow).is_file());
}

// Published-shape pins: the JSON blobs downstream readers (cache
// load, boot recovery, activation) consume are pinned to their
// minimized on-disk shape so a future edit re-adding legacy
// provenance fields (dataset_path, training_cfg*, dataset_revision*)
// parse-fails via deny_unknown_fields before corrupting inference.

/// Pins the published `<head_id>.json` to exactly the minimized
/// `HeadManifest` field set, never the legacy provenance fields.
#[test]
fn published_manifest_carries_minimized_field_set_only() {
    let (tmp, cache) = fresh_workspace();
    let head_id = HeadId::new();
    let mpk = stage_mpk_tempfile(tmp.path(), head_id);
    publish_trained_head(
        tmp.path(),
        &cache,
        PendingHead {
            head_id,
            mpk_tempfile: mpk,
            manifest: sample_manifest(head_id, 7),
        },
        None,
    )
    .unwrap();

    // Inspect keys via serde_json::Value (the wire shape), not
    // read_head_manifest: typed parsing would ignore an unknown key
    // if deny_unknown_fields were dropped, making the check a tautology.
    let bytes = std::fs::read(head_manifest_path(tmp.path(), head_id)).unwrap();
    let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    let obj = v.as_object().expect("manifest is a JSON object");
    let actual: std::collections::BTreeSet<&str> = obj.keys().map(String::as_str).collect();
    let expected: std::collections::BTreeSet<&str> = [
        "head_id",
        "workspace_id",
        "workspace_revision",
        "sha256",
        "n_classes",
        "size_bytes",
        "created_at",
        "labels",
    ]
    .into_iter()
    .collect();
    assert_eq!(
        actual, expected,
        "published HeadManifest must carry exactly the minimized field set; got {actual:?}",
    );
    // Defence in depth against a future expected-set growth quietly
    // re-admitting legacy fields (the equality above already covers it).
    for forbidden in [
        "dataset_path",
        "training_cfg",
        "training_cfg_sha256",
        "dataset_revision",
        "dataset_revision_at_train",
    ] {
        assert!(
            !obj.contains_key(forbidden),
            "legacy field {forbidden:?} must not appear in the manifest",
        );
    }
    let rev = obj["workspace_revision"]
        .as_object()
        .expect("workspace_revision is a sub-object");
    let rev_keys: std::collections::BTreeSet<&str> = rev.keys().map(String::as_str).collect();
    let expected_rev: std::collections::BTreeSet<&str> = ["id", "at"].into_iter().collect();
    assert_eq!(
        rev_keys, expected_rev,
        "workspace_revision sub-object must carry exactly id + at; got {rev_keys:?}",
    );
}

/// Pins the published `heads.json` `HeadRecord` entries to exactly
/// the minimized field set, never the legacy provenance fields.
#[test]
fn published_index_carries_minimized_record_field_set_only() {
    let (tmp, cache) = fresh_workspace();
    let head_id = HeadId::new();
    let mpk = stage_mpk_tempfile(tmp.path(), head_id);
    publish_trained_head(
        tmp.path(),
        &cache,
        PendingHead {
            head_id,
            mpk_tempfile: mpk,
            manifest: sample_manifest(head_id, 11),
        },
        None,
    )
    .unwrap();

    let bytes = std::fs::read(head_index_path(tmp.path())).unwrap();
    let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    let entries = v["heads"].as_array().expect("heads is an array");
    assert_eq!(entries.len(), 1, "exactly one published head");
    let rec = entries[0].as_object().expect("HeadRecord is a JSON object");
    let actual: std::collections::BTreeSet<&str> = rec.keys().map(String::as_str).collect();
    let expected: std::collections::BTreeSet<&str> = [
        "head_id",
        "workspace_revision",
        "sha256",
        "n_classes",
        "size_bytes",
        "created_at",
    ]
    .into_iter()
    .collect();
    assert_eq!(
        actual, expected,
        "published HeadRecord must carry exactly the field set; got {actual:?}",
    );
    for forbidden in [
        "dataset_path",
        "training_cfg_sha256",
        "dataset_revision",
        "dataset_revision_at_train",
        "labels",
        "workspace_id",
    ] {
        assert!(
            !rec.contains_key(forbidden),
            "legacy / non-record field {forbidden:?} must not appear in HeadRecord",
        );
    }
}

/// Read-boundary defence: a hand-staged legacy `<head_id>.json`
/// body parse-fails through `read_head_manifest` (deny_unknown_fields),
/// so stale-binary boots or operator tampering fail closed instead
/// of feeding legacy metadata into the inference path.
#[test]
fn legacy_manifest_shape_parse_fails_on_read() {
    let tmp = tempfile::tempdir().unwrap();
    write_workspace_core(tmp.path(), &sample_core(0, 0)).unwrap();
    write_head_index(tmp.path(), &HeadIndex::default()).unwrap();
    std::fs::create_dir_all(heads_dir(tmp.path())).unwrap();

    let head_id = HeadId::new();
    let round_1_body = serde_json::json!({
        "head_id": head_id.to_string(),
        "workspace_id": ws_id().to_string(),
        // Legacy alias; the schema now requires `workspace_revision`.
        "dataset_revision_at_train": { "id": 5, "at": "2026-05-07T12:00:00Z" },
        "sha256": "abc",
        "n_classes": 3,
        "size_bytes": 1024,
        "created_at": "2026-05-07T12:34:56Z",
        "labels": ["a", "b", "c"],
        // Legacy fields that the current schema drops.
        "dataset_path": "audio/cat",
        "training_cfg_sha256": "deadbeef",
        "training_cfg": { "epochs": 4, "batch_size": 16, "learning_rate": 0.001 },
    });
    std::fs::write(
        head_manifest_path(tmp.path(), head_id),
        serde_json::to_vec(&round_1_body).unwrap(),
    )
    .unwrap();

    let res = read_head_manifest(tmp.path(), head_id);
    assert!(
        res.is_err(),
        "legacy manifest body on disk must parse-fail; got {res:?}",
    );
}

/// Same read-boundary defence for `heads.json`: a hand-staged
/// legacy `HeadRecord` parse-fails through `read_head_index`, which
/// both cache load and boot recovery consume.
#[test]
fn legacy_head_index_shape_parse_fails_on_read() {
    let tmp = tempfile::tempdir().unwrap();
    write_workspace_core(tmp.path(), &sample_core(0, 0)).unwrap();
    std::fs::create_dir_all(heads_dir(tmp.path())).unwrap();

    let head_id = HeadId::new();
    let round_1_index = serde_json::json!({
        "heads": [{
            "head_id": head_id.to_string(),
            "dataset_revision_at_train": { "id": 5, "at": "2026-05-07T12:00:00Z" },
            "sha256": "abc",
            "n_classes": 3,
            "size_bytes": 1024,
            "created_at": "2026-05-07T12:34:56Z",
            "dataset_path": "audio/cat",
            "training_cfg_sha256": "deadbeef",
        }],
    });
    std::fs::write(
        head_index_path(tmp.path()),
        serde_json::to_vec(&round_1_index).unwrap(),
    )
    .unwrap();

    let res = read_head_index(tmp.path());
    assert!(
        res.is_err(),
        "legacy heads.json body on disk must parse-fail; got {res:?}",
    );
}
