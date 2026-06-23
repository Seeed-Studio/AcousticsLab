import type { Messages } from '../types';

// Canonical catalog every other locale's `satisfies Messages` shape is anchored against.
// Wording convention: sentence case; terminal period on full sentences, none on fragment labels.
// House style:
//  - Em dash ' — ' (U+2014, space-padded) is the only inline break/parenthetical dash; never a spaced ASCII hyphen ' - '.
//  - No comma splices: split independent clauses into two sentences (period); em dash only for a tight diagnosis→consequence pair.
//  - Exactly one space after a sentence-terminal period.
//  - Inline separator is ' · ' (U+00B7); in-progress marker is '…' (U+2026). Never ' / ' (except compact key hints), ' | ', or literal '...'.
//  - Archive token is always '.alpkg' (lowercase, leading dot); format names stay uppercase (TFJS, WAV, ALSA). Codec 'opus' is lowercase.
//  - 'Top-K' for standalone control/field labels; 'top-k', 'cpu', 'rss', 'val acc' etc. inside the dense lowercase readout set.
//  - Hyphenate re- + verb consistently (re-train, re-export, re-deploy, re-drop, re-import).
//  - Pipeline/runtime status pills are Title-Case across all surfaces (Deployed / Default / Standby / Detached / Workspace).
//  - Straight quotes and apostrophes only — no curly typography (must round-trip through every locale file).
//  - Lower bounds read 'N or greater'; never hardcode a plural against an interpolated count (use the `${n === 1 ? '' : 's'}` idiom).
export const en = {
  app: {
    name: 'AcousticsLab',
    description:
      'A private, multi-backend, fully-local AI/ML toolkit for developing and deploying real-time sound event detection.'
  },
  routes: {
    dashboard_title: (brand) => brand,
    workspace_list_title: (brand) => `Workspaces · ${brand}`,
    workspace_detail_title: (workspaceName, brand) => `${workspaceName} · ${brand}`
  },
  nav: {
    dashboard: 'Dashboard',
    workspaces: 'Workspaces',
    home_aria: 'AcousticsLab home',
    menu_fallback: 'Menu',
    primary_nav_aria: 'Primary navigation'
  },
  dashboard: {
    limited_support_title: 'Limited browser support',
    visualization_panel: {
      heading: 'Visualization',
      // Split into dot-separated segments so codec/channels can drop on a narrow header while
      // sample rate and window always stay.
      audio_sample_rate: '48 kHz',
      audio_channels: 'mono',
      audio_codec: 'opus',
      audio_window: '3 s window'
    },
    inference_panel: {
      heading: 'Inference'
    },
    configuration_panel: {
      heading: 'Configuration'
    },
    configuration_controls: {
      daemon_unavailable_title: 'Device unavailable',
      daemon_unavailable_default:
        'Configuration will resume automatically when the device is reachable.',
      microphone_heading: 'Microphone',
      source_label: 'Source',
      auto_first_available: 'auto · first available',
      channel_label: 'Channel',
      auto_channel: 'auto',
      inference_cadence_heading: 'Inference cadence',
      overlap_ratio_label: 'Overlap ratio',
      top_k_label: 'Top-K',
      loading: 'loading…',
      kind_alsa: 'ALSA',
      kind_unknown: 'unknown',
      approx_hz: (hz) => `${hz} Hz`,
      khz: (khz) => `${khz} kHz`,
      hz: (rate) => `${rate} Hz`
    },
    top_k_meter: {
      awaiting_first_frame: 'awaiting first inference frame…'
    },
    active_head_card: {
      heading: 'Active model',
      pill_default: 'Default',
      pill_workspace: 'Workspace',
      pill_detached: 'Detached',
      pill_default_title: 'The built-in default model is running.',
      pill_workspace_title: 'A trained workspace model is running.',
      pill_detached_title: 'The source workspace was deleted after this model was activated.',
      loading_active: 'loading active model…',
      activated_label: 'activated',
      class_count_label: (count) => (count === 1 ? 'class' : 'classes'),
      workspace_dt: 'workspace',
      revision_dt: 'revision',
      rev_value: (rev) => `rev ${rev}`,
      deleted_tag: '(deleted)',
      loading: 'loading…',
      ws_title_orphaned_with_name: (name, uuid) => `${name} · ${uuid} (workspace deleted)`,
      ws_title_orphaned: (uuid) => `${uuid} (workspace deleted)`,
      ws_title_with_name: (name, uuid) => `${name} · ${uuid}`
    }
  },
  theme: {
    label: 'Theme',
    label_with_current: (currentLabel) => `Theme: ${currentLabel}`,
    options: {
      auto: 'Auto',
      light: 'Light',
      dark: 'Dark'
    }
  },
  locale: {
    label: 'Language',
    label_with_current: (currentChip) => `Language: ${currentChip}`,
    auto_label: 'Auto'
  },
  health: {
    aria_label: 'System health',
    levels: {
      unknown: 'connecting',
      ok: 'healthy',
      degraded: 'degraded',
      unhealthy: 'unhealthy',
      unreachable: 'unreachable'
    },
    popover: {
      daemon_unreachable_title: 'Device unreachable',
      waiting_first_snapshot: 'waiting for first status snapshot…',
      subsystems_heading: 'Subsystems',
      seconds_ago: (seconds) => `${seconds}s ago`,
      stat_cpu_label: 'cpu',
      stat_rss_label: 'rss',
      stat_disk_free_label: 'disk free',
      uptime_label: 'uptime',
      dropped_count: (count) => `dropped: ${count}`
    }
  },
  common: {
    cancel: 'Cancel',
    dismiss: 'Dismiss'
  },
  error: {
    another_train_running: 'Another training job is already running on this device.',
    another_convert_running: 'Another conversion job is already running on this device.',
    job_conflict: 'Another operation is already in progress on this resource.',
    event_gap: 'The event stream skipped ahead and needs to catch up from logs. Reconnecting…',
    too_early: 'The device is still applying your previous change. Retrying…',
    unavailable: 'The device is temporarily unavailable. Please retry in a moment.',
    internal:
      'The daemon hit an internal error. Please retry. If it persists, check the daemon logs.',
    unknown: 'Something went wrong. Please retry.',
    something_went_wrong: 'Something went wrong.',
    request_failed: (code) => `Request failed (${code}).`
  },
  validation: {
    name: {
      empty: 'Name cannot be empty.',
      max_bytes: (max) => `Name must be ${max} bytes or fewer.`,
      slashes_or_nul: 'Name cannot contain slashes or NUL bytes.',
      starts_or_ends_whitespace: 'Name cannot start or end with whitespace.',
      control_chars: 'Name cannot contain control characters.',
      starts_with_dot: 'Category name cannot start with a dot.',
      starts_with_underscore:
        'Category name cannot start with an underscore (reserved for built-in classes).',
      starts_with_hyphen:
        'Category name cannot start with a hyphen (guards against unquoted shell expansion).',
      bad_chars: 'Only letters, digits, dots, hyphens, and underscores are allowed.',
      category_max_bytes: (max) => `Category name must be ${max} bytes or fewer.`,
      category_empty: 'Category name cannot be empty.'
    },
    cfg: {
      epochs_whole: 'Epochs must be a whole number.',
      epochs_range: (min, max) => `Epochs must be between ${min} and ${max}.`,
      batch_whole: 'Batch size must be a whole number.',
      batch_range: (min, max) => `Batch size must be between ${min} and ${max}.`,
      lr_finite: 'Learning rate must be a finite number.',
      lr_greater_than_zero: 'Learning rate must be greater than 0.',
      lr_max: (max) => `Learning rate must be at most ${max}.`,
      seed_whole: 'Seed must be a whole number.',
      seed_non_negative: 'Seed must be 0 or greater.',
      seed_too_large: 'Seed is too large.',
      split_finite: 'Validation split must be a finite number.',
      split_min: 'Validation split must be 0 or greater.',
      split_max: (max) => `Validation split must be at most ${max}.`
    }
  },
  streams: {
    socket_status: {
      connecting: 'connecting',
      open: 'live',
      closed: 'disconnected',
      error: 'error'
    }
  },
  recorder: {
    mic_error_denied:
      'Microphone access was denied. Allow microphone access in the browser settings and try again.',
    mic_error_not_found: 'No microphone was found. Connect one and try again.',
    mic_error_in_use: 'The microphone is in use by another application. Close it and try again.',
    mic_error_interrupted: 'Microphone capture was interrupted. Try again.',
    mic_error_generic: 'Could not start the microphone. Try again.'
  },
  category: {
    list: {
      heading: 'Dataset',
      description:
        'Each category becomes a class label the trainer learns — Background Noise is required.',
      add_button: 'Add category',
      add_button_aria: 'Add category',
      loading: 'loading categories…',
      load_error: (error) => `Couldn't load categories. ${error}`,
      menu_delete: 'Delete',
      menu_hint_preserved: 'preserved',
      menu_rename: 'Rename',
      menu_rename_hint_busy: 'finish in-progress work first',
      menu_add: 'Add category'
    },
    add_dialog: {
      title: 'Add category',
      name_label: 'Name',
      name_placeholder: 'e.g. cat',
      name_help_prefix:
        'Letters, digits, dots, hyphens, and underscores. The name doubles as the on-disk directory name (e.g. ',
      name_help_code_example: 'datasets/cat/',
      name_help_suffix: ') and as the class label the trainer uses.',
      submit: 'Add',
      error_exact_duplicate: 'A category with this name already exists.',
      error_case_insensitive_duplicate: (existingName) =>
        `Conflicts with existing "${existingName}" (names are case-insensitive on most filesystems).`
    },
    rename_dialog: {
      title: 'Rename category',
      name_label: 'Name',
      name_help:
        'The name doubles as the on-disk directory and the trainer class label, so renaming changes the class label. Existing trained models keep their old labels and are marked stale until you re-train.',
      submit: 'Save',
      error_mandatory: 'Background Noise is preserved and cannot be renamed.',
      error_busy: 'Finish or clear in-progress uploads and deletions before renaming this category.'
    },
    delete_dialog: {
      title: 'Delete this category?',
      body_server: "Removes the dataset folder and every slice inside it. Can't be undone.",
      body_idb:
        'Removes this category from the local list. No slices were uploaded, so nothing on the device changes.',
      submit: 'Delete',
      error_fallback: 'Could not delete the category.',
      error_mandatory_required: 'Background Noise is preserved and cannot be deleted.',
      error_not_found: 'Category not found.'
    },
    slice_card: {
      aria_select: (filename) => `Select slice ${filename}`,
      aria_deselect: (filename) => `Deselect slice ${filename}`,
      aria_play: (filename) => `Play slice ${filename}`,
      title_failed: (errorOrUnknown) => `Upload failed: ${errorOrUnknown}. Right-click to retry.`,
      title_uploading: (progressPct) => `Uploading… ${progressPct}%`,
      title_local: 'Local — awaiting upload',
      title_multi_click_deselect: 'Click to deselect (Esc exits selection)',
      title_multi_click_select: 'Click to add to selection (Esc exits selection)',
      title_playing: 'Playing — click to restart',
      title_idle: 'Click to play (Ctrl/Cmd-click to select)',
      sr_deleting: (filename) => `Deleting slice ${filename}`,
      sr_uploading: (progressPct) => `Uploading ${progressPct}%`,
      retry_aria: (filename) => `Retry upload for slice ${filename}`,
      retry_title_with_error: (errorMessage) => `Upload failed: ${errorMessage}. Click to retry.`,
      retry_title_no_error: 'Upload failed. Click to retry.',
      retry_label: 'retry',
      select_title: 'Select',
      deselect_title: 'Deselect',
      delete_aria: (filename) => `Delete slice ${filename}`,
      delete_title: 'Delete slice',
      slice_select_aria: (filename) => `Select slice ${filename}`,
      slice_deselect_aria: (filename) => `Deselect slice ${filename}`,
      unknown_error: 'unknown error'
    },
    trim_waveform: {
      handles_aria: 'Trim handles, drag to set the start and end of the slice range',
      handle_start_aria: 'Trim start',
      handle_end_aria: 'Trim end',
      selection_aria: 'Slide selection window, drag to move both trim edges together',
      playback_position_aria: 'Playback position',
      value_seconds: (sec) => `${sec} seconds`,
      value_seconds_range: (startSec, endSec) => `${startSec} to ${endSec} seconds`
    },
    slice_pane: {
      heading: 'Slices',
      tips_label: 'Slice module tips',
      tip_audition_title: 'Audition every slice before training.',
      tip_audition_body:
        'A mislabeled row biases the whole class — click cards to play, discard liberally.',
      tip_diversity_title: 'Diversity beats quantity.',
      tip_diversity_body:
        '10 varied takes (distance, angle, background) train better than 30 near-identical copies.',
      quota_above_title: (threshold) => `Above the ${threshold}-slice minimum for training.`,
      quota_below_title: (threshold) =>
        `Below the ${threshold}-slice minimum for training. Slice more to satisfy the quota.`,
      loading: 'loading slices…',
      load_error: (error) => `Couldn't load slices. ${error}`,
      empty_state_prefix: 'No slices yet. Trim the clip in the Input pane and click ',
      empty_state_button: 'Slice',
      empty_state_suffix: ' to fill this grid.',
      select_all_label: 'Select all',
      deselect_all_label: 'Deselect all',
      select_all_title: 'Select all slices (Cmd/Ctrl+A)',
      deselect_all_title: 'Deselect all slices (Cmd/Ctrl+A)',
      done_label: 'Done',
      done_title: 'Exit selection (Esc)',
      delete_title: 'Delete the selected slices (Del / Backspace)',
      delete_disabled_title: 'Select at least one slice to delete',
      delete_inflight_title: (count) => `Deleting ${count} ${count === 1 ? 'slice' : 'slices'}…`,
      delete_inflight_aria: (count) => `Deleting ${count} ${count === 1 ? 'slice' : 'slices'}`,
      delete_aria_count: (count) => `Delete ${count} selected ${count === 1 ? 'slice' : 'slices'}`,
      delete_aria_fallback: 'Delete selected slices',
      delete_label_inflight: (count) => `Deleting ${count}…`,
      delete_label_count: (count) => `Delete ${count}`,
      delete_label_bare: 'Delete',
      menu_play: 'Play',
      menu_stop: 'Stop',
      menu_retry_upload: 'Retry upload',
      menu_select: 'Select',
      menu_deselect: 'Deselect',
      menu_select_all: 'Select all',
      menu_deselect_all: 'Deselect all',
      menu_done_exit: 'Done (exit selection)',
      menu_retry_failed_in_selection: 'Retry failed in selection',
      menu_delete_batch: (count) => `Delete ${count} ${count === 1 ? 'slice' : 'slices'}`,
      menu_delete: 'Delete',
      menu_hint_a: 'Cmd/Ctrl+A',
      menu_hint_esc: 'Esc',
      menu_hint_ctrl_click: 'Ctrl/Cmd-click',
      menu_hint_del_backspace: 'Del / Backspace'
    },
    input_pane: {
      heading: 'Input',
      tips_label: 'Input module tips',
      tip_stream_title: "Prefer the device's sound stream.",
      tip_stream_body:
        "Your slices share the same DSP as inference, so the trained model doesn't see a distribution shift after fine-tune.",
      tip_environment_title: 'Record in the deployment environment.',
      tip_environment_body:
        'A clean studio capture undertrains noise rejection. The real background is half of what the model needs to learn.',
      tip_meter_title: 'Stay green-to-amber on the meter.',
      tip_meter_body: "Rose means clipping, which erases information the trainer can't recover.",
      pane_aria: (categoryDisplay) => `Input module for category ${categoryDisplay}`,
      source_aria: 'Input source',
      loudness_aria: 'Loudness meter',
      source_microphone_group: 'Microphone',
      source_system_default_mic: 'System default microphone',
      source_remembered: (label) => `${label} (remembered)`,
      source_mic_fallback: (n, idFrag) => `Microphone ${n} (${idFrag})`,
      source_mic_remembered_fallback: (idFrag) => `Microphone (${idFrag})`,
      source_mic_default_id: 'default',
      source_live_stream_group: 'Live stream',
      source_daemon_stream: 'Device sound stream',
      source_daemon_stream_with_status: (status) => `Device sound stream · ${status}`,
      drop_zone_title: (cap) => `Drop a WAV file here (up to ${cap}), or click to browse`,
      drop_zone_idle: 'Drag & drop a WAV here',
      drop_zone_browse: 'Browse files',
      record_aria_stream: 'Start capturing from the live sound stream',
      record_aria_mic: 'Start recording from microphone',
      record_label: 'Record',
      record_title_stream_open: (max) => `Capture the live sound stream (auto-stops at ${max}).`,
      record_title_stream_connecting:
        'Device sound stream is connecting. Recording will be available once it opens.',
      record_title_stream_closed:
        'Device sound stream is unreachable. Check the device is running.',
      record_title_stream_unsupported:
        "This browser can't decode the live sound stream here — it needs WebCodecs over a secure (HTTPS) context. Open this page through the secure gateway, or drop or browse for a WAV file instead.",
      capture_stop_aria_stream: 'Stop stream capture',
      capture_stop_aria_mic: 'Stop recording',
      capture_stop_label: 'Stop',
      capture_discard_label: 'Discard',
      capture_encoding: 'Encoding…',
      capture_decoding: 'Decoding…',
      trim_selection_prefix: 'Selection:',
      trim_drag_hint: 'Drag the handles to ≥ 1 s to enable slicing.',
      trim_projected_slices: (count) => `${count} ${count === 1 ? 'slice' : 'slices'} of 1 s each`,
      trim_unused_label: 'unused',
      slice_aria_enabled: (count) => `Slice into ${count} ${count === 1 ? 'slice' : 'slices'}`,
      slice_aria_disabled: 'Slice (selection must be at least 1 second)',
      slice_title_enabled: (count) =>
        `Append ${count} slice${count === 1 ? '' : 's'} to the right pane`,
      slice_title_disabled: 'Selection must be ≥ 1 s to slice',
      slice_label_bare: 'Slice',
      slice_label_count: (count) => `Slice · ${count}`,
      discard_aria: 'Discard clip',
      discard_title: 'Discard clip',
      discard_label: 'Discard',
      play_stop_aria: 'Stop playback',
      play_stop_title: 'Stop playback',
      play_aria: 'Play the trimmed selection',
      play_title: 'Play the trimmed selection',
      export_aria: 'Download as WAV',
      export_title: 'Download as WAV',
      error_file_too_large: (size, cap) =>
        `File is ${size} — the import cap is ${cap}. Trim it shorter and re-export, then drop again.`,
      error_clip_too_short: (clipSecs) =>
        `Clip is only ${clipSecs} s, training needs at least 1 s per clip, so a shorter clip is excluded entirely. Import or record a clip of 1 s or longer.`,
      error_only_one_file:
        'Only one file at a time — the Input slot holds the most recent clip only. Drop a single WAV.',
      error_only_wav: 'Only WAV files are supported.',
      error_could_not_import: 'Could not import the file.',
      error_could_not_discard: 'Could not discard the clip.',
      error_could_not_decode_draft: 'Could not decode the stored draft.',
      error_could_not_save_recording: 'Could not save the recording.',
      error_could_not_capture_stream: 'Could not capture the stream.',
      error_could_not_slice: 'Could not slice the clip.',
      error_wav_too_small_for_header:
        'File is too small to be a WAV (need at least 12 bytes for the header).',
      error_wav_missing_riff: 'Not a WAV file (missing RIFF magic).',
      error_wav_missing_wave: 'Not a WAV file (missing WAVE marker).',
      error_wav_empty: 'File is empty or too small to be a WAV.',
      error_wav_buffer_too_small:
        'WAV buffer too small (need at least 44 bytes for the canonical header).',
      error_web_audio_unavailable: 'Web Audio API is unavailable in this browser.',
      auto_stopped_at_cap: 'Auto-stopped at the duration cap.',
      silent_dropped_suffix: (count) =>
        `${count} silent ${count === 1 ? 'slice' : 'slices'} skipped`
    },
    row: {
      badge_synced: 'Synced',
      badge_uploading: 'Uploading',
      badge_pending: 'Pending',
      badge_failed: 'Failed',
      badge_not_enough: 'Not enough samples',
      badge_not_enough_with_state: (statusLabel) => `Not enough samples · ${statusLabel}`,
      title_synced: (tally) => `${tally} slices uploaded to the device — training-ready.`,
      title_uploading: (tally) => `${tally} slices, some are still uploading to the device.`,
      title_pending: (tally) => `${tally} slices ready but not yet uploaded to the device.`,
      title_failed: (tally) =>
        `${tally} slices, at least one upload failed. Retry from the slice card or discard the failed rows.`,
      title_not_enough_empty: (missing, tally) =>
        `Add ${missing} more slices to satisfy the per-category quota (${tally}).`,
      title_not_enough_synced: (tally, missing) =>
        `${tally} slices uploaded, add ${missing} more to satisfy the per-category quota.`,
      title_not_enough_uploading: (tally, missing) =>
        `${tally} slices, some are still uploading. Need ${missing} more once they finish.`,
      title_not_enough_pending: (tally, missing) =>
        `${tally} slices queued locally, need ${missing} more.`,
      actions_aria: (displayName) => `Actions for ${displayName}`,
      actions_title: 'Category actions',
      actions_title_preserved: 'Preserved — rename and delete disabled',
      badge_deleting: 'Deleting'
    }
  },
  training: {
    pane: {
      heading: 'Train',
      subtitle_other_running: 'Another workspace is training, only one job runs at a time.',
      subtitle_default:
        "Tune a model on this workspace's dataset, old model automatically discarded when new one lands.",
      readiness_loading: 'Loading dataset…',
      readiness_no_categories: 'Add a foreground class with uploaded slices to start training.',
      readiness_background_short: (need) =>
        `Background Noise needs ${need} more uploaded slice${need === 1 ? '' : 's'} to start training.`,
      readiness_foreground_short:
        'At least one foreground class needs 10 uploaded slices to start training.',
      button_starting: 'Starting…',
      button_cancel: 'Cancel',
      button_cancelling: 'Cancelling…',
      button_retrain: 'Re-train',
      button_train: 'Train model',
      button_title_loading: 'Loading dataset…',
      button_title_not_ready_default: 'Readiness reason',
      button_title_form_errors: 'Fix the highlighted hyperparameter fields to enable training.',
      button_title_idle_trained:
        'A model already matches this revision — re-train to try different hyperparameters or a different random seed. Activate any model from the Models section below.',
      button_title_idle_busy: 'Another workspace is training, only one job runs at a time.',
      button_title_idle_ready: 'Train a model on this workspace dataset.',
      button_title_starting: 'Submitting the training request…',
      button_title_running: 'Cancel the running training job.',
      button_title_cancelling: 'Cancelling…',
      summary_chip_epochs: (epochs) => `${epochs} epochs`,
      summary_chip_no_holdout: 'no holdout',
      summary_chip_val: (pctLabel) => `val ${pctLabel}`,
      hyperparameters_disclosure_label: 'Hyperparameters',
      start_error_title: 'Could not start training'
    },
    form: {
      epochs_label: 'Epochs',
      batch_size_label: 'Batch size',
      learning_rate_label: 'Learning rate',
      validation_split_label: 'Validation split',
      validation_split_hint: '· 0 to disable',
      seed_label: 'Seed',
      seed_hint: '· blank for daemon-picked entropy',
      seed_placeholder: '(optional)'
    },
    progress: {
      submitting: 'Submitting…',
      job_short_id: (shortId) => `job ${shortId}…`,
      train_loss_label: 'train loss',
      train_acc_label: 'train acc',
      val_acc_label: 'val acc',
      val_acc_disabled_label: 'val acc · disabled',
      em_dash: ' — '
    },
    logs: {
      heading: 'Logs',
      entry_count: (count) => `${count} ${count === 1 ? 'entry' : 'entries'}`,
      waiting_first_message: 'Waiting for the first message…'
    },
    chart: {
      waiting_first_epoch: 'Waiting for first epoch…',
      legend_loss: 'loss',
      legend_train: 'train',
      legend_val: 'val',
      tooltip_epoch: 'epoch',
      tooltip_loss: 'loss',
      tooltip_train: 'train',
      tooltip_val: 'val',
      chart_aria: 'Training metrics chart'
    },
    history: {
      heading: 'History',
      keeps_last: (cap) => `keeps last ${cap} runs`,
      retention_title: (cap) =>
        `The daemon keeps the ${cap} most recent training-log files per workspace; older JSONL traces are pruned when a new run opens. The published model record (in the Models section below) is unaffected — only the JSONL trace is pruned.`,
      empty_state_prefix: 'No training runs yet for this workspace. Click ',
      empty_state_button: 'Train model',
      empty_state_suffix: ' to start one.',
      hide_older_label: 'Hide older runs',
      show_older_label: (count) => `Show ${count} older ${count === 1 ? 'run' : 'runs'}`,
      hide_older_title: 'Collapse the older runs section back to the recent two.',
      show_older_title: 'Reveal older training runs for this workspace, paged in batches of 5.',
      load_more_label: (count) => `Load ${count} more`,
      load_more_title: 'Fetch the next batch of older training runs from the device.',
      menu_delete: 'Delete',
      menu_deleting: 'Deleting…',
      menu_hint_train_active: 'train active',
      menu_hint_live: 'live',
      delete_error_title: 'Could not delete training log'
    },
    history_item: {
      time_started_pre_ack: 'started',
      time_started: (relative) => `started ${relative}`,
      time_finished: (relative) => relative,
      time_title_started: (absolute) => `started ${absolute}`,
      time_title_finished: (absolute) => `finished ${absolute}`,
      detail_epoch: (current, total) => `epoch ${current}/${total}`,
      detail_class_count: (count) => `${count} ${count === 1 ? 'class' : 'classes'}`,
      detail_val_acc: (pctLabel) => `val ${pctLabel}`,
      detail_train_acc: (pctLabel) => `train ${pctLabel}`,
      detail_stopped_at: (stageLabel) => `stopped at ${stageLabel}`
    },
    summary: {
      completed_aria: 'Completed run summary',
      failed_aria: 'Failed run summary',
      cancelled_aria: 'Cancelled run summary',
      duration_label: 'Duration',
      epochs_label: 'Epochs',
      best_val_at: (epoch) => `Best val @ ${epoch}`,
      final_train_acc_label: 'Final train acc',
      classes_label: 'Classes',
      stopped_at_label: 'Stopped at',
      cancelled_at_label: 'Cancelled at',
      epochs_tooltip_full: 'Ran the full configured epoch count.',
      epochs_tooltip_partial: 'Observed epochs vs. configured epoch count.',
      after_epochs: (run, total) => `after ${run}/${total} epochs`,
      failed_no_diagnostic: 'No diagnostic surfaced. Check daemon logs for details.',
      cancelled_default_reason: 'Stopped at the next training checkpoint.',
      failed_default: 'Training failed.'
    },
    stage: {
      prepare: 'Preparing',
      dataset_scan: 'Scanning dataset',
      feature_extract: 'Extracting features',
      train: 'Training',
      save: 'Saving',
      publish: 'Publishing'
    },
    state: {
      running: 'running',
      completed: 'completed',
      failed: 'failed',
      cancelled: 'cancelled'
    },
    state_submitting: 'submitting',
    store_log: {
      seed_submitted: 'Submitted, waiting for the device to start emitting events…',
      seed_recovered: 'Recovered an in-flight training job from the device.',
      job_submitted: (backbone) => `Job submitted · backbone ${backbone}`,
      job_running: 'Job running',
      phase_prefix: (stageLabel) => `Phase: ${stageLabel}`,
      job_failed: (stageLabel, error) => `Job failed at ${stageLabel} · ${error}`,
      job_cancelled: (stageLabel) => `Job cancelled at ${stageLabel}`,
      job_cancelled_shutdown: (stageLabel) => `Job cancelled at ${stageLabel} (daemon shutdown)`,
      scanned_dataset: (nClasses, nExamples) =>
        `Scanned dataset · ${nClasses} ${nClasses === 1 ? 'class' : 'classes'} · ${nExamples} examples`,
      features_extracted: (kept, dropped, elapsedSec) => {
        const droppedSuffix = dropped > 0 ? ` · dropped ${dropped}` : '';
        return `Features extracted · kept ${kept}${droppedSuffix} · ${elapsedSec}s`;
      },
      train_split: (trainN, valN) => `Train split · ${trainN} train · ${valN} val`,
      epoch_completed: (epoch, epochs, lossLabel, trainAccLabel, valAccLabel) => {
        const valPart = valAccLabel !== null ? ` · val ${valAccLabel}` : '';
        return `Epoch ${epoch}/${epochs} · loss ${lossLabel} · train ${trainAccLabel}${valPart}`;
      },
      train_loop_done: (epochsRun, elapsedSec, bestValAccLabel, bestEpoch) => {
        const bestPart =
          bestValAccLabel !== null && bestEpoch !== null
            ? ` · best val ${bestValAccLabel} @ epoch ${bestEpoch}`
            : '';
        return `Training loop done · ${epochsRun} ${epochsRun === 1 ? 'epoch' : 'epochs'} in ${elapsedSec}s${bestPart}`;
      },
      head_published: (headId, size, nClasses, rev) =>
        `Model published · ${headId} · ${size} · ${nClasses} ${nClasses === 1 ? 'class' : 'classes'} · rev ${rev}`,
      job_completed: (labelsList) =>
        labelsList.length > 0 ? `Job completed · ${labelsList}` : 'Job completed'
    }
  },
  deploy: {
    pane: {
      heading: 'Deploy',
      description:
        'Select a trained model and hot-swap it into live inference seamlessly with zero downtime.',
      pill_deployed: 'Deployed',
      pill_deployed_title: 'A model trained in this workspace is the runtime model.',
      pill_default: 'Default',
      pill_default_title: 'The built-in default model is running.',
      pill_standby: 'Standby',
      pill_standby_title:
        'A model from a different workspace is the runtime model. This workspace is on standby. Deploying one here will replace it.',
      pill_detached: 'Detached',
      pill_detached_title:
        'The workspace that produced the runtime model was deleted. The model is still running.',
      config_disclosure_label: 'Input & Inference config',
      config_chip_freq: (hzLabel) => `freq ${hzLabel} Hz`,
      config_chip_top_k: (topK) => `top-k ${topK}`
    },
    heads_table: {
      heading: 'Models',
      count_label: (count) => `${count} ${count === 1 ? 'model' : 'models'}`,
      // Rotation-cap suffix split off the bare count so it can collapse on a narrow card; carries
      // its own leading comma so it vanishes cleanly when hidden.
      count_retained: (retainedCap) => `, latest ${retainedCap} retained`,
      revert_to_default: 'Revert to default',
      revert_to_id: (shortId) => `Revert to ${shortId}`,
      revert_title: 'Re-deploy the previously running model',
      default_row_headline: 'Default',
      default_row_description: 'Built-in fallback, always available.',
      default_active_title: 'The built-in default model is currently deployed.',
      default_aria_active: 'Default model is active',
      default_aria_deploy: 'Deploy default model',
      default_title_active: 'The default model is already deployed',
      default_title_deploying: 'Deploying…',
      default_title_busy: 'Another model on this list is busy',
      default_title_idle: 'Revert to the built-in default model',
      menu_deploy: 'Deploy',
      menu_export: 'Export as .alpkg',
      menu_exporting: 'Exporting…',
      menu_delete: 'Delete',
      menu_hint_active: 'active',
      menu_hint_deployed: 'deployed',
      error_deploy_head: 'Could not deploy model',
      error_export_head: 'Could not export model',
      error_deploy_default: 'Could not deploy default model'
    },
    head_row: {
      pill_latest: 'Latest',
      pill_latest_title: "Most recent model trained on the workspace's current revision.",
      pill_active: 'Active',
      pill_active_title: 'This model is currently deployed in the inference pipeline.',
      // Fixed-width single-string meta for the model-card popover and delete-confirm card (never degrades).
      meta_line: (size, classCount, rev, relative) =>
        `${size} · ${classCount} ${classCount === 1 ? 'class' : 'classes'} · rev ${rev} · ${relative}`,
      // Row meta renders segment-by-segment (size · classes · rev · age) so size/rev can drop as
      // the row narrows; only these two need a string (size/age come from formatBytes/formatRelative).
      meta_classes: (classCount) => `${classCount} ${classCount === 1 ? 'class' : 'classes'}`,
      meta_rev: (rev) => `rev ${rev}`,
      row_aria_deployed: (shortId) => `Deployed model ${shortId}`,
      row_aria_deploy: (shortId) => `Deploy model ${shortId}`,
      row_title_deployed: 'This model is already deployed',
      row_title_deploying: 'Deploying…',
      row_title_exporting: 'Exporting…',
      row_title_busy: 'Another model on this list is busy',
      row_title_idle: 'Click to hot-swap this model into the inference pipeline',
      export_title_exporting: 'Exporting…',
      export_title_idle: 'Export this model as a .alpkg archive',
      export_aria_exporting: (shortId) => `Exporting model ${shortId}`,
      export_aria_idle: (shortId) => `Export model ${shortId}`,
      info_title: 'View the model card',
      info_aria: (shortId) => `View model card for ${shortId}`
    },
    inference_preview: {
      heading: 'Preview',
      off_title: 'Preview is off',
      off_description:
        "Start the preview to watch the deployed model's spectrogram and top-k stream.",
      start_button: 'Start preview'
    },
    info_dialog: {
      title_with_id: (shortId) => `Model card · ${shortId}`,
      loading: 'Loading classes…',
      error_title: 'Could not load classes',
      retry: 'Retry',
      classes_heading: 'Classes',
      class_labels_aria: 'Trained class labels'
    },
    delete_dialog: {
      title: 'Delete this model?',
      body: "Removes the trained model bytes and its manifest. The dataset and any other models stay. Can't be undone.",
      submit: 'Delete'
    }
  },
  workspace: {
    list: {
      title: 'Workspaces',
      at_cap_subtitle: (max) =>
        `Reached the ${max} workspace limit. Delete one before creating another.`,
      default_subtitle: 'Each workspace holds a labeled dataset and any models trained from it.',
      daemon_unavailable_title: 'Device unavailable',
      loading: 'loading workspaces…',
      empty_title: 'No workspaces yet',
      empty_description:
        'Workspaces are where recordings, labeled samples, and trained models live. Create one to get started.',
      selected_count_aria: (count) => `${count} selected`,
      new_button_label: 'New workspace',
      new_button_aria: 'New workspace',
      new_at_cap_label: (count, max) => `At cap · ${count}/${max}`,
      new_at_cap_title: 'Limit reached. Delete one workspace first.',
      import_button_label: 'Import',
      import_button_aria: 'Import workspace',
      import_button_title: 'Import workspace from an .alpkg or TFJS bundle',
      select_button_label: 'Select',
      done_button_label: 'Done',
      select_all_label: 'Select all',
      deselect_all_label: 'Deselect all',
      bulk_delete_label_count: (count) => `Delete ${count}`,
      bulk_delete_label_bare: 'Delete',
      bulk_delete_aria_count: (count) => `Delete ${count} workspace${count === 1 ? '' : 's'}`,
      bulk_delete_aria_fallback: 'Delete selected workspaces',
      menu_open: 'Open',
      menu_rename: 'Rename',
      menu_export: 'Export',
      menu_delete: 'Delete',
      menu_select_one: 'Select',
      menu_deselect_one: 'Deselect',
      menu_select_all: 'Select all',
      menu_deselect_all: 'Deselect all',
      menu_select_workspaces: 'Select workspaces',
      menu_done_exit: 'Done (exit selection)',
      menu_new: 'New workspace',
      menu_new_at_cap: (max) => `New workspace (at ${max} cap)`,
      menu_import: 'Import workspace'
    },
    detail: {
      back_link: '← Workspaces',
      loading: 'loading workspace…',
      not_found_title: 'Workspace not found',
      not_found_description:
        "It may have been deleted in another tab or via the device directly. Head back to the list to see what's still around.",
      back_to_list_button: 'Back to workspaces',
      load_error_title: "Couldn't load this workspace",
      created_label: (relative) => `created ${relative}`,
      rev_label: (rev) => `rev ${rev}`,
      modified_label: (relative) => `modified ${relative}`,
      live_pill_title: 'Advanced by a recent upload. Reload to refresh the modified timestamp.',
      live_pill: 'live',
      menu_rename: 'Rename',
      menu_export: 'Export',
      menu_import: 'Import',
      menu_delete: 'Delete',
      menu_back_to_list: 'Back to workspaces'
    },
    create_dialog: {
      title: 'New workspace',
      name_label: 'Name',
      name_placeholder: 'my-workspace',
      name_help:
        'Up to 128 characters. No slashes or control characters. The name is the only visible identifier, so pick something memorable.',
      submit: 'Create'
    },
    rename_dialog: {
      title: 'Rename workspace',
      name_label: 'Name',
      name_help:
        'Up to 128 characters. No slashes or control characters. Renaming does not advance the workspace revision — categories, slices, and models stay as they are.',
      submit: 'Save'
    },
    delete_dialog: {
      title: 'Delete this workspace?',
      body: "Removes the dataset, any trained models, and logs. Can't be undone.",
      submit: 'Delete'
    },
    bulk_delete_dialog: {
      title_count: (count) => `Delete ${count} workspace${count === 1 ? '' : 's'}?`,
      body: "Removes each workspace's dataset, trained models, and logs. Can't be undone.",
      submit_count: (count) => `Delete ${count}`
    },
    tool_island: {
      aria_label: 'Workspace actions',
      rename_aria: 'Rename workspace',
      rename_title: 'Rename workspace',
      export_aria: 'Export workspace',
      export_title: 'Export workspace (datasets + models)',
      import_aria: 'Import workspace',
      import_title: 'Import workspace (datasets + models)'
    },
    card: {
      created_label: (relative) => `created ${relative}`,
      select_aria: (name) => `Select workspace ${name}`,
      rename_aria: (name) => `Rename workspace ${name}`,
      deleting: 'deleting'
    },
    import_dialog: {
      title_into: (workspaceName) => `Import into · ${workspaceName}`,
      title_fallback: 'Import',
      step_indicator: (current, total) => `Step ${current} of ${total}`,
      pipeline_error_title: 'Import failed',
      error_invalid_state: 'Inconsistent dialog state — no archive to import.',
      pick_file: {
        drop_zone_title_attr: 'Drop an .alpkg archive or a TFJS bundle here, or click to browse',
        reading: 'Reading…',
        drop_zone_tfjs_staging: 'Drop more files to complete the TFJS bundle',
        drop_zone_idle: 'Drag & drop an .alpkg archive or a TFJS bundle here',
        browse_button: 'Browse files',
        error_empty_drop: 'Drop an .alpkg archive or a TFJS bundle.',
        error_multi_alpkg: (count) => `Pick one .alpkg archive at a time — you picked ${count}.`,
        error_mixed_archive:
          'An .alpkg archive must be picked on its own, not mixed with other files.',
        error_file_count_cap: (max, picked) =>
          `Drop or pick at most ${max} files at once — you picked ${picked}.`,
        error_single_too_large: (name, size, cap) =>
          `"${name}" is ${size} — the per-file cap is ${cap}.`,
        error_total_too_large: (total, cap) =>
          `Selection totals ${total} — the per-drop cap is ${cap}.`,
        error_tfjs_merged_file_count: (mergedCount, cap) =>
          `Staged set would total ${mergedCount} files — the cap is ${cap}. Clear and re-drop a smaller bundle.`,
        error_tfjs_merged_bytes: (mergedBytes, cap) =>
          `Staged set would total ${mergedBytes} — the cap is ${cap}. Clear and re-drop a smaller bundle.`,
        staged_files_heading: 'Staged files',
        staged_files_count: (count) => `${count} ${count === 1 ? 'file' : 'files'}`,
        clear_button: 'Clear',
        error_could_not_read_archive: 'Could not read the archive.',
        error_could_not_read_file: 'Could not read the file.',
        error_could_not_read_picked_files: 'Could not read the picked files.',
        error_could_not_read_model_json: 'Could not read model.json.',
        tfjs_diag_empty_drop: 'Drop the TFJS bundle files (model.json + shards + labels).',
        tfjs_diag_no_model_json: 'No "model.json" in the drop. Include the TFJS manifest.',
        tfjs_diag_ambiguous_model_json: (count) =>
          `Ambiguous bundle: ${count} files named "model.json".`,
        tfjs_diag_multiple_labels_txt:
          'Multiple "labels.txt" files in the drop. Include exactly one.',
        tfjs_diag_multiple_metadata_json:
          'Multiple "metadata.json" files in the drop. Include exactly one.',
        tfjs_diag_both_labels:
          'Both "labels.txt" and "metadata.json" provided. Include only one labels source.',
        tfjs_diag_no_labels: 'No labels file provided. Include "labels.txt" or "metadata.json".',
        tfjs_diag_shard_collision_one: (quotedName) =>
          `Two staged files share the shard name ${quotedName}. Clear staging and drop only the intended copy.`,
        tfjs_diag_shard_collision_many: (quotedNames, overflow) =>
          `Multiple staged files share shard names referenced by "model.json": ${quotedNames}${overflow ? '…' : ''}. Clear staging and drop only the intended copies.`,
        tfjs_diag_missing_shard_one: (quotedName) =>
          `Missing shard ${quotedName} referenced by "model.json".`,
        tfjs_diag_missing_shards_many: (count, quotedNames, overflow) =>
          `Missing ${count} shards referenced by "model.json": ${quotedNames}${overflow ? '…' : ''}.`,
        tfjs_diag_model_json_not_json: 'model.json is not valid JSON.',
        tfjs_diag_model_json_not_object: 'model.json is not a JSON object.',
        tfjs_diag_model_json_no_manifest: 'model.json is missing the "weightsManifest" array.',
        tfjs_diag_model_json_no_shards: 'model.json declares no shard files.'
      },
      pick_target: {
        section_label: 'Import into',
        mode_radio_aria: 'Target workspace mode',
        mode_use_existing: 'Use existing',
        mode_create_new: 'Create new',
        no_workspaces_prefix: 'No workspaces yet — switch to ',
        no_workspaces_link_label: 'Create new',
        no_workspaces_suffix: ' to make one.',
        workspace_list_aria: 'Pick a target workspace',
        workspace_created_label: (relative) => `created ${relative}`,
        create_name_placeholder: 'my-imported-workspace',
        create_will_carry_tags: (tagsCsv) => `Will carry tags from the source: ${tagsCsv}`,
        alpkg_source_card_title: (name, id) => `${name} (${id})`,
        alpkg_source_created_label: (relative) => `created ${relative}`,
        alpkg_source_rev_label: (rev) => `rev ${rev}`,
        alpkg_source_modified_label: (relative) => `modified ${relative}`,
        tfjs_bundle_card_title: 'TFJS bundle',
        tfjs_show_labels_aria: 'Show class labels',
        tfjs_meta_strip: (size, shards, classes, labelsFileName) => {
          const classesPart =
            classes !== null && classes > 0
              ? ` · ${classes} ${classes === 1 ? 'class' : 'classes'}`
              : '';
          const shardsPart = ` · ${shards} ${shards === 1 ? 'shard' : 'shards'}`;
          const labelsPart = labelsFileName !== null ? ` · via ${labelsFileName}` : '';
          return `${size}${classesPart}${shardsPart}${labelsPart}`;
        }
      },
      summary: {
        datasets_heading: 'Datasets',
        datasets_counter: (selected, total) => `selected ${selected} / ${total}`,
        checking_categories: 'Checking target workspace for existing categories…',
        slice_count: (count) => `${count} ${count === 1 ? 'slice' : 'slices'}`,
        rename_button_aria: 'Rename target category',
        rename_button_title_default: 'Rename target category',
        mode_aria: (modeLabel) => `Import action: ${modeLabel}`,
        mode_menu_aria: (sourceName) => `Import action for ${sourceName}`,
        rename_popover_aria: (sourceName) => `Rename target category for ${sourceName}`,
        rename_popover_heading: 'Renaming',
        rename_chips_heading: 'Or reuse existing',
        heads_heading: 'Models',
        heads_cap_tooltip: (cap) =>
          `Up to ${cap} models per workspace. Older non-active models roll off when new ones land — from a re-train or an import.`,
        heads_counter: (selected, existingInTarget, cap, activeInTarget) => {
          const active = activeInTarget > 0 ? ` · active ${activeInTarget} pinned` : '';
          return `selected ${selected} · target ${existingInTarget} / ${cap}${active}`;
        },
        checking_heads: 'Checking target models…',
        displacement_warning: (displaced, cap) =>
          `Importing will displace the ${displaced} oldest non-active model${displaced === 1 ? '' : 's'} to fit the ${cap}-model cap.`,
        head_exists_badge_title: 'A model with this id already exists in the target workspace.',
        head_exists_badge: 'Exists',
        head_show_details_aria: 'Show model details',
        head_class_count: (count) => `${count} ${count === 1 ? 'class' : 'classes'}`,
        head_info_metadata: (size, classes, revisionId, createdAbsolute, createdRelative) => {
          const classesPart =
            classes !== null ? ` · ${classes} ${classes === 1 ? 'class' : 'classes'}` : '';
          const revPart = revisionId !== null ? ` · rev ${revisionId}` : '';
          const createdPart =
            createdAbsolute !== null && createdRelative !== null
              ? ` · ${createdAbsolute} (${createdRelative})`
              : '';
          return `${size}${classesPart}${revPart}${createdPart}`;
        },
        head_classes_heading: 'Classes',
        head_class_labels_aria: 'Trained class labels',
        archive_errors_summary: (count) =>
          `Skipped ${count} archive ${count === 1 ? 'entry' : 'entries'}`,
        tfjs_ignored_unknown: (count, fileList) =>
          `Ignored ${count} unrecognized file${count === 1 ? '' : 's'}: ${fileList}`,
        tfjs_classes_popover_heading: (count) => `Classes (${count})`,
        tfjs_classes_popover_aria: 'Class labels',
        head_disabled_reasons: {
          loading: 'Loading target models…',
          exists: 'Already exists in the target. Pick a different model.',
          ceiling: 'Selection limit reached. Untick another row first.'
        }
      },
      modes: {
        new: 'New',
        merge: 'Merge',
        replace: 'Replace',
        skip: 'Skip'
      },
      mode_tooltips: {
        new: "Create the category from scratch with the archive's slices.",
        merge:
          'Upload archive slices on top of the existing category. Same-sha256 slices overwrite themselves, new ones add to the set.',
        replace:
          'Delete the existing category (and every slice it holds), then upload from the archive.',
        skip: 'Do not import this category.'
      },
      mode_disabled_reasons: {
        new_exists:
          'A category with this target name already exists. Pick Merge to add slices or Replace to wipe and re-import.',
        merge_missing:
          'There is no existing category with this target name. Pick New to create one.',
        replace_missing:
          'There is no existing category with this target name. Pick New to create one.'
      },
      running: {
        progress_replacing_categories: (categoryDisplay, done, total) => {
          const cat = categoryDisplay !== null ? ` · ${categoryDisplay}` : '';
          return `Replacing categories${cat} · ${done} / ${total}`;
        },
        progress_uploading_datasets: (categoryDisplay, done, total) => {
          const cat = categoryDisplay !== null ? ` · ${categoryDisplay}` : '';
          if (typeof done === 'number' && typeof total === 'number') {
            return `Uploading slices${cat} · ${done} / ${total}`;
          }
          return `Uploading slices${cat}`;
        },
        progress_importing_heads: (index1, total, subPhase) => {
          const sub = subPhase !== null ? ` (${subPhase})` : '';
          return `Importing model ${index1} / ${total}${sub}`;
        },
        progress_uploading_tfjs: (done, total) => `Uploading TFJS files · ${done} / ${total}`,
        progress_converting_tfjs: 'Converting TFJS bundle…',
        ds_pending: 'Pending',
        ds_replacing: 'Replacing',
        ds_uploading_counter: (uploaded, total) => `${uploaded} / ${total}`,
        ds_done_uploaded: (uploaded) => `${uploaded} uploaded`,
        ds_failed_count: (failed) => `${failed} failed`,
        ds_failed_label: 'Failed',
        ds_failed_title_count: (failed) =>
          `${failed} slice${failed === 1 ? '' : 's'} failed to upload`,
        head_queued: 'Queued',
        head_skipped_badge_title:
          'The model id already exists on disk and the orchestrator skipped it (idempotent re-import).',
        head_per_log_not_started:
          "Not started yet — log lines will appear once this model's import begins.",
        head_per_log_no_events: 'No events recorded.',
        log_count: (count) => `${count} ${count === 1 ? 'log' : 'logs'}`
      },
      head_phase: {
        queued: 'Queued',
        uploading_files: 'Uploading files',
        starting_convert: 'Starting convert',
        converting: 'Converting',
        cleaning_up: 'Cleaning up',
        done: 'Done',
        failed: 'Failed'
      },
      head_outcome: {
        imported: 'Imported',
        replaced: 'Replaced',
        skipped: 'Skipped',
        failed: 'Failed'
      },
      convert_stage: {
        prepare: 'Preparing',
        read_manifest: 'Reading manifest',
        validate_manifest: 'Validating manifest',
        verify_mpk: 'Verifying MPK',
        stage_mpk: 'Staging MPK',
        read_model_json: 'Reading model.json',
        stage_shards: 'Staging shards',
        extract_weights: 'Extracting weights',
        read_labels: 'Reading labels',
        stage_head_mpk: 'Staging model MPK',
        publish_head: 'Publishing model'
      },
      convert_event: {
        job_submitted: (converter) => `Job submitted via ${converter}`,
        job_running: 'Job running',
        phase: (stageLabel) => `Phase: ${stageLabel}`,
        manifest_validated: (classes) => `Manifest validated · ${classes} classes`,
        mpk_verified: (size) => `MPK verified · ${size}`,
        weights_extracted: (classes, inDim) =>
          `Weights extracted · ${classes} classes · ${inDim} in_dim`,
        labels_loaded: (labels) => `Labels loaded · ${labels} labels`,
        head_published: (idempotentSkip) =>
          `Model published${idempotentSkip ? ' (already on disk, skipped)' : ''}`,
        job_completed: (classes) => `Job completed · ${classes} classes`,
        job_failed: (category, error) => `Job failed · ${category} · ${error}`
      },
      done: {
        conflict_detail: (storedSha8, incomingSha8) =>
          `Target already holds a model with this id but a different sha256 (${storedSha8} vs incoming ${incomingSha8}).`,
        retry_button: 'Replace existing & retry'
      },
      footer: {
        cancel: 'Cancel',
        back: 'Back',
        next: 'Next',
        import: 'Import',
        importing: 'Importing…',
        back_to_selection: 'Back to selection',
        done: 'Done'
      }
    },
    export_dialog: {
      title: (workspaceName) => `Export workspace · ${workspaceName}`,
      load_error_title: "Couldn't load this workspace",
      loading: 'Loading workspace…',
      nothing_to_export: 'This workspace has no categories and no models yet — nothing to export.',
      datasets_heading: 'Datasets',
      heads_heading: 'Models',
      select_all: 'Select all',
      deselect_all: 'Deselect all',
      row_empty: 'empty',
      row_slice_count: (count) => `${count} ${count === 1 ? 'slice' : 'slices'}`,
      head_meta_title: (size, classCount) =>
        `${size} · ${classCount} ${classCount === 1 ? 'class' : 'classes'}`,
      head_meta_classes: (count) => `${count} ${count === 1 ? 'class' : 'classes'}`,
      pending_warning:
        'Slices still uploading or pending in the selection will be excluded — only on-disk slices ship.',
      progress_preparing_workspace: 'Reading workspace metadata…',
      progress_fetching_slices: 'Fetching slices…',
      progress_listing_slices: 'Listing slices…',
      progress_fetched_slices: (done, total) => `Fetched ${done} / ${total} slices…`,
      progress_validating_heads: 'Validating models…',
      progress_validated_heads: (done, total) => `Validated ${done} / ${total} models…`,
      progress_packing: 'Packing archive…',
      progress_downloading: 'Starting download…',
      error_default: 'Export failed',
      error_in_category: (categoryDisplay) => `Export failed in "${categoryDisplay}"`,
      error_for_head: (shortId) => `Export failed for model ${shortId}`,
      exporting: 'Exporting…',
      export_aria: 'Export selected items',
      export_button: 'Export'
    }
  }
} satisfies Messages;
