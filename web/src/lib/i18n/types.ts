// Declare catalogs with `satisfies Messages` (not `as const`): missing keys / wrong arg shapes fail at compile time while function values keep their callable signatures.

export interface Messages {
  app: {
    /** Untranslated by convention, but routed through the catalog so all user-visible text shares one surface. */
    name: string;
    description: string;
  };

  routes: {
    dashboard_title: (brand: string) => string;
    workspace_list_title: (brand: string) => string;
    workspace_detail_title: (workspaceName: string, brand: string) => string;
  };

  nav: {
    dashboard: string;
    workspaces: string;
    home_aria: string;
    /** Mobile menu trigger fallback when the current route matches no tab (defensive; should never paint). */
    menu_fallback: string;
    primary_nav_aria: string;
  };

  /** `configuration_controls` is the form body shared between the dashboard Configuration panel and the workspace deploy disclosure. */
  dashboard: {
    limited_support_title: string;
    visualization_panel: {
      heading: string;
      audio_sample_rate: string;
      audio_channels: string;
      audio_codec: string;
      audio_window: string;
    };
    inference_panel: {
      heading: string;
    };
    configuration_panel: {
      heading: string;
    };
    configuration_controls: {
      daemon_unavailable_title: string;
      daemon_unavailable_default: string;
      microphone_heading: string;
      source_label: string;
      auto_first_available: string;
      channel_label: string;
      auto_channel: string;
      inference_cadence_heading: string;
      overlap_ratio_label: string;
      top_k_label: string;
      loading: string;
      kind_alsa: string;
      kind_unknown: string;
      approx_hz: (hz: string) => string;
      khz: (khz: string) => string;
      hz: (rate: number) => string;
    };
    top_k_meter: {
      awaiting_first_frame: string;
    };
    active_head_card: {
      heading: string;
      pill_default: string;
      pill_workspace: string;
      pill_detached: string;
      pill_default_title: string;
      pill_workspace_title: string;
      pill_detached_title: string;
      loading_active: string;
      activated_label: string;
      class_count_label: (count: number) => string;
      workspace_dt: string;
      revision_dt: string;
      rev_value: (rev: number) => string;
      deleted_tag: string;
      loading: string;
      ws_title_orphaned_with_name: (name: string, uuid: string) => string;
      ws_title_orphaned: (uuid: string) => string;
      ws_title_with_name: (name: string, uuid: string) => string;
    };
  };

  /** `label_with_current` is a function so the locale owns separator/word-order around the embedded current-mode label, not just the prefix. */
  theme: {
    label: string;
    label_with_current: (currentLabel: string) => string;
    options: {
      auto: string;
      light: string;
      dark: string;
    };
  };

  /** Concrete language labels come from the locale registry, not the catalog; `label_with_current` is a function so a locale owns wording around the code. */
  locale: {
    label: string;
    label_with_current: (currentChip: string) => string;
    auto_label: string;
  };

  health: {
    aria_label: string;
    levels: {
      unknown: string;
      ok: string;
      degraded: string;
      unhealthy: string;
      unreachable: string;
    };
    popover: {
      daemon_unreachable_title: string;
      waiting_first_snapshot: string;
      subsystems_heading: string;
      seconds_ago: (seconds: number) => string;
      stat_cpu_label: string;
      stat_rss_label: string;
      stat_disk_free_label: string;
      uptime_label: string;
      dropped_count: (count: number) => string;
    };
  };

  /** Keyed by the daemon `{error, code}` envelope. Fallback chain: unmapped codes use the daemon prose verbatim; `something_went_wrong` covers empty messages; `request_failed` is the last-resort code-only form. */
  error: {
    another_train_running: string;
    another_convert_running: string;
    job_conflict: string;
    event_gap: string;
    too_early: string;
    unavailable: string;
    internal: string;
    unknown: string;
    something_went_wrong: string;
    request_failed: (code: string) => string;
  };

  /** `name.*` is shared by the workspace and category name validators (rules overlap mechanically); `cfg.*` is per-field training-config validation. */
  validation: {
    name: {
      empty: string;
      max_bytes: (max: number) => string;
      slashes_or_nul: string;
      starts_or_ends_whitespace: string;
      control_chars: string;
      starts_with_dot: string;
      starts_with_underscore: string;
      starts_with_hyphen: string;
      bad_chars: string;
      category_max_bytes: (max: number) => string;
      category_empty: string;
    };
    cfg: {
      epochs_whole: string;
      epochs_range: (min: number, max: number) => string;
      batch_whole: string;
      batch_range: (min: number, max: number) => string;
      lr_finite: string;
      lr_greater_than_zero: string;
      lr_max: (max: number) => string;
      seed_whole: string;
      seed_non_negative: string;
      seed_too_large: string;
      split_finite: string;
      split_min: string;
      split_max: (max: number) => string;
    };
  };

  /** Only context-free atoms that recur identically everywhere; anything that could vary by surface lives per-surface so a locale can translate it per context. */
  common: {
    cancel: string;
    dismiss: string;
  };

  /** `socket_status` is the SocketState enum's operator-facing translations, shared by every status-pill consumer. */
  streams: {
    socket_status: {
      connecting: string;
      open: string;
      closed: string;
      error: string;
    };
  };

  /** The recorder maps `getUserMedia` DOMException flavours onto these sentences, falling back to `mic_error_generic`. */
  recorder: {
    mic_error_denied: string;
    mic_error_not_found: string;
    mic_error_in_use: string;
    mic_error_interrupted: string;
    mic_error_generic: string;
  };

  category: {
    list: {
      heading: string;
      description: string;
      add_button: string;
      add_button_aria: string;
      loading: string;
      load_error: (error: string) => string;
      menu_delete: string;
      menu_hint_preserved: string;
      menu_rename: string;
      /** Rename waits for quiescence: an in-flight slice PUT/DELETE bakes the old category name and could re-create or orphan the old directory. */
      menu_rename_hint_busy: string;
      menu_add: string;
    };
    add_dialog: {
      title: string;
      name_label: string;
      name_placeholder: string;
      name_help_prefix: string;
      name_help_code_example: string;
      name_help_suffix: string;
      submit: string;
      error_exact_duplicate: string;
      error_case_insensitive_duplicate: (existingName: string) => string;
    };
    /** Directory name doubles as the trainer class label, so a rename bumps the workspace revision and marks prior heads stale daemon-side; inference is unaffected. */
    rename_dialog: {
      title: string;
      name_label: string;
      name_help: string;
      submit: string;
      /** Defence-in-depth throw (UI also disables Rename on the mandatory Background Noise row). */
      error_mandatory: string;
      /** Throw when an in-flight upload or committed-slice delete still bakes the old name; rename waits for quiescence. */
      error_busy: string;
    };
    delete_dialog: {
      title: string;
      body_server: string;
      body_idb: string;
      submit: string;
      error_fallback: string;
      /** Defence-in-depth throw when deleting the mandatory Background Noise row without `force: true` (UI also disables that Delete item). */
      error_mandatory_required: string;
      /** Defence-in-depth throw if the named category isn't in the slice listing; the UI only surfaces listed rows. */
      error_not_found: string;
    };
    row: {
      badge_synced: string;
      badge_uploading: string;
      badge_pending: string;
      badge_failed: string;
      badge_not_enough: string;
      badge_not_enough_with_state: (statusLabel: string) => string;
      title_synced: (tally: string) => string;
      title_uploading: (tally: string) => string;
      title_pending: (tally: string) => string;
      title_failed: (tally: string) => string;
      title_not_enough_empty: (missing: number, tally: string) => string;
      title_not_enough_synced: (tally: string, missing: number) => string;
      title_not_enough_uploading: (tally: string, missing: number) => string;
      title_not_enough_pending: (tally: string, missing: number) => string;
      /** Kebab overflow-button copy; the Rename/Delete labels inside the menu it opens live under `category.list`. */
      actions_aria: (displayName: string) => string;
      actions_title: string;
      actions_title_preserved: string;
      /** Stored lowercase because the badge applies a CSS capitalize transform. */
      badge_deleting: string;
    };
    slice_card: {
      aria_select: (filename: string) => string;
      aria_deselect: (filename: string) => string;
      aria_play: (filename: string) => string;
      title_failed: (errorOrUnknown: string) => string;
      title_uploading: (progressPct: number) => string;
      title_local: string;
      title_multi_click_deselect: string;
      title_multi_click_select: string;
      title_playing: string;
      title_idle: string;
      sr_deleting: (filename: string) => string;
      sr_uploading: (progressPct: number) => string;
      retry_aria: (filename: string) => string;
      retry_title_with_error: (errorMessage: string) => string;
      retry_title_no_error: string;
      retry_label: string;
      select_title: string;
      deselect_title: string;
      delete_aria: (filename: string) => string;
      delete_title: string;
      slice_select_aria: (filename: string) => string;
      slice_deselect_aria: (filename: string) => string;
      unknown_error: string;
    };
    trim_waveform: {
      handles_aria: string;
      handle_start_aria: string;
      handle_end_aria: string;
      selection_aria: string;
      playback_position_aria: string;
      /** `aria-valuetext` for trim-handle sliders; pre-formatted seconds, catalog owns the unit word so a locale can change it without touching markup. */
      value_seconds: (sec: string) => string;
      /** `aria-valuetext` for the slide-window slider; both bounds pre-formatted, catalog owns the range connector and unit word. */
      value_seconds_range: (startSec: string, endSec: string) => string;
    };
    slice_pane: {
      heading: string;
      tips_label: string;
      tip_audition_title: string;
      tip_audition_body: string;
      tip_diversity_title: string;
      tip_diversity_body: string;
      quota_above_title: (threshold: number) => string;
      quota_below_title: (threshold: number) => string;
      loading: string;
      load_error: (error: string) => string;
      empty_state_prefix: string;
      empty_state_button: string;
      empty_state_suffix: string;
      select_all_label: string;
      deselect_all_label: string;
      select_all_title: string;
      deselect_all_title: string;
      done_label: string;
      done_title: string;
      delete_title: string;
      delete_disabled_title: string;
      delete_inflight_title: (count: number) => string;
      delete_inflight_aria: (count: number) => string;
      delete_aria_count: (count: number) => string;
      delete_aria_fallback: string;
      delete_label_inflight: (count: number) => string;
      delete_label_count: (count: number) => string;
      delete_label_bare: string;
      menu_play: string;
      menu_stop: string;
      menu_retry_upload: string;
      menu_select: string;
      menu_deselect: string;
      menu_select_all: string;
      menu_deselect_all: string;
      menu_done_exit: string;
      menu_retry_failed_in_selection: string;
      menu_delete_batch: (count: number) => string;
      menu_delete: string;
      menu_hint_a: string;
      menu_hint_esc: string;
      menu_hint_ctrl_click: string;
      menu_hint_del_backspace: string;
    };
    input_pane: {
      heading: string;
      tips_label: string;
      tip_stream_title: string;
      tip_stream_body: string;
      tip_environment_title: string;
      tip_environment_body: string;
      tip_meter_title: string;
      tip_meter_body: string;
      pane_aria: (categoryDisplay: string) => string;
      source_aria: string;
      loudness_aria: string;
      source_microphone_group: string;
      source_system_default_mic: string;
      source_remembered: (label: string) => string;
      /** Fallback labels for before mic permission is granted, when `MediaDeviceInfo.label` is empty; `idFrag` is the short deviceId slice, or `source_mic_default_id` when the id is also empty. */
      source_mic_fallback: (n: number, idFrag: string) => string;
      source_mic_remembered_fallback: (idFrag: string) => string;
      source_mic_default_id: string;
      source_live_stream_group: string;
      source_daemon_stream: string;
      source_daemon_stream_with_status: (status: string) => string;
      drop_zone_title: (cap: string) => string;
      drop_zone_idle: string;
      drop_zone_browse: string;
      record_aria_stream: string;
      record_aria_mic: string;
      /** Visible Record-button text shared by stream- and mic-capture modes; the aria-label disambiguates the source. */
      record_label: string;
      record_title_stream_open: (max: string) => string;
      record_title_stream_connecting: string;
      record_title_stream_closed: string;
      record_title_stream_unsupported: string;
      capture_stop_aria_stream: string;
      capture_stop_aria_mic: string;
      capture_stop_label: string;
      capture_discard_label: string;
      capture_encoding: string;
      capture_decoding: string;
      trim_selection_prefix: string;
      trim_drag_hint: string;
      trim_projected_slices: (count: number) => string;
      trim_unused_label: string;
      slice_aria_enabled: (count: number) => string;
      slice_aria_disabled: string;
      slice_title_enabled: (count: number) => string;
      slice_title_disabled: string;
      slice_label_bare: string;
      slice_label_count: (count: number) => string;
      discard_aria: string;
      discard_title: string;
      discard_label: string;
      play_stop_aria: string;
      play_stop_title: string;
      play_aria: string;
      play_title: string;
      export_aria: string;
      export_title: string;
      error_file_too_large: (size: string, cap: string) => string;
      /** Clips shorter than one 1-second training window would be zero-padded, NaN their whole spectrogram, and be silently dropped at training; rejecting at import surfaces that up front. */
      error_clip_too_short: (clipSecs: string) => string;
      /** The Input slot is single-tenant (most recent clip only), so a drop of >1 file is rejected. */
      error_only_one_file: string;
      error_only_wav: string;
      error_could_not_import: string;
      error_could_not_discard: string;
      error_could_not_decode_draft: string;
      error_could_not_save_recording: string;
      error_could_not_capture_stream: string;
      error_could_not_slice: string;
      /** WAV-magic + canonical-decode reasons, surfaced as the decoder's specific cause instead of the generic `error_only_wav` fallback. */
      error_wav_too_small_for_header: string;
      error_wav_missing_riff: string;
      error_wav_missing_wave: string;
      error_wav_empty: string;
      error_wav_buffer_too_small: string;
      /** For the rare browser build exposing neither `AudioContext` nor `webkitAudioContext`. */
      error_web_audio_unavailable: string;
      auto_stopped_at_cap: string;
      silent_dropped_suffix: (count: number) => string;
    };
  };

  /** `stage` / `state` labels are functions so the label utilities read through them and stay reactive on locale switch. */
  training: {
    pane: {
      heading: string;
      subtitle_other_running: string;
      subtitle_default: string;
      readiness_loading: string;
      readiness_no_categories: string;
      readiness_background_short: (need: number) => string;
      readiness_foreground_short: string;
      button_starting: string;
      button_cancel: string;
      button_cancelling: string;
      button_retrain: string;
      button_train: string;
      button_title_loading: string;
      button_title_not_ready_default: string;
      button_title_form_errors: string;
      button_title_idle_trained: string;
      button_title_idle_busy: string;
      button_title_idle_ready: string;
      button_title_starting: string;
      button_title_running: string;
      button_title_cancelling: string;
      summary_chip_epochs: (epochs: number) => string;
      summary_chip_no_holdout: string;
      summary_chip_val: (pctLabel: string) => string;
      hyperparameters_disclosure_label: string;
      start_error_title: string;
    };
    form: {
      epochs_label: string;
      batch_size_label: string;
      learning_rate_label: string;
      validation_split_label: string;
      validation_split_hint: string;
      seed_label: string;
      seed_hint: string;
      seed_placeholder: string;
    };
    progress: {
      submitting: string;
      job_short_id: (shortId: string) => string;
      train_loss_label: string;
      train_acc_label: string;
      val_acc_label: string;
      val_acc_disabled_label: string;
      em_dash: string;
    };
    logs: {
      heading: string;
      entry_count: (count: number) => string;
      waiting_first_message: string;
    };
    chart: {
      waiting_first_epoch: string;
      legend_loss: string;
      legend_train: string;
      legend_val: string;
      tooltip_epoch: string;
      tooltip_loss: string;
      tooltip_train: string;
      tooltip_val: string;
      chart_aria: string;
    };
    history: {
      heading: string;
      keeps_last: (cap: number) => string;
      retention_title: (cap: number) => string;
      empty_state_prefix: string;
      empty_state_button: string;
      empty_state_suffix: string;
      hide_older_label: string;
      show_older_label: (count: number) => string;
      hide_older_title: string;
      show_older_title: string;
      load_more_label: (count: number) => string;
      load_more_title: string;
      menu_delete: string;
      menu_deleting: string;
      menu_hint_train_active: string;
      menu_hint_live: string;
      delete_error_title: string;
    };
    history_item: {
      time_started_pre_ack: string;
      time_started: (relative: string) => string;
      time_finished: (relative: string) => string;
      time_title_started: (absolute: string) => string;
      time_title_finished: (absolute: string) => string;
      detail_epoch: (current: number, total: number) => string;
      detail_class_count: (count: number) => string;
      detail_val_acc: (pctLabel: string) => string;
      detail_train_acc: (pctLabel: string) => string;
      detail_stopped_at: (stageLabel: string) => string;
    };
    summary: {
      completed_aria: string;
      failed_aria: string;
      cancelled_aria: string;
      duration_label: string;
      epochs_label: string;
      best_val_at: (epoch: number) => string;
      final_train_acc_label: string;
      classes_label: string;
      stopped_at_label: string;
      cancelled_at_label: string;
      epochs_tooltip_full: string;
      epochs_tooltip_partial: string;
      after_epochs: (run: number, total: number) => string;
      failed_no_diagnostic: string;
      cancelled_default_reason: string;
      /** Headline fallback when the daemon's typed `error` and the trainer's last progress message are both empty. */
      failed_default: string;
    };
    stage: {
      prepare: string;
      dataset_scan: string;
      feature_extract: string;
      train: string;
      save: string;
      publish: string;
    };
    state: {
      running: string;
      completed: string;
      failed: string;
      cancelled: string;
    };
    /** Pre-ack pseudo-state kept distinct from the four-variant `TrainingJobState` enum so it translates without leaking into the wire-shape contract. */
    state_submitting: string;
    /** Training log lines composed by the training store from SSE events. */
    store_log: {
      /** Local seed line shown before the daemon's first event, for immediate feedback during the admission window. */
      seed_submitted: string;
      /** Local seed line shown after re-binding to an in-flight job across a page reload. */
      seed_recovered: string;
      job_submitted: (backbone: string) => string;
      job_running: string;
      phase_prefix: (stageLabel: string) => string;
      job_failed: (stageLabel: string, error: string) => string;
      job_cancelled: (stageLabel: string) => string;
      job_cancelled_shutdown: (stageLabel: string) => string;
      scanned_dataset: (nClasses: number, nExamples: number) => string;
      /** `dropped` is NaN + I/O drops; the catalog hides the dropped suffix when zero so locales own the word order. */
      features_extracted: (kept: number, dropped: number, elapsedSec: string) => string;
      train_split: (trainN: number, valN: number) => string;
      /** `lossLabel` / `trainAccLabel` are pre-formatted (non-finite renders `-`); `valAccLabel === null` hides the val suffix (holdout disabled). */
      epoch_completed: (
        epoch: number,
        epochs: number,
        lossLabel: string,
        trainAccLabel: string,
        valAccLabel: string | null
      ) => string;
      /** `bestValAccLabel` / `bestEpoch` are both `null` when there was no holdout or the value is non-finite, hiding the best-val suffix; typed args (not a pre-composed string) let locales reorder it without caller churn. */
      train_loop_done: (
        epochsRun: number,
        elapsedSec: string,
        bestValAccLabel: string | null,
        bestEpoch: number | null
      ) => string;
      head_published: (headId: string, size: string, nClasses: number, rev: number) => string;
      /** `labelsList` is pre-formatted so reserved synthetics get their pretty form; empty string suppresses the suffix (no classes on the run). */
      job_completed: (labelsList: string) => string;
    };
  };

  /** Status badges (`Active`/`Latest`/`Default`) stay per-surface, not a shared namespace, to preserve each badge's operator context. */
  deploy: {
    /** Status pills are paired label + tooltip (`pill_*` / `pill_*_title`) so each state translates as a coherent unit. */
    pane: {
      heading: string;
      description: string;
      pill_deployed: string;
      pill_deployed_title: string;
      pill_default: string;
      pill_default_title: string;
      pill_standby: string;
      pill_standby_title: string;
      pill_detached: string;
      pill_detached_title: string;
      config_disclosure_label: string;
      config_chip_freq: (hzLabel: string) => string;
      config_chip_top_k: (topK: number) => string;
    };
    heads_table: {
      heading: string;
      count_label: (count: number) => string;
      count_retained: (retainedCap: number) => string;
      revert_to_default: string;
      revert_to_id: (shortId: string) => string;
      revert_title: string;
      default_row_headline: string;
      default_row_description: string;
      default_active_title: string;
      default_aria_active: string;
      default_aria_deploy: string;
      default_title_active: string;
      default_title_deploying: string;
      default_title_busy: string;
      default_title_idle: string;
      menu_deploy: string;
      menu_export: string;
      menu_exporting: string;
      menu_delete: string;
      menu_hint_active: string;
      menu_hint_deployed: string;
      error_deploy_head: string;
      error_export_head: string;
      error_deploy_default: string;
    };
    head_row: {
      pill_latest: string;
      pill_latest_title: string;
      pill_active: string;
      pill_active_title: string;
      meta_line: (size: string, classCount: number, rev: number, relative: string) => string;
      meta_classes: (classCount: number) => string;
      meta_rev: (rev: number) => string;
      row_aria_deployed: (shortId: string) => string;
      row_aria_deploy: (shortId: string) => string;
      row_title_deployed: string;
      row_title_deploying: string;
      row_title_exporting: string;
      row_title_busy: string;
      row_title_idle: string;
      export_title_exporting: string;
      export_title_idle: string;
      export_aria_exporting: (shortId: string) => string;
      export_aria_idle: (shortId: string) => string;
      info_title: string;
      info_aria: (shortId: string) => string;
    };
    inference_preview: {
      heading: string;
      off_title: string;
      off_description: string;
      start_button: string;
    };
    /** Shows only the trained class labels; status pills (Active/Latest) live on the row header. */
    info_dialog: {
      title_with_id: (shortId: string) => string;
      loading: string;
      error_title: string;
      retry: string;
      classes_heading: string;
      class_labels_aria: string;
    };
    delete_dialog: {
      title: string;
      body: string;
      submit: string;
    };
  };

  /** Keys stay per-surface (not DRY'd into shared verbs) so a translator sees each string in context; the same English word may translate differently per surface. */
  workspace: {
    list: {
      title: string;
      at_cap_subtitle: (max: number) => string;
      default_subtitle: string;
      daemon_unavailable_title: string;
      loading: string;
      empty_title: string;
      empty_description: string;
      selected_count_aria: (count: number) => string;
      new_button_label: string;
      new_button_aria: string;
      new_at_cap_label: (count: number, max: number) => string;
      new_at_cap_title: string;
      import_button_label: string;
      import_button_aria: string;
      import_button_title: string;
      select_button_label: string;
      done_button_label: string;
      select_all_label: string;
      deselect_all_label: string;
      bulk_delete_label_count: (count: number) => string;
      bulk_delete_label_bare: string;
      bulk_delete_aria_count: (count: number) => string;
      bulk_delete_aria_fallback: string;
      menu_open: string;
      menu_rename: string;
      menu_export: string;
      menu_delete: string;
      menu_select_one: string;
      menu_deselect_one: string;
      menu_select_all: string;
      menu_deselect_all: string;
      menu_select_workspaces: string;
      menu_done_exit: string;
      menu_new: string;
      menu_new_at_cap: (max: number) => string;
      menu_import: string;
    };
    /** Covers the /workspaces/[id] chrome around the page title, which is the workspace name (data, never translated). */
    detail: {
      back_link: string;
      loading: string;
      not_found_title: string;
      not_found_description: string;
      back_to_list_button: string;
      load_error_title: string;
      created_label: (relative: string) => string;
      rev_label: (rev: number) => string;
      modified_label: (relative: string) => string;
      live_pill_title: string;
      live_pill: string;
      menu_rename: string;
      menu_export: string;
      menu_import: string;
      menu_delete: string;
      menu_back_to_list: string;
    };
    create_dialog: {
      title: string;
      name_label: string;
      name_placeholder: string;
      name_help: string;
      submit: string;
    };
    rename_dialog: {
      title: string;
      name_label: string;
      name_help: string;
      submit: string;
    };
    delete_dialog: {
      title: string;
      body: string;
      submit: string;
    };
    /** `title_count` / `submit_count` plurals are handled English-inline for now, pending an Intl.PluralRules dispatch. */
    bulk_delete_dialog: {
      title_count: (count: number) => string;
      body: string;
      submit_count: (count: number) => string;
    };
    tool_island: {
      aria_label: string;
      rename_aria: string;
      rename_title: string;
      export_aria: string;
      export_title: string;
      import_aria: string;
      import_title: string;
    };
    card: {
      created_label: (relative: string) => string;
      select_aria: (name: string) => string;
      rename_aria: (name: string) => string;
      deleting: string;
    };
    /** Step machine pick-file -> (pick-target?) -> summary -> running -> done; `into-current`/`pick-target` modes share every step but the target picker, and the alpkg/tfjs archive branches share chrome but differ in per-step copy. */
    import_dialog: {
      title_into: (workspaceName: string) => string;
      title_fallback: string;
      step_indicator: (current: number, total: number) => string;
      pipeline_error_title: string;
      /** Defensive throw if the pipeline runs with neither an ALPKG archive nor a TFJS bundle staged; the step-1 gate prevents this in production. */
      error_invalid_state: string;
      pick_file: {
        drop_zone_title_attr: string;
        reading: string;
        drop_zone_tfjs_staging: string;
        drop_zone_idle: string;
        browse_button: string;
        error_empty_drop: string;
        error_multi_alpkg: (count: number) => string;
        error_mixed_archive: string;
        error_file_count_cap: (max: number, picked: number) => string;
        error_single_too_large: (name: string, size: string, cap: string) => string;
        error_total_too_large: (total: string, cap: string) => string;
        error_tfjs_merged_file_count: (mergedCount: number, cap: number) => string;
        error_tfjs_merged_bytes: (mergedBytes: string, cap: string) => string;
        staged_files_heading: string;
        staged_files_count: (count: number) => string;
        clear_button: string;
        /** Catch-block fallbacks for when the caught value isn't an Error instance (so its `.message` can't be used). */
        error_could_not_read_archive: string;
        error_could_not_read_file: string;
        error_could_not_read_picked_files: string;
        error_could_not_read_model_json: string;
        /** TFJS-bundle classification diagnostics; `*_one`/`*_many` pairs handle singular vs. plural shard wording, `quotedNames` is the pre-joined pre-quoted (≤3) filename list, and `overflow` true drives the trailing ellipsis. */
        tfjs_diag_empty_drop: string;
        tfjs_diag_no_model_json: string;
        tfjs_diag_ambiguous_model_json: (count: number) => string;
        tfjs_diag_multiple_labels_txt: string;
        tfjs_diag_multiple_metadata_json: string;
        tfjs_diag_both_labels: string;
        tfjs_diag_no_labels: string;
        tfjs_diag_shard_collision_one: (quotedName: string) => string;
        tfjs_diag_shard_collision_many: (quotedNames: string, overflow: boolean) => string;
        tfjs_diag_missing_shard_one: (quotedName: string) => string;
        tfjs_diag_missing_shards_many: (
          count: number,
          quotedNames: string,
          overflow: boolean
        ) => string;
        tfjs_diag_model_json_not_json: string;
        tfjs_diag_model_json_not_object: string;
        tfjs_diag_model_json_no_manifest: string;
        tfjs_diag_model_json_no_shards: string;
      };
      pick_target: {
        section_label: string;
        mode_radio_aria: string;
        mode_use_existing: string;
        mode_create_new: string;
        no_workspaces_prefix: string;
        no_workspaces_suffix: string;
        no_workspaces_link_label: string;
        workspace_list_aria: string;
        workspace_created_label: (relative: string) => string;
        create_name_placeholder: string;
        create_will_carry_tags: (tagsCsv: string) => string;
        alpkg_source_card_title: (name: string, id: string) => string;
        alpkg_source_created_label: (relative: string) => string;
        alpkg_source_rev_label: (rev: number) => string;
        alpkg_source_modified_label: (relative: string) => string;
        tfjs_bundle_card_title: string;
        tfjs_show_labels_aria: string;
        tfjs_meta_strip: (
          size: string,
          shards: number,
          classes: number | null,
          labelsFileName: string | null
        ) => string;
      };
      summary: {
        datasets_heading: string;
        datasets_counter: (selected: number, total: number) => string;
        checking_categories: string;
        slice_count: (count: number) => string;
        rename_button_aria: string;
        rename_button_title_default: string;
        mode_aria: (modeLabel: string) => string;
        mode_menu_aria: (sourceName: string) => string;
        rename_popover_aria: (sourceName: string) => string;
        rename_popover_heading: string;
        rename_chips_heading: string;
        heads_heading: string;
        heads_cap_tooltip: (cap: number) => string;
        heads_counter: (
          selected: number,
          existingInTarget: number,
          cap: number,
          activeInTarget: number
        ) => string;
        checking_heads: string;
        displacement_warning: (displaced: number, cap: number) => string;
        head_exists_badge_title: string;
        head_exists_badge: string;
        head_show_details_aria: string;
        head_class_count: (count: number) => string;
        head_info_metadata: (
          size: string,
          classes: number | null,
          revisionId: number | null,
          createdAbsolute: string | null,
          createdRelative: string | null
        ) => string;
        head_classes_heading: string;
        head_class_labels_aria: string;
        archive_errors_summary: (count: number) => string;
        tfjs_ignored_unknown: (count: number, fileList: string) => string;
        tfjs_classes_popover_heading: (count: number) => string;
        tfjs_classes_popover_aria: string;
        /** Disabled-row tooltips: `loading` (heads still fetching), `exists` (id already in target, row unchecked), `ceiling` (selection cap reached, row unchecked). */
        head_disabled_reasons: {
          loading: string;
          exists: string;
          ceiling: string;
        };
      };
      modes: {
        new: string;
        merge: string;
        replace: string;
        skip: string;
      };
      mode_tooltips: {
        new: string;
        merge: string;
        replace: string;
        skip: string;
      };
      mode_disabled_reasons: {
        new_exists: string;
        merge_missing: string;
        replace_missing: string;
      };
      running: {
        progress_replacing_categories: (
          categoryDisplay: string | null,
          done: number,
          total: number
        ) => string;
        progress_uploading_datasets: (
          categoryDisplay: string | null,
          done: number | null,
          total: number | null
        ) => string;
        progress_importing_heads: (
          index1: number,
          total: number,
          subPhase: string | null
        ) => string;
        progress_uploading_tfjs: (done: number, total: number) => string;
        progress_converting_tfjs: string;
        ds_pending: string;
        ds_replacing: string;
        ds_uploading_counter: (uploaded: number, total: number) => string;
        ds_done_uploaded: (uploaded: number) => string;
        ds_failed_count: (failed: number) => string;
        ds_failed_label: string;
        ds_failed_title_count: (failed: number) => string;
        head_queued: string;
        head_skipped_badge_title: string;
        head_per_log_not_started: string;
        head_per_log_no_events: string;
        log_count: (count: number) => string;
      };
      head_phase: {
        queued: string;
        uploading_files: string;
        starting_convert: string;
        converting: string;
        cleaning_up: string;
        done: string;
        failed: string;
      };
      head_outcome: {
        imported: string;
        replaced: string;
        skipped: string;
        failed: string;
      };
      convert_stage: {
        prepare: string;
        read_manifest: string;
        validate_manifest: string;
        verify_mpk: string;
        stage_mpk: string;
        read_model_json: string;
        stage_shards: string;
        extract_weights: string;
        read_labels: string;
        stage_head_mpk: string;
        publish_head: string;
      };
      convert_event: {
        job_submitted: (converter: string) => string;
        job_running: string;
        phase: (stageLabel: string) => string;
        manifest_validated: (classes: number) => string;
        mpk_verified: (size: string) => string;
        weights_extracted: (classes: number, inDim: number) => string;
        labels_loaded: (labels: number) => string;
        head_published: (idempotentSkip: boolean) => string;
        job_completed: (classes: number) => string;
        job_failed: (category: string, error: string) => string;
      };
      done: {
        conflict_detail: (storedSha8: string, incomingSha8: string) => string;
        retry_button: string;
      };
      footer: {
        cancel: string;
        back: string;
        next: string;
        import: string;
        importing: string;
        back_to_selection: string;
        done: string;
      };
    };
    /** Slice / class count plurals are handled English-inline today, pending Intl.PluralRules. */
    export_dialog: {
      title: (workspaceName: string) => string;
      load_error_title: string;
      loading: string;
      nothing_to_export: string;
      datasets_heading: string;
      heads_heading: string;
      select_all: string;
      deselect_all: string;
      row_empty: string;
      row_slice_count: (count: number) => string;
      head_meta_title: (size: string, classCount: number) => string;
      head_meta_classes: (count: number) => string;
      pending_warning: string;
      progress_preparing_workspace: string;
      progress_fetching_slices: string;
      progress_listing_slices: string;
      progress_fetched_slices: (done: number, total: number) => string;
      progress_validating_heads: string;
      progress_validated_heads: (done: number, total: number) => string;
      progress_packing: string;
      progress_downloading: string;
      error_default: string;
      error_in_category: (categoryDisplay: string) => string;
      error_for_head: (shortId: string) => string;
      exporting: string;
      export_aria: string;
      export_button: string;
    };
  };
}
