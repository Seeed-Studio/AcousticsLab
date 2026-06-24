import type { Messages } from '../types';

// 日本語 (ja). Mirrors en.ts 1:1 (keys, params, ${...}, comments); see en.ts for conventions. ja
// style: full-width 。、（）; 「」for label/name quotes, straight "..." for code tokens; half-width
// space between a numeral and a JA word/unit; keep ' · ', '…', numbers, units, tokens, ${...} ASCII;
// break dash ' — '; 'Background Noise' stays literal.
export const ja = {
  app: {
    name: 'AcousticsLab',
    description:
      'リアルタイム音響イベント検出を開発・デプロイするための、プライベートかつマルチバックエンドで完全ローカル動作する AI/ML ツールキット。'
  },
  routes: {
    dashboard_title: (brand) => brand,
    workspace_list_title: (brand) => `ワークスペース · ${brand}`,
    workspace_detail_title: (workspaceName, brand) => `${workspaceName} · ${brand}`
  },
  nav: {
    dashboard: 'ダッシュボード',
    workspaces: 'ワークスペース',
    home_aria: 'AcousticsLab ホーム',
    menu_fallback: 'メニュー',
    primary_nav_aria: 'メインナビゲーション'
  },
  dashboard: {
    limited_support_title: 'ブラウザのサポートが限定的です',
    visualization_panel: {
      heading: '可視化',
      // Dot-separated segments so codec/channels can drop on a narrow header; rate and window always stay.
      audio_sample_rate: '48 kHz',
      audio_channels: 'モノラル',
      audio_codec: 'opus',
      audio_window: '3 s ウィンドウ'
    },
    inference_panel: {
      heading: '推論'
    },
    configuration_panel: {
      heading: '設定'
    },
    configuration_controls: {
      daemon_unavailable_title: 'デバイスを利用できません',
      daemon_unavailable_default: 'デバイスに到達できると、設定は自動的に再開されます。',
      microphone_heading: 'マイク',
      source_label: 'ソース',
      auto_first_available: '自動 · 最初に利用可能なもの',
      channel_label: 'チャンネル',
      auto_channel: '自動',
      inference_cadence_heading: '推論間隔',
      overlap_ratio_label: 'オーバーラップ比率',
      top_k_label: 'Top-K',
      loading: '読み込み中…',
      kind_alsa: 'ALSA',
      kind_unknown: '不明',
      approx_hz: (hz) => `${hz} Hz`,
      khz: (khz) => `${khz} kHz`,
      hz: (rate) => `${rate} Hz`
    },
    top_k_meter: {
      awaiting_first_frame: '最初の推論フレームを待機中…'
    },
    active_head_card: {
      heading: 'アクティブモデル',
      pill_default: 'デフォルト',
      pill_workspace: 'ワークスペース',
      pill_detached: '分離済み',
      pill_default_title: '組み込みのデフォルトモデルが稼働中です。',
      pill_workspace_title: 'トレーニング済みのワークスペースモデルが稼働中です。',
      pill_detached_title:
        'このモデルがアクティブ化された後、ソースワークスペースが削除されました。',
      loading_active: 'アクティブモデルを読み込み中…',
      activated_label: 'アクティブ化済み',
      class_count_label: () => 'クラス',
      workspace_dt: 'ワークスペース',
      revision_dt: 'リビジョン',
      rev_value: (rev) => `rev ${rev}`,
      deleted_tag: '（削除済み）',
      loading: '読み込み中…',
      ws_title_orphaned_with_name: (name, uuid) => `${name} · ${uuid}（ワークスペース削除済み）`,
      ws_title_orphaned: (uuid) => `${uuid}（ワークスペース削除済み）`,
      ws_title_with_name: (name, uuid) => `${name} · ${uuid}`
    }
  },
  theme: {
    label: 'テーマ',
    label_with_current: (currentLabel) => `テーマ：${currentLabel}`,
    options: {
      auto: '自動',
      light: 'ライト',
      dark: 'ダーク'
    }
  },
  locale: {
    label: '言語',
    label_with_current: (currentChip) => `言語：${currentChip}`
  },
  health: {
    aria_label: 'システムヘルス',
    levels: {
      unknown: '接続中',
      ok: '正常',
      degraded: '低下',
      unhealthy: '異常',
      unreachable: '到達不可'
    },
    popover: {
      daemon_unreachable_title: 'デバイスに到達できません',
      waiting_first_snapshot: '最初のステータススナップショットを待機中…',
      subsystems_heading: 'サブシステム',
      seconds_ago: (seconds) => `${seconds}s 前`,
      stat_cpu_label: 'cpu',
      stat_rss_label: 'rss',
      stat_disk_free_label: 'ディスク空き',
      uptime_label: '稼働時間',
      dropped_count: (count) => `ドロップ：${count}`
    }
  },
  common: {
    cancel: 'キャンセル',
    dismiss: '閉じる'
  },
  error: {
    another_train_running: 'このデバイスでは別のトレーニングジョブがすでに実行中です。',
    another_convert_running: 'このデバイスでは別の変換ジョブがすでに実行中です。',
    job_conflict: 'このリソースでは別の操作がすでに進行中です。',
    event_gap: 'イベントストリームが先に進んだため、ログから追いつく必要があります。再接続中…',
    too_early: 'デバイスは前回の変更をまだ適用中です。再試行中…',
    unavailable: 'デバイスは一時的に利用できません。しばらくしてから再試行してください。',
    internal:
      'デーモンで内部エラーが発生しました。再試行してください。継続する場合はデーモンのログを確認してください。',
    unknown: '問題が発生しました。再試行してください。',
    something_went_wrong: '問題が発生しました。',
    request_failed: (code) => `リクエストに失敗しました（${code}）。`
  },
  validation: {
    name: {
      empty: '名前は空にできません。',
      max_bytes: (max) => `名前は ${max} バイト以下にしてください。`,
      slashes_or_nul: '名前にスラッシュや NUL バイトを含めることはできません。',
      starts_or_ends_whitespace: '名前の先頭や末尾に空白文字を使うことはできません。',
      control_chars: '名前に制御文字を含めることはできません。',
      starts_with_dot: 'カテゴリ名はドットで始めることはできません。',
      starts_with_underscore:
        'カテゴリ名はアンダースコアで始めることはできません（組み込みクラス用に予約されています）。',
      starts_with_hyphen:
        'カテゴリ名はハイフンで始めることはできません（セキュリティ上の理由から）。',
      bad_chars: '使用できるのは英字、数字、ドット、ハイフン、アンダースコアのみです。',
      category_max_bytes: (max) => `カテゴリ名は ${max} バイト以下にしてください。`,
      category_empty: 'カテゴリ名は空にできません。'
    },
    cfg: {
      epochs_whole: 'エポックは整数にしてください。',
      epochs_range: (min, max) => `エポックは ${min} から ${max} の間にしてください。`,
      batch_whole: 'バッチサイズは整数にしてください。',
      batch_range: (min, max) => `バッチサイズは ${min} から ${max} の間にしてください。`,
      lr_finite: '学習率は有限の数値にしてください。',
      lr_greater_than_zero: '学習率は 0 より大きくしてください。',
      lr_max: (max) => `学習率は ${max} 以下にしてください。`,
      seed_whole: 'シードは整数にしてください。',
      seed_non_negative: 'シードは 0 以上にしてください。',
      seed_too_large: 'シードが大きすぎます。',
      split_finite: '検証分割は有限の数値にしてください。',
      split_min: '検証分割は 0 以上にしてください。',
      split_max: (max) => `検証分割は ${max} 以下にしてください。`
    }
  },
  streams: {
    socket_status: {
      connecting: '接続中',
      open: 'ライブ',
      closed: '切断',
      error: 'エラー'
    }
  },
  recorder: {
    mic_error_denied:
      'マイクへのアクセスが拒否されました。ブラウザの設定でマイクへのアクセスを許可してから再試行してください。',
    mic_error_not_found: 'マイクが見つかりませんでした。接続してから再試行してください。',
    mic_error_in_use: 'マイクは別のアプリケーションで使用中です。閉じてから再試行してください。',
    mic_error_interrupted: 'マイクのキャプチャが中断されました。再試行してください。',
    mic_error_generic: 'マイクを開始できませんでした。再試行してください。'
  },
  category: {
    list: {
      heading: 'データセット',
      description:
        '各カテゴリはトレーナーが学習するクラスラベルになります — Background Noise は必須です。',
      add_button: 'カテゴリを追加',
      add_button_aria: 'カテゴリを追加',
      loading: 'カテゴリを読み込み中…',
      load_error: (error) => `カテゴリを読み込めませんでした。${error}`,
      menu_delete: '削除',
      menu_hint_preserved: '保持',
      menu_rename: '名前を変更',
      menu_rename_hint_busy: '先に進行中の作業を完了してください',
      menu_add: 'カテゴリを追加'
    },
    add_dialog: {
      title: 'カテゴリを追加',
      name_label: '名前',
      name_placeholder: '例：cat',
      name_help_prefix:
        '英字、数字、ドット、ハイフン、アンダースコア。この名前はディスク上のディレクトリ名（例：',
      name_help_code_example: 'datasets/cat/',
      name_help_suffix: '）を兼ね、トレーナーが使うクラスラベルにもなります。',
      submit: '追加',
      error_exact_duplicate: 'この名前のカテゴリはすでに存在します。',
      error_case_insensitive_duplicate: (existingName) =>
        `既存の「${existingName}」と競合します（ほとんどのファイルシステムで名前は大文字小文字を区別しません）。`
    },
    rename_dialog: {
      title: 'カテゴリの名前を変更',
      name_label: '名前',
      name_help:
        'この名前はディスク上のディレクトリとトレーナーのクラスラベルを兼ねるため、名前の変更はクラスラベルを変更します。既存のトレーニング済みモデルは古いラベルのまま保持され、再トレーニングするまで古い状態として表示されます。',
      submit: '保存',
      error_mandatory: 'Background Noise は保持され、名前を変更できません。',
      error_busy:
        'このカテゴリの名前を変更する前に、進行中のアップロードと削除を完了するか解除してください。'
    },
    delete_dialog: {
      title: 'このカテゴリを削除しますか？',
      body_server:
        'データセットフォルダとその中のすべてのスライスを削除します。この操作は元に戻せません。',
      body_idb:
        'このカテゴリをローカルのリストから削除します。スライスはアップロードされていないため、デバイス上では何も変わりません。',
      submit: '削除',
      error_fallback: 'カテゴリを削除できませんでした。',
      error_mandatory_required: 'Background Noise は保持され、削除できません。',
      error_not_found: 'カテゴリが見つかりません。'
    },
    slice_card: {
      aria_select: (filename) => `スライス ${filename} を選択`,
      aria_deselect: (filename) => `スライス ${filename} の選択を解除`,
      aria_play: (filename) => `スライス ${filename} を再生`,
      title_failed: (errorOrUnknown) =>
        `アップロードに失敗しました：${errorOrUnknown}。右クリックで再試行します。`,
      title_uploading: (progressPct) => `アップロード中… ${progressPct}%`,
      title_local: 'ローカル — アップロード待ち',
      title_multi_click_deselect: 'クリックで選択を解除（Esc で選択を終了）',
      title_multi_click_select: 'クリックで選択に追加（Esc で選択を終了）',
      title_playing: '再生中 — クリックで再生し直します',
      title_idle: 'クリックで再生（Ctrl/Cmd-click で選択）',
      sr_deleting: (filename) => `スライス ${filename} を削除中`,
      sr_uploading: (progressPct) => `アップロード中 ${progressPct}%`,
      retry_aria: (filename) => `スライス ${filename} のアップロードを再試行`,
      retry_title_with_error: (errorMessage) =>
        `アップロードに失敗しました：${errorMessage}。クリックで再試行します。`,
      retry_title_no_error: 'アップロードに失敗しました。クリックで再試行します。',
      retry_label: '再試行',
      select_title: '選択',
      deselect_title: '選択を解除',
      delete_aria: (filename) => `スライス ${filename} を削除`,
      delete_title: 'スライスを削除',
      slice_select_aria: (filename) => `スライス ${filename} を選択`,
      slice_deselect_aria: (filename) => `スライス ${filename} の選択を解除`,
      unknown_error: '不明なエラー'
    },
    trim_waveform: {
      handles_aria: 'トリミングハンドル。ドラッグしてスライス範囲の開始と終了を設定します',
      handle_start_aria: 'トリミング開始',
      handle_end_aria: 'トリミング終了',
      selection_aria:
        '選択ウィンドウのスライド。ドラッグして両端のトリミング位置を同時に移動します',
      playback_position_aria: '再生位置',
      value_seconds: (sec) => `${sec} 秒`,
      value_seconds_range: (startSec, endSec) => `${startSec} から ${endSec} 秒`
    },
    slice_pane: {
      heading: 'スライス',
      tips_label: 'スライスモジュールのヒント',
      tip_audition_title: 'トレーニング前にすべてのスライスを試聴しましょう。',
      tip_audition_body:
        'ラベル付けを誤った 1 行がクラス全体に偏りを生みます — カードをクリックして再生し、ためらわず破棄しましょう。',
      tip_diversity_title: '多様性は量に勝ります。',
      tip_diversity_body:
        '10 件の多様なテイク（距離、角度、背景）は、ほぼ同一の 30 件のコピーよりも質の高いトレーニングになります。',
      quota_above_title: (threshold) =>
        `トレーニングに必要な最小 ${threshold} スライスを超えています。`,
      quota_below_title: (threshold) =>
        `トレーニングに必要な最小 ${threshold} スライスを下回っています。必要数を満たすにはさらにスライスしてください。`,
      loading: 'スライスを読み込み中…',
      load_error: (error) => `スライスを読み込めませんでした。${error}`,
      empty_state_prefix: 'まだスライスがありません。入力ペインでクリップをトリミングし、',
      empty_state_button: 'スライス',
      empty_state_suffix: ' をクリックしてこのグリッドを埋めます。',
      select_all_label: 'すべて選択',
      deselect_all_label: 'すべて選択解除',
      select_all_title: 'すべてのスライスを選択（Cmd/Ctrl+A）',
      deselect_all_title: 'すべてのスライスの選択を解除（Cmd/Ctrl+A）',
      done_label: '完了',
      done_title: '選択を終了（Esc）',
      delete_title: '選択したスライスを削除（Del / Backspace）',
      delete_disabled_title: '削除するスライスを 1 つ以上選択してください',
      delete_inflight_title: (count) => `${count} スライスを削除中…`,
      delete_inflight_aria: (count) => `${count} スライスを削除中`,
      delete_aria_count: (count) => `選択した ${count} スライスを削除`,
      delete_aria_fallback: '選択したスライスを削除',
      delete_label_inflight: (count) => `${count} スライスを削除中…`,
      delete_label_count: (count) => `${count} スライスを削除`,
      delete_label_bare: '削除',
      menu_play: '再生',
      menu_stop: '停止',
      menu_retry_upload: 'アップロードを再試行',
      menu_select: '選択',
      menu_deselect: '選択を解除',
      menu_select_all: 'すべて選択',
      menu_deselect_all: 'すべて選択解除',
      menu_done_exit: '完了（選択を終了）',
      menu_retry_failed_in_selection: '選択内の失敗を再試行',
      menu_delete_batch: (count) => `${count} スライスを削除`,
      menu_delete: '削除',
      menu_hint_a: 'Cmd/Ctrl+A',
      menu_hint_esc: 'Esc',
      menu_hint_ctrl_click: 'Ctrl/Cmd-click',
      menu_hint_del_backspace: 'Del / Backspace'
    },
    input_pane: {
      heading: '入力',
      tips_label: '入力モジュールのヒント',
      tip_stream_title: 'デバイスのサウンドストリームを優先しましょう。',
      tip_stream_body:
        'スライスは推論と同じ DSP を共有するため、ファインチューニング後にトレーニング済みモデルが分布シフトの影響を受けません。',
      tip_environment_title: 'デプロイ先の環境で録音しましょう。',
      tip_environment_body:
        'クリーンなスタジオ収録はノイズ除去のトレーニングが不足します。実際の背景ノイズは、モデルが学習すべき内容のおよそ半分を占めるのが理想です。',
      tip_meter_title: 'デシベルメーターで緑から黄の状態を保ちましょう。',
      tip_meter_body:
        '赤はクリッピングを意味します。情報が失われ、トレーナーが学習できなくなります。',
      pane_aria: (categoryDisplay) => `カテゴリ ${categoryDisplay} の入力モジュール`,
      source_aria: '入力ソース',
      loudness_aria: '音量メーター',
      source_microphone_group: 'マイク',
      source_system_default_mic: 'システム既定のマイク',
      source_remembered: (label) => `${label}（記憶済み）`,
      source_mic_fallback: (n, idFrag) => `マイク ${n}（${idFrag}）`,
      source_mic_remembered_fallback: (idFrag) => `マイク（${idFrag}）`,
      source_mic_default_id: 'default',
      source_live_stream_group: 'ライブストリーム',
      source_daemon_stream: 'デバイスのサウンドストリーム',
      source_daemon_stream_with_status: (status) => `デバイスのサウンドストリーム · ${status}`,
      drop_zone_title: (cap) =>
        `WAV ファイルをここにドロップ（最大 ${cap}）、またはクリックして参照`,
      drop_zone_idle: 'WAV をここにドラッグ＆ドロップ',
      drop_zone_browse: 'ファイルを参照',
      record_aria_stream: 'ライブサウンドストリームからのキャプチャを開始',
      record_aria_mic: 'マイクからの録音を開始',
      record_label: '録音',
      record_title_stream_open: (max) =>
        `ライブサウンドストリームをキャプチャします（${max} で自動停止）。`,
      record_title_stream_connecting:
        'デバイスのサウンドストリームに接続中です。開いたら録音が可能になります。',
      record_title_stream_closed:
        'デバイスのサウンドストリームに到達できません。デバイスが稼働中か確認してください。',
      record_title_stream_unsupported:
        'このブラウザではここでライブサウンドストリームをデコードできません — セキュア（HTTPS）コンテキストでの WebCodecs が必要です。セキュアゲートウェイ経由でこのページを開くか、代わりに WAV ファイルをドロップまたは参照してください。',
      capture_stop_aria_stream: 'ストリームキャプチャを停止',
      capture_stop_aria_mic: '録音を停止',
      capture_stop_label: '停止',
      capture_discard_label: '破棄',
      capture_encoding: 'エンコード中…',
      capture_decoding: 'デコード中…',
      trim_selection_prefix: '選択範囲：',
      trim_drag_hint: 'ハンドルを ≥ 1 s までドラッグするとスライスが有効になります。',
      trim_projected_slices: (count) => `各 1 s の ${count} スライス`,
      trim_unused_label: '未使用',
      slice_aria_enabled: (count) => `${count} スライスに分割`,
      slice_aria_disabled: 'スライス（選択範囲は 1 秒以上必要です）',
      slice_title_enabled: (count) => `${count} スライスを右ペインに追加`,
      slice_title_disabled: 'スライスするには選択範囲が ≥ 1 s 必要です',
      slice_label_bare: 'スライス',
      slice_label_count: (count) => `スライス · ${count}`,
      discard_aria: 'クリップを破棄',
      discard_title: 'クリップを破棄',
      discard_label: '破棄',
      play_stop_aria: '再生を停止',
      play_stop_title: '再生を停止',
      play_aria: 'トリミングした選択範囲を再生',
      play_title: 'トリミングした選択範囲を再生',
      export_aria: 'WAV としてダウンロード',
      export_title: 'WAV としてダウンロード',
      error_file_too_large: (size, cap) =>
        `ファイルは ${size} です — インポート上限は ${cap} です。短くトリミングして再エクスポートし、もう一度ドロップしてください。`,
      error_clip_too_short: (clipSecs) =>
        `クリップは ${clipSecs} s しかなく、トレーニングにはクリップごとに少なくとも 1 s が必要なため、短いクリップは完全に除外されます。1 s 以上のクリップをインポートまたは録音してください。`,
      error_only_one_file:
        '一度に扱えるファイルは 1 つだけです — 入力スロットは最新のクリップのみを保持します。WAV を 1 つドロップしてください。',
      error_only_wav: 'サポートされているのは WAV ファイルのみです。',
      error_could_not_import: 'ファイルをインポートできませんでした。',
      error_could_not_discard: 'クリップを破棄できませんでした。',
      error_could_not_decode_draft: '保存されたドラフトをデコードできませんでした。',
      error_could_not_save_recording: '録音を保存できませんでした。',
      error_could_not_capture_stream: 'ストリームをキャプチャできませんでした。',
      error_could_not_slice: 'クリップをスライスできませんでした。',
      error_wav_too_small_for_header:
        'ファイルが小さすぎて WAV ではありません（ヘッダーに少なくとも 12 バイト必要です）。',
      error_wav_missing_riff: 'WAV ファイルではありません（RIFF マジックがありません）。',
      error_wav_missing_wave: 'WAV ファイルではありません（WAVE マーカーがありません）。',
      error_wav_empty: 'ファイルが空か小さすぎて WAV ではありません。',
      error_wav_buffer_too_small:
        'WAV バッファが小さすぎます（標準ヘッダーに少なくとも 44 バイト必要です）。',
      error_web_audio_unavailable: 'このブラウザでは Web Audio API を利用できません。',
      auto_stopped_at_cap: '時間上限で自動停止しました。',
      silent_dropped_suffix: (count) => `無音の ${count} スライスをスキップしました`
    },
    row: {
      badge_synced: '同期済み',
      badge_uploading: 'アップロード中',
      badge_pending: '保留中',
      badge_failed: '失敗',
      badge_not_enough: 'サンプル不足',
      badge_not_enough_with_state: (statusLabel) => `サンプル不足 · ${statusLabel}`,
      title_synced: (tally) =>
        `${tally} スライスをデバイスにアップロード済み — トレーニング可能です。`,
      title_uploading: (tally) => `${tally} スライス。一部はまだデバイスにアップロード中です。`,
      title_pending: (tally) =>
        `${tally} スライスは準備済みですが、まだデバイスにアップロードされていません。`,
      title_failed: (tally) =>
        `${tally} スライス。少なくとも 1 件のアップロードに失敗しました。スライスカードから再試行するか、失敗した行を破棄してください。`,
      title_not_enough_empty: (missing, tally) =>
        `カテゴリごとの必要数（${tally}）を満たすには、さらに ${missing} スライスを追加してください。`,
      title_not_enough_synced: (tally, missing) =>
        `${tally} スライスをアップロード済み。カテゴリごとの必要数を満たすには、さらに ${missing} スライスを追加してください。`,
      title_not_enough_uploading: (tally, missing) =>
        `${tally} スライス。一部はまだアップロード中です。完了後にさらに ${missing} スライス必要です。`,
      title_not_enough_pending: (tally, missing) =>
        `${tally} スライスをローカルでキューに追加済み。さらに ${missing} スライス必要です。`,
      actions_aria: (displayName) => `${displayName} のアクション`,
      actions_title: 'カテゴリのアクション',
      actions_title_preserved: '保持 — 名前の変更と削除は無効です',
      badge_deleting: '削除中'
    }
  },
  training: {
    pane: {
      heading: 'トレーニング',
      subtitle_other_running:
        '別のワークスペースがトレーニング中です。同時に実行できるジョブは 1 つだけです。',
      subtitle_default:
        'このワークスペースのデータセットでモデルを調整します。新しいモデルが完成すると古いモデルは自動的に破棄されます。',
      readiness_loading: 'データセットを読み込み中…',
      readiness_no_categories:
        'トレーニングを開始するには、アップロード済みスライスを含む前景クラスを追加してください。',
      readiness_background_short: (need) =>
        `トレーニングを開始するには Background Noise にあと ${need} スライスのアップロードが必要です。`,
      readiness_foreground_short:
        'トレーニングを開始するには、少なくとも 1 つの前景クラスに 10 スライスのアップロードが必要です。',
      button_starting: '開始中…',
      button_cancel: 'キャンセル',
      button_cancelling: 'キャンセル中…',
      button_retrain: '再トレーニング',
      button_train: 'モデルをトレーニング',
      button_title_loading: 'データセットを読み込み中…',
      button_title_not_ready_default: '準備状況の理由',
      button_title_form_errors:
        'トレーニングを有効にするには、強調表示されたハイパーパラメータ欄を修正してください。',
      button_title_idle_trained:
        'このリビジョンに一致するモデルがすでにあります — 異なるハイパーパラメータや別の乱数シードを試すには再トレーニングしてください。下の「モデル」セクションから任意のモデルをアクティブ化できます。',
      button_title_idle_busy:
        '別のワークスペースがトレーニング中です。同時に実行できるジョブは 1 つだけです。',
      button_title_idle_ready: 'このワークスペースのデータセットでモデルをトレーニングします。',
      button_title_starting: 'トレーニングリクエストを送信中…',
      button_title_running: '実行中のトレーニングジョブをキャンセルします。',
      button_title_cancelling: 'キャンセル中…',
      summary_chip_epochs: (epochs) => `${epochs} エポック`,
      summary_chip_no_holdout: 'ホールドアウトなし',
      summary_chip_val: (pctLabel) => `検証 ${pctLabel}`,
      hyperparameters_disclosure_label: 'ハイパーパラメータ',
      start_error_title: 'トレーニングを開始できませんでした'
    },
    form: {
      epochs_label: 'エポック',
      batch_size_label: 'バッチサイズ',
      learning_rate_label: '学習率',
      validation_split_label: '検証分割',
      validation_split_hint: '· 0 で無効化',
      seed_label: 'シード',
      seed_hint: '· 空欄でデーモンが選んだエントロピーを使用',
      seed_placeholder: '（任意）'
    },
    progress: {
      submitting: '送信中…',
      job_short_id: (shortId) => `ジョブ ${shortId}…`,
      train_loss_label: 'トレーニング損失',
      train_acc_label: 'トレーニング精度',
      val_acc_label: '検証精度',
      val_acc_disabled_label: '検証精度 · 無効',
      em_dash: ' — '
    },
    logs: {
      heading: 'ログ',
      entry_count: (count) => `${count} 件`,
      waiting_first_message: '最初のメッセージを待機中…'
    },
    chart: {
      waiting_first_epoch: '最初のエポックを待機中…',
      legend_loss: '損失',
      legend_train: 'トレーニング',
      legend_val: '検証',
      tooltip_epoch: 'エポック',
      tooltip_loss: '損失',
      tooltip_train: 'トレーニング',
      tooltip_val: '検証',
      chart_aria: 'トレーニング指標チャート'
    },
    history: {
      heading: '履歴',
      keeps_last: (cap) => `最新 ${cap} 件を保持`,
      retention_title: (cap) =>
        `デーモンはワークスペースごとに最新 ${cap} 件のトレーニングログファイルを保持します。新しい実行が開始されると、古い JSONL トレースは削除されます。発行済みのモデル記録（下の「モデル」セクション）は影響を受けません — 削除されるのは JSONL トレースのみです。`,
      empty_state_prefix: 'このワークスペースにはまだトレーニング実行がありません。',
      empty_state_button: 'モデルをトレーニング',
      empty_state_suffix: ' をクリックして開始します。',
      hide_older_label: '古い実行を非表示',
      show_older_label: (count) => `古い実行 ${count} 件を表示`,
      hide_older_title: '古い実行セクションを直近 2 件に折りたたみます。',
      show_older_title:
        'このワークスペースの古いトレーニング実行を、5 件ずつのバッチで表示します。',
      load_more_label: (count) => `さらに ${count} 件を読み込む`,
      load_more_title: 'デバイスから古いトレーニング実行の次のバッチを取得します。',
      menu_delete: '削除',
      menu_deleting: '削除中…',
      menu_hint_train_active: 'トレーニング中',
      menu_hint_live: 'ライブ',
      delete_error_title: 'トレーニングログを削除できませんでした'
    },
    history_item: {
      time_started_pre_ack: '開始',
      time_started: (relative) => `${relative}に開始`,
      time_finished: (relative) => relative,
      time_title_started: (absolute) => `${absolute}に開始`,
      time_title_finished: (absolute) => `${absolute}に完了`,
      detail_epoch: (current, total) => `エポック ${current}/${total}`,
      detail_class_count: (count) => `${count} クラス`,
      detail_val_acc: (pctLabel) => `精度 ${pctLabel}`,
      detail_train_acc: (pctLabel) => `トレーニング ${pctLabel}`,
      detail_stopped_at: (stageLabel) => `${stageLabel}に停止`
    },
    summary: {
      completed_aria: '完了した実行の概要',
      failed_aria: '失敗した実行の概要',
      cancelled_aria: 'キャンセルされた実行の概要',
      duration_label: '所要時間',
      epochs_label: 'エポック',
      best_val_at: (epoch) => `最良の検証精度 @ ${epoch}`,
      final_train_acc_label: '最終トレーニング精度',
      classes_label: 'クラス',
      stopped_at_label: '停止段階',
      cancelled_at_label: 'キャンセル段階',
      epochs_tooltip_full: '設定されたエポック数を完了しました。',
      epochs_tooltip_partial: '実測エポック数と設定エポック数の比較。',
      after_epochs: (run, total) => `${run}/${total} エポック後`,
      failed_no_diagnostic:
        '診断情報が表示されませんでした。詳細はデーモンのログを確認してください。',
      cancelled_default_reason: '次のトレーニングチェックポイントで停止しました。',
      failed_default: 'トレーニングに失敗しました。'
    },
    stage: {
      prepare: '準備中',
      dataset_scan: 'データセットをスキャン中',
      feature_extract: '特徴を抽出中',
      train: 'トレーニング中',
      save: '保存中',
      publish: '発行中'
    },
    state: {
      running: '実行中',
      completed: '完了',
      failed: '失敗',
      cancelled: 'キャンセル済み'
    },
    state_submitting: '送信中',
    store_log: {
      seed_submitted: '送信しました。デバイスがイベントの発行を開始するのを待機中…',
      seed_recovered: 'デバイスから進行中のトレーニングジョブを復元しました。',
      job_submitted: (backbone) => `ジョブを送信 · バックボーン ${backbone}`,
      job_running: 'ジョブ実行中',
      phase_prefix: (stageLabel) => `フェーズ：${stageLabel}`,
      job_failed: (stageLabel, error) => `${stageLabel}にジョブが失敗 · ${error}`,
      job_cancelled: (stageLabel) => `${stageLabel}にジョブをキャンセル`,
      job_cancelled_shutdown: (stageLabel) => `${stageLabel}にジョブをキャンセル（デーモン停止）`,
      scanned_dataset: (nClasses, nExamples) =>
        `データセットをスキャン · ${nClasses} クラス · ${nExamples} 件`,
      features_extracted: (kept, dropped, elapsedSec) => {
        const droppedSuffix = dropped > 0 ? ` · ドロップ ${dropped}` : '';
        return `特徴を抽出 · 保持 ${kept}${droppedSuffix} · ${elapsedSec}s`;
      },
      train_split: (trainN, valN) => `トレーニング分割 · トレーニング ${trainN} · 検証 ${valN}`,
      epoch_completed: (epoch, epochs, lossLabel, trainAccLabel, valAccLabel) => {
        const valPart = valAccLabel !== null ? ` · 検証 ${valAccLabel}` : '';
        return `エポック ${epoch}/${epochs} · 損失 ${lossLabel} · トレーニング ${trainAccLabel}${valPart}`;
      },
      train_loop_done: (epochsRun, elapsedSec, bestValAccLabel, bestEpoch) => {
        const bestPart =
          bestValAccLabel !== null && bestEpoch !== null
            ? ` · 最良の検証精度 ${bestValAccLabel} @ エポック ${bestEpoch}`
            : '';
        return `トレーニングループ完了 · ${epochsRun} エポック · ${elapsedSec}s${bestPart}`;
      },
      head_published: (headId, size, nClasses, rev) =>
        `モデルを発行 · ${headId} · ${size} · ${nClasses} クラス · rev ${rev}`,
      job_completed: (labelsList) =>
        labelsList.length > 0 ? `ジョブ完了 · ${labelsList}` : 'ジョブ完了'
    }
  },
  deploy: {
    pane: {
      heading: 'デプロイ',
      description:
        'トレーニング済みモデルを閲覧して選択し、リアルタイム推論へシームレスにホットスワップします。',
      pill_deployed: 'デプロイ済み',
      pill_deployed_title: 'このワークスペースでトレーニングされたモデルがランタイムモデルです。',
      pill_default: 'デフォルト',
      pill_default_title: '組み込みのデフォルトモデルが稼働中です。',
      pill_standby: '待機中',
      pill_standby_title:
        '別のワークスペースのモデルがランタイムモデルです。このワークスペースは待機中です。ここでデプロイすると置き換わります。',
      pill_detached: '分離済み',
      pill_detached_title:
        'ランタイムモデルを生成したワークスペースは削除されました。モデルはまだ稼働中です。',
      config_disclosure_label: '入力と推論の設定',
      config_chip_freq: (hzLabel) => `freq ${hzLabel} Hz`,
      config_chip_top_k: (topK) => `top-k ${topK}`
    },
    heads_table: {
      heading: 'モデル',
      count_label: (count) => `${count} 件`,
      // Split off the bare count so it can collapse on a narrow card; carries its own leading comma.
      count_retained: (retainedCap) => `、最新 ${retainedCap} 件を保持`,
      revert_to_default: 'デフォルトに戻す',
      revert_to_id: (shortId) => `${shortId} に戻す`,
      revert_title: '以前稼働していたモデルを再デプロイ',
      default_row_headline: 'デフォルト',
      default_row_description: '組み込みのフォールバック。常に利用可能です。',
      default_active_title: '組み込みのデフォルトモデルが現在デプロイされています。',
      default_aria_active: 'デフォルトモデルがアクティブです',
      default_aria_deploy: 'デフォルトモデルをデプロイ',
      default_title_active: 'デフォルトモデルはすでにデプロイされています',
      default_title_deploying: 'デプロイ中…',
      default_title_busy: 'このリスト内の別のモデルがビジー状態です',
      default_title_idle: '組み込みのデフォルトモデルに戻します',
      menu_deploy: 'デプロイ',
      menu_export: 'ALPKG としてエクスポート',
      menu_exporting: 'エクスポート中…',
      menu_delete: '削除',
      menu_hint_active: 'アクティブ',
      menu_hint_deployed: 'デプロイ済み',
      error_deploy_head: 'モデルをデプロイできませんでした',
      error_export_head: 'モデルをエクスポートできませんでした',
      error_deploy_default: 'デフォルトモデルをデプロイできませんでした'
    },
    head_row: {
      pill_latest: '最新',
      pill_latest_title: 'ワークスペースの現在のリビジョンでトレーニングされた最新モデルです。',
      pill_active: 'アクティブ',
      pill_active_title: 'このモデルは現在、推論パイプラインにデプロイされています。',
      // Fixed-width single-string meta for the model-card popover and delete-confirm card.
      meta_line: (size, classCount, rev, relative) =>
        `${size} · ${classCount} クラス · rev ${rev} · ${relative}`,
      // Row meta renders segment-by-segment so size/rev can drop as the row narrows (size/age come
      // from format utils, not the catalog).
      meta_classes: (classCount) => `${classCount} クラス`,
      meta_rev: (rev) => `rev ${rev}`,
      row_aria_deployed: (shortId) => `デプロイ済みモデル ${shortId}`,
      row_aria_deploy: (shortId) => `モデル ${shortId} をデプロイ`,
      row_title_deployed: 'このモデルはすでにデプロイされています',
      row_title_deploying: 'デプロイ中…',
      row_title_exporting: 'エクスポート中…',
      row_title_busy: 'このリスト内の別のモデルがビジー状態です',
      row_title_idle: 'クリックしてこのモデルを推論パイプラインにホットスワップします',
      export_title_exporting: 'エクスポート中…',
      export_title_idle: 'このモデルを ALPKG アーカイブとしてエクスポート',
      export_aria_exporting: (shortId) => `モデル ${shortId} をエクスポート中`,
      export_aria_idle: (shortId) => `モデル ${shortId} をエクスポート`,
      info_title: 'モデルカードを表示',
      info_aria: (shortId) => `${shortId} のモデルカードを表示`
    },
    inference_preview: {
      heading: 'プレビュー',
      off_title: 'プレビューはオフです',
      off_description:
        'プレビューを開始すると、デプロイ済みモデルのスペクトログラムと top-k ストリームを確認できます。',
      start_button: 'プレビューを開始'
    },
    info_dialog: {
      title_with_id: (shortId) => `モデルカード · ${shortId}`,
      loading: 'クラスを読み込み中…',
      error_title: 'クラスを読み込めませんでした',
      retry: '再試行',
      classes_heading: 'クラス',
      class_labels_aria: 'トレーニング済みのクラスラベル'
    },
    delete_dialog: {
      title: 'このモデルを削除しますか？',
      body: 'トレーニング済みモデルのバイト列とそのマニフェストを削除します。データセットやその他のモデルは残ります。この操作は元に戻せません。',
      submit: '削除'
    }
  },
  workspace: {
    list: {
      title: 'ワークスペース',
      at_cap_subtitle: (max) =>
        `ワークスペース上限の ${max} 件に達しました。もう 1 つ作成する前に 1 つ削除してください。`,
      default_subtitle:
        '各ワークスペースには、ラベル付きデータセットとそこからトレーニングされたモデルが保持されます。',
      daemon_unavailable_title: 'デバイスを利用できません',
      loading: 'ワークスペースを読み込み中…',
      empty_title: 'まだワークスペースがありません',
      empty_description:
        'ワークスペースは、録音、ラベル付きサンプル、トレーニング済みモデルが格納される場所です。1 つ作成して始めましょう。',
      selected_count_aria: (count) => `${count} 件選択済み`,
      new_button_label: '新規ワークスペース',
      new_button_aria: '新規ワークスペース',
      new_at_cap_label: (count, max) => `上限到達 · ${count}/${max}`,
      new_at_cap_title: '上限に達しました。先にワークスペースを 1 つ削除してください。',
      import_button_label: 'インポート',
      import_button_aria: 'ワークスペースをインポート',
      import_button_title: 'ALPKG または TFJS バンドルからワークスペースをインポート',
      select_button_label: '選択',
      done_button_label: '完了',
      select_all_label: 'すべて選択',
      deselect_all_label: 'すべて選択解除',
      bulk_delete_label_count: (count) => `${count} 件を削除`,
      bulk_delete_label_bare: '削除',
      bulk_delete_aria_count: (count) => `${count} 件のワークスペースを削除`,
      bulk_delete_aria_fallback: '選択したワークスペースを削除',
      menu_open: '開く',
      menu_rename: '名前を変更',
      menu_export: 'エクスポート',
      menu_delete: '削除',
      menu_select_one: '選択',
      menu_deselect_one: '選択を解除',
      menu_select_all: 'すべて選択',
      menu_deselect_all: 'すべて選択解除',
      menu_select_workspaces: 'ワークスペースを選択',
      menu_done_exit: '完了（選択を終了）',
      menu_new: '新規ワークスペース',
      menu_new_at_cap: (max) => `新規ワークスペース（上限 ${max} 件に到達）`,
      menu_import: 'ワークスペースをインポート'
    },
    detail: {
      back_link: '← ワークスペース',
      loading: 'ワークスペースを読み込み中…',
      not_found_title: 'ワークスペースが見つかりません',
      not_found_description:
        '別のタブで、またはデバイスから直接削除された可能性があります。リストに戻って残っているものを確認してください。',
      back_to_list_button: 'ワークスペースに戻る',
      load_error_title: 'このワークスペースを読み込めませんでした',
      created_label: (relative) => `${relative}に作成`,
      rev_label: (rev) => `rev ${rev}`,
      modified_label: (relative) => `${relative}に変更`,
      live_pill_title:
        '最近のアップロードで進みました。変更タイムスタンプを更新するには再読み込みしてください。',
      live_pill: 'ライブ',
      menu_rename: '名前を変更',
      menu_export: 'エクスポート',
      menu_import: 'インポート',
      menu_delete: '削除',
      menu_back_to_list: 'ワークスペースに戻る'
    },
    create_dialog: {
      title: '新規ワークスペース',
      name_label: '名前',
      name_placeholder: 'my-workspace',
      name_help:
        '最大 128 文字。スラッシュや制御文字は使えません。名前は唯一の表示識別子なので、覚えやすいものを選んでください。',
      submit: '作成'
    },
    rename_dialog: {
      title: 'ワークスペースの名前を変更',
      name_label: '名前',
      name_help:
        '最大 128 文字。スラッシュや制御文字は使えません。名前の変更でワークスペースのリビジョンは進みません — カテゴリ、スライス、モデルはそのまま維持されます。',
      submit: '保存'
    },
    delete_dialog: {
      title: 'このワークスペースを削除しますか？',
      body: 'データセット、トレーニング済みモデル、ログを削除します。この操作は元に戻せません。',
      submit: '削除'
    },
    bulk_delete_dialog: {
      title_count: (count) => `${count} 件のワークスペースを削除しますか？`,
      body: '各ワークスペースのデータセット、トレーニング済みモデル、ログを削除します。この操作は元に戻せません。',
      submit_count: (count) => `${count} 件を削除`
    },
    tool_island: {
      aria_label: 'ワークスペースのアクション',
      rename_aria: 'ワークスペースの名前を変更',
      rename_title: 'ワークスペースの名前を変更',
      export_aria: 'ワークスペースをエクスポート',
      export_title: 'ワークスペースをエクスポート（データセット + モデル）',
      import_aria: 'ワークスペースをインポート',
      import_title: 'ワークスペースをインポート（データセット + モデル）'
    },
    card: {
      created_label: (relative) => `${relative}に作成`,
      select_aria: (name) => `ワークスペース ${name} を選択`,
      rename_aria: (name) => `ワークスペース ${name} の名前を変更`,
      deleting: '削除中'
    },
    import_dialog: {
      title_into: (workspaceName) => `インポート先 · ${workspaceName}`,
      title_fallback: 'インポート',
      step_indicator: (current, total) => `ステップ ${current} / ${total}`,
      pipeline_error_title: 'インポートに失敗しました',
      error_invalid_state: 'ダイアログの状態が不整合です — インポートするアーカイブがありません。',
      pick_file: {
        drop_zone_title_attr:
          'ALPKG アーカイブまたは TFJS バンドルをここにドロップ、またはクリックして参照',
        reading: '読み込み中…',
        drop_zone_tfjs_staging: 'TFJS バンドルを完成させるにはさらにファイルをドロップしてください',
        drop_zone_idle: 'ALPKG アーカイブまたは TFJS バンドルをここにドラッグ＆ドロップ',
        browse_button: 'ファイルを参照',
        error_empty_drop: 'ALPKG アーカイブまたは TFJS バンドルをドロップしてください。',
        error_multi_alpkg: (count) =>
          `ALPKG アーカイブは一度に 1 つだけ選択してください — ${count} 件選択されました。`,
        error_mixed_archive:
          'ALPKG アーカイブは単独で選択する必要があり、他のファイルと混在させることはできません。',
        error_file_count_cap: (max, picked) =>
          `一度にドロップまたは選択できるのは最大 ${max} ファイルです — ${picked} 件選択されました。`,
        error_single_too_large: (name, size, cap) =>
          `「${name}」は ${size} です — ファイルごとの上限は ${cap} です。`,
        error_total_too_large: (total, cap) =>
          `選択範囲は合計 ${total} です — 1 回のドロップの上限は ${cap} です。`,
        error_tfjs_merged_file_count: (mergedCount, cap) =>
          `ステージングしたセットは合計 ${mergedCount} ファイルになります — 上限は ${cap} です。クリアして、より小さなバンドルを再ドロップしてください。`,
        error_tfjs_merged_bytes: (mergedBytes, cap) =>
          `ステージングしたセットは合計 ${mergedBytes} になります — 上限は ${cap} です。クリアして、より小さなバンドルを再ドロップしてください。`,
        staged_files_heading: 'ステージング済みファイル',
        staged_files_count: (count) => `${count} ファイル`,
        clear_button: 'クリア',
        error_could_not_read_archive: 'アーカイブを読み込めませんでした。',
        error_could_not_read_file: 'ファイルを読み込めませんでした。',
        error_could_not_read_picked_files: '選択したファイルを読み込めませんでした。',
        error_could_not_read_model_json: 'model.json を読み込めませんでした。',
        tfjs_diag_empty_drop:
          'TFJS バンドルのファイル（model.json + シャード + ラベル）をドロップしてください。',
        tfjs_diag_no_model_json:
          'ドロップ内に "model.json" がありません。TFJS マニフェストを含めてください。',
        tfjs_diag_ambiguous_model_json: (count) =>
          `バンドルが不明確です：${count} 個のファイルが "model.json" という名前です。`,
        tfjs_diag_multiple_labels_txt:
          'ドロップ内に "labels.txt" ファイルが複数あります。1 つだけ含めてください。',
        tfjs_diag_multiple_metadata_json:
          'ドロップ内に "metadata.json" ファイルが複数あります。1 つだけ含めてください。',
        tfjs_diag_both_labels:
          '"labels.txt" と "metadata.json" の両方が提供されました。ラベルソースは 1 つだけ含めてください。',
        tfjs_diag_no_labels:
          'ラベルファイルが提供されていません。"labels.txt" または "metadata.json" を含めてください。',
        tfjs_diag_shard_collision_one: (quotedName) =>
          `2 つのステージング済みファイルがシャード名 ${quotedName} を共有しています。ステージングをクリアし、目的のコピーだけをドロップしてください。`,
        tfjs_diag_shard_collision_many: (quotedNames, overflow) =>
          `複数のステージング済みファイルが "model.json" の参照するシャード名を共有しています：${quotedNames}${overflow ? '…' : ''}。ステージングをクリアし、目的のコピーだけをドロップしてください。`,
        tfjs_diag_missing_shard_one: (quotedName) =>
          `"model.json" が参照するシャード ${quotedName} がありません。`,
        tfjs_diag_missing_shards_many: (count, quotedNames, overflow) =>
          `"model.json" が参照するシャードが ${count} 個ありません：${quotedNames}${overflow ? '…' : ''}。`,
        tfjs_diag_model_json_not_json: 'model.json は有効な JSON ではありません。',
        tfjs_diag_model_json_not_object: 'model.json は JSON オブジェクトではありません。',
        tfjs_diag_model_json_no_manifest: 'model.json に "weightsManifest" 配列がありません。',
        tfjs_diag_model_json_no_shards: 'model.json はシャードファイルを宣言していません。'
      },
      pick_target: {
        section_label: 'インポート先',
        mode_radio_aria: 'ターゲットワークスペースのモード',
        mode_use_existing: '既存を使用',
        mode_create_new: '新規作成',
        no_workspaces_prefix: 'まだワークスペースがありません — ',
        no_workspaces_link_label: '新規作成',
        no_workspaces_suffix: ' に切り替えて作成してください。',
        workspace_list_aria: 'ターゲットワークスペースを選択',
        workspace_created_label: (relative) => `${relative}に作成`,
        create_name_placeholder: 'my-imported-workspace',
        create_will_carry_tags: (tagsCsv) => `ソースからタグを引き継ぎます：${tagsCsv}`,
        alpkg_source_card_title: (name, id) => `${name} (${id})`,
        alpkg_source_created_label: (relative) => `${relative}に作成`,
        alpkg_source_rev_label: (rev) => `rev ${rev}`,
        alpkg_source_modified_label: (relative) => `${relative}に変更`,
        tfjs_bundle_card_title: 'TFJS バンドル',
        tfjs_show_labels_aria: 'クラスラベルを表示',
        tfjs_meta_strip: (size, shards, classes, labelsFileName) => {
          const classesPart = classes !== null && classes > 0 ? ` · ${classes} クラス` : '';
          const shardsPart = ` · ${shards} シャード`;
          const labelsPart = labelsFileName !== null ? ` · ${labelsFileName} 経由` : '';
          return `${size}${classesPart}${shardsPart}${labelsPart}`;
        }
      },
      summary: {
        datasets_heading: 'データセット',
        datasets_counter: (selected, total) => `選択済み ${selected} / ${total}`,
        checking_categories: 'ターゲットワークスペースの既存カテゴリを確認中…',
        slice_count: (count) => `${count} スライス`,
        rename_button_aria: 'ターゲットカテゴリの名前を変更',
        rename_button_title_default: 'ターゲットカテゴリの名前を変更',
        mode_aria: (modeLabel) => `インポート操作：${modeLabel}`,
        mode_menu_aria: (sourceName) => `${sourceName} のインポート操作`,
        rename_popover_aria: (sourceName) => `${sourceName} のターゲットカテゴリの名前を変更`,
        rename_popover_heading: '名前を変更',
        rename_chips_heading: 'または既存を再利用',
        heads_heading: 'モデル',
        heads_cap_tooltip: (cap) =>
          `ワークスペースごとに最大 ${cap} 件のモデル。新しいモデルが追加されると — 再トレーニングまたはインポートを問わず — 古い非アクティブモデルがロールオフされます。`,
        heads_counter: (selected, existingInTarget, cap, activeInTarget) => {
          const active = activeInTarget > 0 ? ` · アクティブ ${activeInTarget} 件固定` : '';
          return `選択済み ${selected} · ターゲット ${existingInTarget} / ${cap}${active}`;
        },
        checking_heads: 'ターゲットモデルを確認中…',
        displacement_warning: (displaced, cap) =>
          `インポートすると、${cap} 件のモデル上限に収めるため、最も古い非アクティブモデル ${displaced} 件が押し出されます。`,
        head_exists_badge_title: 'この ID のモデルはターゲットワークスペースにすでに存在します。',
        head_exists_badge: '既存',
        head_show_details_aria: 'モデルの詳細を表示',
        head_class_count: (count) => `${count} クラス`,
        head_info_metadata: (size, classes, revisionId, createdAbsolute, createdRelative) => {
          const classesPart = classes !== null ? ` · ${classes} クラス` : '';
          const revPart = revisionId !== null ? ` · rev ${revisionId}` : '';
          const createdPart =
            createdAbsolute !== null && createdRelative !== null
              ? ` · ${createdAbsolute} (${createdRelative})`
              : '';
          return `${size}${classesPart}${revPart}${createdPart}`;
        },
        head_classes_heading: 'クラス',
        head_class_labels_aria: 'トレーニング済みのクラスラベル',
        archive_errors_summary: (count) => `アーカイブエントリ ${count} 件をスキップしました`,
        tfjs_ignored_unknown: (count, fileList) =>
          `認識できないファイル ${count} 件を無視しました：${fileList}`,
        tfjs_classes_popover_heading: (count) => `クラス (${count})`,
        tfjs_classes_popover_aria: 'クラスラベル',
        head_disabled_reasons: {
          loading: 'ターゲットモデルを読み込み中…',
          exists: 'ターゲットにすでに存在します。別のモデルを選択してください。',
          ceiling: '選択上限に達しました。先に別の行のチェックを外してください。'
        }
      },
      modes: {
        new: '新規',
        merge: 'マージ',
        replace: '置換',
        skip: 'スキップ'
      },
      mode_tooltips: {
        new: 'アーカイブのスライスでカテゴリを一から作成します。',
        merge:
          'アーカイブのスライスを既存のカテゴリに上乗せしてアップロードします。同一 SHA256 のスライスは自身を上書きし、新しいものはセットに追加されます。',
        replace:
          '既存のカテゴリ（およびそれが保持するすべてのスライス）を削除してから、アーカイブからアップロードします。',
        skip: 'このカテゴリをインポートしません。'
      },
      mode_disabled_reasons: {
        new_exists:
          'このターゲット名のカテゴリはすでに存在します。スライスを追加するには「マージ」を、消去して再インポートするには「置換」を選択してください。',
        merge_missing:
          'このターゲット名の既存カテゴリはありません。作成するには「新規」を選択してください。',
        replace_missing:
          'このターゲット名の既存カテゴリはありません。作成するには「新規」を選択してください。'
      },
      running: {
        progress_replacing_categories: (categoryDisplay, done, total) => {
          const cat = categoryDisplay !== null ? ` · ${categoryDisplay}` : '';
          return `カテゴリを置換中${cat} · ${done} / ${total}`;
        },
        progress_uploading_datasets: (categoryDisplay, done, total) => {
          const cat = categoryDisplay !== null ? ` · ${categoryDisplay}` : '';
          if (typeof done === 'number' && typeof total === 'number') {
            return `スライスをアップロード中${cat} · ${done} / ${total}`;
          }
          return `スライスをアップロード中${cat}`;
        },
        progress_importing_heads: (index1, total, subPhase) => {
          const sub = subPhase !== null ? ` (${subPhase})` : '';
          return `モデルをインポート中 ${index1} / ${total}${sub}`;
        },
        progress_uploading_tfjs: (done, total) =>
          `TFJS ファイルをアップロード中 · ${done} / ${total}`,
        progress_converting_tfjs: 'TFJS バンドルを変換中…',
        ds_pending: '保留中',
        ds_replacing: '置換中',
        ds_uploading_counter: (uploaded, total) => `${uploaded} / ${total}`,
        ds_done_uploaded: (uploaded) => `${uploaded} 件アップロード済み`,
        ds_failed_count: (failed) => `${failed} 件失敗`,
        ds_failed_label: '失敗',
        ds_failed_title_count: (failed) => `${failed} スライスのアップロードに失敗しました`,
        head_queued: 'キュー待ち',
        head_skipped_badge_title:
          'モデル ID はディスク上にすでに存在し、オーケストレーターがスキップしました（冪等な再インポート）。',
        head_per_log_not_started:
          'まだ開始されていません — このモデルのインポートが始まるとログ行が表示されます。',
        head_per_log_no_events: '記録されたイベントはありません。',
        log_count: (count) => `${count} 件`
      },
      head_phase: {
        queued: 'キュー待ち',
        uploading_files: 'ファイルをアップロード中',
        starting_convert: '変換を開始中',
        converting: '変換中',
        cleaning_up: 'クリーンアップ中',
        done: '完了',
        failed: '失敗'
      },
      head_outcome: {
        imported: 'インポート済み',
        replaced: '置換済み',
        skipped: 'スキップ済み',
        failed: '失敗'
      },
      convert_stage: {
        prepare: '準備中',
        read_manifest: 'マニフェストを読み込み中',
        validate_manifest: 'マニフェストを検証中',
        verify_mpk: 'MPK を検証中',
        stage_mpk: 'MPK をステージング中',
        read_model_json: 'model.json を読み込み中',
        stage_shards: 'シャードをステージング中',
        extract_weights: '重みを抽出中',
        read_labels: 'ラベルを読み込み中',
        stage_head_mpk: 'モデル MPK をステージング中',
        publish_head: 'モデルを発行中'
      },
      convert_event: {
        job_submitted: (converter) => `${converter} 経由でジョブを送信しました`,
        job_running: 'ジョブ実行中',
        phase: (stageLabel) => `フェーズ：${stageLabel}`,
        manifest_validated: (classes) => `マニフェストを検証 · ${classes} クラス`,
        mpk_verified: (size) => `MPK を検証 · ${size}`,
        weights_extracted: (classes, inDim) => `重みを抽出 · ${classes} クラス · ${inDim} in_dim`,
        labels_loaded: (labels) => `ラベルを読み込み · ${labels} 個`,
        head_published: (idempotentSkip) =>
          `モデルを発行${idempotentSkip ? '（すでにディスク上にあるためスキップ）' : ''}`,
        job_completed: (classes) => `ジョブ完了 · ${classes} クラス`,
        job_failed: (category, error) => `ジョブ失敗 · ${category} · ${error}`
      },
      done: {
        conflict_detail: (storedSha8, incomingSha8) =>
          `ターゲットにはこの ID で SHA256 が異なるモデルがすでに存在します（${storedSha8} に対し受信側は ${incomingSha8}）。`,
        retry_button: '既存を置換して再試行'
      },
      footer: {
        cancel: 'キャンセル',
        back: '戻る',
        next: '次へ',
        import: 'インポート',
        importing: 'インポート中…',
        back_to_selection: '選択に戻る',
        done: '完了'
      }
    },
    export_dialog: {
      title: (workspaceName) => `ワークスペースをエクスポート · ${workspaceName}`,
      load_error_title: 'このワークスペースを読み込めませんでした',
      loading: 'ワークスペースを読み込み中…',
      nothing_to_export:
        'このワークスペースにはまだカテゴリもモデルもありません — エクスポートする内容がありません。',
      datasets_heading: 'データセット',
      heads_heading: 'モデル',
      select_all: 'すべて選択',
      deselect_all: 'すべて選択解除',
      row_empty: '空',
      row_slice_count: (count) => `${count} スライス`,
      head_meta_title: (size, classCount) => `${size} · ${classCount} クラス`,
      head_meta_classes: (count) => `${count} クラス`,
      pending_warning:
        '選択範囲内でまだアップロード中または保留中のスライスは除外されます — ディスク上のスライスのみが含まれます。',
      progress_preparing_workspace: 'ワークスペースのメタデータを読み込み中…',
      progress_fetching_slices: 'スライスを取得中…',
      progress_listing_slices: 'スライスを一覧中…',
      progress_fetched_slices: (done, total) => `${done} / ${total} スライスを取得済み…`,
      progress_validating_heads: 'モデルを検証中…',
      progress_validated_heads: (done, total) => `${done} / ${total} モデルを検証済み…`,
      progress_packing: 'アーカイブをパッキング中…',
      progress_downloading: 'ダウンロードを開始中…',
      error_default: 'エクスポートに失敗しました',
      error_in_category: (categoryDisplay) => `「${categoryDisplay}」でエクスポートに失敗しました`,
      error_for_head: (shortId) => `モデル ${shortId} のエクスポートに失敗しました`,
      exporting: 'エクスポート中…',
      export_aria: '選択した項目をエクスポート',
      export_button: 'エクスポート'
    }
  }
} satisfies Messages;
