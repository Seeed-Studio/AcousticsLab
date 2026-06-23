import type { Messages } from '../types';

// 简体中文 (zh-CN). Mirrors en.ts 1:1 (keys, params, ${...}, comments); see en.ts for conventions. zh
// style: full-width punctuation; keep ' · ', '…', numbers, units, tokens, ${...} ASCII; break dash
// '——'; 'Background Noise' stays literal.
export const zh = {
  app: {
    name: 'AcousticsLab',
    description: '私有、多后端、完全本地运行的 AI/ML 工具包，用于开发和部署实时声音事件检测。'
  },
  routes: {
    dashboard_title: (brand) => brand,
    workspace_list_title: (brand) => `工作区 · ${brand}`,
    workspace_detail_title: (workspaceName, brand) => `${workspaceName} · ${brand}`
  },
  nav: {
    dashboard: '仪表盘',
    workspaces: '工作区',
    home_aria: 'AcousticsLab 主页',
    menu_fallback: '菜单',
    primary_nav_aria: '主导航'
  },
  dashboard: {
    limited_support_title: '浏览器支持有限',
    visualization_panel: {
      heading: '可视化',
      // Dot-separated segments so codec/channels can drop on a narrow header; rate and window always stay.
      audio_sample_rate: '48 kHz',
      audio_channels: '单声道',
      audio_codec: 'opus',
      audio_window: '3 s 窗口'
    },
    inference_panel: {
      heading: '推理'
    },
    configuration_panel: {
      heading: '配置'
    },
    configuration_controls: {
      daemon_unavailable_title: '设备不可用',
      daemon_unavailable_default: '当设备可达时，配置将自动恢复。',
      microphone_heading: '麦克风',
      source_label: '输入源',
      auto_first_available: '自动 · 第一个可用',
      channel_label: '声道',
      auto_channel: '自动',
      inference_cadence_heading: '推理节奏',
      overlap_ratio_label: '重叠比例',
      top_k_label: 'Top-K',
      loading: '加载中…',
      kind_alsa: 'ALSA',
      kind_unknown: '未知',
      approx_hz: (hz) => `${hz} Hz`,
      khz: (khz) => `${khz} kHz`,
      hz: (rate) => `${rate} Hz`
    },
    top_k_meter: {
      awaiting_first_frame: '等待首个推理帧…'
    },
    active_head_card: {
      heading: '活动模型',
      pill_default: '默认',
      pill_workspace: '工作区',
      pill_detached: '已分离',
      pill_default_title: '内置的默认模型正在运行。',
      pill_workspace_title: '已训练的工作区模型正在运行。',
      pill_detached_title: '该模型激活后，其来源工作区已删除。',
      loading_active: '正在加载活动模型…',
      activated_label: '激活时间',
      class_count_label: () => '类别',
      workspace_dt: '工作区',
      revision_dt: '修订',
      rev_value: (rev) => `rev ${rev}`,
      deleted_tag: '（已删除）',
      loading: '加载中…',
      ws_title_orphaned_with_name: (name, uuid) => `${name} · ${uuid}（工作区已删除）`,
      ws_title_orphaned: (uuid) => `${uuid}（工作区已删除）`,
      ws_title_with_name: (name, uuid) => `${name} · ${uuid}`
    }
  },
  theme: {
    label: '主题',
    label_with_current: (currentLabel) => `主题：${currentLabel}`,
    options: {
      auto: '自动',
      light: '浅色',
      dark: '深色'
    }
  },
  locale: {
    label: '语言',
    label_with_current: (currentChip) => `语言：${currentChip}`
  },
  health: {
    aria_label: '系统健康',
    levels: {
      unknown: '连接中',
      ok: '正常',
      degraded: '降级',
      unhealthy: '异常',
      unreachable: '不可达'
    },
    popover: {
      daemon_unreachable_title: '设备不可达',
      waiting_first_snapshot: '正在等待首个状态快照…',
      subsystems_heading: '子系统',
      seconds_ago: (seconds) => `${seconds}s 前`,
      stat_cpu_label: 'CPU',
      stat_rss_label: 'RSS',
      stat_disk_free_label: '可用磁盘',
      uptime_label: '运行时间',
      dropped_count: (count) => `丢弃：${count}`
    }
  },
  common: {
    cancel: '取消',
    dismiss: '忽略'
  },
  error: {
    another_train_running: '此设备上已有另一个训练任务正在运行。',
    another_convert_running: '此设备上已有另一个转换任务正在运行。',
    job_conflict: '该资源上已有另一项操作正在进行中。',
    event_gap: '事件流出现跳跃，需要从日志中追赶。正在重新连接…',
    too_early: '设备仍在应用上一次更改。正在重试…',
    unavailable: '设备暂时不可用。请稍后重试。',
    internal: '守护进程遇到内部错误。请重试。若持续出现，请检查守护进程日志。',
    unknown: '出现错误。请重试。',
    something_went_wrong: '出现错误。',
    request_failed: (code) => `请求失败（${code}）。`
  },
  validation: {
    name: {
      empty: '名称不能为空。',
      max_bytes: (max) => `名称不能超过 ${max} 字节。`,
      slashes_or_nul: '名称不能包含斜杠或 NUL 字节。',
      starts_or_ends_whitespace: '名称不能以空白字符开头或结尾。',
      control_chars: '名称不能包含控制字符。',
      starts_with_dot: '类别名称不能以点号开头。',
      starts_with_underscore: '类别名称不能以下划线开头（保留给内置类别）。',
      starts_with_hyphen: '类别名称不能以连字符开头（出于安全性考量）。',
      bad_chars: '只允许使用字母、数字、点号、连字符和下划线。',
      category_max_bytes: (max) => `类别名称不能超过 ${max} 字节。`,
      category_empty: '类别名称不能为空。'
    },
    cfg: {
      epochs_whole: '轮次必须为整数。',
      epochs_range: (min, max) => `轮次必须介于 ${min} 和 ${max} 之间。`,
      batch_whole: '批大小必须为整数。',
      batch_range: (min, max) => `批大小必须介于 ${min} 和 ${max} 之间。`,
      lr_finite: '学习率必须为有限数值。',
      lr_greater_than_zero: '学习率必须大于 0。',
      lr_max: (max) => `学习率不能超过 ${max}。`,
      seed_whole: '随机种子必须为整数。',
      seed_non_negative: '随机种子必须大于或等于 0。',
      seed_too_large: '随机种子过大。',
      split_finite: '验证集比例必须为有限数值。',
      split_min: '验证集比例必须大于或等于 0。',
      split_max: (max) => `验证集比例不能超过 ${max}。`
    }
  },
  streams: {
    socket_status: {
      connecting: '连接中',
      open: '实时',
      closed: '已断开',
      error: '错误'
    }
  },
  recorder: {
    mic_error_denied: '麦克风访问被拒绝。请在浏览器设置中允许麦克风访问，然后重试。',
    mic_error_not_found: '未找到麦克风。请连接麦克风后重试。',
    mic_error_in_use: '麦克风正被其他应用占用。请关闭该应用，然后重试。',
    mic_error_interrupted: '麦克风采集被中断。请重试。',
    mic_error_generic: '无法启动麦克风。请重试。'
  },
  category: {
    list: {
      heading: '数据集',
      description: '每个类别都会成为训练器学习的类别标签——Background Noise 为必需项。',
      add_button: '添加类别',
      add_button_aria: '添加类别',
      loading: '正在加载类别…',
      load_error: (error) => `无法加载类别。${error}`,
      menu_delete: '删除',
      menu_hint_preserved: '内置保留',
      menu_rename: '重命名',
      menu_rename_hint_busy: '请先完成进行中的操作',
      menu_add: '添加类别'
    },
    add_dialog: {
      title: '添加类别',
      name_label: '名称',
      name_placeholder: '例如 cat',
      name_help_prefix: '字母、数字、点、连字符和下划线。该名称同时用作磁盘上的目录名（例如 ',
      name_help_code_example: 'datasets/cat/',
      name_help_suffix: '），并作为训练器使用的类别标签。',
      submit: '添加',
      error_exact_duplicate: '已存在同名类别。',
      error_case_insensitive_duplicate: (existingName) =>
        `与现有的 "${existingName}" 冲突（在大多数文件系统上名称不区分大小写）。`
    },
    rename_dialog: {
      title: '重命名类别',
      name_label: '名称',
      name_help:
        '该名称同时用作磁盘上的目录名和训练器的类别标签，因此重命名会更改类别标签。已训练的模型仍保留旧标签，并标记为已过期，直到重新训练。',
      submit: '保存',
      error_mandatory: 'Background Noise 为内置保留项，无法重命名。',
      error_busy: '在重命名此类别前，请先完成或清除进行中的上传和删除操作。'
    },
    delete_dialog: {
      title: '删除此类别？',
      body_server: '将移除数据集文件夹及其中的每个切片。此操作无法撤销。',
      body_idb: '将从本地列表中移除此类别。由于尚未上传任何切片，设备不会发生任何更改。',
      submit: '删除',
      error_fallback: '无法删除该类别。',
      error_mandatory_required: 'Background Noise 为内置保留项，无法删除。',
      error_not_found: '未找到类别。'
    },
    slice_card: {
      aria_select: (filename) => `选择切片 ${filename}`,
      aria_deselect: (filename) => `取消选择切片 ${filename}`,
      aria_play: (filename) => `播放切片 ${filename}`,
      title_failed: (errorOrUnknown) => `上传失败：${errorOrUnknown}。右键单击以重试。`,
      title_uploading: (progressPct) => `上传中… ${progressPct}%`,
      title_local: '本地——等待上传',
      title_multi_click_deselect: '单击以取消选择（Esc 退出选择）',
      title_multi_click_select: '单击以添加到选区（Esc 退出选择）',
      title_playing: '播放中——单击以重新播放',
      title_idle: '单击以播放（Ctrl/Cmd 单击以选择）',
      sr_deleting: (filename) => `正在删除切片 ${filename}`,
      sr_uploading: (progressPct) => `上传中 ${progressPct}%`,
      retry_aria: (filename) => `重试上传切片 ${filename}`,
      retry_title_with_error: (errorMessage) => `上传失败：${errorMessage}。单击以重试。`,
      retry_title_no_error: '上传失败。单击以重试。',
      retry_label: '重试',
      select_title: '选择',
      deselect_title: '取消选择',
      delete_aria: (filename) => `删除切片 ${filename}`,
      delete_title: '删除切片',
      slice_select_aria: (filename) => `选择切片 ${filename}`,
      slice_deselect_aria: (filename) => `取消选择切片 ${filename}`,
      unknown_error: '未知错误'
    },
    trim_waveform: {
      handles_aria: '裁剪手柄，拖动以设置切片范围的起点和终点',
      handle_start_aria: '裁剪起点',
      handle_end_aria: '裁剪终点',
      selection_aria: '滑动选区窗口，拖动以同时移动两侧裁剪边缘',
      playback_position_aria: '播放位置',
      value_seconds: (sec) => `${sec} 秒`,
      value_seconds_range: (startSec, endSec) => `${startSec} 至 ${endSec} 秒`
    },
    slice_pane: {
      heading: '切片',
      tips_label: '切片模块提示',
      tip_audition_title: '在训练前试听每个切片。',
      tip_audition_body: '一行标注错误的数据会使整个类别产生偏差——单击卡片即可播放，请大胆丢弃。',
      tip_diversity_title: '多样性胜过数量。',
      tip_diversity_body:
        '10 段多样化的录音（距离、角度、背景）比 30 个几乎相同的副本训练效果更好。',
      quota_above_title: (threshold) => `高于训练所需的 ${threshold} 个切片最低要求。`,
      quota_below_title: (threshold) =>
        `低于训练所需的 ${threshold} 个切片最低要求。请切分更多切片以满足配额。`,
      loading: '正在加载切片…',
      load_error: (error) => `无法加载切片。${error}`,
      empty_state_prefix: '尚无切片。在输入面板中裁剪片段，然后单击 ',
      empty_state_button: '切分',
      empty_state_suffix: ' 以填充此网格。',
      select_all_label: '全选',
      deselect_all_label: '取消全选',
      select_all_title: '全选所有切片（Cmd/Ctrl+A）',
      deselect_all_title: '取消全选所有切片（Cmd/Ctrl+A）',
      done_label: '完成',
      done_title: '退出选择（Esc）',
      delete_title: '删除选中的切片（Del / Backspace）',
      delete_disabled_title: '请至少选择一个切片以删除',
      delete_inflight_title: (count) => `正在删除 ${count} 个切片…`,
      delete_inflight_aria: (count) => `正在删除 ${count} 个切片`,
      delete_aria_count: (count) => `删除 ${count} 个选中的切片`,
      delete_aria_fallback: '删除选中的切片',
      delete_label_inflight: (count) => `正在删除 ${count} 个…`,
      delete_label_count: (count) => `删除 ${count} 个`,
      delete_label_bare: '删除',
      menu_play: '播放',
      menu_stop: '停止',
      menu_retry_upload: '重试上传',
      menu_select: '选择',
      menu_deselect: '取消选择',
      menu_select_all: '全选',
      menu_deselect_all: '取消全选',
      menu_done_exit: '完成（退出选择）',
      menu_retry_failed_in_selection: '重试选区中失败的项',
      menu_delete_batch: (count) => `删除 ${count} 个切片`,
      menu_delete: '删除',
      menu_hint_a: 'Cmd/Ctrl+A',
      menu_hint_esc: 'Esc',
      menu_hint_ctrl_click: 'Ctrl/Cmd 单击',
      menu_hint_del_backspace: 'Del / Backspace'
    },
    input_pane: {
      heading: '输入',
      tips_label: '输入模块提示',
      tip_stream_title: '优先使用设备的声音流。',
      tip_stream_body: '这些切片与推理共享相同的 DSP，因此微调后训练出的模型不会遇到分布偏移。',
      tip_environment_title: '在部署环境中录制。',
      tip_environment_body:
        '干净的录音棚采集会使噪声抑制训练不足。真实的背景是模型需要学习内容的一半。',
      tip_meter_title: '使仪表保持在绿色到琥珀色之间。',
      tip_meter_body: '玫红色表示削波，会抹去训练器无法恢复的信息。',
      pane_aria: (categoryDisplay) => `类别 ${categoryDisplay} 的输入模块`,
      source_aria: '输入源',
      loudness_aria: '响度仪表',
      source_microphone_group: '麦克风',
      source_system_default_mic: '系统默认麦克风',
      source_remembered: (label) => `${label}（已记住）`,
      source_mic_fallback: (n, idFrag) => `麦克风 ${n}（${idFrag}）`,
      source_mic_remembered_fallback: (idFrag) => `麦克风（${idFrag}）`,
      source_mic_default_id: 'default',
      source_live_stream_group: '实时流',
      source_daemon_stream: '设备声音流',
      source_daemon_stream_with_status: (status) => `设备声音流 · ${status}`,
      drop_zone_title: (cap) => `将 WAV 文件拖放到此处（最大 ${cap}），或单击以浏览`,
      drop_zone_idle: '将 WAV 拖放到此处',
      drop_zone_browse: '浏览文件',
      record_aria_stream: '开始从实时声音流采集',
      record_aria_mic: '开始从麦克风录制',
      record_label: '录制',
      record_title_stream_open: (max) => `采集实时声音流（在 ${max} 时自动停止）。`,
      record_title_stream_connecting: '设备声音流正在连接。流打开后即可开始录制。',
      record_title_stream_closed: '设备声音流不可达。请检查设备是否正在运行。',
      record_title_stream_unsupported:
        '此浏览器无法在这里解码实时声音流——它需要在安全（HTTPS）上下文中使用 WebCodecs。请通过安全网关打开此页面，或改为拖放或浏览 WAV 文件。',
      capture_stop_aria_stream: '停止流采集',
      capture_stop_aria_mic: '停止录制',
      capture_stop_label: '停止',
      capture_discard_label: '丢弃',
      capture_encoding: '编码中…',
      capture_decoding: '解码中…',
      trim_selection_prefix: '选区：',
      trim_drag_hint: '将手柄拖动至 ≥ 1 s 以启用切分。',
      trim_projected_slices: (count) => `${count} 个切片，每个 1 s`,
      trim_unused_label: '未使用',
      slice_aria_enabled: (count) => `切分为 ${count} 个切片`,
      slice_aria_disabled: '切分（选区必须至少为 1 秒）',
      slice_title_enabled: (count) => `将 ${count} 个切片追加到右侧面板`,
      slice_title_disabled: '选区必须 ≥ 1 s 才能切分',
      slice_label_bare: '切分',
      slice_label_count: (count) => `切分 · ${count}`,
      discard_aria: '丢弃片段',
      discard_title: '丢弃片段',
      discard_label: '丢弃',
      play_stop_aria: '停止播放',
      play_stop_title: '停止播放',
      play_aria: '播放裁剪后的选区',
      play_title: '播放裁剪后的选区',
      export_aria: '下载为 WAV',
      export_title: '下载为 WAV',
      error_file_too_large: (size, cap) =>
        `文件大小为 ${size}——导入上限为 ${cap}。请裁剪得更短后重新导出，然后再次拖放。`,
      error_clip_too_short: (clipSecs) =>
        `片段仅 ${clipSecs} s，训练要求每个片段至少 1 s，因此较短的片段将被完全排除。请导入或录制时长为 1 s 或更长的片段。`,
      error_only_one_file: '一次只能处理一个文件——输入槽仅保留最近的片段。请拖放单个 WAV。',
      error_only_wav: '仅支持 WAV 文件。',
      error_could_not_import: '无法导入该文件。',
      error_could_not_discard: '无法丢弃该片段。',
      error_could_not_decode_draft: '无法解码已存储的草稿。',
      error_could_not_save_recording: '无法保存录音。',
      error_could_not_capture_stream: '无法采集该流。',
      error_could_not_slice: '无法切分该片段。',
      error_wav_too_small_for_header: '文件太小，不是有效的 WAV（标头至少需要 12 字节）。',
      error_wav_missing_riff: '不是 WAV 文件（缺少 RIFF 魔数）。',
      error_wav_missing_wave: '不是 WAV 文件（缺少 WAVE 标记）。',
      error_wav_empty: '文件为空或太小，不是有效的 WAV。',
      error_wav_buffer_too_small: 'WAV 缓冲区太小（标准标头至少需要 44 字节）。',
      error_web_audio_unavailable: '此浏览器中无法使用 Web Audio API。',
      auto_stopped_at_cap: '已在时长上限处自动停止。',
      silent_dropped_suffix: (count) => `已跳过 ${count} 个静音切片`
    },
    row: {
      badge_synced: '已同步',
      badge_uploading: '上传中',
      badge_pending: '待处理',
      badge_failed: '失败',
      badge_not_enough: '样本不足',
      badge_not_enough_with_state: (statusLabel) => `样本不足 · ${statusLabel}`,
      title_synced: (tally) => `${tally} 个切片已上传到设备——可用于训练。`,
      title_uploading: (tally) => `${tally} 个切片，部分仍在上传到设备。`,
      title_pending: (tally) => `${tally} 个切片已就绪但尚未上传到设备。`,
      title_failed: (tally) =>
        `${tally} 个切片，至少有一项上传失败。请从切片卡片重试或丢弃失败的行。`,
      title_not_enough_empty: (missing, tally) =>
        `再添加 ${missing} 个切片以满足每个类别的配额（${tally}）。`,
      title_not_enough_synced: (tally, missing) =>
        `${tally} 个切片已上传，再添加 ${missing} 个以满足每个类别的配额。`,
      title_not_enough_uploading: (tally, missing) =>
        `${tally} 个切片，部分仍在上传。完成后还需 ${missing} 个。`,
      title_not_enough_pending: (tally, missing) =>
        `${tally} 个切片已在本地排队，还需 ${missing} 个。`,
      actions_aria: (displayName) => `${displayName} 的操作`,
      actions_title: '类别操作',
      actions_title_preserved: '内置保留——重命名和删除已禁用',
      badge_deleting: '删除中'
    }
  },
  training: {
    pane: {
      heading: '训练',
      subtitle_other_running: '已有另一个工作区正在训练，同一时间只能运行一个任务。',
      subtitle_default: '在此工作区的数据集上调优模型，新模型就绪后会自动丢弃旧模型。',
      readiness_loading: '正在加载数据集…',
      readiness_no_categories: '添加一个含已上传切片的前景类别即可开始训练。',
      readiness_background_short: (need) =>
        `Background Noise 还需 ${need} 个已上传切片才能开始训练。`,
      readiness_foreground_short: '至少一个前景类别需要 10 个已上传切片才能开始训练。',
      button_starting: '正在开始…',
      button_cancel: '取消',
      button_cancelling: '正在取消…',
      button_retrain: '重新训练',
      button_train: '训练模型',
      button_title_loading: '正在加载数据集…',
      button_title_not_ready_default: '未就绪原因',
      button_title_form_errors: '修正高亮的超参数字段以启用训练。',
      button_title_idle_trained:
        '已有模型与此修订匹配——重新训练可尝试不同的超参数或不同的随机种子。可从下方的"模型"区激活任意模型。',
      button_title_idle_busy: '已有另一个工作区正在训练，同一时间只能运行一个任务。',
      button_title_idle_ready: '在此工作区数据集上训练模型。',
      button_title_starting: '正在提交训练请求…',
      button_title_running: '取消正在运行的训练任务。',
      button_title_cancelling: '正在取消…',
      summary_chip_epochs: (epochs) => `${epochs} 轮次`,
      summary_chip_no_holdout: '无留出集',
      summary_chip_val: (pctLabel) => `验证 ${pctLabel}`,
      hyperparameters_disclosure_label: '超参数',
      start_error_title: '无法开始训练'
    },
    form: {
      epochs_label: '轮次',
      batch_size_label: '批大小',
      learning_rate_label: '学习率',
      validation_split_label: '验证集比例',
      validation_split_hint: '· 0 表示禁用',
      seed_label: '随机种子',
      seed_hint: '· 留空则由守护进程选取熵',
      seed_placeholder: '（可选）'
    },
    progress: {
      submitting: '正在提交…',
      job_short_id: (shortId) => `任务 ${shortId}…`,
      train_loss_label: '训练损失',
      train_acc_label: '训练准确率',
      val_acc_label: '验证准确率',
      val_acc_disabled_label: '验证准确率 · 已禁用',
      em_dash: ' — '
    },
    logs: {
      heading: '日志',
      entry_count: (count) => `${count} 条`,
      waiting_first_message: '正在等待第一条消息…'
    },
    chart: {
      waiting_first_epoch: '正在等待第一个轮次…',
      legend_loss: '损失',
      legend_train: '训练',
      legend_val: '验证',
      tooltip_epoch: '轮次',
      tooltip_loss: '损失',
      tooltip_train: '训练',
      tooltip_val: '验证',
      chart_aria: '训练指标图表'
    },
    history: {
      heading: '历史',
      keeps_last: (cap) => `保留最近 ${cap} 次`,
      retention_title: (cap) =>
        `守护进程为每个工作区保留最近 ${cap} 个训练日志文件；新的运行开启时会清除较旧的 JSONL 轨迹。已发布的模型记录（在下方的"模型"区）不受影响——仅清除 JSONL 轨迹。`,
      empty_state_prefix: '此工作区暂无训练运行。单击 ',
      empty_state_button: '训练模型',
      empty_state_suffix: ' 即可开始。',
      hide_older_label: '隐藏较旧的运行',
      show_older_label: (count) => `显示 ${count} 次较旧的运行`,
      hide_older_title: '将较旧的运行区折叠回最近两次。',
      show_older_title: '展开此工作区较旧的训练运行，每批 5 次分页加载。',
      load_more_label: (count) => `再加载 ${count} 次`,
      load_more_title: '从设备获取下一批较旧的训练运行。',
      menu_delete: '删除',
      menu_deleting: '正在删除…',
      menu_hint_train_active: '训练进行中',
      menu_hint_live: '实时',
      delete_error_title: '无法删除训练日志'
    },
    history_item: {
      time_started_pre_ack: '已开始',
      time_started: (relative) => `开始于 ${relative}`,
      time_finished: (relative) => relative,
      time_title_started: (absolute) => `开始于 ${absolute}`,
      time_title_finished: (absolute) => `完成于 ${absolute}`,
      detail_epoch: (current, total) => `轮次 ${current}/${total}`,
      detail_class_count: (count) => `${count} 个类别`,
      detail_val_acc: (pctLabel) => `验证 ${pctLabel}`,
      detail_train_acc: (pctLabel) => `训练 ${pctLabel}`,
      detail_stopped_at: (stageLabel) => `停止于 ${stageLabel}`
    },
    summary: {
      completed_aria: '已完成运行摘要',
      failed_aria: '失败运行摘要',
      cancelled_aria: '已取消运行摘要',
      duration_label: '时长',
      epochs_label: '轮次',
      best_val_at: (epoch) => `最佳验证 @ ${epoch}`,
      final_train_acc_label: '最终训练准确率',
      classes_label: '类别',
      stopped_at_label: '停止于',
      cancelled_at_label: '取消于',
      epochs_tooltip_full: '运行了完整配置的轮次数。',
      epochs_tooltip_partial: '实际轮次与配置轮次数的对比。',
      after_epochs: (run, total) => `在 ${run}/${total} 轮次后`,
      failed_no_diagnostic: '未出现诊断信息。请查看守护进程日志了解详情。',
      cancelled_default_reason: '在下一个训练检查点停止。',
      failed_default: '训练失败。'
    },
    stage: {
      prepare: '准备',
      dataset_scan: '扫描数据集',
      feature_extract: '提取特征',
      train: '训练',
      save: '保存',
      publish: '发布'
    },
    state: {
      running: '运行中',
      completed: '已完成',
      failed: '失败',
      cancelled: '已取消'
    },
    state_submitting: '提交中',
    store_log: {
      seed_submitted: '已提交，正在等待设备开始发出事件…',
      seed_recovered: '已从设备恢复进行中的训练任务。',
      job_submitted: (backbone) => `任务已提交 · 骨干网络 ${backbone}`,
      job_running: '任务运行中',
      phase_prefix: (stageLabel) => `阶段：${stageLabel}`,
      job_failed: (stageLabel, error) => `任务在 ${stageLabel} 失败 · ${error}`,
      job_cancelled: (stageLabel) => `任务在 ${stageLabel} 取消`,
      job_cancelled_shutdown: (stageLabel) => `任务在 ${stageLabel} 取消（守护进程关闭）`,
      scanned_dataset: (nClasses, nExamples) =>
        `已扫描数据集 · ${nClasses} 个类别 · ${nExamples} 个样本`,
      features_extracted: (kept, dropped, elapsedSec) => {
        const droppedSuffix = dropped > 0 ? ` · 丢弃 ${dropped}` : '';
        return `已提取特征 · 保留 ${kept}${droppedSuffix} · ${elapsedSec}s`;
      },
      train_split: (trainN, valN) => `训练集划分 · ${trainN} 训练 · ${valN} 验证`,
      epoch_completed: (epoch, epochs, lossLabel, trainAccLabel, valAccLabel) => {
        const valPart = valAccLabel !== null ? ` · 验证 ${valAccLabel}` : '';
        return `轮次 ${epoch}/${epochs} · 损失 ${lossLabel} · 训练 ${trainAccLabel}${valPart}`;
      },
      train_loop_done: (epochsRun, elapsedSec, bestValAccLabel, bestEpoch) => {
        const bestPart =
          bestValAccLabel !== null && bestEpoch !== null
            ? ` · 最佳验证 ${bestValAccLabel} @ 轮次 ${bestEpoch}`
            : '';
        return `训练循环完成 · ${epochsRun} 轮次 · 耗时 ${elapsedSec}s${bestPart}`;
      },
      head_published: (headId, size, nClasses, rev) =>
        `模型已发布 · ${headId} · ${size} · ${nClasses} 个类别 · rev ${rev}`,
      job_completed: (labelsList) =>
        labelsList.length > 0 ? `任务已完成 · ${labelsList}` : '任务已完成'
    }
  },
  deploy: {
    pane: {
      heading: '部署',
      description: '选择已训练的模型，将其无缝热替换到实时推理中，实现零停机。',
      pill_deployed: '已部署',
      pill_deployed_title: '在本工作区训练的模型正作为运行时模型。',
      pill_default: '默认',
      pill_default_title: '内置的默认模型正在运行。',
      pill_standby: '待命',
      pill_standby_title:
        '来自其他工作区的模型正作为运行时模型。本工作区处于待命状态。在此部署模型将替换它。',
      pill_detached: '已分离',
      pill_detached_title: '生成该运行时模型的工作区已被删除。该模型仍在运行。',
      config_disclosure_label: '输入与推理配置',
      config_chip_freq: (hzLabel) => `freq ${hzLabel} Hz`,
      config_chip_top_k: (topK) => `top-k ${topK}`
    },
    heads_table: {
      heading: '模型',
      count_label: (count) => `${count} 个模型`,
      // Split off the bare count so it can collapse on a narrow card; carries its own leading comma.
      count_retained: (retainedCap) => `，保留最新 ${retainedCap} 个`,
      revert_to_default: '还原为默认',
      revert_to_id: (shortId) => `还原为 ${shortId}`,
      revert_title: '重新部署先前运行的模型',
      default_row_headline: '默认',
      default_row_description: '内置的回退模型，始终可用。',
      default_active_title: '内置的默认模型当前已部署。',
      default_aria_active: '默认模型为活动模型',
      default_aria_deploy: '部署默认模型',
      default_title_active: '默认模型已部署',
      default_title_deploying: '部署中…',
      default_title_busy: '此列表中的另一个模型正忙',
      default_title_idle: '还原为内置的默认模型',
      menu_deploy: '部署',
      menu_export: '导出为 .alpkg',
      menu_exporting: '导出中…',
      menu_delete: '删除',
      menu_hint_active: '活动',
      menu_hint_deployed: '已部署',
      error_deploy_head: '无法部署模型',
      error_export_head: '无法导出模型',
      error_deploy_default: '无法部署默认模型'
    },
    head_row: {
      pill_latest: '最新',
      pill_latest_title: '在工作区当前修订上训练的最新模型。',
      pill_active: '活动',
      pill_active_title: '此模型当前已部署在推理管线中。',
      // Fixed-width single-string meta for the model-card popover and delete-confirm card.
      meta_line: (size, classCount, rev, relative) =>
        `${size} · ${classCount} 个类别 · rev ${rev} · ${relative}`,
      // Row meta renders segment-by-segment so size/rev can drop as the row narrows (size/age come
      // from format utils, not the catalog).
      meta_classes: (classCount) => `${classCount} 个类别`,
      meta_rev: (rev) => `rev ${rev}`,
      row_aria_deployed: (shortId) => `已部署模型 ${shortId}`,
      row_aria_deploy: (shortId) => `部署模型 ${shortId}`,
      row_title_deployed: '此模型已部署',
      row_title_deploying: '部署中…',
      row_title_exporting: '导出中…',
      row_title_busy: '此列表中的另一个模型正忙',
      row_title_idle: '单击将此模型热替换到推理管线中',
      export_title_exporting: '导出中…',
      export_title_idle: '将此模型导出为 .alpkg 归档文件',
      export_aria_exporting: (shortId) => `正在导出模型 ${shortId}`,
      export_aria_idle: (shortId) => `导出模型 ${shortId}`,
      info_title: '查看模型卡片',
      info_aria: (shortId) => `查看 ${shortId} 的模型卡片`
    },
    inference_preview: {
      heading: '预览',
      off_title: '预览已关闭',
      off_description: '启动预览以查看已部署模型的频谱图和 top-k 流。',
      start_button: '启动预览'
    },
    info_dialog: {
      title_with_id: (shortId) => `模型卡片 · ${shortId}`,
      loading: '正在加载类别…',
      error_title: '无法加载类别',
      retry: '重试',
      classes_heading: '类别',
      class_labels_aria: '已训练的类别标签'
    },
    delete_dialog: {
      title: '删除此模型？',
      body: '移除已训练的模型字节数据及其清单。数据集和其他模型将保留。此操作无法撤销。',
      submit: '删除'
    }
  },
  workspace: {
    list: {
      title: '工作区',
      at_cap_subtitle: (max) => `已达到 ${max} 个工作区的上限。请先删除一个再创建新的。`,
      default_subtitle: '每个工作区保存一个带标签的数据集以及由它训练出的所有模型。',
      daemon_unavailable_title: '设备不可用',
      loading: '正在加载工作区…',
      empty_title: '尚无工作区',
      empty_description: '工作区是录音、带标签样本和已训练的模型的存放之处。创建一个即可开始。',
      selected_count_aria: (count) => `已选择 ${count} 个`,
      new_button_label: '新建工作区',
      new_button_aria: '新建工作区',
      new_at_cap_label: (count, max) => `已达上限 · ${count}/${max}`,
      new_at_cap_title: '已达上限。请先删除一个工作区。',
      import_button_label: '导入',
      import_button_aria: '导入工作区',
      import_button_title: '从 .alpkg 或 TFJS 包导入工作区',
      select_button_label: '选择',
      done_button_label: '完成',
      select_all_label: '全选',
      deselect_all_label: '取消全选',
      bulk_delete_label_count: (count) => `删除 ${count} 个`,
      bulk_delete_label_bare: '删除',
      bulk_delete_aria_count: (count) => `删除 ${count} 个工作区`,
      bulk_delete_aria_fallback: '删除所选工作区',
      menu_open: '打开',
      menu_rename: '重命名',
      menu_export: '导出',
      menu_delete: '删除',
      menu_select_one: '选择',
      menu_deselect_one: '取消选择',
      menu_select_all: '全选',
      menu_deselect_all: '取消全选',
      menu_select_workspaces: '选择工作区',
      menu_done_exit: '完成（退出选择）',
      menu_new: '新建工作区',
      menu_new_at_cap: (max) => `新建工作区（已达 ${max} 个上限）`,
      menu_import: '导入工作区'
    },
    detail: {
      back_link: '← 工作区',
      loading: '正在加载工作区…',
      not_found_title: '未找到工作区',
      not_found_description: '可能已在其他标签页中或直接通过设备删除。返回列表查看其余工作区。',
      back_to_list_button: '返回工作区',
      load_error_title: '无法加载此工作区',
      created_label: (relative) => `创建于 ${relative}`,
      rev_label: (rev) => `rev ${rev}`,
      modified_label: (relative) => `修改于 ${relative}`,
      live_pill_title: '因近期上传而推进。重新加载以刷新修改时间戳。',
      live_pill: '实时',
      menu_rename: '重命名',
      menu_export: '导出',
      menu_import: '导入',
      menu_delete: '删除',
      menu_back_to_list: '返回工作区'
    },
    create_dialog: {
      title: '新建工作区',
      name_label: '名称',
      name_placeholder: 'my-workspace',
      name_help:
        '最多 128 个字符。不含斜杠或控制字符。名称是唯一可见的标识符，请选择一个易记的名称。',
      submit: '创建'
    },
    rename_dialog: {
      title: '重命名工作区',
      name_label: '名称',
      name_help:
        '最多 128 个字符。不含斜杠或控制字符。重命名不会推进工作区修订——类别、切片和模型保持不变。',
      submit: '保存'
    },
    delete_dialog: {
      title: '删除此工作区？',
      body: '将移除数据集、所有已训练的模型和日志。此操作无法撤销。',
      submit: '删除'
    },
    bulk_delete_dialog: {
      title_count: (count) => `删除 ${count} 个工作区？`,
      body: '将移除每个工作区的数据集、已训练的模型和日志。此操作无法撤销。',
      submit_count: (count) => `删除 ${count} 个`
    },
    tool_island: {
      aria_label: '工作区操作',
      rename_aria: '重命名工作区',
      rename_title: '重命名工作区',
      export_aria: '导出工作区',
      export_title: '导出工作区（数据集 + 模型）',
      import_aria: '导入工作区',
      import_title: '导入工作区（数据集 + 模型）'
    },
    card: {
      created_label: (relative) => `创建于 ${relative}`,
      select_aria: (name) => `选择工作区 ${name}`,
      rename_aria: (name) => `重命名工作区 ${name}`,
      deleting: '删除中'
    },
    import_dialog: {
      title_into: (workspaceName) => `导入到 · ${workspaceName}`,
      title_fallback: '导入',
      step_indicator: (current, total) => `第 ${current} 步，共 ${total} 步`,
      pipeline_error_title: '导入失败',
      error_invalid_state: '对话框状态不一致——没有可导入的归档文件。',
      pick_file: {
        drop_zone_title_attr: '将 .alpkg 归档文件或 TFJS 包拖放到此处，或单击浏览',
        reading: '读取中…',
        drop_zone_tfjs_staging: '拖放更多文件以完成 TFJS 包',
        drop_zone_idle: '将 .alpkg 归档文件或 TFJS 包拖放到此处',
        browse_button: '浏览文件',
        error_empty_drop: '请拖放 .alpkg 归档文件或 TFJS 包。',
        error_multi_alpkg: (count) => `一次只能选择一个 .alpkg 归档文件——已选择 ${count} 个。`,
        error_mixed_archive: '.alpkg 归档文件必须单独选择，不能与其他文件混选。',
        error_file_count_cap: (max, picked) =>
          `一次最多拖放或选择 ${max} 个文件——已选择 ${picked} 个。`,
        error_single_too_large: (name, size, cap) => `"${name}"为 ${size}——单文件上限为 ${cap}。`,
        error_total_too_large: (total, cap) => `选区共计 ${total}——单次拖放上限为 ${cap}。`,
        error_tfjs_merged_file_count: (mergedCount, cap) =>
          `暂存集将总计 ${mergedCount} 个文件——上限为 ${cap}。请清除并重新拖放更小的包。`,
        error_tfjs_merged_bytes: (mergedBytes, cap) =>
          `暂存集将总计 ${mergedBytes}——上限为 ${cap}。请清除并重新拖放更小的包。`,
        staged_files_heading: '暂存文件',
        staged_files_count: (count) => `${count} 个文件`,
        clear_button: '清除',
        error_could_not_read_archive: '无法读取归档文件。',
        error_could_not_read_file: '无法读取文件。',
        error_could_not_read_picked_files: '无法读取所选文件。',
        error_could_not_read_model_json: '无法读取 model.json。',
        tfjs_diag_empty_drop: '请拖放 TFJS 包文件（model.json + 分片 + labels）。',
        tfjs_diag_no_model_json: '拖放内容中没有 "model.json"。请包含 TFJS 清单。',
        tfjs_diag_ambiguous_model_json: (count) =>
          `包不明确：有 ${count} 个名为 "model.json" 的文件。`,
        tfjs_diag_multiple_labels_txt: '拖放内容中有多个 "labels.txt" 文件。请只包含一个。',
        tfjs_diag_multiple_metadata_json: '拖放内容中有多个 "metadata.json" 文件。请只包含一个。',
        tfjs_diag_both_labels: '同时提供了 "labels.txt" 和 "metadata.json"。请只包含一个标签来源。',
        tfjs_diag_no_labels: '未提供标签文件。请包含 "labels.txt" 或 "metadata.json"。',
        tfjs_diag_shard_collision_one: (quotedName) =>
          `两个暂存文件共用分片名 ${quotedName}。请清除暂存，只拖放需要的那一份。`,
        tfjs_diag_shard_collision_many: (quotedNames, overflow) =>
          `多个暂存文件共用 "model.json" 引用的分片名：${quotedNames}${overflow ? '…' : ''}。请清除暂存，只拖放需要的副本。`,
        tfjs_diag_missing_shard_one: (quotedName) => `缺少 "model.json" 引用的分片 ${quotedName}。`,
        tfjs_diag_missing_shards_many: (count, quotedNames, overflow) =>
          `缺少 ${count} 个 "model.json" 引用的分片：${quotedNames}${overflow ? '…' : ''}。`,
        tfjs_diag_model_json_not_json: 'model.json 不是有效的 JSON。',
        tfjs_diag_model_json_not_object: 'model.json 不是 JSON 对象。',
        tfjs_diag_model_json_no_manifest: 'model.json 缺少 "weightsManifest" 数组。',
        tfjs_diag_model_json_no_shards: 'model.json 未声明任何分片文件。'
      },
      pick_target: {
        section_label: '导入到',
        mode_radio_aria: '目标工作区模式',
        mode_use_existing: '使用现有',
        mode_create_new: '新建',
        no_workspaces_prefix: '尚无工作区——切换到',
        no_workspaces_link_label: '新建',
        no_workspaces_suffix: '以创建一个。',
        workspace_list_aria: '选择目标工作区',
        workspace_created_label: (relative) => `创建于 ${relative}`,
        create_name_placeholder: 'my-imported-workspace',
        create_will_carry_tags: (tagsCsv) => `将从源携带标签：${tagsCsv}`,
        alpkg_source_card_title: (name, id) => `${name} (${id})`,
        alpkg_source_created_label: (relative) => `创建于 ${relative}`,
        alpkg_source_rev_label: (rev) => `rev ${rev}`,
        alpkg_source_modified_label: (relative) => `修改于 ${relative}`,
        tfjs_bundle_card_title: 'TFJS 包',
        tfjs_show_labels_aria: '显示类别标签',
        tfjs_meta_strip: (size, shards, classes, labelsFileName) => {
          const classesPart = classes !== null && classes > 0 ? ` · ${classes} 个类别` : '';
          const shardsPart = ` · ${shards} 个分片`;
          const labelsPart = labelsFileName !== null ? ` · 经由 ${labelsFileName}` : '';
          return `${size}${classesPart}${shardsPart}${labelsPart}`;
        }
      },
      summary: {
        datasets_heading: '数据集',
        datasets_counter: (selected, total) => `已选择 ${selected} / ${total}`,
        checking_categories: '正在检查目标工作区中的现有类别…',
        slice_count: (count) => `${count} 个切片`,
        rename_button_aria: '重命名目标类别',
        rename_button_title_default: '重命名目标类别',
        mode_aria: (modeLabel) => `导入操作：${modeLabel}`,
        mode_menu_aria: (sourceName) => `${sourceName} 的导入操作`,
        rename_popover_aria: (sourceName) => `重命名 ${sourceName} 的目标类别`,
        rename_popover_heading: '重命名',
        rename_chips_heading: '或复用现有',
        heads_heading: '模型',
        heads_cap_tooltip: (cap) =>
          `每个工作区最多 ${cap} 个模型。新模型加入时——无论来自重新训练还是导入——较旧的非活动模型会滚动移除。`,
        heads_counter: (selected, existingInTarget, cap, activeInTarget) => {
          const active = activeInTarget > 0 ? ` · 活动 ${activeInTarget} 个已固定` : '';
          return `已选择 ${selected} · 目标 ${existingInTarget} / ${cap}${active}`;
        },
        checking_heads: '正在检查目标模型…',
        displacement_warning: (displaced, cap) =>
          `导入将挤出 ${displaced} 个最旧的非活动模型，以适应 ${cap} 个模型的上限。`,
        head_exists_badge_title: '目标工作区中已存在具有此 id 的模型。',
        head_exists_badge: '已存在',
        head_show_details_aria: '显示模型详情',
        head_class_count: (count) => `${count} 个类别`,
        head_info_metadata: (size, classes, revisionId, createdAbsolute, createdRelative) => {
          const classesPart = classes !== null ? ` · ${classes} 个类别` : '';
          const revPart = revisionId !== null ? ` · rev ${revisionId}` : '';
          const createdPart =
            createdAbsolute !== null && createdRelative !== null
              ? ` · ${createdAbsolute} (${createdRelative})`
              : '';
          return `${size}${classesPart}${revPart}${createdPart}`;
        },
        head_classes_heading: '类别',
        head_class_labels_aria: '已训练的类别标签',
        archive_errors_summary: (count) => `已跳过 ${count} 条归档条目`,
        tfjs_ignored_unknown: (count, fileList) => `已忽略 ${count} 个无法识别的文件：${fileList}`,
        tfjs_classes_popover_heading: (count) => `类别 (${count})`,
        tfjs_classes_popover_aria: '类别标签',
        head_disabled_reasons: {
          loading: '正在加载目标模型…',
          exists: '目标中已存在。请选择其他模型。',
          ceiling: '已达选择上限。请先取消勾选另一行。'
        }
      },
      modes: {
        new: '新建',
        merge: '合并',
        replace: '替换',
        skip: '跳过'
      },
      mode_tooltips: {
        new: '使用归档文件中的切片从头创建该类别。',
        merge:
          '将归档文件的切片上传到现有类别之上。相同 sha256 的切片会覆盖自身，新切片则加入集合。',
        replace: '删除现有类别（及其包含的每个切片），然后从归档文件上传。',
        skip: '不导入此类别。'
      },
      mode_disabled_reasons: {
        new_exists:
          '已存在使用此目标名称的类别。选择"合并"以添加切片，或选择"替换"以清空并重新导入。',
        merge_missing: '不存在使用此目标名称的现有类别。选择"新建"以创建一个。',
        replace_missing: '不存在使用此目标名称的现有类别。选择"新建"以创建一个。'
      },
      running: {
        progress_replacing_categories: (categoryDisplay, done, total) => {
          const cat = categoryDisplay !== null ? ` · ${categoryDisplay}` : '';
          return `正在替换类别${cat} · ${done} / ${total}`;
        },
        progress_uploading_datasets: (categoryDisplay, done, total) => {
          const cat = categoryDisplay !== null ? ` · ${categoryDisplay}` : '';
          if (typeof done === 'number' && typeof total === 'number') {
            return `正在上传切片${cat} · ${done} / ${total}`;
          }
          return `正在上传切片${cat}`;
        },
        progress_importing_heads: (index1, total, subPhase) => {
          const sub = subPhase !== null ? ` (${subPhase})` : '';
          return `正在导入模型 ${index1} / ${total}${sub}`;
        },
        progress_uploading_tfjs: (done, total) => `正在上传 TFJS 文件 · ${done} / ${total}`,
        progress_converting_tfjs: '正在转换 TFJS 包…',
        ds_pending: '待处理',
        ds_replacing: '替换中',
        ds_uploading_counter: (uploaded, total) => `${uploaded} / ${total}`,
        ds_done_uploaded: (uploaded) => `已上传 ${uploaded} 个`,
        ds_failed_count: (failed) => `${failed} 个失败`,
        ds_failed_label: '失败',
        ds_failed_title_count: (failed) => `${failed} 个切片上传失败`,
        head_queued: '排队中',
        head_skipped_badge_title: '模型 id 已存在于磁盘上，编排器已跳过它（幂等重新导入）。',
        head_per_log_not_started: '尚未开始——此模型的导入开始后将出现日志行。',
        head_per_log_no_events: '未记录任何事件。',
        log_count: (count) => `${count} 条日志`
      },
      head_phase: {
        queued: '排队中',
        uploading_files: '正在上传文件',
        starting_convert: '正在启动转换',
        converting: '正在转换',
        cleaning_up: '正在清理',
        done: '完成',
        failed: '失败'
      },
      head_outcome: {
        imported: '已导入',
        replaced: '已替换',
        skipped: '已跳过',
        failed: '失败'
      },
      convert_stage: {
        prepare: '正在准备',
        read_manifest: '正在读取清单',
        validate_manifest: '正在校验清单',
        verify_mpk: '正在验证 MPK',
        stage_mpk: '正在暂存 MPK',
        read_model_json: '正在读取 model.json',
        stage_shards: '正在暂存分片',
        extract_weights: '正在提取权重',
        read_labels: '正在读取 labels',
        stage_head_mpk: '正在暂存模型 MPK',
        publish_head: '正在发布模型'
      },
      convert_event: {
        job_submitted: (converter) => `任务已通过 ${converter} 提交`,
        job_running: '任务运行中',
        phase: (stageLabel) => `阶段：${stageLabel}`,
        manifest_validated: (classes) => `清单已校验 · ${classes} 个类别`,
        mpk_verified: (size) => `MPK 已验证 · ${size}`,
        weights_extracted: (classes, inDim) => `权重已提取 · ${classes} 个类别 · ${inDim} in_dim`,
        labels_loaded: (labels) => `标签已加载 · ${labels} 个标签`,
        head_published: (idempotentSkip) =>
          `模型已发布${idempotentSkip ? '（已在磁盘上，已跳过）' : ''}`,
        job_completed: (classes) => `任务已完成 · ${classes} 个类别`,
        job_failed: (category, error) => `任务失败 · ${category} · ${error}`
      },
      done: {
        conflict_detail: (storedSha8, incomingSha8) =>
          `目标已持有具有此 id 但 sha256 不同的模型（${storedSha8} 对比传入的 ${incomingSha8}）。`,
        retry_button: '替换现有并重试'
      },
      footer: {
        cancel: '取消',
        back: '返回',
        next: '下一步',
        import: '导入',
        importing: '正在导入…',
        back_to_selection: '返回选择',
        done: '完成'
      }
    },
    export_dialog: {
      title: (workspaceName) => `导出工作区 · ${workspaceName}`,
      load_error_title: '无法加载此工作区',
      loading: '正在加载工作区…',
      nothing_to_export: '此工作区尚无类别和模型——没有可导出的内容。',
      datasets_heading: '数据集',
      heads_heading: '模型',
      select_all: '全选',
      deselect_all: '取消全选',
      row_empty: '空',
      row_slice_count: (count) => `${count} 个切片`,
      head_meta_title: (size, classCount) => `${size} · ${classCount} 个类别`,
      head_meta_classes: (count) => `${count} 个类别`,
      pending_warning: '选区中仍在上传或待处理的切片将被排除——仅导出已在磁盘上的切片。',
      progress_preparing_workspace: '正在读取工作区元数据…',
      progress_fetching_slices: '正在获取切片…',
      progress_listing_slices: '正在列出切片…',
      progress_fetched_slices: (done, total) => `已获取 ${done} / ${total} 个切片…`,
      progress_validating_heads: '正在校验模型…',
      progress_validated_heads: (done, total) => `已校验 ${done} / ${total} 个模型…`,
      progress_packing: '正在打包归档文件…',
      progress_downloading: '正在开始下载…',
      error_default: '导出失败',
      error_in_category: (categoryDisplay) => `在"${categoryDisplay}"中导出失败`,
      error_for_head: (shortId) => `模型 ${shortId} 导出失败`,
      exporting: '正在导出…',
      export_aria: '导出所选项目',
      export_button: '导出'
    }
  }
} satisfies Messages;
