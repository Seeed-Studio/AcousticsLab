import type { Messages } from '../types';

// Deutsch (de). Mirrors en.ts 1:1 (keys, params, ${...}, comments); see en.ts for conventions. de
// style: impersonal Infinitivstil (no du/Sie); nouns capitalised; real singular/plural; „…“ for
// name/label quotes, straight "..." for code tokens; keep ' · ', '…', numbers, units, tokens,
// ${...} ASCII; break dash ' – ' (en dash); 'Background Noise' stays literal.
export const de = {
  app: {
    name: 'AcousticsLab',
    description:
      'Ein privates, multi-Backend-fähiges, vollständig lokales KI-/ML-Toolkit zum Entwickeln und Bereitstellen von Echtzeit-Erkennung von Geräuschereignissen.'
  },
  routes: {
    dashboard_title: (brand) => brand,
    workspace_list_title: (brand) => `Arbeitsbereiche · ${brand}`,
    workspace_detail_title: (workspaceName, brand) => `${workspaceName} · ${brand}`
  },
  nav: {
    dashboard: 'Dashboard',
    workspaces: 'Arbeitsbereiche',
    home_aria: 'AcousticsLab-Startseite',
    menu_fallback: 'Menü',
    primary_nav_aria: 'Hauptnavigation'
  },
  dashboard: {
    limited_support_title: 'Eingeschränkte Browserunterstützung',
    visualization_panel: {
      heading: 'Visualisierung',
      // Dot-separated segments so codec/channels can drop on a narrow header; rate and window always stay.
      audio_sample_rate: '48 kHz',
      audio_channels: 'mono',
      audio_codec: 'opus',
      audio_window: '3 s Fenster'
    },
    inference_panel: {
      heading: 'Inferenz'
    },
    configuration_panel: {
      heading: 'Konfiguration'
    },
    configuration_controls: {
      daemon_unavailable_title: 'Gerät nicht verfügbar',
      daemon_unavailable_default:
        'Die Konfiguration wird automatisch fortgesetzt, sobald das Gerät erreichbar ist.',
      microphone_heading: 'Mikrofon',
      source_label: 'Quelle',
      auto_first_available: 'automatisch · erstes verfügbares',
      channel_label: 'Kanal',
      auto_channel: 'automatisch',
      inference_cadence_heading: 'Inferenztakt',
      overlap_ratio_label: 'Überlappungsverhältnis',
      top_k_label: 'Top-K',
      loading: 'wird geladen…',
      kind_alsa: 'ALSA',
      kind_unknown: 'unbekannt',
      approx_hz: (hz) => `${hz} Hz`,
      khz: (khz) => `${khz} kHz`,
      hz: (rate) => `${rate} Hz`
    },
    top_k_meter: {
      awaiting_first_frame: 'Warten auf ersten Inferenz-Frame…'
    },
    active_head_card: {
      heading: 'Aktives Modell',
      pill_default: 'Standard',
      pill_workspace: 'Arbeitsbereich',
      pill_detached: 'Abgetrennt',
      pill_default_title: 'Das integrierte Standardmodell läuft.',
      pill_workspace_title: 'Ein trainiertes Arbeitsbereichsmodell läuft.',
      pill_detached_title:
        'Der Quell-Arbeitsbereich wurde gelöscht, nachdem dieses Modell aktiviert wurde.',
      loading_active: 'aktives Modell wird geladen…',
      activated_label: 'aktiviert',
      class_count_label: (count) => (count === 1 ? 'Klasse' : 'Klassen'),
      workspace_dt: 'Arbeitsbereich',
      revision_dt: 'Revision',
      rev_value: (rev) => `rev ${rev}`,
      deleted_tag: '(gelöscht)',
      loading: 'wird geladen…',
      ws_title_orphaned_with_name: (name, uuid) => `${name} · ${uuid} (Arbeitsbereich gelöscht)`,
      ws_title_orphaned: (uuid) => `${uuid} (Arbeitsbereich gelöscht)`,
      ws_title_with_name: (name, uuid) => `${name} · ${uuid}`
    }
  },
  theme: {
    label: 'Design',
    label_with_current: (currentLabel) => `Design: ${currentLabel}`,
    options: {
      auto: 'Automatisch',
      light: 'Hell',
      dark: 'Dunkel'
    }
  },
  locale: {
    label: 'Sprache',
    label_with_current: (currentChip) => `Sprache: ${currentChip}`
  },
  health: {
    aria_label: 'Systemzustand',
    levels: {
      unknown: 'Verbinden…',
      ok: 'Fehlerfrei',
      degraded: 'Beeinträchtigt',
      unhealthy: 'Fehlerhaft',
      unreachable: 'Nicht erreichbar'
    },
    popover: {
      daemon_unreachable_title: 'Gerät nicht erreichbar',
      waiting_first_snapshot: 'Warten auf erste Statusaufnahme…',
      subsystems_heading: 'Subsysteme',
      seconds_ago: (seconds) => `vor ${seconds}s`,
      stat_cpu_label: 'cpu',
      stat_rss_label: 'rss',
      stat_disk_free_label: 'freier Speicher',
      uptime_label: 'Betriebszeit',
      dropped_count: (count) => `verworfen: ${count}`
    }
  },
  common: {
    cancel: 'Abbrechen',
    dismiss: 'Schließen'
  },
  error: {
    another_train_running: 'Auf diesem Gerät läuft bereits ein anderer Trainingsauftrag.',
    another_convert_running: 'Auf diesem Gerät läuft bereits ein anderer Konvertierungsauftrag.',
    job_conflict: 'Auf dieser Ressource läuft bereits ein anderer Vorgang.',
    event_gap:
      'Der Ereignisstrom hat einen Sprung gemacht und muss aus den Protokollen aufholen. Verbindung wird wiederhergestellt…',
    too_early: 'Das Gerät wendet noch die vorherige Änderung an. Wird erneut versucht…',
    unavailable: 'Das Gerät ist vorübergehend nicht verfügbar. In Kürze erneut versuchen.',
    internal:
      'Im Daemon ist ein interner Fehler aufgetreten. Erneut versuchen. Falls das Problem bestehen bleibt, die Daemon-Protokolle prüfen.',
    unknown: 'Etwas ist schiefgelaufen. Erneut versuchen.',
    something_went_wrong: 'Etwas ist schiefgelaufen.',
    request_failed: (code) => `Anfrage fehlgeschlagen (${code}).`
  },
  validation: {
    name: {
      empty: 'Der Name darf nicht leer sein.',
      max_bytes: (max) => `Der Name darf höchstens ${max} Bytes lang sein.`,
      slashes_or_nul: 'Der Name darf keine Schrägstriche oder NUL-Bytes enthalten.',
      starts_or_ends_whitespace: 'Der Name darf nicht mit einem Leerzeichen beginnen oder enden.',
      control_chars: 'Der Name darf keine Steuerzeichen enthalten.',
      starts_with_dot: 'Der Kategoriename darf nicht mit einem Punkt beginnen.',
      starts_with_underscore:
        'Der Kategoriename darf nicht mit einem Unterstrich beginnen (für integrierte Klassen reserviert).',
      starts_with_hyphen:
        'Der Kategoriename darf nicht mit einem Bindestrich beginnen (aus Sicherheitsgründen).',
      bad_chars: 'Nur Buchstaben, Ziffern, Punkte, Bindestriche und Unterstriche sind zulässig.',
      category_max_bytes: (max) => `Der Kategoriename darf höchstens ${max} Bytes lang sein.`,
      category_empty: 'Der Kategoriename darf nicht leer sein.'
    },
    cfg: {
      epochs_whole: 'Die Anzahl der Epochen muss eine ganze Zahl sein.',
      epochs_range: (min, max) => `Die Anzahl der Epochen muss zwischen ${min} und ${max} liegen.`,
      batch_whole: 'Die Batchgröße muss eine ganze Zahl sein.',
      batch_range: (min, max) => `Die Batchgröße muss zwischen ${min} und ${max} liegen.`,
      lr_finite: 'Die Lernrate muss eine endliche Zahl sein.',
      lr_greater_than_zero: 'Die Lernrate muss größer als 0 sein.',
      lr_max: (max) => `Die Lernrate darf höchstens ${max} betragen.`,
      seed_whole: 'Der Seed muss eine ganze Zahl sein.',
      seed_non_negative: 'Der Seed muss 0 oder größer sein.',
      seed_too_large: 'Der Seed ist zu groß.',
      split_finite: 'Der Validierungsanteil muss eine endliche Zahl sein.',
      split_min: 'Der Validierungsanteil muss 0 oder größer sein.',
      split_max: (max) => `Der Validierungsanteil darf höchstens ${max} betragen.`
    }
  },
  streams: {
    socket_status: {
      connecting: 'Verbinden…',
      open: 'live',
      closed: 'Getrennt',
      error: 'Fehler'
    }
  },
  recorder: {
    mic_error_denied:
      'Der Mikrofonzugriff wurde verweigert. Mikrofonzugriff in den Browsereinstellungen erlauben und erneut versuchen.',
    mic_error_not_found: 'Kein Mikrofon gefunden. Eines anschließen und erneut versuchen.',
    mic_error_in_use:
      'Das Mikrofon wird von einer anderen Anwendung verwendet. Diese schließen und erneut versuchen.',
    mic_error_interrupted: 'Die Mikrofonaufnahme wurde unterbrochen. Erneut versuchen.',
    mic_error_generic: 'Das Mikrofon konnte nicht gestartet werden. Erneut versuchen.'
  },
  category: {
    list: {
      heading: 'Datensatz',
      description:
        'Jede Kategorie wird zu einem Klassenlabel, das der Trainer lernt – Background Noise ist erforderlich.',
      add_button: 'Kategorie hinzufügen',
      add_button_aria: 'Kategorie hinzufügen',
      loading: 'Kategorien werden geladen…',
      load_error: (error) => `Kategorien konnten nicht geladen werden. ${error}`,
      menu_delete: 'Löschen',
      menu_hint_preserved: 'beibehalten',
      menu_rename: 'Umbenennen',
      menu_rename_hint_busy: 'zuerst laufende Arbeit abschließen',
      menu_add: 'Kategorie hinzufügen'
    },
    add_dialog: {
      title: 'Kategorie hinzufügen',
      name_label: 'Name',
      name_placeholder: 'z. B. cat',
      name_help_prefix:
        'Buchstaben, Ziffern, Punkte, Bindestriche und Unterstriche. Der Name dient zugleich als Verzeichnisname auf dem Datenträger (z. B. ',
      name_help_code_example: 'datasets/cat/',
      name_help_suffix: ') und als das Klassenlabel, das der Trainer verwendet.',
      submit: 'Hinzufügen',
      error_exact_duplicate: 'Eine Kategorie mit diesem Namen existiert bereits.',
      error_case_insensitive_duplicate: (existingName) =>
        `Steht im Konflikt mit dem vorhandenen „${existingName}“ (Namen sind auf den meisten Dateisystemen nicht case-sensitiv).`
    },
    rename_dialog: {
      title: 'Kategorie umbenennen',
      name_label: 'Name',
      name_help:
        'Der Name dient zugleich als Verzeichnis auf dem Datenträger und als Klassenlabel des Trainers, daher ändert das Umbenennen das Klassenlabel. Bestehende trainierte Modelle behalten ihre alten Labels und werden als veraltet markiert, bis neu trainiert wird.',
      submit: 'Speichern',
      error_mandatory: 'Background Noise wird beibehalten und kann nicht umbenannt werden.',
      error_busy:
        'Laufende Uploads und Löschvorgänge abschließen oder verwerfen, bevor diese Kategorie umbenannt wird.'
    },
    delete_dialog: {
      title: 'Diese Kategorie löschen?',
      body_server:
        'Entfernt den Datensatzordner und jeden darin enthaltenen Slice. Kann nicht rückgängig gemacht werden.',
      body_idb:
        'Entfernt diese Kategorie aus der lokalen Liste. Es wurden keine Slices hochgeladen, daher ändert sich auf dem Gerät nichts.',
      submit: 'Löschen',
      error_fallback: 'Die Kategorie konnte nicht gelöscht werden.',
      error_mandatory_required: 'Background Noise wird beibehalten und kann nicht gelöscht werden.',
      error_not_found: 'Kategorie nicht gefunden.'
    },
    slice_card: {
      aria_select: (filename) => `Slice ${filename} auswählen`,
      aria_deselect: (filename) => `Auswahl von Slice ${filename} aufheben`,
      aria_play: (filename) => `Slice ${filename} abspielen`,
      title_failed: (errorOrUnknown) =>
        `Upload fehlgeschlagen: ${errorOrUnknown}. Mit Rechtsklick erneut versuchen.`,
      title_uploading: (progressPct) => `Wird hochgeladen… ${progressPct}%`,
      title_local: 'Lokal – Upload ausstehend',
      title_multi_click_deselect: 'Klicken, um Auswahl aufzuheben (Esc beendet die Auswahl)',
      title_multi_click_select: 'Klicken, um zur Auswahl hinzuzufügen (Esc beendet die Auswahl)',
      title_playing: 'Wird abgespielt – klicken zum Neustart',
      title_idle: 'Klicken zum Abspielen (Ctrl/Cmd-click zum Auswählen)',
      sr_deleting: (filename) => `Slice ${filename} wird gelöscht`,
      sr_uploading: (progressPct) => `Wird hochgeladen ${progressPct}%`,
      retry_aria: (filename) => `Upload für Slice ${filename} erneut versuchen`,
      retry_title_with_error: (errorMessage) =>
        `Upload fehlgeschlagen: ${errorMessage}. Klicken, um erneut zu versuchen.`,
      retry_title_no_error: 'Upload fehlgeschlagen. Klicken, um erneut zu versuchen.',
      retry_label: 'erneut versuchen',
      select_title: 'Auswählen',
      deselect_title: 'Auswahl aufheben',
      delete_aria: (filename) => `Slice ${filename} löschen`,
      delete_title: 'Slice löschen',
      slice_select_aria: (filename) => `Slice ${filename} auswählen`,
      slice_deselect_aria: (filename) => `Auswahl von Slice ${filename} aufheben`,
      unknown_error: 'unbekannter Fehler'
    },
    trim_waveform: {
      handles_aria: 'Zuschneidegriffe, ziehen, um Anfang und Ende des Slice-Bereichs festzulegen',
      handle_start_aria: 'Zuschneidebeginn',
      handle_end_aria: 'Zuschneideende',
      selection_aria:
        'Auswahlfenster verschieben, ziehen, um beide Zuschneidekanten gemeinsam zu bewegen',
      playback_position_aria: 'Wiedergabeposition',
      value_seconds: (sec) => `${sec} Sekunden`,
      value_seconds_range: (startSec, endSec) => `${startSec} bis ${endSec} Sekunden`
    },
    slice_pane: {
      heading: 'Slices',
      tips_label: 'Tipps zum Slice-Modul',
      tip_audition_title: 'Jeden Slice vor dem Training anhören.',
      tip_audition_body:
        'Eine falsch beschriftete Zeile verzerrt die ganze Klasse – Karten anklicken zum Abspielen, großzügig verwerfen.',
      tip_diversity_title: 'Vielfalt schlägt Menge.',
      tip_diversity_body:
        '10 abwechslungsreiche Aufnahmen (Abstand, Winkel, Hintergrund) trainieren besser als 30 nahezu identische Kopien.',
      quota_above_title: (threshold) =>
        `Über der Mindestanzahl von ${threshold} Slices für das Training.`,
      quota_below_title: (threshold) =>
        `Unter der Mindestanzahl von ${threshold} Slices für das Training. Mehr schneiden, um die Mindestanzahl zu erreichen.`,
      loading: 'Slices werden geladen…',
      load_error: (error) => `Slices konnten nicht geladen werden. ${error}`,
      empty_state_prefix: 'Noch keine Slices. Den Clip im Eingabebereich zuschneiden und auf ',
      empty_state_button: 'Schneiden',
      empty_state_suffix: ' klicken, um dieses Raster zu füllen.',
      select_all_label: 'Alle auswählen',
      deselect_all_label: 'Auswahl aufheben',
      select_all_title: 'Alle Slices auswählen (Cmd/Ctrl+A)',
      deselect_all_title: 'Auswahl aller Slices aufheben (Cmd/Ctrl+A)',
      done_label: 'Fertig',
      done_title: 'Auswahl beenden (Esc)',
      delete_title: 'Die ausgewählten Slices löschen (Del / Backspace)',
      delete_disabled_title: 'Mindestens einen Slice zum Löschen auswählen',
      delete_inflight_title: (count) =>
        `${count} ${count === 1 ? 'Slice' : 'Slices'} werden gelöscht…`,
      delete_inflight_aria: (count) =>
        `${count} ${count === 1 ? 'Slice' : 'Slices'} werden gelöscht`,
      delete_aria_count: (count) =>
        `${count} ausgewählte${count === 1 ? 'n Slice' : ' Slices'} löschen`,
      delete_aria_fallback: 'Ausgewählte Slices löschen',
      delete_label_inflight: (count) => `${count} werden gelöscht…`,
      delete_label_count: (count) => `${count} löschen`,
      delete_label_bare: 'Löschen',
      menu_play: 'Abspielen',
      menu_stop: 'Stoppen',
      menu_retry_upload: 'Upload erneut versuchen',
      menu_select: 'Auswählen',
      menu_deselect: 'Auswahl aufheben',
      menu_select_all: 'Alle auswählen',
      menu_deselect_all: 'Auswahl aufheben',
      menu_done_exit: 'Fertig (Auswahl beenden)',
      menu_retry_failed_in_selection: 'Fehlgeschlagene in Auswahl erneut versuchen',
      menu_delete_batch: (count) => `${count} ${count === 1 ? 'Slice' : 'Slices'} löschen`,
      menu_delete: 'Löschen',
      menu_hint_a: 'Cmd/Ctrl+A',
      menu_hint_esc: 'Esc',
      menu_hint_ctrl_click: 'Ctrl/Cmd-click',
      menu_hint_del_backspace: 'Del / Backspace'
    },
    input_pane: {
      heading: 'Eingabe',
      tips_label: 'Tipps zum Eingabemodul',
      tip_stream_title: 'Den Audiostream des Geräts bevorzugen.',
      tip_stream_body:
        'Die Slices nutzen dasselbe DSP wie die Inferenz, sodass das trainierte Modell nach dem Feintuning keine Verteilungsverschiebung erlebt.',
      tip_environment_title: 'In der Einsatzumgebung aufnehmen.',
      tip_environment_body:
        'Eine saubere Studioaufnahme untertrainiert die Geräuschunterdrückung. Das echte Hintergrundrauschen sollte etwa die Hälfte dessen ausmachen, was das Modell lernen muss.',
      tip_meter_title: 'Die Dezibelanzeige im Bereich von Grün bis Bernstein halten.',
      tip_meter_body:
        'Rot bedeutet Übersteuerung – Informationen gehen verloren, was den Trainer am Lernen hindert.',
      pane_aria: (categoryDisplay) => `Eingabemodul für Kategorie ${categoryDisplay}`,
      source_aria: 'Eingabequelle',
      loudness_aria: 'Pegelanzeige',
      source_microphone_group: 'Mikrofon',
      source_system_default_mic: 'Systemstandard-Mikrofon',
      source_remembered: (label) => `${label} (gemerkt)`,
      source_mic_fallback: (n, idFrag) => `Mikrofon ${n} (${idFrag})`,
      source_mic_remembered_fallback: (idFrag) => `Mikrofon (${idFrag})`,
      source_mic_default_id: 'Standard',
      source_live_stream_group: 'Livestream',
      source_daemon_stream: 'Geräte-Audiostream',
      source_daemon_stream_with_status: (status) => `Geräte-Audiostream · ${status}`,
      drop_zone_title: (cap) =>
        `Eine WAV-Datei hier ablegen (bis zu ${cap}) oder klicken zum Durchsuchen`,
      drop_zone_idle: 'Eine WAV-Datei hierher ziehen und ablegen',
      drop_zone_browse: 'Dateien durchsuchen',
      record_aria_stream: 'Erfassung aus dem Live-Audiostream starten',
      record_aria_mic: 'Aufnahme vom Mikrofon starten',
      record_label: 'Aufnehmen',
      record_title_stream_open: (max) =>
        `Den Live-Audiostream erfassen (stoppt automatisch bei ${max}).`,
      record_title_stream_connecting:
        'Der Geräte-Audiostream verbindet sich. Die Aufnahme ist verfügbar, sobald er geöffnet ist.',
      record_title_stream_closed:
        'Der Geräte-Audiostream ist nicht erreichbar. Prüfen, ob das Gerät läuft.',
      record_title_stream_unsupported:
        'Dieser Browser kann den Live-Audiostream hier nicht decodieren – er benötigt WebCodecs über einen sicheren (HTTPS) Kontext. Diese Seite über das sichere Gateway öffnen oder stattdessen eine WAV-Datei ablegen oder durchsuchen.',
      capture_stop_aria_stream: 'Streamerfassung stoppen',
      capture_stop_aria_mic: 'Aufnahme stoppen',
      capture_stop_label: 'Stoppen',
      capture_discard_label: 'Verwerfen',
      capture_encoding: 'Wird codiert…',
      capture_decoding: 'Wird decodiert…',
      trim_selection_prefix: 'Auswahl:',
      trim_drag_hint: 'Die Griffe auf ≥ 1 s ziehen, um das Schneiden zu aktivieren.',
      trim_projected_slices: (count) => `${count} ${count === 1 ? 'Slice' : 'Slices'} zu je 1 s`,
      trim_unused_label: 'ungenutzt',
      slice_aria_enabled: (count) => `In ${count} ${count === 1 ? 'Slice' : 'Slices'} schneiden`,
      slice_aria_disabled: 'Schneiden (Auswahl muss mindestens 1 Sekunde betragen)',
      slice_title_enabled: (count) =>
        `${count} Slice${count === 1 ? '' : 's'} an den rechten Bereich anhängen`,
      slice_title_disabled: 'Auswahl muss ≥ 1 s betragen, um zu schneiden',
      slice_label_bare: 'Schneiden',
      slice_label_count: (count) => `Schneiden · ${count}`,
      discard_aria: 'Clip verwerfen',
      discard_title: 'Clip verwerfen',
      discard_label: 'Verwerfen',
      play_stop_aria: 'Wiedergabe stoppen',
      play_stop_title: 'Wiedergabe stoppen',
      play_aria: 'Die zugeschnittene Auswahl abspielen',
      play_title: 'Die zugeschnittene Auswahl abspielen',
      export_aria: 'Als WAV herunterladen',
      export_title: 'Als WAV herunterladen',
      error_file_too_large: (size, cap) =>
        `Die Datei ist ${size} groß – die Importgrenze ist ${cap}. Kürzer zuschneiden und neu exportieren, dann erneut ablegen.`,
      error_clip_too_short: (clipSecs) =>
        `Der Clip ist nur ${clipSecs} s lang, das Training benötigt mindestens 1 s pro Clip, daher wird ein kürzerer Clip vollständig ausgeschlossen. Einen Clip von 1 s oder länger importieren oder aufnehmen.`,
      error_only_one_file:
        'Nur eine Datei auf einmal – der Eingabeslot hält nur den jüngsten Clip. Eine einzelne WAV-Datei ablegen.',
      error_only_wav: 'Nur WAV-Dateien werden unterstützt.',
      error_could_not_import: 'Die Datei konnte nicht importiert werden.',
      error_could_not_discard: 'Der Clip konnte nicht verworfen werden.',
      error_could_not_decode_draft: 'Der gespeicherte Entwurf konnte nicht decodiert werden.',
      error_could_not_save_recording: 'Die Aufnahme konnte nicht gespeichert werden.',
      error_could_not_capture_stream: 'Der Stream konnte nicht erfasst werden.',
      error_could_not_slice: 'Der Clip konnte nicht geschnitten werden.',
      error_wav_too_small_for_header:
        'Die Datei ist zu klein für eine WAV (mindestens 12 Bytes für den Header erforderlich).',
      error_wav_missing_riff: 'Keine WAV-Datei (RIFF-Magic fehlt).',
      error_wav_missing_wave: 'Keine WAV-Datei (WAVE-Marker fehlt).',
      error_wav_empty: 'Die Datei ist leer oder zu klein für eine WAV.',
      error_wav_buffer_too_small:
        'WAV-Puffer zu klein (mindestens 44 Bytes für den kanonischen Header erforderlich).',
      error_web_audio_unavailable: 'Die Web Audio API ist in diesem Browser nicht verfügbar.',
      auto_stopped_at_cap: 'Automatisch bei der Längenobergrenze gestoppt.',
      silent_dropped_suffix: (count) =>
        `${count} ${count === 1 ? 'stiller Slice' : 'stille Slices'} übersprungen`
    },
    row: {
      badge_synced: 'Synchronisiert',
      badge_uploading: 'Wird hochgeladen',
      badge_pending: 'Ausstehend',
      badge_failed: 'Fehlgeschlagen',
      badge_not_enough: 'Zu wenige Beispiele',
      badge_not_enough_with_state: (statusLabel) => `Zu wenige Beispiele · ${statusLabel}`,
      title_synced: (tally) => `${tally} Slices auf das Gerät hochgeladen – trainingsbereit.`,
      title_uploading: (tally) => `${tally} Slices, einige werden noch auf das Gerät hochgeladen.`,
      title_pending: (tally) =>
        `${tally} Slices bereit, aber noch nicht auf das Gerät hochgeladen.`,
      title_failed: (tally) =>
        `${tally} Slices, mindestens ein Upload ist fehlgeschlagen. Über die Slice-Karte erneut versuchen oder die fehlgeschlagenen Zeilen verwerfen.`,
      title_not_enough_empty: (missing, tally) =>
        `${missing} weitere Slices hinzufügen, um die Mindestanzahl pro Kategorie zu erreichen (${tally}).`,
      title_not_enough_synced: (tally, missing) =>
        `${tally} Slices hochgeladen, ${missing} weitere hinzufügen, um die Mindestanzahl pro Kategorie zu erreichen.`,
      title_not_enough_uploading: (tally, missing) =>
        `${tally} Slices, einige werden noch hochgeladen. Nach Abschluss sind ${missing} weitere nötig.`,
      title_not_enough_pending: (tally, missing) =>
        `${tally} Slices lokal in der Warteschlange, ${missing} weitere nötig.`,
      actions_aria: (displayName) => `Aktionen für ${displayName}`,
      actions_title: 'Kategorieaktionen',
      actions_title_preserved: 'Beibehalten – Umbenennen und Löschen deaktiviert',
      badge_deleting: 'wird gelöscht'
    }
  },
  training: {
    pane: {
      heading: 'Training',
      subtitle_other_running:
        'Ein anderer Arbeitsbereich trainiert gerade – es läuft immer nur ein Auftrag.',
      subtitle_default:
        'Ein Modell auf dem Datensatz dieses Arbeitsbereichs feinabstimmen – das alte Modell wird automatisch verworfen, sobald das neue eintrifft.',
      readiness_loading: 'Datensatz wird geladen…',
      readiness_no_categories:
        'Eine Vordergrundklasse mit hochgeladenen Slices hinzufügen, um das Training zu starten.',
      readiness_background_short: (need) =>
        `Background Noise benötigt ${need} weitere${need === 1 ? 'n hochgeladenen Slice' : ' hochgeladene Slices'}, um das Training zu starten.`,
      readiness_foreground_short:
        'Mindestens eine Vordergrundklasse benötigt 10 hochgeladene Slices, um das Training zu starten.',
      button_starting: 'Wird gestartet…',
      button_cancel: 'Abbrechen',
      button_cancelling: 'Wird abgebrochen…',
      button_retrain: 'Neu trainieren',
      button_train: 'Modell trainieren',
      button_title_loading: 'Datensatz wird geladen…',
      button_title_not_ready_default: 'Bereitschaftsgrund',
      button_title_form_errors:
        'Die hervorgehobenen Hyperparameterfelder korrigieren, um das Training zu aktivieren.',
      button_title_idle_trained:
        'Ein Modell passt bereits zu dieser Revision – neu trainieren, um andere Hyperparameter oder einen anderen Zufalls-Seed auszuprobieren. Im Abschnitt „Modelle“ unten lässt sich jedes Modell aktivieren.',
      button_title_idle_busy:
        'Ein anderer Arbeitsbereich trainiert gerade – es läuft immer nur ein Auftrag.',
      button_title_idle_ready: 'Ein Modell auf dem Datensatz dieses Arbeitsbereichs trainieren.',
      button_title_starting: 'Trainingsanfrage wird übermittelt…',
      button_title_running: 'Den laufenden Trainingsauftrag abbrechen.',
      button_title_cancelling: 'Wird abgebrochen…',
      summary_chip_epochs: (epochs) => `${epochs} Epochen`,
      summary_chip_no_holdout: 'kein Holdout',
      summary_chip_val: (pctLabel) => `Val. ${pctLabel}`,
      hyperparameters_disclosure_label: 'Hyperparameter',
      start_error_title: 'Training konnte nicht gestartet werden'
    },
    form: {
      epochs_label: 'Epochen',
      batch_size_label: 'Batchgröße',
      learning_rate_label: 'Lernrate',
      validation_split_label: 'Validierungsanteil',
      validation_split_hint: '· 0 zum Deaktivieren',
      seed_label: 'Seed',
      seed_hint: '· leer für vom Daemon gewählte Entropie',
      seed_placeholder: '(optional)'
    },
    progress: {
      submitting: 'Wird übermittelt…',
      job_short_id: (shortId) => `Auftrag ${shortId}…`,
      train_loss_label: 'Train.-Verlust',
      train_acc_label: 'Train.-Genau.',
      val_acc_label: 'Val.-Genau.',
      val_acc_disabled_label: 'Val.-Genau. · deaktiviert',
      em_dash: ' – '
    },
    logs: {
      heading: 'Protokolle',
      entry_count: (count) => `${count} ${count === 1 ? 'Eintrag' : 'Einträge'}`,
      waiting_first_message: 'Warten auf die erste Meldung…'
    },
    chart: {
      waiting_first_epoch: 'Warten auf die erste Epoche…',
      legend_loss: 'Verlust',
      legend_train: 'Training',
      legend_val: 'Validierung',
      tooltip_epoch: 'Epoche',
      tooltip_loss: 'Verlust',
      tooltip_train: 'Training',
      tooltip_val: 'Validierung',
      chart_aria: 'Diagramm der Trainingsmetriken'
    },
    history: {
      heading: 'Verlauf',
      keeps_last: (cap) => `behält die letzten ${cap} Läufe`,
      retention_title: (cap) =>
        `Der Daemon behält pro Arbeitsbereich die ${cap} jüngsten Trainingsprotokolldateien; ältere JSONL-Spuren werden entfernt, wenn ein neuer Lauf beginnt. Der veröffentlichte Modelldatensatz (im Abschnitt „Modelle“ unten) ist davon nicht betroffen – nur die JSONL-Spur wird entfernt.`,
      empty_state_prefix: 'Noch keine Trainingsläufe für diesen Arbeitsbereich. Auf ',
      empty_state_button: 'Modell trainieren',
      empty_state_suffix: ' klicken, um einen zu starten.',
      hide_older_label: 'Ältere Läufe ausblenden',
      show_older_label: (count) => `${count} ältere${count === 1 ? 'n Lauf' : ' Läufe'} anzeigen`,
      hide_older_title: 'Den Abschnitt mit älteren Läufen auf die letzten zwei zurückklappen.',
      show_older_title:
        'Ältere Trainingsläufe für diesen Arbeitsbereich einblenden, in Stapeln zu je 5 paginiert.',
      load_more_label: (count) => `${count} weitere laden`,
      load_more_title: 'Den nächsten Stapel älterer Trainingsläufe vom Gerät abrufen.',
      menu_delete: 'Löschen',
      menu_deleting: 'Wird gelöscht…',
      menu_hint_train_active: 'Training aktiv',
      menu_hint_live: 'live',
      delete_error_title: 'Trainingsprotokoll konnte nicht gelöscht werden'
    },
    history_item: {
      time_started_pre_ack: 'gestartet',
      time_started: (relative) => `gestartet ${relative}`,
      time_finished: (relative) => relative,
      time_title_started: (absolute) => `gestartet ${absolute}`,
      time_title_finished: (absolute) => `beendet ${absolute}`,
      detail_epoch: (current, total) => `Epoche ${current}/${total}`,
      detail_class_count: (count) => `${count} ${count === 1 ? 'Klasse' : 'Klassen'}`,
      detail_val_acc: (pctLabel) => `Genau. ${pctLabel}`,
      detail_train_acc: (pctLabel) => `Train. ${pctLabel}`,
      detail_stopped_at: (stageLabel) => `gestoppt bei ${stageLabel}`
    },
    summary: {
      completed_aria: 'Zusammenfassung des abgeschlossenen Laufs',
      failed_aria: 'Zusammenfassung des fehlgeschlagenen Laufs',
      cancelled_aria: 'Zusammenfassung des abgebrochenen Laufs',
      duration_label: 'Dauer',
      epochs_label: 'Epochen',
      best_val_at: (epoch) => `Beste Val.-Genau. @ ${epoch}`,
      final_train_acc_label: 'Finale Train.-Genau.',
      classes_label: 'Klassen',
      stopped_at_label: 'Gestoppt bei',
      cancelled_at_label: 'Abgebrochen bei',
      epochs_tooltip_full: 'Die volle konfigurierte Epochenanzahl wurde durchlaufen.',
      epochs_tooltip_partial: 'Beobachtete Epochen vs. konfigurierte Epochenanzahl.',
      after_epochs: (run, total) => `nach ${run}/${total} Epochen`,
      failed_no_diagnostic: 'Keine Diagnose aufgetaucht. Die Daemon-Protokolle auf Details prüfen.',
      cancelled_default_reason: 'Am nächsten Trainings-Checkpoint gestoppt.',
      failed_default: 'Training fehlgeschlagen.'
    },
    stage: {
      prepare: 'Vorbereiten',
      dataset_scan: 'Datensatz wird gescannt',
      feature_extract: 'Merkmale werden extrahiert',
      train: 'Training',
      save: 'Speichern',
      publish: 'Veröffentlichen'
    },
    state: {
      running: 'läuft',
      completed: 'abgeschlossen',
      failed: 'fehlgeschlagen',
      cancelled: 'abgebrochen'
    },
    state_submitting: 'wird übermittelt',
    store_log: {
      seed_submitted: 'Übermittelt, warte darauf, dass das Gerät Ereignisse zu senden beginnt…',
      seed_recovered: 'Einen laufenden Trainingsauftrag vom Gerät wiederhergestellt.',
      job_submitted: (backbone) => `Auftrag übermittelt · Backbone ${backbone}`,
      job_running: 'Auftrag läuft',
      phase_prefix: (stageLabel) => `Phase: ${stageLabel}`,
      job_failed: (stageLabel, error) => `Auftrag fehlgeschlagen bei ${stageLabel} · ${error}`,
      job_cancelled: (stageLabel) => `Auftrag abgebrochen bei ${stageLabel}`,
      job_cancelled_shutdown: (stageLabel) =>
        `Auftrag abgebrochen bei ${stageLabel} (Daemon heruntergefahren)`,
      scanned_dataset: (nClasses, nExamples) =>
        `Datensatz gescannt · ${nClasses} ${nClasses === 1 ? 'Klasse' : 'Klassen'} · ${nExamples} Beispiele`,
      features_extracted: (kept, dropped, elapsedSec) => {
        const droppedSuffix = dropped > 0 ? ` · ${dropped} verworfen` : '';
        return `Merkmale extrahiert · ${kept} behalten${droppedSuffix} · ${elapsedSec}s`;
      },
      train_split: (trainN, valN) =>
        `Trainingsaufteilung · ${trainN} Training · ${valN} Validierung`,
      epoch_completed: (epoch, epochs, lossLabel, trainAccLabel, valAccLabel) => {
        const valPart = valAccLabel !== null ? ` · Val. ${valAccLabel}` : '';
        return `Epoche ${epoch}/${epochs} · Verlust ${lossLabel} · Train. ${trainAccLabel}${valPart}`;
      },
      train_loop_done: (epochsRun, elapsedSec, bestValAccLabel, bestEpoch) => {
        const bestPart =
          bestValAccLabel !== null && bestEpoch !== null
            ? ` · beste Val.-Genau. ${bestValAccLabel} @ Epoche ${bestEpoch}`
            : '';
        return `Trainingsschleife abgeschlossen · ${epochsRun} ${epochsRun === 1 ? 'Epoche' : 'Epochen'} in ${elapsedSec}s${bestPart}`;
      },
      head_published: (headId, size, nClasses, rev) =>
        `Modell veröffentlicht · ${headId} · ${size} · ${nClasses} ${nClasses === 1 ? 'Klasse' : 'Klassen'} · rev ${rev}`,
      job_completed: (labelsList) =>
        labelsList.length > 0 ? `Auftrag abgeschlossen · ${labelsList}` : 'Auftrag abgeschlossen'
    }
  },
  deploy: {
    pane: {
      heading: 'Bereitstellung',
      description:
        'Trainierte Modelle durchsuchen und auswählen, um sie nahtlos per Hot-Swap in die Echtzeit-Inferenz zu übernehmen.',
      pill_deployed: 'Bereitgestellt',
      pill_deployed_title:
        'Ein in diesem Arbeitsbereich trainiertes Modell ist das Laufzeitmodell.',
      pill_default: 'Standard',
      pill_default_title: 'Das integrierte Standardmodell läuft.',
      pill_standby: 'Standby',
      pill_standby_title:
        'Ein Modell aus einem anderen Arbeitsbereich ist das Laufzeitmodell. Dieser Arbeitsbereich ist im Standby. Wird hier eines bereitgestellt, ersetzt es jenes.',
      pill_detached: 'Abgetrennt',
      pill_detached_title:
        'Der Arbeitsbereich, der das Laufzeitmodell erzeugt hat, wurde gelöscht. Das Modell läuft weiterhin.',
      config_disclosure_label: 'Eingabe- & Inferenzkonfiguration',
      config_chip_freq: (hzLabel) => `freq ${hzLabel} Hz`,
      config_chip_top_k: (topK) => `top-k ${topK}`
    },
    heads_table: {
      heading: 'Modelle',
      count_label: (count) => `${count} ${count === 1 ? 'Modell' : 'Modelle'}`,
      // Split off the bare count so it can collapse on a narrow card; carries its own leading comma.
      count_retained: (retainedCap) => `, neueste ${retainedCap} behalten`,
      revert_to_default: 'Auf Standard zurücksetzen',
      revert_to_id: (shortId) => `Auf ${shortId} zurücksetzen`,
      revert_title: 'Das zuvor laufende Modell erneut bereitstellen',
      default_row_headline: 'Standard',
      default_row_description: 'Integrierter Fallback, immer verfügbar.',
      default_active_title: 'Das integrierte Standardmodell ist derzeit bereitgestellt.',
      default_aria_active: 'Standardmodell ist aktiv',
      default_aria_deploy: 'Standardmodell bereitstellen',
      default_title_active: 'Das Standardmodell ist bereits bereitgestellt',
      default_title_deploying: 'Wird bereitgestellt…',
      default_title_busy: 'Ein anderes Modell in dieser Liste ist beschäftigt',
      default_title_idle: 'Auf das integrierte Standardmodell zurücksetzen',
      menu_deploy: 'Bereitstellen',
      menu_export: 'Als ALPKG exportieren',
      menu_exporting: 'Wird exportiert…',
      menu_delete: 'Löschen',
      menu_hint_active: 'aktiv',
      menu_hint_deployed: 'bereitgestellt',
      error_deploy_head: 'Modell konnte nicht bereitgestellt werden',
      error_export_head: 'Modell konnte nicht exportiert werden',
      error_deploy_default: 'Standardmodell konnte nicht bereitgestellt werden'
    },
    head_row: {
      pill_latest: 'Neueste',
      pill_latest_title:
        'Jüngstes Modell, das auf der aktuellen Revision des Arbeitsbereichs trainiert wurde.',
      pill_active: 'Aktiv',
      pill_active_title: 'Dieses Modell ist derzeit in der Inferenzpipeline bereitgestellt.',
      // Fixed-width single-string meta for the model-card popover and delete-confirm card.
      meta_line: (size, classCount, rev, relative) =>
        `${size} · ${classCount} ${classCount === 1 ? 'Klasse' : 'Klassen'} · rev ${rev} · ${relative}`,
      // Row meta renders segment-by-segment so size/rev can drop as the row narrows (size/age come
      // from format utils, not the catalog).
      meta_classes: (classCount) => `${classCount} ${classCount === 1 ? 'Klasse' : 'Klassen'}`,
      meta_rev: (rev) => `rev ${rev}`,
      row_aria_deployed: (shortId) => `Bereitgestelltes Modell ${shortId}`,
      row_aria_deploy: (shortId) => `Modell ${shortId} bereitstellen`,
      row_title_deployed: 'Dieses Modell ist bereits bereitgestellt',
      row_title_deploying: 'Wird bereitgestellt…',
      row_title_exporting: 'Wird exportiert…',
      row_title_busy: 'Ein anderes Modell in dieser Liste ist beschäftigt',
      row_title_idle:
        'Klicken, um dieses Modell per Hot-Swap in die Inferenzpipeline zu übernehmen',
      export_title_exporting: 'Wird exportiert…',
      export_title_idle: 'Dieses Modell als ALPKG-Archiv exportieren',
      export_aria_exporting: (shortId) => `Modell ${shortId} wird exportiert`,
      export_aria_idle: (shortId) => `Modell ${shortId} exportieren`,
      info_title: 'Modellkarte ansehen',
      info_aria: (shortId) => `Modellkarte für ${shortId} ansehen`
    },
    inference_preview: {
      heading: 'Vorschau',
      off_title: 'Vorschau ist aus',
      off_description:
        'Die Vorschau starten, um Spektrogramm und top-k-Stream des bereitgestellten Modells zu beobachten.',
      start_button: 'Vorschau starten'
    },
    info_dialog: {
      title_with_id: (shortId) => `Modellkarte · ${shortId}`,
      loading: 'Klassen werden geladen…',
      error_title: 'Klassen konnten nicht geladen werden',
      retry: 'Erneut versuchen',
      classes_heading: 'Klassen',
      class_labels_aria: 'Trainierte Klassenlabels'
    },
    delete_dialog: {
      title: 'Dieses Modell löschen?',
      body: 'Entfernt die Bytes des trainierten Modells und sein Manifest. Der Datensatz und alle anderen Modelle bleiben erhalten. Kann nicht rückgängig gemacht werden.',
      submit: 'Löschen'
    }
  },
  workspace: {
    list: {
      title: 'Arbeitsbereiche',
      at_cap_subtitle: (max) =>
        `Das Limit von ${max} Arbeitsbereichen ist erreicht. Einen löschen, bevor ein weiterer erstellt wird.`,
      default_subtitle:
        'Jeder Arbeitsbereich enthält einen beschrifteten Datensatz und alle daraus trainierten Modelle.',
      daemon_unavailable_title: 'Gerät nicht verfügbar',
      loading: 'Arbeitsbereiche werden geladen…',
      empty_title: 'Noch keine Arbeitsbereiche',
      empty_description:
        'In Arbeitsbereichen leben Aufnahmen, beschriftete Beispiele und trainierte Modelle. Einen erstellen, um loszulegen.',
      selected_count_aria: (count) => `${count} ausgewählt`,
      new_button_label: 'Neuer Arbeitsbereich',
      new_button_aria: 'Neuer Arbeitsbereich',
      new_at_cap_label: (count, max) => `Limit erreicht · ${count}/${max}`,
      new_at_cap_title: 'Limit erreicht. Zuerst einen Arbeitsbereich löschen.',
      import_button_label: 'Importieren',
      import_button_aria: 'Arbeitsbereich importieren',
      import_button_title: 'Arbeitsbereich aus einem ALPKG- oder TFJS-Bundle importieren',
      select_button_label: 'Auswählen',
      done_button_label: 'Fertig',
      select_all_label: 'Alle auswählen',
      deselect_all_label: 'Auswahl aufheben',
      bulk_delete_label_count: (count) => `${count} löschen`,
      bulk_delete_label_bare: 'Löschen',
      bulk_delete_aria_count: (count) =>
        `${count} ${count === 1 ? 'Arbeitsbereich' : 'Arbeitsbereiche'} löschen`,
      bulk_delete_aria_fallback: 'Ausgewählte Arbeitsbereiche löschen',
      menu_open: 'Öffnen',
      menu_rename: 'Umbenennen',
      menu_export: 'Exportieren',
      menu_delete: 'Löschen',
      menu_select_one: 'Auswählen',
      menu_deselect_one: 'Auswahl aufheben',
      menu_select_all: 'Alle auswählen',
      menu_deselect_all: 'Auswahl aufheben',
      menu_select_workspaces: 'Arbeitsbereiche auswählen',
      menu_done_exit: 'Fertig (Auswahl beenden)',
      menu_new: 'Neuer Arbeitsbereich',
      menu_new_at_cap: (max) => `Neuer Arbeitsbereich (Limit von ${max})`,
      menu_import: 'Arbeitsbereich importieren'
    },
    detail: {
      back_link: '← Arbeitsbereiche',
      loading: 'Arbeitsbereich wird geladen…',
      not_found_title: 'Arbeitsbereich nicht gefunden',
      not_found_description:
        'Möglicherweise wurde er in einem anderen Tab oder direkt über das Gerät gelöscht. Zur Liste zurückkehren, um zu sehen, was noch vorhanden ist.',
      back_to_list_button: 'Zurück zu den Arbeitsbereichen',
      load_error_title: 'Dieser Arbeitsbereich konnte nicht geladen werden',
      created_label: (relative) => `erstellt ${relative}`,
      rev_label: (rev) => `rev ${rev}`,
      modified_label: (relative) => `geändert ${relative}`,
      live_pill_title:
        'Durch einen kürzlichen Upload fortgeschritten. Neu laden, um den Änderungszeitstempel zu aktualisieren.',
      live_pill: 'live',
      menu_rename: 'Umbenennen',
      menu_export: 'Exportieren',
      menu_import: 'Importieren',
      menu_delete: 'Löschen',
      menu_back_to_list: 'Zurück zu den Arbeitsbereichen'
    },
    create_dialog: {
      title: 'Neuer Arbeitsbereich',
      name_label: 'Name',
      name_placeholder: 'my-workspace',
      name_help:
        'Bis zu 128 Zeichen. Keine Schrägstriche oder Steuerzeichen. Der Name ist die einzige sichtbare Kennung, daher etwas Einprägsames wählen.',
      submit: 'Erstellen'
    },
    rename_dialog: {
      title: 'Arbeitsbereich umbenennen',
      name_label: 'Name',
      name_help:
        'Bis zu 128 Zeichen. Keine Schrägstriche oder Steuerzeichen. Das Umbenennen treibt die Revision des Arbeitsbereichs nicht voran – Kategorien, Slices und Modelle bleiben, wie sie sind.',
      submit: 'Speichern'
    },
    delete_dialog: {
      title: 'Diesen Arbeitsbereich löschen?',
      body: 'Entfernt den Datensatz, alle trainierten Modelle und Protokolle. Kann nicht rückgängig gemacht werden.',
      submit: 'Löschen'
    },
    bulk_delete_dialog: {
      title_count: (count) =>
        `${count} ${count === 1 ? 'Arbeitsbereich' : 'Arbeitsbereiche'} löschen?`,
      body: 'Entfernt von jedem Arbeitsbereich den Datensatz, die trainierten Modelle und Protokolle. Kann nicht rückgängig gemacht werden.',
      submit_count: (count) => `${count} löschen`
    },
    tool_island: {
      aria_label: 'Arbeitsbereichsaktionen',
      rename_aria: 'Arbeitsbereich umbenennen',
      rename_title: 'Arbeitsbereich umbenennen',
      export_aria: 'Arbeitsbereich exportieren',
      export_title: 'Arbeitsbereich exportieren (Datensätze + Modelle)',
      import_aria: 'Arbeitsbereich importieren',
      import_title: 'Arbeitsbereich importieren (Datensätze + Modelle)'
    },
    card: {
      created_label: (relative) => `erstellt ${relative}`,
      select_aria: (name) => `Arbeitsbereich ${name} auswählen`,
      rename_aria: (name) => `Arbeitsbereich ${name} umbenennen`,
      deleting: 'wird gelöscht'
    },
    import_dialog: {
      title_into: (workspaceName) => `Importieren in · ${workspaceName}`,
      title_fallback: 'Importieren',
      step_indicator: (current, total) => `Schritt ${current} von ${total}`,
      pipeline_error_title: 'Import fehlgeschlagen',
      error_invalid_state: 'Inkonsistenter Dialogzustand – kein Archiv zum Importieren.',
      pick_file: {
        drop_zone_title_attr:
          'Ein ALPKG-Archiv oder ein TFJS-Bundle hier ablegen oder klicken zum Durchsuchen',
        reading: 'Wird gelesen…',
        drop_zone_tfjs_staging: 'Weitere Dateien ablegen, um das TFJS-Bundle zu vervollständigen',
        drop_zone_idle: 'Ein ALPKG-Archiv oder ein TFJS-Bundle hierher ziehen und ablegen',
        browse_button: 'Dateien durchsuchen',
        error_empty_drop: 'Ein ALPKG-Archiv oder ein TFJS-Bundle ablegen.',
        error_multi_alpkg: (count) =>
          `Ein ALPKG-Archiv auf einmal auswählen – ausgewählt wurden ${count}.`,
        error_mixed_archive:
          'Ein ALPKG-Archiv muss allein ausgewählt werden, nicht gemischt mit anderen Dateien.',
        error_file_count_cap: (max, picked) =>
          `Höchstens ${max} Dateien auf einmal ablegen oder auswählen – ausgewählt wurden ${picked}.`,
        error_single_too_large: (name, size, cap) =>
          `„${name}“ ist ${size} groß – die Grenze pro Datei ist ${cap}.`,
        error_total_too_large: (total, cap) =>
          `Die Auswahl umfasst insgesamt ${total} – die Grenze pro Ablage ist ${cap}.`,
        error_tfjs_merged_file_count: (mergedCount, cap) =>
          `Der Staging-Satz käme auf insgesamt ${mergedCount} Dateien – die Grenze ist ${cap}. Leeren und ein kleineres Bundle erneut ablegen.`,
        error_tfjs_merged_bytes: (mergedBytes, cap) =>
          `Der Staging-Satz käme auf insgesamt ${mergedBytes} – die Grenze ist ${cap}. Leeren und ein kleineres Bundle erneut ablegen.`,
        staged_files_heading: 'Bereitgestellte Dateien',
        staged_files_count: (count) => `${count} ${count === 1 ? 'Datei' : 'Dateien'}`,
        clear_button: 'Leeren',
        error_could_not_read_archive: 'Das Archiv konnte nicht gelesen werden.',
        error_could_not_read_file: 'Die Datei konnte nicht gelesen werden.',
        error_could_not_read_picked_files: 'Die ausgewählten Dateien konnten nicht gelesen werden.',
        error_could_not_read_model_json: 'model.json konnte nicht gelesen werden.',
        tfjs_diag_empty_drop: 'Die TFJS-Bundle-Dateien ablegen (model.json + Shards + Labels).',
        tfjs_diag_no_model_json: 'Kein "model.json" in der Ablage. Das TFJS-Manifest beifügen.',
        tfjs_diag_ambiguous_model_json: (count) =>
          `Mehrdeutiges Bundle: ${count} Dateien mit dem Namen "model.json".`,
        tfjs_diag_multiple_labels_txt:
          'Mehrere "labels.txt"-Dateien in der Ablage. Genau eine beifügen.',
        tfjs_diag_multiple_metadata_json:
          'Mehrere "metadata.json"-Dateien in der Ablage. Genau eine beifügen.',
        tfjs_diag_both_labels:
          'Sowohl "labels.txt" als auch "metadata.json" angegeben. Nur eine Labelquelle beifügen.',
        tfjs_diag_no_labels:
          'Keine Labeldatei angegeben. "labels.txt" oder "metadata.json" beifügen.',
        tfjs_diag_shard_collision_one: (quotedName) =>
          `Zwei bereitgestellte Dateien teilen sich den Shard-Namen ${quotedName}. Das Staging leeren und nur die beabsichtigte Kopie ablegen.`,
        tfjs_diag_shard_collision_many: (quotedNames, overflow) =>
          `Mehrere bereitgestellte Dateien teilen sich von "model.json" referenzierte Shard-Namen: ${quotedNames}${overflow ? '…' : ''}. Das Staging leeren und nur die beabsichtigten Kopien ablegen.`,
        tfjs_diag_missing_shard_one: (quotedName) =>
          `Fehlender Shard ${quotedName}, referenziert von "model.json".`,
        tfjs_diag_missing_shards_many: (count, quotedNames, overflow) =>
          `${count} fehlende Shards, referenziert von "model.json": ${quotedNames}${overflow ? '…' : ''}.`,
        tfjs_diag_model_json_not_json: 'model.json ist kein gültiges JSON.',
        tfjs_diag_model_json_not_object: 'model.json ist kein JSON-Objekt.',
        tfjs_diag_model_json_no_manifest: 'model.json fehlt das "weightsManifest"-Array.',
        tfjs_diag_model_json_no_shards: 'model.json deklariert keine Shard-Dateien.'
      },
      pick_target: {
        section_label: 'Importieren in',
        mode_radio_aria: 'Zielarbeitsbereich-Modus',
        mode_use_existing: 'Vorhandenen verwenden',
        mode_create_new: 'Neu erstellen',
        no_workspaces_prefix: 'Noch keine Arbeitsbereiche – wechseln zu ',
        no_workspaces_link_label: 'Neu erstellen',
        no_workspaces_suffix: ', um einen anzulegen.',
        workspace_list_aria: 'Einen Zielarbeitsbereich auswählen',
        workspace_created_label: (relative) => `erstellt ${relative}`,
        create_name_placeholder: 'my-imported-workspace',
        create_will_carry_tags: (tagsCsv) => `Übernimmt Tags aus der Quelle: ${tagsCsv}`,
        alpkg_source_card_title: (name, id) => `${name} (${id})`,
        alpkg_source_created_label: (relative) => `erstellt ${relative}`,
        alpkg_source_rev_label: (rev) => `rev ${rev}`,
        alpkg_source_modified_label: (relative) => `geändert ${relative}`,
        tfjs_bundle_card_title: 'TFJS-Bundle',
        tfjs_show_labels_aria: 'Klassenlabels anzeigen',
        tfjs_meta_strip: (size, shards, classes, labelsFileName) => {
          const classesPart =
            classes !== null && classes > 0
              ? ` · ${classes} ${classes === 1 ? 'Klasse' : 'Klassen'}`
              : '';
          const shardsPart = ` · ${shards} ${shards === 1 ? 'Shard' : 'Shards'}`;
          const labelsPart = labelsFileName !== null ? ` · über ${labelsFileName}` : '';
          return `${size}${classesPart}${shardsPart}${labelsPart}`;
        }
      },
      summary: {
        datasets_heading: 'Datensätze',
        datasets_counter: (selected, total) => `${selected} / ${total} ausgewählt`,
        checking_categories: 'Zielarbeitsbereich wird auf vorhandene Kategorien geprüft…',
        slice_count: (count) => `${count} ${count === 1 ? 'Slice' : 'Slices'}`,
        rename_button_aria: 'Zielkategorie umbenennen',
        rename_button_title_default: 'Zielkategorie umbenennen',
        mode_aria: (modeLabel) => `Importaktion: ${modeLabel}`,
        mode_menu_aria: (sourceName) => `Importaktion für ${sourceName}`,
        rename_popover_aria: (sourceName) => `Zielkategorie für ${sourceName} umbenennen`,
        rename_popover_heading: 'Umbenennen',
        rename_chips_heading: 'Oder vorhandene wiederverwenden',
        heads_heading: 'Modelle',
        heads_cap_tooltip: (cap) =>
          `Bis zu ${cap} Modelle pro Arbeitsbereich. Ältere nicht aktive Modelle fallen weg, wenn neue eintreffen – aus einem erneuten Training oder einem Import.`,
        heads_counter: (selected, existingInTarget, cap, activeInTarget) => {
          const active = activeInTarget > 0 ? ` · aktive ${activeInTarget} angeheftet` : '';
          return `${selected} ausgewählt · Ziel ${existingInTarget} / ${cap}${active}`;
        },
        checking_heads: 'Zielmodelle werden geprüft…',
        displacement_warning: (displaced, cap) =>
          `Der Import verdrängt ${displaced === 1 ? 'das älteste nicht aktive Modell' : `die ${displaced} ältesten nicht aktiven Modelle`}, um in die Obergrenze von ${cap} Modellen zu passen.`,
        head_exists_badge_title:
          'Ein Modell mit dieser ID existiert bereits im Zielarbeitsbereich.',
        head_exists_badge: 'Vorhanden',
        head_show_details_aria: 'Modelldetails anzeigen',
        head_class_count: (count) => `${count} ${count === 1 ? 'Klasse' : 'Klassen'}`,
        head_info_metadata: (size, classes, revisionId, createdAbsolute, createdRelative) => {
          const classesPart =
            classes !== null ? ` · ${classes} ${classes === 1 ? 'Klasse' : 'Klassen'}` : '';
          const revPart = revisionId !== null ? ` · rev ${revisionId}` : '';
          const createdPart =
            createdAbsolute !== null && createdRelative !== null
              ? ` · ${createdAbsolute} (${createdRelative})`
              : '';
          return `${size}${classesPart}${revPart}${createdPart}`;
        },
        head_classes_heading: 'Klassen',
        head_class_labels_aria: 'Trainierte Klassenlabels',
        archive_errors_summary: (count) =>
          `${count} ${count === 1 ? 'Archiveintrag' : 'Archiveinträge'} übersprungen`,
        tfjs_ignored_unknown: (count, fileList) =>
          `${count} nicht erkannte${count === 1 ? ' Datei' : ' Dateien'} ignoriert: ${fileList}`,
        tfjs_classes_popover_heading: (count) => `Klassen (${count})`,
        tfjs_classes_popover_aria: 'Klassenlabels',
        head_disabled_reasons: {
          loading: 'Zielmodelle werden geladen…',
          exists: 'Im Ziel bereits vorhanden. Ein anderes Modell auswählen.',
          ceiling: 'Auswahllimit erreicht. Zuerst eine andere Zeile abwählen.'
        }
      },
      modes: {
        new: 'Neu',
        merge: 'Zusammenführen',
        replace: 'Ersetzen',
        skip: 'Überspringen'
      },
      mode_tooltips: {
        new: 'Die Kategorie von Grund auf mit den Slices des Archivs erstellen.',
        merge:
          'Archiv-Slices über die vorhandene Kategorie hochladen. Slices mit gleichem SHA256 überschreiben sich selbst, neue kommen zum Satz hinzu.',
        replace:
          'Die vorhandene Kategorie (und jeden enthaltenen Slice) löschen, dann aus dem Archiv hochladen.',
        skip: 'Diese Kategorie nicht importieren.'
      },
      mode_disabled_reasons: {
        new_exists:
          'Eine Kategorie mit diesem Zielnamen existiert bereits. „Zusammenführen“ wählen, um Slices hinzuzufügen, oder „Ersetzen“, um zu löschen und neu zu importieren.',
        merge_missing:
          'Es gibt keine vorhandene Kategorie mit diesem Zielnamen. „Neu“ wählen, um eine zu erstellen.',
        replace_missing:
          'Es gibt keine vorhandene Kategorie mit diesem Zielnamen. „Neu“ wählen, um eine zu erstellen.'
      },
      running: {
        progress_replacing_categories: (categoryDisplay, done, total) => {
          const cat = categoryDisplay !== null ? ` · ${categoryDisplay}` : '';
          return `Kategorien werden ersetzt${cat} · ${done} / ${total}`;
        },
        progress_uploading_datasets: (categoryDisplay, done, total) => {
          const cat = categoryDisplay !== null ? ` · ${categoryDisplay}` : '';
          if (typeof done === 'number' && typeof total === 'number') {
            return `Slices werden hochgeladen${cat} · ${done} / ${total}`;
          }
          return `Slices werden hochgeladen${cat}`;
        },
        progress_importing_heads: (index1, total, subPhase) => {
          const sub = subPhase !== null ? ` (${subPhase})` : '';
          return `Modell ${index1} / ${total} wird importiert${sub}`;
        },
        progress_uploading_tfjs: (done, total) =>
          `TFJS-Dateien werden hochgeladen · ${done} / ${total}`,
        progress_converting_tfjs: 'TFJS-Bundle wird konvertiert…',
        ds_pending: 'Ausstehend',
        ds_replacing: 'Wird ersetzt',
        ds_uploading_counter: (uploaded, total) => `${uploaded} / ${total}`,
        ds_done_uploaded: (uploaded) => `${uploaded} hochgeladen`,
        ds_failed_count: (failed) => `${failed} fehlgeschlagen`,
        ds_failed_label: 'Fehlgeschlagen',
        ds_failed_title_count: (failed) =>
          `${failed} ${failed === 1 ? 'Slice' : 'Slices'} konnten nicht hochgeladen werden`,
        head_queued: 'In Warteschlange',
        head_skipped_badge_title:
          'Die Modell-ID existiert bereits auf dem Datenträger und der Orchestrator hat sie übersprungen (idempotenter Re-Import).',
        head_per_log_not_started:
          'Noch nicht gestartet – Protokollzeilen erscheinen, sobald der Import dieses Modells beginnt.',
        head_per_log_no_events: 'Keine Ereignisse aufgezeichnet.',
        log_count: (count) => `${count} ${count === 1 ? 'Protokoll' : 'Protokolle'}`
      },
      head_phase: {
        queued: 'In Warteschlange',
        uploading_files: 'Dateien werden hochgeladen',
        starting_convert: 'Konvertierung wird gestartet',
        converting: 'Wird konvertiert',
        cleaning_up: 'Wird aufgeräumt',
        done: 'Fertig',
        failed: 'Fehlgeschlagen'
      },
      head_outcome: {
        imported: 'Importiert',
        replaced: 'Ersetzt',
        skipped: 'Übersprungen',
        failed: 'Fehlgeschlagen'
      },
      convert_stage: {
        prepare: 'Vorbereiten',
        read_manifest: 'Manifest wird gelesen',
        validate_manifest: 'Manifest wird validiert',
        verify_mpk: 'MPK wird verifiziert',
        stage_mpk: 'MPK wird bereitgestellt',
        read_model_json: 'model.json wird gelesen',
        stage_shards: 'Shards werden bereitgestellt',
        extract_weights: 'Gewichte werden extrahiert',
        read_labels: 'Labels werden gelesen',
        stage_head_mpk: 'Modell-MPK wird bereitgestellt',
        publish_head: 'Modell wird veröffentlicht'
      },
      convert_event: {
        job_submitted: (converter) => `Auftrag über ${converter} übermittelt`,
        job_running: 'Auftrag läuft',
        phase: (stageLabel) => `Phase: ${stageLabel}`,
        manifest_validated: (classes) => `Manifest validiert · ${classes} Klassen`,
        mpk_verified: (size) => `MPK verifiziert · ${size}`,
        weights_extracted: (classes, inDim) =>
          `Gewichte extrahiert · ${classes} Klassen · ${inDim} in_dim`,
        labels_loaded: (labels) => `Labels geladen · ${labels} Labels`,
        head_published: (idempotentSkip) =>
          `Modell veröffentlicht${idempotentSkip ? ' (bereits auf dem Datenträger, übersprungen)' : ''}`,
        job_completed: (classes) => `Auftrag abgeschlossen · ${classes} Klassen`,
        job_failed: (category, error) => `Auftrag fehlgeschlagen · ${category} · ${error}`
      },
      done: {
        conflict_detail: (storedSha8, incomingSha8) =>
          `Das Ziel hält bereits ein Modell mit dieser ID, aber einem anderen SHA256 (${storedSha8} vs. eingehend ${incomingSha8}).`,
        retry_button: 'Vorhandenes ersetzen & erneut versuchen'
      },
      footer: {
        cancel: 'Abbrechen',
        back: 'Zurück',
        next: 'Weiter',
        import: 'Importieren',
        importing: 'Wird importiert…',
        back_to_selection: 'Zurück zur Auswahl',
        done: 'Fertig'
      }
    },
    export_dialog: {
      title: (workspaceName) => `Arbeitsbereich exportieren · ${workspaceName}`,
      load_error_title: 'Dieser Arbeitsbereich konnte nicht geladen werden',
      loading: 'Arbeitsbereich wird geladen…',
      nothing_to_export:
        'Dieser Arbeitsbereich hat noch keine Kategorien und keine Modelle – nichts zu exportieren.',
      datasets_heading: 'Datensätze',
      heads_heading: 'Modelle',
      select_all: 'Alle auswählen',
      deselect_all: 'Auswahl aufheben',
      row_empty: 'leer',
      row_slice_count: (count) => `${count} ${count === 1 ? 'Slice' : 'Slices'}`,
      head_meta_title: (size, classCount) =>
        `${size} · ${classCount} ${classCount === 1 ? 'Klasse' : 'Klassen'}`,
      head_meta_classes: (count) => `${count} ${count === 1 ? 'Klasse' : 'Klassen'}`,
      pending_warning:
        'In der Auswahl noch hochladende oder ausstehende Slices werden ausgeschlossen – nur Slices auf dem Datenträger werden mitgeliefert.',
      progress_preparing_workspace: 'Arbeitsbereichs-Metadaten werden gelesen…',
      progress_fetching_slices: 'Slices werden abgerufen…',
      progress_listing_slices: 'Slices werden aufgelistet…',
      progress_fetched_slices: (done, total) => `${done} / ${total} Slices abgerufen…`,
      progress_validating_heads: 'Modelle werden validiert…',
      progress_validated_heads: (done, total) => `${done} / ${total} Modelle validiert…`,
      progress_packing: 'Archiv wird gepackt…',
      progress_downloading: 'Download wird gestartet…',
      error_default: 'Export fehlgeschlagen',
      error_in_category: (categoryDisplay) => `Export in „${categoryDisplay}“ fehlgeschlagen`,
      error_for_head: (shortId) => `Export für Modell ${shortId} fehlgeschlagen`,
      exporting: 'Wird exportiert…',
      export_aria: 'Ausgewählte Elemente exportieren',
      export_button: 'Exportieren'
    }
  }
} satisfies Messages;
