//! Mic arbitrator integration + unit tests.

#![cfg(test)]

use super::*;
use crate::audio_io::mock::Waveform;
use crate::common::ids::MicId;
use arc_swap::ArcSwap;

/// Wraps `ArcSwap` as `dyn MicSettingsStore` while keeping the inner cell
/// reachable for mid-run mutations.
fn arcswap_store_into_dyn(s: Arc<ArcSwap<MicSettings>>) -> Arc<dyn MicSettingsStore> {
    Arc::new(ArcSwapStore(s))
}

fn alsa_candidate(id: &str, channels: Vec<u16>) -> MicCandidate {
    MicCandidate {
        id: MicId::parse(id).expect("test mic id literal"),
        source: CandidateSource::Alsa {
            hw_spec: "hw:1,0".into(),
            period_size: 1024,
            buffer_size: 4096,
        },
        channels,
    }
}

fn mock_candidate(id: &str, channels: Vec<u16>, n_wave: usize) -> MicCandidate {
    MicCandidate {
        id: MicId::parse(id).expect("test mic id literal"),
        source: CandidateSource::Mock {
            waveforms: vec![Waveform::Silence; n_wave],
            period_size: 512,
            sample_rate: SampleRate::VALUE,
        },
        channels,
    }
}

#[test]
fn validate_accepts_well_formed_alsa() {
    assert!(alsa_candidate("a", vec![0, 1]).validate().is_ok());
}

#[test]
fn validate_accepts_well_formed_mock() {
    assert!(mock_candidate("m", vec![0, 2], 4).validate().is_ok());
}

#[test]
fn validate_rejects_empty_channels() {
    assert_eq!(
        alsa_candidate("a", vec![]).validate(),
        Err(CandidateError::EmptyChannels),
    );
}

#[test]
fn validate_rejects_duplicate_channel() {
    assert_eq!(
        alsa_candidate("a", vec![0, 1, 0]).validate(),
        Err(CandidateError::DuplicateChannel(0)),
    );
}

/// Static cap defends `AlsaSource::open`'s u16 arithmetic against silent
/// overflow on typo configs.
#[test]
fn validate_rejects_channel_above_static_cap() {
    let too_high = MAX_CHANNEL_INDEX + 1;
    let err = alsa_candidate("a", vec![0, too_high]).validate();
    assert_eq!(
        err,
        Err(CandidateError::ChannelIndexTooLarge {
            channel: too_high,
            cap: MAX_CHANNEL_INDEX,
        }),
    );
}

/// Per-index cap is inclusive. Uses Alsa (channel count negotiated at open),
/// not Mock, which would also trip `MAX_CHANNELS` needing that many waveforms.
#[test]
fn validate_accepts_channel_at_static_cap() {
    let max = MAX_CHANNEL_INDEX;
    let c = MicCandidate {
        id: MicId::from_static("at-cap"),
        source: CandidateSource::Alsa {
            hw_spec: "hw:1,0".into(),
            period_size: 1024,
            buffer_size: 4096,
        },
        channels: vec![max],
    };
    assert_eq!(c.validate(), Ok(()));
}

#[test]
fn validate_rejects_empty_hw_spec() {
    let c = MicCandidate {
        id: MicId::from_static("a"),
        source: CandidateSource::Alsa {
            hw_spec: "".into(),
            period_size: 1024,
            buffer_size: 4096,
        },
        channels: vec![0],
    };
    assert_eq!(c.validate(), Err(CandidateError::EmptyHwSpec));
}

#[test]
fn validate_rejects_zero_period() {
    let c = MicCandidate {
        id: MicId::from_static("a"),
        source: CandidateSource::Alsa {
            hw_spec: "hw:1,0".into(),
            period_size: 0,
            buffer_size: 4096,
        },
        channels: vec![0],
    };
    assert_eq!(c.validate(), Err(CandidateError::InvalidPeriodSize(0)));
}

#[test]
fn validate_rejects_buffer_smaller_than_period() {
    let c = MicCandidate {
        id: MicId::from_static("a"),
        source: CandidateSource::Alsa {
            hw_spec: "hw:1,0".into(),
            period_size: 4096,
            buffer_size: 1024,
        },
        channels: vec![0],
    };
    assert_eq!(
        c.validate(),
        Err(CandidateError::InvalidBufferSize {
            period: 4096,
            buffer: 1024,
        }),
    );
}

#[test]
fn validate_rejects_empty_mock_waveforms() {
    let c = MicCandidate {
        id: MicId::from_static("m"),
        source: CandidateSource::Mock {
            waveforms: vec![],
            period_size: 512,
            sample_rate: 44_100,
        },
        channels: vec![0],
    };
    assert_eq!(c.validate(), Err(CandidateError::EmptyMockWaveforms));
}

#[test]
fn validate_rejects_whitelist_exceeding_mock_waveforms() {
    let c = mock_candidate("m", vec![0, 3], 2);
    assert_eq!(
        c.validate(),
        Err(CandidateError::MockWhitelistOutOfRange {
            whitelist_max: 3,
            waveform_count: 2,
        }),
    );
}

#[test]
fn validate_rejects_zero_mock_sample_rate() {
    let c = MicCandidate {
        id: MicId::from_static("m"),
        source: CandidateSource::Mock {
            waveforms: vec![Waveform::Silence; 2],
            period_size: 512,
            sample_rate: 0,
        },
        channels: vec![0],
    };
    assert_eq!(c.validate(), Err(CandidateError::InvalidMockSampleRate(0)));
}

/// Oversized period_size is rejected before it can violate the `Writer::push`
/// safety margin or trigger an oversize scratch allocation.
#[test]
fn validate_rejects_oversized_alsa_period_size() {
    let too_big = MAX_PERIOD_FRAMES + 1;
    let c = MicCandidate {
        id: MicId::from_static("a"),
        source: CandidateSource::Alsa {
            hw_spec: "hw:1,0".into(),
            period_size: too_big,
            buffer_size: too_big * 4,
        },
        channels: vec![0],
    };
    assert_eq!(
        c.validate(),
        Err(CandidateError::PeriodSizeTooLarge {
            period: too_big,
            cap: MAX_PERIOD_FRAMES,
        }),
    );
}

/// Cap is inclusive: `MAX_PERIOD_FRAMES` is allowed; safety-margin invariant
/// still holds (`max_push_len 65536 >> cap`).
#[test]
fn validate_accepts_alsa_period_at_static_cap() {
    let c = MicCandidate {
        id: MicId::from_static("a"),
        source: CandidateSource::Alsa {
            hw_spec: "hw:1,0".into(),
            period_size: MAX_PERIOD_FRAMES,
            buffer_size: MAX_PERIOD_FRAMES * 4,
        },
        channels: vec![0],
    };
    assert_eq!(c.validate(), Ok(()));
}

/// Mock pushes through the same writer as ALSA, so its period is capped too.
#[test]
fn validate_rejects_oversized_mock_period_size() {
    let too_big = MAX_PERIOD_FRAMES + 1;
    let c = MicCandidate {
        id: MicId::from_static("m"),
        source: CandidateSource::Mock {
            waveforms: vec![Waveform::Silence; 2],
            period_size: too_big,
            sample_rate: 44_100,
        },
        channels: vec![0],
    };
    assert_eq!(
        c.validate(),
        Err(CandidateError::PeriodSizeTooLarge {
            period: too_big,
            cap: MAX_PERIOD_FRAMES,
        }),
    );
}

#[test]
fn validate_rejects_oversized_alsa_buffer_absolute_cap() {
    // period = MAX_PERIOD_FRAMES makes 16*period == MAX_BUFFER_FRAMES, so an
    // over-cap buffer trips both bounds; assert only the kind, not which fired.
    let too_big = MAX_BUFFER_FRAMES + 1;
    let c = MicCandidate {
        id: MicId::from_static("a"),
        source: CandidateSource::Alsa {
            hw_spec: "hw:1,0".into(),
            period_size: MAX_PERIOD_FRAMES,
            buffer_size: too_big,
        },
        channels: vec![0],
    };
    assert!(matches!(
        c.validate(),
        Err(CandidateError::BufferSizeTooLarge { .. }),
    ));
}

/// buffer_size > 16x period_size is rejected (catches the samples-vs-ms typo).
#[test]
fn validate_rejects_buffer_exceeding_period_multiplier() {
    let c = MicCandidate {
        id: MicId::from_static("a"),
        source: CandidateSource::Alsa {
            hw_spec: "hw:1,0".into(),
            period_size: 1024,
            buffer_size: 1024 * 17, // > 16x period yet < absolute cap
        },
        channels: vec![0],
    };
    assert!(matches!(
        c.validate(),
        Err(CandidateError::BufferSizeTooLarge { .. }),
    ));
}

#[test]
fn validate_rejects_oversized_channel_whitelist() {
    let c = MicCandidate {
        id: MicId::from_static("a"),
        source: CandidateSource::Alsa {
            hw_spec: "hw:1,0".into(),
            period_size: 1024,
            buffer_size: 4096,
        },
        channels: (0..(MAX_CHANNELS as u16 + 1)).collect(),
    };
    assert_eq!(
        c.validate(),
        Err(CandidateError::TooManyChannels {
            count: MAX_CHANNELS + 1,
            cap: MAX_CHANNELS,
        }),
    );
}

#[test]
fn validate_rejects_oversized_mock_waveforms() {
    let c = MicCandidate {
        id: MicId::from_static("m"),
        source: CandidateSource::Mock {
            waveforms: vec![Waveform::Silence; MAX_CHANNELS + 1],
            period_size: 512,
            sample_rate: 44_100,
        },
        channels: vec![0],
    };
    assert_eq!(
        c.validate(),
        Err(CandidateError::TooManyChannels {
            count: MAX_CHANNELS + 1,
            cap: MAX_CHANNELS,
        }),
    );
}

/// sample_rate below MIN: a too-low rate causes a pathological resampler ratio.
#[test]
fn validate_rejects_sample_rate_below_min() {
    let c = MicCandidate {
        id: MicId::from_static("m"),
        source: CandidateSource::Mock {
            waveforms: vec![Waveform::Silence; 2],
            period_size: 512,
            sample_rate: MIN_SAMPLE_RATE - 1,
        },
        channels: vec![0],
    };
    assert_eq!(
        c.validate(),
        Err(CandidateError::SampleRateOutOfRange {
            rate: MIN_SAMPLE_RATE - 1,
            min: MIN_SAMPLE_RATE,
            max: MAX_SAMPLE_RATE,
        }),
    );
}

/// sample_rate above MAX: a huge-rate typo would allocate multi-MB sinc tables.
#[test]
fn validate_rejects_sample_rate_above_max() {
    let c = MicCandidate {
        id: MicId::from_static("m"),
        source: CandidateSource::Mock {
            waveforms: vec![Waveform::Silence; 2],
            period_size: 512,
            sample_rate: MAX_SAMPLE_RATE + 1,
        },
        channels: vec![0],
    };
    assert_eq!(
        c.validate(),
        Err(CandidateError::SampleRateOutOfRange {
            rate: MAX_SAMPLE_RATE + 1,
            min: MIN_SAMPLE_RATE,
            max: MAX_SAMPLE_RATE,
        }),
    );
}

#[test]
fn validate_accepts_sample_rate_at_bounds() {
    for rate in [MIN_SAMPLE_RATE, MAX_SAMPLE_RATE] {
        let c = MicCandidate {
            id: MicId::from_static("m"),
            source: CandidateSource::Mock {
                waveforms: vec![Waveform::Silence; 2],
                period_size: 512,
                sample_rate: rate,
            },
            channels: vec![0],
        };
        assert_eq!(c.validate(), Ok(()), "rate {rate} should be accepted");
    }
}

#[test]
fn catalogue_validate_propagates_offending_id() {
    let catalogue = MicCatalogue {
        candidates: vec![
            alsa_candidate("good", vec![0]),
            mock_candidate("bad", vec![5], 2),
        ],
    };
    let err = catalogue.validate().expect_err("should reject");
    assert_eq!(err.0, MicId::from_static("bad"));
    assert!(matches!(
        err.1,
        CandidateError::MockWhitelistOutOfRange { .. }
    ));
}

/// Candidate ids must be unique: resolution is first-match-wins, so a
/// duplicate-id candidate is silently dead. Catch at validation.
#[test]
fn catalogue_validate_rejects_duplicate_ids() {
    let catalogue = MicCatalogue {
        candidates: vec![
            alsa_candidate("front", vec![0]),
            alsa_candidate("rear", vec![0]),
            alsa_candidate("front", vec![1]),
        ],
    };
    let err = catalogue.validate().expect_err("must reject");
    assert_eq!(err.0, MicId::from_static("front"));
    assert_eq!(
        err.1,
        CandidateError::DuplicateMicId(MicId::from_static("front"))
    );
}

#[test]
fn catalogue_validate_accepts_distinct_ids() {
    let catalogue = MicCatalogue {
        candidates: vec![
            alsa_candidate("a", vec![0]),
            alsa_candidate("b", vec![0]),
            alsa_candidate("c", vec![0]),
        ],
    };
    assert_eq!(catalogue.validate(), Ok(()));
}

#[test]
fn catalogue_find_returns_matching_candidate() {
    let catalogue = MicCatalogue {
        candidates: vec![alsa_candidate("a", vec![0]), alsa_candidate("b", vec![1])],
    };
    assert_eq!(
        catalogue.find(&MicId::from_static("b")).map(|c| &c.id),
        Some(&MicId::from_static("b")),
    );
    assert!(catalogue.find(&MicId::from_static("missing")).is_none());
}

/// `Fixed { id }` for an unknown mic is rejected: guards against an
/// accepted-but-runtime-impossible policy.
#[test]
fn validate_policy_rejects_unknown_fixed_id() {
    let catalogue = MicCatalogue {
        candidates: vec![alsa_candidate("front", vec![0])],
    };
    let policy = MicPolicy {
        mic: MicSelection::Fixed {
            id: MicId::from_static("rear"),
        },
        channel: ChannelSelection::Auto,
    };
    let err = policy
        .validate_against(&catalogue)
        .expect_err("should reject");
    assert_eq!(
        err,
        PolicyValidationError::UnknownMicId(MicId::from_static("rear"))
    );
}

/// `Fixed { channel }` not in that mic's catalogue whitelist is rejected.
#[test]
fn validate_policy_rejects_channel_outside_catalogue_whitelist() {
    let catalogue = MicCatalogue {
        candidates: vec![alsa_candidate("front", vec![0, 2])],
    };
    let policy = MicPolicy {
        mic: MicSelection::Fixed {
            id: MicId::from_static("front"),
        },
        channel: ChannelSelection::Fixed { channel: 1 }, // not in whitelist [0, 2]
    };
    let err = policy
        .validate_against(&catalogue)
        .expect_err("should reject");
    match err {
        PolicyValidationError::ChannelNotAvailable {
            mic,
            channel,
            available,
        } => {
            assert_eq!(mic, MicId::from_static("front"));
            assert_eq!(channel, 1);
            assert_eq!(available, vec![0, 2]);
        }
        other => panic!("unexpected error variant: {other:?}"),
    }
}

/// `FirstAvailable` is always valid: resolved mic is unknown until runtime, so
/// the per-mic channel check defers to `pick_slot`.
#[test]
fn validate_policy_accepts_first_available_with_fixed_channel() {
    let catalogue = MicCatalogue {
        candidates: vec![alsa_candidate("a", vec![0])],
    };
    let policy = MicPolicy {
        mic: MicSelection::FirstAvailable,
        channel: ChannelSelection::Fixed { channel: 99 }, // unverifiable until runtime
    };
    assert_eq!(policy.validate_against(&catalogue), Ok(()));
}

#[test]
fn validate_policy_accepts_fixed_mic_with_auto_channel() {
    let catalogue = MicCatalogue {
        candidates: vec![alsa_candidate("a", vec![0, 1, 2])],
    };
    let policy = MicPolicy {
        mic: MicSelection::Fixed {
            id: MicId::from_static("a"),
        },
        channel: ChannelSelection::Auto,
    };
    assert_eq!(policy.validate_against(&catalogue), Ok(()));
}

#[test]
fn policy_default_is_first_available_auto() {
    let p = MicPolicy::default();
    assert_eq!(p.mic, MicSelection::FirstAvailable);
    assert_eq!(p.channel, ChannelSelection::Auto);
}

#[test]
fn hysteresis_linear_3db_is_correct() {
    let cfg = MicArbitratorConfig::default();
    let h = cfg.hysteresis_linear();
    assert!((h - 1.4125).abs() < 1e-3, "h={h}"); // 10^(3/20)
}

/// Catalogue and policy persist in separate files; pins round-trip of the
/// catalogue half (launch config).
#[test]
fn toml_round_trip_preserves_catalogue() {
    let original = MicCatalogue {
        candidates: vec![
            MicCandidate {
                id: MicId::from_static("front"),
                source: CandidateSource::Alsa {
                    hw_spec: "hw:1,0".into(),
                    period_size: 1024,
                    buffer_size: 4096,
                },
                channels: vec![0, 1],
            },
            MicCandidate {
                id: MicId::from_static("dev-mock"),
                source: CandidateSource::Mock {
                    waveforms: vec![
                        Waveform::Silence,
                        Waveform::Sine {
                            freq_hz: 1000.0,
                            amplitude: 0.25,
                        },
                    ],
                    period_size: 512,
                    sample_rate: 44_100,
                },
                channels: vec![1],
            },
        ],
    };
    let s = toml::to_string_pretty(&original).expect("serialize");
    let back: MicCatalogue = toml::from_str(&s).expect("deserialize");
    assert_eq!(back, original, "catalogue toml round-trip mismatch:\n{s}");
}

#[test]
fn toml_round_trip_preserves_policy() {
    let original = MicPolicy {
        mic: MicSelection::Fixed {
            id: MicId::from_static("front"),
        },
        channel: ChannelSelection::Fixed { channel: 0 },
    };
    let s = toml::to_string_pretty(&original).expect("serialize");
    let back: MicPolicy = toml::from_str(&s).expect("deserialize");
    assert_eq!(back, original, "policy toml round-trip mismatch:\n{s}");
}

/// `deny_unknown_fields` makes an operator typo fail the parse rather than
/// silently load as default and never report the failed edit.
#[test]
fn mic_policy_rejects_unknown_field() {
    let bad = r#"
        unknown_typo = "ignored"
        [mic]
        kind = "first_available"
        [channel]
        kind = "auto"
    "#;
    let err = toml::from_str::<MicPolicy>(bad).expect_err("unknown field must reject");
    let msg = err.to_string();
    assert!(
        msg.contains("unknown_typo") || msg.contains("unknown field"),
        "expected unknown-field rejection, got: {msg}",
    );
}

/// Same gate on the internally-tagged `MicSelection`: a field foreign to the
/// chosen variant must reject.
#[test]
fn mic_selection_rejects_unknown_field() {
    let bad = r#"
        kind = "fixed"
        id = "front"
        bogus_field = 42
    "#;
    let err = toml::from_str::<MicSelection>(bad).expect_err("unknown field must reject");
    let msg = err.to_string();
    assert!(
        msg.contains("bogus_field") || msg.contains("unknown field"),
        "expected unknown-field rejection, got: {msg}",
    );
}

fn slot_state(rms: f32) -> SlotState {
    SlotState { rms }
}

#[test]
fn pick_slot_empty_returns_none() {
    let now = Instant::now();
    let cfg = MicArbitratorConfig::default();
    assert_eq!(
        pick_slot(
            &ChannelSelection::Auto,
            &[],
            &[],
            None,
            None,
            now,
            cfg.hysteresis_linear(),
            cfg.dwell,
        ),
        None,
    );
}

#[test]
fn pick_slot_auto_no_current_picks_loudest() {
    let now = Instant::now();
    let cfg = MicArbitratorConfig::default();
    let slots = [slot_state(0.05), slot_state(0.30), slot_state(0.10)];
    let whitelist = [0_u16, 1, 2];
    let chosen = pick_slot(
        &ChannelSelection::Auto,
        &slots,
        &whitelist,
        None,
        None,
        now,
        cfg.hysteresis_linear(),
        cfg.dwell,
    );
    assert_eq!(chosen, Some(1));
}

/// Hysteresis blocks a marginal-louder switch.
#[test]
fn pick_slot_auto_hysteresis_blocks_marginal() {
    let now = Instant::now();
    let cfg = MicArbitratorConfig {
        hysteresis_db: 3.0,
        dwell: Duration::ZERO,
        ..MicArbitratorConfig::default()
    };
    let slots = [slot_state(0.10), slot_state(0.13)]; // alt only ~2.3 dB louder
    let whitelist = [0_u16, 1];
    let chosen = pick_slot(
        &ChannelSelection::Auto,
        &slots,
        &whitelist,
        Some(0),
        Some(now),
        now,
        cfg.hysteresis_linear(),
        cfg.dwell,
    );
    assert_eq!(
        chosen,
        Some(0),
        "should NOT switch (insufficient hysteresis)"
    );
}

/// Clear margin (> hysteresis) + dwell satisfied -> switch.
#[test]
fn pick_slot_auto_clear_margin_after_dwell_switches() {
    let now = Instant::now();
    let cfg = MicArbitratorConfig::default();
    let slots = [slot_state(0.10), slot_state(0.30)]; // ~9.5 dB louder
    let whitelist = [0_u16, 1];
    let chosen = pick_slot(
        &ChannelSelection::Auto,
        &slots,
        &whitelist,
        Some(0),
        Some(now - Duration::from_secs(1)),
        now,
        cfg.hysteresis_linear(),
        cfg.dwell,
    );
    assert_eq!(chosen, Some(1), "should switch (clear margin + dwell ok)");
}

/// Dwell not satisfied -> keep the active slot even with a clear margin.
#[test]
fn pick_slot_auto_dwell_blocks_recent_switch() {
    let now = Instant::now();
    let cfg = MicArbitratorConfig {
        dwell: Duration::from_millis(250),
        ..MicArbitratorConfig::default()
    };
    let slots = [slot_state(0.10), slot_state(0.50)]; // 14 dB louder
    let whitelist = [0_u16, 1];
    let chosen = pick_slot(
        &ChannelSelection::Auto,
        &slots,
        &whitelist,
        Some(0),
        Some(now - Duration::from_millis(100)), // within 250 ms dwell
        now,
        cfg.hysteresis_linear(),
        cfg.dwell,
    );
    assert_eq!(chosen, Some(0), "should NOT switch (dwell not satisfied)");
}

/// A whitelisted Fixed channel is honoured even when another slot is louder.
#[test]
fn pick_slot_fixed_honours_named_channel() {
    let now = Instant::now();
    let cfg = MicArbitratorConfig::default();
    // Sparse whitelist: slot 0 -> channel 0, slot 1 -> channel 2.
    let slots = [slot_state(0.99), slot_state(0.05)];
    let whitelist = [0_u16, 2];
    let chosen = pick_slot(
        &ChannelSelection::Fixed { channel: 2 },
        &slots,
        &whitelist,
        Some(0),
        Some(now - Duration::from_secs(10)),
        now,
        cfg.hysteresis_linear(),
        cfg.dwell,
    );
    assert_eq!(chosen, Some(1));
}

/// Fixed channel absent from the whitelist falls back to Auto rather than
/// emitting silence: policy can name a channel the candidate doesn't expose.
#[test]
fn pick_slot_fixed_off_whitelist_falls_back_to_auto() {
    let now = Instant::now();
    let cfg = MicArbitratorConfig::default();
    let slots = [slot_state(0.10), slot_state(0.40)];
    let whitelist = [0_u16, 2]; // no channel 1
    let chosen = pick_slot(
        &ChannelSelection::Fixed { channel: 1 },
        &slots,
        &whitelist,
        None,
        None,
        now,
        cfg.hysteresis_linear(),
        cfg.dwell,
    );
    assert_eq!(chosen, Some(1)); // fell back to Auto -> loudest slot
}

#[test]
fn resolve_first_available_no_active_picks_zero() {
    let cands = vec![alsa_candidate("a", vec![0]), alsa_candidate("b", vec![0])];
    assert_eq!(
        resolve_desired_idx(&MicSelection::FirstAvailable, &cands, None),
        Some(0),
    );
}

#[test]
fn resolve_first_available_active_present_is_sticky() {
    let cands = vec![alsa_candidate("a", vec![0]), alsa_candidate("b", vec![0])];
    let active = MicId::from_static("b"); // sticky: return its index though "a" is first
    assert_eq!(
        resolve_desired_idx(&MicSelection::FirstAvailable, &cands, Some(&active)),
        Some(1),
    );
}

#[test]
fn resolve_first_available_active_gone_falls_back_to_zero() {
    let cands = vec![alsa_candidate("a", vec![0]), alsa_candidate("b", vec![0])];
    let active = MicId::from_static("c"); // not in candidates
    assert_eq!(
        resolve_desired_idx(&MicSelection::FirstAvailable, &cands, Some(&active)),
        Some(0),
    );
}

#[test]
fn resolve_first_available_empty_candidates_is_none() {
    assert_eq!(
        resolve_desired_idx(&MicSelection::FirstAvailable, &[], None),
        None,
    );
}

#[test]
fn resolve_fixed_returns_named_or_none() {
    let cands = vec![alsa_candidate("a", vec![0]), alsa_candidate("b", vec![0])];
    assert_eq!(
        resolve_desired_idx(
            &MicSelection::Fixed {
                id: MicId::from_static("b")
            },
            &cands,
            None,
        ),
        Some(1),
    );
    assert_eq!(
        resolve_desired_idx(
            &MicSelection::Fixed {
                id: MicId::from_static("missing")
            },
            &cands,
            Some(&MicId::from_static("a")),
        ),
        None,
    );
}

#[test]
fn block_rms_silence_is_zero() {
    assert_eq!(block_rms(&[0.0; 256]), 0.0);
}

#[test]
fn block_rms_constant_matches_value() {
    let r = block_rms(&[0.5_f32; 1024]);
    assert!((r - 0.5).abs() < 1e-6, "rms={r}");
}

#[test]
fn block_rms_sine_amplitude_over_sqrt2() {
    // RMS of an amplitude-A sine is A/sqrt(2).
    let n = 4096;
    let samples: Vec<f32> = (0..n)
        .map(|i| (2.0 * std::f32::consts::PI * 1000.0 / 44100.0 * i as f32).sin())
        .collect();
    let r = block_rms(&samples);
    let expected = 1.0 / 2f32.sqrt();
    assert!((r - expected).abs() < 1e-2);
}

#[test]
fn ema_alpha_block_eq_window_is_one_minus_e_inverse() {
    let a = ema_alpha(Duration::from_millis(100), Duration::from_millis(100));
    assert!((a - (1.0 - (-1.0_f32).exp())).abs() < 1e-6, "alpha={a}");
}

#[test]
fn ema_alpha_zero_window_is_one() {
    assert_eq!(ema_alpha(Duration::from_millis(10), Duration::ZERO), 1.0);
}

#[test]
fn alpha_for_frames_uses_cached_value_for_nominal_period() {
    let cached = 0.123_456_f32;
    assert_eq!(
        alpha_for_frames(1024, 1024, 44_100, cached, Duration::from_millis(100)),
        cached,
    );
}

/// A short non-zero read (hot-unplug/partial-read edge) must not use the
/// full-period alpha, else the EMA over-weights the partial RMS sample.
#[test]
fn alpha_for_frames_recomputes_for_short_read() {
    let window = Duration::from_millis(100);
    let cached = ema_alpha(Duration::from_secs_f64(1024.0 / 44_100.0), window);
    let short = alpha_for_frames(256, 1024, 44_100, cached, window);
    let expected = ema_alpha(Duration::from_secs_f64(256.0 / 44_100.0), window);

    assert!((short - expected).abs() < 1e-7, "short alpha = {short}");
    assert!(short < cached, "short read should have less EMA weight");
}

/// `reset_per_channel_fir` clears FIR history + pending output of every `Some`
/// resampler and leaves `None` slots untouched, so post-reset output is
/// bit-identical to a fresh resampler (no pre-reset phantom leaks). This is the
/// arbitrator's xrun-recovery contract; the real path is Linux/`alsa-real` only,
/// so this macOS-runnable test keeps it honest.
#[test]
fn reset_per_channel_fir_clears_fir_state() {
    // Slot 0 `Some` (48 k -> 44.1 k), slot 1 `None` (native): reset zeros 0,
    // ignores 1.
    let mut resamplers: Vec<Option<Streaming>> = vec![Some(Streaming::new(48_000, 44_100)), None];

    let primer: Vec<f32> = (0..2 * 1024)
        .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 48_000.0).sin())
        .collect();
    if let Some(r) = &mut resamplers[0] {
        let _ = r.process(&primer).expect("primer test");
        assert!(r.pending() > 0, "primer should have produced output");
    }

    reset_per_channel_fir(&mut resamplers);

    match &resamplers[0] {
        Some(r) => assert_eq!(r.pending(), 0, "slot 0 still has pending output"),
        None => panic!("slot 0 must remain Some"),
    }
    assert!(resamplers[1].is_none()); // native-rate slot untouched

    // Bit-identity probe: reset resampler must match a fresh one iff reset truly
    // zeroed FIR + accumulators.
    let mut fresh = Streaming::new(48_000, 44_100);
    let probe: Vec<f32> = (0..1024)
        .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 48_000.0).sin())
        .collect();
    let reset_slot = resamplers[0].as_mut().unwrap();
    let _ = reset_slot.process(&probe).expect("test");
    let _ = fresh.process(&probe).expect("test");
    let out_reset = reset_slot.take_output();
    let out_fresh = fresh.take_output();
    assert_eq!(
        out_reset.len(),
        out_fresh.len(),
        "reset resampler output length differs from fresh",
    );
    for (i, (a, b)) in out_reset.iter().zip(out_fresh.iter()).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "reset resampler diverged from fresh at sample {i}: {a} vs {b}",
        );
    }
}

/// Helper must accept `&mut []` without panic: the run loop calls it with
/// zero slots for a rate-equal source.
#[test]
fn reset_per_channel_fir_empty_slice_is_noop() {
    let mut empty: Vec<Option<Streaming>> = Vec::new();
    reset_per_channel_fir(&mut empty);
    assert!(empty.is_empty());
}

/// Non-native source rates are normalized before entering `AudioBuffer`: the
/// writer head tracks 44.1 kHz resampled output, not the 48 kHz frame count.
#[test]
fn process_period_resamples_48khz_source_to_canonical_buffer_rate() {
    const PERIOD_FRAMES: usize = 960;
    const PERIODS: usize = 50;
    // rubato's sinc resampler has deterministic startup latency: after 1 s of
    // 48 kHz input it has emitted ~47_104 input-rate frames, not 48_000 yet.
    const CONVERTIBLE_INPUT_FRAMES_AFTER_LATENCY: f64 = 47_104.0;

    let buf = AudioBuffer::new(131_072);
    let mut writer = buf.take_writer();
    let cfg = arb_test_cfg();
    let candidate = MicCandidate {
        id: MicId::from_static("mock-48k"),
        source: CandidateSource::Mock {
            waveforms: vec![Waveform::Silence],
            period_size: PERIOD_FRAMES,
            sample_rate: 48_000,
        },
        channels: vec![0],
    };
    let src = crate::audio_io::source::MockSource::open(
        &candidate,
        Arc::new(std::sync::atomic::AtomicBool::new(false)),
    )
    .expect("open 48 kHz mock source");
    let mut state = ArbitratorState::new();
    state.boot(crate::audio_io::source::ActiveSource::Mock(src), &cfg);

    assert_eq!(state.cached_rate, 48_000);
    assert_eq!(state.cached_period_frames, PERIOD_FRAMES);
    assert!(
        state.resamplers[0].is_some(),
        "48 kHz source must allocate a source-rate -> 44.1 kHz resampler",
    );

    for _ in 0..PERIODS {
        state.interleaved_scratch[..PERIOD_FRAMES].fill(0.25);
        process_period(
            &mut state,
            &mut writer,
            PERIOD_FRAMES,
            &ChannelSelection::Auto,
            &cfg,
            cfg.hysteresis_linear(),
        );
    }

    let produced = writer.head_pos();
    let expected_ideal = (CONVERTIBLE_INPUT_FRAMES_AFTER_LATENCY * 44_100.0 / 48_000.0) as u64;
    assert!(
        (expected_ideal - 200..=expected_ideal + 200).contains(&produced),
        "48 kHz source should write ~44.1 kHz-resampled output; produced {produced}, expected ~{expected_ideal}",
    );
    assert!(
        produced < 48_000,
        "writer head advanced at input rate instead of canonical buffer rate: {produced}",
    );
}

/// `single_pass_demux_and_rms` must clamp NaN/Inf to zero, else the slot's EMA
/// RMS stays NaN forever and the audio buffer inherits non-finite samples.
#[test]
fn single_pass_demux_clamps_non_finite_to_zero() {
    // 4 frames, 2 channels: ch0 = [0.5, NaN, 0.5, +Inf]; ch1 = [0, 0, -Inf, 0.5].
    let interleaved: Vec<f32> = vec![
        0.5,
        0.0,
        f32::NAN,
        0.0,
        0.5,
        f32::NEG_INFINITY,
        f32::INFINITY,
        0.5,
    ];
    let whitelist = [0_u16, 1];
    const FRAMES: usize = 4;
    const STRIDE: usize = FRAMES;
    // Per-slot scratch laid out `slot*stride + frame`. Production uses
    // STRIDE==cached_period_frames>=frames so short reads leave each stride tail
    // untouched; here STRIDE==FRAMES.
    let mut slot_scratch_flat = vec![f32::NAN; whitelist.len() * STRIDE];
    let mut sum_sq = vec![0.0_f32; whitelist.len()];

    single_pass_demux_and_rms(
        &interleaved,
        2,
        &whitelist,
        &mut slot_scratch_flat,
        &mut sum_sq,
        FRAMES,
        STRIDE,
    );

    for slot_idx in 0..whitelist.len() {
        let offset = slot_idx * STRIDE;
        let dst = &slot_scratch_flat[offset..offset + FRAMES];
        for &s in dst {
            assert!(s.is_finite(), "non-finite sample leaked through: {s}");
        }
    }
    // ch0 [0.5,0,0.5,0] -> sum_sq 0.5; ch1 [0,0,0,0.5] -> sum_sq 0.25.
    assert!((sum_sq[0] - 0.5).abs() < 1e-6);
    assert!((sum_sq[1] - 0.25).abs() < 1e-6);
}

use crate::audio_buffer::AudioBuffer;
use std::sync::Arc;

/// RMS of one 4096-sample `Ready` peek of the live tail.
fn tail_rms(reader: &crate::audio_buffer::Reader) -> f32 {
    let mut sample = vec![0.0f32; 4096];
    let st = reader.peek_into(&mut sample);
    assert_eq!(st, crate::audio_buffer::ReadStatus::Ready);
    block_rms(&sample)
}

fn arb_test_cfg() -> MicArbitratorConfig {
    MicArbitratorConfig {
        hysteresis_db: 3.0,
        dwell: Duration::from_millis(50),
        rms_window: Duration::from_millis(50),
        mic_failover_after: Duration::from_secs(1),
        failover_retry_interval: Duration::from_millis(100),
        // SCHED_FIFO/pin needs root on most CI hosts; None -> SCHED_OTHER.
        sched_pin: None,
        sched_priority: None,
        // Anchor tests override via `..arb_test_cfg()`.
        timing_anchor: None,
    }
}

/// ch0 silent, ch1 sine: Auto picks ch1 and the buffer tail RMS reflects it.
#[test]
fn integration_auto_picks_loud_channel_inside_one_mic() {
    let buf = AudioBuffer::new(65_536); // ~1.49 s (next pow2 above 44_100)
    let writer = buf.take_writer();
    let reader = buf.reader_at(0);

    let candidate = MicCandidate {
        id: MicId::from_static("dual"),
        source: CandidateSource::Mock {
            waveforms: vec![
                Waveform::Silence,
                Waveform::Sine {
                    freq_hz: 1000.0,
                    amplitude: 0.5,
                },
            ],
            period_size: 256,
            sample_rate: SampleRate::VALUE,
        },
        channels: vec![0, 1],
    };
    let settings = Arc::new(ArcSwap::from_pointee(MicSettings {
        catalogue: Arc::new(MicCatalogue {
            candidates: vec![candidate],
        }),
        policy: MicPolicy {
            mic: MicSelection::FirstAvailable,
            channel: ChannelSelection::Auto,
        },
    }));
    let arb = MicArbitrator::start(writer, arcswap_store_into_dyn(settings), arb_test_cfg());

    std::thread::sleep(Duration::from_millis(300));

    let mut sample = vec![0.0f32; 4096];
    let st = reader.peek_into(&mut sample);
    assert_eq!(
        st,
        crate::audio_buffer::ReadStatus::Ready,
        "buffer should be ready"
    );
    let r = block_rms(&sample);
    // Amplitude-0.5 sine -> RMS ~0.354; silence would be 0.
    assert!(
        r > 0.2,
        "tail RMS = {r}; expected > 0.2 (loud-channel should win)",
    );

    arb.stop();
}

/// With a `SharedTimingAnchor` the arbitrator publishes a fresh anchor after
/// each `Writer::push`. Interval snapshots must show head_pos/captured_at
/// monotone non-decreasing (equality when no push lands between snapshots),
/// sample_rate_hz == SampleRate::VALUE, and a final head_pos>0 && captured_at>0
/// proving a real push replaced the boot placeholder.
#[test]
fn integration_producer_publishes_monotonic_timing_anchor() {
    use crate::common::time::{BufferTimingAnchor, shared_timing_anchor};
    let buf = AudioBuffer::new(65_536);
    let writer = buf.take_writer();

    let candidate = MicCandidate {
        id: MicId::from_static("anchor-test"),
        source: CandidateSource::Mock {
            waveforms: vec![Waveform::Sine {
                freq_hz: 440.0,
                amplitude: 0.3,
            }],
            period_size: 256,
            sample_rate: SampleRate::VALUE,
        },
        channels: vec![0],
    };
    let settings = Arc::new(ArcSwap::from_pointee(MicSettings {
        catalogue: Arc::new(MicCatalogue {
            candidates: vec![candidate],
        }),
        policy: MicPolicy {
            mic: MicSelection::FirstAvailable,
            channel: ChannelSelection::Auto,
        },
    }));

    let anchor_cell = shared_timing_anchor();
    assert_eq!(
        **anchor_cell.load(),
        BufferTimingAnchor::boot_placeholder(),
        "fresh cell must initialise to the boot placeholder",
    );

    let cfg = MicArbitratorConfig {
        timing_anchor: Some(anchor_cell.clone()),
        ..arb_test_cfg()
    };
    let arb = MicArbitrator::start(writer, arcswap_store_into_dyn(settings), cfg);

    // Sample at intervals to observe progression; 300 ms covers >50 pushes at
    // 256 frames / 44.1 kHz (~5.8 ms/period).
    let mut snapshots: Vec<BufferTimingAnchor> = Vec::with_capacity(8);
    let snapshot_intervals = 8;
    let interval = Duration::from_millis(300 / snapshot_intervals as u64);
    for _ in 0..snapshot_intervals {
        std::thread::sleep(interval);
        snapshots.push(**anchor_cell.load());
    }

    arb.stop();

    for w in snapshots.windows(2) {
        assert!(
            w[1].head_pos >= w[0].head_pos,
            "head_pos went backwards: {} -> {}",
            w[0].head_pos,
            w[1].head_pos,
        );
    }
    for w in snapshots.windows(2) {
        assert!(
            w[1].captured_at.as_micros() >= w[0].captured_at.as_micros(),
            "captured_at went backwards: {} -> {}",
            w[0].captured_at.as_micros(),
            w[1].captured_at.as_micros(),
        );
    }
    for s in &snapshots {
        assert_eq!(
            s.sample_rate_hz,
            SampleRate::VALUE,
            "anchor must record the canonical buffer rate",
        );
    }
    // head_pos > 0 proves the production push path ran (cell no longer the boot
    // placeholder).
    let last = *snapshots.last().expect("at least one snapshot");
    assert!(
        last.head_pos > 0,
        "no push observed; producer never published an anchor; last={last:?}",
    );

    // captured_at past placeholder zero proves `CaptureTime::now()` ran.
    assert!(
        last.captured_at.as_micros() > 0,
        "captured_at still at boot placeholder after stop; last={last:?}",
    );
}

/// Both channels active: ch0 loud 500 Hz (RMS ~0.283), ch1 quiet 2 kHz (RMS
/// ~0.071). Distinct freqs expose a frequency-sensitive demux/RMS bug; ~12 dB
/// margin clears the 3 dB hysteresis so ch0 wins.
#[test]
fn integration_auto_picks_louder_of_two_active_channels() {
    let buf = AudioBuffer::new(65_536);
    let writer = buf.take_writer();
    let reader = buf.reader_at(0);

    let candidate = MicCandidate {
        id: MicId::from_static("dual-active"),
        source: CandidateSource::Mock {
            waveforms: vec![
                Waveform::Sine {
                    freq_hz: 500.0,
                    amplitude: 0.4,
                },
                Waveform::Sine {
                    freq_hz: 2000.0,
                    amplitude: 0.1,
                },
            ],
            period_size: 256,
            sample_rate: SampleRate::VALUE,
        },
        channels: vec![0, 1],
    };
    let settings = Arc::new(ArcSwap::from_pointee(MicSettings {
        catalogue: Arc::new(MicCatalogue {
            candidates: vec![candidate],
        }),
        policy: MicPolicy::default(),
    }));
    let arb = MicArbitrator::start(writer, arcswap_store_into_dyn(settings), arb_test_cfg());

    std::thread::sleep(Duration::from_millis(300));

    let r = tail_rms(&reader);
    // ch0 ~0.283, ch1 ~0.071; picking ch1 would land ~0.07, well below 0.2.
    assert!(
        r > 0.2,
        "tail RMS {r} suggests arbitrator picked the quieter channel",
    );

    arb.stop();
}

/// Loudness alternates every 0.5 s in opposite phase (`PingPongSine`); the
/// arbitrator must follow each flip (ch0, ch1, ch0) through the
/// EMA x hysteresis x dwell interaction or it latches and the tail goes quiet on
/// the low half. New-loud EMA exceeds old by >1.41x within ~46 ms, so switches
/// land ~546 ms and ~1046 ms. Sampling at 800 ms catches "never switches"; at
/// 1300 ms catches "switches once then latches".
#[test]
fn integration_auto_switches_when_loudness_alternates_between_channels() {
    let buf = AudioBuffer::new(131_072);
    let writer = buf.take_writer();
    let mut reader = buf.reader_at(0);

    let half = 22_050; // 0.5 s @ 44.1 k
    let candidate = MicCandidate {
        id: MicId::from_static("ping-pong"),
        source: CandidateSource::Mock {
            waveforms: vec![
                Waveform::PingPongSine {
                    freq_hz: 500.0,
                    high_amp: 0.5,
                    low_amp: 0.02,
                    half_period_samples: half,
                    inverted: false,
                },
                Waveform::PingPongSine {
                    freq_hz: 2000.0,
                    high_amp: 0.5,
                    low_amp: 0.02,
                    half_period_samples: half,
                    inverted: true,
                },
            ],
            period_size: 256,
            sample_rate: SampleRate::VALUE,
        },
        channels: vec![0, 1],
    };
    let settings = Arc::new(ArcSwap::from_pointee(MicSettings {
        catalogue: Arc::new(MicCatalogue {
            candidates: vec![candidate],
        }),
        policy: MicPolicy::default(),
    }));
    let arb = MicArbitrator::start(writer, arcswap_store_into_dyn(settings), arb_test_cfg());

    // ~800 ms: flip at 500 ms made ch1 loud (switch ~546 ms); the 2048-sample
    // (~46 ms) tail lies entirely after the switch.
    std::thread::sleep(Duration::from_millis(800));
    reader.seek_latest(2048);
    let mut sample = vec![0.0f32; 2048];
    let st = reader.peek_into(&mut sample);
    assert_eq!(st, crate::audio_buffer::ReadStatus::Ready);
    let r1 = block_rms(&sample);
    assert!(
        r1 > 0.2,
        "first flip: expected arbitrator to follow ch0->ch1 switch; \
             tail RMS {r1} suggests it latched on ch0 (now in its low half)",
    );

    // ~1300 ms: flip at 1000 ms made ch0 loud again; switch expected ~1046 ms.
    std::thread::sleep(Duration::from_millis(500));
    reader.seek_latest(2048);
    let st = reader.peek_into(&mut sample);
    assert_eq!(st, crate::audio_buffer::ReadStatus::Ready);
    let r2 = block_rms(&sample);
    assert!(
        r2 > 0.2,
        "second flip: expected arbitrator to follow ch1->ch0 switch-back; \
             tail RMS {r2} suggests it latched on ch1 (now in its low half)",
    );

    arb.stop();
}

/// Hot-swapping `MicSettings` to a different candidate id (Fixed policy forcing
/// it) must tear down the old source, open the new one, and converge on its loud
/// channel. Exercises the mic-swap + post-swap initial channel pick, not the
/// in-mic dwell/hysteresis the `pick_slot_*` tests cover.
#[test]
fn integration_post_mic_swap_picks_loud_channel_of_new_mic() {
    let buf = AudioBuffer::new(131_072);
    let writer = buf.take_writer();
    let mut reader = buf.reader_at(0);

    let make_candidate = |amp_ch0: f32, amp_ch1: f32| MicCandidate {
        id: MicId::from_static("dual"),
        source: CandidateSource::Mock {
            waveforms: vec![
                Waveform::Sine {
                    freq_hz: 500.0,
                    amplitude: amp_ch0,
                },
                Waveform::Sine {
                    freq_hz: 2000.0,
                    amplitude: amp_ch1,
                },
            ],
            period_size: 256,
            sample_rate: SampleRate::VALUE,
        },
        channels: vec![0, 1],
    };

    let settings = Arc::new(ArcSwap::from_pointee(MicSettings {
        catalogue: Arc::new(MicCatalogue {
            candidates: vec![make_candidate(0.5, 0.05)], // phase 1: ch0 loud
        }),
        policy: MicPolicy::default(),
    }));
    let arb = MicArbitrator::start(
        writer,
        arcswap_store_into_dyn(settings.clone()),
        arb_test_cfg(),
    );

    std::thread::sleep(Duration::from_millis(200)); // phase 1: open + converge

    // Phase 2: pin policy to a new id (`dual-flipped`, ch1 loud), forcing a
    // mic-switch (tear down + re-open), not a channel switch.
    settings.store(Arc::new(MicSettings {
        catalogue: Arc::new(MicCatalogue {
            candidates: vec![
                make_candidate(0.5, 0.05),
                MicCandidate {
                    id: MicId::from_static("dual-flipped"),
                    source: CandidateSource::Mock {
                        waveforms: vec![
                            Waveform::Sine {
                                freq_hz: 500.0,
                                amplitude: 0.05,
                            },
                            Waveform::Sine {
                                freq_hz: 2000.0,
                                amplitude: 0.5,
                            },
                        ],
                        period_size: 256,
                        sample_rate: SampleRate::VALUE,
                    },
                    channels: vec![0, 1],
                },
            ],
        }),
        policy: MicPolicy {
            mic: MicSelection::Fixed {
                id: MicId::from_static("dual-flipped"),
            },
            channel: ChannelSelection::Auto,
        },
    }));

    std::thread::sleep(Duration::from_millis(400)); // re-open + settle on ch1 + dwell

    // Tail dominated by the loud sine (both phases used amplitude 0.5).
    reader.seek_latest(2048);
    let mut sample = vec![0.0f32; 2048];
    let st = reader.peek_into(&mut sample);
    assert_eq!(st, crate::audio_buffer::ReadStatus::Ready);
    let r = block_rms(&sample);
    assert!(r > 0.2, "post-switch RMS = {r}; expected > 0.2");

    arb.stop();
}

/// Demux honours the whitelist, not all device channels: whitelisting only the
/// silent channels (excluding the loud one) keeps the tail quiet.
#[test]
fn integration_whitelist_filters_to_subset() {
    let buf = AudioBuffer::new(65_536);
    let writer = buf.take_writer();
    let reader = buf.reader_at(0);

    let candidate = MicCandidate {
        id: MicId::from_static("triple"),
        // Waveforms: silent, loud, silent.
        source: CandidateSource::Mock {
            waveforms: vec![
                Waveform::Silence,
                Waveform::Sine {
                    freq_hz: 1000.0,
                    amplitude: 0.5,
                },
                Waveform::Silence,
            ],
            period_size: 256,
            sample_rate: SampleRate::VALUE,
        },
        channels: vec![0, 2], // excludes the loud ch1
    };
    let settings = Arc::new(ArcSwap::from_pointee(MicSettings {
        catalogue: Arc::new(MicCatalogue {
            candidates: vec![candidate],
        }),
        policy: MicPolicy::default(),
    }));
    let arb = MicArbitrator::start(writer, arcswap_store_into_dyn(settings), arb_test_cfg());

    std::thread::sleep(Duration::from_millis(300));

    let r = tail_rms(&reader);
    assert!(
        r < 0.05,
        "whitelist excluded the loud channel; tail RMS = {r}; expected < 0.05",
    );

    arb.stop();
}

/// `Fixed` opens the named candidate even when an earlier-listed one is also
/// openable (no FirstAvailable "earlier" preference).
#[test]
fn integration_fixed_selection_opens_named_mic() {
    let buf = AudioBuffer::new(65_536);
    let writer = buf.take_writer();
    let reader = buf.reader_at(0);

    let cand_quiet = MicCandidate {
        id: MicId::from_static("quiet"),
        source: CandidateSource::Mock {
            waveforms: vec![Waveform::Silence],
            period_size: 256,
            sample_rate: SampleRate::VALUE,
        },
        channels: vec![0],
    };
    let cand_loud = MicCandidate {
        id: MicId::from_static("loud"),
        source: CandidateSource::Mock {
            waveforms: vec![Waveform::Sine {
                freq_hz: 1000.0,
                amplitude: 0.5,
            }],
            period_size: 256,
            sample_rate: SampleRate::VALUE,
        },
        channels: vec![0],
    };
    let settings = Arc::new(ArcSwap::from_pointee(MicSettings {
        catalogue: Arc::new(MicCatalogue {
            candidates: vec![cand_quiet, cand_loud],
        }),
        policy: MicPolicy {
            mic: MicSelection::Fixed {
                id: MicId::from_static("loud"),
            },
            channel: ChannelSelection::Auto,
        },
    }));
    let arb = MicArbitrator::start(writer, arcswap_store_into_dyn(settings), arb_test_cfg());

    std::thread::sleep(Duration::from_millis(300));

    let r = tail_rms(&reader);
    assert!(r > 0.2, "Fixed selection didn't pick named loud mic; r={r}");

    arb.stop();
}

/// Stop is honoured promptly even mid mock sleep: teardown within a few periods.
#[test]
fn integration_stop_is_prompt() {
    let buf = AudioBuffer::new(65_536);
    let writer = buf.take_writer();
    let cand = MicCandidate {
        id: MicId::from_static("paced"),
        source: CandidateSource::Mock {
            waveforms: vec![Waveform::Silence],
            period_size: 2048, // ~50 ms so a stop during mock pacing is meaningful
            sample_rate: SampleRate::VALUE,
        },
        channels: vec![0],
    };
    let settings = Arc::new(ArcSwap::from_pointee(MicSettings {
        catalogue: Arc::new(MicCatalogue {
            candidates: vec![cand],
        }),
        policy: MicPolicy::default(),
    }));
    let arb = MicArbitrator::start(writer, arcswap_store_into_dyn(settings), arb_test_cfg());

    std::thread::sleep(Duration::from_millis(120)); // enter the read+pace hot loop

    let t = Instant::now();
    arb.stop();
    let elapsed = t.elapsed();
    // 2 ms slice + slop; allow one period (50 ms) on a loaded CI box.
    assert!(
        elapsed < Duration::from_millis(50),
        "stop took too long: {elapsed:?}",
    );
}

/// `signal_stop` returns without joining; a later `stop()` joins promptly
/// because the loop already saw the cancel. The daemon relies on this to signal
/// the producer first (consumers drain into a quiet pipeline), then join last.
#[test]
fn integration_signal_stop_returns_without_joining() {
    let buf = AudioBuffer::new(65_536);
    let writer = buf.take_writer();
    let cand = MicCandidate {
        id: MicId::from_static("paced"),
        source: CandidateSource::Mock {
            waveforms: vec![Waveform::Silence],
            period_size: 2048,
            sample_rate: SampleRate::VALUE,
        },
        channels: vec![0],
    };
    let settings = Arc::new(ArcSwap::from_pointee(MicSettings {
        catalogue: Arc::new(MicCatalogue {
            candidates: vec![cand],
        }),
        policy: MicPolicy::default(),
    }));
    let arb = MicArbitrator::start(writer, arcswap_store_into_dyn(settings), arb_test_cfg());

    std::thread::sleep(Duration::from_millis(120)); // hit a thread actively producing

    let t_signal = Instant::now();
    arb.signal_stop();
    let signal_elapsed = t_signal.elapsed();
    // Signal is a flag store + unpark (sub-us); a join slipping in would blow
    // the 5 ms ceiling since `stop()` typically takes 5-50 ms.
    assert!(
        signal_elapsed < Duration::from_millis(5),
        "signal_stop blocked for {signal_elapsed:?}; must not join",
    );

    arb.signal_stop(); // idempotent no-op

    // Loop has observed stop since the first signal, so this join is a no-wait.
    let t_join = Instant::now();
    arb.stop();
    let join_elapsed = t_join.elapsed();
    assert!(
        join_elapsed < Duration::from_millis(50),
        "post-signal stop took too long: {join_elapsed:?}",
    );
}

/// Stop is prompt even in the no-source `park_timeout` branch (empty catalogue
/// parks for `failover_retry_interval`); a `thread::sleep` here would hold
/// teardown back by up to one interval.
#[test]
fn integration_stop_is_prompt_with_no_source_open() {
    let buf = AudioBuffer::new(65_536);
    let writer = buf.take_writer();
    // Empty catalogue -> no source opened -> loop sits in the no-source branch.
    let settings = Arc::new(ArcSwap::from_pointee(MicSettings {
        catalogue: Arc::new(MicCatalogue::default()),
        policy: MicPolicy::default(),
    }));
    // Long interval so a sleep-instead-of-park regression clearly overruns.
    let cfg = MicArbitratorConfig {
        failover_retry_interval: Duration::from_secs(1),
        ..arb_test_cfg()
    };
    let arb = MicArbitrator::start(writer, arcswap_store_into_dyn(settings), cfg);

    std::thread::sleep(Duration::from_millis(50)); // enter the park_timeout branch

    let t = Instant::now();
    arb.stop();
    let elapsed = t.elapsed();
    // Without the unpark this would be ~1 s (failover_retry_interval).
    assert!(
        elapsed < Duration::from_millis(50),
        "stop took too long with no source open: {elapsed:?} \
             (expected unpark to wake park_timeout immediately)",
    );
}

/// FirstAvailable failover: first candidate is broken (ALSA absent on macOS),
/// so the arbitrator falls through to the working mock.
#[test]
fn integration_first_available_falls_through_broken_candidate() {
    let buf = AudioBuffer::new(65_536);
    let writer = buf.take_writer();
    let reader = buf.reader_at(0);

    let broken = MicCandidate {
        id: MicId::from_static("broken-alsa"),
        source: CandidateSource::Alsa {
            hw_spec: "hw:99,99".into(),
            period_size: 1024,
            buffer_size: 4096,
        },
        channels: vec![0],
    };
    let working = MicCandidate {
        id: MicId::from_static("working-mock"),
        source: CandidateSource::Mock {
            waveforms: vec![Waveform::Sine {
                freq_hz: 1000.0,
                amplitude: 0.5,
            }],
            period_size: 256,
            sample_rate: SampleRate::VALUE,
        },
        channels: vec![0],
    };
    let settings = Arc::new(ArcSwap::from_pointee(MicSettings {
        catalogue: Arc::new(MicCatalogue {
            candidates: vec![broken, working],
        }),
        policy: MicPolicy::default(),
    }));
    let arb = MicArbitrator::start(writer, arcswap_store_into_dyn(settings), arb_test_cfg());

    std::thread::sleep(Duration::from_millis(300));

    let r = tail_rms(&reader);
    assert!(r > 0.2, "expected fallback to working-mock; tail RMS = {r}",);

    arb.stop();
}

/// `validate` accepts the defaults and rejects each independently-bad field.
#[test]
fn arbitrator_config_validate_rejects_bad_fields() {
    MicArbitratorConfig::default()
        .validate()
        .expect("default must validate");

    let bad = MicArbitratorConfig {
        hysteresis_db: -1.0,
        ..MicArbitratorConfig::default()
    };
    let err = bad.validate().expect_err("negative hysteresis must reject");
    assert!(err.contains("hysteresis_db"), "{err}");

    let bad = MicArbitratorConfig {
        hysteresis_db: f32::NAN,
        ..MicArbitratorConfig::default()
    };
    bad.validate().expect_err("NaN hysteresis must reject");

    let bad = MicArbitratorConfig {
        rms_window: Duration::ZERO,
        ..MicArbitratorConfig::default()
    };
    let err = bad.validate().expect_err("zero rms_window must reject");
    assert!(err.contains("rms_window"), "{err}");

    let bad = MicArbitratorConfig {
        mic_failover_after: Duration::ZERO,
        ..MicArbitratorConfig::default()
    };
    let err = bad.validate().expect_err("zero failover must reject");
    assert!(err.contains("mic_failover_after"), "{err}");

    let bad = MicArbitratorConfig {
        failover_retry_interval: Duration::ZERO,
        ..MicArbitratorConfig::default()
    };
    let err = bad.validate().expect_err("zero retry_interval must reject");
    assert!(err.contains("failover_retry_interval"), "{err}");
}

/// `start` self-validates and panics on a bad config, so a call site can't
/// bypass the gate by forgetting an upstream `validate()`.
#[test]
#[should_panic(expected = "invalid MicArbitratorConfig")]
fn start_self_validates_invalid_config_panics() {
    let buf = AudioBuffer::new(65_536);
    let writer = buf.take_writer();
    let candidate = MicCandidate {
        id: MicId::from_static("dual"),
        source: CandidateSource::Mock {
            waveforms: vec![Waveform::Silence],
            period_size: 256,
            sample_rate: SampleRate::VALUE,
        },
        channels: vec![0],
    };
    let settings = Arc::new(ArcSwap::from_pointee(MicSettings {
        catalogue: Arc::new(MicCatalogue {
            candidates: vec![candidate],
        }),
        policy: MicPolicy {
            mic: MicSelection::FirstAvailable,
            channel: ChannelSelection::Auto,
        },
    }));
    let bad = MicArbitratorConfig {
        hysteresis_db: -1.0,
        ..arb_test_cfg()
    };
    let _ = MicArbitrator::start(writer, arcswap_store_into_dyn(settings), bad);
}

#[test]
fn toml_defaults_fill_missing_alsa_fields() {
    let toml_text = r#"
            candidates = [
              { id = "a", source = { kind = "alsa", hw_spec = "hw:1,0" }, channels = [0] },
            ]
        "#;
    let s: MicCatalogue = toml::from_str(toml_text).expect("deserialize");
    match &s.candidates[0].source {
        CandidateSource::Alsa {
            period_size,
            buffer_size,
            ..
        } => {
            assert_eq!(*period_size, 1024_usize);
            assert_eq!(*buffer_size, 4096_usize);
        }
        _ => panic!("expected Alsa source"),
    }
}
