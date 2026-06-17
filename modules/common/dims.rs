//! Typed `u32`-backed dimension newtypes for the audio + ML pipeline.
//!
//! `u32` is target-independent (unlike `usize`) so wire/config values
//! round-trip identically across architectures; `USIZE` is the cast-free
//! companion for array-dim positions.

macro_rules! u32_newtype {
    ($(#[$attr:meta])* $name:ident, $value:expr) => {
        $(#[$attr])*
        #[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
        #[derive(serde::Serialize, serde::Deserialize)]
        #[serde(transparent)]
        pub struct $name(u32);

        impl $name {
            pub const VALUE: u32 = $value;
            pub const USIZE: usize = $value as usize;

            /// For tests; prod paths want the canonical [`Self::default`].
            #[inline]
            pub const fn new(v: u32) -> Self {
                Self(v)
            }

            #[inline]
            pub const fn get(self) -> u32 {
                self.0
            }

            #[inline]
            pub const fn as_usize(self) -> usize {
                self.0 as usize
            }
        }

        impl Default for $name {
            fn default() -> Self {
                Self($value)
            }
        }

        impl ::core::fmt::Display for $name {
            fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                ::core::fmt::Display::fmt(&self.0, f)
            }
        }
    };
}

u32_newtype! {
    /// Microphone capture sample rate (44.1 kHz).
    SampleRate, 44_100
}

u32_newtype! {
    /// Inference window length in samples (~1.0 s at [`SampleRate`]).
    WaveformLen, 44_032
}

u32_newtype! {
    /// Number of spectrogram time-frames per inference window.
    NFrames, 43
}

u32_newtype! {
    /// Number of spectrogram frequency bins per frame.
    NBins, 232
}

u32_newtype! {
    /// Hop size between adjacent spectrogram frames, in samples.
    HopSamples, 1_024
}

u32_newtype! {
    /// Backbone output / head input feature-vector length.
    BackboneFeatureDim, 2_000
}

/// Upper bound on a loaded head's `n_classes`, bounding the
/// `BACKBONE_FEATURE_DIM x MAX_N_CLASSES x 4` f32 weight allocation a corrupt
/// `head.mpk` could trigger before cold-path validation. Re-exported (never
/// redefined) by the `model` and `inference::head` validators so they can't drift.
pub const MAX_N_CLASSES: usize = 100_000;

pub const BACKBONE_FEATURE_DIM: usize = BackboneFeatureDim::USIZE;

#[cfg(test)]
mod tests {
    use super::*;

    /// Pins canonical values: a change here means every consumer needs review.
    #[test]
    fn canonical_values_are_pinned() {
        assert_eq!(SampleRate::VALUE, 44_100);
        assert_eq!(WaveformLen::VALUE, 44_032);
        assert_eq!(NFrames::VALUE, 43);
        assert_eq!(NBins::VALUE, 232);
        assert_eq!(HopSamples::VALUE, 1_024);
        assert_eq!(BackboneFeatureDim::VALUE, 2_000);
    }

    #[test]
    fn usize_matches_value() {
        assert_eq!(SampleRate::USIZE, SampleRate::VALUE as usize);
        assert_eq!(WaveformLen::USIZE, WaveformLen::VALUE as usize);
        assert_eq!(
            BackboneFeatureDim::USIZE,
            BackboneFeatureDim::VALUE as usize
        );
    }

    #[test]
    fn default_is_canonical() {
        assert_eq!(SampleRate::default().get(), SampleRate::VALUE);
        assert_eq!(WaveformLen::default().get(), WaveformLen::VALUE);
        assert_eq!(
            BackboneFeatureDim::default().as_usize(),
            BackboneFeatureDim::USIZE
        );
    }

    #[test]
    fn new_constructs_arbitrary_value() {
        let half = WaveformLen::new(WaveformLen::VALUE / 2);
        assert_eq!(half.get(), 22_016);
        assert_ne!(half, WaveformLen::default());
    }

    #[test]
    fn usize_works_in_array_dim_position() {
        let _: [f32; WaveformLen::USIZE] = [0.0; 44_032];
        let _: [f32; BackboneFeatureDim::USIZE] = [0.0; 2_000];
    }

    #[test]
    fn display_writes_inner_number() {
        assert_eq!(format!("{}", WaveformLen::default()), "44032");
        assert_eq!(format!("{}", NFrames::new(7)), "7");
    }
}
