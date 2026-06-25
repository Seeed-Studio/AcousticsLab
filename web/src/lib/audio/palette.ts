// Spectrogram colormap: 8 interpolated stops/mode, brand-green tinted (hue ≈132°) with a subtle
// viridis-like glide (blue-green low energy -> yellow-green peak); chroma -> 0 at the floor so silence
// dissolves into the canvas. floor == --color-canvas (light #fafafa / dark #0a0a0a). Steps are EVEN in L*
// (equal dB -> equal perceived density). LIGHT is lifted/compressed (peak ~L*0.55, light/airy); DARK keeps
// the original wide range (near-black floor -> bright peak). One ramp feeds the live canvas AND the IDB PNG
// thumbnails (cache keyed (sha,theme), never auto-invalidated) — so changing these stops MUST bump
// DB_VERSION and clear the spectrogram stores (see db.ts), else old slices keep their old thumbnails.

// Resolved theme value, never the raw mode (which can be 'auto').
export type SpectrogramTheme = 'light' | 'dark';

const LIGHT_STOPS: readonly (readonly [number, number, number])[] = [
  [250, 250, 250], // floor == --color-canvas (#fafafa), chroma 0
  [217, 236, 216],
  [191, 219, 185],
  [167, 201, 156],
  [146, 183, 126],
  [126, 164, 97],
  [109, 145, 65],
  [93, 127, 26] // peak: medium green ~L*0.55 (light/airy)
];

// Even L* steps like light but ascending over the original WIDE range (near-black floor -> bright
// yellow-green peak); floor == --color-canvas dark (#0a0a0a).
const DARK_STOPS: readonly (readonly [number, number, number])[] = [
  [10, 10, 10], // floor == --color-canvas dark (#0a0a0a), chroma 0
  [23, 44, 25],
  [40, 74, 38],
  [67, 109, 55],
  [96, 143, 71],
  [135, 185, 92],
  [176, 225, 111],
  [218, 255, 129] // peak: bright yellow-green
];

function stopsFor(theme: SpectrogramTheme): readonly (readonly [number, number, number])[] {
  return theme === 'dark' ? DARK_STOPS : LIGHT_STOPS;
}

const LAST = LIGHT_STOPS.length - 1; // both palettes share this length

// Finite t clamped to [0, 1]; NaN is NOT guarded (would index stops -> undefined and throw) since
// every caller feeds finite t.
export function spectrogramColor(t: number, theme: SpectrogramTheme): [number, number, number] {
  const stops = stopsFor(theme);
  const clamped = t < 0 ? 0 : t > 1 ? 1 : t;
  const scaled = clamped * LAST;
  const i = Math.floor(scaled);
  const frac = scaled - i;
  const a = stops[i];
  const b = stops[Math.min(LAST, i + 1)]; // clamp upper stop at t === 1 where i === LAST
  return [
    Math.round(a[0] + (b[0] - a[0]) * frac),
    Math.round(a[1] + (b[1] - a[1]) * frac),
    Math.round(a[2] + (b[2] - a[2]) * frac)
  ];
}

// Interleaved RGB LUT of n evenly-spaced colours as 3·n bytes [r₀,g₀,b₀,r₁,…] for direct
// imageData.data writes; Uint8ClampedArray mirrors ImageData.data so writers skip channel clamps.
export function buildSpectrogramLut(n: number, theme: SpectrogramTheme): Uint8ClampedArray {
  const lut = new Uint8ClampedArray(n * 3);
  const denom = n > 1 ? n - 1 : 1; // keep divisor positive for degenerate n === 1
  for (let i = 0; i < n; i++) {
    const [r, g, b] = spectrogramColor(i / denom, theme);
    const o = i * 3;
    lut[o] = r;
    lut[o + 1] = g;
    lut[o + 2] = b;
  }
  return lut;
}

// Default dB range for FFT_SIZE/2-normalised magnitudes (full-scale ~-6 dB, noise ~-70 dB): [-80, 0]
// spreads the useful band over the ramp's middle ~60%. Both renderers MUST share this or the
// cross-surface colour invariant breaks; override only when both surfaces move together.
export const SPECTROGRAM_DB_FLOOR = -80;
export const SPECTROGRAM_DB_CEILING = 0;

// Normalised FFT magnitude -> clamped palette index in [0, paletteN). The 1e-10 epsilon keeps log10
// finite on silent bins, flooring them to index 0.
export function magnitudeToPaletteIndex(
  magnitude: number,
  paletteN: number,
  floor: number = SPECTROGRAM_DB_FLOOR,
  ceiling: number = SPECTROGRAM_DB_CEILING
): number {
  const db = 20 * Math.log10(magnitude < 1e-10 ? 1e-10 : magnitude);
  let idx = Math.floor(((db - floor) * (paletteN - 1)) / (ceiling - floor));
  if (idx < 0) idx = 0;
  else if (idx >= paletteN) idx = paletteN - 1;
  return idx;
}
