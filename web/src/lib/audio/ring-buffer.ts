// Shared ring-buffer helpers keeping the streams and recorder PCM pipelines' wrap logic in lock-step.

// Returns the next `writeIdx`; callers own and bump `totalWritten` themselves.
export function pushToRing(ring: Float32Array, writeIdx: number, frame: Float32Array): number {
  const r = ring.length;
  if (r === 0) return writeIdx;
  const n = frame.length;
  if (n === 0) return writeIdx;
  const space = r - writeIdx;
  if (n <= space) {
    ring.set(frame, writeIdx);
  } else {
    ring.set(frame.subarray(0, space), writeIdx);
    ring.set(frame.subarray(space), 0);
  }
  return (writeIdx + n) % r;
}

// Bins before the oldest available sample read as 0 (flat baseline); allocation-free so callers reuse `lo`/`hi` across RAFs.
export function envelopeFromRing(
  ring: Float32Array,
  totalWritten: number,
  endSample: number,
  samples: number,
  bins: number,
  lo: Float32Array,
  hi: Float32Array
): void {
  const n = Math.min(bins, lo.length, hi.length);
  if (n <= 0) return;
  lo.fill(0, 0, n);
  hi.fill(0, 0, n);

  if (totalWritten === 0 || samples <= 0) return;
  const r = ring.length;
  if (r === 0) return;
  const oldestAvailable = Math.max(0, totalWritten - r);
  const clampedEnd = Math.max(oldestAvailable, Math.min(Math.floor(endSample), totalWritten));
  const requestedStart = clampedEnd - samples;
  const samplesPerBin = samples / n;

  for (let x = 0; x < n; x++) {
    const rawStart = Math.floor(requestedStart + x * samplesPerBin);
    const rawEnd = Math.floor(requestedStart + (x + 1) * samplesPerBin);
    const start = Math.max(rawStart, oldestAvailable);
    const stop = Math.min(rawEnd, clampedEnd);
    if (stop <= start) continue;

    let min = 0;
    let max = 0;
    for (let p = start; p < stop; p++) {
      const v = ring[p % r];
      if (v < min) min = v;
      if (v > max) max = v;
    }
    lo[x] = min;
    hi[x] = max;
  }
}
