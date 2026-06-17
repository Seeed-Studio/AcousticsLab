// `mm:ss` up to an hour, `h:mm:ss` past, rounded to nearest second.
export function formatDuration(ms: number): string {
  if (!Number.isFinite(ms) || ms < 0) return '0:00';
  const totalSec = Math.round(ms / 1000);
  const hr = Math.floor(totalSec / 3600);
  const min = Math.floor((totalSec % 3600) / 60);
  const sec = totalSec % 60;
  const pad2 = (n: number): string => (n < 10 ? `0${n}` : `${n}`);
  if (hr > 0) return `${hr}:${pad2(min)}:${pad2(sec)}`;
  return `${min}:${pad2(sec)}`;
}

// Unit-suffixed duration to disambiguate from other tokens in inline metadata; sub-second renders "<1s" to stay nonzero.
export function formatDurationHuman(ms: number): string {
  if (!Number.isFinite(ms) || ms <= 0) return '<1s';
  const totalSec = Math.round(ms / 1000);
  if (totalSec === 0) return '<1s';
  if (totalSec < 60) return `${totalSec}s`;
  const hr = Math.floor(totalSec / 3600);
  const min = Math.floor((totalSec % 3600) / 60);
  const sec = totalSec % 60;
  if (hr > 0) return min === 0 ? `${hr}h` : `${hr}h ${min}m`;
  return sec === 0 ? `${min}m` : `${min}m ${sec}s`;
}

// Binary IEC units to match the browser's binary IDB/quota reporting (decimal would mismatch DevTools storage readouts).
const SIZE_UNITS = ['B', 'KiB', 'MiB', 'GiB'] as const;
export function formatBytes(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes < 0) return '0 B';
  if (bytes < 1024) return `${bytes} B`;
  let value = bytes;
  let unit = 0;
  while (value >= 1024 && unit < SIZE_UNITS.length - 1) {
    value /= 1024;
    unit++;
  }
  let fixed = value >= 10 ? value.toFixed(0) : value.toFixed(1);
  // [1023.5,1024) survives the loop but toFixed(0) rounds to "1024"; carry into the next unit to render "1 MiB" not "1024 KiB".
  if (fixed === '1024' && unit < SIZE_UNITS.length - 1) {
    unit++;
    fixed = '1.0';
  }
  const trimmed = fixed.endsWith('.0') ? fixed.slice(0, -2) : fixed;
  return `${trimmed} ${SIZE_UNITS[unit]}`;
}

// MM:SS.t with a tenths decimal so the live recording clock visibly ticks.
export function formatRecordingClock(ms: number): string {
  if (!Number.isFinite(ms) || ms < 0) return '0:00.0';
  const totalTenths = Math.floor(ms / 100);
  const sec = Math.floor(totalTenths / 10);
  const tenths = totalTenths % 10;
  const min = Math.floor(sec / 60);
  const ss = sec % 60;
  return `${min}:${ss < 10 ? '0' : ''}${ss}.${tenths}`;
}
