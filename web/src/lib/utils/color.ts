export function hexToRgba(hex: string, alpha: number): string {
  let h = hex.replace('#', '').trim();
  // Expand shorthand first: the prod CSS minifier rewrites `#ffffff` to `#fff`,
  // so getComputedStyle yields 3/4-digit hex that a 6-hex-only gate would reject.
  if (h.length === 3 || h.length === 4) {
    h = h
      .split('')
      .map((c) => c + c)
      .join('');
  }
  // Non-hex input (rgb/oklch/named/unresolved var) would parseInt to NaN, so fall
  // back to grey; 8-digit hex is accepted but its trailing alpha pair is ignored.
  if (!/^[0-9a-fA-F]{6}([0-9a-fA-F]{2})?$/.test(h)) {
    return `rgba(128, 128, 128, ${alpha})`;
  }
  const r = parseInt(h.slice(0, 2), 16);
  const g = parseInt(h.slice(2, 4), 16);
  const b = parseInt(h.slice(4, 6), 16);
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}
