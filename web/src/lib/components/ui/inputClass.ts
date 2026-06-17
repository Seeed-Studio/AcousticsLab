// Focus ring/halo lives globally on `input:focus-visible` in app.css, not as
// `focus-visible:` utilities here: Tailwind's `@layer utilities` loses to the
// unlayered accent-focus rule by cascade-layer precedence regardless of specificity.
// `hasError` swaps only the unfocused border; the focused danger border+halo come
// from the app.css `[aria-invalid='true']:focus-visible` rule, so callers must pair
// `inputClass(true)` with `aria-invalid={true}`. Disabled+hover repeats the static
// colour so the border never flickers mid-submit.
export function inputClass(hasError = false): string {
  const palette = hasError
    ? 'border-danger-line hover:border-danger-line disabled:hover:border-danger-line'
    : 'border-line hover:border-line-strong disabled:hover:border-line';
  return `block w-full rounded-md border ${palette} bg-surface px-2.5 py-1.5 text-sm text-fg transition-colors disabled:cursor-wait disabled:bg-page disabled:text-fg-subtle`;
}
