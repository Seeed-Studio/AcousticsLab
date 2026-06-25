<script lang="ts">
  import type { Snippet } from 'svelte';
  import Spinner from '$lib/components/Spinner.svelte';

  // warning = primary shape, amber palette flagging a large-consequence action.
  // tonal = filled teal secondary-action CTA (brand secondary), ranking below the green `primary` but
  // above the neutral outline `secondary`. RESERVED — defined but no caller yet.
  export type ButtonVariant = 'primary' | 'secondary' | 'tonal' | 'warning' | 'destructive';
  export type ButtonSize = 'sm' | 'md';

  interface Props {
    children?: Snippet;
    variant?: ButtonVariant;
    size?: ButtonSize;
    type?: 'button' | 'submit' | 'reset';
    disabled?: boolean;
    // Shows a leading Spinner and folds into isDisabled, blocking clicks.
    loading?: boolean;
    ariaLabel?: string;
    title?: string;
    onclick?: (e: MouseEvent) => void;
    // Escape hatch (e.g. `w-full`), appended after base classes.
    class?: string;
  }
  let {
    children,
    variant = 'primary',
    size = 'md',
    type = 'button',
    disabled = false,
    loading = false,
    ariaLabel,
    title,
    onclick,
    class: extraClass = ''
  }: Props = $props();

  // Saturated variants wash to `bg-disabled` when disabled so the hue can't leak.
  const VARIANT_CLASSES: Readonly<Record<ButtonVariant, string>> = {
    // primary: brand #77ba2a fill, self-bordered (border-primary = fill), white label (--color-primary-fg).
    // White-on-#77ba2a is 2.38:1 in light, below AA by design; dark-teal label in dark (4.72).
    primary:
      'bg-primary text-primary-fg border-primary hover:bg-primary-hover hover:border-primary-hover disabled:bg-disabled disabled:text-disabled-fg disabled:border-disabled',
    secondary:
      'bg-surface text-fg border-line hover:border-line-strong hover:bg-surface-2 disabled:bg-page disabled:text-disabled-fg disabled:border-line',
    tonal:
      'bg-secondary text-secondary-fg border-secondary hover:bg-secondary-hover hover:border-secondary-hover disabled:bg-disabled disabled:text-disabled-fg disabled:border-disabled',
    warning:
      'bg-warning text-warning-fg border-warning hover:bg-warning-hover hover:border-warning-hover disabled:bg-disabled disabled:text-disabled-fg disabled:border-disabled',
    destructive:
      'bg-danger text-danger-fg border-danger hover:bg-danger-hover hover:border-danger-hover disabled:bg-disabled disabled:text-disabled-fg disabled:border-disabled'
  };

  // Heights match the tab + select rhythm.
  const SIZE_CLASSES: Readonly<Record<ButtonSize, string>> = {
    sm: 'text-xs px-2.5 py-1',
    md: 'text-sm px-3.5 py-1.5'
  };

  let isDisabled = $derived(disabled || loading);
</script>

<button
  {type}
  disabled={isDisabled}
  aria-label={ariaLabel}
  {title}
  {onclick}
  class="inline-flex shrink-0 items-center justify-center gap-1.5 rounded-md border font-medium transition duration-200 ease-out active:scale-[0.98] disabled:cursor-not-allowed disabled:active:scale-100 {SIZE_CLASSES[
    size
  ]} {VARIANT_CLASSES[variant]} {loading ? 'cursor-wait' : ''} {extraClass}"
>
  {#if loading}
    <!-- Spinner stroke is `currentColor`; inherits the button's text color. -->
    <Spinner class="h-3 w-3" />
  {/if}
  {@render children?.()}
</button>
