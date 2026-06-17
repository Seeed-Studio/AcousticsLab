<script lang="ts">
  import type { Snippet } from 'svelte';
  import Spinner from '$lib/components/Spinner.svelte';

  // warning = primary shape, amber palette flagging a large-consequence action.
  export type ButtonVariant = 'primary' | 'secondary' | 'warning' | 'destructive';
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
    primary:
      'bg-accent text-fg-on-accent border-accent hover:bg-accent-hover hover:border-accent-hover disabled:bg-disabled disabled:text-disabled-fg disabled:border-disabled',
    secondary:
      'bg-surface text-fg border-line hover:border-line-strong hover:bg-surface-2 disabled:bg-page disabled:text-disabled-fg disabled:border-line',
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
