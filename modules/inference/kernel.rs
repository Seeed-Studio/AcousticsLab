//! Zero-allocation CPU kernels for the inference hot loop: caller-owned
//! buffers, no steady-state heap activity. [`top_k_indices_into`] allocates
//! only when `out`'s capacity is short; the engine pre-reserves it.

/// `logits[c] = bias[c] + sum_i features[i] * weight[i, c]`, row-major
/// weight of shape `[in_features, n_classes]`. Zero alloc; inner FMA loop
/// auto-vectorizes to NEON, bypassing Burn's heap-allocating path.
pub fn head_forward(features: &[f32], weight: &[f32], bias: &[f32], logits: &mut [f32]) {
    let n = logits.len();
    // Guards `chunks_exact(0)` panic for direct callers (HeadInner::validate gates n_classes >= 1).
    assert!(n > 0, "head_forward called with empty logits");
    // Runtime (not debug_) asserts are codegen-load-bearing: the length proofs let LLVM
    // memcpy copy_from_slice and elide per-cell bounds checks so the FMA loop vectorizes to NEON fmla.
    assert_eq!(bias.len(), n, "bias shape mismatches logits");
    assert_eq!(
        weight.len(),
        features.len() * n,
        "weight shape mismatches features x logits",
    );
    logits.copy_from_slice(bias);
    for (f, row) in features.iter().zip(weight.chunks_exact(n)) {
        for (out, &w) in logits.iter_mut().zip(row.iter()) {
            *out += f * w;
        }
    }
}

/// Numerically-stable softmax: `probs[i] = exp(logits[i] - max) / sum`, subtracting max before
/// `exp` to avoid overflow. Length mismatch is a runtime assert (else stale tail scaled as valid probs).
pub fn softmax_into(logits: &[f32], probs: &mut [f32]) {
    assert_eq!(
        logits.len(),
        probs.len(),
        "softmax shape mismatch: logits.len()={}, probs.len()={}",
        logits.len(),
        probs.len(),
    );
    // A NaN logit survives the max-fold (`>` drops NaN) then poisons exp via (NaN - m); short-circuit to uniform.
    let mut m = f32::NEG_INFINITY;
    for &v in logits.iter() {
        if v.is_nan() {
            let n = probs.len();
            let p = if n == 0 { 0.0 } else { 1.0 / n as f32 };
            probs.fill(p);
            return;
        }
        if v > m {
            m = v;
        }
    }
    // +inf: normal path's (+inf - +inf) = NaN poisons all; softmax limit splits mass equally over +inf entries, finite -> 0.
    if m == f32::INFINITY {
        let inf_count = logits.iter().filter(|&&l| l == f32::INFINITY).count();
        let p = 1.0 / inf_count as f32;
        for (out, &l) in probs.iter_mut().zip(logits.iter()) {
            *out = if l == f32::INFINITY { p } else { 0.0 };
        }
        return;
    }
    // All -inf or n == 0: explicit uniform avoids -inf - -inf = NaN.
    if m == f32::NEG_INFINITY {
        let n = probs.len();
        let p = if n == 0 { 0.0 } else { 1.0 / n as f32 };
        probs.fill(p);
        return;
    }
    let mut sum = 0.0f32;
    for (p, &l) in probs.iter_mut().zip(logits.iter()) {
        let e = (l - m).exp();
        *p = e;
        sum += e;
    }
    // sum >= 1 here: the l == m entry contributes exp(0) = 1, so the divide is safe.
    let inv = 1.0 / sum;
    for p in probs.iter_mut() {
        *p *= inv;
    }
}

/// Fill `out` with the top-`k` indices of `xs`, sorted descending; `k = 0` yields
/// empty, `k > xs.len()` is capped, NaN sorts as Equal. O(n + k log k) via
/// partition-then-sort. Zero alloc if `out.capacity() >= xs.len()`.
pub fn top_k_indices_into(xs: &[f32], k: usize, out: &mut Vec<usize>) {
    out.clear();
    let n = xs.len();
    let k_capped = k.min(n);
    if k_capped == 0 {
        return;
    }
    out.extend(0..n);
    let cmp = |&a: &usize, &b: &usize| {
        xs[b]
            .partial_cmp(&xs[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    };
    if k_capped < n {
        out.select_nth_unstable_by(k_capped - 1, cmp);
        out.truncate(k_capped);
    }
    out.sort_unstable_by(cmp);
}

/// Transpose frame-major `[N_FRAMES][N_BINS]` into bin-major `[bin][frame]` for librknnrt's
/// NHWC ingestion: rv1126b/RKNN-toolkit2 2.4.0 reports `dims=[1, H=bins, W=1, C=frames]`, so
/// feeding frame-major silently yields near-uniform softmax. `out.len()` must equal `N_FRAMES * N_BINS`.
pub fn transpose_frame_major_to_bin_major<const N_FRAMES: usize, const N_BINS: usize>(
    spec: &[[f32; N_BINS]; N_FRAMES],
    out: &mut [f32],
) {
    // Length proof elides per-cell bounds checks in the loop below.
    assert_eq!(out.len(), N_FRAMES * N_BINS);
    for (f, row) in spec.iter().enumerate() {
        for (b, &v) in row.iter().enumerate() {
            out[b * N_FRAMES + f] = v;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn head_forward_bias_only() {
        let features = vec![0.0; 4];
        let weight = vec![999.0; 4 * 3];
        let bias = vec![1.0, 2.0, 3.0];
        let mut logits = vec![0.0; 3];
        head_forward(&features, &weight, &bias, &mut logits);
        assert_eq!(logits, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn head_forward_first_feature_picks_first_row() {
        let features = vec![1.0, 0.0, 0.0];
        let weight = vec![
            10.0, 20.0, 30.0, // row 0 (feature i=0): the one features=[1,0,0] selects
            -1.0, -2.0, -3.0, // row 1 (feature i=1)
            100.0, 200.0, 300.0, // row 2 (feature i=2)
        ];
        let bias = vec![0.5, 0.5, 0.5];
        let mut logits = vec![0.0; 3];
        head_forward(&features, &weight, &bias, &mut logits);
        assert_eq!(logits, vec![10.5, 20.5, 30.5]);
    }

    #[test]
    fn softmax_sums_to_one() {
        let logits = vec![1.0, 2.0, 3.0];
        let mut probs = vec![0.0; 3];
        softmax_into(&logits, &mut probs);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "sum={sum}");
        assert!(probs[2] > probs[1] && probs[1] > probs[0]);
    }

    /// ~1000 logits would overflow `exp` without max-subtraction.
    #[test]
    fn softmax_stable_under_huge_logits() {
        let logits = vec![1000.0, 1001.0, 999.0];
        let mut probs = vec![0.0; 3];
        softmax_into(&logits, &mut probs);
        assert!(probs.iter().all(|p| p.is_finite()));
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "sum={sum}");
    }

    #[test]
    fn softmax_uniform_on_neg_inf_logits() {
        let logits = vec![f32::NEG_INFINITY; 4];
        let mut probs = vec![0.0; 4];
        softmax_into(&logits, &mut probs);
        for &p in probs.iter() {
            assert!((p - 0.25).abs() < 1e-6, "p={p}");
        }
    }

    #[test]
    fn softmax_uniform_on_nan_logit_at_non_max_position() {
        let logits = vec![1.0_f32, f32::NAN, 3.0, 2.0];
        let mut probs = vec![0.0_f32; 4];
        softmax_into(&logits, &mut probs);
        for &p in probs.iter() {
            assert!(p.is_finite(), "NaN propagated to probs: {p}");
            assert!((p - 0.25).abs() < 1e-6, "p={p}");
        }
    }

    #[test]
    fn softmax_one_hot_on_single_pos_inf_logit() {
        let logits = vec![1.0_f32, f32::INFINITY, 3.0, 2.0];
        let mut probs = vec![0.0_f32; 4];
        softmax_into(&logits, &mut probs);
        for &p in probs.iter() {
            assert!(p.is_finite(), "NaN/inf propagated to probs: {p}");
        }
        assert_eq!(probs, vec![0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn softmax_splits_mass_across_multiple_pos_inf_logits() {
        let logits = vec![f32::INFINITY, 3.0, f32::INFINITY, 2.0];
        let mut probs = vec![0.0_f32; 4];
        softmax_into(&logits, &mut probs);
        assert_eq!(probs, vec![0.5, 0.0, 0.5, 0.0]);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "sum={sum}");
    }

    #[test]
    fn top_k_basic() {
        let xs = vec![0.1, 0.5, 0.3, 0.05, 0.05];
        let mut out = Vec::with_capacity(8);
        top_k_indices_into(&xs, 3, &mut out);
        assert_eq!(out, vec![1, 2, 0]);
    }

    #[test]
    fn top_k_caps_at_len() {
        let xs = vec![0.4, 0.6];
        let mut out = Vec::with_capacity(8);
        top_k_indices_into(&xs, 100, &mut out);
        assert_eq!(out, vec![1, 0]);
    }

    #[test]
    fn top_k_zero_yields_empty() {
        let xs = vec![0.4, 0.6];
        let mut out = vec![999; 4];
        top_k_indices_into(&xs, 0, &mut out);
        assert!(out.is_empty());
    }

    #[test]
    fn top_k_reuse_does_not_allocate_when_capacity_reserved() {
        let xs = vec![0.2; 32];
        let mut out: Vec<usize> = Vec::with_capacity(32);
        let initial_cap = out.capacity();
        for _ in 0..10 {
            top_k_indices_into(&xs, 5, &mut out);
        }
        assert!(
            out.capacity() <= initial_cap,
            "capacity grew across reuses (initial={initial_cap}, now={})",
            out.capacity(),
        );
    }

    #[test]
    fn transpose_basic() {
        let mut spec = [[0.0f32; 3]; 2];
        spec[0] = [1.0, 2.0, 3.0];
        spec[1] = [10.0, 20.0, 30.0];
        let mut flat = vec![0.0f32; 6];
        transpose_frame_major_to_bin_major::<2, 3>(&spec, &mut flat);
        assert_eq!(flat, vec![1.0, 10.0, 2.0, 20.0, 3.0, 30.0]);
    }
}
