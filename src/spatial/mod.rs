//! Spatial statistics for spatial transcriptomics data.
//!
//! The central task is identifying **spatially variable genes** — genes whose
//! expression is organised in space rather than scattered at random across the
//! tissue. Both statistics here answer that question against a spatial neighbour
//! graph, and disagree usefully: Moran's I is a global correlation and responds to
//! broad gradients and domains, while Geary's C is built from pairwise differences
//! and is more sensitive to sharp local boundaries.
//!
//! # The weight matrix
//!
//! Both take a sparse `spots x spots` weight matrix `W`, typically a k-nearest-
//! neighbour graph over the spatial coordinates. Weights need not be symmetric or
//! normalised; the statistics account for `W` explicitly. Self-weights `w_ii` should
//! be zero and are ignored if present.
//!
//! # Inference
//!
//! p-values come from the normality-assumption ("normal approximation") variance,
//! which is standard practice and cheap. It assumes the expression values are
//! approximately normal — reasonable for log-normalised data, less so for raw counts.
//! For heavily zero-inflated genes prefer a permutation null.

use num_traits::AsPrimitive;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use single_utilities::traits::FloatOpsTS;
use statrs::distribution::ContinuousCDF;

use crate::testing::utils::{standard_normal, Rng, SparseMatrixRef};
use crate::testing::TestResult;

/// Weight-matrix moments shared by both statistics.
///
/// `s0 = sum_ij w_ij`, `s1 = 0.5 * sum_ij (w_ij + w_ji)^2`, and
/// `s2 = sum_i (sum_j w_ij + sum_j w_ji)^2`.
#[derive(Debug, Clone, Copy)]
pub struct WeightMoments {
    pub s0: f64,
    pub s1: f64,
    pub s2: f64,
    pub n: usize,
}

impl WeightMoments {
    /// Compute the moments of a sparse spatial weight matrix.
    ///
    /// `w` must be square. Self-weights are ignored.
    pub fn from_weights<T, N, I>(w: SparseMatrixRef<T, N, I>) -> anyhow::Result<Self>
    where
        T: FloatOpsTS,
        N: AsPrimitive<usize> + Send + Sync,
        I: AsPrimitive<usize> + Send + Sync,
    {
        let n = w.n_rows;
        if n != w.n_cols {
            return Err(anyhow::anyhow!(
                "Single-Statistics | Spatial weight matrix must be square, got {}x{}. Error code: SS-SPT-001",
                w.n_rows,
                w.n_cols
            ));
        }
        if n == 0 {
            return Err(anyhow::anyhow!(
                "Single-Statistics | Spatial weight matrix is empty. Error code: SS-SPT-002"
            ));
        }

        // Row sums and column sums, and s1 which needs w_ji alongside w_ij.
        let mut row_sums = vec![0.0f64; n];
        let mut col_sums = vec![0.0f64; n];
        let mut s0 = 0.0f64;

        for (i, row_sum) in row_sums.iter_mut().enumerate() {
            let (cols, vals) = w.get_major(i);
            for (c, &v) in cols.iter().zip(vals.iter()) {
                let j = c.as_();
                if j == i {
                    continue; // ignore self-weights
                }
                let val = v.to_f64().unwrap_or(0.0);
                *row_sum += val;
                col_sums[j] += val;
                s0 += val;
            }
        }

        // s1 needs the transpose, so look each w_ji up directly.
        let mut s1 = 0.0f64;
        for i in 0..n {
            let (cols, vals) = w.get_major(i);
            for (c, &v) in cols.iter().zip(vals.iter()) {
                let j = c.as_();
                if j == i {
                    continue;
                }
                let w_ij = v.to_f64().unwrap_or(0.0);
                let w_ji = w.get_entry(j, i).to_f64().unwrap_or(0.0);
                let sum = w_ij + w_ji;
                s1 += sum * sum;
            }
        }
        s1 *= 0.5;

        let s2: f64 = (0..n)
            .map(|i| {
                let t = row_sums[i] + col_sums[i];
                t * t
            })
            .sum();

        if s0 <= 0.0 {
            return Err(anyhow::anyhow!(
                "Single-Statistics | Spatial weight matrix has zero total weight. Error code: SS-SPT-003"
            ));
        }

        Ok(Self { s0, s1, s2, n })
    }
}

/// Gather one gene's expression across all spots, materialising implicit zeros.
fn gene_vector<T, N, I>(matrix: &SparseMatrixRef<T, N, I>, gene: usize, n_spots: usize) -> Vec<f64>
where
    T: FloatOpsTS,
    N: AsPrimitive<usize> + Send + Sync,
    I: AsPrimitive<usize> + Send + Sync,
{
    let mut x = vec![0.0f64; n_spots];
    let (cols, vals) = matrix.get_major(gene);
    for (c, &v) in cols.iter().zip(vals.iter()) {
        let idx = c.as_();
        if idx < n_spots {
            x[idx] = v.to_f64().unwrap_or(0.0);
        }
    }
    x
}

#[inline]
fn two_sided_p(z: f64) -> f64 {
    if !z.is_finite() {
        return 1.0;
    }
    (2.0 * (1.0 - standard_normal().cdf(z.abs()))).clamp(0.0, 1.0)
}


#[derive(Clone, Copy, PartialEq)]
enum Stat {
    Moran,
    Geary,
}

/// Both statistics are functions of the deviations from the mean, which is why a
/// permutation can shuffle `dev` directly — the mean and total sum of squares are
/// invariant under relabelling, so only the cross term changes.
fn stat_from_dev<TW, NW, IW>(
    stat: Stat,
    dev: &[f64],
    w: &SparseMatrixRef<TW, NW, IW>,
    s0: f64,
    denom: f64,
    n: f64,
) -> f64
where
    TW: FloatOpsTS,
    NW: AsPrimitive<usize> + Send + Sync,
    IW: AsPrimitive<usize> + Send + Sync,
{
    let mut acc = 0.0f64;
    for i in 0..dev.len() {
        let (cols, vals) = w.get_major(i);
        for (c, &v) in cols.iter().zip(vals.iter()) {
            let j = c.as_();
            if j == i {
                continue;
            }
            let wij = v.to_f64().unwrap_or(0.0);
            acc += match stat {
                Stat::Moran => wij * dev[i] * dev[j],
                Stat::Geary => {
                    let d = dev[i] - dev[j];
                    wij * d * d
                }
            };
        }
    }

    match stat {
        Stat::Moran => (n / s0) * (acc / denom),
        Stat::Geary => ((n - 1.0) * acc) / (2.0 * s0 * denom),
    }
}

/// Shared driver. `n_perm == 0` uses the analytic normality-assumption variance;
/// otherwise the null comes from shuffling spot labels.
fn run<T, N, I, TW, NW, IW>(
    stat: Stat,
    matrix: SparseMatrixRef<T, N, I>,
    weights: SparseMatrixRef<TW, NW, IW>,
    n_perm: usize,
    seed: u64,
) -> anyhow::Result<Vec<TestResult<f64>>>
where
    T: FloatOpsTS,
    N: AsPrimitive<usize> + Send + Sync,
    I: AsPrimitive<usize> + Send + Sync,
    TW: FloatOpsTS,
    NW: AsPrimitive<usize> + Send + Sync,
    IW: AsPrimitive<usize> + Send + Sync,
{
    let m = WeightMoments::from_weights(weights)?;
    let n_spots = m.n;

    if matrix.n_cols != n_spots {
        return Err(anyhow::anyhow!(
            "Single-Statistics | Matrix has {} spots but the weight matrix has {}. Error code: SS-SPT-004",
            matrix.n_cols,
            n_spots
        ));
    }

    let n = n_spots as f64;
    let s0_sq = m.s0 * m.s0;
    let expected = match stat {
        Stat::Moran => -1.0 / (n - 1.0),
        Stat::Geary => 1.0,
    };

    // Depends only on W, so hoist it out of the per-gene loop.
    let variance = match stat {
        Stat::Moran => {
            let num = n * n * m.s1 - n * m.s2 + 3.0 * s0_sq;
            num / (s0_sq * (n * n - 1.0)) - expected * expected
        }
        Stat::Geary => {
            ((2.0 * m.s1 + m.s2) * (n - 1.0) - 4.0 * s0_sq) / (2.0 * (n + 1.0) * s0_sq)
        }
    };

    let results = (0..matrix.n_rows)
        .into_par_iter()
        .map(|gene| {
            let x = gene_vector(&matrix, gene, n_spots);
            let mean = x.iter().sum::<f64>() / n;
            let dev: Vec<f64> = x.iter().map(|v| v - mean).collect();
            let denom: f64 = dev.iter().map(|d| d * d).sum();

            if denom <= 0.0 {
                // A constant gene has no spatial signal to detect.
                return TestResult::new(f64::NAN, 1.0)
                    .with_metadata("z_score", f64::NAN)
                    .with_metadata("expected", expected);
            }

            let value = stat_from_dev(stat, &dev, &weights, m.s0, denom, n);

            if n_perm == 0 {
                let z = if variance > 0.0 {
                    (value - expected) / variance.sqrt()
                } else {
                    f64::NAN
                };
                return TestResult::new(value, two_sided_p(z))
                    .with_metadata("z_score", z)
                    .with_metadata("expected", expected)
                    .with_metadata("variance", variance);
            }

            // Permutation null: relabel spots, recompute, count at-least-as-extreme.
            let mut rng = Rng::new(seed.wrapping_add(gene as u64));
            let mut shuffled = dev.clone();
            let mut extreme = 0usize;
            let mut sum = 0.0f64;
            let mut sum_sq = 0.0f64;

            for _ in 0..n_perm {
                rng.shuffle(&mut shuffled);
                let v = stat_from_dev(stat, &shuffled, &weights, m.s0, denom, n);
                sum += v;
                sum_sq += v * v;
                // Geary is inverted: small values mean strong clustering.
                let as_extreme = match stat {
                    Stat::Moran => (v - expected).abs() >= (value - expected).abs(),
                    Stat::Geary => (v - 1.0).abs() >= (value - 1.0).abs(),
                };
                if as_extreme {
                    extreme += 1;
                }
            }

            let np = n_perm as f64;
            let null_mean = sum / np;
            let null_var = (sum_sq / np - null_mean * null_mean).max(0.0);
            let z = if null_var > 0.0 {
                (value - null_mean) / null_var.sqrt()
            } else {
                f64::NAN
            };

            TestResult::new(value, (extreme + 1) as f64 / (np + 1.0))
                .with_metadata("z_score", z)
                .with_metadata("expected", null_mean)
                .with_metadata("variance", null_var)
        })
        .collect();

    Ok(results)
}

/// Moran's I for every gene against a spatial weight matrix.
///
/// `matrix` is genes (major axis) × spots (minor axis); `weights` is `spots x spots`.
///
/// I is roughly `+1` for strong spatial clustering, `0` for no spatial structure, and
/// negative for a checkerboard pattern. The null expectation is `-1/(n-1)`, not zero.
pub fn morans_i<T, N, I, TW, NW, IW>(
    matrix: SparseMatrixRef<T, N, I>,
    weights: SparseMatrixRef<TW, NW, IW>,
) -> anyhow::Result<Vec<TestResult<f64>>>
where
    T: FloatOpsTS,
    N: AsPrimitive<usize> + Send + Sync,
    I: AsPrimitive<usize> + Send + Sync,
    TW: FloatOpsTS,
    NW: AsPrimitive<usize> + Send + Sync,
    IW: AsPrimitive<usize> + Send + Sync,
{
    run(Stat::Moran, matrix, weights, 0, 0)
}

/// Geary's C for every gene against a spatial weight matrix.
///
/// C is centred on `1.0`: below 1 is positive spatial autocorrelation (neighbouring
/// spots are similar), above 1 is negative. Built from squared pairwise differences,
/// so it reacts more sharply to local discontinuities than Moran's I.
pub fn gearys_c<T, N, I, TW, NW, IW>(
    matrix: SparseMatrixRef<T, N, I>,
    weights: SparseMatrixRef<TW, NW, IW>,
) -> anyhow::Result<Vec<TestResult<f64>>>
where
    T: FloatOpsTS,
    N: AsPrimitive<usize> + Send + Sync,
    I: AsPrimitive<usize> + Send + Sync,
    TW: FloatOpsTS,
    NW: AsPrimitive<usize> + Send + Sync,
    IW: AsPrimitive<usize> + Send + Sync,
{
    run(Stat::Geary, matrix, weights, 0, 0)
}

/// [`morans_i`] with a permutation null instead of the normality assumption.
///
/// Prefer this for zero-inflated genes, where the analytic variance is unreliable.
/// Costs `n_perm` extra passes over the weight matrix per gene, so it is usually run
/// on candidate genes rather than the whole matrix.
pub fn morans_i_permutation<T, N, I, TW, NW, IW>(
    matrix: SparseMatrixRef<T, N, I>,
    weights: SparseMatrixRef<TW, NW, IW>,
    n_perm: usize,
    seed: u64,
) -> anyhow::Result<Vec<TestResult<f64>>>
where
    T: FloatOpsTS,
    N: AsPrimitive<usize> + Send + Sync,
    I: AsPrimitive<usize> + Send + Sync,
    TW: FloatOpsTS,
    NW: AsPrimitive<usize> + Send + Sync,
    IW: AsPrimitive<usize> + Send + Sync,
{
    run(Stat::Moran, matrix, weights, n_perm.max(1), seed)
}

/// [`gearys_c`] with a permutation null instead of the normality assumption.
pub fn gearys_c_permutation<T, N, I, TW, NW, IW>(
    matrix: SparseMatrixRef<T, N, I>,
    weights: SparseMatrixRef<TW, NW, IW>,
    n_perm: usize,
    seed: u64,
) -> anyhow::Result<Vec<TestResult<f64>>>
where
    T: FloatOpsTS,
    N: AsPrimitive<usize> + Send + Sync,
    I: AsPrimitive<usize> + Send + Sync,
    TW: FloatOpsTS,
    NW: AsPrimitive<usize> + Send + Sync,
    IW: AsPrimitive<usize> + Send + Sync,
{
    run(Stat::Geary, matrix, weights, n_perm.max(1), seed)
}
