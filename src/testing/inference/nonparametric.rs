//! Non-parametric statistical tests for single-cell data analysis.
//!
//! This module implements non-parametric statistical tests that make fewer assumptions about
//! data distribution. These tests are particularly useful for single-cell data which often
//! exhibits non-normal distributions, high sparsity, and outliers.
//!
//! The primary test implemented is the Mann-Whitney U test (also known as the Wilcoxon 
//! rank-sum test), which compares the distributions of two groups without assuming normality.

use std::{cmp::Ordering, f64};

use rayon::iter::{IntoParallelIterator, ParallelIterator};
use single_utilities::traits::FloatOpsTS;
use statrs::distribution::{ChiSquared, ContinuousCDF};

use crate::testing::{Alternative, TestResult};
use crate::testing::utils::{standard_normal, SparseMatrixRef, SprsView};
use num_traits::AsPrimitive;

/// Perform Mann-Whitney U tests on all genes comparing two groups of cells.
pub fn mann_whitney_matrix_groups<T, I, Iptr, IptrStorage, IndStorage, DataStorage>(
    matrix: &sprs::CsMatBase<T, I, IptrStorage, IndStorage, DataStorage, Iptr>,
    group1_indices: &[usize],
    group2_indices: &[usize],
    alternative: Alternative,
) -> anyhow::Result<Vec<TestResult<f64>>>
where
    T: FloatOpsTS,
    I: sprs::indexing::SpIndex + AsPrimitive<usize>,
    Iptr: sprs::indexing::SpIndex + AsPrimitive<usize>,
    IptrStorage: std::ops::Deref<Target = [Iptr]> + Send + Sync,
    IndStorage: std::ops::Deref<Target = [I]> + Send + Sync,
    DataStorage: std::ops::Deref<Target = [T]> + Send + Sync,
    f64: std::convert::From<T>,
{
    mann_whitney_sparse(
        SprsView::new(matrix).as_matrix_ref(),
        group1_indices,
        group2_indices,
        alternative,
    )
}

/// Perform Mann-Whitney U tests on a sparse matrix represented by raw components.
pub fn mann_whitney_sparse<T, N, I>(
    matrix: SparseMatrixRef<T, N, I>,
    group1_indices: &[usize],
    group2_indices: &[usize],
    alternative: Alternative,
) -> anyhow::Result<Vec<TestResult<f64>>>
where
    T: FloatOpsTS,
    N: AsPrimitive<usize> + Send + Sync,
    I: AsPrimitive<usize> + Send + Sync,
    f64: std::convert::From<T>,
{
    if group1_indices.is_empty() || group2_indices.is_empty() {
        return Err(anyhow::anyhow!(
            "Single-Statistics | Group indices cannot be empty. Error code: SS-NP-001"
        ));
    }

    let nrows = matrix.n_rows;
    let n_group1 = group1_indices.len();
    let n_group2 = group2_indices.len();

    // Mapping from column index to group ID (0 for none, 1 for group1, 2 for group2)
    let mut cell_groups = vec![0u8; matrix.n_cols];
    for &idx in group1_indices {
        if idx < cell_groups.len() { cell_groups[idx] = 1; }
    }
    for &idx in group2_indices {
        if idx < cell_groups.len() { cell_groups[idx] = 2; }
    }

    let results: Vec<_> = (0..nrows)
        .into_par_iter()
        .map(|row| {
            let start = matrix.maj_ind[row].as_();
            let end = matrix.maj_ind[row + 1].as_();
            
            let mut x_nonzero = Vec::new();
            let mut y_nonzero = Vec::new();
            let mut g1_nz_count = 0;
            let mut g2_nz_count = 0;

            for i in start..end {
                let col = matrix.min_ind[i].as_();
                let val = f64::from(matrix.val[i]);
                
                // A stored entry that happens to be 0.0 must still be treated as a
                // zero. Counting it as a non-zero would drop it from both the
                // non-zero list and the zero count, removing it from `nx`/`ny`
                // entirely and corrupting the rank sums.
                if val == 0.0 {
                    continue;
                }

                match cell_groups[col] {
                    1 => {
                        x_nonzero.push(val);
                        g1_nz_count += 1;
                    }
                    2 => {
                        y_nonzero.push(val);
                        g2_nz_count += 1;
                    }
                    _ => {}
                }
            }
            
            let x_zeros = n_group1 - g1_nz_count;
            let y_zeros = n_group2 - g2_nz_count;

            mann_whitney_from_sparse_parts(x_nonzero, y_nonzero, x_zeros, y_zeros, alternative)
        })
        .collect();

    Ok(results)
}

/// Rank a sorted non-zero list sitting above a shared block of `n_zeros` tied zeros,
/// calling `emit(group, rank)` per element. Returns sum(t^3 - t) over every tie group,
/// the zero block included.
///
/// Callers add their own zero-block rank sums; only the walk and the tie correction
/// are shared, since Mann-Whitney tracks one group and Kruskal-Wallis tracks k.
fn rank_above_zero_block<G: Copy>(
    sorted: &[(f64, G)],
    n_zeros: usize,
    mut emit: impl FnMut(G, f64),
) -> f64 {
    let z = n_zeros as f64;
    let mut tie = if n_zeros > 0 { z * z * z - z } else { 0.0 };
    let mut rank = z + 1.0;

    let mut i = 0;
    while i < sorted.len() {
        let val = sorted[i].0;
        let start = i;
        while i < sorted.len() && sorted[i].0 == val {
            i += 1;
        }
        let count = (i - start) as f64;
        let avg = rank + (count - 1.0) / 2.0;
        for &(_, g) in &sorted[start..i] {
            emit(g, avg);
        }
        if count > 1.0 {
            tie += count * count * count - count;
        }
        rank += count;
    }
    tie
}

/// Core MW-U logic optimized for sparse scRNA-seq data (many zeros).
fn mann_whitney_from_sparse_parts(
    x_nonzero: Vec<f64>,
    y_nonzero: Vec<f64>,
    x_zeros: usize,
    y_zeros: usize,
    alternative: Alternative,
) -> TestResult<f64> {
    let nx = x_zeros + x_nonzero.len();
    let ny = y_zeros + y_nonzero.len();

    if nx == 0 || ny == 0 {
        return TestResult::new(f64::NAN, 1.0);
    }

    let mut combined_nz: Vec<(f64, u8)> = Vec::with_capacity(x_nonzero.len() + y_nonzero.len());
    for v in x_nonzero { combined_nz.push((v, 0)); }
    for v in y_nonzero { combined_nz.push((v, 1)); }
    combined_nz.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));

    let n_total = (nx + ny) as f64;
    let n_zeros = x_zeros + y_zeros;

    // Group 1's share of the tied zero block, then the non-zeros above it.
    let mut rank_sum_x = x_zeros as f64 * ((n_zeros as f64 + 1.0) / 2.0);
    let tie_correction = rank_above_zero_block(&combined_nz, n_zeros, |group, rank| {
        if group == 0 {
            rank_sum_x += rank;
        }
    });

    let nx_f = nx as f64;
    let ny_f = ny as f64;
    let u_x = rank_sum_x - (nx_f * (nx_f + 1.0)) / 2.0;
    let u_y = (nx_f * ny_f) - u_x;
    let mean_u = nx_f * ny_f / 2.0;
    
    let var_u = (nx_f * ny_f / (n_total * (n_total - 1.0))) * 
                ((n_total.powi(3) - n_total - tie_correction) / 12.0);

    let (u_stat, z) = match alternative {
        Alternative::TwoSided => {
            let u = u_x.min(u_y);
            let z_score = if var_u > 0.0 {
                ((u - mean_u).abs() - 0.5).max(0.0) / var_u.sqrt()
            } else { 0.0 };
            (u, z_score)
        },
        Alternative::Greater => {
            let z_score = if var_u > 0.0 {
                (u_x - mean_u - 0.5) / var_u.sqrt()
            } else { 0.0 };
            (u_x, z_score)
        },
        Alternative::Less => {
            let z_score = if var_u > 0.0 {
                (u_x - mean_u + 0.5) / var_u.sqrt()
            } else { 0.0 };
            (u_x, z_score)
        }
    };

    let p = calculate_p_value(z, alternative, nx_f, ny_f);
    TestResult::new(u_stat, p)
        .with_metadata("z_score", z)
        .with_metadata("var_u", var_u)
        .with_metadata("tie_correction", tie_correction)
}

/// Assign average (mid-) ranks to `values`, 1-based, in the original order.
///
/// Returns the ranks alongside the tie-correction term `sum(t^3 - t)` accumulated
/// over each group of tied values, which both the signed-rank and Kruskal-Wallis
/// variance corrections need.
fn average_ranks(values: &[f64]) -> (Vec<f64>, f64) {
    let n = values.len();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_unstable_by(|&a, &b| {
        values[a].partial_cmp(&values[b]).unwrap_or(Ordering::Equal)
    });

    let mut ranks = vec![0.0; n];
    let mut tie_correction = 0.0;
    let mut i = 0;
    while i < n {
        let v = values[order[i]];
        let start = i;
        while i < n && values[order[i]] == v {
            i += 1;
        }
        // Positions start+1 ..= i are 1-based ranks; their mean is the mid-rank.
        let avg = (start + 1 + i) as f64 / 2.0;
        for &idx in &order[start..i] {
            ranks[idx] = avg;
        }
        let count = (i - start) as f64;
        if count > 1.0 {
            tie_correction += count * count * count - count;
        }
    }

    (ranks, tie_correction)
}

/// Wilcoxon signed-rank test for two *paired* samples.
///
/// This is the paired counterpart to the Mann-Whitney U test (which is itself the
/// Wilcoxon *rank-sum* test). Pairs are formed elementwise, so `x` and `y` must be
/// the same length and ordered consistently.
///
/// Zero differences are discarded before ranking (Wilcoxon's original treatment,
/// matching R's `wilcox.test(paired = TRUE)`), and the reported statistic is `V`,
/// the sum of ranks over positive differences.
///
/// # Errors
/// Returns an error if `x` and `y` have different lengths.
pub fn wilcoxon_signed_rank(
    x: &[f64],
    y: &[f64],
    alternative: Alternative,
) -> anyhow::Result<TestResult<f64>> {
    if x.len() != y.len() {
        return Err(anyhow::anyhow!(
            "Single-Statistics | Wilcoxon signed-rank requires paired samples of equal length, got {} and {}. Error code: SS-NP-002",
            x.len(),
            y.len()
        ));
    }

    let diffs: Vec<f64> = x
        .iter()
        .zip(y.iter())
        .map(|(&a, &b)| a - b)
        .filter(|d| d.is_finite() && *d != 0.0)
        .collect();

    Ok(signed_rank_from_diffs(&diffs, alternative))
}

/// One-sample Wilcoxon signed-rank test against a hypothesised median `mu0`.
pub fn wilcoxon_signed_rank_one_sample(
    x: &[f64],
    mu0: f64,
    alternative: Alternative,
) -> TestResult<f64> {
    let diffs: Vec<f64> = x
        .iter()
        .map(|&a| a - mu0)
        .filter(|d| d.is_finite() && *d != 0.0)
        .collect();

    signed_rank_from_diffs(&diffs, alternative)
}

/// Core signed-rank computation over already-filtered non-zero differences.
fn signed_rank_from_diffs(diffs: &[f64], alternative: Alternative) -> TestResult<f64> {
    let n = diffs.len();
    if n == 0 {
        // Every pair was tied; there is no evidence of a shift in either direction.
        return TestResult::new(0.0, 1.0);
    }

    let abs_diffs: Vec<f64> = diffs.iter().map(|d| d.abs()).collect();
    let (ranks, tie_correction) = average_ranks(&abs_diffs);

    let w_plus: f64 = diffs
        .iter()
        .zip(ranks.iter())
        .filter(|(d, _)| **d > 0.0)
        .map(|(_, r)| *r)
        .sum();

    let n_f = n as f64;
    let mean_w = n_f * (n_f + 1.0) / 4.0;
    let var_w = n_f * (n_f + 1.0) * (2.0 * n_f + 1.0) / 24.0 - tie_correction / 48.0;

    let z = if var_w > 0.0 {
        let sd = var_w.sqrt();
        match alternative {
            Alternative::TwoSided => ((w_plus - mean_w).abs() - 0.5).max(0.0) / sd,
            Alternative::Greater => (w_plus - mean_w - 0.5) / sd,
            Alternative::Less => (w_plus - mean_w + 0.5) / sd,
        }
    } else {
        0.0
    };

    let p_value = if !z.is_finite() {
        1.0
    } else {
        match alternative {
            Alternative::TwoSided => (2.0 * (1.0 - standard_normal().cdf(z.abs()))).clamp(0.0, 1.0),
            Alternative::Greater => 1.0 - standard_normal().cdf(z),
            Alternative::Less => standard_normal().cdf(z),
        }
    };

    TestResult::new(w_plus, p_value)
        .with_metadata("z_score", z)
        .with_metadata("n_pairs", n_f)
        .with_metadata("var_w", var_w)
}

/// Kruskal-Wallis H test: a non-parametric one-way ANOVA across `k >= 2` groups.
///
/// Generalises the Mann-Whitney U test to more than two groups. Returns the
/// tie-corrected H statistic and its chi-square p-value on `k - 1` degrees of freedom.
///
/// # Errors
/// Returns an error if fewer than two groups are supplied or if every group is empty.
pub fn kruskal_wallis(groups: &[&[f64]]) -> anyhow::Result<TestResult<f64>> {
    let k = groups.len();
    if k < 2 {
        return Err(anyhow::anyhow!(
            "Single-Statistics | Kruskal-Wallis requires at least two groups, got {}. Error code: SS-NP-003",
            k
        ));
    }

    let pooled: Vec<f64> = groups.iter().flat_map(|g| g.iter().copied()).collect();
    let n_total = pooled.len();
    if n_total == 0 {
        return Err(anyhow::anyhow!(
            "Single-Statistics | Kruskal-Wallis received only empty groups. Error code: SS-NP-004"
        ));
    }

    let (ranks, tie_correction) = average_ranks(&pooled);

    let n_f = n_total as f64;
    let mut h_sum = 0.0;
    let mut offset = 0usize;
    let mut non_empty = 0usize;
    for g in groups {
        if g.is_empty() {
            continue;
        }
        non_empty += 1;
        let rank_sum: f64 = ranks[offset..offset + g.len()].iter().sum();
        h_sum += rank_sum * rank_sum / g.len() as f64;
        offset += g.len();
    }

    if non_empty < 2 {
        return Ok(TestResult::new(0.0, 1.0));
    }

    let mut h = 12.0 / (n_f * (n_f + 1.0)) * h_sum - 3.0 * (n_f + 1.0);

    // Tie correction: divide by 1 - sum(t^3 - t) / (N^3 - N)
    let denom = 1.0 - tie_correction / (n_f * n_f * n_f - n_f);
    if denom > 0.0 {
        h /= denom;
    }

    let df = (non_empty - 1) as f64;
    let p_value = match ChiSquared::new(df) {
        Ok(dist) if h.is_finite() && h >= 0.0 => (1.0 - dist.cdf(h)).clamp(0.0, 1.0),
        _ => 1.0,
    };

    Ok(TestResult::new(h, p_value)
        .with_degrees_of_freedom(df)
        .with_metadata("tie_correction", tie_correction))
}

/// Kruskal-Wallis H test across `k` groups for every gene in a sparse matrix.
///
/// `group_ids` assigns each cell (column) to a group; cells whose id is not present
/// in `unique_groups` are ignored. Unlike the two-group tests this places no limit
/// on the number of groups.
///
/// Like [`mann_whitney_sparse`], this exploits sparsity by treating unstored entries
/// as zeros and ranking the resulting zero block analytically, so it assumes zero is
/// the *smallest* value in the matrix. That holds for counts and log1p-normalised
/// data; it does not hold for centred or scaled data.
pub fn kruskal_wallis_sparse<T, N, I>(
    matrix: SparseMatrixRef<T, N, I>,
    group_ids: &[usize],
) -> anyhow::Result<Vec<TestResult<f64>>>
where
    T: FloatOpsTS,
    N: AsPrimitive<usize> + Send + Sync,
    I: AsPrimitive<usize> + Send + Sync,
{
    if group_ids.len() != matrix.n_cols {
        return Err(anyhow::anyhow!(
            "Single-Statistics | group_ids has length {} but the matrix has {} columns. Error code: SS-NP-005",
            group_ids.len(),
            matrix.n_cols
        ));
    }

    let unique_groups = crate::testing::utils::extract_unique_groups(group_ids);
    let k = unique_groups.len();
    if k < 2 {
        return Err(anyhow::anyhow!(
            "Single-Statistics | Kruskal-Wallis requires at least two distinct groups, found {}. Error code: SS-NP-003",
            k
        ));
    }

    // Dense id -> compact index map, so the per-gene loop indexes an array.
    let max_id = unique_groups.last().copied().unwrap_or(0);
    let mut group_slot = vec![usize::MAX; max_id + 1];
    for (slot, &gid) in unique_groups.iter().enumerate() {
        group_slot[gid] = slot;
    }
    let cell_slot: Vec<usize> = group_ids.iter().map(|&g| group_slot[g]).collect();

    let mut group_sizes = vec![0usize; k];
    for &slot in &cell_slot {
        group_sizes[slot] += 1;
    }

    let results: Vec<_> = (0..matrix.n_rows)
        .into_par_iter()
        .map(|row| {
            let start = matrix.maj_ind[row].as_();
            let end = matrix.maj_ind[row + 1].as_();

            let mut nonzero: Vec<(f64, usize)> = Vec::with_capacity(end - start);
            let mut nonzero_per_group = vec![0usize; k];

            for i in start..end {
                let val = matrix.val[i].to_f64().unwrap_or(0.0);
                if val == 0.0 {
                    continue;
                }
                let slot = cell_slot[matrix.min_ind[i].as_()];
                nonzero.push((val, slot));
                nonzero_per_group[slot] += 1;
            }

            let zeros_per_group: Vec<usize> = group_sizes
                .iter()
                .zip(nonzero_per_group.iter())
                .map(|(&size, &nz)| size - nz)
                .collect();

            kruskal_wallis_from_sparse_parts(nonzero, &zeros_per_group, &group_sizes)
        })
        .collect();

    Ok(results)
}

/// Core Kruskal-Wallis computation for one gene, with the zero block ranked in bulk.
fn kruskal_wallis_from_sparse_parts(
    mut nonzero: Vec<(f64, usize)>,
    zeros_per_group: &[usize],
    group_sizes: &[usize],
) -> TestResult<f64> {
    let k = group_sizes.len();
    let n_total: usize = group_sizes.iter().sum();
    if n_total == 0 {
        return TestResult::new(0.0, 1.0);
    }

    let mut rank_sums = vec![0.0f64; k];

    // The zeros all tie at the bottom of the ranking, split across the groups.
    let n_zeros: usize = zeros_per_group.iter().sum();
    let avg_rank_zeros = (n_zeros as f64 + 1.0) / 2.0;
    for (slot, &count) in zeros_per_group.iter().enumerate() {
        rank_sums[slot] += count as f64 * avg_rank_zeros;
    }

    nonzero.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
    let tie_correction = rank_above_zero_block(&nonzero, n_zeros, |slot, rank| {
        rank_sums[slot] += rank;
    });

    let n_f = n_total as f64;
    let mut h_sum = 0.0;
    let mut non_empty = 0usize;
    for (slot, &size) in group_sizes.iter().enumerate() {
        if size == 0 {
            continue;
        }
        non_empty += 1;
        h_sum += rank_sums[slot] * rank_sums[slot] / size as f64;
    }

    if non_empty < 2 {
        return TestResult::new(0.0, 1.0);
    }

    let mut h = 12.0 / (n_f * (n_f + 1.0)) * h_sum - 3.0 * (n_f + 1.0);

    let denom = 1.0 - tie_correction / (n_f * n_f * n_f - n_f);
    if denom > 0.0 {
        h /= denom;
    }

    let df = (non_empty - 1) as f64;
    let p_value = match ChiSquared::new(df) {
        Ok(dist) if h.is_finite() && h >= 0.0 => (1.0 - dist.cdf(h)).clamp(0.0, 1.0),
        _ => 1.0,
    };

    TestResult::new(h, p_value)
        .with_degrees_of_freedom(df)
        .with_metadata("tie_correction", tie_correction)
}

/// Wilcoxon signed-rank test for every gene in a sparse matrix, over paired cells.
///
/// `group1_indices[i]` is paired with `group2_indices[i]`, so the two slices must be
/// the same length and ordered consistently (e.g. the same donor before and after
/// treatment). Pairs where both cells are unexpressed contribute a zero difference
/// and are discarded, as in the dense test.
pub fn wilcoxon_signed_rank_sparse<T, N, I>(
    matrix: SparseMatrixRef<T, N, I>,
    group1_indices: &[usize],
    group2_indices: &[usize],
    alternative: Alternative,
) -> anyhow::Result<Vec<TestResult<f64>>>
where
    T: FloatOpsTS,
    N: AsPrimitive<usize> + Send + Sync,
    I: AsPrimitive<usize> + Send + Sync,
{
    if group1_indices.len() != group2_indices.len() {
        return Err(anyhow::anyhow!(
            "Single-Statistics | Wilcoxon signed-rank requires paired groups of equal length, got {} and {}. Error code: SS-NP-002",
            group1_indices.len(),
            group2_indices.len()
        ));
    }
    if group1_indices.is_empty() {
        return Err(anyhow::anyhow!(
            "Single-Statistics | Group indices cannot be empty. Error code: SS-NP-001"
        ));
    }

    let results: Vec<_> = (0..matrix.n_rows)
        .into_par_iter()
        .map(|row| {
            let diffs: Vec<f64> = group1_indices
                .iter()
                .zip(group2_indices.iter())
                .map(|(&c1, &c2)| {
                    let a = matrix.get_entry(row, c1).to_f64().unwrap_or(0.0);
                    let b = matrix.get_entry(row, c2).to_f64().unwrap_or(0.0);
                    a - b
                })
                .filter(|d| d.is_finite() && *d != 0.0)
                .collect();

            signed_rank_from_diffs(&diffs, alternative)
        })
        .collect();

    Ok(results)
}

/// Public API for two samples (dense).
pub fn mann_whitney_optimized(x: &[f64], y: &[f64], alternative: Alternative) -> TestResult<f64> {
    let mut x_nz = Vec::new();
    let mut x_z = 0;
    for &v in x { if v.is_finite() { if v == 0.0 { x_z += 1; } else { x_nz.push(v); } } }

    let mut y_nz = Vec::new();
    let mut y_z = 0;
    for &v in y { if v.is_finite() { if v == 0.0 { y_z += 1; } else { y_nz.push(v); } } }

    mann_whitney_from_sparse_parts(x_nz, y_nz, x_z, y_z, alternative)
}

#[inline]
fn calculate_p_value(z: f64, alternative: Alternative, nx: f64, ny: f64) -> f64 {
    if nx < 3.0 || ny < 3.0 { return 1.0; }
    if !z.is_finite() { return 1.0; }

    match alternative {
        Alternative::TwoSided => 2.0 * (1.0 - standard_normal().cdf(z.abs())),
        Alternative::Greater => 1.0 - standard_normal().cdf(z),
        Alternative::Less => standard_normal().cdf(z),
    }
}

