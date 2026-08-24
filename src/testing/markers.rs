//! Marker gene statistics for single-cell differential expression.
//!
//! A p-value alone is rarely enough to call a marker gene. This module produces the
//! descriptive statistics that accompany it in practice — group means, the fraction
//! of cells expressing the gene in each group, log fold change, and the AUROC — in a
//! single parallel pass over the matrix.
//!
//! # Zero handling
//!
//! Unstored entries count as zeros, and an *explicitly stored* `0.0` is treated
//! identically to an absent one, so `pct_group1`/`pct_group2` measure genuine
//! expression rather than storage layout.

use std::cmp::Ordering;

use num_traits::AsPrimitive;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use single_utilities::traits::FloatOpsTS;

use crate::testing::utils::SparseMatrixRef;

/// Descriptive statistics for one gene in a two-group comparison.
#[derive(Debug, Clone, PartialEq)]
pub struct MarkerStats {
    /// Mean expression across group 1 cells, counting unexpressed cells as zero.
    pub mean_group1: f64,
    /// Mean expression across group 2 cells, counting unexpressed cells as zero.
    pub mean_group2: f64,
    /// Fraction of group 1 cells with non-zero expression (Seurat's `pct.1`).
    pub pct_group1: f64,
    /// Fraction of group 2 cells with non-zero expression (Seurat's `pct.2`).
    pub pct_group2: f64,
    /// `log2((mean_group1 + pseudocount) / (mean_group2 + pseudocount))`.
    pub log2_fold_change: f64,
    /// Area under the ROC curve for separating group 1 from group 2 on this gene.
    ///
    /// `0.5` means no discrimination, `1.0` means group 1 is uniformly higher, and
    /// `0.0` means group 2 is uniformly higher. Equivalent to the Mann-Whitney U
    /// statistic normalised by `n1 * n2`, ties counted as half.
    pub auroc: f64,
}

impl MarkerStats {
    /// Difference in detection rate between the groups (`pct.1 - pct.2`).
    pub fn delta_pct(&self) -> f64 {
        self.pct_group1 - self.pct_group2
    }

    /// How far the AUROC sits from chance, in `[0, 0.5]`.
    ///
    /// Useful for ranking markers irrespective of direction.
    pub fn auroc_power(&self) -> f64 {
        (self.auroc - 0.5).abs()
    }
}

/// Compute [`MarkerStats`] for every gene in one parallel pass.
///
/// `pseudocount` is added to both means before the log ratio, keeping the result
/// finite when a group has no expression. `1.0` matches the common convention for
/// log-normalised data.
pub fn marker_statistics<T, N, I>(
    matrix: SparseMatrixRef<T, N, I>,
    group1_indices: &[usize],
    group2_indices: &[usize],
    pseudocount: f64,
) -> anyhow::Result<Vec<MarkerStats>>
where
    T: FloatOpsTS,
    N: AsPrimitive<usize> + Send + Sync,
    I: AsPrimitive<usize> + Send + Sync,
{
    if group1_indices.is_empty() || group2_indices.is_empty() {
        return Err(anyhow::anyhow!(
            "Single-Statistics | Group indices cannot be empty. Error code: SS-MK-001"
        ));
    }

    let n1 = group1_indices.len();
    let n2 = group2_indices.len();

    let mut cell_groups = vec![0u8; matrix.n_cols];
    for &idx in group1_indices {
        if idx < cell_groups.len() {
            cell_groups[idx] = 1;
        }
    }
    for &idx in group2_indices {
        if idx < cell_groups.len() {
            cell_groups[idx] = 2;
        }
    }

    let results = (0..matrix.n_rows)
        .into_par_iter()
        .map(|row| {
            let (cols, vals) = matrix.get_major(row);

            let mut sum1 = 0.0f64;
            let mut sum2 = 0.0f64;
            // (value, group) for the non-zeros, reused for the rank sum below.
            let mut nonzero: Vec<(f64, u8)> = Vec::with_capacity(cols.len());

            for (col_idx, &value) in cols.iter().zip(vals.iter()) {
                let val = value.to_f64().unwrap_or(0.0);
                if val == 0.0 {
                    continue;
                }
                match cell_groups[col_idx.as_()] {
                    1 => {
                        sum1 += val;
                        nonzero.push((val, 1));
                    }
                    2 => {
                        sum2 += val;
                        nonzero.push((val, 2));
                    }
                    _ => {}
                }
            }

            let nz1 = nonzero.iter().filter(|(_, g)| *g == 1).count();
            let nz2 = nonzero.len() - nz1;

            let n1_f = n1 as f64;
            let n2_f = n2 as f64;
            let mean1 = sum1 / n1_f;
            let mean2 = sum2 / n2_f;

            MarkerStats {
                mean_group1: mean1,
                mean_group2: mean2,
                pct_group1: nz1 as f64 / n1_f,
                pct_group2: nz2 as f64 / n2_f,
                log2_fold_change: ((mean1 + pseudocount) / (mean2 + pseudocount)).log2(),
                auroc: auroc_from_parts(&mut nonzero, n1 - nz1, n2 - nz2, n1_f, n2_f),
            }
        })
        .collect();

    Ok(results)
}

/// AUROC via the rank-sum identity `AUC = U1 / (n1 * n2)`, with the shared zero
/// block ranked in bulk rather than materialised.
fn auroc_from_parts(
    nonzero: &mut [(f64, u8)],
    zeros1: usize,
    zeros2: usize,
    n1_f: f64,
    n2_f: f64,
) -> f64 {
    if n1_f == 0.0 || n2_f == 0.0 {
        return f64::NAN;
    }

    nonzero.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));

    let n_zeros = zeros1 + zeros2;
    let mut rank_sum_1 = 0.0f64;

    // The zeros are the smallest values and all tie with one another.
    let mut current_rank = 1.0f64;
    if n_zeros > 0 {
        let n_zeros_f = n_zeros as f64;
        rank_sum_1 += zeros1 as f64 * ((n_zeros_f + 1.0) / 2.0);
        current_rank += n_zeros_f;
    }

    let mut i = 0;
    while i < nonzero.len() {
        let val = nonzero[i].0;
        let start = i;
        while i < nonzero.len() && nonzero[i].0 == val {
            i += 1;
        }
        let count = (i - start) as f64;
        let avg_rank = current_rank + (count - 1.0) / 2.0;
        for &(_, group) in &nonzero[start..i] {
            if group == 1 {
                rank_sum_1 += avg_rank;
            }
        }
        current_rank += count;
    }

    let u1 = rank_sum_1 - n1_f * (n1_f + 1.0) / 2.0;
    u1 / (n1_f * n2_f)
}

/// Filter gene indices down to plausible marker candidates before testing.
///
/// Mirrors the pre-filtering step in common single-cell pipelines: a gene is kept
/// only if it is detected in at least `min_pct` of the cells in *either* group and
/// its absolute log2 fold change reaches `min_log2fc`. Running the statistical test
/// on the survivors alone both saves time and reduces the multiple-testing burden.
pub fn filter_marker_candidates(
    stats: &[MarkerStats],
    min_pct: f64,
    min_log2fc: f64,
) -> Vec<usize> {
    stats
        .iter()
        .enumerate()
        .filter_map(|(idx, s)| {
            let detected = s.pct_group1.max(s.pct_group2) >= min_pct;
            let changed = s.log2_fold_change.abs() >= min_log2fc;
            if detected && changed { Some(idx) } else { None }
        })
        .collect()
}
