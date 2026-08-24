use std::collections::HashMap;

use anyhow::anyhow;
use indicatif::ParallelProgressIterator;
use ndarray::Array2;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use single_utilities::traits::FloatOpsTS;
use single_utilities::types::PathwayNetwork;
use crate::testing::utils::{SparseMatrixRef, SprsView};
use num_traits::AsPrimitive;

// Following the general implementation presented here, but adapted to this crate's
// container-agnostic sparse view and multithreading:
// https://github.com/scverse/decoupler/blob/main/src/decoupler/mt/_aucell.py

/// Resolve the ranking cutoff `n_up` from either an absolute count or a fraction
/// of the feature count, clamped to the range `[2, n_var]`.
fn validate_n_up(
    n_var: usize,
    n_up_abs: Option<usize>,
    n_up_frac: Option<f32>,
) -> anyhow::Result<usize> {
    match (n_up_abs, n_up_frac) {
        (None, None) => {
            let nup = (n_var as f32 * 0.05).ceil() as usize;
            Ok(nup.clamp(2, n_var.max(2)))
        }
        (None, Some(x)) => {
            let frac = (x * n_var as f32).ceil() as usize;
            Ok(frac.clamp(2, n_var.max(2)))
        }
        (Some(x), None) => Ok(x.clamp(2, n_var.max(2))),
        (Some(_), Some(_)) => Err(anyhow!(
            "Cannot define both, n_up_abs AND n_up_frac, only one of them can be defined."
        )),
    }
}

/// Compute the AUC enrichment score of every pathway for a single observation.
///
/// `all_values` must already be sorted by descending expression value.
fn au_cell_internal(
    all_values: Vec<(usize, f32)>,
    pathway_network: &PathwayNetwork,
    n_up: usize,
    n_src: usize,
) -> anyhow::Result<Vec<f32>> {
    let mut rank_map: HashMap<usize, usize> = HashMap::with_capacity(all_values.len());
    for (rank, (idx, _)) in all_values.iter().enumerate() {
        rank_map.insert(*idx, rank + 1);
    }

    // temporarily no paralellization here to prevent nesting...
    let mut v: Vec<(usize, f32)> = (0..n_src)
        .map(|j| {
            let functional_set = pathway_network.get_pathway_features(j);

            let x_th = 1..=functional_set.len();
            let x_th: Vec<usize> = x_th.filter(|&v| v < n_up).collect();

            let max_auc: f32 = x_th
                .iter()
                .enumerate()
                .map(|(i, &val)| {
                    let next = if i < x_th.len() - 1 {
                        x_th[i + 1] as f32
                    } else {
                        n_up as f32
                    };
                    (next - val as f32) * val as f32
                })
                .sum();

            let mut x: Vec<usize> = functional_set
                .iter()
                .filter_map(|&idx| rank_map.get(&idx).copied())
                .collect();

            x.sort_unstable();
            x.retain(|&rank| rank <= n_up);

            let y: Vec<f32> = (1..=x.len()).map(|i| i as f32).collect();

            let mut x_f32: Vec<f32> = x.iter().map(|&r| r as f32).collect();

            x_f32.push(n_up as f32);

            let auc: f32 = x_f32
                .windows(2)
                .zip(y.iter())
                .map(|(window, &y_val)| (window[1] - window[0]) * y_val)
                .sum();
            let enrich_v = if max_auc > 0.0 { auc / max_auc } else { 0.0 };
            (j, enrich_v)
        })
        .collect();

    v.sort_unstable_by_key(|entry| entry.0);
    let v: Vec<f32> = v.iter().map(|a| a.1).collect();

    Ok(v)
}

/// Run AUCell over an [`sprs`] matrix.
///
/// Works for either storage order: the scored axis is always the major one, so pass
/// a CSR `cells x genes` matrix or a CSC `genes x cells` matrix to score cells.
/// See [`crate::testing::utils::sparse_ref_from_sprs`] for the full contract.
pub fn au_cell<T, I, Iptr, IptrStorage, IndStorage, DataStorage>(
    matrix: &sprs::CsMatBase<T, I, IptrStorage, IndStorage, DataStorage, Iptr>,
    pathway_network: &PathwayNetwork,
    n_up_abs: Option<usize>,
    n_up_frac: Option<f32>,
    verbose: bool,
) -> anyhow::Result<Array2<f32>>
where
    T: FloatOpsTS,
    I: sprs::indexing::SpIndex + AsPrimitive<usize>,
    Iptr: sprs::indexing::SpIndex + AsPrimitive<usize>,
    IptrStorage: std::ops::Deref<Target = [Iptr]> + Send + Sync,
    IndStorage: std::ops::Deref<Target = [I]> + Send + Sync,
    DataStorage: std::ops::Deref<Target = [T]> + Send + Sync,
{
    au_cell_sparse(
        SprsView::new(matrix).as_matrix_ref(),
        pathway_network,
        n_up_abs,
        n_up_frac,
        verbose,
    )
}

pub fn au_cell_sparse<T, N, I>(
    matrix: SparseMatrixRef<T, N, I>,
    pathway_network: &PathwayNetwork,
    n_up_abs: Option<usize>,
    n_up_frac: Option<f32>,
    verbose: bool,
) -> anyhow::Result<Array2<f32>>
where
    T: FloatOpsTS,
    N: AsPrimitive<usize> + Send + Sync,
    I: AsPrimitive<usize> + Send + Sync,
{
    let (n_obs, n_vars) = (matrix.n_rows, matrix.n_cols);
    let n_src = pathway_network.get_num_pathways();
    let n_up = validate_n_up(n_vars, n_up_abs, n_up_frac)?;

    let score_row = |i: usize| -> anyhow::Result<(usize, Vec<f32>)> {
        let (cols, vals) = matrix.get_major(i);
        let mut all_values: Vec<(usize, f32)> = cols
            .iter()
            .zip(vals.iter())
            .map(|(&idx, val)| (idx.as_(), val.to_f32().unwrap()))
            .collect();
        all_values.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let re = au_cell_internal(all_values, pathway_network, n_up, n_src)?;
        Ok((i, re))
    };

    let res: anyhow::Result<Vec<(usize, Vec<f32>)>> = match verbose {
        true => (0..n_obs)
            .into_par_iter()
            .progress_count(n_obs as u64)
            .map(score_row)
            .collect(),
        false => (0..n_obs).into_par_iter().map(score_row).collect(),
    };

    let mut res = res?;
    res.sort_unstable_by_key(|(row, _)| *row);

    let res_vec: Vec<f32> = res.into_iter().flat_map(|(_, v)| v).collect();
    let array = Array2::from_shape_vec((n_obs, n_src), res_vec)?;
    Ok(array)
}

