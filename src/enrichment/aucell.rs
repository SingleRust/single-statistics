use std::collections::HashMap;

use anyhow::anyhow;
use indicatif::ParallelProgressIterator;
use nalgebra_sparse::csc::CscCol;
use nalgebra_sparse::{CscMatrix, CsrMatrix, csr::CsrRow};
use ndarray::Array2;
use rayon::iter::{ParallelBridge, ParallelIterator};
use single_utilities::traits::FloatOpsTS;
use single_utilities::types::PathwayNetwork;
use crate::testing::utils::SparseMatrixRef;
use num_traits::AsPrimitive;

// ... (validate_n_up and au_cell_internal remain the same) ...

pub fn au_cell_csr<T: FloatOpsTS>(
    matrix: &CsrMatrix<T>,
    pathway_network: &PathwayNetwork,
    n_up_abs: Option<usize>,
    n_up_frac: Option<f32>,
    verbose: bool,
) -> anyhow::Result<Array2<f32>> {
    let smr = SparseMatrixRef {
        maj_ind: matrix.row_offsets(),
        min_ind: matrix.col_indices(),
        val: matrix.values(),
        n_rows: matrix.nrows(),
        n_cols: matrix.ncols(),
    };
    au_cell_sparse(smr, pathway_network, n_up_abs, n_up_frac, verbose)
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

    let res: anyhow::Result<Vec<(usize, Vec<f32>)>> = match verbose {
        true => (0..n_obs)
            .into_par_iter()
            .progress_count(n_obs as u64)
            .map(|i| {
                let (cols, vals) = matrix.get_major(i);
                let mut all_values: Vec<(usize, f32)> = cols
                    .iter()
                    .zip(vals.iter())
                    .map(|(&idx, val)| (idx.as_(), val.to_f32().unwrap()))
                    .collect();
                all_values.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                let re = au_cell_internal(all_values, pathway_network, n_up, n_src)?;
                Ok((i, re))
            })
            .collect(),
        false => (0..n_obs)
            .into_par_iter()
            .map(|i| {
                let (cols, vals) = matrix.get_major(i);
                let mut all_values: Vec<(usize, f32)> = cols
                    .iter()
                    .zip(vals.iter())
                    .map(|(&idx, val)| (idx.as_(), val.to_f32().unwrap()))
                    .collect();
                all_values.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                let re = au_cell_internal(all_values, pathway_network, n_up, n_src)?;
                Ok((i, re))
            })
            .collect(),
    };

    let mut res = res?;
    res.sort_unstable_by(|a, b| a.0.cmp(&b.0));

    let res_vec: Vec<f32> = res.into_iter().flat_map(|(_, v)| v).collect();
    let array = Array2::from_shape_vec((n_obs, n_src), res_vec)?;
    Ok(array)
}

pub fn au_cell_csc<T: FloatOpsTS>(
    matrix: &CscMatrix<T>,
    pathway_network: &PathwayNetwork,
    n_up_abs: Option<usize>,
    n_up_frac: Option<f32>,
    verbose: bool,
) -> anyhow::Result<Array2<f32>> {
    let smr = SparseMatrixRef {
        maj_ind: matrix.col_offsets(),
        min_ind: matrix.row_indices(),
        val: matrix.values(),
        n_rows: matrix.ncols(),
        n_cols: matrix.nrows(),
    };
    au_cell_sparse(smr, pathway_network, n_up_abs, n_up_frac, verbose)
}
