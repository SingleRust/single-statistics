// Following the general implementation presented here, But adapted to nalgebra_sparse and multithreading: https://github.com/scverse/decoupler/blob/main/src/decoupler/mt/_aucell.py

use std::mem::offset_of;

use nalgebra_sparse::CsrMatrix;
use ndarray::Array2;
use single_utilities::traits::{FloatOps, FloatOpsTS};

fn validate_n_up<T>(n_genes: usize, n_up: Option<T>) -> anyhow::Result<usize> 
where 
T: FloatOps {
    let n_up_value = match n_up {
        Some(val) => {
            let n_up_int = num_traits::Float::ceil(val).to_usize().unwrap();
            n_up_int
        },
        None => {
            let n_up_float = T::from(0.05).unwrap() * T::from(n_genes).unwrap();
            let n_up_ceil = num_traits::Float::ceil(n_up_float);
            let n_up_int = n_up_ceil.to_usize().unwrap();
            n_up_int.clamp(2, n_genes)
        },
    };

    if n_up_value <= 1 || n_up_value > n_genes {
        return Err(anyhow::anyhow!(
            "For n_genes={}, n_up={} must be between 1 and {}",
            n_genes, n_up_value, n_genes
        ));
    }
    Ok(n_up_value)
}

fn get_gene_set(
    connectivity: &[usize],
    starts: &[usize],
    offsets: &[usize],
    source_idx: usize
) -> Vec<usize> {
    let start = starts[source_idx];
    let offset = offsets[source_idx];
    connectivity[start..start + offset].to_vec()
}

fn rank_data<T>(values: &[T]) -> Vec<usize> 
where
    T: FloatOps 
    {
        let mut indexed_values: Vec<(usize, T)> = values
        .iter()
        .enumerate()
        .map(|(i, &v)| (i,v))
        .collect();

        indexed_values.sort_by(|a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut ranks = vec![0; values.len()];
        for (rank, (original_idx, _)) in indexed_values.iter().enumerate() {
            ranks[*original_idx] = rank + 1;
        }

        ranks
    }



fn compute_chunk_size(n_cells: usize, n_genes: usize) -> usize {
    let n_cores = rayon::current_num_threads();

    let base_chunk_size = if n_genes > 20000 {
        200
    } else if n_genes > 10000 {
        500
    } else {
        1000
    };

    let min_chunks = n_cores;
    let max_chunk_size = (n_cells + min_chunks - 1) / min_chunks;
    base_chunk_size.min(max_chunk_size).max(1)
}

pub(crate) fn aucell_compute<T>(
    matrix: &CsrMatrix<T>,
    connectivity: &[usize],
    starts: &[usize],
    offsets: &[usize],
    n_up: Option<T>
) -> anyhow::Result<Array2<T>>
where 
    T: FloatOpsTS {
        let n_cells = matrix.nrows();
        let n_genes = matrix.ncols();
        let n_sources = starts.len();

        if connectivity.is_empty() || starts.is_empty() || offsets.is_empty() {
            return Err(anyhow::anyhow!("Connectivity arrays cannot be empty!!"))
        }

        if starts.len() != offsets.len() {
            return Err(anyhow::anyhow!("Starts and offsets must have the same length!"))
        }

        todo!()
    }