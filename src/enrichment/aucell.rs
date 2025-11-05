use std::{collections::HashMap, hash::Hash};

use anyhow::anyhow;
use indicatif::ParallelProgressIterator;
use nalgebra_sparse::csc::CscCol;
use nalgebra_sparse::{CscMatrix, CsrMatrix, csr::CsrRow};
use ndarray::Array2;
use num_traits::Float;
use rayon::iter::{IntoParallelIterator, ParallelBridge, ParallelIterator};
use single_utilities::traits::FloatOpsTS;

use crate::enrichment::utils::getset;

// Following the general implementation presented here, But adapted to nalgebra_sparse and multithreading: https://github.com/scverse/decoupler/blob/main/src/decoupler/mt/_aucell.py
fn validate_net(
    source: Vec<String>,
    target: Vec<String>,
    weights: Option<Vec<f32>>,
    verbose: bool,
) -> anyhow::Result<HashMap<String, Vec<(String, f32)>>> {
    let len_source = source.len();
    let len_target = target.len();
    if (len_source != len_target) {
        return Err(anyhow!(
            "Source and target must have the same length in order to be used for network construction!"
        ));
    }

    let mut map: HashMap<String, Vec<(String, f32)>> = HashMap::new();
    let mut current_src: String = "".to_string();
    let mut current_target_weight: HashMap<String, f32> = HashMap::new();
    for (i, src) in source.iter().enumerate() {
        if current_src.is_empty() {
            // never set a value in there
            current_src = src.clone();
        }

        if current_src != *src {
            // incase this is a different node now
            if !current_target_weight.is_empty() {
                let data: Vec<(String, f32)> = current_target_weight
                    .iter()
                    .map(|(key, value)| (key.clone(), *value))
                    .collect();
                map.insert(current_src, data);
                // cleanup
                current_target_weight.clear();
                current_src = src.clone();
            }
        }

        let src_target = target[i].clone();
        let src_target_weight = match &weights {
            Some(we) => we[i],
            None => 1f32,
        };
        current_target_weight.insert(src_target, src_target_weight);
    }

    if !current_target_weight.is_empty() {
        let data: Vec<(String, f32)> = current_target_weight
            .iter()
            .map(|(key, value)| (key.clone(), *value))
            .collect();
        map.insert(current_src, data);
    }

    Ok(map)
}

fn validate_n_up(
    n_var: usize,
    n_up_abs: Option<usize>,
    n_up_frac: Option<f32>,
) -> anyhow::Result<usize> {
    match (n_up_abs, n_up_frac) {
        (None, None) => {
            let mut nup = (n_var as f32 * 0.05).ceil() as usize;
            nup = nup.max(n_var).min(2);
            Ok(nup)
        }
        (None, Some(x)) => {
            let frac = (x * n_var as f32).ceil() as usize;
            Ok(frac.max(n_var).min(2))
        }
        (Some(x), None) => Ok(x.max(n_var).min(2)),
        (Some(_), Some(_)) => Err(anyhow!(
            "Cannot define both, n_up_abs AND n_up_frac, only one of them can be defined."
        )),
    }
}

fn au_cell_internal(
    all_values: Vec<(usize, f32)>,
    cnct: &[usize],
    starts: &[usize],
    offsets: &[usize],
    n_up: usize,
    n_src: usize,
) -> anyhow::Result<Vec<f32>> {
    let mut rank_map: HashMap<usize, usize> = HashMap::new();
    for (rank, (idx, _)) in all_values.iter().enumerate() {
        rank_map.insert(*idx, rank + 1);
    }

    // temporarily no paralellization here to prevent nesting...
    let mut v: Vec<(usize, f32)> = (0..n_src)
        .map(|j| {
            // dont know if we should actually parallelize here!
            let functional_set = getset(cnct, starts, offsets, j);

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

    v.sort_unstable_by(|&a, b| a.0.cmp(&b.0));
    let v: Vec<f32> = v.iter().map(|a| a.1).collect();

    Ok(v)
}

fn au_cell_csr_row<T: FloatOpsTS>(
    lane: CsrRow<T>,
    cnct: &[usize],
    starts: &[usize],
    offsets: &[usize],
    n_up: usize,
    n_src: usize,
) -> anyhow::Result<Vec<f32>> {
    let mut all_values: Vec<(usize, f32)> = lane
        .col_indices()
        .iter()
        .zip(lane.values().iter())
        .map(|(&idx, val)| (idx, val.to_f32().unwrap()))
        .collect();

    all_values.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    au_cell_internal(all_values, cnct, starts, offsets, n_up, n_src)
}

fn au_cell_csc_row<T: FloatOpsTS>(
    lane: CscCol<T>,
    cnct: &[usize],
    starts: &[usize],
    offsets: &[usize],
    n_up: usize,
    n_src: usize,
) -> anyhow::Result<Vec<f32>> {
    let mut all_values: Vec<(usize, f32)> = lane
        .row_indices()
        .iter()
        .zip(lane.values().iter())
        .map(|(&idx, val)| (idx, val.to_f32().unwrap()))
        .collect();

    all_values.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    au_cell_internal(all_values, cnct, starts, offsets, n_up, n_src)
}

pub fn au_cell_csr<T: FloatOpsTS>(
    matrix: &CsrMatrix<T>,
    cnct: &[usize],
    starts: &[usize],
    offsets: &[usize],
    n_up_abs: Option<usize>,
    n_up_frac: Option<f32>,
    verbose: bool,
) -> anyhow::Result<Array2<f32>> {
    let (n_obs, n_vars) = (matrix.nrows(), matrix.ncols());
    let n_src = starts.len();
    let n_up = validate_n_up(n_vars, n_up_abs, n_up_frac)?;

    let res: anyhow::Result<Vec<(usize, Vec<f32>)>> = match verbose {
        true => matrix
            .row_iter()
            .enumerate()
            .par_bridge()
            .progress_count(n_obs as u64)
            .map(|(i, r)| {
                let re = au_cell_csr_row(r, cnct, starts, offsets, n_up, n_src)?;
                Ok((i, re))
            })
            .collect(),
        false => matrix
            .row_iter()
            .enumerate()
            .par_bridge()
            .map(|(i, r)| {
                let re = au_cell_csr_row(r, cnct, starts, offsets, n_up, n_src)?;
                Ok((i, re))
            })
            .collect(),
    };

    let mut res = res?;
    res.sort_unstable_by(|a, b| a.0.cmp(&b.0));

    let res: Vec<f32> = res.into_iter().flat_map(|(_, v)| v).collect();
    let array = Array2::from_shape_vec((n_obs, n_vars), res)?;
    Ok(array)
}

pub fn au_cell_csc<T: FloatOpsTS>(
    matrix: CscMatrix<T>,
    cnct: &[usize],
    starts: &[usize],
    offsets: &[usize],
    n_up_abs: Option<usize>,
    n_up_frac: Option<f32>,
    verbose: bool,
) -> anyhow::Result<Array2<f32>> {
    let (n_obs, n_vars) = (matrix.ncols(), matrix.nrows());
    let n_src = starts.len();
    let n_up = validate_n_up(n_vars, n_up_abs, n_up_frac)?;

    let res: anyhow::Result<Vec<(usize, Vec<f32>)>> = match verbose {
        true => matrix
            .col_iter()
            .enumerate()
            .par_bridge()
            .progress_count(n_obs as u64)
            .map(|(i, r)| {
                let re = au_cell_csc_row(r, cnct, starts, offsets, n_up, n_src)?;
                Ok((i, re))
            })
            .collect(),
        false => matrix
            .col_iter()
            .enumerate()
            .par_bridge()
            .map(|(i, r)| {
                let re = au_cell_csc_row(r, cnct, starts, offsets, n_up, n_src)?;
                Ok((i, re))
            })
            .collect(),
    };

    let mut res = res?;
    res.sort_unstable_by(|a, b| a.0.cmp(&b.0));

    let res: Vec<f32> = res.into_iter().flat_map(|(_, v)| v).collect();
    let array = Array2::from_shape_vec((n_obs, n_vars), res)?;
    Ok(array)
}
