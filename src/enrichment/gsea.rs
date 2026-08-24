//! GSEA. Rank genes by any metric, walk the list, score each pathway.

use rayon::prelude::*;
use single_utilities::types::PathwayNetwork;

use crate::testing::correction::benjamini_hochberg_correction;
use crate::testing::utils::Rng;

#[derive(Debug, Clone)]
pub struct GseaResult {
    pub pathway_index: usize,
    pub size: usize,
    pub es: f64,
    pub nes: f64,
    pub p_value: f64,
    pub adjusted_p_value: f64,
    /// Genes driving the score, in rank order.
    pub leading_edge: Vec<usize>,
}

/// ES from sorted hit positions. Between hits the score only slides down by a
/// constant, so the extremes sit either side of a hit — O(k), not O(n).
///
/// Returns (ES, index in the ranked list where it peaks).
fn es_at(pos: &[usize], w: &[f64], n: usize) -> (f64, usize) {
    let k = pos.len();
    if k == 0 || k == n {
        return (0.0, 0);
    }
    let n_r: f64 = pos.iter().map(|&p| w[p]).sum();
    if n_r <= 0.0 {
        return (0.0, 0);
    }

    let miss = 1.0 / (n - k) as f64;
    let (mut sum, mut best, mut peak) = (0.0f64, 0.0f64, 0);

    for (j, &p) in pos.iter().enumerate() {
        let dip = (p - j) as f64 * miss; // misses seen before this hit
        let before = sum / n_r - dip;
        if before.abs() > best.abs() {
            best = before;
            peak = p.saturating_sub(1);
        }
        sum += w[p];
        let after = sum / n_r - dip;
        if after.abs() > best.abs() {
            best = after;
            peak = p;
        }
    }
    (best, peak)
}

/// Null from random same-size sets. Samples positions directly, so no gene lookup.
fn null_es(w: &[f64], n: usize, k: usize, n_perm: usize, seed: u64) -> Vec<f64> {
    let mut rng = Rng::new(seed);
    let mut idx: Vec<usize> = (0..n).collect();
    let mut pos = vec![0usize; k];

    (0..n_perm)
        .map(|_| {
            // partial Fisher-Yates: idx[..k] becomes a uniform random k-subset
            for i in 0..k {
                let j = i + rng.below(n - i);
                idx.swap(i, j);
            }
            pos.copy_from_slice(&idx[..k]);
            pos.sort_unstable();
            es_at(&pos, w, n).0
        })
        .collect()
}

/// Mean of null scores sharing `es`'s sign; used to normalise ES into NES.
fn mean_same_sign(nulls: &[f64], es: f64) -> f64 {
    let (mut sum, mut count) = (0.0, 0usize);
    for v in nulls {
        if (*v >= 0.0) == (es >= 0.0) {
            sum += v.abs();
            count += 1;
        }
    }
    if count == 0 { f64::NAN } else { sum / count as f64 }
}

/// Run GSEA. `metric[g]` ranks gene `g` (t-stat, log2FC, signal-to-noise, ...).
/// `weight` is the classic exponent p: 1.0 weighted, 0.0 for a plain KS test.
pub fn gsea(
    metric: &[f64],
    pathways: &[Vec<usize>],
    n_perm: usize,
    weight: f64,
    seed: u64,
) -> anyhow::Result<Vec<GseaResult>> {
    if metric.is_empty() {
        return Err(anyhow::anyhow!(
            "Single-Statistics | Empty ranking metric. Error code: SS-GSEA-001"
        ));
    }
    if pathways.is_empty() {
        return Err(anyhow::anyhow!(
            "Single-Statistics | No pathways supplied. Error code: SS-GSEA-002"
        ));
    }

    let n = metric.len();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_unstable_by(|&a, &b| {
        metric[b]
            .partial_cmp(&metric[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let w: Vec<f64> = order.iter().map(|&g| metric[g].abs().powf(weight)).collect();
    let mut rank_of = vec![0usize; n];
    for (p, &g) in order.iter().enumerate() {
        rank_of[g] = p;
    }

    let mut results: Vec<GseaResult> = pathways
        .par_iter()
        .enumerate()
        .map(|(idx, genes)| {
            // sort+dedup handles duplicates and gives the order es_at wants
            let mut pos: Vec<usize> = genes.iter().filter(|&&g| g < n).map(|&g| rank_of[g]).collect();
            pos.sort_unstable();
            pos.dedup();
            let size = pos.len();

            let (es, peak) = es_at(&pos, &w, n);

            let leading_edge: Vec<usize> = if es >= 0.0 {
                pos.iter().take_while(|&&p| p <= peak).map(|&p| order[p]).collect()
            } else {
                pos.iter().skip_while(|&&p| p < peak).map(|&p| order[p]).collect()
            };

            let (nes, p_value) = if size == 0 || es == 0.0 || n_perm == 0 {
                (f64::NAN, 1.0)
            } else {
                let nulls = null_es(&w, n, size, n_perm, seed.wrapping_add(idx as u64));
                let extreme = nulls
                    .iter()
                    .filter(|v| if es >= 0.0 { **v >= es } else { **v <= es })
                    .count();
                // Opposite-sign nulls are never more extreme, so count against all
                // permutations. NES still normalises by the same-sign mean.
                let p = (extreme + 1) as f64 / (n_perm + 1) as f64;
                (es / mean_same_sign(&nulls, es), p.min(1.0))
            };

            GseaResult {
                pathway_index: idx,
                size,
                es,
                nes,
                p_value,
                adjusted_p_value: f64::NAN,
                leading_edge,
            }
        })
        .collect();

    let raw: Vec<f64> = results.iter().map(|r| r.p_value).collect();
    for (r, adj) in results.iter_mut().zip(benjamini_hochberg_correction(&raw)?) {
        r.adjusted_p_value = adj;
    }
    Ok(results)
}

/// Same, taking a [`PathwayNetwork`].
pub fn gsea_network(
    metric: &[f64],
    network: &PathwayNetwork,
    n_perm: usize,
    weight: f64,
    seed: u64,
) -> anyhow::Result<Vec<GseaResult>> {
    let sets: Vec<Vec<usize>> = (0..network.get_num_pathways())
        .map(|i| network.get_pathway_features(i).to_vec())
        .collect();
    gsea(metric, &sets, n_perm, weight, seed)
}
