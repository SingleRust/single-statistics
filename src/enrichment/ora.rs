//! Over-Representation Analysis (ORA).
//!
//! Given a query set of genes — typically the significant hits from a differential
//! expression run — ORA asks, for each pathway, whether the overlap with that pathway
//! is larger than chance would produce. The null is the hypergeometric distribution:
//! drawing `n` genes without replacement from a universe of `N`, of which `K` belong
//! to the pathway.
//!
//! Unlike GSEA, ORA needs only a gene *list*, not a ranking, which makes it the
//! natural follow-up to a thresholded DE result.
//!
//! # Choosing the universe
//!
//! The universe should be the set of genes that *could* have been detected — usually
//! the genes retained after filtering, not every gene in the annotation. An inflated
//! universe inflates significance.

use std::collections::HashSet;

use rayon::iter::{IntoParallelIterator, ParallelIterator};
use statrs::distribution::{Discrete, DiscreteCDF, Hypergeometric};

use crate::testing::correction::benjamini_hochberg_correction;

/// Enrichment result for a single pathway.
#[derive(Debug, Clone, PartialEq)]
pub struct OraResult {
    /// Index of the pathway in the supplied collection.
    pub pathway_index: usize,
    /// Genes shared between the query set and the pathway (`k`).
    pub overlap: usize,
    /// Pathway size within the universe (`K`).
    pub pathway_size: usize,
    /// Query set size within the universe (`n`).
    pub query_size: usize,
    /// Universe size (`N`).
    pub universe_size: usize,
    /// Ratio of observed overlap to its expectation under the null.
    ///
    /// Infinite when the expected overlap is zero but the observed one is not.
    pub fold_enrichment: f64,
    /// One-sided hypergeometric p-value for over-representation, `P(X >= k)`.
    pub p_value: f64,
    /// Benjamini-Hochberg adjusted p-value across all tested pathways.
    pub adjusted_p_value: f64,
}

/// Run over-representation analysis of `query` against each pathway.
///
/// Every input is a collection of gene identifiers as `usize` indices. Genes in
/// `query` or in a pathway that fall outside `universe` are ignored, so callers can
/// pass full annotations without pre-intersecting them.
///
/// Results come back in the order the pathways were supplied, each carrying a
/// BH-adjusted p-value computed across the whole collection.
///
/// # Errors
/// Returns an error if the universe or the pathway collection is empty.
pub fn over_representation_analysis(
    query: &[usize],
    pathways: &[Vec<usize>],
    universe: &[usize],
) -> anyhow::Result<Vec<OraResult>> {
    if universe.is_empty() {
        return Err(anyhow::anyhow!(
            "Single-Statistics | ORA universe cannot be empty. Error code: SS-ORA-001"
        ));
    }
    if pathways.is_empty() {
        return Err(anyhow::anyhow!(
            "Single-Statistics | ORA requires at least one pathway. Error code: SS-ORA-002"
        ));
    }

    let universe_set: HashSet<usize> = universe.iter().copied().collect();
    let n_universe = universe_set.len();

    // Restrict the query to the universe and de-duplicate it.
    let query_set: HashSet<usize> = query
        .iter()
        .copied()
        .filter(|g| universe_set.contains(g))
        .collect();
    let n_query = query_set.len();

    let mut results: Vec<OraResult> = (0..pathways.len())
        .into_par_iter()
        .map(|idx| {
            let pathway_set: HashSet<usize> = pathways[idx]
                .iter()
                .copied()
                .filter(|g| universe_set.contains(g))
                .collect();
            let k_pathway = pathway_set.len();
            let overlap = query_set.intersection(&pathway_set).count();

            let expected = (n_query as f64) * (k_pathway as f64) / (n_universe as f64);
            let fold_enrichment = if expected > 0.0 {
                overlap as f64 / expected
            } else if overlap > 0 {
                f64::INFINITY
            } else {
                0.0
            };

            // P(X >= overlap) under Hypergeometric(N, K, n).
            let p_value = if overlap == 0 || k_pathway == 0 || n_query == 0 {
                1.0
            } else {
                match Hypergeometric::new(n_universe as u64, k_pathway as u64, n_query as u64) {
                    Ok(dist) => {
                        let upper_tail = 1.0 - dist.cdf(overlap as u64 - 1);
                        // Recover precision in the far tail, where 1 - cdf cancels.
                        if upper_tail <= 0.0 {
                            let hi = k_pathway.min(n_query);
                            (overlap..=hi).map(|i| dist.pmf(i as u64)).sum::<f64>()
                        } else {
                            upper_tail
                        }
                        .clamp(0.0, 1.0)
                    }
                    Err(_) => 1.0,
                }
            };

            OraResult {
                pathway_index: idx,
                overlap,
                pathway_size: k_pathway,
                query_size: n_query,
                universe_size: n_universe,
                fold_enrichment,
                p_value,
                adjusted_p_value: f64::NAN, // filled in below
            }
        })
        .collect();

    let raw: Vec<f64> = results.iter().map(|r| r.p_value).collect();
    let adjusted = benjamini_hochberg_correction(&raw)?;
    for (r, adj) in results.iter_mut().zip(adjusted) {
        r.adjusted_p_value = adj;
    }

    Ok(results)
}

/// Convenience wrapper returning only pathways significant at `alpha`, sorted by
/// adjusted p-value (most significant first).
pub fn significant_pathways(
    query: &[usize],
    pathways: &[Vec<usize>],
    universe: &[usize],
    alpha: f64,
) -> anyhow::Result<Vec<OraResult>> {
    let mut results = over_representation_analysis(query, pathways, universe)?;
    results.retain(|r| r.adjusted_p_value < alpha);
    results.sort_by(|a, b| {
        a.adjusted_p_value
            .partial_cmp(&b.adjusted_p_value)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    Ok(results)
}
