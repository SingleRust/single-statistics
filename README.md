# single-statistics

[![Crates.io](https://img.shields.io/crates/v/single-statistics.svg)](https://crates.io/crates/single-statistics)
[![Documentation](https://docs.rs/single-statistics/badge.svg)](https://docs.rs/single-statistics)
[![License: BSD 3-Clause](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](LICENSE.md)

A specialized Rust library for statistical analysis of single-cell data, part of the single-rust ecosystem.

## Overview

`single-statistics` provides robust statistical methods for biological analysis of single-cell data, focusing on differential expression analysis, marker gene identification, and related statistical tests. This crate builds on the foundations provided by `single-algebra` while implementing biologically-relevant statistical approaches optimized for sparse single-cell data.

## Features

- **Differential Expression Analysis**
    - Parametric: Student's and Welch's t-test
    - Rank-based: Mann-Whitney U (Wilcoxon rank-sum), Wilcoxon signed-rank (paired),
      Kruskal-Wallis (two or more groups)
    - Count-based: Fisher's exact test, negative binomial
    - Effect sizes: Cohen's d, Hedges' g, log2 fold change
    - Parallel throughout

- **Marker Genes**
    - Group means, detection rates (`pct.1`/`pct.2`), log2FC and AUROC in one pass
    - Candidate pre-filtering by detection rate and fold change

- **Enrichment**
    - GSEA with permutation null and leading-edge genes
    - ORA (hypergeometric over-representation)
    - AUCell per-cell pathway activity

- **Spatial**
    - Moran's I and Geary's C against a spatial neighbour graph

- **Multiple Testing Correction**
    - Bonferroni correction
    - Benjamini-Hochberg (FDR)
    - Benjamini-Yekutieli
    - Holm-Bonferroni
    - Storey's q-value

- **Statistical Framework**
    - Generic interfaces for statistical tests
    - Container-agnostic sparse support: works on `sprs` matrices in either storage
      order, or on raw CSR/CSC slices passed across an FFI boundary such as PyO3
    - Type-safe operations via traits

## Getting Started

Add the crate to your Cargo.toml:

```toml
[dependencies]
single-statistics = "1.0"
```

## Example Usage

```rust
use sprs::CsMat;
use single_statistics::testing::{MatrixStatTests, TestMethod, TTestType};

fn main() -> anyhow::Result<()> {
    // Expression matrix, genes x cells
    let expression_matrix: CsMat<f64> = // ...

    // Define groups (e.g., cell types, conditions)
    let group_ids = vec![0, 0, 0, 1, 1, 1];

    // Run differential expression analysis
    let results = expression_matrix.differential_expression(
        &group_ids,
        TestMethod::TTest(TTestType::Welch)
    )?;

    // Get significantly differentially expressed genes
    let significant_genes = results.significant_indices(0.05);
    println!("Found {} significant genes", significant_genes.len());

    // Access statistics, p-values, and effect sizes
    if let Some(effect_sizes) = &results.effect_sizes {
        for (i, &gene_idx) in significant_genes.iter().enumerate() {
            println!(
                "Gene {}: statistic = {}, p-value = {}, adjusted p-value = {}, effect size = {}",
                gene_idx,
                results.statistics[gene_idx],
                results.p_values[gene_idx],
                results.adjusted_p_values.as_ref().unwrap()[gene_idx],
                effect_sizes[i]
            );
        }
    }

    Ok(())
}
```

### Marker genes

```rust
use single_statistics::testing::markers::{marker_statistics, filter_marker_candidates};
use single_statistics::testing::utils::SprsView;

let view = SprsView::new(&expression_matrix);
let stats = marker_statistics(view.as_matrix_ref(), &group1, &group2, 1.0)?;

// Keep genes detected in >=10% of either group with |log2FC| >= 0.25
let candidates = filter_marker_candidates(&stats, 0.10, 0.25);
```

### Matrix orientation

Sparse views are *major-oriented*: the major axis is features (genes), the minor
axis is observations (cells). A CSR `genes x cells` matrix and a CSC `cells x genes`
matrix both work directly. Outer-sliced `sprs` views are handled too — their `indptr`
is rebased automatically, and unsliced matrices stay zero-copy.

## Integration with the single-rust Ecosystem

`single-statistics` is designed to work seamlessly with other components of the single-rust ecosystem:

- **single-algebra**: Core algebraic operations for single-cell data
- **single-clustering**: Algorithms for clustering cells
- **single-utilities**: Common utilities for the ecosystem

## Scope

This crate focuses specifically on statistics related to differential expression and marker gene identification. It implements robust, efficient algorithms optimized for sparse data, providing statistical foundations for higher-level analyses in the single-cell domain.

Features in scope:
- Statistical tests relevant to single-cell RNA-seq and spatial transcriptomics
- Implementations of various hypothesis testing methods
- Multiple testing correction
- Effect size and marker gene statistics
- Gene set enrichment (GSEA, ORA, AUCell)

Features out of scope (available in other crates):
- General matrix statistics (in `single-algebra`)
- Basic QC metrics computation (in `single-algebra`)
- Plotting/visualization
- Clustering algorithms (in `single-clustering`)
- Batch correction

## Feature flags

| Flag | Default | Enables |
| --- | --- | --- |
| `spatial` | yes | Moran's I, Geary's C |
| `enrichment` | no | GSEA, ORA, AUCell |

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the BSD 3-Clause License - see the [LICENSE.md](LICENSE.md) file for details.