# single-statistics

[![Crates.io](https://img.shields.io/crates/v/single-statistics.svg)](https://crates.io/crates/single-statistics)
[![Documentation](https://docs.rs/single-statistics/badge.svg)](https://docs.rs/single-statistics)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE.md)

A specialized Rust library for statistical analysis of single-cell data, part of the single-rust ecosystem.

## Overview

`single-statistics` provides robust statistical methods for biological analysis of single-cell data, focusing on differential expression analysis, marker gene identification, and related statistical tests. This crate builds on the foundations provided by `single-algebra` while implementing biologically-relevant statistical approaches optimized for sparse single-cell data.

## Features

- **Differential Expression Analysis**
    - Parametric tests (Student's t-test, Welch's t-test)
    - Non-parametric tests (Mann-Whitney U test)
    - Effect size calculations
    - Parallel implementation for performance

- **Multiple Testing Correction**
    - Bonferroni correction
    - Benjamini-Hochberg (FDR)
    - Benjamini-Yekutieli
    - Holm-Bonferroni
    - Storey's q-value

- **Statistical Framework**
    - Generic interfaces for statistical tests
    - Support for sparse matrix representations
    - Type-safe operations via traits

## Getting Started

Add the crate to your Cargo.toml:

```toml
[dependencies]
single-statistics = "0.1.0"
```

## Example Usage

```rust
use nalgebra_sparse::CsrMatrix;
use single_statistics::testing::{Alternative, MatrixStatTests, TestMethod, TTestType};

fn main() -> anyhow::Result<()> {
    // Create or load your expression matrix (genes x cells)
    let expression_matrix: CsrMatrix<f64> = // ...

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

## Integration with the single-rust Ecosystem

`single-statistics` is designed to work seamlessly with other components of the single-rust ecosystem:

- **single-algebra**: Core algebraic operations for single-cell data
- **single-clustering**: Algorithms for clustering cells
- **single-utilities**: Common utilities for the ecosystem

## Scope

This crate focuses specifically on statistics related to differential expression and marker gene identification. It implements robust, efficient algorithms optimized for sparse data, providing statistical foundations for higher-level analyses in the single-cell domain.

Features in scope:
- Statistical tests relevant to single-cell RNA-seq analysis
- Implementations of various hypothesis testing methods
- Multiple testing correction
- Effect size calculations

Features out of scope (available in other crates):
- General matrix statistics (in `single-algebra`)
- Basic QC metrics computation (in `single-algebra`)
- Plotting/visualization
- Clustering algorithms (in `single-clustering`)
- Batch correction

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the BSD 3-Clause License - see the [LICENSE.md](LICENSE.md) file for details.