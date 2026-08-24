//! Gene set enrichment analysis methods for single-cell data.
//!
//! This module provides various approaches to gene set enrichment analysis, allowing you to
//! determine whether predefined sets of genes show statistically significant enrichment in
//! your single-cell data.
//!
//! ## Available Methods
//!
//! - **GSEA** (`gsea`): Gene Set Enrichment Analysis using ranking-based approaches
//! - **AUCell** (`aucell`): Area Under the Curve method for gene set activity scoring
//! - **ORA** (`ora`): Over-Representation Analysis using hypergeometric testing
//!
//! ## Quick Example
//!
//! ```rust
//! use single_statistics::enrichment::{gsea, over_representation_analysis};
//!
//! // Rank genes by any DE metric, then score pathways.
//! let metric: Vec<f64> = (0..100).map(|i| (100 - i) as f64).collect();
//! let pathways = vec![(0..10).collect::<Vec<_>>()];
//! let hits = gsea(&metric, &pathways, 100, 1.0, 42)?;
//!
//! // Or, from a thresholded gene list:
//! let universe: Vec<usize> = (0..100).collect();
//! let ora = over_representation_analysis(&(0..10).collect::<Vec<_>>(), &pathways, &universe)?;
//! # Ok::<(), anyhow::Error>(())
//! ```

pub mod gsea;
mod aucell;
pub mod ora;

pub use aucell::{au_cell, au_cell_sparse};
pub use gsea::{gsea, gsea_network, GseaResult};
pub use ora::{over_representation_analysis, significant_pathways, OraResult};