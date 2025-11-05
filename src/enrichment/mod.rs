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
//! ```rust,no_run
//! // Gene set enrichment analysis workflow would go here
//! // (Implementation depends on the specific modules)
//! ```

mod gsea;
mod aucell;
mod ora;
pub(crate) mod utils;

pub use aucell::{au_cell_csc, au_cell_csr};