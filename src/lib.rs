//! # single-statistics
//!
//! A specialized Rust library for statistical analysis of single-cell data, part of the single-rust ecosystem.
//!
//! This crate provides robust statistical methods for biological analysis of single-cell data, focusing on
//! differential expression analysis, marker gene identification, and related statistical tests. It is optimized
//! for sparse single-cell matrices and provides both parametric and non-parametric statistical tests.
//!
//! ## Core Features
//!
//! - **Differential Expression Analysis**: t-tests, Mann-Whitney U (Wilcoxon rank-sum),
//!   Wilcoxon signed-rank, Kruskal-Wallis, and Fisher's exact test
//! - **Multiple Testing Correction**: FDR, Bonferroni, and other correction methods
//! - **Effect Size Calculations**: Cohen's d and other effect size measures
//! - **Sparse Matrix Support**: container-agnostic via [`testing::utils::SparseMatrixRef`],
//!   a borrowed view over raw CSR/CSC slices. Implementations are written against that
//!   view, so any backend that can expose `indptr`/`indices`/`data` slices works —
//!   including arrays passed across an FFI boundary such as PyO3. [`sprs`] matrices are
//!   supported directly through [`testing::utils::SprsView`], in either storage order
//!   and including outer-sliced views.
//!
//! ## Quick Start
//!
//! Use the `MatrixStatTests` trait to perform differential expression analysis on sparse matrices.
//! The library supports various statistical tests including t-tests and Mann-Whitney U tests,
//! with automatic multiple testing correction.
//!
//! ## Module Organization
//!
//! - **[`testing`]**: Statistical tests, hypothesis testing, and multiple testing correction
//! - **[`enrichment`]**: Gene set enrichment analysis methods (GSEA, ORA, AUCell)

pub mod testing;
#[cfg(feature = "enrichment")]
pub mod enrichment;
#[cfg(feature = "spatial")]
pub mod spatial;