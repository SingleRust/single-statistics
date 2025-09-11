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
//! - **Differential Expression Analysis**: T-tests, Mann-Whitney U tests, and other statistical methods
//! - **Multiple Testing Correction**: FDR, Bonferroni, and other correction methods
//! - **Effect Size Calculations**: Cohen's d and other effect size measures
//! - **Sparse Matrix Support**: Optimized for `CsrMatrix` from nalgebra-sparse
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
pub mod enrichment;