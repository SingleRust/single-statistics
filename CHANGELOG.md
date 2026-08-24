# Changelog

## 1.0.0

First stable release. It also fixes a p-value bug that affects anything you ran
before, so read that part first.

### t-test p-values were about 4x too small

The large-df tail used a hand-rolled erfc approximation. It took the coefficients
from Abramowitz & Stegun 7.1.28 and put them in the formula from 7.1.26. Those are
two different approximations, so the answer was just wrong.

Anything with df > 100 and |t| >= 2.83 came out roughly 4x too small. With
single-cell data that's more or less every significant gene.

| \|t\| | old | correct |
| --- | --- | --- |
| 3.0 | 3.24e-4 | 1.35e-3 |
| 5.0 | 7.95e-8 | 2.87e-7 |

It uses statrs now. Ranking is unaffected, but the numbers move, so results from
older versions won't reproduce. There's a regression test on it.

### Other fixes

- Holm-Bonferroni had no running max, so output could come out non-monotone. The
  unit test asserted the wrong values as well.
- Fisher's exact counted stored zeros as expressed. CSR matrices carry explicit
  zeros after filtering, so this showed up on real data.
- Mann-Whitney dropped stored zeros. They counted as neither zero nor non-zero and
  fell out of the sample size, which broke the rank sums.
- `chi_square_test` divided by zero expected counts.
- `validate_n_up` had its clamp arguments the wrong way round, so `n_up` was always
  2. AUCell scores change. They were meaningless before.
- Holm, Storey and adaptive Storey checked `p < 0 || p > 1`, which is false for NaN.
  NaN went straight through into the sort.

### Breaking

- sprs replaces nalgebra-sparse. Works with `CsMat`, views and outer-sliced views,
  in either storage order.
- `au_cell_csr` and `au_cell_csc` are now one `au_cell`.
- `effect::*` take `&SparseMatrixRef`. `calculate_log2_fold_change` used to read its
  indices as rows, now reads them as cells like everything else does.
- `TestResult.metadata` is keyed by `&'static str` instead of `String`.
- New `spatial` feature, on by default.

### Added

- Wilcoxon signed-rank, Kruskal-Wallis, negative binomial, zero-inflated (hurdle).
- Kruskal-Wallis lifts the two-group limit on `differential_expression`.
- Marker stats: group means, pct.1/pct.2, log2FC and AUROC in one pass.
- GSEA and ORA. `gsea.rs` and `ora.rs` were empty files before.
- Moran's I and Geary's C, with permutation nulls.
- Confidence intervals and per-gene metadata on results. Both fields existed but
  nothing ever filled them.

### Performance

- t-test: 28.7ms to 0.8ms on 2000 genes x 4000 cells. The matrix traversal was
  single-threaded.
- GSEA: 632ms to 7.3ms on 20k genes, 200 pathways, 1000 permutations. It walked
  every gene on every permutation instead of just the set members.
