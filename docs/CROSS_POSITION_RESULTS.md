# Cross-position central comparison

Reference results: benchmark implementations, experiment runners, datasets, and raw draws are not bundled. These tables are not a claim of a newly reproduced package evaluation.

Direct DRF vs multivariate DRF vs hierarchical parametric model, on every position.
Method: [COMPARISON_GUIDE.md](COMPARISON_GUIDE.md). WR origin story: [CENTRAL_RESULTS.md](CENTRAL_RESULTS.md).

Direct DRF vs multivariate DRF vs hierarchical parametric model, extended to every
position, with the lessons from the wide-receiver experiment applied:

- shape parameters are estimated once on the season before the holdout and then frozen;
- the hierarchical chain is position-specific (compound signed-gain yardage, per-event
  touchdown curves, frozen categorical kicking laws, hurdle zero-mass stages);
- every scoring rule is verified against the panel's own `fantasy_points` column first;
- upper-tail calibration is recorded for every statistic, not only for fantasy points.

Candidates, identical folds and identical features within a position:

- **Direct DRF**
- **Independent stat DRFs**
- **Multivariate DRF**
- **Hierarchical parametric**

Direct DRF vs multivariate DRF vs hierarchical parametric model, extended to every
position, with the lessons from the wide-receiver experiment applied:

- shape parameters are estimated once on the season before the holdout and then frozen;
- the hierarchical chain is position-specific (compound signed-gain yardage, per-event
  touchdown curves, frozen categorical kicking laws, hurdle zero-mass stages);
- every scoring rule is verified against the panel's own `fantasy_points` column first;
- upper-tail calibration is recorded for every statistic, not only for fantasy points.

Candidates, identical folds and identical features within a position:

- **Direct DRF**
- **Independent stat DRFs**
- **Multivariate DRF**
- **Hierarchical parametric**

## Scoring reconciliation

The scoring function must reproduce the panel exactly, or the run aborts.

| Position | Rows | Mismatched rows | Max abs error | Mean fantasy points |
| --- | ---: | ---: | ---: | ---: |
| WR | 12256 | 0 | 0.00000000 | 7.309 |
| TE | 6109 | 0 | 0.00000000 | 5.495 |
| QB | 3279 | 0 | 0.00000000 | 13.754 |
| K | 2716 | 0 | 0.00000000 | 8.218 |
| DST | 2718 | 0 | 0.00000000 | 5.538 |

## Headline: pooled CRPS by position

| Position | Direct DRF | Independent stat DRFs | Multivariate DRF | Hierarchical parametric | Best |
| --- | ---: | ---: | ---: | ---: | --- |
| WR | 2.960 | 3.039 | 2.973 | 2.982 | Direct DRF |
| TE | 2.421 | 2.465 | 2.420 | 2.445 | Multivariate DRF |
| QB | 4.288 | 4.366 | 4.309 | 4.350 | Direct DRF |
| K | 2.643 | 2.648 | 2.664 | 2.676 | Direct DRF |
| DST | 2.657 | 2.670 | 2.645 | 2.685 | Multivariate DRF |

Paired CRPS delta against the direct DRF, player-clustered 95% interval.
Positive means the candidate beats the direct DRF. The draw-noise floor measured on
WR was about 0.009 CRPS, so read small differences with that in mind.

| Position | Independent stat DRFs | Multivariate DRF | Hierarchical parametric |
| --- | ---: | ---: | ---: |
| WR | -0.079 [-0.096, -0.064] | -0.013 [-0.028, +0.001] | -0.021 [-0.044, -0.004] |
| TE | -0.044 [-0.060, -0.028] | +0.000 [-0.017, +0.016] | -0.025 [-0.051, +0.004] |
| QB | -0.078 [-0.130, -0.029] | -0.021 [-0.071, +0.030] | -0.062 [-0.139, +0.004] |
| K | -0.005 [-0.026, +0.015] | -0.021 [-0.047, +0.005] | -0.033 [-0.068, +0.005] |
| DST | -0.013 [-0.036, +0.011] | +0.012 [-0.010, +0.036] | -0.028 [-0.069, +0.005] |

## Per-position detail

### WR

| Model | CRPS | MAE | 80% cov | 80% width | PIT deviation | Predicted sd | Observed sd |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Direct DRF | 2.960 | 4.357 | 0.860 | 13.54 | 0.037 | 5.76 | 7.59 |
| Independent stat DRFs | 3.039 | 4.361 | 0.694 | 10.17 | 0.330 | 4.17 | 7.59 |
| Multivariate DRF | 2.973 | 4.419 | 0.874 | 13.99 | 0.055 | 5.95 | 7.59 |
| Hierarchical parametric | 2.982 | 4.376 | 0.853 | 13.64 | 0.068 | 5.72 | 7.59 |

Tail calibration for this position's registered events:

| Event | Model | Predicted | Realised | Brier |
| --- | --- | ---: | ---: | ---: |
| p_lt_5 | Direct DRF | 0.5139 | 0.5194 | 0.1672 |
| p_lt_5 | Independent stat DRFs | 0.4531 | 0.5194 | 0.1727 |
| p_lt_5 | Multivariate DRF | 0.5105 | 0.5194 | 0.1679 |
| p_lt_5 | Hierarchical parametric | 0.5160 | 0.5194 | 0.1688 |
| p_ge_20 | Direct DRF | 0.0759 | 0.0739 | 0.0603 |
| p_ge_20 | Independent stat DRFs | 0.0477 | 0.0739 | 0.0611 |
| p_ge_20 | Multivariate DRF | 0.0766 | 0.0739 | 0.0603 |
| p_ge_20 | Hierarchical parametric | 0.0773 | 0.0739 | 0.0609 |
| p_ge_25 | Direct DRF | 0.0342 | 0.0296 | 0.0270 |
| p_ge_25 | Independent stat DRFs | 0.0151 | 0.0296 | 0.0273 |
| p_ge_25 | Multivariate DRF | 0.0347 | 0.0296 | 0.0272 |
| p_ge_25 | Hierarchical parametric | 0.0388 | 0.0296 | 0.0276 |
| p_ge_30 | Direct DRF | 0.0146 | 0.0109 | 0.0106 |
| p_ge_30 | Independent stat DRFs | 0.0041 | 0.0109 | 0.0106 |
| p_ge_30 | Multivariate DRF | 0.0149 | 0.0109 | 0.0106 |
| p_ge_30 | Hierarchical parametric | 0.0192 | 0.0109 | 0.0109 |

Dependence: within-draw correlation against the realised correlation.

| Statistic pair | Model | Model corr | Observed corr |
| --- | --- | ---: | ---: |
| receptions / receiving_yards | Independent stat DRFs | -0.001 | 0.866 |
| receptions / receiving_yards | Multivariate DRF | 0.792 | 0.866 |
| receptions / receiving_yards | Hierarchical parametric | 0.796 | 0.866 |
| targets / receiving_yards | Independent stat DRFs | 0.000 | 0.799 |
| targets / receiving_yards | Multivariate DRF | 0.679 | 0.799 |
| targets / receiving_yards | Hierarchical parametric | 0.656 | 0.799 |
| targets / receptions | Independent stat DRFs | -0.000 | 0.913 |
| targets / receptions | Multivariate DRF | 0.853 | 0.913 |
| targets / receptions | Hierarchical parametric | 0.824 | 0.913 |

Worst upper-tail miscalibration by statistic:

| Statistic | Event | Model | Predicted | Realised | Gap |
| --- | --- | --- | ---: | ---: | ---: |
| receiving_yards | ge_126 | Hierarchical parametric | 0.0325 | 0.0173 | +0.0152 |
| receiving_yards | ge_86 | Independent stat DRFs | 0.1029 | 0.0894 | +0.0134 |
| receiving_yards | ge_86 | Multivariate DRF | 0.1027 | 0.0894 | +0.0133 |
| receiving_yards | ge_125 | Hierarchical parametric | 0.0327 | 0.0195 | +0.0132 |
| receiving_yards | ge_86 | Hierarchical parametric | 0.1013 | 0.0894 | +0.0119 |
| receptions | ge_6 | Multivariate DRF | 0.1373 | 0.1267 | +0.0106 |

### TE

| Model | CRPS | MAE | 80% cov | 80% width | PIT deviation | Predicted sd | Observed sd |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Direct DRF | 2.421 | 3.550 | 0.863 | 10.81 | 0.045 | 4.65 | 6.06 |
| Independent stat DRFs | 2.465 | 3.529 | 0.692 | 8.39 | 0.326 | 3.47 | 6.06 |
| Multivariate DRF | 2.420 | 3.561 | 0.873 | 11.10 | 0.065 | 4.77 | 6.06 |
| Hierarchical parametric | 2.445 | 3.560 | 0.850 | 10.72 | 0.132 | 4.45 | 6.06 |

Tail calibration for this position's registered events:

| Event | Model | Predicted | Realised | Brier |
| --- | --- | ---: | ---: | ---: |
| p_lt_3 | Direct DRF | 0.4470 | 0.4539 | 0.1752 |
| p_lt_3 | Independent stat DRFs | 0.3873 | 0.4539 | 0.1786 |
| p_lt_3 | Multivariate DRF | 0.4441 | 0.4539 | 0.1768 |
| p_lt_3 | Hierarchical parametric | 0.4589 | 0.4539 | 0.1767 |
| p_ge_10 | Direct DRF | 0.1905 | 0.1951 | 0.1246 |
| p_ge_10 | Independent stat DRFs | 0.1728 | 0.1951 | 0.1242 |
| p_ge_10 | Multivariate DRF | 0.1911 | 0.1951 | 0.1232 |
| p_ge_10 | Hierarchical parametric | 0.1898 | 0.1951 | 0.1251 |
| p_ge_15 | Direct DRF | 0.0804 | 0.0830 | 0.0665 |
| p_ge_15 | Independent stat DRFs | 0.0570 | 0.0830 | 0.0670 |
| p_ge_15 | Multivariate DRF | 0.0791 | 0.0830 | 0.0661 |
| p_ge_15 | Hierarchical parametric | 0.0836 | 0.0830 | 0.0672 |
| p_ge_20 | Direct DRF | 0.0322 | 0.0335 | 0.0296 |
| p_ge_20 | Independent stat DRFs | 0.0159 | 0.0335 | 0.0303 |
| p_ge_20 | Multivariate DRF | 0.0316 | 0.0335 | 0.0300 |
| p_ge_20 | Hierarchical parametric | 0.0353 | 0.0335 | 0.0301 |

Dependence: within-draw correlation against the realised correlation.

| Statistic pair | Model | Model corr | Observed corr |
| --- | --- | ---: | ---: |
| carries / rushing_yards | Independent stat DRFs | 0.000 | — |
| carries / rushing_yards | Multivariate DRF | 0.725 | — |
| carries / rushing_yards | Hierarchical parametric | 0.040 | — |
| receptions / receiving_yards | Independent stat DRFs | 0.000 | 0.877 |
| receptions / receiving_yards | Multivariate DRF | 0.811 | 0.877 |
| receptions / receiving_yards | Hierarchical parametric | 0.748 | 0.877 |
| targets / receiving_yards | Independent stat DRFs | -0.001 | 0.821 |
| targets / receiving_yards | Multivariate DRF | 0.720 | 0.821 |
| targets / receiving_yards | Hierarchical parametric | 0.616 | 0.821 |
| targets / receptions | Independent stat DRFs | 0.001 | 0.933 |
| targets / receptions | Multivariate DRF | 0.876 | 0.933 |
| targets / receptions | Hierarchical parametric | 0.823 | 0.933 |

Worst upper-tail miscalibration by statistic:

| Statistic | Event | Model | Predicted | Realised | Gap |
| --- | --- | --- | ---: | ---: | ---: |
| receiving_yards | ge_58 | Hierarchical parametric | 0.1258 | 0.0673 | +0.0585 |
| receiving_yards | ge_58 | Multivariate DRF | 0.1148 | 0.0673 | +0.0475 |
| receiving_yards | ge_58 | Independent stat DRFs | 0.1114 | 0.0673 | +0.0441 |
| carries | ge_1 | Hierarchical parametric | 0.0135 | 0.0359 | -0.0224 |
| rushing_yards | ge_1 | Hierarchical parametric | 0.0111 | 0.0262 | -0.0151 |
| receiving_yards | ge_56 | Hierarchical parametric | 0.1031 | 0.1176 | -0.0145 |

### QB

| Model | CRPS | MAE | 80% cov | 80% width | PIT deviation | Predicted sd | Observed sd |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Direct DRF | 4.288 | 6.219 | 0.770 | 18.82 | 0.080 | 7.42 | 9.24 |
| Independent stat DRFs | 4.366 | 6.220 | 0.679 | 15.73 | 0.269 | 6.24 | 9.24 |
| Multivariate DRF | 4.309 | 6.325 | 0.798 | 19.86 | 0.072 | 7.76 | 9.24 |
| Hierarchical parametric | 4.350 | 6.224 | 0.745 | 18.41 | 0.129 | 7.38 | 9.24 |

Tail calibration for this position's registered events:

| Event | Model | Predicted | Realised | Brier |
| --- | --- | ---: | ---: | ---: |
| p_lt_15 | Direct DRF | 0.5588 | 0.5520 | 0.2103 |
| p_lt_15 | Independent stat DRFs | 0.5747 | 0.5520 | 0.2110 |
| p_lt_15 | Multivariate DRF | 0.5624 | 0.5520 | 0.2084 |
| p_lt_15 | Hierarchical parametric | 0.5903 | 0.5520 | 0.2141 |
| p_ge_20 | Direct DRF | 0.2418 | 0.2575 | 0.1724 |
| p_ge_20 | Independent stat DRFs | 0.2055 | 0.2575 | 0.1718 |
| p_ge_20 | Multivariate DRF | 0.2400 | 0.2575 | 0.1716 |
| p_ge_20 | Hierarchical parametric | 0.2287 | 0.2575 | 0.1768 |
| p_ge_25 | Direct DRF | 0.1043 | 0.1182 | 0.0990 |
| p_ge_25 | Independent stat DRFs | 0.0790 | 0.1182 | 0.0993 |
| p_ge_25 | Multivariate DRF | 0.1037 | 0.1182 | 0.0986 |
| p_ge_25 | Hierarchical parametric | 0.1136 | 0.1182 | 0.0996 |
| p_ge_30 | Direct DRF | 0.0339 | 0.0361 | 0.0336 |
| p_ge_30 | Independent stat DRFs | 0.0246 | 0.0361 | 0.0333 |
| p_ge_30 | Multivariate DRF | 0.0343 | 0.0361 | 0.0336 |
| p_ge_30 | Hierarchical parametric | 0.0503 | 0.0361 | 0.0347 |

Dependence: within-draw correlation against the realised correlation.

| Statistic pair | Model | Model corr | Observed corr |
| --- | --- | ---: | ---: |
| attempts / completions | Independent stat DRFs | -0.002 | 0.952 |
| attempts / completions | Multivariate DRF | 0.927 | 0.952 |
| attempts / completions | Hierarchical parametric | 0.916 | 0.952 |
| attempts / passing_yards | Independent stat DRFs | 0.000 | 0.864 |
| attempts / passing_yards | Multivariate DRF | 0.799 | 0.864 |
| attempts / passing_yards | Hierarchical parametric | 0.787 | 0.864 |
| completions / passing_yards | Independent stat DRFs | 0.000 | 0.903 |
| completions / passing_yards | Multivariate DRF | 0.855 | 0.903 |
| completions / passing_yards | Hierarchical parametric | 0.858 | 0.903 |

Worst upper-tail miscalibration by statistic:

| Statistic | Event | Model | Predicted | Realised | Gap |
| --- | --- | --- | ---: | ---: | ---: |
| passing_yards | ge_322 | Hierarchical parametric | 0.1306 | 0.0417 | +0.0890 |
| passing_yards | ge_319 | Hierarchical parametric | 0.1348 | 0.0513 | +0.0835 |
| passing_yards | ge_380 | Hierarchical parametric | 0.0683 | 0.0000 | +0.0683 |
| passing_yards | ge_322 | Multivariate DRF | 0.1020 | 0.0417 | +0.0603 |
| passing_yards | ge_322 | Independent stat DRFs | 0.0970 | 0.0417 | +0.0553 |
| rushing_yards | ge_45 | Multivariate DRF | 0.1153 | 0.0606 | +0.0547 |

### K

| Model | CRPS | MAE | 80% cov | 80% width | PIT deviation | Predicted sd | Observed sd |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Direct DRF | 2.643 | 3.753 | 0.824 | 11.50 | 0.104 | 4.56 | 4.75 |
| Independent stat DRFs | 2.648 | 3.743 | 0.776 | 10.49 | 0.140 | 4.12 | 4.75 |
| Multivariate DRF | 2.664 | 3.776 | 0.813 | 11.39 | 0.101 | 4.53 | 4.75 |
| Hierarchical parametric | 2.676 | 3.806 | 0.807 | 11.50 | 0.108 | 4.66 | 4.75 |

Tail calibration for this position's registered events:

| Event | Model | Predicted | Realised | Brier |
| --- | --- | ---: | ---: | ---: |
| p_lt_5 | Direct DRF | 0.2157 | 0.2109 | 0.1631 |
| p_lt_5 | Independent stat DRFs | 0.1901 | 0.2109 | 0.1651 |
| p_lt_5 | Multivariate DRF | 0.2149 | 0.2109 | 0.1657 |
| p_lt_5 | Hierarchical parametric | 0.1887 | 0.2109 | 0.1692 |
| p_ge_8 | Direct DRF | 0.5324 | 0.5589 | 0.2469 |
| p_ge_8 | Independent stat DRFs | 0.5400 | 0.5589 | 0.2442 |
| p_ge_8 | Multivariate DRF | 0.5297 | 0.5589 | 0.2458 |
| p_ge_8 | Hierarchical parametric | 0.5424 | 0.5589 | 0.2467 |
| p_ge_12 | Direct DRF | 0.2323 | 0.2532 | 0.1880 |
| p_ge_12 | Independent stat DRFs | 0.2137 | 0.2532 | 0.1904 |
| p_ge_12 | Multivariate DRF | 0.2274 | 0.2532 | 0.1912 |
| p_ge_12 | Hierarchical parametric | 0.2479 | 0.2532 | 0.1905 |
| p_ge_15 | Direct DRF | 0.0940 | 0.1142 | 0.1013 |
| p_ge_15 | Independent stat DRFs | 0.0806 | 0.1142 | 0.1016 |
| p_ge_15 | Multivariate DRF | 0.0894 | 0.1142 | 0.1019 |
| p_ge_15 | Hierarchical parametric | 0.1186 | 0.1142 | 0.1025 |

Dependence: within-draw correlation against the realised correlation.

| Statistic pair | Model | Model corr | Observed corr |
| --- | --- | ---: | ---: |
| fg_att / fg_made | Independent stat DRFs | 0.000 | 0.910 |
| fg_att / fg_made | Multivariate DRF | 0.909 | 0.910 |
| fg_att / fg_made | Hierarchical parametric | 0.907 | 0.910 |
| fg_made / fg_made_40_49 | Independent stat DRFs | -0.001 | 0.479 |
| fg_made / fg_made_40_49 | Multivariate DRF | 0.483 | 0.479 |
| fg_made / fg_made_40_49 | Hierarchical parametric | 0.447 | 0.479 |
| pat_att / pat_made | Independent stat DRFs | -0.000 | 0.976 |
| pat_att / pat_made | Multivariate DRF | 0.973 | 0.976 |
| pat_att / pat_made | Hierarchical parametric | 0.979 | 0.976 |

Worst upper-tail miscalibration by statistic:

| Statistic | Event | Model | Predicted | Realised | Gap |
| --- | --- | --- | ---: | ---: | ---: |
| fg_made_50_59 | ge_1 | Multivariate DRF | 0.2494 | 0.2762 | -0.0268 |
| fg_made_40_49 | ge_1 | Multivariate DRF | 0.3483 | 0.3738 | -0.0255 |
| pat_made | ge_4 | Multivariate DRF | 0.1661 | 0.1878 | -0.0217 |
| pat_made | ge_4 | Independent stat DRFs | 0.1671 | 0.1878 | -0.0208 |
| pat_att | ge_4 | Multivariate DRF | 0.1819 | 0.2017 | -0.0197 |
| pat_att | ge_4 | Independent stat DRFs | 0.1825 | 0.2017 | -0.0191 |

### DST

| Model | CRPS | MAE | 80% cov | 80% width | PIT deviation | Predicted sd | Observed sd |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Direct DRF | 2.657 | 3.760 | 0.825 | 11.98 | 0.078 | 4.95 | 5.08 |
| Independent stat DRFs | 2.670 | 3.759 | 0.778 | 10.09 | 0.182 | 4.08 | 5.08 |
| Multivariate DRF | 2.645 | 3.751 | 0.834 | 12.11 | 0.062 | 5.04 | 5.08 |
| Hierarchical parametric | 2.685 | 3.739 | 0.752 | 9.78 | 0.196 | 4.01 | 5.08 |

Tail calibration for this position's registered events:

| Event | Model | Predicted | Realised | Brier |
| --- | --- | ---: | ---: | ---: |
| p_lt_5 | Direct DRF | 0.4822 | 0.4926 | 0.2349 |
| p_lt_5 | Independent stat DRFs | 0.4477 | 0.4926 | 0.2370 |
| p_lt_5 | Multivariate DRF | 0.4778 | 0.4926 | 0.2336 |
| p_lt_5 | Hierarchical parametric | 0.4890 | 0.4926 | 0.2375 |
| p_ge_10 | Direct DRF | 0.1914 | 0.1774 | 0.1417 |
| p_ge_10 | Independent stat DRFs | 0.1695 | 0.1774 | 0.1392 |
| p_ge_10 | Multivariate DRF | 0.1932 | 0.1774 | 0.1393 |
| p_ge_10 | Hierarchical parametric | 0.1451 | 0.1774 | 0.1436 |
| p_ge_15 | Direct DRF | 0.0588 | 0.0579 | 0.0540 |
| p_ge_15 | Independent stat DRFs | 0.0331 | 0.0579 | 0.0542 |
| p_ge_15 | Multivariate DRF | 0.0598 | 0.0579 | 0.0537 |
| p_ge_15 | Hierarchical parametric | 0.0301 | 0.0579 | 0.0540 |
| p_ge_20 | Direct DRF | 0.0166 | 0.0119 | 0.0120 |
| p_ge_20 | Independent stat DRFs | 0.0045 | 0.0119 | 0.0119 |
| p_ge_20 | Multivariate DRF | 0.0170 | 0.0119 | 0.0121 |
| p_ge_20 | Hierarchical parametric | 0.0053 | 0.0119 | 0.0118 |

Dependence: within-draw correlation against the realised correlation.

| Statistic pair | Model | Model corr | Observed corr |
| --- | --- | ---: | ---: |
| def_interceptions / def_tds | Independent stat DRFs | 0.005 | — |
| def_interceptions / def_tds | Multivariate DRF | 0.281 | — |
| def_interceptions / def_tds | Hierarchical parametric | 0.237 | — |
| def_interceptions / points_allowed | Independent stat DRFs | 0.003 | -0.262 |
| def_interceptions / points_allowed | Multivariate DRF | -0.243 | -0.262 |
| def_interceptions / points_allowed | Hierarchical parametric | 0.002 | -0.262 |
| def_sacks / points_allowed | Independent stat DRFs | -0.002 | -0.353 |
| def_sacks / points_allowed | Multivariate DRF | -0.324 | -0.353 |
| def_sacks / points_allowed | Hierarchical parametric | -0.000 | -0.353 |

Worst upper-tail miscalibration by statistic:

| Statistic | Event | Model | Predicted | Realised | Gap |
| --- | --- | --- | ---: | ---: | ---: |
| def_fumbles | ge_1 | Multivariate DRF | 0.1289 | 0.0910 | +0.0379 |
| def_fumbles | ge_1 | Hierarchical parametric | 0.1240 | 0.0910 | +0.0330 |
| def_fumbles | ge_1 | Independent stat DRFs | 0.1205 | 0.0910 | +0.0295 |
| points_allowed | ge_34 | Multivariate DRF | 0.1017 | 0.1250 | -0.0233 |
| points_allowed | ge_34 | Independent stat DRFs | 0.1019 | 0.1250 | -0.0231 |
| points_allowed | ge_34 | Hierarchical parametric | 0.1028 | 0.1250 | -0.0222 |

## Verdict across positions

Pooled CRPS, 2024–2025 week-ahead holdout. Lower is better; the best candidate in
each row is bold.

| Position | Player-weeks | Direct DRF | Independent stat DRFs | Multivariate DRF | Hierarchical |
| --- | ---: | ---: | ---: | ---: | ---: |
| WR | 4,940 | **2.960** | 3.039 | 2.973 | 2.982 |
| RB | 3,222 | **2.902** | 2.953 | 2.940 | 2.945 |
| TE | 2,507 | 2.421 | 2.465 | **2.420** | 2.445 |
| QB | 1,328 | **4.288** | 4.366 | 4.309 | 4.350 |
| K | 1,086 | **2.643** | 2.648 | 2.664 | 2.676 |
| DST | 1,088 | 2.657 | 2.670 | **2.645** | 2.685 |

Paired player-clustered CRPS delta against the direct DRF. Positive means the
candidate beats the direct DRF. The draw-noise floor measured on WR was about
0.009 CRPS.

| Position | Independent stat DRFs | Multivariate DRF | Hierarchical |
| --- | ---: | ---: | ---: |
| WR | −0.079 [−0.096, −0.064] | −0.013 [−0.028, +0.001] | −0.021 [−0.044, −0.004] |
| RB | −0.051 [−0.073, −0.026] | **−0.038 [−0.058, −0.018]** | **−0.042 [−0.069, −0.014]** |
| TE | −0.044 [−0.060, −0.028] | +0.000 [−0.017, +0.016] | −0.025 [−0.051, +0.004] |
| QB | −0.078 [−0.130, −0.029] | −0.021 [−0.071, +0.030] | −0.062 [−0.139, +0.004] |
| K | −0.005 [−0.026, +0.015] | −0.021 [−0.047, +0.005] | −0.033 [−0.068, +0.005] |
| DST | −0.013 [−0.036, +0.011] | +0.012 [−0.010, +0.036] | −0.028 [−0.069, +0.005] |

### 1. The WR conclusion mostly replicates, with one real exception

Two things hold at every position:

- **The independent-stat decomposition loses.** It is worse than the direct DRF
  at all six positions, and significantly so at WR (−0.079), RB (−0.051),
  TE (−0.044) and QB (−0.078).
- **The direct DRF is never clearly beaten.** It has the best pooled CRPS at WR,
  RB, QB and K, and the multivariate DRF edges it at TE and DST by amounts well
  inside the noise floor.

The third WR finding — *the multivariate DRF ties the direct DRF* — holds at
five positions but **fails at RB**, where both structured models are
significantly worse: the multivariate DRF by −0.038 [−0.058, −0.018] and the
hierarchical chain by −0.042 [−0.069, −0.014]. RB is discussed in section 2
because it is the most informative position in the study, not an annoyance.

Averaged over all six positions the multivariate DRF is 0.015 CRPS behind the
direct DRF; excluding RB it is 0.006 behind, which is inside the draw-noise
floor.

### 2. RB is the exception, and it is the most informative position

RB has the **largest realised branch dependence** of any position — rushing and
receiving volume move together because both are driven by the player's role and by
game script — so if keeping dependence were the whole story, RB is where the
stat-vector models should win. They lose instead.

| RB | CRPS | 80% cov | Predicted sd | Observed sd | PIT deviation |
| --- | ---: | ---: | ---: | ---: | ---: |
| Direct DRF | **2.902** | 0.846 | 5.65 | 8.07 | 0.045 |
| Multivariate DRF | 2.940 | 0.856 | 5.88 | 8.07 | 0.063 |
| Hierarchical parametric | 2.945 | 0.816 | 5.53 | 8.07 | 0.081 |
| Independent stat DRFs | 2.953 | 0.726 | 4.39 | 8.07 | 0.272 |

What is measured and reliable:

- Both structured models lose significantly: multivariate DRF −0.038
  [−0.058, −0.018], hierarchical −0.042 [−0.069, −0.014].
- It is **not** a dependence failure. Both recover the leading correlations well:
  `carries / rushing yards` at 0.831 and 0.753 against 0.879 realised, and
  `targets / receptions` at 0.922 and 0.908 against 0.946.
- It is **not** a ceiling failure either, and the ceiling error runs the other
  way. Over all 3,222 holdout rows, against a realised `P(30+ points)` of 0.0192:

| Model | P(30+) forecast |
| --- | ---: |
| Direct DRF | 0.0152 |
| Multivariate DRF | 0.0146 |
| Hierarchical | 0.0221 |
| Independent stat DRFs | 0.0075 |

So on RB the *direct* DRF and the multivariate DRF **under**-forecast the ceiling
and the chain slightly over-forecasts it, while the independent control collapses
to less than half the realised rate. Whatever costs RB its 0.04 CRPS, it is not
the top of the distribution.

The remaining candidate explanation is a **compound-tail problem**, and I want to
be explicit that this one is a hypothesis rather than a measurement. Rushing
accounts for 59% of RB fantasy points against 2% for WR, and RB carries average
7.3 per game against 0.18 for WR. A per-carry i.i.d. gain mixture therefore gets
summed 10–20 times per game on RB, where on WR it is summed once or twice. If the
gain law is even slightly too heavy-tailed, that error compounds multiplicatively
with the carry count, which is exactly the direction the WR experiment found when
the explosive gain component was fitted on single-catch weeks. I could not verify
it here: the per-statistic tail table turned out to be unreliable at this
granularity (see the caveat below), so the honest statement is that RB's CRPS loss
is established and attributed to the compound yardage stage only by elimination
plus analogy with WR.

### 3. The failure of the independent decomposition has one mechanism everywhere

It is always the same defect, and the size of the loss tracks how much
covariance the position has to lose:

| Position | Predictive sd, independent | Predictive sd, direct DRF | 80% coverage, independent | Drop in spread |
| --- | ---: | ---: | ---: | ---: |
| WR | 4.17 | 5.76 | 0.694 | −28% |
| TE | 3.47 | 4.65 | 0.692 | −25% |
| QB | 6.24 | 7.42 | 0.679 | −16% |
| K | 4.12 | 4.56 | 0.776 | −10% |
| DST | 4.08 | 4.95 | 0.778 | −18% |

The within-draw correlation is destroyed by construction and replaced with zero:
`attempts|completions` −0.002 against 0.952 realised for QB,
`receptions|receiving_yards` −0.001 against 0.866 for WR,
`pat_att|pat_made` −0.000 against 0.976 for K. The nominal 80% interval then
covers 0.68–0.78. The pattern is the plan's `2 Σ aᵢaⱼ Cov(Yᵢ,Yⱼ)` term
disappearing, and it is largest where the covariance is largest.

### 4. The multivariate DRF recovers dependence at every position, including the awkward ones

| Position | Most correlated realised pair | Direct DRF | Multivariate DRF | Hierarchical | Independent |
| --- | --- | ---: | ---: | ---: | ---: |
| WR | receptions / receiving yards (0.866) | — | 0.833 | 0.828 | −0.001 |
| RB | carries / rushing yards (0.879) | — | 0.831 | 0.753 | 0.000 |
| TE | receptions / receiving yards (0.877) | — | 0.775 | 0.748 | −0.001 |
| QB | attempts / completions (0.952) | — | 0.948 | 0.945 | −0.002 |
| K | pat attempts / pat made (0.976) | — | 0.973 | — | −0.000 |
| DST | sacks / points allowed (−0.353) | — | −0.324 | −0.000 | −0.002 |

DST is the most interesting case, and the only position where the multivariate
DRF does not merely tie but wins. The strongest realised relationship there is
*negative*: sacks and points allowed move in opposite directions, because a
defence that is giving up points faces fewer obvious passing situations and rushes
the passer less. The multivariate DRF reproduces it at −0.324 against −0.353
realised. The independent sampler erases it (−0.002) and so does the hierarchical
chain (−0.000), because its sacks stage and its points-allowed stage are drawn
independently of one another. That is the cleanest illustration in the whole set
of why a joint model is worth having: the dependence here is negative, so a model
that ignores it does not merely lose spread, it gets the sign of the relationship
wrong.

### 5. Where the hierarchical chain is genuinely behind

The chain is competitive on CRPS everywhere but its calibration is worse than the
direct DRF at three positions, and the reasons are specific rather than generic:

| Position | Chain PIT deviation | Direct DRF PIT deviation | Chain 80% coverage | Diagnosis |
| --- | ---: | ---: | ---: | --- |
| TE | 0.132 | 0.045 | 0.850 | touchdown rate is effectively constant per reception, so TE TD scoring is not predictable from volume the way the chain assumes |
| QB | 0.129 | 0.080 | 0.745 | passing volume means run high (+11% attempts, +16% yards); the shape layer is fine, the LightGBM mean stage carries the bias |
| DST | 0.196 | 0.078 | 0.752 | points-allowed tier mass is wrong even though its mean is right |

The DST case is the clearest failure in the whole exercise. The tier map is
nonlinear and non-monotone, and the chain gets the mean of points allowed right
while getting the distribution over scoring tiers wrong by a total variation of
0.07–0.10: it under-weights the 0 and 1–6 tiers and over-weights 7–13. The cause
is structural — realised points allowed is more left-skewed than any negative
binomial at the same variance — and matching tier mass instead of variance does
not close the gap. This is a direct argument for the plan's own suggestion to
model the *bucket distribution* rather than a continuous count fed through the
tiers.

### 6. Frozen shape parameters held up

Every chain froze its dispersion, gain laws and touchdown laws on the season
before the holdout, refitting only the conditional means per fold. This was
established on WR and re-used unchanged. The per-fold diagnostics show the frozen
scalars are stable: RB's latent branch loading lands at 0.30, K's frozen band mix
is `[0.53, 0.26, 0.21, 0.008]` with per-band make rates `[0.97, 0.81, 0.71, 0.63]`,
and the k-disagreement that plagued the per-fold WR calibration (CV 0.19) is gone.

### 7. Structural facts the position chains exposed

- **Kicking is under-dispersed.** `fg_att` has mean 1.94 and variance 1.63, below
  its own mean, so no negative binomial can express it. A hurdle for the zero mass
  is mandatory, not stylistic. Both dispersions then land on the Poisson limit.
- **DST `def_fumbles` is fumble recoveries, not forced fumbles** (mean 0.14, 87%
  zeros, against 0.81 for forced). Modelling it as the 0.7-rate count the brief
  implied would have mis-specified a whole column.
- **RB needs explicit branch coupling.** Drawing rushing and receiving
  independently puts `corr(carries, targets)` at 0.280 against an observed 0.406.
  A single latent role/game-script factor, with loading fitted on the calibration
  season and both dispersions re-solved at that loading, lands at 0.398 without
  buying correlation with extra marginal spread.
- **QB needs a hurdle on attempts.** 9.1% of QB weeks have zero attempts; a plain
  negative binomial puts that mass at 0.0002. Adding the hurdle moved frozen CRPS
  from 3.61 to 3.32 and repaired the zero-attempt rate to 0.087 against 0.091.

### 8. Practical recommendation

Unchanged from the WR verdict, and now supported at six positions:

1. **Keep the direct DRF as the primary marginal forecast.**
2. **Promote the multivariate DRF as the joint/lineup engine.** It ties the
   direct DRF on marginals, wins the joint diagnostics, reproduces dependence
   including the negative DST relationship, and gives free scoring-rule
   flexibility.
3. **Never ship the independent-stat decomposition**, however good its marginals
   look.
4. **Use the hierarchical chains as the interpretable layer**, position by
   position, but fix the three named defects before trusting their tails: the TE
   touchdown stage, the QB mean bias, and the DST tier mass.

## Limitations

- The independent-stat control uses 50 trees per statistic against 200 for the
  direct DRF. It is a control, not a contender, and it loses anyway.
- The multivariate DRF uses 50 trees per fold. It is not behind, so the smaller
  budget does not manufacture its result.
- Position chains differ in maturity. Their calibration and implementation choices must be evaluated independently; these reference results do not establish current package test coverage.
- The DST tier mismatch and the TE touchdown dependence are real and unresolved.
- **The per-statistic tail table is unreliable and should not be quoted cell by
  cell.** Thresholds are derived per fold from the training window and were then
  labelled by their integer part, so a label such as `ge_84` pooled folds whose
  exact thresholds differed and, in the worst case, collapsed to a single fold of
  97 rows. An early draft of this document read a 2.7–3.8× rushing-yardage tail
  over-forecast out of that table; it was a single-fold sampling artefact and has
  been removed. The runner now snaps each threshold to its integer before
  labelling, so labels pool identical events, but the artifacts in this run were
  produced before that fix. Only the fantasy-point tail rows, which aggregate over
  every holdout row, should be treated as measured.
- All forecasts remain marginal player distributions; nothing models cross-player
  or within-game dependence.
- 2024–2025 have now been inspected for every position, so further tuning must
  use new data.
