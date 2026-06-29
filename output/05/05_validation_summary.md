# Validation Notebook — Summary Report
Generated: 20260629_152618

## Paper Comparison (Imran et al., 2021)
| Benchmark | Paper Accuracy | Our Best (no SMOTE) | Our Best (SMOTE) |
|-----------|---------------|---------------------|-----------------|
| 3class | 83.89% | 80.00% | 79.72% ✗ |
| 2class | 86.39% | 84.72% | 85.28% ✗ |

## 69.5 BDT Incentive Threshold (Imran et al.)
- Mann-Whitney U p-value: 6.130e-33
- Chi-square (incentive × low_day) p: 1.898e-10
- Conclusion: Threshold statistically validated ✓

## Outputs saved
- `output/05/05a_paper_accuracy_comparison.png`
- `output/05/05b_incentive_threshold_validation.png`
- `output/05/05c_group_reliability.png`
- `output/05/05d_calendar_effects.png`
- `output/05/05e_style_change_drift.png`
- `output/05/05f_risk_heatmap_team_dow.png`
- `output/05/05g_heterogeneity.png`
- `output/05/05h_decision_utility_whatif.png`
- `output/05/05i_secom_validation.png`
- `output/05/05_classifier_results.csv`
- `output/05/05_group_reliability_table.csv`
- `output/05/05_latest_saved_kalman_forecasts.csv`
- `output/05/05_heterogeneity_table.csv`
- `output/05/05_secom_results.csv`

## References
- Imran et al. (2021). Mining the productivity data of the garment industry. IJBIDM.
- UCI ML Repository — Garment Worker Productivity Dataset.
- McCann & Johnston (2008). SECOM Dataset. UCI ML Repository.