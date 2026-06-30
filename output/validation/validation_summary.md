# Validation Summary — Exact Model C + Kalman DLM

## Protocol

- The Bayesian model was not refitted. The authoritative Model C posterior, metadata, split indices, and posterior prediction draws were loaded and verified.
- Notebook 05's alternative Bayesian fit and residual Kalman correction were deliberately excluded.
- The paper models were not retrained; published values are reference benchmarks.
- Model C uses no SMOTE. Default and cost-sensitive probability decisions are reported with calibration diagnostics.
- The corrected Kalman v7.1 model is evaluated on its chronological future test block.
- Bayesian + Kalman results are combined only on the dual-holdout intersection.

## 1. Driver validation against the garment paper

- **Incentive: 0 → 69.5 BDT (conditional at median WIP=560.0)**: mean=1.094, 94% interval [0.893, 1.305]; Aligned.
- **WIP coefficient**: mean=0.403, 94% interval [0.047, 0.762]; Not clearly aligned.
- **Number of workers coefficient**: mean=0.213, 94% interval [-0.020, 0.455]; Not clearly aligned.
- **Incentive × WIP interaction**: mean=0.635, 94% interval [0.480, 0.798]; Interaction assessed; the paper gives a rule rather than a signed coefficient.
- **WIP × workers interaction**: mean=-0.366, 94% interval [-0.555, -0.155]; Interaction assessed; the paper gives a rule rather than a signed coefficient.

## 2. Classification performance

- Exact Model C 3-class accuracy: **74.72%** (paper tree ensemble: **83.89%**).
- Exact Model C 3-class macro OvR AUC: **0.647**.
- Exact Model C 2-class default accuracy/AUC: **82.50% / 0.675** (paper GBT+SMOTE: **86.39% / 0.900**).
- Default low precision/recall/F1: **51.72% / 23.44% / 0.323**.
- Cost-sensitive low precision/recall/F1: **28.57% / 37.50% / 0.324**.
- Default probability quality: Brier **0.144**, ECE **0.080**, PR-AUC **0.345**.
- Default false negatives/false positives/expected cost: **49 / 14 / 112.0**.
- Cost-sensitive false negatives/false positives/expected cost: **40 / 60 / 140.0**.
- Exact Model C test MAE/RMSE: **0.095 / 0.130**.

## 3. Actionable incentive thresholds

- **Low WIP** (WIP≈0.0): median threshold **0.0 BDT**, 94% interval **[0.0, 118.0] BDT**, crossing rate **21.4%**; Paper 69.5 BDT lies inside the 94% posterior threshold interval.
- **Median WIP** (WIP≈560.0): median threshold **55.0 BDT**, 94% interval **[45.0, 75.0] BDT**, crossing rate **100.0%**; Paper 69.5 BDT lies inside the 94% posterior threshold interval.
- **High WIP** (WIP≈1075.0): median threshold **45.0 BDT**, 94% interval **[35.0, 65.0] BDT**, crossing rate **100.0%**; Paper 69.5 BDT lies outside the 94% posterior threshold interval.

## 4. What-if decisions

- **Baseline**: expected productivity **0.728**; P(low) **25.05%**; change vs baseline **+0.000** productivity and **+0.00%** low risk.
- **+20 BDT incentive**: expected productivity **0.621**; P(low) **44.81%**; change vs baseline **-0.106** productivity and **+19.76%** low risk.
- **Reduce WIP by 20%**: expected productivity **0.724**; P(low) **25.66%**; change vs baseline **-0.003** productivity and **+0.61%** low risk.
- **+5 workers**: expected productivity **0.746**; P(low) **22.25%**; change vs baseline **+0.019** productivity and **-2.80%** low risk.

## 5. Corrected Kalman added value

- Future-test MAE/RMSE: **0.086 / 0.132**.
- Prediction bias (predicted − actual): **-0.000**.
- 80%/95% interval coverage: **88.5% / 97.9%**.
- Paper-style low AUC/AP/Brier: **0.715 / 0.348 / 0.127**.
- Dual-holdout ensemble 2-class accuracy/AUC: **0.768 / 0.668** on **n=56** rows.

## 6. SECOM status

- SECOM was not executed here because Notebook 05 supplied no executable SECOM implementation. This is recorded as pending rather than reported as completed.

## 7. Notebook 05 migration safeguards

- Adopted: comparison tables, ECE/cost metrics, richer figures, paper-linked driver contrasts, posterior threshold uncertainty, and scenario plots.
- Excluded: a new Bayesian fit, a newly fitted residual correction, and fallback to the old `dlm_forecasts.csv` file.

## Important limitations

- The paper benchmark's exact random split rows were not published.
- Predictive what-if differences are not causal treatment effects.
- The corrected Kalman v7.1 model tracks residual-level drift rather than separate time-varying incentive and WIP slopes.
- Missing-WIP sensitivity is not a separately fitted latent-state WIP model.

## Generated files

- `validation_actionable_threshold_draws.csv`
- `validation_actionable_threshold_posterior.csv`
- `validation_actionable_thresholds.csv`
- `validation_artifact_audit.csv`
- `validation_bayesian_binary_bootstrap_ci.csv`
- `validation_bayesian_paper_predictions.csv`
- `validation_bayesian_regression_metrics.csv`
- `validation_bayesian_regression_scatter.png`
- `validation_budget_allocation.csv`
- `validation_budget_allocation_summary.csv`
- `validation_budget_utility.png`
- `validation_calendar_effects.csv`
- `validation_confusion_matrices.png`
- `validation_dual_holdout_comparison.png`
- `validation_dual_holdout_decisions.csv`
- `validation_dual_holdout_metrics.csv`
- `validation_exact_model_c_incentive_effect.csv`
- `validation_imbalance_comparison.csv`
- `validation_incentive_productivity_curve.png`
- `validation_incentive_threshold.png`
- `validation_incentive_threshold_curves.csv`
- `validation_kalman_aligned_forecasts.csv`
- `validation_kalman_temporal_metrics.csv`
- `validation_loo_comparison_copy.csv`
- `validation_missing_wip_sensitivity.csv`
- `validation_model_c_reconstruction_check.csv`
- `validation_notebook_05_migration_audit.csv`
- `validation_objective_coverage_audit.csv`
- `validation_paper_benchmark_comparison.png`
- `validation_paper_classification_metrics.csv`
- `validation_paper_comparison.csv`
- `validation_paper_driver_alignment.csv`
- `validation_paper_driver_and_incentive_effect.png`
- `validation_paper_reference_metrics.csv`
- `validation_posterior_driver_table.csv`
- `validation_posterior_drivers.png`
- `validation_roc_pr_calibration.png`
- `validation_style_change_drift.csv`
- `validation_style_change_drift.png`
- `validation_team_heterogeneity.csv`
- `validation_team_heterogeneity.png`
- `validation_whatif_expected_productivity.png`
- `validation_whatif_low_productivity_probability.png`
- `validation_whatif_productivity_low_risk_rows.csv`
- `validation_whatif_productivity_low_risk_summary.csv`
- `validation_whatif_row_level.csv`
- `validation_whatif_summary.csv`
- `validation_whatif_uplift_vs_risk_reduction.png`