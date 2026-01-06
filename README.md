# ProdFusion
FYP : From Insight to Forecast: Bayesian &amp; Kalman Models for Garment Manufacturing Productivity.
# Overview

Build a rigorous, **time-aware analytics and forecasting pipeline**—grounded in **Bayesian modeling** and **Kalman filtering (Dynamic Linear Models)**—to explain, predict, and optimize **team-day productivity** in a garment factory using the **UCI Garment Productivity** dataset.

The results will be compared with the findings of the paper *Mining the productivity data of the garment industry*. Specifically, we will cross-validate our Bayesian findings against the paper’s baselines on the questions it actually studied:

1. **Drivers of low productivity**  
   Compare posterior effect sizes and interaction terms (e.g., incentive, WIP, workers, incentive × WIP) with the paper’s rule-based importance and thresholds (notably the ~69.5 BDT incentive split).

2. **Classification performance**  
   For both **3-class** (low/moderate/normal) and **2-class** (low vs. not-low) formulations, report accuracy/AUC on the paper’s split and benchmark against its top models (tree ensemble; GBT + SMOTE).

3. **Actionable thresholds**  
   Check whether Bayesian *what-if* and monotone/spline estimates imply breakpoints consistent with the paper’s rules.

4. **Class-imbalance handling**  
   While Bayesian models do not use SMOTE, compare calibrated probability outputs and cost-sensitive results to the paper’s oversampled 2-class scores to assess whether we match or improve detection of “low” days.

> **Note:** Time-aware forecasting, drift detection, and causal policy optimization are **outside the paper’s scope** and will be presented as **added value**, not direct validations.

---

## External Validation Dataset: SECOM (UCI)

We will also validate the method on the **SECOM** dataset (UCI, semiconductor manufacturing):

- **Size:** 1,567 runs × 590+ anonymized sensor features  
- **Target:** Yield label (−1 = pass, 1 = fail)  
- **Properties:** Timestamped, heavy missingness, strong class imbalance  

SECOM is a strong proxy for line-productivity analytics and is well-suited for:

- Bayesian GLMs with **sparsity priors**
- **Dynamic Linear Models / Kalman filtering** for sensor or tool drift
- Time-aware validation under non-stationarity

Prior studies benchmark SECOM for preprocessing, imbalance handling, and drift-aware industrial analytics (e.g., Salem et al., 2018; Park, 2024; Arpaia et al., 2022).

---

## Research Questions

### Descriptive / Diagnostic
- How do **incentive, WIP, overtime, SMV, and staffing** relate to productivity?
- Are effects **non-linear** (e.g., diminishing returns to incentive)?
- Do interactions (e.g., **WIP × workers**, **incentive × WIP**) meaningfully shift outcomes?
- Are there **calendar effects** (weekday, quarter-of-month) and **style-change dips**?

### Predictive
- What is each team’s **one-day-ahead productivity forecast** with calibrated uncertainty?
- What is the probability of a **low-productivity day tomorrow**,  
  \[
  \Pr(y_{t+1} < \tau)?
  \]

### Prescriptive / What-if
- Expected uplift from:
  - Raising incentives by \(\Delta\)
  - Reducing WIP by \(\Delta\)
  - Adding \(k\) workers
- Simple, actionable thresholds (e.g., incentive \(\ge \tau\), WIP \(\le \tau\)) that minimize low-productivity risk.

### Stability / Heterogeneity
- Do incentive/WIP effects **drift over time** (change-points)?
- Which teams benefit most/least from specific levers (**heterogeneous effects**)?

---

## Dataset & Granularity

- **Unit:** Team × day (Jan–Mar 2015)
- **Target:** `actual_productivity ∈ [0, 1]`
- **Key covariates:**
  - incentive
  - WIP (with missingness)
  - over_time
  - SMV
  - no_of_workers
  - no_of_style_change
  - department, day, quarter

---

## Methodology (Bayesian + Kalman, End-to-End)

### 1) Leakage-Safe Time Splits
- Train up to date \(T\)
- Validate on \((T, T+\Delta]\)
- Test on a held-out **future block**
- Group by team where appropriate

---

### 2) Data Model: Two Complementary Lenses

#### A. Bayesian Beta (or Logit-Normal) Regression

- **Likelihood:**  
  \( y_t \in (0,1) \) via Beta regression
- **Link:**  
  \[
  \text{logit}(\mu_t) = X_t^\top \beta + u_{\text{team}}
  \]
- **Priors:** Weakly informative; hierarchical team effects \(u_{\text{team}}\)
- **Nonlinearity:** Monotone splines for incentive
- **Interactions:** WIP × workers, incentive × WIP
- **Missing WIP:**  
  - Joint latent modeling **or**
  - Missingness indicator (compare both)

#### B. Dynamic Linear Model (Kalman Filter)

- Transform \(y_t\) via logit to \(z_t \in \mathbb{R}\)

**Observation equation**
\[
z_t = X_t^\top \theta_t + v_t, \quad v_t \sim \mathcal{N}(0, R)
\]

**State evolution (drift)**
\[
\theta_t = \theta_{t-1} + w_t, \quad w_t \sim \mathcal{N}(0, Q)
\]

- **Seasonality:** Weekday states or dummies
- **Change-points:** Temporary inflation of \(Q\) at style changes
- **Latent WIP option:**  
  - WIP follows a local-level state
  - Observed WIP updates the state; missing WIP is inferred

**Why both?**
- Bayesian GLM: interpretable global/pooled effects with credible intervals  
- DLM: captures time variation and enables rolling forecasts and online updates

---

### 3) Hyperparameter Learning
- **Kalman EM** to estimate \(Q, R\) or discount-factor tuning on rolling validation
- **Bayesian models:** Prior sensitivity + PSIS-LOO to compare:
  - Link functions
  - Spline complexity

---

### 4) Inference → Decisions
- **Uplift curves:**  
  \[
  \mathbb{E}[\Delta y \mid \Delta \text{incentive}]
  \]
  stratified by team/weekday
- **Risk controls:**  
  Alerts when \(\Pr(y_{t+1} < \tau) > p^\*\)
- **Threshold rules:**  
  e.g., raise incentive if risk > \(p^\*\) and WIP ≤ cap
- **Budgeted allocation:**  
  Greedy optimization using posterior means/variances to maximize expected productivity under an incentive budget

---

## Outputs & Deliverables

### Reports & Dashboards
- Driver analysis (posterior effect sizes, PDP/ICE-style summaries with uncertainty)
- Rolling 1-day-ahead forecasts with 80/95% bands by team
- Heatmaps of low-day risk and interactive *what-if* calculators

### Model Artifacts
- PyMC model for Beta / Logit-Normal regression (saved posterior)
- NumPy/Python DLM with EM; per-team or pooled execution

### Playbook
- Data prep (time splits, encoding, leakage checks)
- Missing-WIP handling strategies
- Threshold recommendation rules
- Budget-allocation procedure

---

## Validation & Metrics

- **Forecasting:** Rolling MAE/RMSE on original scale; interval coverage
- **Classification proxy:** AUC / PR for “low day” threshold
- **Stability:** Coefficient-drift diagnostics; change-point flags aligned with style changes
- **Decision utility:** Expected gain vs. baseline incentives (offline policy evaluation)

---

## Risks & Mitigations

- **Missing WIP (MNAR risk):**  
  Missingness model + latent-state WIP; sensitivity analysis
- **Small samples per team:**  
  Hierarchical pooling; partial pooling of \(Q\) across teams
- **Non-stationarity:**  
  Time-varying coefficients; change-point handling
- **Leakage:**  
  Strict temporal splits; audit feature timestamps (e.g., WIP as start-of-day or latent)

---

## References

- Imran, A. A., Rahim, M. S., & Ahmed, T. (2021). *Mining the productivity data of the garment industry*. **International Journal of Business Intelligence and Data Mining**. DOI: 10.1504/IJBIDM.2021.118183.  
- UCI ML Repository — *Productivity Prediction of Garment Employees* (Dataset). DOI: 10.24432/C51S6D.  
- McCann, M., & Johnston, A. (2008). *SECOM* (Dataset). UCI ML Repository. DOI: 10.24432/C54305.  
- Salem, M., Taheri, S., & Yuan, J.-S. (2018). *An Experimental Evaluation of Fault Diagnosis from Imbalanced and Incomplete Data for Smart Semiconductor Manufacturing*. **Big Data and Cognitive Computing**, 2(4), 30.  
- Park, H. J. (2024). *Study on Data Preprocessing for Machine Learning Based on Semiconductor Manufacturing Processes*. **Sensors**.  
- Arpaia, P., et al. (2022). *Drift-Free Integration in Inductive Magnetic Field Measurements Achieved by Kalman Filtering*. **Sensors**, 22(1), 182.

