

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


PAGE_FILE = Path(__file__).resolve()
DASHBOARD_DIR = PAGE_FILE.parents[1]

if str(DASHBOARD_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(DASHBOARD_DIR),
    )


from utils.decision_engine import (
    context_table,
    baseline_plan,
    compare_scenario,
    action_candidates,
    incentive_budget_allocation,
    reliability_metrics,
)
from utils.forecast_engine import (
    default_forecast_date,
)


st.set_page_config(
    page_title="What-If & Decision Planner",
    page_icon="🧭",
    layout="wide",
)


# =============================================================================
# HEADER
# =============================================================================

st.title(
    "What-If & Decision Planner"
)

st.caption(
    "Test practical changes, compare expected productivity and low-day risk, "
    "and allocate a limited incentive budget across team-lines."
)


# =============================================================================
# 1. DECISION CONTEXT
# =============================================================================

st.subheader(
    "1. Decision Context"
)


context = context_table().copy()


def team_number(
    label,
):

    digits = "".join(
        character
        for character in str(
            label
        )
        if character.isdigit()
    )

    return int(
        digits
    ) if digits else 9999


c1, c2, c3 = st.columns(
    [
        1.1,
        1.1,
        1.0,
    ]
)


with c1:

    departments = sorted(
        context[
            "department_display"
        ]
        .dropna()
        .unique()
        .tolist()
    )

    selected_department = st.selectbox(
        "Department",
        departments,
        key="decision_department",
    )


department_context = context.loc[
    context[
        "department_display"
    ]
    == selected_department
].copy()


with c2:

    teams = sorted(
        department_context[
            "team_display"
        ]
        .dropna()
        .unique()
        .tolist(),
        key=team_number,
    )

    selected_team = st.selectbox(
        "Team",
        teams,
        key="decision_team",
    )


selected_context = (
    department_context.loc[
        department_context[
            "team_display"
        ]
        == selected_team
    ]
    .iloc[0]
)


series_id = str(
    selected_context[
        "series_id"
    ]
)


with c3:

    forecast_date = st.date_input(
        "Forecast Date",
        value=default_forecast_date(),
        key="decision_forecast_date",
        help=(
            "The planned production date. "
            "Friday is outside the Sat–Thu model calendar."
        ),
    )


# =============================================================================
# 2. CURRENT PLAN
# =============================================================================

base = baseline_plan(
    series_id
)


st.subheader(
    "2. Current Plan"
)


p1, p2, p3, p4, p5, p6 = st.columns(
    6
)


with p1:
    st.metric(
        "Incentive",
        f"{base['incentive']:.0f} BDT",
    )

with p2:
    st.metric(
        "WIP",
        f"{base['wip']:.0f}",
    )

with p3:
    st.metric(
        "Workers",
        f"{base['no_of_workers']:.0f}",
    )

with p4:
    st.metric(
        "Overtime",
        f"{base['over_time']:.0f} min",
    )

with p5:
    st.metric(
        "SMV",
        f"{base['smv']:.2f}",
    )

with p6:
    st.metric(
        "Style Changes",
        f"{base['no_of_style_change']}",
    )


st.caption(
    "Page 5 changes only the three decision levers requested by the project: "
    "**Incentive, WIP and Workers**. Other operating conditions remain fixed "
    "at the selected team-line's latest saved plan."
)


# =============================================================================
# 3. WHAT-IF SCENARIO
# =============================================================================

st.subheader(
    "3. What-If Scenario"
)


with st.form(
    "what_if_form"
):

    w1, w2, w3 = st.columns(
        3
    )


    with w1:

        scenario_incentive = st.number_input(
            "Incentive (BDT)",
            min_value=0.0,
            value=float(
                base[
                    "incentive"
                ]
            ),
            step=5.0,
        )


    with w2:

        scenario_wip = st.number_input(
            "WIP (pieces)",
            min_value=0.0,
            value=float(
                base[
                    "wip"
                ]
            ),
            step=10.0,
        )


    with w3:

        scenario_workers = st.number_input(
            "Number of Workers",
            min_value=0.0,
            value=float(
                base[
                    "no_of_workers"
                ]
            ),
            step=1.0,
        )


    run_scenario = st.form_submit_button(
        "Compare Scenario",
        type="primary",
        use_container_width=True,
    )


# Initialize comparison from current values so the rest of the page
# can render even before the manager changes anything.
try:

    comparison = compare_scenario(
        series_id=series_id,
        forecast_date=forecast_date,
        incentive=scenario_incentive,
        wip=scenario_wip,
        no_of_workers=scenario_workers,
    )

except Exception as exc:

    st.error(
        f"Scenario could not be evaluated: {exc}"
    )

    st.stop()


baseline_result = comparison[
    "baseline_forecast"
]

scenario_result = comparison[
    "scenario_forecast"
]


# =============================================================================
# 4. BASELINE VS SCENARIO
# =============================================================================

st.subheader(
    "4. Baseline vs Scenario"
)


gain_pp = (
    comparison[
        "expected_productivity_gain"
    ]
    * 100
)

risk_change_pp = (
    comparison[
        "low_day_risk_change"
    ]
    * 100
)


r1, r2, r3, r4 = st.columns(
    4
)


with r1:

    st.metric(
        "Current Expected Productivity",
        f"{baseline_result['expected_productivity']:.1%}",
    )


with r2:

    st.metric(
        "Scenario Expected Productivity",
        f"{scenario_result['expected_productivity']:.1%}",
        delta=f"{gain_pp:+.1f} pp",
    )


with r3:

    st.metric(
        "Current Low-Day Risk",
        f"{baseline_result['low_day_risk']:.1%}",
    )


with r4:

    # A negative risk change is an improvement.
    st.metric(
        "Scenario Low-Day Risk",
        f"{scenario_result['low_day_risk']:.1%}",
        delta=f"{risk_change_pp:+.1f} pp",
        delta_color="inverse",
    )


comparison_df = pd.DataFrame(
    {
        "Measure": [
            "Expected Productivity",
            "Low-Day Risk",
            "80% Range",
            "95% Range",
        ],
        "Current Plan": [
            f"{baseline_result['expected_productivity']:.1%}",
            f"{baseline_result['low_day_risk']:.1%}",
            (
                f"{baseline_result['pi80_lower']:.1%} – "
                f"{baseline_result['pi80_upper']:.1%}"
            ),
            (
                f"{baseline_result['pi95_lower']:.1%} – "
                f"{baseline_result['pi95_upper']:.1%}"
            ),
        ],
        "What-If Plan": [
            f"{scenario_result['expected_productivity']:.1%}",
            f"{scenario_result['low_day_risk']:.1%}",
            (
                f"{scenario_result['pi80_lower']:.1%} – "
                f"{scenario_result['pi80_upper']:.1%}"
            ),
            (
                f"{scenario_result['pi95_lower']:.1%} – "
                f"{scenario_result['pi95_upper']:.1%}"
            ),
        ],
    }
)


st.dataframe(
    comparison_df,
    use_container_width=True,
    hide_index=True,
)


if gain_pp > 0 and risk_change_pp < 0:

    st.success(
        f"This scenario is favorable in both model outputs: expected productivity "
        f"improves by {gain_pp:.1f} pp and low-day risk falls by "
        f"{abs(risk_change_pp):.1f} pp."
    )

elif gain_pp > 0:

    st.info(
        f"Expected productivity improves by {gain_pp:.1f} pp, "
        "but review the risk result before adopting the plan."
    )

elif risk_change_pp < 0:

    st.info(
        f"Low-day risk falls by {abs(risk_change_pp):.1f} pp, "
        "although the expected-productivity improvement is limited."
    )

elif (
    abs(gain_pp) < 0.05
    and abs(risk_change_pp) < 0.05
):

    st.info(
        "The proposed values produce almost no model-estimated change "
        "from the current plan."
    )

else:

    st.warning(
        "The proposed scenario does not improve the current plan on the main "
        "model outputs."
    )


st.caption(
    "These are model-based scenario comparisons, not guaranteed causal effects."
)


# =============================================================================
# 5. ACTION OPTIONS
# =============================================================================

st.subheader(
    "5. Action Options"
)


try:

    actions = action_candidates(
        series_id,
        forecast_date,
    )

except Exception as exc:

    st.warning(
        f"Action options could not be generated: {exc}"
    )

    actions = pd.DataFrame()


if not actions.empty:

    actions_display = actions.copy()

    actions_display[
        "Expected Productivity"
    ] = actions_display[
        "expected_productivity"
    ].map(
        lambda value:
            f"{value:.1%}"
    )

    actions_display[
        "Expected Gain"
    ] = actions_display[
        "productivity_gain"
    ].map(
        lambda value:
            f"{value * 100:+.1f} pp"
    )

    actions_display[
        "Low-Day Risk"
    ] = actions_display[
        "low_day_risk"
    ].map(
        lambda value:
            f"{value:.1%}"
    )

    actions_display[
        "Risk Reduction"
    ] = actions_display[
        "risk_reduction"
    ].map(
        lambda value:
            f"{value * 100:+.1f} pp"
    )

    actions_display[
        "Recommended"
    ] = actions_display[
        "recommended"
    ].map(
        {
            True:
                "★ Best tested option",
            False:
                "",
        }
    )


    st.dataframe(
        actions_display[
            [
                "action",
                "Expected Productivity",
                "Expected Gain",
                "Low-Day Risk",
                "Risk Reduction",
                "Recommended",
            ]
        ].rename(
            columns={
                "action":
                    "Tested Action",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )


    best_action = actions.iloc[0]

    st.info(
        f"Among the preset single-lever options tested here, "
        f"**{best_action['action']}** gives the largest modeled risk reduction, "
        f"with an expected productivity change of "
        f"{best_action['productivity_gain'] * 100:+.1f} pp."
    )


st.caption(
    "The action table tests a small preset set of practical changes. "
    "It is a decision aid, not an automatic operating instruction."
)


# =============================================================================
# 6. INCENTIVE BUDGET ALLOCATION
# =============================================================================

st.subheader(
    "6. Incentive Budget Allocation"
)


st.write(
    "Allocate a limited **additional incentive budget** across team-lines "
    "using greedy marginal expected productivity gain. "
    "The allocator stops when no remaining tested incentive increment "
    "has positive modeled productivity gain."
)


b1, b2, b3, b4 = st.columns(
    4
)


with b1:

    budget_scope = st.selectbox(
        "Allocation Scope",
        [
            "All",
            *sorted(
                context[
                    "department_display"
                ]
                .dropna()
                .unique()
                .tolist()
            ),
        ],
    )


with b2:

    incentive_budget = st.number_input(
        "Additional Budget (BDT)",
        min_value=0.0,
        value=200.0,
        step=50.0,
    )


with b3:

    allocation_step = st.number_input(
        "Allocation Step (BDT)",
        min_value=1.0,
        value=10.0,
        step=5.0,
    )


with b4:

    max_extra_line = st.number_input(
        "Max Extra per Team-Line (BDT)",
        min_value=float(
            allocation_step
        ),
        value=max(
            50.0,
            float(
                allocation_step
            ),
        ),
        step=5.0,
    )


run_allocation = st.button(
    "Allocate Incentive Budget",
    type="primary",
    use_container_width=True,
)


if run_allocation:

    try:

        (
            allocation,
            allocation_summary,
        ) = incentive_budget_allocation(
            forecast_date=forecast_date,
            budget_bdt=incentive_budget,
            department=budget_scope,
            increment_bdt=allocation_step,
            max_extra_per_line=max_extra_line,
        )

    except Exception as exc:

        st.error(
            f"Budget allocation could not be completed: {exc}"
        )

        allocation = pd.DataFrame()
        allocation_summary = None


    if (
        allocation_summary is not None
        and not allocation.empty
    ):

        a1, a2, a3, a4 = st.columns(
            4
        )


        with a1:

            st.metric(
                "Budget",
                f"{allocation_summary['budget_bdt']:.0f} BDT",
            )


        with a2:

            st.metric(
                "Allocated",
                f"{allocation_summary['spent_bdt']:.0f} BDT",
            )


        with a3:

            st.metric(
                "Expected Mean Productivity",
                f"{allocation_summary['allocated_mean_productivity']:.1%}",
                delta=(
                    f"{allocation_summary['expected_mean_gain'] * 100:+.2f} pp"
                ),
            )


        with a4:

            risk_delta = (
                allocation_summary[
                    "allocated_mean_risk"
                ]
                -
                allocation_summary[
                    "baseline_mean_risk"
                ]
            )

            st.metric(
                "Expected Mean Low-Day Risk",
                f"{allocation_summary['allocated_mean_risk']:.1%}",
                delta=f"{risk_delta * 100:+.2f} pp",
                delta_color="inverse",
            )


        display_allocation = allocation.loc[
            allocation[
                "extra_incentive"
            ]
            > 0
        ].copy()


        if display_allocation.empty:

            st.info(
                "No positive modeled marginal productivity gain was found "
                "for the tested incentive increments, so the allocator "
                "did not spend the budget."
            )

        else:

            display_allocation[
                "Baseline Incentive"
            ] = display_allocation[
                "baseline_incentive"
            ].map(
                lambda value:
                    f"{value:.0f} BDT"
            )

            display_allocation[
                "Extra Incentive"
            ] = display_allocation[
                "extra_incentive"
            ].map(
                lambda value:
                    f"{value:.0f} BDT"
            )

            display_allocation[
                "Recommended Incentive"
            ] = display_allocation[
                "recommended_incentive"
            ].map(
                lambda value:
                    f"{value:.0f} BDT"
            )

            display_allocation[
                "Expected Gain"
            ] = display_allocation[
                "expected_gain"
            ].map(
                lambda value:
                    f"{value * 100:+.2f} pp"
            )

            display_allocation[
                "Risk Reduction"
            ] = display_allocation[
                "risk_reduction"
            ].map(
                lambda value:
                    f"{value * 100:+.2f} pp"
            )


            st.dataframe(
                display_allocation[
                    [
                        "team",
                        "department",
                        "Baseline Incentive",
                        "Extra Incentive",
                        "Recommended Incentive",
                        "Expected Gain",
                        "Risk Reduction",
                    ]
                ].rename(
                    columns={
                        "team":
                            "Team",
                        "department":
                            "Department",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )


        if allocation_summary[
            "unspent_bdt"
        ] > 0:

            st.caption(
                f"{allocation_summary['unspent_bdt']:.0f} BDT remains unallocated "
                "because the greedy procedure found no further positive modeled "
                "marginal productivity gain within the selected limits."
            )


        st.caption(
            "Budget allocation maximizes modeled expected productivity using "
            "incremental incentive changes. It does not account for worker cost, "
            "WIP-control cost, causal identification, or monetary value of output."
        )


# =============================================================================
# 7. FORECAST RELIABILITY
# =============================================================================

st.subheader(
    "7. Forecast Reliability"
)


reliability = reliability_metrics()


if reliability.get(
    "source"
) == "unavailable":

    st.warning(
        "No reliability artifact is available."
    )

else:

    q1, q2, q3, q4 = st.columns(
        4
    )


    mae = reliability.get(
        "mae"
    )

    rmse = reliability.get(
        "rmse"
    )

    coverage = reliability.get(
        "interval_coverage"
    )

    interval_level = reliability.get(
        "interval_level"
    )


    with q1:

        st.metric(
            "Rolling MAE",
            (
                f"{mae:.3f}"
                if mae is not None
                else "N/A"
            ),
        )


    with q2:

        st.metric(
            "Rolling RMSE",
            (
                f"{rmse:.3f}"
                if rmse is not None
                else "N/A"
            ),
        )


    with q3:

        coverage_label = (
            f"{interval_level:.0%} Interval Coverage"
            if interval_level is not None
            else "Interval Coverage"
        )

        st.metric(
            coverage_label,
            (
                f"{coverage:.1%}"
                if coverage is not None
                else "N/A"
            ),
        )


    with q4:

        alarm_recall = reliability.get(
            "alarm_recall"
        )

        st.metric(
            "Low-Day Alert Recall",
            (
                f"{alarm_recall:.1%}"
                if alarm_recall is not None
                else "N/A"
            ),
        )


    rcol1, rcol2, rcol3, rcol4 = st.columns(
        4
    )


    with rcol1:

        precision = reliability.get(
            "alarm_precision"
        )

        st.metric(
            "Alert Precision",
            (
                f"{precision:.1%}"
                if precision is not None
                else "N/A"
            ),
        )


    with rcol2:

        f1 = reliability.get(
            "alarm_f1"
        )

        st.metric(
            "Alert F1",
            (
                f"{f1:.3f}"
                if f1 is not None
                else "N/A"
            ),
        )


    with rcol3:

        auc = reliability.get(
            "roc_auc"
        )

        st.metric(
            "Low-Day ROC-AUC",
            (
                f"{auc:.3f}"
                if auc is not None
                else "N/A"
            ),
        )


    with rcol4:

        n_eval = reliability.get(
            "n"
        )

        st.metric(
            "Evaluation Rows",
            (
                str(
                    int(
                        n_eval
                    )
                )
                if n_eval is not None
                else "N/A"
            ),
        )


    st.caption(
        f"Reliability source: {reliability.get('source')}. "
        f"Evaluation scope: {reliability.get('evaluation_scope', 'not specified')}."
    )


    if (
        coverage is not None
        and interval_level is not None
    ):

        difference = (
            coverage
            - interval_level
        )

        if abs(
            difference
        ) <= 0.03:

            st.success(
                "Prediction-interval coverage is close to its nominal level."
            )

        elif difference > 0:

            st.info(
                "Prediction intervals are covering more observations than nominal, "
                "which may indicate conservative/wider uncertainty bands."
            )

        else:

            st.warning(
                "Prediction intervals are covering fewer observations than nominal; "
                "uncertainty may be understated."
            )


# =============================================================================
# 8. DECISION SUMMARY
# =============================================================================

st.subheader(
    "8. Decision Summary"
)


decision_sentence = (
    f"For **{selected_team} · {selected_department}**, the tested what-if plan "
    f"changes expected productivity by **{gain_pp:+.1f} pp** and low-day risk "
    f"by **{risk_change_pp:+.1f} pp** relative to the current saved plan."
)


st.info(
    decision_sentence
)


st.markdown(
    """
**Use the five pages together:**

**Daily Overview** → find the team needing attention  
**Next-Day Forecast** → estimate the planned outcome  
**Risk & Alerts** → assess urgency  
**Drivers & Team Trends** → understand why  
**What-If & Decision Planner** → compare actions and allocate resources
"""
)


st.caption(
    "All recommendations are based on fitted-model associations and saved "
    "forecast states. Validate important operating changes with production "
    "knowledge and offline/controlled evaluation before treating them as policy."
)

