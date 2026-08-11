

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


from utils.risk_engine import (
    load_risk_data,
    available_periods,
    snapshot,
    risk_summary,
    calibration_info,
)


st.set_page_config(
    page_title="Risk & Alerts",
    page_icon="⚠️",
    layout="wide",
)


@st.cache_data(
    show_spinner=False
)
def cached_risk_data():

    return load_risk_data()


risk_df = cached_risk_data()

periods = available_periods(
    risk_df
)

if periods.empty:

    st.error(
        "No risk periods are available."
    )

    st.stop()


# ------------------------------------------------------------------
# Header
# ------------------------------------------------------------------

st.title("Risk & Alerts")

st.caption(
    "Identify team-lines most likely to face a low-productivity day "
    "and prioritize management attention."
)


# ------------------------------------------------------------------
# Risk context
# ------------------------------------------------------------------

st.subheader(
    "1. Risk Context"
)

context_col1, context_col2 = st.columns(
    [1.3, 1.0]
)


period_labels = (
    periods["_period_label"]
    .tolist()
)

period_by_label = {
    row["_period_label"]:
        row["_period_key"]
    for _, row
    in periods.iterrows()
}


with context_col1:

    selected_period_label = st.selectbox(
        "Forecast Date",
        period_labels,
        index=len(
            period_labels
        ) - 1,
        help=(
            "Select the production day whose one-day-ahead "
            "risk forecasts you want to review."
        ),
    )


with context_col2:

    department_options = [
        "All"
    ] + sorted(
        risk_df[
            "department_display"
        ]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    selected_department = st.selectbox(
        "Department",
        department_options,
    )


selected_period_key = (
    period_by_label[
        selected_period_label
    ]
)


risk_snapshot = snapshot(
    selected_period_key,
    department=selected_department,
    df=risk_df,
)


if risk_snapshot.empty:

    st.warning(
        "No team-line risk forecasts are available "
        "for the selected context."
    )

    st.stop()


# ------------------------------------------------------------------
# Risk summary
# ------------------------------------------------------------------

st.subheader(
    "2. Risk Summary"
)

summary = risk_summary(
    risk_snapshot
)


m1, m2, m3, m4 = st.columns(
    4
)


with m1:

    st.metric(
        "High-Risk Lines",
        summary[
            "high_risk"
        ],
    )


with m2:

    st.metric(
        "Watch Lines",
        summary[
            "watch"
        ],
    )


with m3:

    st.metric(
        "Healthy Lines",
        summary[
            "healthy"
        ],
    )


with m4:

    st.metric(
        "Average Low-Day Risk",
        f"{summary['average_risk']:.1%}",
    )


calibration = calibration_info()

threshold_fraction = calibration.get(
    "achievement_low_threshold"
)

if threshold_fraction is not None:

    st.caption(
        "Low-Day Risk is the model-estimated probability of productivity "
        f"falling below the calibrated low-day level "
        f"({float(threshold_fraction):.0%} of the selected target)."
    )

else:

    st.caption(
        "Low-Day Risk is the model-estimated probability of productivity "
        "falling below the calibrated low-day threshold."
    )


# ------------------------------------------------------------------
# Heatmap
# ------------------------------------------------------------------

st.subheader(
    "3. Low-Day Risk Heatmap"
)


heat_source = (
    risk_snapshot[
        [
            "department_display",
            "team_display",
            "risk_rank",
            "p_alarm",
            "risk_status",
        ]
    ]
    .copy()
)


def team_number(
    label,
):

    digits = "".join(
        character
        for character in str(label)
        if character.isdigit()
    )

    return (
        int(digits)
        if digits
        else 9999
    )


team_order = sorted(
    heat_source[
        "team_display"
    ]
    .unique()
    .tolist(),
    key=team_number,
)

department_order = sorted(
    heat_source[
        "department_display"
    ]
    .unique()
    .tolist()
)


rank_matrix = (
    heat_source.pivot_table(
        index="department_display",
        columns="team_display",
        values="risk_rank",
        aggfunc="max",
    )
    .reindex(
        index=department_order,
        columns=team_order,
    )
)


risk_matrix = (
    heat_source.pivot_table(
        index="department_display",
        columns="team_display",
        values="p_alarm",
        aggfunc="max",
    )
    .reindex(
        index=department_order,
        columns=team_order,
    )
)


status_matrix = (
    heat_source.pivot_table(
        index="department_display",
        columns="team_display",
        values="risk_status",
        aggfunc="first",
    )
    .reindex(
        index=department_order,
        columns=team_order,
    )
)


text_matrix = np.empty(
    risk_matrix.shape,
    dtype=object,
)


hover_matrix = np.empty(
    risk_matrix.shape,
    dtype=object,
)


for row_index in range(
    risk_matrix.shape[0]
):

    for col_index in range(
        risk_matrix.shape[1]
    ):

        value = (
            risk_matrix.iloc[
                row_index,
                col_index,
            ]
        )

        if pd.isna(value):

            text_matrix[
                row_index,
                col_index,
            ] = ""

            hover_matrix[
                row_index,
                col_index,
            ] = "No forecast"

        else:

            status = (
                status_matrix.iloc[
                    row_index,
                    col_index,
                ]
            )

            text_matrix[
                row_index,
                col_index,
            ] = f"{value:.0%}"

            hover_matrix[
                row_index,
                col_index,
            ] = (
                f"{status}<br>"
                f"Low-day risk: {value:.1%}"
            )


# Discrete manager-facing colors:
# 0 Healthy, 1 Watch, 2 High Risk.
colorscale = [
    [0.000, "#DCFCE7"],
    [0.249, "#DCFCE7"],
    [0.250, "#FEF3C7"],
    [0.749, "#FEF3C7"],
    [0.750, "#FEE2E2"],
    [1.000, "#FEE2E2"],
]


figure = go.Figure(
    data=go.Heatmap(
        z=rank_matrix.to_numpy(
            dtype=float
        ),
        x=team_order,
        y=department_order,
        text=text_matrix,
        customdata=hover_matrix,
        texttemplate="%{text}",
        hovertemplate=(
            "<b>%{y} · %{x}</b><br>"
            "%{customdata}"
            "<extra></extra>"
        ),
        zmin=0,
        zmax=2,
        colorscale=colorscale,
        showscale=False,
        xgap=3,
        ygap=3,
    )
)


figure.update_layout(
    height=max(
        260,
        95
        * len(
            department_order
        )
        + 100,
    ),
    margin=dict(
        l=15,
        r=15,
        t=15,
        b=15,
    ),
    xaxis=dict(
        title="Team",
        side="top",
        tickangle=0,
        showgrid=False,
    ),
    yaxis=dict(
        title="",
        autorange="reversed",
        showgrid=False,
    ),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
)


st.plotly_chart(
    figure,
    use_container_width=True,
    config={
        "displayModeBar": False,
    },
)


legend_a, legend_b, legend_c = st.columns(
    3
)

with legend_a:
    st.success(
        "🟢 Healthy"
    )

with legend_b:
    st.warning(
        "🟠 Watch"
    )

with legend_c:
    st.error(
        "🔴 High Risk"
    )


# ------------------------------------------------------------------
# Priority alerts
# ------------------------------------------------------------------

st.subheader(
    "4. Priority Alerts"
)


priority = risk_snapshot.copy()

priority["Team"] = (
    priority[
        "team_display"
    ]
)

priority["Department"] = (
    priority[
        "department_display"
    ]
)

priority["Low-Day Risk"] = (
    priority["p_alarm"]
    .map(
        lambda value:
            f"{value:.1%}"
    )
)

priority["Forecast"] = (
    priority[
        "predicted_productivity"
    ]
    .map(
        lambda value:
            f"{value:.1%}"
    )
)

priority["Target"] = (
    priority[
        "targeted_productivity"
    ]
    .map(
        lambda value:
            f"{value:.1%}"
    )
)

priority["Gap"] = (
    priority[
        "forecast_gap"
    ]
    .map(
        lambda value:
            f"{value * 100:+.1f} pp"
    )
)


status_icons = {
    "High Risk": "🔴 High Risk",
    "Watch": "🟠 Watch",
    "Healthy": "🟢 Healthy",
}


priority["Status"] = (
    priority[
        "risk_status"
    ]
    .map(
        status_icons
    )
)


priority_display = (
    priority[
        [
            "Team",
            "Department",
            "Low-Day Risk",
            "Forecast",
            "Target",
            "Gap",
            "Status",
        ]
    ]
    .reset_index(
        drop=True
    )
)


priority_display.index = (
    np.arange(
        1,
        len(
            priority_display
        )
        + 1,
    )
)


st.dataframe(
    priority_display,
    use_container_width=True,
    height=min(
        430,
        38
        * (
            len(
                priority_display
            )
            + 1
        ),
    ),
)


# ------------------------------------------------------------------
# Alert detail
# ------------------------------------------------------------------

st.subheader(
    "5. Alert Detail"
)


detail_options = (
    risk_snapshot.apply(
        lambda row:
            f"{row['team_display']} · "
            f"{row['department_display']}",
        axis=1,
    )
    .tolist()
)


selected_line = st.selectbox(
    "Selected Team-Line",
    detail_options,
)


detail_index = (
    detail_options.index(
        selected_line
    )
)

detail = (
    risk_snapshot.iloc[
        detail_index
    ]
)


d1, d2, d3, d4 = st.columns(
    4
)


with d1:

    st.metric(
        "Low-Day Risk",
        f"{detail['p_alarm']:.1%}",
    )


with d2:

    st.metric(
        "Expected Productivity",
        f"{detail['predicted_productivity']:.1%}",
    )


with d3:

    st.metric(
        "Target",
        f"{detail['targeted_productivity']:.1%}",
    )


with d4:

    st.metric(
        "Target Gap",
        f"{detail['forecast_gap'] * 100:+.1f} pp",
    )


detail_status = detail[
    "risk_status"
]


if detail_status == "High Risk":

    st.error(
        "🔴 High Risk — this team-line should be reviewed before production."
    )

elif detail_status == "Watch":

    st.warning(
        "🟠 Watch — review the planned conditions and monitor this team-line."
    )

else:

    st.success(
        "🟢 Healthy — current low-day risk is below the alert ranges."
    )


st.markdown(
    "**Operating conditions on this forecast record**"
)


condition_columns = [
    (
        "Incentive",
        "incentive",
        "{:.1f}",
    ),
    (
        "WIP",
        "wip",
        "{:.0f}",
    ),
    (
        "Workers",
        "no_of_workers",
        "{:.0f}",
    ),
    (
        "Overtime",
        "over_time",
        "{:.0f}",
    ),
    (
        "SMV",
        "smv",
        "{:.2f}",
    ),
    (
        "Style Changes",
        "no_of_style_change",
        "{:.0f}",
    ),
]


available_conditions = [
    item
    for item in condition_columns
    if item[1] in detail.index
    and pd.notna(
        detail[
            item[1]
        ]
    )
]


if available_conditions:

    cols = st.columns(
        min(
            3,
            len(
                available_conditions
            ),
        )
    )

    for index, (
        label,
        source_column,
        value_format,
    ) in enumerate(
        available_conditions
    ):

        with cols[
            index
            % len(cols)
        ]:

            value = float(
                detail[
                    source_column
                ]
            )

            st.metric(
                label,
                value_format.format(
                    value
                ),
            )


st.info(
    "Next decision step: use **Productivity Drivers** to understand why "
    "the team-line is vulnerable, then use the **What-If Planner** to "
    "test possible operating changes."
)

