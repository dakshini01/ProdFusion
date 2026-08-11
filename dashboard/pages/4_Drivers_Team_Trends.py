

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


from utils.driver_engine import (
    available_context,
    latest_snapshot,
    driver_summary,
    driver_curve,
    recent_drift,
    recent_trend,
    movement_summary,
    productivity_direction,
)


st.set_page_config(
    page_title="Drivers & Team Trends",
    page_icon="📈",
    layout="wide",
)


# =============================================================================
# HEADER
# =============================================================================

st.title(
    "Productivity Drivers & Team Trends"
)

st.caption(
    "Understand which operating conditions are associated with productivity "
    "and whether their influence is changing for the selected team."
)


# =============================================================================
# CONTEXT
# =============================================================================

st.subheader(
    "1. Analysis Context"
)


context = available_context()


def team_number(
    label,
):

    digits = "".join(
        character
        for character
        in str(label)
        if character.isdigit()
    )

    return int(
        digits
    ) if digits else 9999


context_col1, context_col2 = st.columns(
    2
)


with context_col1:

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
    )


department_context = context.loc[
    context[
        "department_display"
    ]
    == selected_department
].copy()


with context_col2:

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
        "series"
    ]
)


# =============================================================================
# CURRENT SNAPSHOT
# =============================================================================

snapshot = latest_snapshot(
    series_id
)


st.subheader(
    "2. Current Team Snapshot"
)


expected = float(
    snapshot[
        "predicted_productivity"
    ]
)

target = float(
    snapshot[
        "targeted_productivity"
    ]
)

gap = float(
    snapshot[
        "forecast_gap"
    ]
)

risk = float(
    snapshot[
        "p_alarm"
    ]
)

risk_status = str(
    snapshot[
        "risk_status"
    ]
)


s1, s2, s3, s4 = st.columns(
    4
)


with s1:

    st.metric(
        "Expected Productivity",
        f"{expected:.1%}",
    )


with s2:

    st.metric(
        "Target",
        f"{target:.1%}",
    )


with s3:

    st.metric(
        "Target Gap",
        f"{gap * 100:+.1f} pp",
    )


with s4:

    st.metric(
        "Low-Day Risk",
        f"{risk:.1%}",
    )

    if risk_status == "High Risk":
        st.error(
            "🔴 High Risk"
        )

    elif risk_status == "Watch":
        st.warning(
            "🟠 Watch"
        )

    else:
        st.success(
            "🟢 Healthy"
        )


# =============================================================================
# MAIN DRIVERS
# =============================================================================

st.subheader(
    "3. Main Productivity Drivers"
)


summary = driver_summary(
    series_id
)


def direction_label(
    value,
):

    if value == "Supports":
        return "↑ Supports"

    if value == "Reduces":
        return "↓ Reduces"

    return "↔ Minimal / Mixed"


def format_current(
    row,
):

    value = float(
        row[
            "current_value"
        ]
    )

    unit = str(
        row[
            "unit"
        ]
    )

    driver = row[
        "driver"
    ]

    if driver in {
        "WIP",
        "Workers",
        "Overtime",
        "Style Changes",
    }:

        number = f"{value:.0f}"

    elif driver == "SMV":

        number = f"{value:.2f}"

    else:

        number = f"{value:.1f}"


    if unit and unit != "nan":
        return f"{number} {unit}"

    return number


driver_table = pd.DataFrame(
    {
        "Driver":
            summary[
                "driver"
            ],

        "Influence":
            summary[
                "direction"
            ]
            .map(
                direction_label
            ),

        "Strength":
            summary[
                "strength"
            ],

        "Uncertainty":
            summary[
                "confidence"
            ],

        "Typical-Range Effect":
            summary[
                "effect_mean"
            ]
            .map(
                lambda value:
                    f"{value * 100:+.1f} pp"
            ),

        "Current Condition":
            summary.apply(
                format_current,
                axis=1,
            ),
    }
)


st.dataframe(
    driver_table,
    use_container_width=True,
    hide_index=True,
)


st.caption(
    "Strength ranks the six drivers by the size of their modeled productivity "
    "change across a realistic observed range for this team-line. "
    "'Clear' means the 80% uncertainty interval for that range effect stays "
    "on one side of zero."
)


# =============================================================================
# DRIVER RESPONSE CURVE
# =============================================================================

st.subheader(
    "4. Driver Response & Uncertainty"
)


driver_options = (
    summary[
        "driver"
    ]
    .tolist()
)


selected_driver = st.selectbox(
    "Driver",
    driver_options,
)


curve = driver_curve(
    series_id,
    selected_driver,
)


if curve.empty:

    st.warning(
        "No response curve is available for this driver."
    )

else:

    figure = go.Figure()


    # 95% uncertainty
    figure.add_trace(
        go.Scatter(
            x=curve[
                "driver_value"
            ],
            y=curve[
                "ci95_upper"
            ],
            mode="lines",
            line=dict(
                width=0
            ),
            hoverinfo="skip",
            showlegend=False,
        )
    )


    figure.add_trace(
        go.Scatter(
            x=curve[
                "driver_value"
            ],
            y=curve[
                "ci95_lower"
            ],
            mode="lines",
            line=dict(
                width=0
            ),
            fill="tonexty",
            fillcolor="rgba(148,163,184,0.18)",
            name="95% uncertainty",
            hoverinfo="skip",
        )
    )


    # 80% uncertainty
    figure.add_trace(
        go.Scatter(
            x=curve[
                "driver_value"
            ],
            y=curve[
                "ci80_upper"
            ],
            mode="lines",
            line=dict(
                width=0
            ),
            hoverinfo="skip",
            showlegend=False,
        )
    )


    figure.add_trace(
        go.Scatter(
            x=curve[
                "driver_value"
            ],
            y=curve[
                "ci80_lower"
            ],
            mode="lines",
            line=dict(
                width=0
            ),
            fill="tonexty",
            fillcolor="rgba(100,116,139,0.28)",
            name="80% uncertainty",
            hoverinfo="skip",
        )
    )


    # Expected response
    figure.add_trace(
        go.Scatter(
            x=curve[
                "driver_value"
            ],
            y=curve[
                "expected_productivity"
            ],
            mode="lines",
            name="Expected productivity",
            line=dict(
                width=3
            ),
            hovertemplate=(
                f"{selected_driver}: "
                "%{x:.2f}<br>"
                "Expected productivity: "
                "%{y:.1%}"
                "<extra></extra>"
            ),
        )
    )


    current_value = float(
        curve[
            "current_value"
        ]
        .iloc[0]
    )


    figure.add_vline(
        x=current_value,
        line_dash="dash",
        annotation_text="Current value",
        annotation_position="top",
    )


    # Original paper's incentive split is shown as a reference only.
    if selected_driver == "Incentive":

        xmin = float(
            curve[
                "driver_value"
            ].min()
        )

        xmax = float(
            curve[
                "driver_value"
            ].max()
        )

        if (
            69.5 >= xmin
            and 69.5 <= xmax
        ):

            figure.add_vline(
                x=69.5,
                line_dash="dot",
                annotation_text="Original study: 69.5 BDT",
                annotation_position="bottom",
            )


    unit = str(
        curve[
            "unit"
        ].iloc[0]
    )

    x_title = (
        selected_driver
        if not unit
        or unit == "nan"
        else f"{selected_driver} ({unit})"
    )


    figure.update_layout(
        height=440,
        margin=dict(
            l=20,
            r=20,
            t=30,
            b=20,
        ),
        xaxis_title=x_title,
        yaxis_title="Expected Productivity",
        yaxis_tickformat=".0%",
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
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


    selected_summary = (
        summary.loc[
            summary[
                "driver"
            ]
            == selected_driver
        ]
        .iloc[0]
    )


    st.caption(
        f"Across the displayed realistic range, the estimated mean change is "
        f"{selected_summary['effect_mean'] * 100:+.1f} percentage points "
        f"(80% uncertainty: "
        f"{selected_summary['effect80_lower'] * 100:+.1f} to "
        f"{selected_summary['effect80_upper'] * 100:+.1f} pp). "
        "This is a model-based association, not a guaranteed causal effect."
    )


# =============================================================================
# TEAM-SPECIFIC INTERPRETATION
# =============================================================================

st.subheader(
    "5. Team-Specific Interpretation"
)


clear_support = (
    summary.loc[
        (
            summary[
                "direction"
            ]
            == "Supports"
        )
    ]
    .sort_values(
        "effect_mean",
        ascending=False,
    )
)


clear_reduction = (
    summary.loc[
        (
            summary[
                "direction"
            ]
            == "Reduces"
        )
    ]
    .sort_values(
        "effect_mean",
        ascending=True,
    )
)


interpretation_parts = []


if not clear_support.empty:

    row = clear_support.iloc[0]

    interpretation_parts.append(
        f"**Largest modeled support:** {row['driver']} "
        f"({row['effect_mean'] * 100:+.1f} pp across its typical range)."
    )


if not clear_reduction.empty:

    row = clear_reduction.iloc[0]

    interpretation_parts.append(
        f"**Largest modeled downward association:** {row['driver']} "
        f"({row['effect_mean'] * 100:+.1f} pp across its typical range)."
    )


uncertain = (
    summary.loc[
        summary[
            "confidence"
        ]
        == "Uncertain",
        "driver",
    ]
    .tolist()
)


if uncertain:

    interpretation_parts.append(
        "**Use extra caution interpreting:** "
        + ", ".join(
            uncertain
        )
        + ", because their 80% effect ranges include zero."
    )


if interpretation_parts:

    for text in interpretation_parts:

        st.markdown(
            f"- {text}"
        )

else:

    st.write(
        "No clear positive or negative driver pattern is available for this team-line."
    )


# =============================================================================
# CHANGING TEAM BEHAVIOR
# =============================================================================

st.subheader(
    "6. Changing Team Behavior"
)


drift = recent_drift(
    series_id,
    recent_n=30,
)


default_drift_drivers = [
    driver
    for driver in [
        "Incentive",
        "WIP",
        "Workers",
    ]
    if driver in drift[
        "driver"
    ].unique()
]


selected_drift_drivers = st.multiselect(
    "Show changing influence for",
    sorted(
        drift[
            "driver"
        ]
        .dropna()
        .unique()
        .tolist()
    ),
    default=default_drift_drivers,
)


if selected_drift_drivers:

    drift_plot = drift.loc[
        drift[
            "driver"
        ]
        .isin(
            selected_drift_drivers
        )
    ].copy()


    drift_fig = go.Figure()


    for driver in selected_drift_drivers:

        group = drift_plot.loc[
            drift_plot[
                "driver"
            ]
            == driver
        ].sort_values(
            "day_index"
        )


        drift_fig.add_trace(
            go.Scatter(
                x=group[
                    "day_index"
                ],
                y=group[
                    "influence"
                ],
                mode="lines",
                name=driver,
                hovertemplate=(
                    "Day %{x}<br>"
                    "Influence: %{y:.3f}"
                    "<extra>"
                    + driver
                    + "</extra>"
                ),
            )
        )


    drift_fig.add_hline(
        y=0,
        line_dash="dot",
    )


    # Mark observed style-change days.
    if "no_of_style_change" in drift.columns:

        style_days = (
            drift.loc[
                pd.to_numeric(
                    drift[
                        "no_of_style_change"
                    ],
                    errors="coerce",
                )
                .fillna(0)
                > 0,
                "day_index",
            ]
            .dropna()
            .unique()
        )

        for style_day in style_days:

            drift_fig.add_vline(
                x=float(
                    style_day
                ),
                line_dash="dot",
                opacity=0.25,
            )


    drift_fig.update_layout(
        height=400,
        margin=dict(
            l=20,
            r=20,
            t=20,
            b=20,
        ),
        xaxis_title="Production Day",
        yaxis_title="Changing Influence",
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )


    st.plotly_chart(
        drift_fig,
        use_container_width=True,
        config={
            "displayModeBar": False,
        },
    )


    st.caption(
        "Above zero indicates that the time-varying part of a driver is "
        "supporting productivity; below zero indicates downward pressure. "
        "Vertical dotted markers indicate recorded style-change days. "
        "For incentive, the full relationship also includes the non-linear "
        "response shown in the driver-response chart."
    )


movement = movement_summary(
    series_id,
    recent_n=12,
)


if not movement.empty:

    movement_display = pd.DataFrame(
        {
            "Driver":
                movement[
                    "driver"
                ],

            "Recent Direction":
                movement[
                    "direction"
                ],

            "Recent Movement":
                movement[
                    "movement_label"
                ],
        }
    )


    st.dataframe(
        movement_display,
        use_container_width=True,
        hide_index=True,
    )


    st.caption(
        "'Recent Movement' is a relative within-team drift indicator based on "
        "the range of each changing influence over the recent window. "
        "It is not a formal change-point flag."
    )


# =============================================================================
# PRODUCTIVITY TREND
# =============================================================================

st.subheader(
    "7. Productivity Trend"
)


trend = recent_trend(
    series_id,
    recent_n=30,
)


trend_fig = go.Figure()


trend_fig.add_trace(
    go.Scatter(
        x=trend[
            "day_index"
        ],
        y=trend[
            "actual_productivity"
        ],
        mode="lines+markers",
        name="Actual Productivity",
        hovertemplate=(
            "Day %{x}<br>"
            "Actual: %{y:.1%}"
            "<extra></extra>"
        ),
    )
)


trend_fig.add_trace(
    go.Scatter(
        x=trend[
            "day_index"
        ],
        y=trend[
            "predicted_productivity"
        ],
        mode="lines",
        name="Expected Productivity",
        hovertemplate=(
            "Day %{x}<br>"
            "Expected: %{y:.1%}"
            "<extra></extra>"
        ),
    )
)


if "targeted_productivity" in trend.columns:

    trend_fig.add_trace(
        go.Scatter(
            x=trend[
                "day_index"
            ],
            y=trend[
                "targeted_productivity"
            ],
            mode="lines",
            name="Target",
            line=dict(
                dash="dash"
            ),
            hovertemplate=(
                "Day %{x}<br>"
                "Target: %{y:.1%}"
                "<extra></extra>"
            ),
        )
    )


if "no_of_style_change" in trend.columns:

    style_days = (
        trend.loc[
            pd.to_numeric(
                trend[
                    "no_of_style_change"
                ],
                errors="coerce",
            )
            .fillna(0)
            > 0,
            "day_index",
        ]
        .dropna()
        .unique()
    )

    for style_day in style_days:

        trend_fig.add_vline(
            x=float(
                style_day
            ),
            line_dash="dot",
            opacity=0.25,
        )


trend_fig.update_layout(
    height=400,
    margin=dict(
        l=20,
        r=20,
        t=20,
        b=20,
    ),
    xaxis_title="Production Day",
    yaxis_title="Productivity",
    yaxis_tickformat=".0%",
    hovermode="x unified",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="left",
        x=0,
    ),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
)


st.plotly_chart(
    trend_fig,
    use_container_width=True,
    config={
        "displayModeBar": False,
    },
)


direction = productivity_direction(
    series_id,
    recent_n=7,
)

st.write(
    f"**Recent productivity direction:** {direction}"
)


# =============================================================================
# MANAGEMENT SUMMARY
# =============================================================================

st.subheader(
    "8. Management Summary"
)


top_driver = (
    summary.sort_values(
        "strength_rank"
    )
    .iloc[0]
)


summary_text = (
    f"For **{selected_team} · {selected_department}**, "
    f"the largest modeled driver movement across its typical range is "
    f"**{top_driver['driver']}** "
    f"({top_driver['effect_mean'] * 100:+.1f} pp). "
    f"The recent expected-productivity direction is **{direction.lower()}**, "
    f"and the current low-day status is **{risk_status}**."
)


st.info(
    summary_text
)


st.caption(
    "Next step: use Page 5 — What-If & Decision Planner — to test practical "
    "changes to incentive, WIP and staffing before making an operating decision."
)

