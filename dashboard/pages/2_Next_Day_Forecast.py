

from pathlib import Path
import sys

import pandas as pd
import streamlit as st


PAGE_FILE = Path(__file__).resolve()
DASHBOARD_DIR = PAGE_FILE.parents[1]

if str(DASHBOARD_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(DASHBOARD_DIR),
    )


from utils.forecast_engine import (
    available_context,
    defaults_for_series,
    default_forecast_date,
    forecast_next_day,
)


st.set_page_config(
    page_title="Next-Day Forecast",
    page_icon="↗",
    layout="wide",
)


# -----------------------------------------------------------------------------
# Page header
# -----------------------------------------------------------------------------

st.title("Next-Day Productivity Forecast")

st.caption(
    "Enter the planned operating conditions for one team and estimate "
    "the next production day's productivity, uncertainty and low-day risk."
)


# -----------------------------------------------------------------------------
# Context selectors
# -----------------------------------------------------------------------------

context = available_context()

context["team"] = (
    context["team"]
    .astype(str)
)


def team_sort_key(
    team_label,
):

    try:
        return int(
            str(team_label)
            .replace(
                "Team ",
                "",
            )
        )

    except Exception:
        return 9999


st.subheader("1. Forecast Context")

c1, c2, c3 = st.columns(
    [1.2, 1.2, 1.0]
)


with c1:

    department_options = sorted(
        context[
            "department_display"
        ]
        .dropna()
        .unique()
        .tolist()
    )

    selected_department = st.selectbox(
        "Department",
        department_options,
    )


department_context = context.loc[
    context["department_display"]
    == selected_department
].copy()


with c2:

    team_options = sorted(
        department_context[
            "team_display"
        ]
        .dropna()
        .unique()
        .tolist(),
        key=team_sort_key,
    )

    selected_team = st.selectbox(
        "Team",
        team_options,
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
        help=(
            "The planned production date. "
            "The model uses a Sat–Thu production calendar."
        ),
    )


defaults = defaults_for_series(
    series_id
)


# -----------------------------------------------------------------------------
# Seven manager inputs
# -----------------------------------------------------------------------------

st.subheader("2. Tomorrow's Operating Plan")

st.caption(
    "The fields start from the latest observed values for the selected "
    "team-line. Change them to match the planned production conditions."
)


with st.form(
    "next_day_forecast_form"
):

    row1_a, row1_b, row1_c, row1_d = st.columns(
        4
    )

    with row1_a:

        targeted_productivity = st.number_input(
            "Target Productivity",
            min_value=0.0,
            max_value=1.0,
            value=float(
                defaults[
                    "targeted_productivity"
                ]
            ),
            step=0.01,
            format="%.2f",
            help="0.80 means an 80% productivity target.",
        )


    with row1_b:

        smv = st.number_input(
            "SMV",
            min_value=0.0,
            value=float(
                defaults["smv"]
            ),
            step=0.1,
            help="Standard Minute Value / work-content measure.",
        )


    with row1_c:

        wip = st.number_input(
            "Work in Progress (WIP)",
            min_value=0.0,
            value=float(
                defaults["wip"]
            ),
            step=10.0,
        )


    with row1_d:

        over_time = st.number_input(
            "Overtime (minutes)",
            min_value=0.0,
            value=float(
                defaults["over_time"]
            ),
            step=60.0,
        )


    row2_a, row2_b, row2_c, row2_d = st.columns(
        4
    )


    with row2_a:

        incentive = st.number_input(
            "Incentive",
            min_value=0.0,
            value=float(
                defaults["incentive"]
            ),
            step=5.0,
        )


    with row2_b:

        no_of_workers = st.number_input(
            "Number of Workers",
            min_value=0.0,
            value=float(
                defaults[
                    "no_of_workers"
                ]
            ),
            step=1.0,
        )


    with row2_c:

        no_of_style_change = st.number_input(
            "Style Changes",
            min_value=0,
            value=int(
                defaults[
                    "no_of_style_change"
                ]
            ),
            step=1,
        )


    with row2_d:

        st.write("")
        st.write("")

        submitted = st.form_submit_button(
            "Generate Forecast",
            type="primary",
            use_container_width=True,
        )


# -----------------------------------------------------------------------------
# Forecast result
# -----------------------------------------------------------------------------

if submitted:

    try:

        result = forecast_next_day(
            series_id=series_id,
            forecast_date=forecast_date,
            targeted_productivity=targeted_productivity,
            smv=smv,
            wip=wip,
            over_time=over_time,
            incentive=incentive,
            no_of_workers=no_of_workers,
            no_of_style_change=no_of_style_change,
        )

    except Exception as exc:

        st.error(
            f"Forecast could not be generated: {exc}"
        )

        st.stop()


    st.divider()

    st.subheader("3. Forecast Result")


    expected = result[
        "expected_productivity"
    ]

    target = result[
        "target_productivity"
    ]

    gap_pp = (
        result[
            "forecast_gap"
        ]
        * 100
    )

    low_risk = result[
        "low_day_risk"
    ]

    risk_status = result[
        "risk_status"
    ]


    r1, r2, r3, r4 = st.columns(
        4
    )


    with r1:

        st.metric(
            "Expected Productivity",
            f"{expected:.1%}",
        )


    with r2:

        st.metric(
            "Target Productivity",
            f"{target:.1%}",
        )


    with r3:

        st.metric(
            "Forecast Gap",
            f"{gap_pp:+.1f} pp",
        )


    with r4:

        st.metric(
            "Low-Day Risk",
            f"{low_risk:.1%}",
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


    st.subheader(
        "4. Forecast Uncertainty"
    )


    u1, u2 = st.columns(
        2
    )


    with u1:

        st.info(
            "**80% likely range**\n\n"
            f"{result['pi80_lower']:.1%} – "
            f"{result['pi80_upper']:.1%}"
        )


    with u2:

        st.info(
            "**95% likely range**\n\n"
            f"{result['pi95_lower']:.1%} – "
            f"{result['pi95_upper']:.1%}"
        )


    st.subheader(
        "5. Manager Interpretation"
    )


    if gap_pp < 0:

        target_sentence = (
            f"The expected productivity is "
            f"{abs(gap_pp):.1f} percentage points below "
            f"the {target:.1%} target."
        )

    elif gap_pp > 0:

        target_sentence = (
            f"The expected productivity is "
            f"{gap_pp:.1f} percentage points above "
            f"the {target:.1%} target."
        )

    else:

        target_sentence = (
            "The expected productivity matches the target."
        )


    if risk_status == "High Risk":

        risk_sentence = (
            "The plan is classified as High Risk for a low-productivity day."
        )

    elif risk_status == "Watch":

        risk_sentence = (
            "The plan is in the Watch range and should be reviewed before production."
        )

    else:

        risk_sentence = (
            "The current plan is in the Healthy risk range."
        )


    st.write(
        f"**{result['team']} · {result['department']} — "
        f"{forecast_date.strftime('%d %b %Y')}**"
    )

    st.write(
        target_sentence
        + " "
        + risk_sentence
    )


    st.caption(
        "The 80% and 95% ranges express forecast uncertainty. "
        "The low-day probability is the fused model's probability of "
        "falling below the calibrated low-productivity threshold."
    )

else:

    st.info(
        "Enter or review the seven planned operating values, then click "
        "**Generate Forecast**."
    )

