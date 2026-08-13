
from pathlib import Path
import json
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Production Intelligence",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"

LATEST_PATH = DATA_DIR / "team_latest.csv"
DAILY_PATH = DATA_DIR / "team_daily.csv"
META_PATH = DATA_DIR / "dashboard_meta.json"


@st.cache_data
def load_dashboard_data():
    latest = pd.read_csv(LATEST_PATH)
    daily = pd.read_csv(DAILY_PATH)

    if META_PATH.exists():
        with open(META_PATH, "r", encoding="utf-8") as f:
            meta = json.load(f)
    else:
        meta = {}

    return latest, daily, meta


latest, daily, meta = load_dashboard_data()

for frame in (latest, daily):
    for col in [
        "predicted_productivity",
        "targeted_productivity",
        "forecast_gap",
        "p_alarm",
    ]:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")

if "date" in daily.columns:
    daily["date"] = pd.to_datetime(daily["date"], errors="coerce")
if "date" in latest.columns:
    latest["date"] = pd.to_datetime(latest["date"], errors="coerce")


st.markdown(
    """
    <style>
    .block-container {
        max-width: 1500px;
        padding-top: 1.4rem;
        padding-bottom: 2.5rem;
    }

    [data-testid="stSidebar"] {
        min-width: 250px;
    }

    [data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 14px 16px;
        min-height: 132px;
    }

    [data-testid="stMetricLabel"] p {
        font-weight: 700;
        color: #64748B;
    }

    [data-testid="stWidgetLabel"] p {
        color: #475569;
        font-weight: 650;
    }

    div[data-testid="stDataFrame"] {
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        overflow: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------
with st.sidebar:
    st.title("🏭 Production Intelligence")
    st.caption("Decision Support Dashboard")

    st.markdown("### Daily Overview")
    st.markdown("Team Forecasts")
    st.markdown("Risk & Alerts")
    st.markdown("Productivity Drivers")
    st.markdown("What-If Planner")
    st.markdown("Team Trends")
    st.divider()
    st.markdown("Forecast Reliability")
    st.markdown("About")


# ---------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------
header_left, header_right = st.columns([5, 1.2], vertical_alignment="center")

with header_left:
    st.title("Production Intelligence Dashboard")
    st.caption(
        "Latest available one-day-ahead productivity outlook and operational priorities"
    )

with header_right:
    st.download_button(
        "⬇ Export Report",
        data=latest.to_csv(index=False).encode("utf-8"),
        file_name="daily_productivity_outlook.csv",
        mime="text/csv",
        use_container_width=True,
    )


# ---------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------
f1, f2, f3 = st.columns([1.4, 1.4, 2.2])

with f1:
    department_options = ["All Departments"] + sorted(
        latest["department_display"].dropna().astype(str).unique().tolist()
    )
    selected_department = st.selectbox("Department", department_options)

team_base = latest.copy()
if selected_department != "All Departments":
    team_base = team_base[
        team_base["department_display"].astype(str) == selected_department
    ]

def team_sort_key(label):
    match = re.search(r"(\d+)", str(label))
    return int(match.group(1)) if match else 9999

with f2:
    team_options = ["All Teams"] + sorted(
        team_base["team_display"].dropna().astype(str).unique().tolist(),
        key=team_sort_key,
    )
    selected_team = st.selectbox("Team", team_options)

filtered = latest.copy()

if selected_department != "All Departments":
    filtered = filtered[
        filtered["department_display"].astype(str) == selected_department
    ]

if selected_team != "All Teams":
    filtered = filtered[
        filtered["team_display"].astype(str) == selected_team
    ]

if filtered.empty:
    st.warning("No team-line records match the selected filters.")
    st.stop()

with f3:
    valid_dates = pd.to_datetime(filtered["date"], errors="coerce")
    if valid_dates.notna().any():
        display_period = valid_dates.max().strftime("%d %b %Y")
    else:
        display_period = "Latest available model period"

    st.text_input(
        "Forecast period",
        value=display_period,
        disabled=True,
    )


# ---------------------------------------------------------------------
# KPI definitions and values
# ---------------------------------------------------------------------
expected_productivity = filtered["predicted_productivity"].mean()

high_risk_count = int(
    (filtered["risk_status"].astype(str) == "High Risk").sum()
)

below_target_count = int(
    (filtered["predicted_productivity"] < filtered["targeted_productivity"]).sum()
)

average_low_day_risk = filtered["p_alarm"].mean()

highest_row = filtered.sort_values(
    ["p_alarm", "forecast_gap"],
    ascending=[False, True],
).iloc[0]

highest_team = str(highest_row["team_display"])
highest_department = str(highest_row["department_display"])
highest_risk = float(highest_row["p_alarm"])


k1, k2, k3, k4, k5 = st.columns(5)

with k1:
    st.metric(
        "Expected Productivity",
        f"{expected_productivity:.1%}",
        help=(
            "Mean of the latest one-day-ahead productivity forecasts "
            "for the selected team-lines."
        ),
    )
    st.caption("Mean latest forecast")

with k2:
    st.metric(
        "High-Risk Team-Lines",
        f"{high_risk_count}",
        help=(
            "Number of selected team × department lines whose low-day "
            "probability is at or above the calibrated alarm threshold."
        ),
    )
    st.caption(f"of {len(filtered)} selected team-lines")

with k3:
    st.metric(
        "Below Target",
        f"{below_target_count} / {len(filtered)}",
        help=(
            "Number of selected team-lines whose latest forecast is below "
            "their targeted productivity."
        ),
    )
    st.caption("Forecast < target")

with k4:
    st.metric(
        "Average Low-Day Risk",
        f"{average_low_day_risk:.1%}",
        help=(
            "Mean model-estimated probability of a low-productivity day "
            "across the selected team-lines."
        ),
    )
    st.caption("Mean risk across selection")

with k5:
    st.metric(
        "Highest-Priority Team",
        highest_team,
        help=(
            "The selected team-line with the largest model-estimated "
            "low-day probability."
        ),
    )
    st.caption(f"{highest_department} · {highest_risk:.1%} risk")


# ---------------------------------------------------------------------
# Management outlook
# ---------------------------------------------------------------------
focus = str(highest_row.get("suggested_focus", "Review team conditions"))

if high_risk_count == 0:
    outlook = (
        f"No selected team-lines are currently above the high-risk threshold. "
        f"The highest current risk is {highest_risk:.1%} for "
        f"{highest_team} · {highest_department}."
    )
elif high_risk_count == 1:
    outlook = (
        f"1 selected team-line requires attention. "
        f"{highest_team} · {highest_department} has the highest low-day risk "
        f"at {highest_risk:.1%}. Suggested review: {focus}."
    )
else:
    outlook = (
        f"{high_risk_count} selected team-lines require attention. "
        f"{highest_team} · {highest_department} has the highest low-day risk "
        f"at {highest_risk:.1%}. Suggested first review: {focus}."
    )

st.info(f"**Latest management outlook — {display_period}:** {outlook}")


# ---------------------------------------------------------------------
# Decision panels
# ---------------------------------------------------------------------
left_panel, right_panel = st.columns([1.45, 1.0], gap="large")


# ----------------------------
# Priority table
# ----------------------------
with left_panel:
    st.subheader("Priority Teams")
    st.caption(
        "Latest team-lines ranked first by low-day risk and then by target gap."
    )

    priority = filtered.sort_values(
        ["p_alarm", "forecast_gap"],
        ascending=[False, True],
    ).reset_index(drop=True).copy()

    priority["Priority"] = np.arange(1, len(priority) + 1)

    def status_label(status):
        if status == "High Risk":
            return "🔴 High Risk"
        if status == "Watch":
            return "🟠 Watch"
        return "🟢 Healthy"

    priority["Status"] = priority["risk_status"].astype(str).map(status_label)
    priority["Forecast"] = priority["predicted_productivity"] * 100
    priority["Target"] = priority["targeted_productivity"] * 100
    priority["Gap"] = priority["forecast_gap"] * 100
    priority["Low-Day Risk"] = priority["p_alarm"] * 100

    priority_table = priority[
        [
            "Priority",
            "team_display",
            "department_display",
            "Forecast",
            "Target",
            "Gap",
            "Low-Day Risk",
            "Status",
            "suggested_focus",
        ]
    ].rename(
        columns={
            "team_display": "Team",
            "department_display": "Department",
            "suggested_focus": "Suggested Focus",
        }
    ).head(8)

    st.dataframe(
        priority_table,
        hide_index=True,
        use_container_width=True,
        height=430,
        column_config={
            "Priority": st.column_config.NumberColumn(
                "Priority", format="%d", width="small"
            ),
            "Team": st.column_config.TextColumn("Team", width="small"),
            "Department": st.column_config.TextColumn(
                "Department", width="small"
            ),
            "Forecast": st.column_config.NumberColumn(
                "Forecast", format="%.1f%%", width="small"
            ),
            "Target": st.column_config.NumberColumn(
                "Target", format="%.1f%%", width="small"
            ),
            "Gap": st.column_config.NumberColumn(
                "Gap", format="%+.1f pp", width="small"
            ),
            "Low-Day Risk": st.column_config.ProgressColumn(
                "Low-Day Risk",
                format="%.0f%%",
                min_value=0,
                max_value=100,
                width="medium",
            ),
            "Status": st.column_config.TextColumn("Status", width="medium"),
            "Suggested Focus": st.column_config.TextColumn(
                "Suggested Focus", width="medium"
            ),
        },
    )


# ----------------------------
# Risk heatmap
# ----------------------------
with right_panel:
    st.subheader("Recent Low-Day Risk")
    st.caption(
        "Last seven available production periods for the highest-risk team-lines."
    )

    heat = daily.copy()

    if selected_department != "All Departments":
        heat = heat[
            heat["department_display"].astype(str) == selected_department
        ]

    if selected_team != "All Teams":
        heat = heat[
            heat["team_display"].astype(str) == selected_team
        ]

    if "is_burn_in" in heat.columns:
        burn = (
            heat["is_burn_in"]
            .astype(str)
            .str.lower()
            .isin(["true", "1"])
        )
        heat = heat[~burn]

    heat["Team Line"] = (
        heat["team_display"].astype(str)
        + " · "
        + heat["department_display"].astype(str)
    )

    parsed_dates = pd.to_datetime(heat["date"], errors="coerce")
    real_date_mask = parsed_dates.notna()

    if real_date_mask.any():
        heat["_period_sort"] = parsed_dates
        recent_periods = (
            heat["_period_sort"]
            .dropna()
            .drop_duplicates()
            .sort_values()
            .tail(7)
        )
        heat = heat[heat["_period_sort"].isin(recent_periods)].copy()
        heat["Period"] = heat["_period_sort"].dt.strftime("%d %b")
        ordered_periods = [d.strftime("%d %b") for d in recent_periods]
    else:
        heat["_period_sort"] = pd.to_numeric(
            heat["date_or_order"], errors="coerce"
        )
        recent_periods = (
            heat["_period_sort"]
            .dropna()
            .drop_duplicates()
            .sort_values()
            .tail(7)
        )
        heat = heat[heat["_period_sort"].isin(recent_periods)].copy()
        heat["Period"] = heat["_period_sort"].map(
            lambda x: f"Day {int(x)}" if pd.notna(x) else ""
        )
        ordered_periods = [f"Day {int(x)}" for x in recent_periods]

    if heat.empty:
        st.info("No recent risk history is available for this selection.")
    else:
        line_order = (
            heat.groupby("Team Line")["p_alarm"]
            .mean()
            .sort_values(ascending=False)
            .head(12)
            .index
            .tolist()
        )
        heat = heat[heat["Team Line"].isin(line_order)]

        risk_matrix = heat.pivot_table(
            index="Team Line",
            columns="Period",
            values="p_alarm",
            aggfunc="mean",
        )

        present_columns = [
            c for c in ordered_periods if c in risk_matrix.columns
        ]
        risk_matrix = risk_matrix.reindex(
            index=line_order,
            columns=present_columns,
        )

        risk_pct = risk_matrix * 100

        text_values = np.empty(risk_pct.shape, dtype=object)
        for i in range(risk_pct.shape[0]):
            for j in range(risk_pct.shape[1]):
                value = risk_pct.iat[i, j]
                text_values[i, j] = (
                    "" if pd.isna(value) else f"{value:.0f}%"
                )

        p_star = float(meta.get("p_star", 0.50))
        watch_threshold = float(
            meta.get("watch_threshold", 0.75 * p_star)
        )

        colorscale = [
            [0.00, "#DFF3E8"],
            [max(0.0, min(watch_threshold, 1.0)), "#DFF3E8"],
            [max(0.0, min(watch_threshold + 0.001, 1.0)), "#FFF1C7"],
            [max(0.0, min(p_star, 1.0)), "#FFF1C7"],
            [max(0.0, min(p_star + 0.001, 1.0)), "#F7C9C9"],
            [1.00, "#D95B5B"],
        ]

        fig = go.Figure(
            data=go.Heatmap(
                z=risk_pct.values,
                x=risk_pct.columns.tolist(),
                y=risk_pct.index.tolist(),
                zmin=0,
                zmax=100,
                colorscale=colorscale,
                text=text_values,
                texttemplate="%{text}",
                hovertemplate=(
                    "<b>%{y}</b>"
                    "<br>%{x}"
                    "<br>Low-day risk: %{z:.1f}%"
                    "<extra></extra>"
                ),
                colorbar=dict(
                    title="Risk",
                    ticksuffix="%",
                    thickness=12,
                    len=0.75,
                ),
            )
        )

        fig.update_layout(
            height=max(360, 44 * len(risk_pct.index) + 120),
            margin=dict(l=8, r=8, t=8, b=8),
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",
            font=dict(size=11, color="#334155"),
            xaxis=dict(
                title="",
                tickangle=-30,
                tickfont=dict(color="#64748B"),
            ),
            yaxis=dict(
                title="",
                tickfont=dict(color="#334155"),
            ),
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            theme=None,
            config={"displayModeBar": False},
        )

        st.caption(
            f"🟢 Healthy  ·  🟠 Watch  ·  🔴 High Risk  |  "
            f"High-risk threshold: {p_star:.1%}"
        )


st.caption(
    "Decision-support output only. Suggested focus is a triage cue based on "
    "current operating values and should be combined with production knowledge."
)
