

from pathlib import Path
import json

import numpy as np
import pandas as pd


THIS_FILE = Path(__file__).resolve()
DASHBOARD_DIR = THIS_FILE.parents[1]
DATA_DIR = DASHBOARD_DIR / "data"

TEAM_DAILY_PATH = DATA_DIR / "team_daily.csv"
RUNTIME_CONFIG_PATH = DATA_DIR / "forecast_runtime_config.json"


STATUS_ORDER = {
    "Healthy": 0,
    "Watch": 1,
    "High Risk": 2,
}


def _clean_status(value):

    text = str(value).strip().lower()

    if text in {
        "high risk",
        "high",
        "risk",
        "red",
    }:
        return "High Risk"

    if text in {
        "watch",
        "warning",
        "moderate",
        "orange",
        "amber",
    }:
        return "Watch"

    return "Healthy"


def _safe_percent(value):

    value = pd.to_numeric(
        value,
        errors="coerce",
    )

    return value.clip(
        lower=0.0,
        upper=1.0,
    )


def load_risk_data():

    if not TEAM_DAILY_PATH.exists():
        raise FileNotFoundError(
            f"Dashboard risk source not found: {TEAM_DAILY_PATH}"
        )

    df = pd.read_csv(
        TEAM_DAILY_PATH
    )

    required = [
        "team_display",
        "department_display",
        "predicted_productivity",
        "targeted_productivity",
        "forecast_gap",
        "p_alarm",
        "risk_status",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise KeyError(
            f"team_daily.csv is missing Page 3 columns: {missing}"
        )

    # Rolling forecasts during model burn-in are not manager alerts.
    if "is_burn_in" in df.columns:

        burn = (
            df["is_burn_in"]
            .astype(str)
            .str.lower()
            .isin(
                [
                    "true",
                    "1",
                    "yes",
                ]
            )
        )

        df = df.loc[
            ~burn
        ].copy()

    df["p_alarm"] = _safe_percent(
        df["p_alarm"]
    )

    df["predicted_productivity"] = _safe_percent(
        df["predicted_productivity"]
    )

    df["targeted_productivity"] = _safe_percent(
        df["targeted_productivity"]
    )

    df["forecast_gap"] = pd.to_numeric(
        df["forecast_gap"],
        errors="coerce",
    )

    df["risk_status"] = (
        df["risk_status"]
        .map(_clean_status)
    )

    df["risk_rank"] = (
        df["risk_status"]
        .map(STATUS_ORDER)
        .fillna(0)
        .astype(int)
    )

    df = _add_period_columns(
        df
    )

    return df


def _add_period_columns(df):

    result = df.copy()

    # Prefer a genuine calendar date when one exists.
    if "date" in result.columns:

        raw_date = result["date"]

        parsed = pd.to_datetime(
            raw_date,
            errors="coerce",
        )

        valid_ratio = (
            parsed.notna().mean()
            if len(parsed)
            else 0.0
        )

        if parsed.notna().any():

            plausible = (
                parsed.dropna()
                .dt.year
                .between(
                    2010,
                    2030,
                )
                .mean()
            )

        else:
            plausible = 0.0

        if (
            valid_ratio >= 0.80
            and plausible >= 0.80
        ):

            result["_period_key"] = (
                parsed.dt.normalize()
            )

            result["_period_label"] = (
                parsed.dt.strftime(
                    "%d %b %Y"
                )
            )

            result["_period_type"] = "date"

            return result


    # Otherwise use the model's day index/order.
    if "day_index" in result.columns:

        day_index = pd.to_numeric(
            result["day_index"],
            errors="coerce",
        )

    elif "date_or_order" in result.columns:

        day_index = pd.to_numeric(
            result["date_or_order"],
            errors="coerce",
        )

    else:

        day_index = pd.Series(
            np.arange(
                len(result)
            ),
            index=result.index,
            dtype=float,
        )


    result["_period_key"] = day_index

    def make_day_label(value):

        if pd.isna(value):
            return "Unknown"

        if float(value).is_integer():
            return f"Forecast Day {int(value)}"

        return f"Forecast Day {value}"


    result["_period_label"] = (
        day_index.map(
            make_day_label
        )
    )

    result["_period_type"] = "order"

    return result


def available_periods(df=None):

    if df is None:
        df = load_risk_data()

    periods = (
        df[
            [
                "_period_key",
                "_period_label",
            ]
        ]
        .dropna(
            subset=[
                "_period_key",
            ]
        )
        .drop_duplicates()
        .sort_values(
            "_period_key"
        )
        .reset_index(
            drop=True
        )
    )

    return periods


def snapshot(
    period_key,
    department="All",
    df=None,
):

    if df is None:
        df = load_risk_data()

    selected = df.loc[
        df["_period_key"]
        == period_key
    ].copy()

    # Safety for a duplicated source row: retain one row per team-line.
    sort_columns = [
        "team_display",
        "department_display",
    ]

    if "day_index" in selected.columns:
        sort_columns.append(
            "day_index"
        )

    selected = (
        selected.sort_values(
            sort_columns
        )
        .groupby(
            [
                "team_display",
                "department_display",
            ],
            as_index=False,
            sort=False,
        )
        .tail(1)
        .copy()
    )

    if department != "All":

        selected = selected.loc[
            selected["department_display"]
            == department
        ].copy()

    selected = selected.sort_values(
        [
            "risk_rank",
            "p_alarm",
            "forecast_gap",
        ],
        ascending=[
            False,
            False,
            True,
        ],
    ).reset_index(
        drop=True
    )

    return selected


def risk_summary(frame):

    counts = (
        frame["risk_status"]
        .value_counts()
    )

    return {
        "high_risk":
            int(
                counts.get(
                    "High Risk",
                    0,
                )
            ),

        "watch":
            int(
                counts.get(
                    "Watch",
                    0,
                )
            ),

        "healthy":
            int(
                counts.get(
                    "Healthy",
                    0,
                )
            ),

        "average_risk":
            float(
                frame["p_alarm"].mean()
            )
            if len(frame)
            else np.nan,
    }


def calibration_info():

    default = {
        "p_star": None,
        "achievement_low_threshold": None,
        "watch_threshold": None,
    }

    if not RUNTIME_CONFIG_PATH.exists():
        return default

    try:

        config = json.loads(
            RUNTIME_CONFIG_PATH.read_text()
        )

    except Exception:
        return default

    return {
        "p_star":
            config.get("p_star"),

        "achievement_low_threshold":
            config.get(
                "achievement_low_threshold"
            ),

        "watch_threshold":
            config.get(
                "watch_threshold"
            ),
    }

