

from pathlib import Path

import numpy as np
import pandas as pd


THIS_FILE = Path(__file__).resolve()
DASHBOARD_DIR = THIS_FILE.parents[1]
DATA_DIR = DASHBOARD_DIR / "data"

SUMMARY_PATH = DATA_DIR / "04_driver_summary.csv"
CURVES_PATH = DATA_DIR / "04_driver_curves_latest.csv"
DRIFT_PATH = DATA_DIR / "04_driver_drift.csv"
TEAM_DAILY_PATH = DATA_DIR / "team_daily.csv"


def _load():

    required = [
        SUMMARY_PATH,
        CURVES_PATH,
        DRIFT_PATH,
        TEAM_DAILY_PATH,
    ]

    missing = [
        path
        for path in required
        if not path.exists()
    ]

    if missing:

        raise FileNotFoundError(
            "Page 4 data artifact(s) missing:\n"
            + "\n".join(
                str(path)
                for path in missing
            )
        )

    summary = pd.read_csv(
        SUMMARY_PATH
    )

    curves = pd.read_csv(
        CURVES_PATH
    )

    drift = pd.read_csv(
        DRIFT_PATH
    )

    daily = pd.read_csv(
        TEAM_DAILY_PATH
    )

    for frame in [
        summary,
        curves,
        drift,
        daily,
    ]:

        if "series" in frame.columns:
            frame["series"] = (
                frame["series"]
                .astype(str)
            )

    return (
        summary,
        curves,
        drift,
        daily,
    )


SUMMARY, CURVES, DRIFT, DAILY = _load()


def available_context():

    context = (
        SUMMARY[
            [
                "series",
                "team_display",
                "department_display",
            ]
        ]
        .drop_duplicates()
        .copy()
    )

    return context.reset_index(
        drop=True
    )


def latest_snapshot(
    series_id,
):

    frame = DAILY.loc[
        DAILY["series"].astype(str)
        == str(series_id)
    ].copy()

    if frame.empty:
        raise KeyError(
            f"No daily data for series {series_id}"
        )

    if "is_burn_in" in frame.columns:

        burn = (
            frame[
                "is_burn_in"
            ]
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

        non_burn = frame.loc[
            ~burn
        ]

        if not non_burn.empty:
            frame = non_burn.copy()

    if "day_index" in frame.columns:

        frame = frame.sort_values(
            "day_index"
        )

    return frame.iloc[-1]


def driver_summary(
    series_id,
):

    frame = SUMMARY.loc[
        SUMMARY["series"]
        == str(series_id)
    ].copy()

    return frame.sort_values(
        "strength_rank"
    ).reset_index(
        drop=True
    )


def driver_curve(
    series_id,
    driver,
):

    frame = CURVES.loc[
        (
            CURVES["series"]
            == str(series_id)
        )
        &
        (
            CURVES["driver"]
            == driver
        )
    ].copy()

    return frame.sort_values(
        "driver_value"
    ).reset_index(
        drop=True
    )


def recent_drift(
    series_id,
    recent_n=30,
):

    frame = DRIFT.loc[
        DRIFT["series"]
        == str(series_id)
    ].copy()

    if frame.empty:
        return frame

    if "is_burn_in" in frame.columns:

        burn = (
            frame[
                "is_burn_in"
            ]
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

        frame = frame.loc[
            ~burn
        ].copy()

    unique_days = (
        frame["day_index"]
        .dropna()
        .sort_values()
        .unique()
    )

    keep_days = unique_days[
        -recent_n:
    ]

    return (
        frame.loc[
            frame["day_index"]
            .isin(
                keep_days
            )
        ]
        .sort_values(
            [
                "day_index",
                "driver",
            ]
        )
        .reset_index(
            drop=True
        )
    )


def recent_trend(
    series_id,
    recent_n=30,
):

    frame = DAILY.loc[
        DAILY["series"].astype(str)
        == str(series_id)
    ].copy()

    if frame.empty:
        return frame

    if "is_burn_in" in frame.columns:

        burn = (
            frame[
                "is_burn_in"
            ]
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

        non_burn = frame.loc[
            ~burn
        ]

        if not non_burn.empty:
            frame = non_burn.copy()

    frame = frame.sort_values(
        "day_index"
    )

    return frame.tail(
        recent_n
    ).reset_index(
        drop=True
    )


def movement_summary(
    series_id,
    recent_n=12,
):

    drift = recent_drift(
        series_id,
        recent_n=recent_n,
    )

    if drift.empty:
        return pd.DataFrame()

    rows = []

    for driver, group in drift.groupby(
        "driver"
    ):

        group = group.sort_values(
            "day_index"
        )

        values = pd.to_numeric(
            group["influence"],
            errors="coerce",
        ).dropna()

        if len(values) < 2:
            continue

        delta = float(
            values.iloc[-1]
            - values.iloc[0]
        )

        movement = float(
            values.max()
            - values.min()
        )

        rows.append(
            {
                "driver":
                    driver,

                "recent_delta":
                    delta,

                "recent_range":
                    movement,
            }
        )

    result = pd.DataFrame(
        rows
    )

    if result.empty:
        return result

    result = result.sort_values(
        "recent_range",
        ascending=False,
    ).reset_index(
        drop=True
    )

    # Relative within-team movement ranking, not a formal change-point test.
    n = len(result)

    for index in range(n):

        if index < max(
            1,
            n // 3,
        ):
            label = "Higher movement"

        elif index < max(
            2,
            2 * n // 3,
        ):
            label = "Moderate movement"

        else:
            label = "Lower movement"

        result.loc[
            index,
            "movement_label",
        ] = label

    result["direction"] = np.where(
        result["recent_delta"] > 0,
        "Increasing",
        np.where(
            result["recent_delta"] < 0,
            "Decreasing",
            "Flat",
        ),
    )

    return result


def productivity_direction(
    series_id,
    recent_n=7,
):

    frame = recent_trend(
        series_id,
        recent_n=recent_n,
    )

    values = pd.to_numeric(
        frame[
            "predicted_productivity"
        ],
        errors="coerce",
    ).dropna()

    if len(values) < 2:
        return "Not enough history"

    x = np.arange(
        len(values),
        dtype=float,
    )

    slope = np.polyfit(
        x,
        values.to_numpy(),
        1,
    )[0]

    if slope > 0.002:
        return "Improving"

    if slope < -0.002:
        return "Declining"

    return "Broadly stable"

