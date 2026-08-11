

from pathlib import Path
import json
from datetime import date, datetime

import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    roc_auc_score,
)

from utils.forecast_engine import (
    available_context,
    defaults_for_series,
    default_forecast_date,
    forecast_next_day,
)


THIS_FILE = Path(__file__).resolve()
DASHBOARD_DIR = THIS_FILE.parents[1]
REPO_DIR = DASHBOARD_DIR.parent
DATA_DIR = DASHBOARD_DIR / "data"
MODEL_DIR = REPO_DIR / "models"

TEAM_DAILY_PATH = DATA_DIR / "team_daily.csv"
RUNTIME_CONFIG_PATH = DATA_DIR / "forecast_runtime_config.json"

VALIDATION_SUMMARY_PATH = (
    MODEL_DIR
    / "05_complete_validation_summary.json"
)

VALIDATION_METRICS_PATH = (
    MODEL_DIR
    / "05_complete_fused_validation_metrics.csv"
)

ALARM_METRICS_PATH = (
    MODEL_DIR
    / "05_complete_alarm_metrics.csv"
)


def _as_date(value):

    if isinstance(value, datetime):
        return value.date()

    if isinstance(value, date):
        return value

    if isinstance(value, str):
        return datetime.fromisoformat(
            value
        ).date()

    raise TypeError(
        "forecast_date must be date, datetime or ISO date string."
    )


def _safe_forecast_date(value):

    candidate = _as_date(
        value
    )

    if candidate.weekday() == 4:
        raise ValueError(
            "Friday is outside the Sat–Thu production calendar used by the model."
        )

    return candidate


def context_table():

    return available_context()


def baseline_plan(
    series_id,
):

    row = defaults_for_series(
        series_id
    )

    return {
        "series_id":
            str(series_id),

        "team":
            row["team_display"],

        "department":
            row["department_display"],

        "targeted_productivity":
            float(
                row[
                    "targeted_productivity"
                ]
            ),

        "smv":
            float(
                row[
                    "smv"
                ]
            ),

        "wip":
            float(
                row[
                    "wip"
                ]
            ),

        "over_time":
            float(
                row[
                    "over_time"
                ]
            ),

        "incentive":
            float(
                row[
                    "incentive"
                ]
            ),

        "no_of_workers":
            float(
                row[
                    "no_of_workers"
                ]
            ),

        "no_of_style_change":
            int(
                round(
                    float(
                        row[
                            "no_of_style_change"
                        ]
                    )
                )
            ),
    }


def forecast_plan(
    plan,
    forecast_date,
):

    forecast_date = _safe_forecast_date(
        forecast_date
    )

    return forecast_next_day(
        series_id=plan["series_id"],
        forecast_date=forecast_date,
        targeted_productivity=float(
            plan[
                "targeted_productivity"
            ]
        ),
        smv=float(
            plan["smv"]
        ),
        wip=float(
            plan["wip"]
        ),
        over_time=float(
            plan["over_time"]
        ),
        incentive=float(
            plan["incentive"]
        ),
        no_of_workers=float(
            plan[
                "no_of_workers"
            ]
        ),
        no_of_style_change=int(
            plan[
                "no_of_style_change"
            ]
        ),
    )


def compare_scenario(
    series_id,
    forecast_date,
    incentive=None,
    wip=None,
    no_of_workers=None,
):

    baseline = baseline_plan(
        series_id
    )

    proposed = dict(
        baseline
    )

    if incentive is not None:
        proposed["incentive"] = max(
            0.0,
            float(
                incentive
            ),
        )

    if wip is not None:
        proposed["wip"] = max(
            0.0,
            float(
                wip
            ),
        )

    if no_of_workers is not None:
        proposed["no_of_workers"] = max(
            0.0,
            float(
                no_of_workers
            ),
        )


    baseline_forecast = forecast_plan(
        baseline,
        forecast_date,
    )

    scenario_forecast = forecast_plan(
        proposed,
        forecast_date,
    )


    productivity_gain = (
        scenario_forecast[
            "expected_productivity"
        ]
        -
        baseline_forecast[
            "expected_productivity"
        ]
    )

    risk_change = (
        scenario_forecast[
            "low_day_risk"
        ]
        -
        baseline_forecast[
            "low_day_risk"
        ]
    )


    return {
        "baseline_plan":
            baseline,

        "scenario_plan":
            proposed,

        "baseline_forecast":
            baseline_forecast,

        "scenario_forecast":
            scenario_forecast,

        "expected_productivity_gain":
            float(
                productivity_gain
            ),

        "low_day_risk_change":
            float(
                risk_change
            ),
    }


def action_candidates(
    series_id,
    forecast_date,
):

    baseline = baseline_plan(
        series_id
    )

    base_result = forecast_plan(
        baseline,
        forecast_date,
    )


    candidates = []


    def add_candidate(
        action,
        plan,
    ):

        result = forecast_plan(
            plan,
            forecast_date,
        )

        gain = (
            result[
                "expected_productivity"
            ]
            -
            base_result[
                "expected_productivity"
            ]
        )

        risk_reduction = (
            base_result[
                "low_day_risk"
            ]
            -
            result[
                "low_day_risk"
            ]
        )

        candidates.append(
            {
                "action":
                    action,

                "expected_productivity":
                    result[
                        "expected_productivity"
                    ],

                "productivity_gain":
                    gain,

                "low_day_risk":
                    result[
                        "low_day_risk"
                    ],

                "risk_reduction":
                    risk_reduction,

                "risk_status":
                    result[
                        "risk_status"
                    ],
            }
        )


    # Incentive candidates
    for increase in [
        10.0,
        20.0,
        30.0,
        50.0,
    ]:

        plan = dict(
            baseline
        )

        plan["incentive"] = (
            baseline[
                "incentive"
            ]
            + increase
        )

        add_candidate(
            f"Increase incentive by {increase:.0f} BDT",
            plan,
        )


    # WIP candidates
    for reduction_fraction in [
        0.10,
        0.20,
        0.30,
    ]:

        plan = dict(
            baseline
        )

        plan["wip"] = max(
            0.0,
            baseline[
                "wip"
            ]
            * (
                1.0
                - reduction_fraction
            ),
        )

        add_candidate(
            f"Reduce WIP by {reduction_fraction:.0%}",
            plan,
        )


    # Worker candidates
    for additional_workers in [
        1,
        2,
        3,
        5,
    ]:

        plan = dict(
            baseline
        )

        plan[
            "no_of_workers"
        ] = (
            baseline[
                "no_of_workers"
            ]
            + additional_workers
        )

        add_candidate(
            f"Add {additional_workers} worker"
            + (
                "s"
                if additional_workers != 1
                else ""
            ),
            plan,
        )


    result = pd.DataFrame(
        candidates
    )


    if result.empty:
        return result


    # Decision ranking:
    # primarily prefer risk reduction, then productivity gain.
    result = result.sort_values(
        [
            "risk_reduction",
            "productivity_gain",
        ],
        ascending=[
            False,
            False,
        ],
    ).reset_index(
        drop=True
    )


    result["recommended"] = False

    if len(result):
        result.loc[
            0,
            "recommended",
        ] = True


    return result


def incentive_budget_allocation(
    *,
    forecast_date,
    budget_bdt,
    department="All",
    increment_bdt=10.0,
    max_extra_per_line=50.0,
):

    forecast_date = _safe_forecast_date(
        forecast_date
    )

    budget_bdt = max(
        0.0,
        float(
            budget_bdt
        ),
    )

    increment_bdt = max(
        1.0,
        float(
            increment_bdt
        ),
    )

    max_extra_per_line = max(
        increment_bdt,
        float(
            max_extra_per_line
        ),
    )


    context = context_table().copy()

    if department != "All":

        context = context.loc[
            context[
                "department_display"
            ]
            == department
        ].copy()


    if context.empty:

        return (
            pd.DataFrame(),
            {
                "budget_bdt":
                    budget_bdt,

                "spent_bdt":
                    0.0,

                "unspent_bdt":
                    budget_bdt,

                "baseline_mean_productivity":
                    np.nan,

                "allocated_mean_productivity":
                    np.nan,

                "expected_mean_gain":
                    np.nan,
            },
        )


    state = {}


    for _, row in context.iterrows():

        series_id = str(
            row[
                "series_id"
            ]
        )

        plan = baseline_plan(
            series_id
        )

        result = forecast_plan(
            plan,
            forecast_date,
        )

        state[
            series_id
        ] = {
            "team":
                row[
                    "team_display"
                ],

            "department":
                row[
                    "department_display"
                ],

            "baseline_plan":
                plan,

            "current_plan":
                dict(
                    plan
                ),

            "baseline_result":
                result,

            "current_result":
                result,

            "allocated":
                0.0,
        }


    remaining = budget_bdt


    while (
        remaining
        >= increment_bdt
    ):

        best = None


        for series_id, info in state.items():

            if (
                info[
                    "allocated"
                ]
                + increment_bdt
                > max_extra_per_line
                + 1e-9
            ):

                continue


            candidate_plan = dict(
                info[
                    "current_plan"
                ]
            )

            candidate_plan[
                "incentive"
            ] = (
                candidate_plan[
                    "incentive"
                ]
                + increment_bdt
            )


            candidate_result = forecast_plan(
                candidate_plan,
                forecast_date,
            )


            marginal_gain = (
                candidate_result[
                    "expected_productivity"
                ]
                -
                info[
                    "current_result"
                ][
                    "expected_productivity"
                ]
            )


            marginal_risk_reduction = (
                info[
                    "current_result"
                ][
                    "low_day_risk"
                ]
                -
                candidate_result[
                    "low_day_risk"
                ]
            )


            # We only allocate an increment when its modeled productivity
            # gain is positive. Risk reduction breaks near-ties.
            score = (
                marginal_gain
                / increment_bdt
            )


            candidate = {
                "series_id":
                    series_id,

                "plan":
                    candidate_plan,

                "result":
                    candidate_result,

                "marginal_gain":
                    marginal_gain,

                "marginal_risk_reduction":
                    marginal_risk_reduction,

                "score":
                    score,
            }


            if (
                best is None
                or candidate[
                    "score"
                ]
                > best[
                    "score"
                ]
                + 1e-12
                or (
                    abs(
                        candidate[
                            "score"
                        ]
                        -
                        best[
                            "score"
                        ]
                    )
                    <= 1e-12
                    and candidate[
                        "marginal_risk_reduction"
                    ]
                    >
                    best[
                        "marginal_risk_reduction"
                    ]
                )
            ):

                best = candidate


        if (
            best is None
            or best[
                "marginal_gain"
            ]
            <= 0
        ):

            break


        chosen = state[
            best[
                "series_id"
            ]
        ]

        chosen[
            "current_plan"
        ] = best[
            "plan"
        ]

        chosen[
            "current_result"
        ] = best[
            "result"
        ]

        chosen[
            "allocated"
        ] += increment_bdt

        remaining -= increment_bdt


    rows = []


    for series_id, info in state.items():

        baseline_result = (
            info[
                "baseline_result"
            ]
        )

        final_result = (
            info[
                "current_result"
            ]
        )


        rows.append(
            {
                "series_id":
                    series_id,

                "team":
                    info[
                        "team"
                    ],

                "department":
                    info[
                        "department"
                    ],

                "baseline_incentive":
                    info[
                        "baseline_plan"
                    ][
                        "incentive"
                    ],

                "extra_incentive":
                    info[
                        "allocated"
                    ],

                "recommended_incentive":
                    info[
                        "current_plan"
                    ][
                        "incentive"
                    ],

                "baseline_productivity":
                    baseline_result[
                        "expected_productivity"
                    ],

                "allocated_productivity":
                    final_result[
                        "expected_productivity"
                    ],

                "expected_gain":
                    final_result[
                        "expected_productivity"
                    ]
                    -
                    baseline_result[
                        "expected_productivity"
                    ],

                "baseline_risk":
                    baseline_result[
                        "low_day_risk"
                    ],

                "allocated_risk":
                    final_result[
                        "low_day_risk"
                    ],

                "risk_reduction":
                    baseline_result[
                        "low_day_risk"
                    ]
                    -
                    final_result[
                        "low_day_risk"
                    ],
            }
        )


    allocation = pd.DataFrame(
        rows
    ).sort_values(
        [
            "extra_incentive",
            "expected_gain",
        ],
        ascending=[
            False,
            False,
        ],
    ).reset_index(
        drop=True
    )


    spent = float(
        allocation[
            "extra_incentive"
        ].sum()
    )


    summary = {
        "budget_bdt":
            budget_bdt,

        "spent_bdt":
            spent,

        "unspent_bdt":
            max(
                0.0,
                budget_bdt
                - spent,
            ),

        "baseline_mean_productivity":
            float(
                allocation[
                    "baseline_productivity"
                ].mean()
            ),

        "allocated_mean_productivity":
            float(
                allocation[
                    "allocated_productivity"
                ].mean()
            ),

        "expected_mean_gain":
            float(
                allocation[
                    "expected_gain"
                ].mean()
            ),

        "baseline_mean_risk":
            float(
                allocation[
                    "baseline_risk"
                ].mean()
            ),

        "allocated_mean_risk":
            float(
                allocation[
                    "allocated_risk"
                ].mean()
            ),
    }


    return (
        allocation,
        summary,
    )


def reliability_metrics():

    # ------------------------------------------------------------------
    # Prefer the complete validation notebook outputs when available.
    # ------------------------------------------------------------------

    if VALIDATION_SUMMARY_PATH.exists():

        try:

            summary = json.loads(
                VALIDATION_SUMMARY_PATH.read_text()
            )

            fused = summary.get(
                "fused_forecast",
                {},
            )

            alarm = summary.get(
                "low_achievement_alarm",
                {},
            )

            if not alarm:
                alarm = summary.get(
                    "alarm",
                    {},
                )

            two_class = summary.get(
                "two_class",
                {},
            )


            return {
                "source":
                    "05 complete validation",

                "evaluation_scope":
                    summary.get(
                        "evaluation_scope"
                    ),

                "n":
                    summary.get(
                        "n_primary_eval"
                    ),

                "mae":
                    fused.get(
                        "mae"
                    ),

                "rmse":
                    fused.get(
                        "rmse"
                    ),

                "interval_level":
                    0.90,

                "interval_coverage":
                    fused.get(
                        "pi90_coverage"
                    ),

                "bias":
                    fused.get(
                        "bias_pred_minus_actual"
                    ),

                "alarm_precision":
                    alarm.get(
                        "precision"
                    ),

                "alarm_recall":
                    alarm.get(
                        "recall"
                    ),

                "alarm_f1":
                    alarm.get(
                        "f1"
                    ),

                "alarm_balanced_accuracy":
                    alarm.get(
                        "balanced_accuracy"
                    ),

                "roc_auc":
                    two_class.get(
                        "fused_roc_auc"
                    ),

                "brier":
                    two_class.get(
                        "fused_brier"
                    ),
            }

        except Exception:
            pass


    # ------------------------------------------------------------------
    # Fallback: recompute compact rolling metrics from dashboard history.
    # ------------------------------------------------------------------

    if not TEAM_DAILY_PATH.exists():

        return {
            "source":
                "unavailable"
        }


    df = pd.read_csv(
        TEAM_DAILY_PATH
    )


    if "is_burn_in" in df.columns:

        burn = (
            df[
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

        df = df.loc[
            ~burn
        ].copy()


    # Prefer held-out test rows.
    if "split_code" in df.columns:

        split = pd.to_numeric(
            df[
                "split_code"
            ],
            errors="coerce",
        )

        test = df.loc[
            split == 2
        ].copy()

        if len(test):
            df = test
            scope = "held-out test rows"

        else:
            scope = "post-burn-in rows"

    else:

        scope = "post-burn-in rows"


    required = [
        "actual_productivity",
        "predicted_productivity",
    ]

    if not all(
        column in df.columns
        for column in required
    ):

        return {
            "source":
                "unavailable"
        }


    y = pd.to_numeric(
        df[
            "actual_productivity"
        ],
        errors="coerce",
    )

    pred = pd.to_numeric(
        df[
            "predicted_productivity"
        ],
        errors="coerce",
    )


    valid = (
        y.notna()
        &
        pred.notna()
    )

    y = y.loc[
        valid
    ].to_numpy(
        dtype=float
    )

    pred = pred.loc[
        valid
    ].to_numpy(
        dtype=float
    )


    result = {
        "source":
            "dashboard rolling history",

        "evaluation_scope":
            scope,

        "n":
            int(
                len(y)
            ),

        "mae":
            float(
                mean_absolute_error(
                    y,
                    pred,
                )
            )
            if len(y)
            else None,

        "rmse":
            float(
                np.sqrt(
                    mean_squared_error(
                        y,
                        pred,
                    )
                )
            )
            if len(y)
            else None,

        "bias":
            float(
                np.mean(
                    pred
                    - y
                )
            )
            if len(y)
            else None,

        "interval_level":
            None,

        "interval_coverage":
            None,

        "alarm_precision":
            None,

        "alarm_recall":
            None,

        "alarm_f1":
            None,

        "alarm_balanced_accuracy":
            None,

        "roc_auc":
            None,

        "brier":
            None,
    }


    # Existing fused history uses pi_lower / pi_upper for the saved rolling PI.
    if (
        "pi_lower" in df.columns
        and "pi_upper" in df.columns
    ):

        lo = pd.to_numeric(
            df.loc[
                valid,
                "pi_lower",
            ],
            errors="coerce",
        ).to_numpy(
            dtype=float
        )

        hi = pd.to_numeric(
            df.loc[
                valid,
                "pi_upper",
            ],
            errors="coerce",
        ).to_numpy(
            dtype=float
        )

        ok = (
            np.isfinite(
                lo
            )
            &
            np.isfinite(
                hi
            )
        )

        if ok.any():

            result[
                "interval_coverage"
            ] = float(
                np.mean(
                    (
                        y[
                            ok
                        ]
                        >= lo[
                            ok
                        ]
                    )
                    &
                    (
                        y[
                            ok
                        ]
                        <= hi[
                            ok
                        ]
                    )
                )
            )

            # Fused historical artifact was validated as a 90% PI.
            result[
                "interval_level"
            ] = 0.90


    # Recompute low-day alarm metrics when the saved columns exist.
    if all(
        column in df.columns
        for column in [
            "p_alarm",
            "actual_alarm",
        ]
    ):

        p = pd.to_numeric(
            df.loc[
                valid,
                "p_alarm",
            ],
            errors="coerce",
        ).to_numpy(
            dtype=float
        )

        actual_alarm = pd.to_numeric(
            df.loc[
                valid,
                "actual_alarm",
            ],
            errors="coerce",
        ).to_numpy()


        runtime_config = {}

        if RUNTIME_CONFIG_PATH.exists():

            try:

                runtime_config = json.loads(
                    RUNTIME_CONFIG_PATH.read_text()
                )

            except Exception:

                runtime_config = {}


        p_star = runtime_config.get(
            "p_star",
            0.5,
        )


        ok = (
            np.isfinite(
                p
            )
            &
            np.isfinite(
                actual_alarm
            )
        )


        if ok.any():

            p = np.clip(
                p[
                    ok
                ],
                0.0,
                1.0,
            )

            y_alarm = (
                actual_alarm[
                    ok
                ]
                .astype(int)
            )

            pred_alarm = (
                p
                >= float(
                    p_star
                )
            ).astype(int)


            result[
                "alarm_precision"
            ] = float(
                precision_score(
                    y_alarm,
                    pred_alarm,
                    zero_division=0,
                )
            )

            result[
                "alarm_recall"
            ] = float(
                recall_score(
                    y_alarm,
                    pred_alarm,
                    zero_division=0,
                )
            )

            result[
                "alarm_f1"
            ] = float(
                f1_score(
                    y_alarm,
                    pred_alarm,
                    zero_division=0,
                )
            )

            result[
                "alarm_balanced_accuracy"
            ] = float(
                balanced_accuracy_score(
                    y_alarm,
                    pred_alarm,
                )
            )

            if len(
                np.unique(
                    y_alarm
                )
            ) == 2:

                result[
                    "roc_auc"
                ] = float(
                    roc_auc_score(
                        y_alarm,
                        p,
                    )
                )


    return result

