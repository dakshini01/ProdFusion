

from pathlib import Path
import json
import math
from datetime import date, datetime

import numpy as np
import pandas as pd
from scipy.interpolate import BSpline
from scipy import stats


THIS_FILE = Path(__file__).resolve()
DASHBOARD_DIR = THIS_FILE.parents[1]
DATA_DIR = DASHBOARD_DIR / "data"

STATE_PATH = DATA_DIR / "forecast_runtime_states.npz"
CONFIG_PATH = DATA_DIR / "forecast_runtime_config.json"
DEFAULTS_PATH = DATA_DIR / "forecast_defaults.csv"


def _load_artifacts():

    if not STATE_PATH.exists():
        raise FileNotFoundError(
            f"Forecast runtime state not found: {STATE_PATH}"
        )

    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Forecast runtime config not found: {CONFIG_PATH}"
        )

    states = np.load(
        STATE_PATH,
        allow_pickle=False,
    )

    with open(
        CONFIG_PATH,
        "r",
        encoding="utf-8",
    ) as file_handle:
        config = json.load(file_handle)

    defaults = pd.read_csv(
        DEFAULTS_PATH
    )

    return (
        states,
        config,
        defaults,
    )


STATES, CONFIG, DEFAULTS = _load_artifacts()


SERIES_IDS = [
    str(value)
    for value
    in STATES["series_ids"].tolist()
]

SERIES_TO_INDEX = {
    series_id: index
    for index, series_id
    in enumerate(SERIES_IDS)
}

FEATURE_COLUMNS = list(
    CONFIG["feature_columns"]
)

COL = {
    name: index
    for index, name
    in enumerate(FEATURE_COLUMNS)
}

COL["incentive"] = COL["log_incentive"]
COL["wip"] = COL["log_wip_imputed"]
COL["over_time"] = COL["log_over_time"]
COL["smv"] = COL["log_smv"]
COL["no_of_workers"] = COL["log_workers"]


SCALER_MEAN = np.asarray(
    STATES["scaler_mean"],
    dtype=float,
)

SCALER_SCALE = np.asarray(
    STATES["scaler_scale"],
    dtype=float,
)

SPLINE_KNOTS = np.asarray(
    STATES["spline_knots"],
    dtype=float,
)

W_SPLINE_MEAN = np.asarray(
    STATES["w_spline_mean"],
    dtype=float,
)

SPLINE_DEGREE = int(
    CONFIG["spline_degree"]
)

SPLINE_LOWER = float(
    CONFIG["spline_lower"]
)

SPLINE_UPPER = float(
    CONFIG["spline_upper"]
)

N_BETA_TRANSFORM = int(
    CONFIG["n_beta_transform"]
)

P_STAR = float(
    CONFIG["p_star"]
)

WATCH_THRESHOLD = float(
    CONFIG["watch_threshold"]
)

ACHIEVEMENT_LOW_THRESHOLD = float(
    CONFIG["achievement_low_threshold"]
)


def b_spline_basis(
    x,
    knots,
    degree=3,
):

    x = np.asarray(
        x,
        dtype=float,
    )

    knots = np.asarray(
        knots,
        dtype=float,
    )

    knot_vector = np.concatenate(
        [
            np.repeat(
                knots[0],
                degree,
            ),
            knots,
            np.repeat(
                knots[-1],
                degree,
            ),
        ]
    )

    n_basis = (
        len(knot_vector)
        - degree
        - 1
    )

    basis = np.zeros(
        (
            len(x),
            n_basis,
        ),
        dtype=float,
    )

    for index in range(
        n_basis
    ):

        coeff = np.zeros(
            n_basis,
            dtype=float,
        )

        coeff[index] = 1.0

        spline = BSpline(
            knot_vector,
            coeff,
            degree,
            extrapolate=False,
        )

        basis[:, index] = spline(
            x
        )

    return np.nan_to_num(
        basis,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )


def i_spline_basis(
    x,
    knots,
    degree=3,
):

    basis = b_spline_basis(
        x,
        knots,
        degree,
    )

    cumulative = np.cumsum(
        basis[:, ::-1],
        axis=1,
    )[:, ::-1]

    return cumulative[:, 1:]


def _productivity_from_logit(
    z_value,
):

    z_value = np.clip(
        np.asarray(
            z_value,
            dtype=float,
        ),
        -30,
        30,
    )

    y_beta = (
        1.0
        /
        (
            1.0
            + np.exp(
                -z_value
            )
        )
    )

    y_original = (
        y_beta
        * N_BETA_TRANSFORM
        - 0.5
    ) / (
        N_BETA_TRANSFORM
        - 1
    )

    return np.clip(
        y_original,
        0.0,
        1.0,
    )


def _quarter_label_from_date(
    forecast_date,
):

    if isinstance(
        forecast_date,
        str,
    ):
        forecast_date = datetime.fromisoformat(
            forecast_date
        ).date()

    if isinstance(
        forecast_date,
        datetime,
    ):
        forecast_date = forecast_date.date()

    if not isinstance(
        forecast_date,
        date,
    ):
        raise TypeError(
            "forecast_date must be a date, datetime or ISO date string."
        )

    labels = list(
        CONFIG["quarter_labels"]
    )

    if not labels:
        raise ValueError(
            "No quarter labels are available."
        )

    quarter_index = min(
        (forecast_date.day - 1) // 7,
        len(labels) - 1,
    )

    return labels[
        quarter_index
    ]


def _six_day_dow(
    forecast_date,
):

    if isinstance(
        forecast_date,
        str,
    ):
        forecast_date = datetime.fromisoformat(
            forecast_date
        ).date()

    if isinstance(
        forecast_date,
        datetime,
    ):
        forecast_date = forecast_date.date()

    # Python: Monday=0 ... Sunday=6
    six_day_map = {
        5: 0,  # Saturday
        6: 1,  # Sunday
        0: 2,  # Monday
        1: 3,  # Tuesday
        2: 4,  # Wednesday
        3: 5,  # Thursday
    }

    if forecast_date.weekday() == 4:
        raise ValueError(
            "Friday is outside the six-day Sat–Thu production calendar "
            "used to build the model."
        )

    day_num = six_day_map[
        forecast_date.weekday()
    ]

    sin_dow = math.sin(
        2
        * math.pi
        * day_num
        / 6
    )

    cos_dow = math.cos(
        2
        * math.pi
        * day_num
        / 6
    )

    return (
        sin_dow,
        cos_dow,
    )


def available_context():

    frame = DEFAULTS[
        [
            "series_id",
            "team",
            "team_display",
            "dept_bin",
            "department_display",
        ]
    ].copy()

    return frame.sort_values(
        [
            "department_display",
            "team",
        ]
    ).reset_index(
        drop=True
    )


def defaults_for_series(
    series_id,
):

    series_id = str(
        series_id
    )

    match = DEFAULTS.loc[
        DEFAULTS["series_id"].astype(str)
        == series_id
    ]

    if match.empty:
        raise KeyError(
            f"Unknown series_id: {series_id}"
        )

    return match.iloc[0].to_dict()


def default_forecast_date():

    latest = CONFIG.get(
        "latest_real_date"
    )

    if latest:

        latest_date = (
            datetime.fromisoformat(
                latest
            )
            .date()
        )

        candidate = (
            latest_date
            + pd.Timedelta(
                days=1
            )
        )

        if hasattr(
            candidate,
            "date",
        ):
            candidate = candidate.date()

        # Skip Friday because model workweek is Sat–Thu.
        if candidate.weekday() == 4:
            candidate = (
                candidate
                + pd.Timedelta(
                    days=1
                )
            )

            if hasattr(
                candidate,
                "date",
            ):
                candidate = candidate.date()

        return candidate

    return date.today()


def forecast_next_day(
    *,
    series_id,
    forecast_date,
    targeted_productivity,
    smv,
    wip,
    over_time,
    incentive,
    no_of_workers,
    no_of_style_change,
):

    series_id = str(
        series_id
    )

    if series_id not in SERIES_TO_INDEX:
        raise KeyError(
            f"Unknown team-department series: {series_id}"
        )


    target = float(
        targeted_productivity
    )

    if not 0.0 <= target <= 1.0:
        raise ValueError(
            "Target productivity must be between 0 and 1."
        )


    numeric_inputs = {
        "smv":
            float(smv),

        "wip":
            float(wip),

        "over_time":
            float(over_time),

        "incentive":
            float(incentive),

        "no_of_workers":
            float(no_of_workers),

        "no_of_style_change":
            float(no_of_style_change),
    }


    for name, value in numeric_inputs.items():

        if value < 0:
            raise ValueError(
                f"{name} cannot be negative."
            )


    row = defaults_for_series(
        series_id
    )

    dept_bin = int(
        row["dept_bin"]
    )


    quarter_label = (
        _quarter_label_from_date(
            forecast_date
        )
    )


    (
        sin_dow,
        cos_dow,
    ) = _six_day_dow(
        forecast_date
    )


    # -------------------------------------------------------------
    # Exact raw-to-feature construction used in Notebook 03's
    # forecast_next_day().
    # -------------------------------------------------------------

    log_values = {
        "log_incentive":
            np.log1p(
                max(
                    numeric_inputs[
                        "incentive"
                    ],
                    0.0,
                )
            ),

        "log_wip_imputed":
            np.log1p(
                max(
                    numeric_inputs[
                        "wip"
                    ],
                    0.0,
                )
            ),

        "log_over_time":
            np.log1p(
                max(
                    numeric_inputs[
                        "over_time"
                    ],
                    0.0,
                )
            ),

        "log_smv":
            np.log1p(
                max(
                    numeric_inputs[
                        "smv"
                    ],
                    0.0,
                )
            ),

        "log_workers":
            np.log1p(
                max(
                    numeric_inputs[
                        "no_of_workers"
                    ],
                    0.0,
                )
            ),

        "targeted_productivity":
            target,

        "no_of_style_change":
            numeric_inputs[
                "no_of_style_change"
            ],

        "sin_dow":
            sin_dow,

        "cos_dow":
            cos_dow,
    }


    x_raw = np.asarray(
        [
            log_values[column]
            for column
            in FEATURE_COLUMNS
        ],
        dtype=float,
    )


    x_std = (
        x_raw
        - SCALER_MEAN
    ) / SCALER_SCALE


    incentive_std = (
        x_std[
            COL["incentive"]
        ]
    )

    wip_std = (
        x_std[
            COL["wip"]
        ]
    )

    workers_std = (
        x_std[
            COL["no_of_workers"]
        ]
    )


    spline_x = np.clip(
        [incentive_std],
        SPLINE_LOWER,
        SPLINE_UPPER,
    )


    I_row = i_spline_basis(
        spline_x,
        SPLINE_KNOTS,
        SPLINE_DEGREE,
    )[0]


    spline_value = (
        I_row
        @ W_SPLINE_MEAN
    )


    quarter_value = float(
        CONFIG[
            "quarter_effect_mean"
        ][
            quarter_label
        ]
    )


    dept_value = (
        float(
            CONFIG[
                "b_dept_mean"
            ]
        )
        * dept_bin
    )


    inc_wip_value = (
        float(
            CONFIG[
                "b_inc_wip_mean"
            ]
        )
        * (
            incentive_std
            * wip_std
        )
    )


    wip_workers_value = (
        float(
            CONFIG[
                "b_wip_nw_mean"
            ]
        )
        * (
            wip_std
            * workers_std
        )
    )


    offset_next = (
        spline_value
        + quarter_value
        + dept_value
        + inc_wip_value
        + wip_workers_value
    )


    incentive_positive_flag = (
        1.0
        if numeric_inputs[
            "incentive"
        ] > 0
        else 0.0
    )


    F_next = np.asarray(
        [
            1.0,

            incentive_std
            * incentive_positive_flag,

            wip_std,

            workers_std,

            x_std[
                COL["over_time"]
            ],

            x_std[
                COL["smv"]
            ],

            x_std[
                COL[
                    "no_of_style_change"
                ]
            ],
        ],
        dtype=float,
    )


    state_index = (
        SERIES_TO_INDEX[
            series_id
        ]
    )


    last_m = np.asarray(
        STATES["last_m"][
            state_index
        ],
        dtype=float,
    )


    last_C = np.asarray(
        STATES["last_C"][
            state_index
        ],
        dtype=float,
    )


    Q = np.asarray(
        STATES["Q"][
            state_index
        ],
        dtype=float,
    )


    R = float(
        STATES["R"][
            state_index
        ]
    )


    # One-step-ahead state prediction.
    a_next = last_m
    R_state = last_C + Q


    f_next = (
        offset_next
        + F_next @ a_next
    )


    Qf_next = (
        F_next
        @ R_state
        @ F_next
        + R
    )


    Qf_next = max(
        float(Qf_next),
        1e-8,
    )


    expected = float(
        _productivity_from_logit(
            f_next
        )
    )


    def interval(
        level,
    ):

        critical = stats.norm.ppf(
            1
            - (1 - level)
            / 2
        )

        half_width = (
            critical
            * math.sqrt(
                Qf_next
            )
        )

        lower = float(
            _productivity_from_logit(
                f_next
                - half_width
            )
        )

        upper = float(
            _productivity_from_logit(
                f_next
                + half_width
            )
        )

        return (
            min(
                lower,
                upper,
            ),
            max(
                lower,
                upper,
            ),
        )


    pi80 = interval(
        0.80
    )

    pi95 = interval(
        0.95
    )


    y_threshold = np.clip(
        ACHIEVEMENT_LOW_THRESHOLD
        * target,
        1e-6,
        1 - 1e-6,
    )


    y_threshold_beta = (
        y_threshold
        * N_BETA_TRANSFORM
        - 0.5
    ) / (
        N_BETA_TRANSFORM
        - 1
    )


    y_threshold_beta = np.clip(
        y_threshold_beta,
        1e-6,
        1 - 1e-6,
    )


    z_threshold = math.log(
        y_threshold_beta
        / (
            1
            - y_threshold_beta
        )
    )


    p_alarm = float(
        stats.norm.cdf(
            z_threshold,
            loc=f_next,
            scale=math.sqrt(
                Qf_next
            ),
        )
    )


    if p_alarm >= P_STAR:
        risk_status = "High Risk"

    elif p_alarm >= WATCH_THRESHOLD:
        risk_status = "Watch"

    else:
        risk_status = "Healthy"


    gap = (
        expected
        - target
    )


    return {
        "series_id":
            series_id,

        "team":
            row["team_display"],

        "department":
            row[
                "department_display"
            ],

        "forecast_date":
            str(forecast_date),

        "quarter":
            quarter_label,

        "expected_productivity":
            expected,

        "target_productivity":
            target,

        "forecast_gap":
            gap,

        "pi80_lower":
            pi80[0],

        "pi80_upper":
            pi80[1],

        "pi95_lower":
            pi95[0],

        "pi95_upper":
            pi95[1],

        "low_day_risk":
            p_alarm,

        "risk_status":
            risk_status,

        "p_star":
            P_STAR,

        "watch_threshold":
            WATCH_THRESHOLD,

        "achievement_low_threshold":
            ACHIEVEMENT_LOW_THRESHOLD,
    }

