from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.neutralizer import (
    FactorNeutralizer,
    NeutralizationInputError,
    NeutralizerConfig,
    RankDeficientDesignError,
    neutralize_cross_section,
)


# ---------------------------------------------------------------------------
# Synthetic NSE-500-style fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_cross_section():
    """
    Construct 500 securities with deliberately strong systematic exposures.

    Raw alpha includes:
    - size exposure,
    - momentum exposure,
    - value exposure,
    - sector exposure,
    - idiosyncratic residual,
    - extreme residual outliers to exercise winsorization.
    """

    rng = np.random.default_rng(
        20260909
    )

    n = 500

    tickers = [
        f"TEST{i:03d}.NS"
        for i in range(n)
    ]

    sectors = np.array(
        [
            "Auto",
            "Banks",
            "CapitalGoods",
            "Chemicals",
            "Consumer",
            "Energy",
            "Financials",
            "Healthcare",
            "IT",
            "Metals",
        ]
    )

    sector = rng.choice(
        sectors,
        size=n,
        replace=True,
    )

    log_market_cap = rng.normal(
        loc=10.0,
        scale=1.2,
        size=n,
    )

    # Deliberately correlated with size.
    momentum_12m = (
        0.30
        * log_market_cap
        + rng.normal(
            loc=0.0,
            scale=1.0,
            size=n,
        )
    )

    price_to_book = np.exp(
        rng.normal(
            loc=0.8,
            scale=0.45,
            size=n,
        )
    )

    sector_effect = {
        "Auto": 0.30,
        "Banks": -0.50,
        "CapitalGoods": 0.80,
        "Chemicals": 0.20,
        "Consumer": -0.10,
        "Energy": 0.40,
        "Financials": -0.30,
        "Healthcare": 0.10,
        "IT": 0.55,
        "Metals": -0.60,
    }

    idiosyncratic = rng.normal(
        loc=0.0,
        scale=0.35,
        size=n,
    )

    # Force winsorization.
    idiosyncratic[0] = 8.0
    idiosyncratic[1] = -7.0
    idiosyncratic[2] = 9.0

    raw_score = (
        0.85
        * log_market_cap
        - 0.60
        * momentum_12m
        + 0.40
        * price_to_book
        + np.array(
            [
                sector_effect[
                    value
                ]
                for value
                in sector
            ]
        )
        + idiosyncratic
    )

    return pd.DataFrame(
        {
            "raw_score":
                raw_score,

            "log_market_cap":
                log_market_cap,

            "momentum_12m":
                momentum_12m,

            "price_to_book":
                price_to_book,

            "sector":
                sector,
        },
        index=pd.Index(
            tickers,
            name="ticker",
        ),
    )


# ---------------------------------------------------------------------------
# Core Level-4 gate
# ---------------------------------------------------------------------------


def test_final_alpha_is_machine_precision_neutral_to_continuous_factors(
    synthetic_cross_section,
):
    result = (
        neutralize_cross_section(
            synthetic_cross_section
        )
    )

    alpha = (
        result.alpha_z
        .to_numpy()
    )

    for factor_name in (
        "log_market_cap",
        "momentum_12m",
        "price_to_book",
    ):

        factor = (
            synthetic_cross_section[
                factor_name
            ]
            .to_numpy(
                dtype=float
            )
        )

        correlation = float(
            np.corrcoef(
                alpha,
                factor,
            )[0, 1]
        )

        # Requirement:
        # |r| < 1e-4
        #
        # We enforce a materially tighter gate.
        assert abs(
            correlation
        ) < 1.0e-10


def test_final_alpha_is_machine_precision_neutral_to_every_sector_dummy(
    synthetic_cross_section,
):
    result = (
        neutralize_cross_section(
            synthetic_cross_section
        )
    )

    alpha = (
        result.alpha_z
        .to_numpy()
    )

    sectors = sorted(
        synthetic_cross_section[
            "sector"
        ].unique()
    )

    for sector in sectors:

        dummy = (
            synthetic_cross_section[
                "sector"
            ]
            .eq(
                sector
            )
            .astype(float)
            .to_numpy()
        )

        correlation = float(
            np.corrcoef(
                alpha,
                dummy,
            )[0, 1]
        )

        # Requirement:
        # |r| < 1e-4
        #
        # Again enforce a much tighter bound.
        assert abs(
            correlation
        ) < 1.0e-10


def test_internal_neutrality_diagnostics_pass_strict_gate(
    synthetic_cross_section,
):
    result = (
        neutralize_cross_section(
            synthetic_cross_section
        )
    )

    assert (
        result
        .diagnostics
        .max_abs_final_factor_correlation
        < 1.0e-10
    )

    assert all(
        abs(value)
        < 1.0e-10
        for value
        in result
        .factor_correlations
        .values()
    )


# ---------------------------------------------------------------------------
# Z-score gates
# ---------------------------------------------------------------------------


def test_final_alpha_has_zero_mean(
    synthetic_cross_section,
):
    result = (
        neutralize_cross_section(
            synthetic_cross_section
        )
    )

    mean = float(
        result.alpha_z.mean()
    )

    assert abs(
        mean
    ) < 1.0e-12


def test_final_alpha_has_unit_population_variance(
    synthetic_cross_section,
):
    result = (
        neutralize_cross_section(
            synthetic_cross_section
        )
    )

    std = float(
        np.std(
            result
            .alpha_z
            .to_numpy(),
            ddof=0,
        )
    )

    assert std == pytest.approx(
        1.0,
        abs=1.0e-12,
    )


# ---------------------------------------------------------------------------
# Winsorization gate
# ---------------------------------------------------------------------------


def test_residuals_are_winsorized_at_exactly_three_sigma(
    synthetic_cross_section,
):
    result = (
        neutralize_cross_section(
            synthetic_cross_section
        )
    )

    diagnostics = (
        result.diagnostics
    )

    expected_lower = (
        diagnostics
        .residual_mean_before_winsor
        - (
            3.0
            * diagnostics
            .residual_std_before_winsor
        )
    )

    expected_upper = (
        diagnostics
        .residual_mean_before_winsor
        + (
            3.0
            * diagnostics
            .residual_std_before_winsor
        )
    )

    assert (
        diagnostics
        .winsor_lower_bound
        == pytest.approx(
            expected_lower,
            abs=1.0e-12,
        )
    )

    assert (
        diagnostics
        .winsor_upper_bound
        == pytest.approx(
            expected_upper,
            abs=1.0e-12,
        )
    )

    winsorized = (
        result
        .winsorized_residual
        .to_numpy()
    )

    assert np.all(
        winsorized
        >= (
            expected_lower
            - 1.0e-12
        )
    )

    assert np.all(
        winsorized
        <= (
            expected_upper
            + 1.0e-12
        )
    )

    assert (
        diagnostics
        .n_winsorized_lower
        + diagnostics
        .n_winsorized_upper
        > 0
    )


def test_default_winsor_sigma_is_three():
    config = (
        NeutralizerConfig()
    )

    assert (
        config.winsor_sigma
        == 3.0
    )


# ---------------------------------------------------------------------------
# Cross-sectional integrity
# ---------------------------------------------------------------------------


def test_no_security_is_selected_or_dropped(
    synthetic_cross_section,
):
    """
    Module 1 must output a standardized score ARRAY,
    not qualitative stock picks.
    """

    result = (
        neutralize_cross_section(
            synthetic_cross_section
        )
    )

    assert len(
        result.alpha_z
    ) == len(
        synthetic_cross_section
    )

    assert list(
        result.alpha_z.index
    ) == list(
        synthetic_cross_section.index
    )

    assert (
        result
        .alpha_z
        .notna()
        .all()
    )


def test_result_frame_preserves_complete_cross_section(
    synthetic_cross_section,
):
    result = (
        neutralize_cross_section(
            synthetic_cross_section
        )
    )

    output = (
        result.as_frame()
    )

    assert list(
        output.index
    ) == list(
        synthetic_cross_section.index
    )

    assert list(
        output.columns
    ) == [
        "fitted_systematic_score",
        "raw_residual",
        "winsorized_residual",
        "orthogonalized_residual",
        "alpha_z",
    ]

    assert (
        output
        .notna()
        .all()
        .all()
    )


# ---------------------------------------------------------------------------
# Regression effectiveness
# ---------------------------------------------------------------------------


def test_raw_scores_contain_material_factor_exposure_before_neutralization(
    synthetic_cross_section,
):
    """
    Confirm the test fixture actually contains factor contamination.

    Otherwise a passing neutralization test would be trivial.
    """

    raw = (
        synthetic_cross_section[
            "raw_score"
        ]
        .to_numpy()
    )

    size = (
        synthetic_cross_section[
            "log_market_cap"
        ]
        .to_numpy()
    )

    correlation = float(
        np.corrcoef(
            raw,
            size,
        )[0, 1]
    )

    assert abs(
        correlation
    ) > 0.25


def test_ols_design_is_full_rank(
    synthetic_cross_section,
):
    result = (
        neutralize_cross_section(
            synthetic_cross_section
        )
    )

    assert (
        result
        .diagnostics
        .design_rank
        == result
        .diagnostics
        .n_design_columns
    )


# ---------------------------------------------------------------------------
# Failure handling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "column",
    [
        "raw_score",
        "log_market_cap",
        "momentum_12m",
        "price_to_book",
        "sector",
    ],
)
def test_missing_required_column_is_rejected(
    synthetic_cross_section,
    column,
):
    broken = (
        synthetic_cross_section
        .drop(
            columns=[
                column
            ]
        )
    )

    with pytest.raises(
        NeutralizationInputError
    ):
        neutralize_cross_section(
            broken
        )


@pytest.mark.parametrize(
    "bad_value",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_non_finite_raw_score_is_rejected(
    synthetic_cross_section,
    bad_value,
):
    broken = (
        synthetic_cross_section
        .copy()
    )

    broken.iloc[
        0,
        broken.columns.get_loc(
            "raw_score"
        ),
    ] = bad_value

    with pytest.raises(
        NeutralizationInputError
    ):
        neutralize_cross_section(
            broken
        )


def test_duplicate_security_index_is_rejected(
    synthetic_cross_section,
):
    broken = (
        synthetic_cross_section
        .copy()
    )

    index = list(
        broken.index
    )

    index[1] = index[0]

    broken.index = index

    with pytest.raises(
        NeutralizationInputError
    ):
        neutralize_cross_section(
            broken
        )


def test_rank_deficient_factor_matrix_is_rejected(
    synthetic_cross_section,
):
    """
    Exact factor duplication should not silently produce unstable betas.
    """

    broken = (
        synthetic_cross_section
        .copy()
    )

    broken[
        "momentum_12m"
    ] = broken[
        "log_market_cap"
    ]

    with pytest.raises(
        RankDeficientDesignError
    ):
        neutralize_cross_section(
            broken
        )


def test_constant_continuous_factor_is_rejected(
    synthetic_cross_section,
):
    broken = (
        synthetic_cross_section
        .copy()
    )

    broken[
        "price_to_book"
    ] = 1.0

    with pytest.raises(
        NeutralizationInputError
    ):
        neutralize_cross_section(
            broken
        )


def test_single_sector_cross_section_is_rejected(
    synthetic_cross_section,
):
    broken = (
        synthetic_cross_section
        .copy()
    )

    broken[
        "sector"
    ] = "OnlySector"

    with pytest.raises(
        NeutralizationInputError
    ):
        neutralize_cross_section(
            broken
        )


def test_too_small_cross_section_is_rejected(
    synthetic_cross_section,
):
    broken = (
        synthetic_cross_section
        .iloc[
            :20
        ]
    )

    with pytest.raises(
        NeutralizationInputError
    ):
        neutralize_cross_section(
            broken
        )


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_neutralization_is_deterministic(
    synthetic_cross_section,
):
    neutralizer = (
        FactorNeutralizer()
    )

    first = (
        neutralizer.neutralize(
            synthetic_cross_section
        )
    )

    second = (
        neutralizer.neutralize(
            synthetic_cross_section
        )
    )

    np.testing.assert_array_equal(
        first
        .alpha_z
        .to_numpy(),

        second
        .alpha_z
        .to_numpy(),
    )


# ---------------------------------------------------------------------------
# Auditability
# ---------------------------------------------------------------------------


def test_coefficients_include_style_and_sector_exposures(
    synthetic_cross_section,
):
    result = (
        neutralize_cross_section(
            synthetic_cross_section
        )
    )

    coefficient_names = (
        set(
            result
            .coefficients
            .index
        )
    )

    assert (
        "intercept"
        in coefficient_names
    )

    assert (
        "z_log_market_cap"
        in coefficient_names
    )

    assert (
        "z_momentum_12m"
        in coefficient_names
    )

    assert (
        "z_price_to_book"
        in coefficient_names
    )

    assert any(
        name.startswith(
            "sector["
        )
        for name
        in coefficient_names
    )


def test_one_sector_is_explicitly_retained_as_baseline(
    synthetic_cross_section,
):
    result = (
        neutralize_cross_section(
            synthetic_cross_section
        )
    )

    sectors = sorted(
        synthetic_cross_section[
            "sector"
        ].unique()
    )

    assert (
        result.sector_baseline
        == sectors[0]
    )
