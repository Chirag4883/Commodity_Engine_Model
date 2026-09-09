from __future__ import annotations

from datetime import date
import json
import os
from typing import Any, Mapping

import pandas as pd
import pytest

from src.agent import (
    QuantamentalFeatureAgent,
)

from src.backtest_runner import (
    BacktestConfig,
    HistoricalBacktester,
    HistoricalSnapshot,
    InMemoryForwardReturnProvider,
    SnapshotCandidate,
    SnapshotValidationError,
    YFinancePriceProvider,
    compute_rank_ic,
)

from src.graph_engine import (
    build_seed_graph,
)


# ---------------------------------------------------------------------------
# Official PIT anchor dates
# ---------------------------------------------------------------------------


SOLAR_DATE = date(
    2020,
    12,
    11,
)

TRANSFORMER_DATE = date(
    2022,
    12,
    21,
)


# ---------------------------------------------------------------------------
# Deterministic structured model
# ---------------------------------------------------------------------------


class MappingStructuredLLM:
    """
    Deterministic provider keyed by anonymous candidate ID.

    No external LLM call is required for CI.
    """

    def __init__(
        self,
        responses: Mapping[
            str,
            Mapping[str, Any],
        ],
    ) -> None:

        self.responses = responses

        self.user_prompts: list[
            str
        ] = []

    def complete_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        json_schema: Mapping[
            str,
            Any,
        ],
    ) -> Mapping[
        str,
        Any,
    ]:

        self.user_prompts.append(
            user_prompt
        )

        payload = json.loads(
            user_prompt
        )

        alias = (
            payload[
                "candidate"
            ][
                "anonymized_id"
            ]
        )

        return self.responses[
            alias
        ]


def feature_responses(
) -> Mapping[
    str,
    Mapping[str, Any],
]:

    return {

        "ChokePoint_SolarGlass": {

            "candidate_node_alias":
                "ChokePoint_SolarGlass",

            "E_sub": -0.80,

            "P_pricing": -0.40,

            "T_policy": 0.50,

            "confidence": 0.84,

            "evidence": [
                {
                    "observation": (
                        "Anonymous evidence "
                        "indicates constrained "
                        "imported supply and "
                        "trade protection."
                    ),
                    "effect":
                        "supports_protection",
                }
            ],
        },

        "ChokePoint_TransformerCores": {

            "candidate_node_alias":
                "ChokePoint_TransformerCores",

            "E_sub": -0.70,

            "P_pricing": -0.30,

            "T_policy": 0.50,

            "confidence": 0.79,

            "evidence": [
                {
                    "observation": (
                        "Anonymous evidence "
                        "indicates qualified-core "
                        "sourcing constraints and "
                        "limited substitution."
                    ),
                    "effect":
                        "supports_risk",
                }
            ],
        },

        "BelowThreshold": {

            "candidate_node_alias":
                "BelowThreshold",

            "E_sub": 0.00,

            "P_pricing": 0.00,

            "T_policy": 0.15,

            "confidence": 0.50,

            "evidence": [
                {
                    "observation": (
                        "Evidence is weak "
                        "and close to neutral."
                    ),
                    "effect":
                        "uncertain",
                }
            ],
        },
    }


# ---------------------------------------------------------------------------
# Curated historical snapshots
# ---------------------------------------------------------------------------


def curated_snapshots(
) -> tuple[
    HistoricalSnapshot,
    HistoricalSnapshot,
]:

    solar = HistoricalSnapshot(

        snapshot_id=(
            "solar_glass_"
            "cvd_final_findings"
        ),

        as_of_date=(
            SOLAR_DATE
        ),

        source_text=(
            "In 2020, Borosil Renewables "
            "Limited described competitive "
            "pressure from imported textured "
            "tempered solar glass. "
            "BORORENEW.NS also described "
            "domestic capacity and trade-"
            "remedy protection while some "
            "input costs remained externally "
            "sourced."
        ),

        candidates=(

            SnapshotCandidate(

                anonymized_id=(
                    "ChokePoint_SolarGlass"
                ),

                graph_node_id=(
                    "COMP_SOLAR_GLASS"
                ),

                physical_role=(
                    "Anonymous solar-glass "
                    "choke point linking "
                    "mineral inputs to "
                    "domestic solar-glass "
                    "manufacturing."
                ),
            ),

        ),
    )

    transformer = HistoricalSnapshot(

        snapshot_id=(
            "transformer_core_"
            "surveillance_guideline"
        ),

        as_of_date=(
            TRANSFORMER_DATE
        ),

        source_text=(
            "In 2022, Transformers & "
            "Rectifiers (India) Limited "
            "and Voltamp Transformers "
            "Limited operated in a "
            "transformer supply chain "
            "where qualified CRGO core "
            "sourcing and conformity "
            "requirements affected "
            "supplier flexibility. "
            "TRIL.NS and VOLTAMP.NS "
            "are removed before model "
            "inference."
        ),

        candidates=(

            SnapshotCandidate(

                anonymized_id=(
                    "ChokePoint_"
                    "TransformerCores"
                ),

                graph_node_id=(
                    "COMP_TRANSFORMER_CORES"
                ),

                physical_role=(
                    "Anonymous transformer-"
                    "core choke point "
                    "dependent on qualified "
                    "electrical-steel inputs."
                ),
            ),

        ),
    )

    return (
        solar,
        transformer,
    )


# ---------------------------------------------------------------------------
# Mechanical return fixtures
# ---------------------------------------------------------------------------


def deterministic_returns():

    # IMPORTANT:
    #
    # These values are TEST FIXTURES.
    #
    # They verify the mathematical
    # Rank-IC implementation.
    #
    # They are NOT represented as actual
    # historical returns.

    returns_2020 = {

        "BORORENEW.NS": 0.40,

        "SRF.NS": 0.10,

        "NAVINFLUOR.NS": 0.05,

        "AARTIIND.NS": 0.02,

        "TRIL.NS": -0.03,

        "VOLTAMP.NS": -0.06,
    }

    returns_2022 = {

        "TRIL.NS": 0.30,

        "VOLTAMP.NS": 0.20,

        "BORORENEW.NS": 0.05,

        "SRF.NS": 0.02,

        "NAVINFLUOR.NS": 0.00,

        "AARTIIND.NS": -0.05,
    }

    def scaled(
        values,
        factor,
    ):
        return {
            ticker: (
                value * factor
            )
            for (
                ticker,
                value,
            ) in values.items()
        }

    return {

        SOLAR_DATE: {

            30: scaled(
                returns_2020,
                0.40,
            ),

            90: returns_2020,

            180: scaled(
                returns_2020,
                1.25,
            ),
        },

        TRANSFORMER_DATE: {

            30: scaled(
                returns_2022,
                0.35,
            ),

            90: returns_2022,

            180: scaled(
                returns_2022,
                1.15,
            ),
        },
    }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_backtester(
    return_provider,
):

    client = (
        MappingStructuredLLM(
            feature_responses()
        )
    )

    agent = (
        QuantamentalFeatureAgent(
            client
        )
    )

    runner = (
        HistoricalBacktester(

            graph=(
                build_seed_graph()
            ),

            agent=agent,

            return_provider=(
                return_provider
            ),

            config=(
                BacktestConfig(

                    start_year=2018,

                    end_year=2025,

                    horizons_days=(
                        30,
                        90,
                        180,
                    ),

                    primary_horizon_days=90,

                    signal_threshold=0.15,

                    min_ic_observations=3,
                )
            ),
        )
    )

    return (
        runner,
        client,
    )


# ---------------------------------------------------------------------------
# Walk-forward gates
# ---------------------------------------------------------------------------


def test_curated_walk_forward_runs_chronologically():

    runner, _ = (
        build_backtester(
            InMemoryForwardReturnProvider(
                deterministic_returns()
            )
        )
    )

    (
        solar,
        transformer,
    ) = curated_snapshots()

    # Deliberately reversed input order.
    result = (
        runner.run_walk_forward(
            (
                transformer,
                solar,
            )
        )
    )

    assert [
        item.as_of_date
        for item
        in result.snapshots
    ] == [
        SOLAR_DATE,
        TRANSFORMER_DATE,
    ]


def test_2020_solar_glass_fixture_propagates_to_bororenew():

    runner, _ = (
        build_backtester(
            InMemoryForwardReturnProvider(
                deterministic_returns()
            )
        )
    )

    solar, _ = (
        curated_snapshots()
    )

    result = (
        runner.run_snapshot(
            solar
        )
    )

    signal = (
        result
        .candidate_signals[0]
    )

    # S
    # =
    # 0.50
    # +
    # (-0.40 * -0.80)
    #
    # =
    # 0.82

    assert (
        signal
        .raw_bottleneck_score
        == pytest.approx(
            0.82,
            abs=1e-12,
        )
    )

    assert (
        signal.active
        is True
    )

    expected = (

        0.82

        * (
            (1.0 - 0.05)
            * 0.82
        )

        * (
            (1.0 - 0.05)
            * 0.85
        )
    )

    assert (
        result
        .raw_equity_scores[
            "BORORENEW.NS"
        ]
        == pytest.approx(
            expected,
            abs=1e-12,
        )
    )

    assert (
        result
        .raw_equity_scores[
            "TRIL.NS"
        ]
        == 0.0
    )


def test_2022_transformer_fixture_propagates_to_tril_and_voltamp():

    runner, _ = (
        build_backtester(
            InMemoryForwardReturnProvider(
                deterministic_returns()
            )
        )
    )

    _, transformer = (
        curated_snapshots()
    )

    result = (
        runner.run_snapshot(
            transformer
        )
    )

    signal = (
        result
        .candidate_signals[0]
    )

    # S
    # =
    # 0.50
    # +
    # (-0.30 * -0.70)
    #
    # =
    # 0.71

    assert (
        signal
        .raw_bottleneck_score
        == pytest.approx(
            0.71,
            abs=1e-12,
        )
    )

    assert (
        signal.active
        is True
    )

    expected_tril = (

        0.71

        * (
            (1.0 - 0.06)
            * 0.34
        )

        * (
            (1.0 - 0.10)
            * 0.72
        )
    )

    expected_voltamp = (

        0.71

        * (
            (1.0 - 0.06)
            * 0.34
        )

        * (
            (1.0 - 0.12)
            * 0.68
        )
    )

    assert (
        result
        .raw_equity_scores[
            "TRIL.NS"
        ]
        == pytest.approx(
            expected_tril,
            abs=1e-12,
        )
    )

    assert (
        result
        .raw_equity_scores[
            "VOLTAMP.NS"
        ]
        == pytest.approx(
            expected_voltamp,
            abs=1e-12,
        )
    )


# ---------------------------------------------------------------------------
# PIT leakage regression
# ---------------------------------------------------------------------------


def test_historical_prompts_remain_anonymized():

    runner, client = (
        build_backtester(
            InMemoryForwardReturnProvider(
                deterministic_returns()
            )
        )
    )

    runner.run_walk_forward(
        curated_snapshots()
    )

    combined = "\n".join(
        client.user_prompts
    ).casefold()

    for prohibited in (

        "borosil",

        "bororenew.ns",

        "transformers & rectifiers",

        "voltamp transformers",

        "tril.ns",

        "voltamp.ns",

        "2020",

        "2022",
    ):
        assert (
            prohibited
            not in combined
        )


# ---------------------------------------------------------------------------
# Rank-IC mathematical gate
# ---------------------------------------------------------------------------


def test_rank_ic_math_on_curated_mechanical_fixtures():

    runner, _ = (
        build_backtester(
            InMemoryForwardReturnProvider(
                deterministic_returns()
            )
        )
    )

    result = (
        runner.run_walk_forward(
            curated_snapshots()
        )
    )

    solar_ic = (
        result
        .snapshots[0]
        .rank_ic[90]
        .coefficient
    )

    transformer_ic = (
        result
        .snapshots[1]
        .rank_ic[90]
        .coefficient
    )

    assert (
        solar_ic
        == pytest.approx(
            0.6546536707079772,
            abs=1e-12,
        )
    )

    assert (
        transformer_ic
        == pytest.approx(
            0.8451542547285165,
            abs=1e-12,
        )
    )

    # Mechanical CI gate only.
    #
    # This establishes that the code
    # computes Rank IC correctly for
    # supplied returns.
    #
    # It is NOT an empirical-alpha claim.

    assert (
        result.primary_mean_rank_ic
        == pytest.approx(
            0.7499039627182468,
            abs=1e-12,
        )
    )

    assert (
        result.primary_mean_rank_ic
        > 0.04
    )


# ---------------------------------------------------------------------------
# Organic threshold gate
# ---------------------------------------------------------------------------


def test_signal_threshold_is_strictly_greater_than_threshold():

    provider = (
        InMemoryForwardReturnProvider(
            {
                SOLAR_DATE:
                    deterministic_returns()[
                        SOLAR_DATE
                    ]
            }
        )
    )

    runner, _ = (
        build_backtester(
            provider
        )
    )

    snapshot = (
        HistoricalSnapshot(

            snapshot_id=(
                "threshold_boundary"
            ),

            as_of_date=(
                SOLAR_DATE
            ),

            source_text=(
                "In 2020 the anonymous "
                "supply condition was weak."
            ),

            candidates=(

                SnapshotCandidate(

                    anonymized_id=(
                        "BelowThreshold"
                    ),

                    graph_node_id=(
                        "COMP_SOLAR_GLASS"
                    ),

                    physical_role=(
                        "Anonymous "
                        "physical input."
                    ),
                ),

            ),
        )
    )

    result = (
        runner.run_snapshot(
            snapshot
        )
    )

    assert (
        result
        .candidate_signals[0]
        .raw_bottleneck_score
        == pytest.approx(
            0.15,
            abs=1e-12,
        )
    )

    # Requirement is |S_i| > threshold,
    # not >= threshold.
    assert (
        result
        .candidate_signals[0]
        .active
        is False
    )

    assert all(
        value == 0.0
        for value
        in result
        .raw_equity_scores
        .values()
    )

    assert (
        result
        .rank_ic[90]
        .coefficient
        is None
    )

    assert (
        result
        .rank_ic[90]
        .reason
        ==
        "constant_score_cross_section"
    )


# ---------------------------------------------------------------------------
# PIT window constraints
# ---------------------------------------------------------------------------


def test_pre_2018_snapshot_is_rejected():

    runner, _ = (
        build_backtester(
            InMemoryForwardReturnProvider(
                deterministic_returns()
            )
        )
    )

    snapshot = (
        HistoricalSnapshot(

            snapshot_id=(
                "too_early"
            ),

            as_of_date=(
                date(
                    2017,
                    12,
                    31,
                )
            ),

            source_text=(
                "Anonymous historical "
                "evidence from 2017."
            ),

            candidates=(

                SnapshotCandidate(

                    anonymized_id=(
                        "ChokePoint_SolarGlass"
                    ),

                    graph_node_id=(
                        "COMP_SOLAR_GLASS"
                    ),

                    physical_role=(
                        "Anonymous "
                        "physical input."
                    ),
                ),

            ),
        )
    )

    with pytest.raises(
        SnapshotValidationError
    ):
        runner.run_snapshot(
            snapshot
        )


def test_layer_4_equity_cannot_be_feature_candidate():

    runner, _ = (
        build_backtester(
            InMemoryForwardReturnProvider(
                deterministic_returns()
            )
        )
    )

    snapshot = (
        HistoricalSnapshot(

            snapshot_id=(
                "bad_equity_candidate"
            ),

            as_of_date=(
                SOLAR_DATE
            ),

            source_text=(
                "Anonymous source "
                "dated 2020."
            ),

            candidates=(

                SnapshotCandidate(

                    anonymized_id=(
                        "ChokePoint_SolarGlass"
                    ),

                    graph_node_id=(
                        "BORORENEW.NS"
                    ),

                    physical_role=(
                        "This should "
                        "be rejected."
                    ),
                ),

            ),
        )
    )

    with pytest.raises(
        SnapshotValidationError
    ):
        runner.run_snapshot(
            snapshot
        )


# ---------------------------------------------------------------------------
# Missing/constant cross-section behavior
# ---------------------------------------------------------------------------


def test_rank_ic_omits_missing_returns_programmatically():

    scores = {
        "A": 3.0,
        "B": 2.0,
        "C": 1.0,
        "D": 0.0,
    }

    returns = {
        "A": 0.30,
        "B": None,
        "C": 0.10,
        "D": 0.00,
    }

    result = (
        compute_rank_ic(
            scores,
            returns,
            min_observations=3,
        )
    )

    assert (
        result.n_obs
        == 3
    )

    assert (
        result.coefficient
        == pytest.approx(
            1.0,
            abs=1e-12,
        )
    )


def test_rank_ic_rejects_constant_cross_section_gracefully():

    result = (
        compute_rank_ic(

            {
                "A": 0.0,
                "B": 0.0,
                "C": 0.0,
            },

            {
                "A": 0.10,
                "B": 0.20,
                "C": 0.30,
            },

            min_observations=3,
        )
    )

    assert (
        result.coefficient
        is None
    )

    assert (
        result.reason
        ==
        "constant_score_cross_section"
    )


# ---------------------------------------------------------------------------
# yfinance date-selection mechanics
# ---------------------------------------------------------------------------


def test_yfinance_forward_return_selects_next_trading_day_for_weekend_target():

    provider = (
        YFinancePriceProvider(

            max_market_date_lag_days=7,

            download_buffer_days=14,
        )
    )

    close = pd.Series(

        [
            100.0,
            105.0,
        ],

        index=pd.to_datetime(
            [
                "2020-01-03",
                "2020-01-06",
            ]
        ),
    )

    # Friday + 2 calendar days
    # = Sunday.
    #
    # Monday's close is used.

    value = (
        provider
        ._forward_return_from_series(

            close,

            as_of_date=(
                date(
                    2020,
                    1,
                    3,
                )
            ),

            horizon_days=2,
        )
    )

    assert (
        value
        == pytest.approx(
            0.05,
            abs=1e-12,
        )
    )


def test_yfinance_forward_return_refuses_stale_price_beyond_lag_limit():

    provider = (
        YFinancePriceProvider(

            max_market_date_lag_days=3,

            download_buffer_days=14,
        )
    )

    close = pd.Series(

        [
            100.0,
            110.0,
        ],

        index=pd.to_datetime(
            [
                "2020-01-01",
                "2020-01-20",
            ]
        ),
    )

    value = (
        provider
        ._forward_return_from_series(

            close,

            as_of_date=(
                date(
                    2020,
                    1,
                    1,
                )
            ),

            horizon_days=5,
        )
    )

    assert (
        value
        is None
    )


def test_yfinance_extracts_ticker_first_multiindex_close():

    columns = (
        pd.MultiIndex
        .from_tuples(
            [
                (
                    "AAA.NS",
                    "Close",
                ),
                (
                    "AAA.NS",
                    "Volume",
                ),
                (
                    "BBB.NS",
                    "Close",
                ),
            ]
        )
    )

    frame = pd.DataFrame(

        [
            [
                100.0,
                1000.0,
                200.0,
            ]
        ],

        index=pd.to_datetime(
            [
                "2020-01-01"
            ]
        ),

        columns=columns,
    )

    series = (
        YFinancePriceProvider
        ._extract_close_series(
            frame,
            "BBB.NS",
        )
    )

    assert (
        series
        is not None
    )

    assert (
        float(
            series.iloc[0]
        )
        == 200.0
    )


# ---------------------------------------------------------------------------
# EMPIRICAL NETWORK GATE
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    os.getenv(
        "RUN_LIVE_YFINANCE"
    ) != "1",
    reason=(
        "Set RUN_LIVE_YFINANCE=1 "
        "to run the empirical "
        "Yahoo-price gate"
    ),
)
def test_live_yfinance_90d_mean_rank_ic_exceeds_threshold():
    """
    This is the actual historical-price gate.

    Passing this test — not the fixture-return test above — is required before
    claiming the curated Level-3 observations clear the empirical +0.04 IC
    threshold.
    """

    runner, _ = (
        build_backtester(

            YFinancePriceProvider(

                entry_lag_calendar_days=0,

                max_market_date_lag_days=7,

                download_buffer_days=14,

                timeout_seconds=30,
            )
        )
    )

    result = (
        runner.run_walk_forward(
            curated_snapshots()
        )
    )

    assert (
        result.primary_mean_rank_ic
        is not None
    )

    assert (
        result.primary_mean_rank_ic
        > 0.04
    ), (
        "Empirical 90d mean "
        "Rank IC failed: "
        f"{result.primary_mean_rank_ic:.6f}"
    )
