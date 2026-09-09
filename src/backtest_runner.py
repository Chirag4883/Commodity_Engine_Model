from __future__ import annotations

"""
Level 3 — Historical Walk-Forward Backtester

Pipeline
--------
PIT snapshot
  -> anonymize
  -> structured feature extraction
  -> threshold candidate bottleneck scores
  -> graph shock propagation
  -> locally map terminal graph nodes back to listed tickers
  -> obtain 30d / 90d / 180d forward returns
  -> compute cross-sectional Spearman Rank IC

Research integrity
------------------
The backtest runner is market-data-provider neutral.

- InMemoryForwardReturnProvider:
    deterministic CI/testing only.

- YFinancePriceProvider:
    empirical historical-price implementation.

No arbitrary stock selections occur here. The graph determines the terminal
equity exposure set and all resulting equity signals are numerical scores.
"""

from dataclasses import dataclass, field
from datetime import date, timedelta
import math
from typing import Dict, Mapping, Optional, Protocol, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.agent import CandidateNode, QuantamentalFeatureAgent
from src.anonymizer import PITAnonymizer
from src.graph_engine import Layer, SupplyChainGraph


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class BacktestError(Exception):
    """Base exception for Level-3 failures."""


class BacktestConfigurationError(BacktestError):
    """Invalid research/backtest configuration."""


class SnapshotValidationError(BacktestError):
    """Invalid PIT snapshot metadata."""


class MarketDataError(BacktestError):
    """Historical market-data retrieval/validation failure."""


# ---------------------------------------------------------------------------
# PIT snapshot model
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SnapshotCandidate:
    """
    Local mapping between an anonymous inference candidate and a real graph node.

    graph_node_id NEVER enters the LLM prompt.
    """

    anonymized_id: str
    graph_node_id: str
    physical_role: str

    def __post_init__(self) -> None:
        if not self.anonymized_id.strip():
            raise ValueError("anonymized_id must not be empty")

        if not self.graph_node_id.strip():
            raise ValueError("graph_node_id must not be empty")

        if not self.physical_role.strip():
            raise ValueError("physical_role must not be empty")


@dataclass(frozen=True, slots=True)
class HistoricalSnapshot:
    snapshot_id: str
    as_of_date: date
    source_text: str
    candidates: Tuple[SnapshotCandidate, ...]

    executive_names: Tuple[str, ...] = ()

    company_aliases: Mapping[str, str] = field(
        default_factory=dict
    )

    ticker_aliases: Mapping[str, str] = field(
        default_factory=dict
    )

    event_aliases: Mapping[str, str] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if not self.snapshot_id.strip():
            raise ValueError("snapshot_id must not be empty")

        if not isinstance(self.as_of_date, date):
            raise ValueError(
                "as_of_date must be datetime.date"
            )

        if not self.source_text.strip():
            raise ValueError("source_text must not be empty")

        if not self.candidates:
            raise ValueError(
                "snapshot must contain at least one candidate"
            )


# ---------------------------------------------------------------------------
# Backtest configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BacktestConfig:
    start_year: int = 2018
    end_year: int = 2025

    horizons_days: Tuple[int, ...] = (
        30,
        90,
        180,
    )

    primary_horizon_days: int = 90

    signal_threshold: float = 0.15

    min_ic_observations: int = 3

    def __post_init__(self) -> None:
        if self.start_year < 2016:
            raise BacktestConfigurationError(
                "start_year must be >= 2016 under "
                "the project PIT policy"
            )

        if self.end_year < self.start_year:
            raise BacktestConfigurationError(
                "end_year must be >= start_year"
            )

        if not self.horizons_days:
            raise BacktestConfigurationError(
                "at least one horizon is required"
            )

        if any(
            horizon <= 0
            for horizon in self.horizons_days
        ):
            raise BacktestConfigurationError(
                "horizons must be positive"
            )

        if (
            len(set(self.horizons_days))
            != len(self.horizons_days)
        ):
            raise BacktestConfigurationError(
                "horizons must be unique"
            )

        if (
            self.primary_horizon_days
            not in self.horizons_days
        ):
            raise BacktestConfigurationError(
                "primary_horizon_days must be "
                "present in horizons_days"
            )

        if (
            not math.isfinite(self.signal_threshold)
            or self.signal_threshold < 0
        ):
            raise BacktestConfigurationError(
                "signal_threshold must be finite "
                "and non-negative"
            )

        if self.min_ic_observations < 3:
            raise BacktestConfigurationError(
                "min_ic_observations must be at least 3"
            )


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CandidateSignal:
    anonymized_id: str
    graph_node_id: str

    raw_bottleneck_score: float

    active: bool

    terminal_contributions: Mapping[str, float]


@dataclass(frozen=True, slots=True)
class RankICObservation:
    coefficient: Optional[float]

    n_obs: int

    reason: Optional[str] = None


@dataclass(frozen=True, slots=True)
class SnapshotBacktestResult:
    snapshot_id: str
    as_of_date: date

    candidate_signals: Tuple[
        CandidateSignal,
        ...
    ]

    raw_equity_scores: Mapping[
        str,
        float,
    ]

    forward_returns: Mapping[
        int,
        Mapping[str, Optional[float]],
    ]

    rank_ic: Mapping[
        int,
        RankICObservation,
    ]


@dataclass(frozen=True, slots=True)
class WalkForwardResult:
    snapshots: Tuple[
        SnapshotBacktestResult,
        ...
    ]

    mean_rank_ic: Mapping[
        int,
        Optional[float],
    ]

    primary_horizon_days: int

    @property
    def primary_mean_rank_ic(
        self,
    ) -> Optional[float]:
        return self.mean_rank_ic.get(
            self.primary_horizon_days
        )


# ---------------------------------------------------------------------------
# Forward-return provider interface
# ---------------------------------------------------------------------------


class ForwardReturnProvider(Protocol):

    def get_forward_returns(
        self,
        *,
        tickers: Sequence[str],
        as_of_date: date,
        horizons_days: Sequence[int],
    ) -> Mapping[
        int,
        Mapping[str, Optional[float]],
    ]:
        ...


# ---------------------------------------------------------------------------
# Deterministic CI provider
# ---------------------------------------------------------------------------


class InMemoryForwardReturnProvider:
    """
    Deterministic PIT return provider.

    Intended for:
    - CI
    - mathematical regression tests
    - offline fixtures

    It must NOT be represented as empirical market validation.
    """

    def __init__(
        self,
        returns_by_date: Mapping[
            date,
            Mapping[
                int,
                Mapping[
                    str,
                    Optional[float],
                ],
            ],
        ],
    ) -> None:
        self._returns_by_date = (
            returns_by_date
        )

    def get_forward_returns(
        self,
        *,
        tickers: Sequence[str],
        as_of_date: date,
        horizons_days: Sequence[int],
    ) -> Mapping[
        int,
        Mapping[str, Optional[float]],
    ]:

        if (
            as_of_date
            not in self._returns_by_date
        ):
            raise MarketDataError(
                "No deterministic return fixture "
                f"for {as_of_date.isoformat()}"
            )

        snapshot_data = (
            self._returns_by_date[
                as_of_date
            ]
        )

        output: Dict[
            int,
            Dict[str, Optional[float]],
        ] = {}

        for horizon in horizons_days:

            if horizon not in snapshot_data:
                raise MarketDataError(
                    f"Missing {horizon}d fixture "
                    f"for {as_of_date.isoformat()}"
                )

            horizon_data = (
                snapshot_data[horizon]
            )

            output[horizon] = {
                ticker: horizon_data.get(
                    ticker
                )
                for ticker in tickers
            }

        return output


# ---------------------------------------------------------------------------
# yfinance empirical provider
# ---------------------------------------------------------------------------


class YFinancePriceProvider:
    """
    Historical forward-return provider backed by yfinance.

    Price convention
    ----------------
    yfinance is requested with:

        interval="1d"
        auto_adjust=True

    The entry price is the first available close on or after:

        T0 + entry_lag_calendar_days

    The horizon price is the first available close on or after:

        T0 + horizon_days

    Therefore horizon dates remain anchored to PIT date T0.

    A maximum date lag prevents stale observations from silently becoming
    valid prices across extended market/data gaps.
    """

    def __init__(
        self,
        *,
        entry_lag_calendar_days: int = 0,
        max_market_date_lag_days: int = 7,
        download_buffer_days: int = 14,
        timeout_seconds: int = 20,
    ) -> None:

        if entry_lag_calendar_days < 0:
            raise ValueError(
                "entry_lag_calendar_days "
                "must be non-negative"
            )

        if max_market_date_lag_days < 1:
            raise ValueError(
                "max_market_date_lag_days "
                "must be >= 1"
            )

        if (
            download_buffer_days
            <= max_market_date_lag_days
        ):
            raise ValueError(
                "download_buffer_days must exceed "
                "max_market_date_lag_days"
            )

        if timeout_seconds <= 0:
            raise ValueError(
                "timeout_seconds must be positive"
            )

        self.entry_lag_calendar_days = (
            entry_lag_calendar_days
        )

        self.max_market_date_lag_days = (
            max_market_date_lag_days
        )

        self.download_buffer_days = (
            download_buffer_days
        )

        self.timeout_seconds = (
            timeout_seconds
        )

    def get_forward_returns(
        self,
        *,
        tickers: Sequence[str],
        as_of_date: date,
        horizons_days: Sequence[int],
    ) -> Mapping[
        int,
        Mapping[str, Optional[float]],
    ]:

        unique_tickers = tuple(
            dict.fromkeys(tickers)
        )

        horizons = tuple(
            horizons_days
        )

        if not unique_tickers:
            raise MarketDataError(
                "No tickers supplied to "
                "market-data provider"
            )

        if (
            not horizons
            or any(
                horizon <= 0
                for horizon in horizons
            )
        ):
            raise MarketDataError(
                "Forward-return horizons "
                "must be positive"
            )

        try:
            import yfinance as yf

        except ImportError as exc:
            raise MarketDataError(
                "yfinance is not installed; "
                "install requirements.txt"
            ) from exc

        # yfinance end= is exclusive.
        end_date = (
            as_of_date
            + timedelta(
                days=max(horizons)
            )
            + timedelta(
                days=(
                    self.download_buffer_days
                    + 1
                )
            )
        )

        try:
            data = yf.download(
                tickers=list(
                    unique_tickers
                ),
                start=(
                    as_of_date.isoformat()
                ),
                end=end_date.isoformat(),
                interval="1d",
                auto_adjust=True,
                actions=False,
                repair=False,
                progress=False,
                threads=True,
                group_by="ticker",
                timeout=(
                    self.timeout_seconds
                ),
                multi_level_index=True,
            )

        except Exception as exc:
            raise MarketDataError(
                "yfinance download failed: "
                f"{exc}"
            ) from exc

        if (
            data is None
            or data.empty
        ):
            raise MarketDataError(
                "yfinance returned no "
                "historical data"
            )

        output: Dict[
            int,
            Dict[str, Optional[float]],
        ] = {
            horizon: {}
            for horizon in horizons
        }

        for ticker in unique_tickers:

            close = (
                self._extract_close_series(
                    data,
                    ticker,
                )
            )

            for horizon in horizons:

                output[horizon][ticker] = (
                    self._forward_return_from_series(
                        close,
                        as_of_date=(
                            as_of_date
                        ),
                        horizon_days=(
                            horizon
                        ),
                    )
                )

        return output

    def _forward_return_from_series(
        self,
        close: Optional[pd.Series],
        *,
        as_of_date: date,
        horizon_days: int,
    ) -> Optional[float]:

        if (
            close is None
            or close.empty
        ):
            return None

        series = (
            close
            .dropna()
            .astype(float)
            .sort_index()
        )

        if series.empty:
            return None

        if not isinstance(
            series.index,
            pd.DatetimeIndex,
        ):
            try:
                series.index = (
                    pd.to_datetime(
                        series.index
                    )
                )
            except Exception:
                return None

        if series.index.tz is not None:
            series.index = (
                series.index.tz_localize(
                    None
                )
            )

        entry_target = (
            as_of_date
            + timedelta(
                days=(
                    self.entry_lag_calendar_days
                )
            )
        )

        horizon_target = (
            as_of_date
            + timedelta(
                days=horizon_days
            )
        )

        entry = (
            self._first_price_on_or_after(
                series,
                target_date=entry_target,
            )
        )

        exit_ = (
            self._first_price_on_or_after(
                series,
                target_date=horizon_target,
            )
        )

        if (
            entry is None
            or exit_ is None
        ):
            return None

        (
            entry_date,
            entry_price,
        ) = entry

        (
            exit_date,
            exit_price,
        ) = exit_

        if exit_date <= entry_date:
            return None

        if (
            entry_price <= 0.0
            or exit_price <= 0.0
        ):
            return None

        value = (
            exit_price
            / entry_price
            - 1.0
        )

        if not math.isfinite(value):
            return None

        return float(value)

    def _first_price_on_or_after(
        self,
        series: pd.Series,
        *,
        target_date: date,
    ) -> Optional[
        Tuple[
            pd.Timestamp,
            float,
        ]
    ]:

        target = pd.Timestamp(
            target_date
        )

        eligible = series.loc[
            series.index >= target
        ]

        if eligible.empty:
            return None

        price_date = pd.Timestamp(
            eligible.index[0]
        )

        lag_days = (
            price_date.normalize()
            - target.normalize()
        ).days

        if (
            lag_days
            > self.max_market_date_lag_days
        ):
            return None

        price = float(
            eligible.iloc[0]
        )

        if not math.isfinite(price):
            return None

        return (
            price_date,
            price,
        )

    @staticmethod
    def _extract_close_series(
        data: pd.DataFrame,
        ticker: str,
    ) -> Optional[pd.Series]:

        if data.empty:
            return None

        if isinstance(
            data.columns,
            pd.MultiIndex,
        ):
            # yfinance versions/grouping options
            # can expose either orientation.
            candidates = (
                (
                    ticker,
                    "Close",
                ),
                (
                    "Close",
                    ticker,
                ),
            )

            for key in candidates:

                if key in data.columns:

                    value = data.loc[
                        :,
                        key,
                    ]

                    if isinstance(
                        value,
                        pd.DataFrame,
                    ):
                        value = (
                            value.iloc[
                                :,
                                0,
                            ]
                        )

                    return value.rename(
                        ticker
                    )

            return None

        if "Close" in data.columns:

            value = data[
                "Close"
            ]

            if isinstance(
                value,
                pd.DataFrame,
            ):
                value = (
                    value.iloc[
                        :,
                        0,
                    ]
                )

            return value.rename(
                ticker
            )

        return None


# ---------------------------------------------------------------------------
# Historical backtester
# ---------------------------------------------------------------------------


class HistoricalBacktester:

    def __init__(
        self,
        *,
        graph: SupplyChainGraph,
        agent: QuantamentalFeatureAgent,
        return_provider: ForwardReturnProvider,
        config: Optional[
            BacktestConfig
        ] = None,
    ) -> None:

        self.graph = graph
        self.agent = agent

        self.return_provider = (
            return_provider
        )

        self.config = (
            config
            or BacktestConfig()
        )

        self._terminal_node_to_ticker = (
            self._discover_terminal_nodes()
        )

        self._terminal_tickers = tuple(
            sorted(
                self._terminal_node_to_ticker
                .values()
            )
        )

        if (
            len(self._terminal_tickers)
            < self.config.min_ic_observations
        ):
            raise BacktestConfigurationError(
                "Graph has too few terminal "
                "equities for Rank IC calculation"
            )

    def run_snapshot(
        self,
        snapshot: HistoricalSnapshot,
    ) -> SnapshotBacktestResult:

        self._validate_snapshot(
            snapshot
        )

        # New anonymizer for each PIT snapshot:
        # no cross-period identity state.
        anonymizer = PITAnonymizer(
            company_aliases=(
                snapshot.company_aliases
            ),
            ticker_aliases=(
                snapshot.ticker_aliases
            ),
            executive_names=(
                snapshot.executive_names
            ),
            event_aliases=(
                snapshot.event_aliases
            ),
        )

        document = (
            anonymizer.anonymize(
                snapshot.source_text
            )
        )

        # Every terminal equity receives a numerical
        # cross-sectional score, including zero exposures.
        raw_equity_scores: Dict[
            str,
            float,
        ] = {
            ticker: 0.0
            for ticker
            in self._terminal_tickers
        }

        signals = []

        for candidate in snapshot.candidates:

            graph_node = (
                self.graph.get_node(
                    candidate.graph_node_id
                )
            )

            if (
                graph_node.layer
                == Layer.LISTED_EQUITY
            ):
                raise SnapshotValidationError(
                    "Feature extraction candidates "
                    "must be physical nodes, not "
                    "Layer-4 equities: "
                    f"{candidate.graph_node_id!r}"
                )

            assessment = (
                self.agent.evaluate(
                    document=document,
                    candidate=CandidateNode(
                        anonymized_id=(
                            candidate.anonymized_id
                        ),
                        layer=int(
                            graph_node.layer
                        ),
                        physical_role=(
                            candidate.physical_role
                        ),
                    ),
                )
            )

            raw_score = float(
                assessment
                .raw_bottleneck_score
            )

            # User-specified organic triggering:
            #
            #     |S_i| > threshold
            #
            active = (
                abs(raw_score)
                > self.config.signal_threshold
            )

            terminal_contributions: Dict[
                str,
                float,
            ] = {}

            if active:

                propagation = (
                    self.graph.propagate_shock(
                        candidate.graph_node_id,
                        raw_score,
                    )
                )

                if not propagation.ok:
                    raise BacktestError(
                        propagation.error
                        or (
                            "Graph propagation "
                            "failed"
                        )
                    )

                # Local de-anonymization happens
                # ONLY after inference.
                for (
                    terminal_node_id,
                    contribution,
                ) in (
                    propagation
                    .terminal_shocks
                    .items()
                ):

                    ticker = (
                        self
                        ._terminal_node_to_ticker
                        .get(
                            terminal_node_id
                        )
                    )

                    if ticker is None:
                        continue

                    terminal_contributions[
                        ticker
                    ] = (
                        terminal_contributions
                        .get(
                            ticker,
                            0.0,
                        )
                        + float(
                            contribution
                        )
                    )

                    raw_equity_scores[
                        ticker
                    ] += float(
                        contribution
                    )

            signals.append(
                CandidateSignal(
                    anonymized_id=(
                        candidate.anonymized_id
                    ),
                    graph_node_id=(
                        candidate.graph_node_id
                    ),
                    raw_bottleneck_score=(
                        raw_score
                    ),
                    active=active,
                    terminal_contributions=(
                        terminal_contributions
                    ),
                )
            )

        forward_returns = (
            self.return_provider
            .get_forward_returns(
                tickers=(
                    self._terminal_tickers
                ),
                as_of_date=(
                    snapshot.as_of_date
                ),
                horizons_days=(
                    self.config
                    .horizons_days
                ),
            )
        )

        rank_ic: Dict[
            int,
            RankICObservation,
        ] = {}

        for horizon in (
            self.config.horizons_days
        ):

            if horizon not in forward_returns:
                raise MarketDataError(
                    "Return provider omitted "
                    "required horizon "
                    f"{horizon}d"
                )

            rank_ic[horizon] = (
                compute_rank_ic(
                    raw_equity_scores,
                    forward_returns[
                        horizon
                    ],
                    min_observations=(
                        self.config
                        .min_ic_observations
                    ),
                )
            )

        return SnapshotBacktestResult(
            snapshot_id=(
                snapshot.snapshot_id
            ),
            as_of_date=(
                snapshot.as_of_date
            ),
            candidate_signals=tuple(
                signals
            ),
            raw_equity_scores=dict(
                raw_equity_scores
            ),
            forward_returns={
                horizon: dict(
                    values
                )
                for (
                    horizon,
                    values,
                ) in (
                    forward_returns
                    .items()
                )
            },
            rank_ic=rank_ic,
        )

    def run_walk_forward(
        self,
        snapshots: Sequence[
            HistoricalSnapshot
        ],
    ) -> WalkForwardResult:

        if not snapshots:
            raise SnapshotValidationError(
                "At least one historical "
                "snapshot is required"
            )

        ids = [
            snapshot.snapshot_id
            for snapshot in snapshots
        ]

        if (
            len(set(ids))
            != len(ids)
        ):
            raise SnapshotValidationError(
                "snapshot_id values "
                "must be unique"
            )

        # Walk strictly forward in time regardless
        # of caller ordering.
        ordered = sorted(
            snapshots,
            key=lambda item: (
                item.as_of_date,
                item.snapshot_id,
            ),
        )

        results = tuple(
            self.run_snapshot(
                snapshot
            )
            for snapshot in ordered
        )

        means: Dict[
            int,
            Optional[float],
        ] = {}

        for horizon in (
            self.config.horizons_days
        ):

            coefficients = [
                result
                .rank_ic[horizon]
                .coefficient
                for result in results
                if (
                    result
                    .rank_ic[horizon]
                    .coefficient
                    is not None
                )
            ]

            means[horizon] = (
                float(
                    np.mean(
                        coefficients
                    )
                )
                if coefficients
                else None
            )

        return WalkForwardResult(
            snapshots=results,
            mean_rank_ic=means,
            primary_horizon_days=(
                self.config
                .primary_horizon_days
            ),
        )

    def _validate_snapshot(
        self,
        snapshot: HistoricalSnapshot,
    ) -> None:

        year = (
            snapshot.as_of_date.year
        )

        if not (
            self.config.start_year
            <= year
            <= self.config.end_year
        ):
            raise SnapshotValidationError(
                f"Snapshot "
                f"{snapshot.snapshot_id!r} "
                f"date "
                f"{snapshot.as_of_date.isoformat()} "
                "lies outside configured "
                "walk-forward window "
                f"{self.config.start_year}-"
                f"{self.config.end_year}"
            )

        candidate_ids = [
            candidate.anonymized_id
            for candidate
            in snapshot.candidates
        ]

        if (
            len(set(candidate_ids))
            != len(candidate_ids)
        ):
            raise SnapshotValidationError(
                "Candidate anonymized_id "
                "values must be unique "
                "per snapshot"
            )

        for candidate in (
            snapshot.candidates
        ):

            if not self.graph.has_node(
                candidate.graph_node_id
            ):
                raise SnapshotValidationError(
                    "Unknown graph node "
                    "in snapshot: "
                    f"{candidate.graph_node_id!r}"
                )

    def _discover_terminal_nodes(
        self,
    ) -> Mapping[str, str]:

        mapping: Dict[
            str,
            str,
        ] = {}

        for node_id in (
            self.graph
            .nx_graph
            .nodes
        ):

            node = (
                self.graph.get_node(
                    node_id
                )
            )

            if (
                node.layer
                == Layer.LISTED_EQUITY
            ):

                ticker = (
                    node.ticker
                    or node_id
                )

                if (
                    ticker
                    in mapping.values()
                ):
                    raise (
                        BacktestConfigurationError(
                            "Duplicate terminal "
                            "ticker mapping: "
                            f"{ticker!r}"
                        )
                    )

                mapping[
                    node_id
                ] = ticker

        return mapping


# ---------------------------------------------------------------------------
# Rank IC
# ---------------------------------------------------------------------------


def compute_rank_ic(
    scores: Mapping[str, float],
    forward_returns: Mapping[
        str,
        Optional[float],
    ],
    *,
    min_observations: int = 3,
) -> RankICObservation:
    """
    Cross-sectional Spearman Rank IC.

        Rank IC
        =
        SpearmanCorr(
            S_T0,
            R_T0->T0+h
        )

    Missing/non-finite returns are excluded pairwise.

    Constant score or return cross-sections return coefficient=None rather
    than emitting NaN into later aggregation.
    """

    paired = []

    common_tickers = sorted(
        set(scores)
        .intersection(
            forward_returns
        )
    )

    for ticker in common_tickers:

        score = scores[
            ticker
        ]

        realized_return = (
            forward_returns[
                ticker
            ]
        )

        if realized_return is None:
            continue

        if not math.isfinite(
            float(score)
        ):
            continue

        if not math.isfinite(
            float(
                realized_return
            )
        ):
            continue

        paired.append(
            (
                float(score),
                float(
                    realized_return
                ),
            )
        )

    n_obs = len(
        paired
    )

    if (
        n_obs
        < min_observations
    ):
        return RankICObservation(
            coefficient=None,
            n_obs=n_obs,
            reason=(
                "insufficient_observations"
            ),
        )

    score_values = np.asarray(
        [
            value[0]
            for value in paired
        ],
        dtype=float,
    )

    return_values = np.asarray(
        [
            value[1]
            for value in paired
        ],
        dtype=float,
    )

    if (
        np.ptp(
            score_values
        )
        == 0.0
    ):
        return RankICObservation(
            coefficient=None,
            n_obs=n_obs,
            reason=(
                "constant_score_cross_section"
            ),
        )

    if (
        np.ptp(
            return_values
        )
        == 0.0
    ):
        return RankICObservation(
            coefficient=None,
            n_obs=n_obs,
            reason=(
                "constant_return_cross_section"
            ),
        )

    result = spearmanr(
        score_values,
        return_values,
        nan_policy="raise",
    )

    coefficient = float(
        result.statistic
    )

    if not math.isfinite(
        coefficient
    ):
        return RankICObservation(
            coefficient=None,
            n_obs=n_obs,
            reason=(
                "non_finite_correlation"
            ),
        )

    return RankICObservation(
        coefficient=(
            coefficient
        ),
        n_obs=n_obs,
        reason=None,
    )
