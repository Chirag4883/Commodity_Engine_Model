from __future__ import annotations

"""
Level 5 — Weekly live Module-1 pipeline.

Ingest
  -> deterministic physical-node routing
  -> PIT anonymization
  -> structured feature extraction
  -> graph shock propagation
  -> full Nifty-500 raw cross-section
  -> factor orthogonalization
  -> latest_alpha_factors.parquet

No qualitative stock selection is produced.
"""

import argparse
from concurrent.futures import (
    ThreadPoolExecutor,
    as_completed,
)
from dataclasses import dataclass
from datetime import (
    date,
    datetime,
    timedelta,
    timezone,
)
import json
import math
import os
from pathlib import Path
import re
import tempfile
import time
from typing import (
    Any,
    Dict,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np
import pandas as pd
import requests

from src.agent import (
    CandidateNode,
    QuantamentalFeatureAgent,
    StructuredLLMClient,
)
from src.anonymizer import (
    PITAnonymizer,
)
from src.graph_engine import (
    Layer,
    SupplyChainGraph,
    build_seed_graph,
)
from src.ingestion_live import (
    EventSource,
    IngestedEvent,
    IngestionBatch,
    LiveIngestor,
    MockSource,
    make_event,
)
from src.neutralizer import (
    FactorNeutralizer,
)


MODULE_NAME = (
    "M1_SUPPLY_CHAIN_CHOKE_POINT"
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LivePipelineConfig:

    lookback_days: int = 7

    signal_threshold: float = 0.15

    max_chars_per_candidate: int = 24_000

    openai_model: str = (
        "gpt-5.6-luna"
    )

    output_path: str = (
        "data/"
        "latest_alpha_factors.parquet"
    )

    metadata_path: str = (
        "data/"
        "pipeline_metadata.json"
    )

    def __post_init__(
        self,
    ) -> None:

        if self.lookback_days < 1:
            raise ValueError(
                "lookback_days must be >= 1"
            )

        if (
            not math.isfinite(
                self.signal_threshold
            )
            or self.signal_threshold < 0
        ):
            raise ValueError(
                "signal_threshold must be "
                "finite and non-negative"
            )


# ---------------------------------------------------------------------------
# Deterministic physical-node router
# ---------------------------------------------------------------------------


class PhysicalNodeRouter:
    """
    Product/material text -> physical graph node.

    This is intentionally deterministic.

    Routing identifies candidate physical nodes. It does NOT choose stocks.
    """

    RULES: Tuple[
        Tuple[
            re.Pattern[str],
            str,
        ],
        ...,
    ] = (

        (
            re.compile(
                r"\b("
                r"solar glass|"
                r"textured tempered glass|"
                r"photovoltaic glass"
                r")\b",
                re.IGNORECASE,
            ),
            "COMP_SOLAR_GLASS",
        ),

        (
            re.compile(
                r"\b("
                r"polysilicon|"
                r"pv cells?|"
                r"photovoltaic cells?"
                r")\b",
                re.IGNORECASE,
            ),
            "COMP_PV_CELLS",
        ),

        (
            re.compile(
                r"\b("
                r"crgo|"
                r"grain oriented electrical steel|"
                r"transformer cores?"
                r")\b",
                re.IGNORECASE,
            ),
            "COMP_TRANSFORMER_CORES",
        ),

        (
            re.compile(
                r"\b("
                r"power transformers?|"
                r"distribution transformers?"
                r")\b",
                re.IGNORECASE,
            ),
            "SUB_POWER_TRANSFORMERS",
        ),

        (
            re.compile(
                r"\b("
                r"fluorspar|"
                r"fluorochemicals?|"
                r"hydrogen fluoride|"
                r"anhydrous hydrogen fluoride|"
                r"\bahf\b"
                r")",
                re.IGNORECASE,
            ),
            "COMP_FLUOROCHEMICAL_FEED",
        ),

        (
            re.compile(
                r"\b("
                r"bulk api|"
                r"bulk apis|"
                r"active pharmaceutical ingredient"
                r")\b",
                re.IGNORECASE,
            ),
            "SUB_BULK_APIS",
        ),

        (
            re.compile(
                r"\b("
                r"sulfuric acid|"
                r"sulphuric acid|"
                r"specialty chemicals?"
                r")\b",
                re.IGNORECASE,
            ),
            "COMP_CHEMICAL_PROCESS_INPUTS",
        ),

        (
            re.compile(
                r"\b("
                r"refractory bricks?|"
                r"refractories"
                r")\b",
                re.IGNORECASE,
            ),
            "COMP_REFRACTORY_BRICKS",
        ),
    )

    def route(
        self,
        events: Sequence[
            IngestedEvent
        ],
        graph: SupplyChainGraph,
    ) -> Mapping[
        str,
        Tuple[
            IngestedEvent,
            ...,
        ],
    ]:

        routed: Dict[
            str,
            Dict[
                str,
                IngestedEvent,
            ],
        ] = {}

        for event in events:

            explicit = (
                event.metadata.get(
                    "candidate_node"
                )
            )

            candidate_nodes = set()

            if explicit:
                candidate_nodes.add(
                    explicit
                )

            for (
                pattern,
                node_id,
            ) in self.RULES:

                if pattern.search(
                    event.searchable_text
                ):
                    candidate_nodes.add(
                        node_id
                    )

            for node_id in (
                candidate_nodes
            ):

                if not graph.has_node(
                    node_id
                ):
                    continue

                node = graph.get_node(
                    node_id
                )

                if (
                    node.layer
                    == Layer.LISTED_EQUITY
                ):
                    continue

                routed.setdefault(
                    node_id,
                    {}
                )[
                    event.event_id
                ] = event

        return {
            node_id: tuple(
                sorted(
                    values.values(),
                    key=lambda item: (
                        item.published_date,
                        item.event_id,
                    ),
                )
            )
            for (
                node_id,
                values,
            )
            in routed.items()
        }


# ---------------------------------------------------------------------------
# Structured clients
# ---------------------------------------------------------------------------


class OpenAIStructuredClient:
    """
    Live implementation of the Level-2 StructuredLLMClient protocol.
    """

    def __init__(
        self,
        *,
        model: str,
    ) -> None:

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "openai package is not installed"
            ) from exc

        if not os.getenv(
            "OPENAI_API_KEY"
        ):
            raise RuntimeError(
                "OPENAI_API_KEY is required "
                "for live mode"
            )

        self.model = model
        self.client = OpenAI()

    def complete_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        json_schema: Mapping[
            str,
            Any,
        ],
    ) -> str:

        response = (
            self.client
            .responses
            .create(
                model=self.model,
                input=[
                    {
                        "role": "system",
                        "content":
                            system_prompt,
                    },
                    {
                        "role": "user",
                        "content":
                            user_prompt,
                    },
                ],
                text={
                    "format": {
                        "type":
                            "json_schema",
                        "name":
                            "feature_assessment",
                        "schema":
                            dict(
                                json_schema
                            ),
                        "strict":
                            True,
                    }
                },
            )
        )

        output_text = (
            response.output_text
            or ""
        ).strip()

        if not output_text:
            raise RuntimeError(
                "Structured model returned "
                "no output_text"
            )

        return output_text


class DeterministicMockClient:
    """
    Offline CI feature extractor.

    It exercises the exact Level-2 parsing/validation boundary without
    network/API calls.
    """

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

        payload = json.loads(
            user_prompt
        )

        alias = payload[
            "candidate"
        ][
            "anonymized_id"
        ]

        role = payload[
            "candidate"
        ][
            "physical_role"
        ].casefold()

        if "solar glass" in role:
            values = (
                -0.80,
                -0.40,
                0.60,
            )

        elif (
            "transformer core"
            in role
        ):
            values = (
                -0.70,
                -0.30,
                0.50,
            )

        elif (
            "fluorochemical"
            in role
            or "fluor" in role
        ):
            values = (
                -0.75,
                -0.20,
                0.35,
            )

        else:
            values = (
                -0.35,
                -0.10,
                0.20,
            )

        e_sub, p_pricing, t_policy = (
            values
        )

        return {
            "candidate_node_alias":
                alias,

            "E_sub":
                e_sub,

            "P_pricing":
                p_pricing,

            "T_policy":
                t_policy,

            "confidence":
                0.80,

            "evidence": [
                {
                    "observation": (
                        "Anonymous physical "
                        "supply evidence indicates "
                        "a measurable bottleneck."
                    ),
                    "effect":
                        "supports_risk",
                }
            ],
        }


# ---------------------------------------------------------------------------
# Live Nifty-500 factor provider
# ---------------------------------------------------------------------------


class Nifty500FactorProvider:

    CONSTITUENTS_URL = (
        "https://nsearchives.nseindia.com/"
        "content/indices/"
        "ind_nifty500list.csv"
    )

    def __init__(
        self,
        *,
        max_workers: int = 16,
        max_imputation_fraction: float = 0.35,
    ) -> None:

        self.max_workers = (
            max_workers
        )

        self.max_imputation_fraction = (
            max_imputation_fraction
        )

    def build(
        self,
    ) -> pd.DataFrame:

        try:
            import yfinance as yf
        except ImportError as exc:
            raise RuntimeError(
                "yfinance is required "
                "for live factor construction"
            ) from exc

        universe = (
            self._download_universe()
        )

        tickers = tuple(
            universe.index
        )

        prices = yf.download(
            tickers=list(
                tickers
            ),
            period="15mo",
            interval="1d",
            auto_adjust=True,
            actions=False,
            progress=False,
            threads=True,
            group_by="column",
            multi_level_index=True,
            timeout=30,
        )

        close = (
            self._extract_close_frame(
                prices,
                tickers,
            )
        )

        momentum = {}

        for ticker in tickers:

            if ticker not in (
                close.columns
            ):
                momentum[
                    ticker
                ] = np.nan
                continue

            series = (
                close[
                    ticker
                ]
                .dropna()
                .astype(float)
            )

            if len(series) < 200:
                momentum[
                    ticker
                ] = np.nan
                continue

            lookback_index = max(
                0,
                len(series) - 253,
            )

            base = float(
                series.iloc[
                    lookback_index
                ]
            )

            latest = float(
                series.iloc[-1]
            )

            if (
                base <= 0
                or latest <= 0
            ):
                momentum[
                    ticker
                ] = np.nan
            else:
                momentum[
                    ticker
                ] = (
                    latest
                    / base
                    - 1.0
                )

        fundamentals: Dict[
            str,
            Tuple[
                float,
                float,
            ],
        ] = {}

        def fetch_fundamental(
            ticker: str,
        ) -> Tuple[
            str,
            float,
            float,
        ]:

            market_cap = np.nan
            price_to_book = np.nan

            try:
                security = yf.Ticker(
                    ticker
                )

                info = (
                    security.get_info()
                    or {}
                )

                value = info.get(
                    "marketCap"
                )

                if value is not None:
                    market_cap = float(
                        value
                    )

                value = info.get(
                    "priceToBook"
                )

                if value is not None:
                    price_to_book = float(
                        value
                    )

                if (
                    not math.isfinite(
                        market_cap
                    )
                    or market_cap <= 0
                ):
                    try:
                        value = (
                            security
                            .fast_info[
                                "market_cap"
                            ]
                        )

                        market_cap = float(
                            value
                        )
                    except Exception:
                        pass

            except Exception:
                pass

            return (
                ticker,
                market_cap,
                price_to_book,
            )

        with ThreadPoolExecutor(
            max_workers=(
                self.max_workers
            )
        ) as executor:

            futures = {
                executor.submit(
                    fetch_fundamental,
                    ticker,
                ): ticker
                for ticker
                in tickers
            }

            for future in (
                as_completed(
                    futures
                )
            ):
                (
                    ticker,
                    market_cap,
                    price_to_book,
                ) = future.result()

                fundamentals[
                    ticker
                ] = (
                    market_cap,
                    price_to_book,
                )

        frame = universe.copy()

        frame[
            "market_cap"
        ] = [
            fundamentals.get(
                ticker,
                (
                    np.nan,
                    np.nan,
                ),
            )[0]
            for ticker
            in frame.index
        ]

        frame[
            "price_to_book"
        ] = [
            fundamentals.get(
                ticker,
                (
                    np.nan,
                    np.nan,
                ),
            )[1]
            for ticker
            in frame.index
        ]

        frame[
            "momentum_12m"
        ] = [
            momentum.get(
                ticker,
                np.nan,
            )
            for ticker
            in frame.index
        ]

        frame.loc[
            (
                ~np.isfinite(
                    frame[
                        "market_cap"
                    ]
                )
                | (
                    frame[
                        "market_cap"
                    ]
                    <= 0
                )
            ),
            "market_cap",
        ] = np.nan

        diagnostics = {}

        for column in (
            "market_cap",
            "momentum_12m",
            "price_to_book",
        ):

            missing_before = int(
                frame[
                    column
                ].isna().sum()
            )

            missing_fraction = (
                missing_before
                / len(frame)
            )

            if (
                missing_fraction
                > self.max_imputation_fraction
            ):
                raise RuntimeError(
                    f"Too much missing live "
                    f"factor data in "
                    f"{column}: "
                    f"{missing_fraction:.1%}"
                )

            # Sector/industry median first.
            frame[
                column
            ] = (
                frame.groupby(
                    "sector",
                    observed=True,
                )[
                    column
                ]
                .transform(
                    lambda series:
                        series.fillna(
                            series.median()
                        )
                )
            )

            # Global fallback.
            global_median = float(
                frame[
                    column
                ].median()
            )

            if not math.isfinite(
                global_median
            ):
                raise RuntimeError(
                    f"Unable to impute "
                    f"{column}"
                )

            frame[
                column
            ] = (
                frame[
                    column
                ]
                .fillna(
                    global_median
                )
            )

            diagnostics[
                f"imputed_{column}"
            ] = missing_before

        frame[
            "log_market_cap"
        ] = np.log(
            frame[
                "market_cap"
            ].astype(float)
        )

        result = frame[
            [
                "log_market_cap",
                "momentum_12m",
                "price_to_book",
                "sector",
            ]
        ].copy()

        result.attrs[
            "factor_diagnostics"
        ] = diagnostics

        return result

    def _download_universe(
        self,
    ) -> pd.DataFrame:

        response = requests.get(
            self.CONSTITUENTS_URL,
            timeout=30,
            headers={
                "User-Agent":
                    "Mozilla/5.0 "
                    "(CommodityEngineModel)"
            },
        )

        response.raise_for_status()

        frame = pd.read_csv(
            pd.io.common.StringIO(
                response.text
            )
        )

        required = {
            "Symbol",
            "Industry",
        }

        if not required.issubset(
            frame.columns
        ):
            raise RuntimeError(
                "Unexpected NSE Nifty-500 "
                "constituent schema"
            )

        frame = (
            frame[
                [
                    "Symbol",
                    "Industry",
                ]
            ]
            .dropna()
            .drop_duplicates(
                subset=[
                    "Symbol"
                ]
            )
        )

        frame[
            "ticker"
        ] = (
            frame[
                "Symbol"
            ]
            .astype(str)
            .str.strip()
            + ".NS"
        )

        frame[
            "sector"
        ] = (
            frame[
                "Industry"
            ]
            .astype(str)
            .str.strip()
        )

        frame = (
            frame
            .set_index(
                "ticker"
            )[
                [
                    "sector"
                ]
            ]
        )

        if len(frame) < 450:
            raise RuntimeError(
                "Nifty-500 universe download "
                f"returned only {len(frame)} "
                "securities"
            )

        return frame

    @staticmethod
    def _extract_close_frame(
        data: pd.DataFrame,
        tickers: Sequence[str],
    ) -> pd.DataFrame:

        if (
            data is None
            or data.empty
        ):
            raise RuntimeError(
                "No price history returned"
            )

        if isinstance(
            data.columns,
            pd.MultiIndex,
        ):

            if (
                "Close"
                in data.columns.get_level_values(
                    0
                )
            ):
                close = data[
                    "Close"
                ]

            elif (
                "Close"
                in data.columns.get_level_values(
                    1
                )
            ):
                close = data.xs(
                    "Close",
                    axis=1,
                    level=1,
                )

            else:
                raise RuntimeError(
                    "Unable to locate Close "
                    "prices in yfinance output"
                )

            if isinstance(
                close,
                pd.Series,
            ):
                close = close.to_frame()

            return close

        if (
            len(tickers) == 1
            and "Close"
            in data.columns
        ):
            return pd.DataFrame(
                {
                    tickers[0]:
                        data[
                            "Close"
                        ]
                }
            )

        raise RuntimeError(
            "Unexpected yfinance "
            "price schema"
        )


# ---------------------------------------------------------------------------
# Mock factors / mock ingestion
# ---------------------------------------------------------------------------


def build_mock_factor_frame(
    *,
    n: int = 500,
) -> pd.DataFrame:

    if n < 50:
        raise ValueError(
            "mock universe must contain "
            "at least 50 securities"
        )

    seed_tickers = [
        "TRIL.NS",
        "VOLTAMP.NS",
        "BORORENEW.NS",
        "SRF.NS",
        "NAVINFLUOR.NS",
        "AARTIIND.NS",
    ]

    remaining = [
        f"MOCK{i:03d}.NS"
        for i
        in range(
            n
            - len(
                seed_tickers
            )
        )
    ]

    tickers = (
        seed_tickers
        + remaining
    )

    rng = np.random.default_rng(
        20260909
    )

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

    return pd.DataFrame(
        {
            "log_market_cap":
                rng.normal(
                    10.0,
                    1.2,
                    n,
                ),

            "momentum_12m":
                rng.normal(
                    0.10,
                    0.35,
                    n,
                ),

            "price_to_book":
                np.exp(
                    rng.normal(
                        0.7,
                        0.45,
                        n,
                    )
                ),

            "sector":
                rng.choice(
                    sectors,
                    n,
                ),
        },
        index=pd.Index(
            tickers,
            name="ticker",
        ),
    )


def build_mock_ingestor(
    *,
    as_of_date: Optional[
        date
    ] = None,
) -> LiveIngestor:

    today = (
        as_of_date
        or datetime.now(
            timezone.utc
        ).date()
    )

    events = (
        make_event(
            source=EventSource.MOCK,
            event_type="qco",
            published_date=today,
            title=(
                "Quality requirements "
                "for CRGO transformer cores"
            ),
            url="mock://transformer",
            text=(
                "Qualified CRGO transformer "
                "core inputs face long "
                "qualification lead times."
            ),
            metadata={
                "candidate_node":
                    "COMP_TRANSFORMER_CORES"
            },
        ),

        make_event(
            source=EventSource.MOCK,
            event_type="trade_remedy",
            published_date=today,
            title=(
                "Trade remedy concerning "
                "textured tempered solar glass"
            ),
            url="mock://solar",
            text=(
                "Imported solar glass "
                "faces domestic trade "
                "protection."
            ),
            metadata={
                "candidate_node":
                    "COMP_SOLAR_GLASS"
            },
        ),

        make_event(
            source=EventSource.MOCK,
            event_type="trade_policy",
            published_date=today,
            title=(
                "Supply constraints in "
                "fluorochemical feedstock"
            ),
            url="mock://fluor",
            text=(
                "Fluorochemical feedstock "
                "has concentrated supply "
                "and switching constraints."
            ),
            metadata={
                "candidate_node":
                    "COMP_FLUOROCHEMICAL_FEED"
            },
        ),
    )

    return LiveIngestor(
        (
            MockSource(
                events
            ),
        )
    )


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------


ALPHA_OUTPUT_COLUMNS = (
    "ticker",
    "as_of_utc",
    "module",
    "run_id",
    "pipeline_status",
    "raw_score",
    "alpha_z",
    "log_market_cap",
    "momentum_12m",
    "price_to_book",
    "sector",
    "event_count",
    "routed_candidate_count",
    "source_failure_count",
)


def validate_alpha_frame(
    frame: pd.DataFrame,
) -> None:

    if tuple(
        frame.columns
    ) != ALPHA_OUTPUT_COLUMNS:
        raise RuntimeError(
            "Alpha output schema mismatch"
        )

    if frame.empty:
        raise RuntimeError(
            "Alpha output is empty"
        )

    if frame[
        "ticker"
    ].duplicated().any():
        raise RuntimeError(
            "Duplicate ticker in output"
        )

    numeric_columns = (
        "raw_score",
        "alpha_z",
        "log_market_cap",
        "momentum_12m",
        "price_to_book",
    )

    for column in (
        numeric_columns
    ):
        values = pd.to_numeric(
            frame[
                column
            ],
            errors="coerce",
        ).to_numpy(
            dtype=float
        )

        if not np.all(
            np.isfinite(
                values
            )
        ):
            raise RuntimeError(
                f"Non-finite values in "
                f"{column}"
            )


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def run_pipeline(
    *,
    mode: str,
    config: Optional[
        LivePipelineConfig
    ] = None,
    ingestor: Optional[
        LiveIngestor
    ] = None,
    factor_frame: Optional[
        pd.DataFrame
    ] = None,
    structured_client: Optional[
        StructuredLLMClient
    ] = None,
    as_of_utc: Optional[
        datetime
    ] = None,
) -> Mapping[
    str,
    Any,
]:

    cfg = (
        config
        or LivePipelineConfig()
    )

    if mode not in (
        "mock",
        "live",
    ):
        raise ValueError(
            "mode must be 'mock' or 'live'"
        )

    started_perf = (
        time.perf_counter()
    )

    started_utc = (
        as_of_utc
        or datetime.now(
            timezone.utc
        )
    )

    if started_utc.tzinfo is None:
        started_utc = (
            started_utc.replace(
                tzinfo=timezone.utc
            )
        )

    run_id = (
        started_utc.strftime(
            "%Y%m%dT%H%M%SZ"
        )
    )

    step_seconds: Dict[
        str,
        float,
    ] = {}

    graph = build_seed_graph()

    # ----------------------------------------------------------------------
    # Ingestion
    # ----------------------------------------------------------------------

    t0 = time.perf_counter()

    if ingestor is None:
        ingestor = (
            build_mock_ingestor(
                as_of_date=(
                    started_utc.date()
                )
            )
            if mode == "mock"
            else LiveIngestor.default()
        )

    batch = ingestor.collect(
        start_date=(
            started_utc.date()
            - timedelta(
                days=(
                    cfg.lookback_days
                    - 1
                )
            )
        ),
        end_date=(
            started_utc.date()
        ),
    )

    step_seconds[
        "ingestion"
    ] = (
        time.perf_counter()
        - t0
    )

    # ----------------------------------------------------------------------
    # Routing
    # ----------------------------------------------------------------------

    t0 = time.perf_counter()

    router = (
        PhysicalNodeRouter()
    )

    routed = router.route(
        batch.events,
        graph,
    )

    step_seconds[
        "routing"
    ] = (
        time.perf_counter()
        - t0
    )

    # ----------------------------------------------------------------------
    # Factors / universe
    # ----------------------------------------------------------------------

    t0 = time.perf_counter()

    if factor_frame is None:
        factor_frame = (
            build_mock_factor_frame()
            if mode == "mock"
            else (
                Nifty500FactorProvider()
                .build()
            )
        )

    factors = (
        factor_frame.copy()
    )

    required_factor_columns = {
        "log_market_cap",
        "momentum_12m",
        "price_to_book",
        "sector",
    }

    if not required_factor_columns.issubset(
        factors.columns
    ):
        raise RuntimeError(
            "Factor frame missing "
            "required columns"
        )

    if not factors.index.is_unique:
        raise RuntimeError(
            "Factor-frame ticker index "
            "must be unique"
        )

    factor_diagnostics = dict(
        getattr(
            factor_frame,
            "attrs",
            {},
        ).get(
            "factor_diagnostics",
            {},
        )
    )

    step_seconds[
        "factor_data"
    ] = (
        time.perf_counter()
        - t0
    )

    # ----------------------------------------------------------------------
    # Feature extraction + graph propagation
    # ----------------------------------------------------------------------

    t0 = time.perf_counter()

    client = structured_client

    if client is None:
        client = (
            DeterministicMockClient()
            if mode == "mock"
            else OpenAIStructuredClient(
                model=(
                    os.getenv(
                        "OPENAI_MODEL",
                        cfg.openai_model,
                    )
                )
            )
        )

    agent = (
        QuantamentalFeatureAgent(
            client
        )
    )

    raw_scores: Dict[
        str,
        float,
    ] = {
        ticker: 0.0
        for ticker
        in factors.index
    }

    active_candidates = 0

    for candidate_number, (
        node_id,
        candidate_events,
    ) in enumerate(
        sorted(
            routed.items()
        ),
        start=1,
    ):

        node = graph.get_node(
            node_id
        )

        alias = (
            f"Candidate_"
            f"{candidate_number:03d}"
        )

        chunks = []

        company_aliases = {}

        for event_number, event in enumerate(
            candidate_events,
            start=1,
        ):

            company_name = (
                event.metadata.get(
                    "company_name",
                    "",
                )
                .strip()
            )

            if company_name:
                company_aliases[
                    company_name
                ] = (
                    f"Issuer_"
                    f"{event_number:03d}"
                )

            chunks.append(
                (
                    f"Source: "
                    f"{event.source.value}\n"
                    f"Date: "
                    f"{event.published_date.isoformat()}\n"
                    f"Title: "
                    f"{event.title}\n"
                    f"Text: "
                    f"{event.text}"
                )
            )

        source_text = (
            "\n\n---\n\n"
            .join(
                chunks
            )[
                :cfg.max_chars_per_candidate
            ]
        )

        anonymizer = (
            PITAnonymizer(
                company_aliases=(
                    company_aliases
                )
            )
        )

        document = (
            anonymizer.anonymize(
                source_text
            )
        )

        assessment = (
            agent.evaluate(
                document=document,
                candidate=(
                    CandidateNode(
                        anonymized_id=(
                            alias
                        ),
                        layer=int(
                            node.layer
                        ),
                        physical_role=(
                            f"{node.label} "
                            f"physical supply-chain "
                            f"choke point."
                        ),
                    )
                ),
            )
        )

        score = float(
            assessment
            .raw_bottleneck_score
        )

        if (
            abs(score)
            <= cfg.signal_threshold
        ):
            continue

        active_candidates += 1

        propagated = (
            graph.propagate_shock(
                node_id,
                score,
            )
        )

        if not propagated.ok:
            raise RuntimeError(
                propagated.error
                or (
                    "Graph propagation "
                    "failed"
                )
            )

        for (
            ticker,
            contribution,
        ) in (
            propagated
            .terminal_shocks
            .items()
        ):

            if ticker not in raw_scores:
                continue

            raw_scores[
                ticker
            ] += float(
                contribution
            )

    step_seconds[
        "feature_and_graph"
    ] = (
        time.perf_counter()
        - t0
    )

    # ----------------------------------------------------------------------
    # Factor orthogonalization
    # ----------------------------------------------------------------------

    t0 = time.perf_counter()

    cross_section = (
        factors[
            [
                "log_market_cap",
                "momentum_12m",
                "price_to_book",
                "sector",
            ]
        ]
        .copy()
    )

    cross_section[
        "raw_score"
    ] = pd.Series(
        raw_scores
    ).reindex(
        cross_section.index
    ).astype(float)

    raw_array = (
        cross_section[
            "raw_score"
        ]
        .to_numpy(
            dtype=float
        )
    )

    if (
        np.std(
            raw_array,
            ddof=0,
        )
        <= np.finfo(float).eps
    ):
        # Mathematically there is no standardized cross-sectional alpha when
        # the entire raw vector is identical. Output the neutral all-zero
        # vector rather than manufacturing variation.
        alpha_z = pd.Series(
            0.0,
            index=(
                cross_section.index
            ),
            name="alpha_z",
        )

        pipeline_status = (
            "no_signal"
        )

        neutrality_diagnostics = None

    else:

        neutralizer = (
            FactorNeutralizer()
        )

        result = (
            neutralizer
            .neutralize(
                cross_section[
                    [
                        "raw_score",
                        "log_market_cap",
                        "momentum_12m",
                        "price_to_book",
                        "sector",
                    ]
                ]
            )
        )

        alpha_z = (
            result.alpha_z
        )

        pipeline_status = "ok"

        neutrality_diagnostics = {
            "max_abs_factor_correlation":
                result
                .diagnostics
                .max_abs_final_factor_correlation,

            "final_mean":
                result
                .diagnostics
                .final_mean,

            "final_std":
                result
                .diagnostics
                .final_std,

            "design_rank":
                result
                .diagnostics
                .design_rank,

            "design_columns":
                result
                .diagnostics
                .n_design_columns,
        }

    step_seconds[
        "neutralization"
    ] = (
        time.perf_counter()
        - t0
    )

    # ----------------------------------------------------------------------
    # Persist
    # ----------------------------------------------------------------------

    t0 = time.perf_counter()

    output = pd.DataFrame(
        {
            "ticker":
                cross_section.index
                .astype(str),

            "as_of_utc":
                pd.Timestamp(
                    started_utc
                ),

            "module":
                MODULE_NAME,

            "run_id":
                run_id,

            "pipeline_status":
                pipeline_status,

            "raw_score":
                cross_section[
                    "raw_score"
                ]
                .to_numpy(
                    dtype=float
                ),

            "alpha_z":
                alpha_z
                .reindex(
                    cross_section.index
                )
                .to_numpy(
                    dtype=float
                ),

            "log_market_cap":
                cross_section[
                    "log_market_cap"
                ]
                .to_numpy(
                    dtype=float
                ),

            "momentum_12m":
                cross_section[
                    "momentum_12m"
                ]
                .to_numpy(
                    dtype=float
                ),

            "price_to_book":
                cross_section[
                    "price_to_book"
                ]
                .to_numpy(
                    dtype=float
                ),

            "sector":
                cross_section[
                    "sector"
                ]
                .astype(str)
                .to_numpy(),

            "event_count":
                len(
                    batch.events
                ),

            "routed_candidate_count":
                len(
                    routed
                ),

            "source_failure_count":
                len(
                    batch.source_failures
                ),
        },
        columns=(
            ALPHA_OUTPUT_COLUMNS
        ),
    )

    validate_alpha_frame(
        output
    )

    output_path = Path(
        cfg.output_path
    )

    metadata_path = Path(
        cfg.metadata_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    persistence_started = (
    time.perf_counter()
    )

    _atomic_write_parquet(
        output,
        output_path,
    )

    step_seconds[
        "persistence"
    ] = (
        time.perf_counter()
        - persistence_started
    )

    total_duration = (
        time.perf_counter()
        - started_perf
    )

    metadata = {
        "module":
            MODULE_NAME,

        "run_id":
            run_id,

        "mode":
            mode,

        "pipeline_status":
            pipeline_status,

        "as_of_utc":
            started_utc
            .isoformat(),

        "event_count":
            len(
                batch.events
            ),

        "source_failures":
            dict(
                batch.source_failures
            ),

        "routed_candidate_count":
            len(
                routed
            ),

        "active_candidate_count":
            active_candidates,

        "output_rows":
            len(
                output
            ),

        "alpha_mean":
            float(
                output[
                    "alpha_z"
                ].mean()
            ),

        "alpha_std":
            float(
                np.std(
                    output[
                        "alpha_z"
                    ].to_numpy(),
                    ddof=0,
                )
            ),

        "factor_diagnostics":
            factor_diagnostics,

        "neutrality_diagnostics":
            neutrality_diagnostics,

        "step_seconds":
            {
                key:
                    float(
                        value
                    )
                for (
                    key,
                    value,
                )
                in step_seconds.items()
            },

        "total_duration_seconds":
            float(
                total_duration
            ),

        "output_path":
            str(
                output_path
            ),
    }

    _atomic_write_json(
        metadata,
        metadata_path,
    )

    return metadata


# ---------------------------------------------------------------------------
# Atomic persistence
# ---------------------------------------------------------------------------


def _atomic_write_parquet(
    frame: pd.DataFrame,
    path: Path,
) -> None:

    temporary = path.with_name(
        f".{path.name}.tmp"
    )

    frame.to_parquet(
        temporary,
        index=False,
        engine="pyarrow",
    )

    os.replace(
        temporary,
        path,
    )


def _atomic_write_json(
    payload: Mapping[
        str,
        Any,
    ],
    path: Path,
) -> None:

    temporary = path.with_name(
        f".{path.name}.tmp"
    )

    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            default=str,
        ),
        encoding="utf-8",
    )

    os.replace(
        temporary,
        path,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=[
            "mock",
            "live",
        ],
        required=True,
    )

    parser.add_argument(
        "--output",
        default=(
            "data/"
            "latest_alpha_factors.parquet"
        ),
    )

    parser.add_argument(
        "--metadata",
        default=(
            "data/"
            "pipeline_metadata.json"
        ),
    )

    args = parser.parse_args()

    config = (
        LivePipelineConfig(
            output_path=(
                args.output
            ),
            metadata_path=(
                args.metadata
            ),
        )
    )

    metadata = run_pipeline(
        mode=args.mode,
        config=config,
    )

    print(
        json.dumps(
            metadata,
            indent=2,
            sort_keys=True,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
