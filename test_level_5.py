from __future__ import annotations

from datetime import (
    datetime,
    timezone,
)
import importlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import pytest
import requests

from src.live_pipeline import (
    ALPHA_OUTPUT_COLUMNS,
    DeterministicMockClient,
    LivePipelineConfig,
    build_mock_factor_frame,
    build_mock_ingestor,
    run_pipeline,
    validate_alpha_frame,
)


@pytest.fixture
def fixed_time():
    return datetime(
        2026,
        9,
        9,
        10,
        30,
        tzinfo=timezone.utc,
    )


@pytest.fixture
def output_paths(
    tmp_path,
):
    return (
        tmp_path
        / "latest_alpha_factors.parquet",

        tmp_path
        / "pipeline_metadata.json",
    )


def test_mock_ingestion_contains_unique_events(
    fixed_time,
):
    ingestor = (
        build_mock_ingestor(
            as_of_date=(
                fixed_time.date()
            )
        )
    )

    batch = (
        ingestor.collect(
            start_date=(
                fixed_time.date()
            ),
            end_date=(
                fixed_time.date()
            ),
        )
    )

    assert (
        len(
            batch.events
        )
        == 3
    )

    event_ids = [
        event.event_id
        for event
        in batch.events
    ]

    assert (
        len(
            event_ids
        )
        == len(
            set(
                event_ids
            )
        )
    )

    assert (
        batch.source_failures
        == {}
    )


def test_mock_factor_frame_is_full_500_cross_section():
    frame = (
        build_mock_factor_frame()
    )

    assert len(
        frame
    ) == 500

    assert frame.index.is_unique

    assert {
        "TRIL.NS",
        "VOLTAMP.NS",
        "BORORENEW.NS",
        "SRF.NS",
        "NAVINFLUOR.NS",
        "AARTIIND.NS",
    }.issubset(
        set(
            frame.index
        )
    )


def test_end_to_end_mock_pipeline_creates_artifacts(
    output_paths,
    fixed_time,
):
    (
        parquet_path,
        metadata_path,
    ) = output_paths

    config = (
        LivePipelineConfig(
            output_path=str(
                parquet_path
            ),
            metadata_path=str(
                metadata_path
            ),
        )
    )

    metadata = (
        run_pipeline(
            mode="mock",
            config=config,
            as_of_utc=fixed_time,
        )
    )

    assert (
        parquet_path.exists()
    )

    assert (
        metadata_path.exists()
    )

    assert (
        metadata[
            "pipeline_status"
        ]
        == "ok"
    )


def test_parquet_schema_integrity(
    output_paths,
    fixed_time,
):
    (
        parquet_path,
        metadata_path,
    ) = output_paths

    run_pipeline(
        mode="mock",
        config=(
            LivePipelineConfig(
                output_path=str(
                    parquet_path
                ),
                metadata_path=str(
                    metadata_path
                ),
            )
        ),
        as_of_utc=fixed_time,
    )

    frame = pd.read_parquet(
        parquet_path
    )

    assert tuple(
        frame.columns
    ) == ALPHA_OUTPUT_COLUMNS

    validate_alpha_frame(
        frame
    )

    assert len(
        frame
    ) == 500

    assert (
        frame[
            "ticker"
        ]
        .nunique()
        == 500
    )


def test_final_mock_alpha_is_standardized(
    output_paths,
    fixed_time,
):
    (
        parquet_path,
        metadata_path,
    ) = output_paths

    run_pipeline(
        mode="mock",
        config=(
            LivePipelineConfig(
                output_path=str(
                    parquet_path
                ),
                metadata_path=str(
                    metadata_path
                ),
            )
        ),
        as_of_utc=fixed_time,
    )

    frame = pd.read_parquet(
        parquet_path
    )

    alpha = (
        frame[
            "alpha_z"
        ]
        .to_numpy(
            dtype=float
        )
    )

    assert abs(
        float(
            np.mean(
                alpha
            )
        )
    ) < 1.0e-10

    assert float(
        np.std(
            alpha,
            ddof=0,
        )
    ) == pytest.approx(
        1.0,
        abs=1.0e-10,
    )


def test_pipeline_output_remains_factor_neutral(
    output_paths,
    fixed_time,
):
    (
        parquet_path,
        metadata_path,
    ) = output_paths

    run_pipeline(
        mode="mock",
        config=(
            LivePipelineConfig(
                output_path=str(
                    parquet_path
                ),
                metadata_path=str(
                    metadata_path
                ),
            )
        ),
        as_of_utc=fixed_time,
    )

    frame = pd.read_parquet(
        parquet_path
    )

    alpha = (
        frame[
            "alpha_z"
        ]
        .to_numpy(
            dtype=float
        )
    )

    for factor in (
        "log_market_cap",
        "momentum_12m",
        "price_to_book",
    ):
        correlation = float(
            np.corrcoef(
                alpha,
                frame[
                    factor
                ].to_numpy(
                    dtype=float
                ),
            )[0, 1]
        )

        assert abs(
            correlation
        ) < 1.0e-8

    for sector in (
        frame[
            "sector"
        ].unique()
    ):

        dummy = (
            frame[
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

        assert abs(
            correlation
        ) < 1.0e-8


def test_pipeline_generates_no_qualitative_pick_columns(
    output_paths,
    fixed_time,
):
    (
        parquet_path,
        metadata_path,
    ) = output_paths

    run_pipeline(
        mode="mock",
        config=(
            LivePipelineConfig(
                output_path=str(
                    parquet_path
                ),
                metadata_path=str(
                    metadata_path
                ),
            )
        ),
        as_of_utc=fixed_time,
    )

    frame = pd.read_parquet(
        parquet_path
    )

    prohibited = {
        "pick",
        "recommendation",
        "buy",
        "sell",
        "action",
        "target_price",
    }

    assert prohibited.isdisjoint(
        {
            column.casefold()
            for column
            in frame.columns
        }
    )


def test_metadata_contains_all_pipeline_stages(
    output_paths,
    fixed_time,
):
    (
        parquet_path,
        metadata_path,
    ) = output_paths

    run_pipeline(
        mode="mock",
        config=(
            LivePipelineConfig(
                output_path=str(
                    parquet_path
                ),
                metadata_path=str(
                    metadata_path
                ),
            )
        ),
        as_of_utc=fixed_time,
    )

    metadata = json.loads(
        metadata_path.read_text(
            encoding="utf-8"
        )
    )

    stages = (
        metadata[
            "step_seconds"
        ]
    )

    assert {
        "ingestion",
        "routing",
        "factor_data",
        "feature_and_graph",
        "neutralization",
    }.issubset(
        stages
    )

    assert all(
        value >= 0.0
        for value
        in stages.values()
    )


def test_mock_pipeline_finishes_inside_ci_runtime_budget(
    output_paths,
    fixed_time,
):
    (
        parquet_path,
        metadata_path,
    ) = output_paths

    started = (
        time.perf_counter()
    )

    run_pipeline(
        mode="mock",
        config=(
            LivePipelineConfig(
                output_path=str(
                    parquet_path
                ),
                metadata_path=str(
                    metadata_path
                ),
            )
        ),
        as_of_utc=fixed_time,
    )

    duration = (
        time.perf_counter()
        - started
    )

    # Dry-run gate is deliberately far below the GitHub job timeout.
    assert duration < 30.0


def test_mock_mode_does_not_require_openai_api_key(
    monkeypatch,
    output_paths,
    fixed_time,
):
    monkeypatch.delenv(
        "OPENAI_API_KEY",
        raising=False,
    )

    (
        parquet_path,
        metadata_path,
    ) = output_paths

    metadata = run_pipeline(
        mode="mock",
        config=(
            LivePipelineConfig(
                output_path=str(
                    parquet_path
                ),
                metadata_path=str(
                    metadata_path
                ),
            )
        ),
        as_of_utc=fixed_time,
    )

    assert (
        metadata[
            "pipeline_status"
        ]
        == "ok"
    )


def test_mock_mode_makes_no_http_requests(
    monkeypatch,
    output_paths,
    fixed_time,
):
    def forbidden(
        *args,
        **kwargs,
    ):
        raise AssertionError(
            "Network call attempted "
            "during mock mode"
        )

    monkeypatch.setattr(
        requests.Session,
        "request",
        forbidden,
    )

    monkeypatch.setattr(
        requests,
        "get",
        forbidden,
    )

    (
        parquet_path,
        metadata_path,
    ) = output_paths

    run_pipeline(
        mode="mock",
        config=(
            LivePipelineConfig(
                output_path=str(
                    parquet_path
                ),
                metadata_path=str(
                    metadata_path
                ),
            )
        ),
        as_of_utc=fixed_time,
    )

    assert (
        parquet_path.exists()
    )


def test_output_is_deterministic_for_fixed_mock_inputs(
    tmp_path,
    fixed_time,
):
    first_parquet = (
        tmp_path
        / "first.parquet"
    )

    first_metadata = (
        tmp_path
        / "first.json"
    )

    second_parquet = (
        tmp_path
        / "second.parquet"
    )

    second_metadata = (
        tmp_path
        / "second.json"
    )

    run_pipeline(
        mode="mock",
        config=(
            LivePipelineConfig(
                output_path=str(
                    first_parquet
                ),
                metadata_path=str(
                    first_metadata
                ),
            )
        ),
        as_of_utc=fixed_time,
    )

    run_pipeline(
        mode="mock",
        config=(
            LivePipelineConfig(
                output_path=str(
                    second_parquet
                ),
                metadata_path=str(
                    second_metadata
                ),
            )
        ),
        as_of_utc=fixed_time,
    )

    first = (
        pd.read_parquet(
            first_parquet
        )
    )

    second = (
        pd.read_parquet(
            second_parquet
        )
    )

    pd.testing.assert_frame_equal(
        first,
        second,
        check_exact=True,
    )


def test_streamlit_dashboard_module_imports():
    module = (
        importlib.import_module(
            "app"
        )
    )

    assert callable(
        module.main
    )
