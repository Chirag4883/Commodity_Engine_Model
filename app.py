from __future__ import annotations

"""
Streamlit monitoring interface for Module 1.

This dashboard visualizes quantitative outputs and graph structure.
It does not generate discretionary stock recommendations.
"""

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import streamlit as st

from src.graph_engine import (
    Layer,
    build_seed_graph,
)


DEFAULT_ALPHA_PATH = (
    "data/"
    "latest_alpha_factors.parquet"
)

DEFAULT_METADATA_PATH = (
    "data/"
    "pipeline_metadata.json"
)

DEFAULT_IC_PATH = (
    "data/"
    "rank_ic_history.parquet"
)


@st.cache_data
def load_alpha(
    path: str,
) -> pd.DataFrame:

    frame = pd.read_parquet(
        path
    )

    return frame


@st.cache_data
def load_rank_ic(
    path: str,
) -> pd.DataFrame:

    return pd.read_parquet(
        path
    )


@st.cache_data
def load_metadata(
    path: str,
) -> dict:

    return json.loads(
        Path(
            path
        ).read_text(
            encoding="utf-8"
        )
    )


def render_alpha_distribution(
    frame: pd.DataFrame,
) -> None:

    st.subheader(
        "Cross-Sectional Alpha Distribution"
    )

    figure, axis = plt.subplots(
        figsize=(
            9,
            4,
        )
    )

    axis.hist(
        frame[
            "alpha_z"
        ].to_numpy(),
        bins=40,
    )

    axis.axvline(
        0.0,
        linewidth=1.0,
    )

    axis.set_xlabel(
        "Module-1 residual alpha Z-score"
    )

    axis.set_ylabel(
        "Number of securities"
    )

    axis.set_title(
        "NSE 500 Cross-Section"
    )

    st.pyplot(
        figure,
        clear_figure=True,
    )


def render_raw_vs_neutralized(
    frame: pd.DataFrame,
) -> None:

    st.subheader(
        "Raw vs Neutralized Signal"
    )

    chart = (
        frame[
            [
                "raw_score",
                "alpha_z",
            ]
        ]
        .copy()
    )

    st.scatter_chart(
        chart,
        x="raw_score",
        y="alpha_z",
    )


def render_graph() -> None:

    st.subheader(
        "Supply-Chain Knowledge Graph"
    )

    graph_engine = (
        build_seed_graph()
    )

    graph = (
        graph_engine
        .nx_graph
    )

    positions = {}

    for layer in Layer:

        nodes = [
            node_id
            for node_id
            in graph.nodes
            if (
                graph_engine
                .get_node(
                    node_id
                )
                .layer
                == layer
            )
        ]

        y_values = (
            np.linspace(
                -1.0,
                1.0,
                max(
                    len(nodes),
                    1,
                ),
            )
        )

        for (
            node_id,
            y_value,
        ) in zip(
            nodes,
            y_values,
        ):
            positions[
                node_id
            ] = (
                float(
                    layer
                ),
                float(
                    y_value
                ),
            )

    figure, axis = plt.subplots(
        figsize=(
            15,
            8,
        )
    )

    nx.draw_networkx_edges(
        graph,
        positions,
        ax=axis,
        arrows=True,
        arrowsize=12,
        width=0.8,
    )

    nx.draw_networkx_nodes(
        graph,
        positions,
        ax=axis,
        node_size=500,
    )

    labels = {
        node_id:
            graph_engine
            .get_node(
                node_id
            )
            .label
        for node_id
        in graph.nodes
    }

    nx.draw_networkx_labels(
        graph,
        positions,
        labels=labels,
        ax=axis,
        font_size=6,
    )

    axis.set_axis_off()

    axis.set_title(
        "L0 Raw Materials → "
        "L4 Listed Equity Exposures"
    )

    st.pyplot(
        figure,
        clear_figure=True,
    )


def render_rank_ic(
    path: str,
) -> None:

    st.subheader(
        "Historical Rank IC Tracking"
    )

    if not Path(
        path
    ).exists():

        st.info(
            "No persisted Rank-IC history "
            "is available yet. "
            "The dashboard will display it "
            "when Level-3 empirical runs "
            "are persisted to "
            "rank_ic_history.parquet."
        )

        return

    frame = load_rank_ic(
        path
    )

    required = {
        "as_of_date",
        "horizon_days",
        "rank_ic",
    }

    if not required.issubset(
        frame.columns
    ):

        st.warning(
            "Rank-IC history file has "
            "an unexpected schema."
        )

        return

    frame = frame.copy()

    frame[
        "as_of_date"
    ] = pd.to_datetime(
        frame[
            "as_of_date"
        ]
    )

    pivot = frame.pivot_table(
        index="as_of_date",
        columns="horizon_days",
        values="rank_ic",
        aggfunc="mean",
    )

    st.line_chart(
        pivot
    )


def main() -> None:

    st.set_page_config(
        page_title=(
            "Commodity Choke-Point "
            "Alpha Monitor"
        ),
        layout="wide",
    )

    st.title(
        "Module 1 — "
        "Supply-Chain & Commodity "
        "Choke-Point Alpha"
    )

    st.caption(
        "Quantitative monitoring only. "
        "Outputs are cross-sectional "
        "factor scores, not discretionary "
        "stock recommendations."
    )

    alpha_path = os.getenv(
        "ALPHA_FILE",
        DEFAULT_ALPHA_PATH,
    )

    metadata_path = os.getenv(
        "PIPELINE_METADATA_FILE",
        DEFAULT_METADATA_PATH,
    )

    ic_path = os.getenv(
        "RANK_IC_FILE",
        DEFAULT_IC_PATH,
    )

    if not Path(
        alpha_path
    ).exists():

        st.error(
            f"Missing alpha artifact: "
            f"{alpha_path}"
        )

        st.stop()

    frame = load_alpha(
        alpha_path
    )

    metadata = {}

    if Path(
        metadata_path
    ).exists():
        metadata = (
            load_metadata(
                metadata_path
            )
        )

    columns = st.columns(
        5
    )

    columns[0].metric(
        "Cross-section",
        f"{len(frame):,}",
    )

    columns[1].metric(
        "Events ingested",
        int(
            frame[
                "event_count"
            ].iloc[0]
        ),
    )

    columns[2].metric(
        "Routed nodes",
        int(
            frame[
                "routed_candidate_count"
            ].iloc[0]
        ),
    )

    columns[3].metric(
        "Alpha mean",
        f"{frame['alpha_z'].mean():.3e}",
    )

    columns[4].metric(
        "Alpha σ",
        (
            f"{np.std(frame['alpha_z'], ddof=0):.4f}"
        ),
    )

    status = (
        frame[
            "pipeline_status"
        ]
        .iloc[0]
    )

    st.write(
        f"Pipeline status: "
        f"**{status}**"
    )

    render_alpha_distribution(
        frame
    )

    render_raw_vs_neutralized(
        frame
    )

    render_graph()

    render_rank_ic(
        ic_path
    )

    st.subheader(
        "Full Cross-Sectional Factor Array"
    )

    display_columns = [
        "ticker",
        "raw_score",
        "alpha_z",
        "log_market_cap",
        "momentum_12m",
        "price_to_book",
        "sector",
    ]

    st.dataframe(
        frame[
            display_columns
        ]
        .sort_values(
            "ticker"
        ),
        use_container_width=True,
        hide_index=True,
    )

    if metadata:

        with st.expander(
            "Pipeline diagnostics"
        ):
            st.json(
                metadata
            )


if __name__ == "__main__":
    main()
