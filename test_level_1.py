import math

import networkx as nx
import pytest

from src.graph_engine import (
    Layer,
    build_seed_graph,
)


@pytest.fixture(scope="module")
def graph():
    return build_seed_graph()


def test_seed_graph_is_valid(graph):
    """
    Master validation gate.
    """
    report = graph.validate()

    assert report.is_valid, (
        "Seed graph validation failed:\n"
        + "\n".join(report.errors)
    )


def test_graph_is_directed_acyclic_graph(graph):
    """
    Required Level-1 gate:
    graph must be a DAG.
    """
    nx_graph = graph.nx_graph

    assert nx.is_directed(nx_graph)
    assert nx.is_directed_acyclic_graph(nx_graph)


def test_every_edge_advances_exactly_one_layer(graph):
    """
    Required Level-1 gate:
        L0 -> L1 -> L2 -> L3 -> L4

    No layer skipping and no same-layer/backward edges.
    """
    nx_graph = graph.nx_graph

    for source, target in nx_graph.edges:
        source_layer = graph.get_node(source).layer
        target_layer = graph.get_node(target).layer

        assert int(target_layer) == int(source_layer) + 1, (
            f"Invalid layer transition: "
            f"{source} L{int(source_layer)} -> "
            f"{target} L{int(target_layer)}"
        )


def test_equity_nodes_are_terminal(graph):
    """
    Layer-4 listed equities must have no downstream children.
    """
    nx_graph = graph.nx_graph

    for node_id in nx_graph.nodes:
        node = graph.get_node(node_id)

        if node.layer == Layer.LISTED_EQUITY:
            assert nx_graph.out_degree(node_id) == 0


def test_fluorspar_to_srf_has_full_layer_continuity(graph):
    expected_path = (
        "RAW_FLUORSPAR",
        "INT_AHF",
        "COMP_FLUOROCHEMICAL_FEED",
        "SUB_FLUOROCHEMICAL_OPERATIONS",
        "SRF.NS",
    )

    paths = graph.path_transmission_weights(
        "RAW_FLUORSPAR",
        "SRF.NS",
    )

    assert len(paths) == 1
    assert paths[0].path == expected_path

    layers = [
        graph.get_node(node_id).layer
        for node_id in expected_path
    ]

    assert layers == [
        Layer.RAW_FEEDSTOCK,
        Layer.REFINED_INTERMEDIATE,
        Layer.CRITICAL_COMPONENT,
        Layer.SUBASSEMBLY,
        Layer.LISTED_EQUITY,
    ]


def test_fluorspar_to_navinfluor_has_full_layer_continuity(graph):
    expected_path = (
        "RAW_FLUORSPAR",
        "INT_AHF",
        "COMP_FLUOROCHEMICAL_FEED",
        "SUB_FLUOROCHEMICAL_OPERATIONS",
        "NAVINFLUOR.NS",
    )

    paths = graph.path_transmission_weights(
        "RAW_FLUORSPAR",
        "NAVINFLUOR.NS",
    )

    assert len(paths) == 1
    assert paths[0].path == expected_path


def test_srf_transmission_weight_matches_analytical_formula(graph):
    """
    Required Level-1 mathematical gate.

    RAW_FLUORSPAR -> INT_AHF
        (1 - 0.20) * 0.75 = 0.60

    INT_AHF -> COMP_FLUOROCHEMICAL_FEED
        (1 - 0.10) * 0.80 = 0.72

    COMP_FLUOROCHEMICAL_FEED -> SUB_FLUOROCHEMICAL_OPERATIONS
        (1 - 0.25) * 0.60 = 0.45

    SUB_FLUOROCHEMICAL_OPERATIONS -> SRF.NS
        (1 - 0.10) * 0.50 = 0.45
    """
    expected = (
        ((1.0 - 0.20) * 0.75)
        * ((1.0 - 0.10) * 0.80)
        * ((1.0 - 0.25) * 0.60)
        * ((1.0 - 0.10) * 0.50)
    )

    assert expected == pytest.approx(
        0.08748,
        rel=0.0,
        abs=1e-12,
    )

    actual = graph.transmission_weight(
        "RAW_FLUORSPAR",
        "SRF.NS",
    )

    assert actual == pytest.approx(
        expected,
        rel=0.0,
        abs=1e-12,
    )


def test_navinfluor_transmission_weight_matches_analytical_formula(graph):
    """
    Required Level-1 mathematical gate.
    """
    expected = (
        ((1.0 - 0.20) * 0.75)
        * ((1.0 - 0.10) * 0.80)
        * ((1.0 - 0.25) * 0.60)
        * ((1.0 - 0.05) * 0.55)
    )

    assert expected == pytest.approx(
        0.101574,
        rel=0.0,
        abs=1e-12,
    )

    actual = graph.transmission_weight(
        "RAW_FLUORSPAR",
        "NAVINFLUOR.NS",
    )

    assert actual == pytest.approx(
        expected,
        rel=0.0,
        abs=1e-12,
    )


def test_non_unit_fluorspar_shock_propagates_correctly(graph):
    """
    Confirm:
        downstream_shock
        = transmission_weight * source_shock
    """
    source_shock = 0.80

    result = graph.propagate_shock(
        "RAW_FLUORSPAR",
        source_shock,
    )

    assert result.ok
    assert result.error is None

    expected_srf = source_shock * 0.08748
    expected_navin = source_shock * 0.101574

    assert result.terminal_shocks["SRF.NS"] == pytest.approx(
        expected_srf,
        rel=0.0,
        abs=1e-12,
    )

    assert result.terminal_shocks["NAVINFLUOR.NS"] == pytest.approx(
        expected_navin,
        rel=0.0,
        abs=1e-12,
    )

    assert result.terminal_shocks["SRF.NS"] == pytest.approx(
        0.069984,
        rel=0.0,
        abs=1e-12,
    )

    assert result.terminal_shocks["NAVINFLUOR.NS"] == pytest.approx(
        0.0812592,
        rel=0.0,
        abs=1e-12,
    )


def test_unrelated_equities_do_not_receive_fluorspar_shock(graph):
    result = graph.propagate_shock(
        "RAW_FLUORSPAR",
        1.0,
    )

    assert result.ok

    assert "TRIL.NS" not in result.terminal_shocks
    assert "VOLTAMP.NS" not in result.terminal_shocks
    assert "BORORENEW.NS" not in result.terminal_shocks
    assert "AARTIIND.NS" not in result.terminal_shocks


def test_invalid_source_node_fails_gracefully(graph):
    """
    Required Level-1 gate:
    invalid node must not generate an unhandled exception.
    """
    result = graph.propagate_shock(
        "RAW_NOT_A_REAL_COMMODITY",
        1.0,
    )

    assert result.ok is False
    assert result.node_shocks == {}
    assert result.terminal_shocks == {}
    assert result.error is not None
    assert "Unknown supply-chain node" in result.error


@pytest.mark.parametrize(
    "invalid_magnitude",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_invalid_shock_magnitude_fails_gracefully(
    graph,
    invalid_magnitude,
):
    result = graph.propagate_shock(
        "RAW_FLUORSPAR",
        invalid_magnitude,
    )

    assert result.ok is False
    assert result.error is not None
    assert result.node_shocks == {}
    assert result.terminal_shocks == {}


def test_zero_shock_remains_zero_downstream(graph):
    result = graph.propagate_shock(
        "RAW_FLUORSPAR",
        0.0,
    )

    assert result.ok

    assert result.node_shocks["RAW_FLUORSPAR"] == 0.0
    assert result.node_shocks["SRF.NS"] == 0.0
    assert result.node_shocks["NAVINFLUOR.NS"] == 0.0


def test_negative_shock_preserves_sign(graph):
    """
    The engine supports signed shocks.

    This matters later when positive and negative supply shocks are represented
    by the same transmission machinery.
    """
    source_shock = -0.50

    result = graph.propagate_shock(
        "RAW_FLUORSPAR",
        source_shock,
    )

    assert result.ok

    assert result.terminal_shocks["SRF.NS"] == pytest.approx(
        source_shock * 0.08748,
        rel=0.0,
        abs=1e-12,
    )

    assert result.terminal_shocks["NAVINFLUOR.NS"] == pytest.approx(
        source_shock * 0.101574,
        rel=0.0,
        abs=1e-12,
    )


def test_all_edge_transmission_factors_are_finite_and_bounded(graph):
    """
    Since:
        substitutability in [0, 1]
        cost_share in [0, 1]

    every edge transmission factor must lie in [0, 1].
    """
    nx_graph = graph.nx_graph

    for source, target in nx_graph.edges:
        edge = graph.get_edge(source, target)

        assert math.isfinite(edge.transmission_factor)
        assert 0.0 <= edge.transmission_factor <= 1.0
