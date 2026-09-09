from __future__ import annotations

"""
Level 1 — Supply-Chain Knowledge Graph Engine

The graph is a directed acyclic graph with strict adjacent-layer continuity:

    Layer 0 -> Layer 1 -> Layer 2 -> Layer 3 -> Layer 4

Shock transmission across one edge e is:

    w_e = (1 - substitutability_e) * cost_share_e

For a path P from source S to target T:

    Shock(T | P) = Shock(S) * product(w_e for e in P)

If multiple distinct DAG paths reach the same node, their contributions are
summed by linear superposition. No clipping or normalization is performed.

This module deliberately does NOT:
- produce stock recommendations,
- generate cross-sectional alpha scores,
- perform factor neutralization,
- fetch market data,
- perform LLM inference.

Those belong to later levels/modules.
"""

from dataclasses import dataclass
from enum import Enum, IntEnum
import math
from typing import Dict, Iterable, Mapping, Optional, Tuple

import networkx as nx


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class GraphEngineError(Exception):
    """Base exception for graph-engine failures."""


class GraphValidationError(GraphEngineError):
    """Raised when a graph invariant is violated."""


class GraphNodeNotFoundError(GraphEngineError):
    """Raised by strict APIs when a node does not exist."""


# ---------------------------------------------------------------------------
# Domain model
# ---------------------------------------------------------------------------


class Layer(IntEnum):
    RAW_FEEDSTOCK = 0
    REFINED_INTERMEDIATE = 1
    CRITICAL_COMPONENT = 2
    SUBASSEMBLY = 3
    LISTED_EQUITY = 4


class NodeKind(str, Enum):
    RAW = "raw_feedstock"
    INTERMEDIATE = "refined_intermediate"
    COMPONENT = "critical_component"
    SUBASSEMBLY = "subassembly"
    EQUITY = "listed_equity"


EXPECTED_KIND_BY_LAYER: Mapping[Layer, NodeKind] = {
    Layer.RAW_FEEDSTOCK: NodeKind.RAW,
    Layer.REFINED_INTERMEDIATE: NodeKind.INTERMEDIATE,
    Layer.CRITICAL_COMPONENT: NodeKind.COMPONENT,
    Layer.SUBASSEMBLY: NodeKind.SUBASSEMBLY,
    Layer.LISTED_EQUITY: NodeKind.EQUITY,
}


@dataclass(frozen=True, slots=True)
class SupplyNode:
    node_id: str
    label: str
    layer: Layer
    kind: NodeKind
    ticker: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.node_id or not self.node_id.strip():
            raise ValueError("node_id must be a non-empty string")

        if not self.label or not self.label.strip():
            raise ValueError("label must be a non-empty string")

        expected_kind = EXPECTED_KIND_BY_LAYER[self.layer]
        if self.kind != expected_kind:
            raise ValueError(
                f"Node {self.node_id!r}: kind={self.kind.value!r} does not "
                f"match layer={int(self.layer)}; expected {expected_kind.value!r}"
            )

        if self.layer == Layer.LISTED_EQUITY:
            if not self.ticker:
                raise ValueError(
                    f"Listed-equity node {self.node_id!r} requires ticker"
                )
        elif self.ticker is not None:
            raise ValueError(
                f"Non-equity node {self.node_id!r} must not specify ticker"
            )


@dataclass(frozen=True, slots=True)
class SupplyEdge:
    source: str
    target: str
    substitutability: float
    cost_share: float
    lead_time_months: float

    def __post_init__(self) -> None:
        for field_name, value in (
            ("substitutability", self.substitutability),
            ("cost_share", self.cost_share),
            ("lead_time_months", self.lead_time_months),
        ):
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(
                    f"{field_name} must be a finite numeric value; got {value!r}"
                )

        if not 0.0 <= self.substitutability <= 1.0:
            raise ValueError(
                "substitutability must lie in [0.0, 1.0]"
            )

        if not 0.0 <= self.cost_share <= 1.0:
            raise ValueError(
                "cost_share must lie in [0.0, 1.0]"
            )

        if self.lead_time_months < 0.0:
            raise ValueError(
                "lead_time_months must be non-negative"
            )

    @property
    def transmission_factor(self) -> float:
        """
        One-edge multiplicative attenuation factor.

        factor = (1 - substitutability) * cost_share
        """
        return (1.0 - self.substitutability) * self.cost_share


@dataclass(frozen=True, slots=True)
class PathTransmission:
    path: Tuple[str, ...]
    transmission_weight: float


@dataclass(frozen=True, slots=True)
class GraphValidationReport:
    is_valid: bool
    errors: Tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PropagationResult:
    ok: bool
    source_node: str
    shock_magnitude: float
    node_shocks: Mapping[str, float]
    terminal_shocks: Mapping[str, float]
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Graph engine
# ---------------------------------------------------------------------------


class SupplyChainGraph:
    """
    Production-oriented wrapper around networkx.DiGraph.

    Invariants
    ----------
    1. Graph must remain directed and acyclic.
    2. Every edge must advance exactly one layer:
           L0 -> L1 -> L2 -> L3 -> L4
    3. Layer-4 nodes must be terminal.
    4. Node kind must match node layer.
    """

    def __init__(
        self,
        nodes: Optional[Iterable[SupplyNode]] = None,
        edges: Optional[Iterable[SupplyEdge]] = None,
    ) -> None:
        self._graph = nx.DiGraph()

        if nodes:
            for node in nodes:
                self.add_node(node)

        if edges:
            for edge in edges:
                self.add_edge(edge)

    @property
    def nx_graph(self) -> nx.DiGraph:
        """
        Return a defensive copy for inspection/testing.

        External callers therefore cannot mutate internal graph state.
        """
        return self._graph.copy()

    @property
    def node_count(self) -> int:
        return self._graph.number_of_nodes()

    @property
    def edge_count(self) -> int:
        return self._graph.number_of_edges()

    def has_node(self, node_id: str) -> bool:
        return self._graph.has_node(node_id)

    def get_node(self, node_id: str) -> SupplyNode:
        if node_id not in self._graph:
            raise GraphNodeNotFoundError(
                f"Unknown supply-chain node: {node_id!r}"
            )
        return self._graph.nodes[node_id]["spec"]

    def get_edge(self, source: str, target: str) -> SupplyEdge:
        if not self._graph.has_edge(source, target):
            raise GraphValidationError(
                f"Edge {source!r} -> {target!r} does not exist"
            )
        return self._graph.edges[source, target]["spec"]

    def add_node(self, node: SupplyNode) -> None:
        if node.node_id in self._graph:
            raise GraphValidationError(
                f"Duplicate node_id: {node.node_id!r}"
            )

        self._graph.add_node(
            node.node_id,
            spec=node,
            label=node.label,
            layer=int(node.layer),
            kind=node.kind.value,
            ticker=node.ticker,
        )

    def add_edge(self, edge: SupplyEdge) -> None:
        if edge.source not in self._graph:
            raise GraphValidationError(
                f"Cannot add edge: source node {edge.source!r} does not exist"
            )

        if edge.target not in self._graph:
            raise GraphValidationError(
                f"Cannot add edge: target node {edge.target!r} does not exist"
            )

        if self._graph.has_edge(edge.source, edge.target):
            raise GraphValidationError(
                f"Duplicate edge: {edge.source!r} -> {edge.target!r}"
            )

        source_node = self.get_node(edge.source)
        target_node = self.get_node(edge.target)

        if int(target_node.layer) - int(source_node.layer) != 1:
            raise GraphValidationError(
                "Layer continuity violation: "
                f"{edge.source!r} is L{int(source_node.layer)}, "
                f"{edge.target!r} is L{int(target_node.layer)}. "
                "Edges must advance exactly one layer."
            )

        self._graph.add_edge(
            edge.source,
            edge.target,
            spec=edge,
            substitutability=edge.substitutability,
            cost_share=edge.cost_share,
            lead_time_months=edge.lead_time_months,
            transmission_factor=edge.transmission_factor,
        )

        # Defensive DAG enforcement during graph construction.
        if not nx.is_directed_acyclic_graph(self._graph):
            self._graph.remove_edge(edge.source, edge.target)
            raise GraphValidationError(
                f"Edge {edge.source!r} -> {edge.target!r} would create a cycle"
            )

    def validate(self) -> GraphValidationReport:
        errors = []

        if not nx.is_directed_acyclic_graph(self._graph):
            errors.append("Graph contains at least one directed cycle.")

        for node_id in self._graph.nodes:
            node = self.get_node(node_id)

            expected_kind = EXPECTED_KIND_BY_LAYER[node.layer]
            if node.kind != expected_kind:
                errors.append(
                    f"{node_id}: kind {node.kind.value!r} inconsistent "
                    f"with layer L{int(node.layer)}."
                )

            if (
                node.layer == Layer.LISTED_EQUITY
                and self._graph.out_degree(node_id) != 0
            ):
                errors.append(
                    f"{node_id}: Layer-4 equity nodes must be terminal."
                )

        for source, target in self._graph.edges:
            source_node = self.get_node(source)
            target_node = self.get_node(target)

            if int(target_node.layer) != int(source_node.layer) + 1:
                errors.append(
                    f"{source} -> {target}: invalid layer transition "
                    f"L{int(source_node.layer)} -> L{int(target_node.layer)}."
                )

            edge = self.get_edge(source, target)

            if not 0.0 <= edge.substitutability <= 1.0:
                errors.append(
                    f"{source} -> {target}: invalid substitutability."
                )

            if not 0.0 <= edge.cost_share <= 1.0:
                errors.append(
                    f"{source} -> {target}: invalid cost_share."
                )

            if edge.lead_time_months < 0.0:
                errors.append(
                    f"{source} -> {target}: invalid lead_time_months."
                )

        return GraphValidationReport(
            is_valid=len(errors) == 0,
            errors=tuple(errors),
        )

    def path_transmission_weights(
        self,
        source_node: str,
        target_node: str,
    ) -> Tuple[PathTransmission, ...]:
        """
        Return the exact multiplicative weight for every DAG path between
        source_node and target_node.

        This is primarily an audit/debugging interface. Production propagation
        uses dynamic programming and does not enumerate all paths.
        """
        self._require_node(source_node)
        self._require_node(target_node)

        paths = []

        for raw_path in nx.all_simple_paths(
            self._graph,
            source=source_node,
            target=target_node,
        ):
            weight = 1.0

            for u, v in zip(raw_path[:-1], raw_path[1:]):
                weight *= self.get_edge(u, v).transmission_factor

            paths.append(
                PathTransmission(
                    path=tuple(raw_path),
                    transmission_weight=weight,
                )
            )

        return tuple(paths)

    def transmission_weight(
        self,
        source_node: str,
        target_node: str,
    ) -> float:
        """
        Aggregate source->target transmission coefficient.

        For one path this equals the exact path product.

        For multiple paths:
            total_weight = sum(path_weight)
        """
        self._require_node(source_node)
        self._require_node(target_node)

        if source_node == target_node:
            return 1.0

        result = self.propagate_shock(
            source_node=source_node,
            shock_magnitude=1.0,
        )

        if not result.ok:
            raise GraphEngineError(
                result.error or "Unexpected propagation failure"
            )

        return float(result.node_shocks.get(target_node, 0.0))

    def propagate_shock(
        self,
        source_node: str,
        shock_magnitude: float,
    ) -> PropagationResult:
        """
        Propagate a shock downstream using multiplicative attenuation.

        Graceful-failure contract
        -------------------------
        Invalid user-supplied source nodes or magnitudes do NOT raise an
        unhandled exception. Instead:

            result.ok == False
            result.error contains a diagnostic string.

        Multiple paths are aggregated by linear superposition.
        """
        if source_node not in self._graph:
            return PropagationResult(
                ok=False,
                source_node=source_node,
                shock_magnitude=shock_magnitude,
                node_shocks={},
                terminal_shocks={},
                error=f"Unknown supply-chain node: {source_node!r}",
            )

        if (
            not isinstance(shock_magnitude, (int, float))
            or not math.isfinite(shock_magnitude)
        ):
            return PropagationResult(
                ok=False,
                source_node=source_node,
                shock_magnitude=shock_magnitude,
                node_shocks={},
                terminal_shocks={},
                error="shock_magnitude must be a finite numeric value",
            )

        shock_magnitude = float(shock_magnitude)

        # Dynamic-programming propagation across topological order.
        #
        # If a node has several parents, all incoming path contributions
        # accumulate in node_shocks[node].
        node_shocks: Dict[str, float] = {
            source_node: shock_magnitude
        }

        topological_order = list(nx.topological_sort(self._graph))

        try:
            source_position = topological_order.index(source_node)
        except ValueError:
            # Should be impossible for a valid graph, but return gracefully.
            return PropagationResult(
                ok=False,
                source_node=source_node,
                shock_magnitude=shock_magnitude,
                node_shocks={},
                terminal_shocks={},
                error="Source node absent from topological ordering",
            )

        for current_node in topological_order[source_position:]:
            if current_node not in node_shocks:
                continue

            current_shock = node_shocks[current_node]

            for successor in self._graph.successors(current_node):
                edge = self.get_edge(current_node, successor)

                contribution = (
                    current_shock * edge.transmission_factor
                )

                node_shocks[successor] = (
                    node_shocks.get(successor, 0.0)
                    + contribution
                )

        terminal_shocks = {
            node_id: shock
            for node_id, shock in node_shocks.items()
            if self.get_node(node_id).layer == Layer.LISTED_EQUITY
        }

        return PropagationResult(
            ok=True,
            source_node=source_node,
            shock_magnitude=shock_magnitude,
            node_shocks=dict(node_shocks),
            terminal_shocks=terminal_shocks,
            error=None,
        )

    def _require_node(self, node_id: str) -> None:
        if node_id not in self._graph:
            raise GraphNodeNotFoundError(
                f"Unknown supply-chain node: {node_id!r}"
            )


# ---------------------------------------------------------------------------
# Seed graph
# ---------------------------------------------------------------------------
#
# IMPORTANT:
# The numerical edge parameters below are deterministic seed priors for
# engineering/testing. They are NOT asserted to be empirical current-period
# cost shares. Production calibrations should eventually come from PIT source
# data and versioned calibration fixtures.
# ---------------------------------------------------------------------------


SEED_NODES: Tuple[SupplyNode, ...] = (
    # -----------------------------------------------------------------------
    # Layer 0 — Raw feedstocks / minerals
    # -----------------------------------------------------------------------
    SupplyNode(
        "RAW_SILICA_SAND",
        "Silica Sand",
        Layer.RAW_FEEDSTOCK,
        NodeKind.RAW,
    ),
    SupplyNode(
        "RAW_FLUORSPAR",
        "Fluorspar",
        Layer.RAW_FEEDSTOCK,
        NodeKind.RAW,
    ),
    SupplyNode(
        "RAW_IRON_ORE",
        "Iron Ore",
        Layer.RAW_FEEDSTOCK,
        NodeKind.RAW,
    ),
    SupplyNode(
        "RAW_COKING_COAL",
        "Coking Coal",
        Layer.RAW_FEEDSTOCK,
        NodeKind.RAW,
    ),
    SupplyNode(
        "RAW_CRUDE",
        "Crude Oil",
        Layer.RAW_FEEDSTOCK,
        NodeKind.RAW,
    ),
    SupplyNode(
        "RAW_SODA_ASH",
        "Soda Ash",
        Layer.RAW_FEEDSTOCK,
        NodeKind.RAW,
    ),

    # -----------------------------------------------------------------------
    # Layer 1 — Refined intermediates
    # -----------------------------------------------------------------------
    SupplyNode(
        "INT_POLYSILICON",
        "Polysilicon",
        Layer.REFINED_INTERMEDIATE,
        NodeKind.INTERMEDIATE,
    ),
    SupplyNode(
        "INT_AHF",
        "Anhydrous Hydrogen Fluoride",
        Layer.REFINED_INTERMEDIATE,
        NodeKind.INTERMEDIATE,
    ),
    SupplyNode(
        "INT_CRGO_STEEL",
        "CRGO Steel",
        Layer.REFINED_INTERMEDIATE,
        NodeKind.INTERMEDIATE,
    ),
    SupplyNode(
        "INT_INDUSTRIAL_SULFURIC_ACID",
        "Industrial Sulfuric Acid",
        Layer.REFINED_INTERMEDIATE,
        NodeKind.INTERMEDIATE,
    ),
    SupplyNode(
        "INT_HIGH_PURITY_SILICA",
        "High-Purity Silica",
        Layer.REFINED_INTERMEDIATE,
        NodeKind.INTERMEDIATE,
    ),
    SupplyNode(
        "INT_DENSE_SODA_ASH",
        "Dense Soda Ash",
        Layer.REFINED_INTERMEDIATE,
        NodeKind.INTERMEDIATE,
    ),
    SupplyNode(
        "INT_CALCINED_CARBON",
        "Calcined Carbon Feed",
        Layer.REFINED_INTERMEDIATE,
        NodeKind.INTERMEDIATE,
    ),

    # -----------------------------------------------------------------------
    # Layer 2 — Critical components / tools
    # -----------------------------------------------------------------------
    SupplyNode(
        "COMP_TRANSFORMER_CORES",
        "Transformer Cores",
        Layer.CRITICAL_COMPONENT,
        NodeKind.COMPONENT,
    ),
    SupplyNode(
        "COMP_SOLAR_GLASS",
        "Solar Glass",
        Layer.CRITICAL_COMPONENT,
        NodeKind.COMPONENT,
    ),
    SupplyNode(
        "COMP_REFRACTORY_BRICKS",
        "Refractory Bricks",
        Layer.CRITICAL_COMPONENT,
        NodeKind.COMPONENT,
    ),
    SupplyNode(
        "COMP_PV_CELLS",
        "PV Cells",
        Layer.CRITICAL_COMPONENT,
        NodeKind.COMPONENT,
    ),
    SupplyNode(
        "COMP_FLUOROCHEMICAL_FEED",
        "Fluorochemical Feedstock",
        Layer.CRITICAL_COMPONENT,
        NodeKind.COMPONENT,
    ),
    SupplyNode(
        "COMP_CHEMICAL_PROCESS_INPUTS",
        "Chemical Process Inputs",
        Layer.CRITICAL_COMPONENT,
        NodeKind.COMPONENT,
    ),

    # -----------------------------------------------------------------------
    # Layer 3 — Sub-assemblies / operating value-chain segments
    # -----------------------------------------------------------------------
    SupplyNode(
        "SUB_POWER_TRANSFORMERS",
        "Power Transformers",
        Layer.SUBASSEMBLY,
        NodeKind.SUBASSEMBLY,
    ),
    SupplyNode(
        "SUB_PV_MODULES",
        "PV Modules",
        Layer.SUBASSEMBLY,
        NodeKind.SUBASSEMBLY,
    ),
    SupplyNode(
        "SUB_BULK_APIS",
        "Bulk APIs",
        Layer.SUBASSEMBLY,
        NodeKind.SUBASSEMBLY,
    ),
    SupplyNode(
        "SUB_SOLAR_GLASS_MANUFACTURING",
        "Solar Glass Manufacturing",
        Layer.SUBASSEMBLY,
        NodeKind.SUBASSEMBLY,
    ),
    SupplyNode(
        "SUB_FLUOROCHEMICAL_OPERATIONS",
        "Fluorochemical Operations",
        Layer.SUBASSEMBLY,
        NodeKind.SUBASSEMBLY,
    ),
    SupplyNode(
        "SUB_SPECIALTY_CHEMICAL_OPERATIONS",
        "Specialty Chemical Operations",
        Layer.SUBASSEMBLY,
        NodeKind.SUBASSEMBLY,
    ),
    SupplyNode(
        "SUB_HIGH_TEMP_PROCESS_SYSTEMS",
        "High-Temperature Process Systems",
        Layer.SUBASSEMBLY,
        NodeKind.SUBASSEMBLY,
    ),

    # -----------------------------------------------------------------------
    # Layer 4 — Listed equity exposures
    # -----------------------------------------------------------------------
    SupplyNode(
        "TRIL.NS",
        "TRIL.NS",
        Layer.LISTED_EQUITY,
        NodeKind.EQUITY,
        ticker="TRIL.NS",
    ),
    SupplyNode(
        "VOLTAMP.NS",
        "VOLTAMP.NS",
        Layer.LISTED_EQUITY,
        NodeKind.EQUITY,
        ticker="VOLTAMP.NS",
    ),
    SupplyNode(
        "BORORENEW.NS",
        "BORORENEW.NS",
        Layer.LISTED_EQUITY,
        NodeKind.EQUITY,
        ticker="BORORENEW.NS",
    ),
    SupplyNode(
        "SRF.NS",
        "SRF.NS",
        Layer.LISTED_EQUITY,
        NodeKind.EQUITY,
        ticker="SRF.NS",
    ),
    SupplyNode(
        "NAVINFLUOR.NS",
        "NAVINFLUOR.NS",
        Layer.LISTED_EQUITY,
        NodeKind.EQUITY,
        ticker="NAVINFLUOR.NS",
    ),
    SupplyNode(
        "AARTIIND.NS",
        "AARTIIND.NS",
        Layer.LISTED_EQUITY,
        NodeKind.EQUITY,
        ticker="AARTIIND.NS",
    ),
)


SEED_EDGES: Tuple[SupplyEdge, ...] = (
    # -----------------------------------------------------------------------
    # L0 -> L1
    # -----------------------------------------------------------------------
    SupplyEdge(
        "RAW_SILICA_SAND",
        "INT_HIGH_PURITY_SILICA",
        substitutability=0.20,
        cost_share=0.70,
        lead_time_months=3.0,
    ),
    SupplyEdge(
        "RAW_SILICA_SAND",
        "INT_POLYSILICON",
        substitutability=0.35,
        cost_share=0.40,
        lead_time_months=8.0,
    ),
    SupplyEdge(
        "RAW_FLUORSPAR",
        "INT_AHF",
        substitutability=0.20,
        cost_share=0.75,
        lead_time_months=6.0,
    ),
    SupplyEdge(
        "RAW_IRON_ORE",
        "INT_CRGO_STEEL",
        substitutability=0.30,
        cost_share=0.22,
        lead_time_months=4.0,
    ),
    SupplyEdge(
        "RAW_COKING_COAL",
        "INT_CRGO_STEEL",
        substitutability=0.25,
        cost_share=0.18,
        lead_time_months=5.0,
    ),
    SupplyEdge(
        "RAW_COKING_COAL",
        "INT_CALCINED_CARBON",
        substitutability=0.20,
        cost_share=0.55,
        lead_time_months=4.0,
    ),
    SupplyEdge(
        "RAW_CRUDE",
        "INT_INDUSTRIAL_SULFURIC_ACID",
        substitutability=0.65,
        cost_share=0.08,
        lead_time_months=2.0,
    ),
    SupplyEdge(
        "RAW_SODA_ASH",
        "INT_DENSE_SODA_ASH",
        substitutability=0.10,
        cost_share=0.90,
        lead_time_months=3.0,
    ),

    # -----------------------------------------------------------------------
    # L1 -> L2
    # -----------------------------------------------------------------------
    SupplyEdge(
        "INT_CRGO_STEEL",
        "COMP_TRANSFORMER_CORES",
        substitutability=0.08,
        cost_share=0.62,
        lead_time_months=7.0,
    ),
    SupplyEdge(
        "INT_HIGH_PURITY_SILICA",
        "COMP_SOLAR_GLASS",
        substitutability=0.15,
        cost_share=0.45,
        lead_time_months=5.0,
    ),
    SupplyEdge(
        "INT_DENSE_SODA_ASH",
        "COMP_SOLAR_GLASS",
        substitutability=0.25,
        cost_share=0.18,
        lead_time_months=4.0,
    ),
    SupplyEdge(
        "INT_POLYSILICON",
        "COMP_PV_CELLS",
        substitutability=0.12,
        cost_share=0.55,
        lead_time_months=6.0,
    ),

    # Fluorine-chain test fixture:
    # factor = (1 - 0.10) * 0.80 = 0.72
    SupplyEdge(
        "INT_AHF",
        "COMP_FLUOROCHEMICAL_FEED",
        substitutability=0.10,
        cost_share=0.80,
        lead_time_months=5.0,
    ),

    SupplyEdge(
        "INT_INDUSTRIAL_SULFURIC_ACID",
        "COMP_CHEMICAL_PROCESS_INPUTS",
        substitutability=0.30,
        cost_share=0.25,
        lead_time_months=2.0,
    ),
    SupplyEdge(
        "INT_CALCINED_CARBON",
        "COMP_REFRACTORY_BRICKS",
        substitutability=0.25,
        cost_share=0.35,
        lead_time_months=5.0,
    ),

    # -----------------------------------------------------------------------
    # L2 -> L3
    # -----------------------------------------------------------------------
    SupplyEdge(
        "COMP_TRANSFORMER_CORES",
        "SUB_POWER_TRANSFORMERS",
        substitutability=0.06,
        cost_share=0.34,
        lead_time_months=9.0,
    ),
    SupplyEdge(
        "COMP_SOLAR_GLASS",
        "SUB_PV_MODULES",
        substitutability=0.25,
        cost_share=0.10,
        lead_time_months=3.0,
    ),
    SupplyEdge(
        "COMP_PV_CELLS",
        "SUB_PV_MODULES",
        substitutability=0.15,
        cost_share=0.55,
        lead_time_months=5.0,
    ),
    SupplyEdge(
        "COMP_SOLAR_GLASS",
        "SUB_SOLAR_GLASS_MANUFACTURING",
        substitutability=0.05,
        cost_share=0.82,
        lead_time_months=4.0,
    ),

    # Fluorine-chain test fixture:
    # factor = (1 - 0.25) * 0.60 = 0.45
    SupplyEdge(
        "COMP_FLUOROCHEMICAL_FEED",
        "SUB_FLUOROCHEMICAL_OPERATIONS",
        substitutability=0.25,
        cost_share=0.60,
        lead_time_months=8.0,
    ),

    SupplyEdge(
        "COMP_CHEMICAL_PROCESS_INPUTS",
        "SUB_BULK_APIS",
        substitutability=0.40,
        cost_share=0.28,
        lead_time_months=3.0,
    ),
    SupplyEdge(
        "COMP_CHEMICAL_PROCESS_INPUTS",
        "SUB_SPECIALTY_CHEMICAL_OPERATIONS",
        substitutability=0.35,
        cost_share=0.30,
        lead_time_months=4.0,
    ),
    SupplyEdge(
        "COMP_REFRACTORY_BRICKS",
        "SUB_HIGH_TEMP_PROCESS_SYSTEMS",
        substitutability=0.30,
        cost_share=0.12,
        lead_time_months=5.0,
    ),

    # -----------------------------------------------------------------------
    # L3 -> L4
    # -----------------------------------------------------------------------
    SupplyEdge(
        "SUB_POWER_TRANSFORMERS",
        "TRIL.NS",
        substitutability=0.10,
        cost_share=0.72,
        lead_time_months=9.0,
    ),
    SupplyEdge(
        "SUB_POWER_TRANSFORMERS",
        "VOLTAMP.NS",
        substitutability=0.12,
        cost_share=0.68,
        lead_time_months=8.0,
    ),
    SupplyEdge(
        "SUB_SOLAR_GLASS_MANUFACTURING",
        "BORORENEW.NS",
        substitutability=0.05,
        cost_share=0.85,
        lead_time_months=6.0,
    ),

    # Fluorine-chain test fixture:
    # SRF factor   = (1 - 0.10) * 0.50 = 0.45
    # Navin factor = (1 - 0.05) * 0.55 = 0.5225
    SupplyEdge(
        "SUB_FLUOROCHEMICAL_OPERATIONS",
        "SRF.NS",
        substitutability=0.10,
        cost_share=0.50,
        lead_time_months=6.0,
    ),
    SupplyEdge(
        "SUB_FLUOROCHEMICAL_OPERATIONS",
        "NAVINFLUOR.NS",
        substitutability=0.05,
        cost_share=0.55,
        lead_time_months=7.0,
    ),

    SupplyEdge(
        "SUB_SPECIALTY_CHEMICAL_OPERATIONS",
        "AARTIIND.NS",
        substitutability=0.15,
        cost_share=0.62,
        lead_time_months=5.0,
    ),
)


def build_seed_graph() -> SupplyChainGraph:
    """
    Construct and validate the deterministic Level-1 seed graph.
    """
    graph = SupplyChainGraph(
        nodes=SEED_NODES,
        edges=SEED_EDGES,
    )

    report = graph.validate()

    if not report.is_valid:
        raise GraphValidationError(
            "Seed graph failed validation: "
            + "; ".join(report.errors)
        )

    return graph


if __name__ == "__main__":
    graph = build_seed_graph()

    result = graph.propagate_shock(
        source_node="RAW_FLUORSPAR",
        shock_magnitude=1.0,
    )

    print(f"nodes={graph.node_count}")
    print(f"edges={graph.edge_count}")
    print(f"valid={graph.validate().is_valid}")
    print(f"terminal_shocks={result.terminal_shocks}")
