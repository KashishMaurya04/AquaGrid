"""
distribution/engine.py – Network-Aware Water Distribution Optimization Engine.

Bridges the gap between mathematical fair water allocation (volume entitlement)
and physical water network conduits (hydraulic topology and pipe capacity constraints).

Core Responsibilities:
----------------------
1. Evaluates topological reachability and finds transport routes from a designated
   water supply source node (e.g. Tank / Reservoir) to consumer distribution zones.
2. Supports two distinct routing objectives:
   - SHORTEST_PATH: Minimum cumulative routing cost (via Dijkstra's algorithm).
   - WIDEST_PATH: Maximum bottleneck delivery capacity (via Maximum Bottleneck Path algorithm).
3. Identifies the conduit bottleneck along the selected path:
       C_path = min(C_1, C_2, ..., C_k)
4. Computes delivery rate requirements and theoretical volumetric delivery:
       Q_required = V_allocated / T_duration
       V_max = C_path * T_duration
       V_deliverable = min(V_allocated, V_max)
       V_undelivered = V_allocated - V_deliverable
5. Classifies delivery feasibility (FEASIBLE, CAPACITY_CONSTRAINED, UNREACHABLE, UNMAPPED, ZERO_ALLOCATION).
6. Detects shared-pipe contention across multiple consumer zones operating simultaneously.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Sequence

from dsa.dijkstra import dijkstra
from dsa.graph import Graph
from dsa.widest_path import widest_path


# Feasibility Status Constants
STATUS_FEASIBLE = "FEASIBLE"
STATUS_CAPACITY_CONSTRAINED = "CAPACITY_CONSTRAINED"
STATUS_UNREACHABLE = "UNREACHABLE"
STATUS_UNMAPPED = "UNMAPPED"
STATUS_ZERO_ALLOCATION = "ZERO_ALLOCATION"

FEASIBILITY_STATUSES = {
    STATUS_FEASIBLE,
    STATUS_CAPACITY_CONSTRAINED,
    STATUS_UNREACHABLE,
    STATUS_UNMAPPED,
    STATUS_ZERO_ALLOCATION,
}

# Routing Strategy Constants
STRATEGY_SHORTEST_PATH = "SHORTEST_PATH"
STRATEGY_WIDEST_PATH = "WIDEST_PATH"

SUPPORTED_ROUTING_STRATEGIES = {
    STRATEGY_SHORTEST_PATH,
    STRATEGY_WIDEST_PATH,
}


@dataclass
class ZoneDeliveryInput:
    """Input parameters for evaluating delivery feasibility for a consumer zone."""
    zone_id: int
    zone_name: str
    destination_node_id: int | None
    allocated_volume_l: float | int | Decimal

    def __post_init__(self) -> None:
        try:
            val = float(self.allocated_volume_l)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid allocated volume for zone '{self.zone_name}' (#{self.zone_id}): {exc}"
            ) from exc

        if math.isnan(val) or math.isinf(val):
            raise ValueError(
                f"Allocated volume for zone '{self.zone_name}' (#{self.zone_id}) must be a finite number."
            )
        if val < 0:
            raise ValueError(
                f"Allocated volume for zone '{self.zone_name}' (#{self.zone_id}) cannot be negative (got {val})."
            )
        self.allocated_volume_l = val


@dataclass
class ZoneDeliveryResult:
    """Detailed delivery feasibility outcome for an individual consumer zone."""
    zone_id: int
    zone_name: str
    destination_node_id: int | None
    allocated_volume_l: float = 0.0
    route_node_ids: list[int] = field(default_factory=list)
    route_pipe_ids: list[int] = field(default_factory=list)
    route_cost: float | None = None
    bottleneck_pipe_id: int | None = None
    bottleneck_pipe_name: str | None = None
    bottleneck_capacity_lpm: float | None = None
    required_rate_lpm: float = 0.0
    maximum_deliverable_l: float = 0.0
    deliverable_volume_l: float = 0.0
    undelivered_volume_l: float = 0.0
    delivery_efficiency_pct: float = 0.0
    feasibility_status: str = STATUS_UNMAPPED
    explanation: str = ""

    def __getitem__(self, item: str) -> Any:
        if hasattr(self, item):
            return getattr(self, item)
        raise KeyError(item)

    def get(self, item: str, default: Any = None) -> Any:
        return getattr(self, item, default)

    def to_dict(self) -> dict[str, Any]:
        return {
            "zone_id": self.zone_id,
            "zone_name": self.zone_name,
            "destination_node_id": self.destination_node_id,
            "allocated_volume_l": self.allocated_volume_l,
            "route_node_ids": list(self.route_node_ids),
            "route_pipe_ids": list(self.route_pipe_ids),
            "route_cost": self.route_cost,
            "bottleneck_pipe_id": self.bottleneck_pipe_id,
            "bottleneck_pipe_name": self.bottleneck_pipe_name,
            "bottleneck_capacity_lpm": self.bottleneck_capacity_lpm,
            "required_rate_lpm": self.required_rate_lpm,
            "maximum_deliverable_l": self.maximum_deliverable_l,
            "deliverable_volume_l": self.deliverable_volume_l,
            "undelivered_volume_l": self.undelivered_volume_l,
            "delivery_efficiency_pct": self.delivery_efficiency_pct,
            "feasibility_status": self.feasibility_status,
            "explanation": self.explanation,
        }


@dataclass
class DistributionPlanResult:
    """Holistic distribution optimization plan across all consumer zones."""
    source_node_id: int
    duration_minutes: float
    routing_strategy: str
    total_allocated_l: float
    total_deliverable_l: float
    total_undelivered_l: float
    overall_delivery_efficiency_pct: float
    zone_results: list[ZoneDeliveryResult]
    shared_pipe_warnings: list[dict[str, Any]] = field(default_factory=list)
    source_node_name: str = ""
    has_unreachable_zones: bool = False
    has_constrained_zones: bool = False
    has_unmapped_zones: bool = False
    notes: str = ""

    def __getitem__(self, item: str) -> Any:
        if hasattr(self, item):
            return getattr(self, item)
        raise KeyError(item)

    def get(self, item: str, default: Any = None) -> Any:
        return getattr(self, item, default)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_node_id": self.source_node_id,
            "source_node_name": self.source_node_name,
            "duration_minutes": self.duration_minutes,
            "routing_strategy": self.routing_strategy,
            "total_allocated_l": self.total_allocated_l,
            "total_deliverable_l": self.total_deliverable_l,
            "total_undelivered_l": self.total_undelivered_l,
            "overall_delivery_efficiency_pct": self.overall_delivery_efficiency_pct,
            "zone_results": [z.to_dict() for z in self.zone_results],
            "shared_pipe_warnings": list(self.shared_pipe_warnings),
            "has_unreachable_zones": self.has_unreachable_zones,
            "has_constrained_zones": self.has_constrained_zones,
            "has_unmapped_zones": self.has_unmapped_zones,
            "notes": self.notes,
        }


def _normalize_strategy(raw_strategy: str) -> str:
    """Normalize and validate routing strategy string."""
    clean = str(raw_strategy).strip().upper()
    if clean in ("SHORTEST_PATH", "SHORTEST", "DIJKSTRA", "MIN_COST"):
        return STRATEGY_SHORTEST_PATH
    if clean in ("WIDEST_PATH", "WIDEST", "MAX_CAPACITY", "MAX_BOTTLENECK", "BOTTLENECK"):
        return STRATEGY_WIDEST_PATH
    raise ValueError(
        f"Unknown routing strategy '{raw_strategy}'. "
        f"Supported strategies: {', '.join(sorted(SUPPORTED_ROUTING_STRATEGIES))}."
    )


def _coerce_zone_input(item: ZoneDeliveryInput | dict[str, Any]) -> ZoneDeliveryInput:
    """Coerce input item into a validated ZoneDeliveryInput instance."""
    if isinstance(item, ZoneDeliveryInput):
        return item
    if isinstance(item, dict):
        zid = int(item.get("zone_id", 0))
        zname = str(item.get("zone_name", f"Zone-{zid}"))
        dest_raw = item.get("destination_node_id", item.get("node_id"))
        dest_id = int(dest_raw) if dest_raw is not None else None
        vol = item.get("allocated_volume_l", item.get("allocated_amount_l", 0.0))
        return ZoneDeliveryInput(
            zone_id=zid,
            zone_name=zname,
            destination_node_id=dest_id,
            allocated_volume_l=vol,
        )
    raise TypeError(f"Expected ZoneDeliveryInput or dict, got {type(item).__name__}")


def assess_distribution(
    graph: Graph,
    source_node_id: int,
    zones: Sequence[ZoneDeliveryInput | dict[str, Any]],
    duration_minutes: float | int = 60.0,
    routing_strategy: str = STRATEGY_SHORTEST_PATH,
) -> DistributionPlanResult:
    """
    Assess network delivery feasibility for a set of consumer zones from a supply source.

    Args:
        graph: Adjacency-list Graph instance representing the water network.
        source_node_id: ID of the water supply source node (e.g. Tank/Reservoir).
        zones: Collection of ZoneDeliveryInput instances or dictionaries.
        duration_minutes: Available delivery period in minutes (must be strictly positive).
        routing_strategy: 'SHORTEST_PATH' or 'WIDEST_PATH'.

    Returns:
        DistributionPlanResult containing holistic metrics, per-zone outcomes, and
        shared-pipe contention diagnostics.
    """
    # ── 1. Validate Duration ──────────────────────────────────────────────────
    try:
        dur = float(duration_minutes)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid delivery duration: {exc}") from exc

    if math.isnan(dur) or math.isinf(dur):
        raise ValueError("Delivery duration must be a finite number.")
    if dur <= 0:
        raise ValueError(
            f"Delivery duration must be strictly positive (got {duration_minutes} minutes)."
        )

    # ── 2. Validate Graph and Source Node ─────────────────────────────────────
    if graph.node_count == 0:
        raise ValueError("Network graph is empty. No nodes exist in the network.")

    if not graph.has_node(source_node_id):
        raise ValueError(
            f"Source node #{source_node_id} does not exist in the network graph."
        )

    source_meta = graph.nodes.get(source_node_id, {})
    if source_meta.get("status") != "active":
        raise ValueError(
            f"Source node #{source_node_id} ('{source_meta.get('name', '')}') is inactive."
        )
    source_name = source_meta.get("name", f"Node-{source_node_id}")

    # ── 3. Validate Strategy ──────────────────────────────────────────────────
    strategy = _normalize_strategy(routing_strategy)

    # Coerce input zones
    coerced_zones = [_coerce_zone_input(z) for z in zones]

    zone_results: list[ZoneDeliveryResult] = []
    # Map for shared-pipe contention analysis: pipe_id -> list of (zone_id, zone_name, flow_rate_lpm)
    pipe_usage: dict[int, list[tuple[int, str, float]]] = {}
    pipe_metadata: dict[int, dict[str, Any]] = {}

    # ── 4. Evaluate Each Consumer Zone ────────────────────────────────────────
    for z in coerced_zones:
        alloc_vol = float(z.allocated_volume_l)

        # Case A: Zero Allocated Volume
        if math.isclose(alloc_vol, 0.0, abs_tol=1e-9):
            zone_results.append(
                ZoneDeliveryResult(
                    zone_id=z.zone_id,
                    zone_name=z.zone_name,
                    destination_node_id=z.destination_node_id,
                    allocated_volume_l=0.0,
                    route_node_ids=[],
                    route_pipe_ids=[],
                    route_cost=0.0,
                    bottleneck_pipe_id=None,
                    bottleneck_pipe_name=None,
                    bottleneck_capacity_lpm=None,
                    required_rate_lpm=0.0,
                    maximum_deliverable_l=0.0,
                    deliverable_volume_l=0.0,
                    undelivered_volume_l=0.0,
                    delivery_efficiency_pct=100.0,
                    feasibility_status=STATUS_ZERO_ALLOCATION,
                    explanation=(
                        f"Zone '{z.zone_name}' has 0.0 L allocated. "
                        f"No physical water delivery is required."
                    ),
                )
            )
            continue

        required_rate = round(alloc_vol / dur, 4)

        # Case B: Unmapped Zone (No destination node)
        if z.destination_node_id is None:
            zone_results.append(
                ZoneDeliveryResult(
                    zone_id=z.zone_id,
                    zone_name=z.zone_name,
                    destination_node_id=None,
                    allocated_volume_l=round(alloc_vol, 2),
                    route_node_ids=[],
                    route_pipe_ids=[],
                    route_cost=None,
                    bottleneck_pipe_id=None,
                    bottleneck_pipe_name=None,
                    bottleneck_capacity_lpm=None,
                    required_rate_lpm=required_rate,
                    maximum_deliverable_l=0.0,
                    deliverable_volume_l=0.0,
                    undelivered_volume_l=round(alloc_vol, 2),
                    delivery_efficiency_pct=0.0,
                    feasibility_status=STATUS_UNMAPPED,
                    explanation=(
                        f"Zone '{z.zone_name}' is not mapped to any physical network node. "
                        f"Delivery cannot be simulated or routed until a destination node is assigned."
                    ),
                )
            )
            continue

        dest_id = int(z.destination_node_id)

        # Case C: Co-Located Zone (Source == Destination)
        if dest_id == source_node_id:
            zone_results.append(
                ZoneDeliveryResult(
                    zone_id=z.zone_id,
                    zone_name=z.zone_name,
                    destination_node_id=dest_id,
                    allocated_volume_l=round(alloc_vol, 2),
                    route_node_ids=[source_node_id],
                    route_pipe_ids=[],
                    route_cost=0.0,
                    bottleneck_pipe_id=None,
                    bottleneck_pipe_name=None,
                    bottleneck_capacity_lpm=None,
                    required_rate_lpm=required_rate,
                    maximum_deliverable_l=round(alloc_vol, 2),
                    deliverable_volume_l=round(alloc_vol, 2),
                    undelivered_volume_l=0.0,
                    delivery_efficiency_pct=100.0,
                    feasibility_status=STATUS_FEASIBLE,
                    explanation=(
                        f"Zone '{z.zone_name}' is co-located at supply source #{source_node_id}. "
                        f"Water is available directly at the reservoir without conduit transport constraints."
                    ),
                )
            )
            continue

        # Case D: Network Pathfinding
        if strategy == STRATEGY_SHORTEST_PATH:
            route_res = dijkstra(graph, source_node_id, dest_id)
        else:
            route_res = widest_path(graph, source_node_id, dest_id)

        # Case E: Destination Unreachable
        if not route_res.found:
            zone_results.append(
                ZoneDeliveryResult(
                    zone_id=z.zone_id,
                    zone_name=z.zone_name,
                    destination_node_id=dest_id,
                    allocated_volume_l=round(alloc_vol, 2),
                    route_node_ids=[],
                    route_pipe_ids=[],
                    route_cost=None,
                    bottleneck_pipe_id=None,
                    bottleneck_pipe_name=None,
                    bottleneck_capacity_lpm=None,
                    required_rate_lpm=required_rate,
                    maximum_deliverable_l=0.0,
                    deliverable_volume_l=0.0,
                    undelivered_volume_l=round(alloc_vol, 2),
                    delivery_efficiency_pct=0.0,
                    feasibility_status=STATUS_UNREACHABLE,
                    explanation=(
                        f"Destination node #{dest_id} is unreachable from source #{source_node_id}. "
                        f"No active physical pipe path connects the supply node to this zone."
                    ),
                )
            )
            continue

        # Case F: Path Found – Analyze Conduit Bottleneck
        pipe_path = route_res.pipe_path
        if not pipe_path:
            # Degenerate case of path with no pipes
            zone_results.append(
                ZoneDeliveryResult(
                    zone_id=z.zone_id,
                    zone_name=z.zone_name,
                    destination_node_id=dest_id,
                    allocated_volume_l=round(alloc_vol, 2),
                    route_node_ids=route_res.node_path,
                    route_pipe_ids=[],
                    route_cost=0.0,
                    bottleneck_pipe_id=None,
                    bottleneck_pipe_name=None,
                    bottleneck_capacity_lpm=None,
                    required_rate_lpm=required_rate,
                    maximum_deliverable_l=round(alloc_vol, 2),
                    deliverable_volume_l=round(alloc_vol, 2),
                    undelivered_volume_l=0.0,
                    delivery_efficiency_pct=100.0,
                    feasibility_status=STATUS_FEASIBLE,
                    explanation=f"Direct zero-pipe connection to destination node #{dest_id}.",
                )
            )
            continue

        # Find the bottleneck pipe (minimum capacity_lpm along the path)
        bottleneck_edge = min(pipe_path, key=lambda p: float(p.get("capacity_lpm", 0.0)))
        bottleneck_cap = float(bottleneck_edge["capacity_lpm"])
        bottleneck_pid = int(bottleneck_edge["pipe_id"])
        bottleneck_pname = str(bottleneck_edge.get("name", f"Pipe-{bottleneck_pid}"))

        # Volumetric calculations
        v_max = round(bottleneck_cap * dur, 2)
        v_deliv = round(min(alloc_vol, v_max), 2)
        v_undeliv = round(alloc_vol - v_deliv, 2)
        eff_pct = round((v_deliv / alloc_vol * 100.0) if alloc_vol > 0 else 100.0, 2)

        if v_deliv >= alloc_vol - 1e-9:
            status = STATUS_FEASIBLE
            explanation = (
                f"Feasible: Selected {strategy.lower().replace('_', ' ')} of {len(pipe_path)} pipe(s) "
                f"provides adequate capacity (bottleneck {bottleneck_cap:.1f} L/min >= required {required_rate:.1f} L/min). "
                f"100% of allocated volume ({alloc_vol:.1f} L) can be delivered in {dur:.1f} minutes "
                f"at total routing cost {route_res.total_cost:.2f}."
            )
        else:
            status = STATUS_CAPACITY_CONSTRAINED
            explanation = (
                f"Capacity Constrained: Delivery is restricted by bottleneck pipe '{bottleneck_pname}' "
                f"(#{bottleneck_pid}) with conduit capacity {bottleneck_cap:.1f} L/min. "
                f"Required delivery rate is {required_rate:.1f} L/min. Over {dur:.1f} minutes, "
                f"only {v_deliv:.1f} L of the {alloc_vol:.1f} L allocation can be delivered ({eff_pct:.1f}%), "
                f"leaving {v_undeliv:.1f} L undelivered."
            )

        # Record pipe usage for shared-pipe contention analysis
        for edge in pipe_path:
            pid = int(edge["pipe_id"])
            if pid not in pipe_usage:
                pipe_usage[pid] = []
                pipe_metadata[pid] = {
                    "pipe_id": pid,
                    "pipe_name": str(edge.get("name", f"Pipe-{pid}")),
                    "capacity_lpm": float(edge.get("capacity_lpm", 0.0)),
                }
            pipe_usage[pid].append((z.zone_id, z.zone_name, required_rate))

        zone_results.append(
            ZoneDeliveryResult(
                zone_id=z.zone_id,
                zone_name=z.zone_name,
                destination_node_id=dest_id,
                allocated_volume_l=round(alloc_vol, 2),
                route_node_ids=route_res.node_path,
                route_pipe_ids=route_res.pipe_ids,
                route_cost=round(float(route_res.total_cost), 4) if route_res.total_cost is not None else None,
                bottleneck_pipe_id=bottleneck_pid,
                bottleneck_pipe_name=bottleneck_pname,
                bottleneck_capacity_lpm=round(bottleneck_cap, 4),
                required_rate_lpm=required_rate,
                maximum_deliverable_l=v_max,
                deliverable_volume_l=v_deliv,
                undelivered_volume_l=v_undeliv,
                delivery_efficiency_pct=eff_pct,
                feasibility_status=status,
                explanation=explanation,
            )
        )

    # ── 5. Shared-Pipe Contention Diagnostic ──────────────────────────────────
    shared_pipe_warnings: list[dict[str, Any]] = []
    for pid in sorted(pipe_usage.keys()):
        users = pipe_usage[pid]
        meta = pipe_metadata[pid]
        cap = meta["capacity_lpm"]
        total_flow = round(sum(rate for _, _, rate in users), 4)

        if total_flow > cap + 1e-9:
            excess = round(total_flow - cap, 4)
            affected_zones = [
                {"zone_id": zid, "zone_name": zname, "required_rate_lpm": round(rate, 2)}
                for zid, zname, rate in users
            ]
            warning_msg = (
                f"Contention detected on '{meta['pipe_name']}' (#{pid}): aggregate required flow "
                f"of {total_flow:.2f} L/min across {len(affected_zones)} zone(s) exceeds conduit capacity "
                f"of {cap:.2f} L/min by {excess:.2f} L/min. Operating these routes simultaneously "
                f"will cause mutual delivery deficits unless scheduled sequentially."
            )
            shared_pipe_warnings.append({
                "pipe_id": pid,
                "pipe_name": meta["pipe_name"],
                "pipe_capacity_lpm": round(cap, 2),
                "aggregate_required_lpm": round(total_flow, 2),
                "excess_demand_lpm": round(excess, 2),
                "affected_zone_ids": [z["zone_id"] for z in affected_zones],
                "affected_zone_names": [z["zone_name"] for z in affected_zones],
                "affected_zones": affected_zones,
                "warning_message": warning_msg,
            })

    # ── 6. Aggregate Holistic Plan Metrics ────────────────────────────────────
    tot_alloc = round(sum(z.allocated_volume_l for z in coerced_zones), 2)
    tot_deliv = round(sum(zr.deliverable_volume_l for zr in zone_results), 2)
    tot_undeliv = round(tot_alloc - tot_deliv, 2)
    overall_eff = round((tot_deliv / tot_alloc * 100.0) if tot_alloc > 0 else 100.0, 2)

    has_unreachable = any(zr.feasibility_status == STATUS_UNREACHABLE for zr in zone_results)
    has_constrained = any(zr.feasibility_status == STATUS_CAPACITY_CONSTRAINED for zr in zone_results)
    has_unmapped = any(zr.feasibility_status == STATUS_UNMAPPED for zr in zone_results)

    notes_parts: list[str] = []
    if shared_pipe_warnings:
        notes_parts.append(
            f"Notice: {len(shared_pipe_warnings)} shared pipe(s) exceed individual conduit capacity "
            f"under concurrent delivery. Individual zone feasibility estimates assume sequential or non-conflicting flow."
        )
    if has_unreachable:
        notes_parts.append("One or more zones are topologically unreachable from the selected supply source.")
    if has_unmapped:
        notes_parts.append("One or more zones lack a mapped network destination node.")

    plan_notes = " ".join(notes_parts) if notes_parts else "All routed zones assessed successfully."

    return DistributionPlanResult(
        source_node_id=source_node_id,
        source_node_name=source_name,
        duration_minutes=dur,
        routing_strategy=strategy,
        total_allocated_l=tot_alloc,
        total_deliverable_l=tot_deliv,
        total_undelivered_l=tot_undeliv,
        overall_delivery_efficiency_pct=overall_eff,
        zone_results=zone_results,
        shared_pipe_warnings=shared_pipe_warnings,
        has_unreachable_zones=has_unreachable,
        has_constrained_zones=has_constrained,
        has_unmapped_zones=has_unmapped,
        notes=plan_notes,
    )
