"""
dsa/widest_path.py – Maximum Bottleneck Path (Widest Path) Algorithm for AquaGrid.

Computes the path between water network nodes that maximizes the minimum conduit
capacity (bottleneck capacity) using a priority queue (Python's `heapq`).

Mathematical Objective:
-----------------------
For a path P = (e_1, e_2, ..., e_k):
    WidestCapacity(P) = min(capacity_lpm(e_i) for i in 1..k)
The algorithm finds P* in P(source, destination) such that:
    WidestCapacity(P*) = max(WidestCapacity(P) for P in P(source, destination))

Secondary Objective (Deterministic Tie-Breaking):
-------------------------------------------------
Among paths with identical maximum bottleneck capacity, the algorithm selects
the path with minimal cumulative routing cost, followed by fewer pipe hops,
followed by deterministic node ordering.

Time and Space Complexity:
--------------------------
- Time Complexity: O((V + E) log V)
  Using a binary max-heap (implemented via min-heap with negated bottleneck values),
  each vertex is relaxed with logarithmic priority queue operations.
- Space Complexity: O(V + E)
  Storage for the priority queue, bottleneck maps, predecessor references, and path dictionaries.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from dsa.graph import Graph


@dataclass
class WidestPathResult:
    """
    Encapsulates the result of a Maximum Bottleneck Path query.

    Supports both attribute access (result.bottleneck_capacity) and dict-style
    indexing (result["bottleneck_capacity"]) for seamless template and engine integration.
    """
    source: int | None
    destination: int | None
    found: bool
    bottleneck_capacity: float | None
    total_cost: float | None
    node_path: list[int] = field(default_factory=list)
    pipe_ids: list[int] = field(default_factory=list)
    pipe_names: list[str] = field(default_factory=list)
    pipe_path: list[dict[str, Any]] = field(default_factory=list)
    message: str = ""

    def __getitem__(self, item: str) -> Any:
        if hasattr(self, item):
            return getattr(self, item)
        raise KeyError(item)

    def get(self, item: str, default: Any = None) -> Any:
        return getattr(self, item, default)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "destination": self.destination,
            "found": self.found,
            "bottleneck_capacity": self.bottleneck_capacity,
            "total_cost": self.total_cost,
            "node_path": list(self.node_path),
            "pipe_ids": list(self.pipe_ids),
            "pipe_names": list(self.pipe_names),
            "pipe_path": list(self.pipe_path),
            "message": self.message,
        }


def widest_path(graph: Graph, source: int, destination: int) -> WidestPathResult:
    """
    Find the Maximum Bottleneck Path (Widest Path) from source to destination.

    Args:
        graph: Adjacency-list Graph instance (from dsa.graph).
        source: Start node ID.
        destination: Target node ID.

    Returns:
        WidestPathResult containing bottleneck_capacity, total_cost, node_path,
        pipe_ids, pipe_names, and pipe_path.

    Raises:
        ValueError: If any active pipe capacity is non-positive or non-finite.
    """
    # ── 1. Validate empty graph and existence of source / destination ──────────
    if graph.node_count == 0:
        return WidestPathResult(
            source=source,
            destination=destination,
            found=False,
            bottleneck_capacity=None,
            total_cost=None,
            message="Network is empty. No nodes exist in the graph.",
        )

    if not graph.has_node(source):
        return WidestPathResult(
            source=source,
            destination=destination,
            found=False,
            bottleneck_capacity=None,
            total_cost=None,
            message=f"Source node #{source} does not exist in the network.",
        )

    if not graph.has_node(destination):
        return WidestPathResult(
            source=source,
            destination=destination,
            found=False,
            bottleneck_capacity=None,
            total_cost=None,
            message=f"Destination node #{destination} does not exist in the network.",
        )

    # ── 2. Validate endpoint active status ────────────────────────────────────
    source_meta = graph.nodes.get(source, {})
    dest_meta = graph.nodes.get(destination, {})

    if source_meta.get("status") != "active":
        return WidestPathResult(
            source=source,
            destination=destination,
            found=False,
            bottleneck_capacity=None,
            total_cost=None,
            message=f"Source node #{source} ('{source_meta.get('name', '')}') is inactive.",
        )

    if dest_meta.get("status") != "active":
        return WidestPathResult(
            source=source,
            destination=destination,
            found=False,
            bottleneck_capacity=None,
            total_cost=None,
            message=f"Destination node #{destination} ('{dest_meta.get('name', '')}') is inactive.",
        )

    # ── 3. Validate identical source and destination endpoints ────────────────
    if source == destination:
        return WidestPathResult(
            source=source,
            destination=destination,
            found=False,
            bottleneck_capacity=None,
            total_cost=None,
            node_path=[],
            pipe_ids=[],
            pipe_names=[],
            pipe_path=[],
            message="Source and destination must be different.",
        )

    # ── 4. Initialize Widest-Path data structures ─────────────────────────────
    # Best known bottleneck capacity from source to each vertex
    best_bottleneck: dict[int, float] = {source: float("inf")}

    # Best known cumulative routing cost corresponding to the best bottleneck
    best_cost: dict[int, float] = {source: 0.0}

    # Best known hop count corresponding to the best bottleneck
    best_hops: dict[int, int] = {source: 0}

    # Predecessor map: node_id -> (predecessor_node_id, edge_dict)
    predecessors: dict[int, tuple[int, dict[str, Any]]] = {}

    # Set of settled vertices whose optimal widest path is finalized
    visited: set[int] = set()

    # Max-Priority Queue implemented using Python heapq with negated capacity.
    # Heap tuple: (-bottleneck_capacity, cumulative_cost, hop_count, sequence_id, node_id)
    # 1. Smaller negative bottleneck = larger actual bottleneck capacity (primary max-bottleneck).
    # 2. Smaller cumulative_cost = lower total routing cost (secondary min-cost tie-break).
    # 3. Smaller hop_count = fewer conduit segments (tertiary min-hop tie-break).
    # 4. sequence_id = unique monotonic integer guaranteeing deterministic tie-breaking.
    seq: int = 0
    heap: list[tuple[float, float, int, int, int]] = []
    heapq.heappush(heap, (-float("inf"), 0.0, 0, seq, source))

    # ── 5. Priority Queue Relaxation Loop ─────────────────────────────────────
    while heap:
        neg_b, curr_cost, curr_hops, _, u = heapq.heappop(heap)
        curr_b = -neg_b

        if u in visited:
            continue
        visited.add(u)

        # Early termination: destination reached with maximal bottleneck capacity
        if u == destination:
            break

        # Stale heap entry check
        if curr_b < best_bottleneck.get(u, 0.0):
            continue

        # Explore all active incident/outgoing edges (supporting parallel pipes)
        for edge in graph.get_outgoing_edges(u):
            if edge.get("status") != "active":
                continue

            v = edge.get("neighbor_id")
            if v is None:
                continue

            v_meta = graph.nodes.get(v, {})
            if v_meta.get("status") != "active":
                continue

            # Validate pipe capacity
            try:
                cap = float(edge.get("capacity_lpm", 0.0))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid pipe capacity on pipe #{edge.get('pipe_id')}: {exc}"
                ) from exc

            if cap <= 0 or math.isnan(cap) or math.isinf(cap):
                raise ValueError(
                    f"Pipe capacity must be strictly positive and finite (got {cap} "
                    f"on pipe '{edge.get('name') or edge.get('pipe_id')}')."
                )

            # Validate pipe cost
            try:
                edge_cost = float(edge.get("cost", 0.0))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid routing cost on pipe #{edge.get('pipe_id')}: {exc}"
                ) from exc

            if edge_cost < 0 or math.isnan(edge_cost) or math.isinf(edge_cost):
                raise ValueError(
                    f"Pipe cost cannot be negative or non-finite (got {edge_cost} "
                    f"on pipe '{edge.get('name') or edge.get('pipe_id')}')."
                )

            # Candidate bottleneck to neighbor v is the minimum of path bottleneck so far and edge capacity
            cand_b = min(curr_b, cap)
            cand_cost = curr_cost + edge_cost
            cand_hops = curr_hops + 1

            prev_b = best_bottleneck.get(v, 0.0)
            prev_cost = best_cost.get(v, float("inf"))
            prev_hops = best_hops.get(v, 1000000000)

            should_update = False
            # Condition 1: Strictly larger bottleneck capacity
            if cand_b > prev_b:
                should_update = True
            # Condition 2: Equal bottleneck capacity, but lower routing cost
            elif math.isclose(cand_b, prev_b, rel_tol=1e-9) and cand_cost < prev_cost:
                should_update = True
            # Condition 3: Equal capacity and equal cost, but fewer hops
            elif (
                math.isclose(cand_b, prev_b, rel_tol=1e-9)
                and math.isclose(cand_cost, prev_cost, rel_tol=1e-9)
                and cand_hops < prev_hops
            ):
                should_update = True

            if should_update:
                best_bottleneck[v] = cand_b
                best_cost[v] = cand_cost
                best_hops[v] = cand_hops
                predecessors[v] = (u, edge)
                seq += 1
                heapq.heappush(heap, (-cand_b, cand_cost, cand_hops, seq, v))

    # ── 6. Check if destination was reached ───────────────────────────────────
    if destination not in predecessors:
        return WidestPathResult(
            source=source,
            destination=destination,
            found=False,
            bottleneck_capacity=None,
            total_cost=None,
            message=f"Destination node #{destination} is not reachable from source node #{source}.",
        )

    # ── 7. Reconstruct path from destination back to source ────────────────────
    curr = destination
    reversed_nodes: list[int] = [curr]
    reversed_pipes: list[dict[str, Any]] = []

    while curr != source:
        prev_node, edge_used = predecessors[curr]
        reversed_pipes.append(edge_used)
        reversed_nodes.append(prev_node)
        curr = prev_node

    ordered_node_path = list(reversed(reversed_nodes))
    ordered_pipe_path = list(reversed(reversed_pipes))
    ordered_pipe_ids = [
        int(p["pipe_id"]) for p in ordered_pipe_path if p.get("pipe_id") is not None
    ]
    ordered_pipe_names = [str(p.get("name", "")) for p in ordered_pipe_path]

    final_bottleneck = min(float(p["capacity_lpm"]) for p in ordered_pipe_path)
    final_cost = sum(float(p.get("cost", 0.0)) for p in ordered_pipe_path)

    return WidestPathResult(
        source=source,
        destination=destination,
        found=True,
        bottleneck_capacity=round(final_bottleneck, 4),
        total_cost=round(final_cost, 4),
        node_path=ordered_node_path,
        pipe_ids=ordered_pipe_ids,
        pipe_names=ordered_pipe_names,
        pipe_path=ordered_pipe_path,
        message=(
            f"Widest path found with bottleneck capacity {final_bottleneck:.2f} L/min "
            f"and routing cost {final_cost:.2f} across {len(ordered_pipe_path)} pipe(s)."
        ),
    )


# Alias for intuitive invocation
find_widest_path = widest_path

