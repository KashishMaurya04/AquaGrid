"""
dsa/dijkstra.py – Dijkstra's Shortest-Path Algorithm for AquaGrid.

Computes optimal shortest paths between water network nodes based on pipe routing
costs using a min-priority queue (Python's `heapq`).

Features & Constraints:
-----------------------
- Priority Queue: Python `heapq` as the min-priority queue.
- Edge Weights: Uses `cost` (routing weight) as edge weight. Rejects negative weights,
  allows zero weights.
- Filtering: Only active pipes (`status == 'active'`) and active, valid endpoint nodes
  (`status == 'active'`) are considered.
- Directionality: Fully respects the graph's `directed` mode (Bidirectional vs Directed).
- Parallel Pipes: Correctly supports multiple pipes between the same pair of nodes,
  evaluating each independently and selecting the optimal pipe.
- Edge Cases: Handles empty networks, invalid node IDs, inactive nodes, unreachable
  destinations, and source == destination gracefully.

Time and Space Complexity:
--------------------------
- Time Complexity: O((V + E) log V)
  Using a binary min-heap (`heapq`), each vertex is extracted at most once, and each
  incident edge is relaxed, taking O(log V) per heap push operation.
- Space Complexity: O(V + E)
  Storage for the min-heap, distances dictionary, predecessor mappings, and path records.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from dsa.graph import Graph


@dataclass
class DijkstraResult:
    """
    Encapsulates the result of a Dijkstra shortest-path query.

    Supports both attribute access (result.total_cost) and dict-style indexing
    (result["total_cost"]) for seamless integration with templates and tests.
    """
    source: int | None
    destination: int | None
    found: bool
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
            "total_cost": self.total_cost,
            "node_path": list(self.node_path),
            "pipe_ids": list(self.pipe_ids),
            "pipe_names": list(self.pipe_names),
            "pipe_path": list(self.pipe_path),
            "message": self.message,
        }


def dijkstra(graph: Graph, source: int, destination: int) -> DijkstraResult:
    """
    Compute the shortest path from `source` to `destination` using Dijkstra's algorithm.

    Args:
        graph: An adjacency-list `Graph` instance (from `dsa.graph`).
        source: The start node ID.
        destination: The target node ID.

    Returns:
        DijkstraResult containing `source`, `destination`, `found`, `total_cost`,
        `node_path`, `pipe_ids`, `pipe_names`, and `pipe_path`.

    Raises:
        ValueError: If any edge weight (routing cost) is negative.
    """
    # ── 1. Validate empty graph and existence of source / destination ──────────
    if graph.node_count == 0:
        return DijkstraResult(
            source=source,
            destination=destination,
            found=False,
            total_cost=float("inf"),
            message="Network is empty. No nodes exist in the graph.",
        )

    if not graph.has_node(source):
        return DijkstraResult(
            source=source,
            destination=destination,
            found=False,
            total_cost=float("inf"),
            message=f"Source node #{source} does not exist in the network.",
        )

    if not graph.has_node(destination):
        return DijkstraResult(
            source=source,
            destination=destination,
            found=False,
            total_cost=float("inf"),
            message=f"Destination node #{destination} does not exist in the network.",
        )

    # ── 2. Validate endpoint active status ────────────────────────────────────
    source_meta = graph.nodes.get(source, {})
    dest_meta = graph.nodes.get(destination, {})

    if source_meta.get("status") != "active":
        return DijkstraResult(
            source=source,
            destination=destination,
            found=False,
            total_cost=float("inf"),
            message=f"Source node #{source} ('{source_meta.get('name', '')}') is inactive.",
        )

    if dest_meta.get("status") != "active":
        return DijkstraResult(
            source=source,
            destination=destination,
            found=False,
            total_cost=float("inf"),
            message=f"Destination node #{destination} ('{dest_meta.get('name', '')}') is inactive.",
        )

    # ── 3. Validate identical source and destination endpoints ────────────────
    if source == destination:
        return DijkstraResult(
            source=source,
            destination=destination,
            found=False,
            total_cost=None,
            node_path=[],
            pipe_ids=[],
            pipe_names=[],
            pipe_path=[],
            message="Source and destination must be different.",
        )

    # ── 4. Initialize Dijkstra data structures ────────────────────────────────
    # Map node_id -> minimum tentative distance from source
    distances: dict[int, float] = {source: 0.0}

    # Map node_id -> (predecessor_node_id, pipe_edge_dict)
    predecessors: dict[int, tuple[int, dict[str, Any]]] = {}

    # Set of settled vertices whose shortest distance is finalized
    visited: set[int] = set()

    # Min-priority queue: tuples of (cumulative_cost, node_id)
    # Python heapq orders by first tuple element (cumulative_cost)
    heap: list[tuple[float, int]] = []
    heapq.heappush(heap, (0.0, source))

    # ── 5. Main Dijkstra Relaxation Loop ──────────────────────────────────────
    while heap:
        curr_dist, u = heapq.heappop(heap)

        if u in visited:
            continue
        visited.add(u)

        # Early termination: destination reached with minimal distance
        if u == destination:
            break

        # If a stale heap entry has higher distance than already found, skip
        if curr_dist > distances.get(u, float("inf")):
            continue

        # Explore all active incident/outgoing edges (including parallel pipes)
        for edge in graph.get_outgoing_edges(u):
            # Inactive pipes must be ignored
            if edge.get("status") != "active":
                continue

            v = edge.get("neighbor_id")
            if v is None:
                continue

            # Inactive endpoint nodes must not be traversed
            v_meta = graph.nodes.get(v, {})
            if v_meta.get("status") != "active":
                continue

            weight = float(edge.get("cost", 0.0))
            if weight < 0:
                raise ValueError(
                    f"Routing cost cannot be negative (got {weight} on pipe #{edge.get('pipe_id')}). "
                    f"Dijkstra's algorithm requires non-negative edge weights."
                )

            new_dist = curr_dist + weight

            if new_dist < distances.get(v, float("inf")):
                distances[v] = new_dist
                predecessors[v] = (u, edge)
                heapq.heappush(heap, (new_dist, v))

    # ── 6. Check if destination was reached ───────────────────────────────────
    if destination not in predecessors:
        return DijkstraResult(
            source=source,
            destination=destination,
            found=False,
            total_cost=float("inf"),
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
    final_cost = float(distances[destination])

    return DijkstraResult(
        source=source,
        destination=destination,
        found=True,
        total_cost=final_cost,
        node_path=ordered_node_path,
        pipe_ids=ordered_pipe_ids,
        pipe_names=ordered_pipe_names,
        pipe_path=ordered_pipe_path,
        message=f"Shortest path found with total routing cost {final_cost:.2f}.",
    )


# Alias for intuitive invocation
find_shortest_path = dijkstra



def create_demo_four_node_network(directed: bool = False) -> Graph:
    """
    Create the demonstration network specified in Requirement 3:
        Nodes:
            1: Tank A
            2: Junction B
            3: Junction C
            4: Valve D
        Pipes:
            A -> B: cost 10.0
            A -> C: cost 4.0
            C -> B: cost 2.0
            B -> D: cost 5.0
            C -> D: cost 12.0
        Expected shortest route from A (1) to D (4):
            A -> C -> B -> D
            Total cost: 11.0 (4 + 2 + 5 = 11)
    """
    from dsa.graph import Graph
    g = Graph(directed=directed)
    g.add_node(1, name="Tank A", node_type="tank", status="active")
    g.add_node(2, name="Junction B", node_type="junction", status="active")
    g.add_node(3, name="Junction C", node_type="junction", status="active")
    g.add_node(4, name="Valve D", node_type="valve", status="active")

    g.add_edge(1, 2, pipe_id=101, name="Pipe A-B", cost=10.0, length_m=1000.0, status="active")
    g.add_edge(1, 3, pipe_id=102, name="Pipe A-C", cost=4.0, length_m=400.0, status="active")
    g.add_edge(3, 2, pipe_id=103, name="Pipe C-B", cost=2.0, length_m=200.0, status="active")
    g.add_edge(2, 4, pipe_id=104, name="Pipe B-D", cost=5.0, length_m=500.0, status="active")
    g.add_edge(3, 4, pipe_id=105, name="Pipe C-D", cost=12.0, length_m=1200.0, status="active")
    return g
