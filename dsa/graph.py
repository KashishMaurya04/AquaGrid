"""
dsa/graph.py – AquaGrid Graph Engine.

Implements an adjacency-list graph representation and Breadth-First Search (BFS)
for topological reachability simulation in water distribution networks.

Data Structure
--------------
- The graph is represented as an adjacency list using Python dictionaries and lists:
      adjacency_list: dict[int, list[int]]
  where each key is a vertex (node ID) and each value is an ordered list of
  neighboring node IDs connected via active pipes.
- Metadata for vertices (name, type, location, status) and edges (pipe ID, name,
  length, capacity, cost, status) are stored in accompanying dictionaries for
  rich graph simulation inspection without coupling the graph engine to database
  or web frameworks.

Time and Space Complexity
-------------------------
1. Graph Construction (from database or manual addition):
   - Time Complexity: O(V + E), where V is the number of network nodes and E is
     the number of pipes. Loading and inserting each node and edge takes O(1)
     amortized time.
   - Space Complexity: O(V + E) for the adjacency list and edge/node metadata.
     Isolated nodes are represented with an empty list O(1) space.

2. Breadth-First Search (breadth_first_search):
   - Time Complexity: O(V + E)
     Every reachable vertex is enqueued and dequeued at most once via collections.deque
     with O(1) popleft/append operations. Each incident edge is traversed at most
     once (or twice in undirected mode). Visited set lookups are O(1) average time.
   - Space Complexity: O(V)
     The visited set, traversal queue, and result list each store at most V node IDs.
"""

from __future__ import annotations

import sqlite3
from collections import deque
from typing import Any


class Graph:
    """
    Adjacency-list graph representation of a water distribution network.

    Vertices represent network nodes (tanks, junctions, valves).
    Edges represent valid, active pipes connecting pairs of nodes.
    """

    def __init__(self, directed: bool = False) -> None:
        """
        Initialize an empty network graph.

        Args:
            directed: If False (default), pipes are treated as bidirectional
                      conduits for reachability. If True, reachability only
                      follows from_node -> to_node direction.
        """
        self.directed: bool = directed
        # Core adjacency list: vertex ID -> list of neighbor vertex IDs
        self.adjacency_list: dict[int, list[int]] = {}
        # Incident active edges list: vertex ID -> list of incident edge dictionaries
        # Preserves all parallel pipes and individual routing costs
        self.adj_edges: dict[int, list[dict[str, Any]]] = {}
        # Node metadata: vertex ID -> dictionary of node properties
        self.nodes: dict[int, dict[str, Any]] = {}
        # Active edges metadata: list of active pipe dictionaries
        self.edges: list[dict[str, Any]] = []
        # Inactive pipes metadata: list of inactive pipes (ignored for reachability)
        self.inactive_pipes: list[dict[str, Any]] = []

    # ── Node and Edge Management ──────────────────────────────────────────────

    def add_node(
        self,
        node_id: int,
        name: str = "",
        node_type: str = "junction",
        location: str | None = None,
        status: str = "active",
        **extra: Any,
    ) -> None:
        """
        Add a network node (vertex) to the graph.

        Includes isolated nodes with an empty adjacency list.
        If the node already exists, updates its metadata without clearing edges.
        """
        if node_id not in self.adjacency_list:
            self.adjacency_list[node_id] = []
        if node_id not in self.adj_edges:
            self.adj_edges[node_id] = []

        self.nodes[node_id] = {
            "id": node_id,
            "name": name or f"Node-{node_id}",
            "node_type": node_type,
            "location": location,
            "status": status,
            **extra,
        }

    def has_node(self, node_id: int) -> bool:
        """Return True if node_id is a vertex in the graph."""
        return node_id in self.adjacency_list

    def add_edge(
        self,
        from_node_id: int,
        to_node_id: int,
        pipe_id: int | None = None,
        name: str = "",
        status: str = "active",
        length_m: float = 0.0,
        capacity_lpm: float = 0.0,
        cost: float = 1.0,
        **extra: Any,
    ) -> bool:
        """
        Add a pipe (edge) to the graph.

        Only ACTIVE pipes are added to the reachability adjacency list.
        Inactive pipes are recorded in `inactive_pipes` and ignored for reachability.

        Returns:
            True if the edge was added to the reachability adjacency list,
            False if it was ignored (e.g. inactive status).
        """
        cost_val = float(cost)
        if cost_val < 0:
            raise ValueError(
                f"Routing cost cannot be negative (got {cost_val} for pipe '{name or pipe_id}'). "
                f"Dijkstra's shortest path requires non-negative edge weights."
            )

        edge_data = {
            "pipe_id": pipe_id,
            "name": name or (f"Pipe-{pipe_id}" if pipe_id else f"{from_node_id}->{to_node_id}"),
            "from_node_id": from_node_id,
            "to_node_id": to_node_id,
            "status": status,
            "length_m": float(length_m),
            "capacity_lpm": float(capacity_lpm),
            "cost": cost_val,
            **extra,
        }

        # Ensure both endpoints exist in the graph (e.g. for standalone testing)
        if from_node_id not in self.adjacency_list:
            self.add_node(from_node_id)
        if to_node_id not in self.adjacency_list:
            self.add_node(to_node_id)

        # Inactive pipes must be ignored for reachability and routing
        if status != "active":
            self.inactive_pipes.append(edge_data)
            return False

        # Add to active edges collection
        self.edges.append(edge_data)

        # Add to incident active edges list (preserves parallel pipes and edge identities)
        fwd_edge = dict(edge_data)
        fwd_edge["neighbor_id"] = to_node_id
        self.adj_edges[from_node_id].append(fwd_edge)

        # Update adjacency list (avoiding duplicate neighbor entries for BFS)
        if to_node_id not in self.adjacency_list[from_node_id]:
            self.adjacency_list[from_node_id].append(to_node_id)

        if not self.directed:
            rev_edge = dict(edge_data)
            rev_edge["neighbor_id"] = from_node_id
            self.adj_edges[to_node_id].append(rev_edge)

            if from_node_id not in self.adjacency_list[to_node_id]:
                self.adjacency_list[to_node_id].append(from_node_id)

        return True

    def get_outgoing_edges(self, node_id: int) -> list[dict[str, Any]]:
        """Return all active outgoing/incident edges from node_id, including parallel pipes."""
        return list(self.adj_edges.get(node_id, []))

    def dijkstra(self, source: int, destination: int) -> Any:
        """Find the shortest path from source to destination using Dijkstra's algorithm."""
        from dsa.dijkstra import dijkstra
        return dijkstra(self, source, destination)

    def widest_path(self, source: int, destination: int) -> Any:
        """Find the maximum bottleneck path from source to destination."""
        from dsa.widest_path import widest_path
        return widest_path(self, source, destination)

    def get_neighbors(self, node_id: int) -> list[int]:
        """Return a copy of neighbor node IDs reachable from node_id."""
        return list(self.adjacency_list.get(node_id, []))

    @property
    def node_count(self) -> int:
        """Total number of vertices in the graph."""
        return len(self.adjacency_list)

    @property
    def edge_count(self) -> int:
        """Total number of active reachability edges in the graph."""
        return len(self.edges)

    # ── Breadth-First Search (BFS) ─────────────────────────────────────────────

    def breadth_first_search(self, start_node_id: int) -> list[int]:
        """
        Perform Breadth-First Search starting from `start_node_id`.

        Traverses the graph layer-by-layer to discover all reachable network nodes.
        Uses `collections.deque` for FIFO queue operations and a `set` for O(1)
        visited-state lookups to guarantee each node is visited at most once.

        Args:
            start_node_id: The ID of the root node to begin traversal from.

        Returns:
            An ordered list of reachable node IDs, starting with `start_node_id`.
            Returns an empty list `[]` if `start_node_id` is invalid, missing,
            or if the graph is empty.
        """
        # Gracefully handle missing nodes, invalid IDs, and empty graphs
        if start_node_id not in self.adjacency_list:
            return []

        visited: set[int] = set()
        queue: deque[int] = deque([start_node_id])
        visited.add(start_node_id)
        traversal_order: list[int] = []

        while queue:
            current = queue.popleft()
            traversal_order.append(current)

            for neighbor in self.adjacency_list.get(current, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        return traversal_order

    def bfs_detailed(self, start_node_id: int) -> dict[str, Any]:
        """
        Perform BFS and return comprehensive simulation details for the Graph Explorer.

        Provides traversal distance (levels), discovery parents, active connections
        used within the reachable component, and isolation diagnostics.
        """
        if start_node_id not in self.adjacency_list:
            return {
                "start_node_id": start_node_id,
                "found": False,
                "reachable_node_ids": [],
                "nodes": [],
                "connections_used": [],
                "is_isolated": False,
                "total_nodes": self.node_count,
            }

        visited: set[int] = set()
        queue: deque[tuple[int, int]] = deque([(start_node_id, 0)])  # (node_id, depth)
        visited.add(start_node_id)

        ordered_nodes: list[dict[str, Any]] = []
        depth_map: dict[int, int] = {start_node_id: 0}

        while queue:
            current_id, depth = queue.popleft()
            node_meta = self.nodes.get(current_id, {"id": current_id, "name": f"Node-{current_id}"})
            ordered_nodes.append({
                **node_meta,
                "bfs_step": len(ordered_nodes) + 1,
                "depth": depth,
            })

            for neighbor in self.adjacency_list.get(current_id, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    depth_map[neighbor] = depth + 1
                    queue.append((neighbor, depth + 1))

        reachable_ids = [n["id"] for n in ordered_nodes]
        reachable_set = set(reachable_ids)

        # Active connections connecting reachable nodes
        connections_used: list[dict[str, Any]] = []
        for edge in self.edges:
            # An edge is part of this connected subgraph if both ends are reachable
            if edge["from_node_id"] in reachable_set and edge["to_node_id"] in reachable_set:
                connections_used.append(edge)

        is_isolated = (len(reachable_ids) == 1 and len(self.adjacency_list.get(start_node_id, [])) == 0)

        return {
            "start_node_id": start_node_id,
            "found": True,
            "reachable_node_ids": reachable_ids,
            "nodes": ordered_nodes,
            "connections_used": connections_used,
            "is_isolated": is_isolated,
            "total_nodes": self.node_count,
            "total_reachable": len(reachable_ids),
        }

    # ── Database Loader ───────────────────────────────────────────────────────

    @classmethod
    def from_db(
        cls,
        db_module: Any = None,
        db_path: str | None = None,
        directed: bool = False,
    ) -> Graph:
        """
        Build and populate a Graph from the SQLite database.

        Loads:
        1. All records from `network_nodes` as vertices (including isolated nodes).
        2. All records from `pipes`, adding active pipes as edges and ignoring
           inactive pipes for reachability.

        Args:
            db_module: Optional database access module (defaults to project's `database.py`).
            db_path: Optional explicit SQLite database file path.
            directed: Whether to build a directed graph (default False).
        """
        graph = cls(directed=directed)

        # If a db_module is provided, use its connection or query functions
        if db_module is not None:
            nodes = db_module.get_all_nodes()
            pipes = db_module.get_all_pipes()
        elif db_path is not None:
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            with conn:
                nodes = conn.execute("SELECT * FROM network_nodes ORDER BY id").fetchall()
                pipes = conn.execute("SELECT * FROM pipes ORDER BY id").fetchall()
        else:
            # Import project database module dynamically to use current configuration
            import database as db
            nodes = db.get_all_nodes()
            pipes = db.get_all_pipes()

        # Step 1: Add all nodes as vertices (including isolated nodes)
        for node in nodes:
            graph.add_node(
                node_id=int(node["id"]),
                name=node["name"],
                node_type=node["node_type"],
                location=node["location"],
                status=node["status"],
                created_at=node["created_at"],
            )

        # Step 2: Add pipes as edges (ignoring inactive pipes for reachability)
        for pipe in pipes:
            from_id = int(pipe["from_node_id"])
            to_id = int(pipe["to_node_id"])
            status = pipe["status"]

            # Respect foreign keys: if either node is somehow missing, ensure vertex exists
            if from_id not in graph.adjacency_list:
                graph.add_node(from_id)
            if to_id not in graph.adjacency_list:
                graph.add_node(to_id)

            graph.add_edge(
                from_node_id=from_id,
                to_node_id=to_id,
                pipe_id=int(pipe["id"]),
                name=pipe["name"],
                status=status,
                length_m=float(pipe["length_m"]),
                capacity_lpm=float(pipe["capacity_lpm"]),
                cost=float(pipe["cost"]),
                created_at=pipe["created_at"],
            )

        return graph


# ── Standalone module-level functions ─────────────────────────────────────────

def breadth_first_search(graph_or_start_id: Graph | int, start_node_id: int | None = None) -> list[int]:
    """
    Breadth-First Search helper function.

    Supports both calling styles:
        breadth_first_search(graph, start_node_id)
        breadth_first_search(start_node_id)  [when called on a Graph instance or loaded from db]
    """
    if isinstance(graph_or_start_id, Graph):
        if start_node_id is None:
            raise ValueError("start_node_id is required when passing a Graph instance.")
        return graph_or_start_id.breadth_first_search(start_node_id)

    # If called as breadth_first_search(start_node_id), load graph from default DB
    start_id = int(graph_or_start_id)
    graph = load_graph_from_db()
    return graph.breadth_first_search(start_id)


def load_graph_from_db(
    db_module: Any = None,
    db_path: str | None = None,
    directed: bool = False,
) -> Graph:
    """Load and return an adjacency-list Graph from the database."""
    return Graph.from_db(db_module=db_module, db_path=db_path, directed=directed)

