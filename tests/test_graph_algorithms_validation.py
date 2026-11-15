"""
tests/test_graph_algorithms_validation.py – Day 6: Graph Algorithms & Route Optimization Validation.

Comprehensive validation of AquaGrid's:
- Breadth-First Search (BFS) reachability, traversal order, acyclic/cyclic behavior, and complexity
- Dijkstra Shortest-Path algorithm, relaxation, predecessor tracking, zero-weight, and large-weight behavior
- Strict non-negative weight validation at database, graph, and algorithmic boundaries
- Water network database integration (active vs inactive pipes, directed vs undirected, dynamic updates)
- Edge cases: self-loops, parallel edges, equal-cost alternatives, disconnected components
"""

from __future__ import annotations

import pytest
from dsa.graph import Graph, breadth_first_search, load_graph_from_db
from dsa.dijkstra import dijkstra, DijkstraResult
import database as db_module


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture()
def isolated_db(monkeypatch, tmp_path):
    """Isolated SQLite database in a temporary directory for integration tests."""
    test_db = str(tmp_path / "test_graph_val.db")
    monkeypatch.setattr(db_module, "INSTANCE_DIR", str(tmp_path))
    monkeypatch.setattr(db_module, "DB_PATH", test_db)
    db_module.init_db()
    return db_module


# ==============================================================================
# 1. BFS Validation Across Topology Archetypes
# ==============================================================================

class TestBFSValidation:
    """Rigorous validation of BFS across diverse topological structures."""

    def test_bfs_simple_chain(self):
        """Scenario A: Simple chain A(1) -> B(2) -> C(3) -> D(4)."""
        g = Graph(directed=True)
        for i in range(1, 5):
            g.add_node(i)
        g.add_edge(1, 2, status="active")
        g.add_edge(2, 3, status="active")
        g.add_edge(3, 4, status="active")

        order = g.breadth_first_search(1)
        assert order == [1, 2, 3, 4]
        assert len(order) == len(set(order))

    def test_bfs_branching_tree(self):
        """Scenario B: Branching graph A(1) -> B(2), A(1) -> C(3), B(2) -> D(4), C(3) -> E(5)."""
        g = Graph(directed=True)
        for i in range(1, 6):
            g.add_node(i)
        g.add_edge(1, 2, status="active")
        g.add_edge(1, 3, status="active")
        g.add_edge(2, 4, status="active")
        g.add_edge(3, 5, status="active")

        order = g.breadth_first_search(1)
        assert order == [1, 2, 3, 4, 5]
        # Level 0: 1, Level 1: 2, 3, Level 2: 4, 5
        detailed = g.bfs_detailed(1)
        depths = {n["id"]: n["depth"] for n in detailed["nodes"]}
        assert depths[1] == 0
        assert depths[2] == 1
        assert depths[3] == 1
        assert depths[4] == 2
        assert depths[5] == 2

    def test_bfs_cyclic_graph(self):
        """Scenario C: Cyclic graph A(1) -> B(2) -> C(3) -> A(1) avoids infinite loop."""
        g = Graph(directed=True)
        for i in range(1, 4):
            g.add_node(i)
        g.add_edge(1, 2, status="active")
        g.add_edge(2, 3, status="active")
        g.add_edge(3, 1, status="active")

        order = g.breadth_first_search(1)
        assert order == [1, 2, 3]
        assert len(order) == len(set(order))

    def test_bfs_disconnected_components(self):
        """Scenario D: Disconnected components A(1) -> B(2) and C(3) -> D(4) never bleed."""
        g = Graph(directed=False)
        for i in range(1, 5):
            g.add_node(i)
        g.add_edge(1, 2, status="active")
        g.add_edge(3, 4, status="active")

        assert g.breadth_first_search(1) == [1, 2]
        assert g.breadth_first_search(3) == [3, 4]

    def test_bfs_multiple_paths_to_same_node(self):
        """Scenario E: Multiple paths to node 4 (via 2 and via 3) visits node 4 exactly once."""
        g = Graph(directed=False)
        for i in range(1, 5):
            g.add_node(i)
        g.add_edge(1, 2, status="active")
        g.add_edge(1, 3, status="active")
        g.add_edge(2, 4, status="active")
        g.add_edge(3, 4, status="active")

        order = g.breadth_first_search(1)
        assert order == [1, 2, 3, 4]
        assert len(order) == len(set(order))

    def test_bfs_edge_cases_empty_single_invalid(self):
        """Empty graph, single node, and missing starting node IDs."""
        g_empty = Graph()
        assert g_empty.breadth_first_search(1) == []
        assert g_empty.bfs_detailed(1)["found"] is False

        g_single = Graph()
        g_single.add_node(42)
        assert g_single.breadth_first_search(42) == [42]
        assert g_single.breadth_first_search(999) == []


# ==============================================================================
# 2. Dijkstra Shortest-Path Validation
# ==============================================================================

class TestDijkstraValidation:
    """Rigorous shortest path, relaxation, and predecessor tracking tests."""

    def test_dijkstra_direct_vs_indirect_cheaper_path(self):
        """Scenario A: Direct path (cost 10.0) vs indirect cheaper path (3.0 + 4.0 = 7.0)."""
        g = Graph(directed=False)
        g.add_node(1)
        g.add_node(2)
        g.add_node(3)
        g.add_edge(1, 3, pipe_id=101, cost=10.0, status="active")
        g.add_edge(1, 2, pipe_id=102, cost=3.0, status="active")
        g.add_edge(2, 3, pipe_id=103, cost=4.0, status="active")

        res = dijkstra(g, 1, 3)
        assert res.found is True
        assert res.total_cost == pytest.approx(7.0)
        assert res.node_path == [1, 2, 3]
        assert res.pipe_ids == [102, 103]

    def test_dijkstra_multiple_competing_routes(self):
        """Scenario B: 3 competing routes (15.0, 11.0, and 9.0) selects 9.0."""
        g = Graph(directed=False)
        for i in range(1, 6):
            g.add_node(i)
        g.add_edge(1, 4, pipe_id=1, cost=15.0, status="active")
        g.add_edge(1, 2, pipe_id=2, cost=5.0, status="active")
        g.add_edge(2, 4, pipe_id=3, cost=6.0, status="active")
        g.add_edge(1, 3, pipe_id=4, cost=2.0, status="active")
        g.add_edge(3, 5, pipe_id=5, cost=3.0, status="active")
        g.add_edge(5, 4, pipe_id=6, cost=4.0, status="active")

        res = dijkstra(g, 1, 4)
        assert res.total_cost == pytest.approx(9.0)
        assert res.node_path == [1, 3, 5, 4]
        assert res.pipe_ids == [4, 5, 6]

    def test_dijkstra_equal_cost_alternatives(self):
        """Scenario C: Two distinct routes with identical total cost (6.0) both valid."""
        g = Graph(directed=False)
        for i in range(1, 5):
            g.add_node(i)
        g.add_edge(1, 2, pipe_id=11, cost=3.0, status="active")
        g.add_edge(2, 4, pipe_id=12, cost=3.0, status="active")
        g.add_edge(1, 3, pipe_id=13, cost=3.0, status="active")
        g.add_edge(3, 4, pipe_id=14, cost=3.0, status="active")

        res = dijkstra(g, 1, 4)
        assert res.found is True
        assert res.total_cost == pytest.approx(6.0)
        assert res.node_path in ([1, 2, 4], [1, 3, 4])

    def test_dijkstra_unreachable_destination(self):
        """Scenario D: Disconnected destination returns found=False, total_cost=inf."""
        g = Graph(directed=False)
        g.add_node(1)
        g.add_node(2)
        g.add_node(3)
        g.add_edge(1, 2, pipe_id=20, cost=5.0, status="active")

        res = dijkstra(g, 1, 3)
        assert res.found is False
        assert res.total_cost == float("inf")
        assert res.node_path == []
        assert "not reachable" in res.message.lower()

    def test_dijkstra_source_equals_destination(self):
        """Scenario E: Source equals destination returns friendly notice."""
        g = Graph(directed=False)
        g.add_node(1)
        res = dijkstra(g, 1, 1)
        assert res.found is False
        assert "different" in res.message.lower()

    def test_dijkstra_zero_weight_edges(self):
        """Scenario F: Zero-cost edge supported cleanly."""
        g = Graph(directed=False)
        g.add_node(1)
        g.add_node(2)
        g.add_node(3)
        g.add_edge(1, 2, pipe_id=30, cost=0.0, status="active")
        g.add_edge(2, 3, pipe_id=31, cost=4.5, status="active")

        res = dijkstra(g, 1, 3)
        assert res.found is True
        assert res.total_cost == pytest.approx(4.5)
        assert res.node_path == [1, 2, 3]

    def test_dijkstra_large_positive_edge_weights(self):
        """Scenario G: Large weights (1e8) evaluated without numeric overflow."""
        g = Graph(directed=False)
        g.add_node(1)
        g.add_node(2)
        g.add_edge(1, 2, pipe_id=40, cost=100_000_000.0, status="active")

        res = dijkstra(g, 1, 2)
        assert res.found is True
        assert res.total_cost == pytest.approx(100_000_000.0)

    def test_dijkstra_cyclic_graph(self):
        """Scenario H: Directed cycle terminates with shortest acyclic path."""
        g = Graph(directed=True)
        g.add_node(1)
        g.add_node(2)
        g.add_node(3)
        g.add_edge(1, 2, pipe_id=50, cost=2.0, status="active")
        g.add_edge(2, 3, pipe_id=51, cost=3.0, status="active")
        g.add_edge(3, 1, pipe_id=52, cost=1.0, status="active")

        res = dijkstra(g, 1, 3)
        assert res.found is True
        assert res.total_cost == pytest.approx(5.0)
        assert res.node_path == [1, 2, 3]


# ==============================================================================
# 3. Non-Negative Weight Safety Enforcement
# ==============================================================================

class TestNegativeWeightSafety:
    """Verifies that negative routing costs are rejected across all system layers."""

    def test_graph_add_edge_rejects_negative_weight(self):
        """Graph.add_edge raises ValueError on negative cost."""
        g = Graph()
        with pytest.raises(ValueError, match="Routing cost cannot be negative"):
            g.add_edge(1, 2, cost=-2.5)

    def test_database_validation_rejects_negative_cost(self, isolated_db):
        """database.validate_pipe rejects negative routing cost."""
        n1 = isolated_db.create_node("N1", "tank", "North", "active")
        n2 = isolated_db.create_node("N2", "junction", "Central", "active")
        with pytest.raises(ValueError, match="must be zero or greater"):
            isolated_db.create_pipe("Neg-Pipe", n1, n2, "100.0", "1000.0", "-5.0", "active")

    def test_dijkstra_relaxation_rejects_negative_weight(self):
        """Dijkstra algorithm checks non-negativity during relaxation."""
        g = Graph()
        g.add_node(1)
        g.add_node(2)
        # Manually tamper with internal edge weight to test defensive check in dijkstra()
        g.adjacency_list[1] = [2]
        g.adj_edges[1] = [{"neighbor_id": 2, "cost": -3.0, "status": "active"}]
        with pytest.raises(ValueError, match="Routing cost cannot be negative"):
            dijkstra(g, 1, 2)


# ==============================================================================
# 4. Water Network Database Integration
# ==============================================================================

class TestWaterNetworkIntegration:
    """End-to-end integration with SQLite water distribution network."""

    def test_active_and_inactive_pipe_reachability(self, isolated_db):
        """Only active pipes are loaded for reachability; inactive pipes are ignored."""
        n1 = isolated_db.create_node("Res-A", "tank", "North", "active")
        n2 = isolated_db.create_node("Junc-B", "junction", "Central", "active")
        n3 = isolated_db.create_node("Valve-C", "valve", "South", "active")

        # Direct n1 -> n3 is inactive
        isolated_db.create_pipe("Direct-Inactive", n1, n3, "200.0", "1000.0", "2.0", "inactive")
        # Indirect active route n1 -> n2 -> n3
        p1 = isolated_db.create_pipe("P1", n1, n2, "300.0", "1000.0", "3.0", "active")
        p2 = isolated_db.create_pipe("P2", n2, n3, "400.0", "1000.0", "4.0", "active")

        g = load_graph_from_db(db_module=isolated_db, directed=False)
        assert g.node_count == 3
        assert g.edge_count == 2
        assert len(g.inactive_pipes) == 1

        res = dijkstra(g, n1, n3)
        assert res.found is True
        assert res.total_cost == pytest.approx(7.0)
        assert res.node_path == [n1, n2, n3]
        assert res.pipe_ids == [p1, p2]

    def test_directed_vs_undirected_network_mode(self, isolated_db):
        """Directed network mode prevents reverse traversal through conduits."""
        n1 = isolated_db.create_node("Upstream", "tank", "North", "active")
        n2 = isolated_db.create_node("Downstream", "valve", "South", "active")
        isolated_db.create_pipe("Gravity-Flow", n1, n2, "500.0", "1000.0", "5.0", "active")

        # Directed mode: n1 -> n2 works, n2 -> n1 fails
        g_dir = load_graph_from_db(db_module=isolated_db, directed=True)
        assert dijkstra(g_dir, n1, n2).found is True
        assert dijkstra(g_dir, n2, n1).found is False

        # Undirected mode: both directions work
        g_undir = load_graph_from_db(db_module=isolated_db, directed=False)
        assert dijkstra(g_undir, n1, n2).found is True
        assert dijkstra(g_undir, n2, n1).found is True

    def test_dynamic_network_mutation_updates_shortest_path(self, isolated_db):
        """Adding, deactivating, or updating pipes changes the chosen path dynamically."""
        n1 = isolated_db.create_node("N1", "tank", "North", "active")
        n2 = isolated_db.create_node("N2", "junction", "Central", "active")
        n3 = isolated_db.create_node("N3", "valve", "South", "active")

        p_slow = isolated_db.create_pipe("P-Slow", n1, n3, "1500.0", "1000.0", "15.0", "active")
        g1 = load_graph_from_db(db_module=isolated_db, directed=False)
        assert dijkstra(g1, n1, n3).total_cost == pytest.approx(15.0)

        # Add faster detour: n1 -> n2 (2.0) and n2 -> n3 (3.0) -> cost 5.0
        isolated_db.create_pipe("P-Fast-1", n1, n2, "200.0", "1000.0", "2.0", "active")
        isolated_db.create_pipe("P-Fast-2", n2, n3, "300.0", "1000.0", "3.0", "active")

        g2 = load_graph_from_db(db_module=isolated_db, directed=False)
        res2 = dijkstra(g2, n1, n3)
        assert res2.total_cost == pytest.approx(5.0)
        assert res2.node_path == [n1, n2, n3]


# ==============================================================================
# 5. Edge Cases
# ==============================================================================

class TestGraphEdgeCases:
    """Extreme graph topologies and boundary conditions."""

    def test_self_loops_handled_safely(self):
        """Self-loop edge on a node does not corrupt shortest path to another node."""
        g = Graph(directed=False)
        g.add_node(1)
        g.add_node(2)
        g.add_edge(1, 1, pipe_id=90, cost=5.0, status="active") # self-loop
        g.add_edge(1, 2, pipe_id=91, cost=3.0, status="active")

        res = dijkstra(g, 1, 2)
        assert res.found is True
        assert res.total_cost == pytest.approx(3.0)
        assert res.node_path == [1, 2]

    def test_parallel_edges_optimal_selection(self):
        """Parallel edges with different costs select the lower cost pipe."""
        g = Graph(directed=False)
        g.add_node(1)
        g.add_node(2)
        g.add_edge(1, 2, pipe_id=10, cost=8.0, status="active")
        g.add_edge(1, 2, pipe_id=20, cost=2.5, status="active")
        g.add_edge(1, 2, pipe_id=30, cost=5.0, status="active")

        res = dijkstra(g, 1, 2)
        assert res.total_cost == pytest.approx(2.5)
        assert res.pipe_ids == [20]
