"""
tests/test_dijkstra.py – Unit test suite for Dijkstra's Shortest-Path Algorithm.

Covers:
  1. Competing routes (picks minimum cost path over paths with fewer hops but higher cost)
  2. Zero-weight edges (allowed, cost adds up accurately)
  3. Inactive pipes (must be ignored, path takes active alternative)
  4. Parallel pipes between the same pair of nodes (selects lower cost pipe, preserves pipe identity)
  5. Disconnected nodes / components (unreachable destination returns found=False, node_path=[])
  6. Invalid node IDs (missing source, missing destination, non-existent node IDs)
  7. Source equals destination (returns found=True, cost=0.0, node_path=[source], pipe_ids=[])
  8. Negative weight rejection (raises ValueError)
  9. Inactive endpoint node (cannot route to or through inactive node)
 10. Directed vs Bidirectional mode behavior
 11. Database integration test with tmp_db SQLite fixture
"""

import os
import sys
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dsa.graph import Graph, load_graph_from_db
from dsa.dijkstra import dijkstra, find_shortest_path, DijkstraResult


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture()
def tmp_db(monkeypatch, tmp_path):
    """Isolate database tests with a fresh SQLite DB."""
    import database as db_module

    test_db = str(tmp_path / "test_dijkstra_aquagrid.db")
    monkeypatch.setattr(db_module, "INSTANCE_DIR", str(tmp_path))
    monkeypatch.setattr(db_module, "DB_PATH", test_db)
    db_module.init_db()
    return db_module


# ═════════════════════════════════════════════════════════════════════════════
#  UNIT TESTS FOR DIJKSTRA'S ALGORITHM
# ═════════════════════════════════════════════════════════════════════════════

class TestDijkstraShortestPath:
    """Test Dijkstra shortest path on in-memory Graph structures."""

    def test_competing_routes(self):
        """
        Verify that Dijkstra chooses the path with minimal routing cost,
        even if another path has fewer hops.

        Path 1 (1 hop): 1 -> 4, cost = 10.0
        Path 2 (2 hops): 1 -> 2 -> 4, cost = 3.0 + 4.0 = 7.0
        Path 3 (3 hops): 1 -> 3 -> 5 -> 4, cost = 1.0 + 1.5 + 2.0 = 4.5

        Dijkstra must pick Path 3: [1, 3, 5, 4] with total cost 4.5.
        """
        g = Graph(directed=False)
        g.add_node(1, name="Source")
        g.add_node(2, name="Via 2")
        g.add_node(3, name="Via 3")
        g.add_node(4, name="Target")
        g.add_node(5, name="Via 5")

        # Path 1
        g.add_edge(1, 4, pipe_id=101, name="Direct-HighCost", cost=10.0, status="active")
        # Path 2
        g.add_edge(1, 2, pipe_id=102, name="P1-2", cost=3.0, status="active")
        g.add_edge(2, 4, pipe_id=103, name="P2-4", cost=4.0, status="active")
        # Path 3 (Optimal)
        g.add_edge(1, 3, pipe_id=104, name="P1-3", cost=1.0, status="active")
        g.add_edge(3, 5, pipe_id=105, name="P3-5", cost=1.5, status="active")
        g.add_edge(5, 4, pipe_id=106, name="P5-4", cost=2.0, status="active")

        result = dijkstra(g, 1, 4)

        assert result.found is True
        assert result.total_cost == pytest.approx(4.5)
        assert result.node_path == [1, 3, 5, 4]
        assert result.pipe_ids == [104, 105, 106]
        assert result.pipe_names == ["P1-3", "P3-5", "P5-4"]

    def test_zero_weight_edges(self):
        """
        Zero-cost edges (e.g. frictionless connection) are valid and must be
        handled correctly without infinite loops or errors.
        """
        g = Graph(directed=False)
        g.add_node(1)
        g.add_node(2)
        g.add_node(3)

        g.add_edge(1, 2, pipe_id=201, name="ZeroCost-1", cost=0.0, status="active")
        g.add_edge(2, 3, pipe_id=202, name="ZeroCost-2", cost=0.0, status="active")

        res = dijkstra(g, 1, 3)
        assert res.found is True
        assert res.total_cost == 0.0
        assert res.node_path == [1, 2, 3]
        assert res.pipe_ids == [201, 202]

    def test_inactive_pipe_ignored(self):
        """
        If the shortest path physically uses an inactive pipe, Dijkstra must
        ignore it and select the active alternative route.
        """
        g = Graph(directed=False)
        g.add_node(1)
        g.add_node(2)
        g.add_node(3)

        # Direct pipe 1 -> 3 has cost 1.0 but is inactive
        g.add_edge(1, 3, pipe_id=301, name="Short-Inactive", cost=1.0, status="inactive")
        # Longer active path 1 -> 2 -> 3 with cost 3.0 + 3.0 = 6.0
        g.add_edge(1, 2, pipe_id=302, name="Active-A", cost=3.0, status="active")
        g.add_edge(2, 3, pipe_id=303, name="Active-B", cost=3.0, status="active")

        res = dijkstra(g, 1, 3)
        assert res.found is True
        assert res.total_cost == pytest.approx(6.0)
        assert res.node_path == [1, 2, 3]
        assert 301 not in res.pipe_ids
        assert res.pipe_ids == [302, 303]

    def test_parallel_pipes(self):
        """
        When two or more parallel pipes exist between the same pair of nodes,
        Dijkstra must select the one with minimal cost and preserve its identity.
        """
        g = Graph(directed=False)
        g.add_node(1, name="Source")
        g.add_node(2, name="Dest")

        # Two parallel pipes between node 1 and 2
        g.add_edge(1, 2, pipe_id=401, name="Parallel-Heavy", cost=8.5, status="active")
        g.add_edge(1, 2, pipe_id=402, name="Parallel-Optimal", cost=2.1, status="active")

        res = dijkstra(g, 1, 2)
        assert res.found is True
        assert res.total_cost == pytest.approx(2.1)
        assert res.node_path == [1, 2]
        assert res.pipe_ids == [402]
        assert res.pipe_names == ["Parallel-Optimal"]

    def test_disconnected_nodes(self):
        """
        If destination is unreachable from source (disconnected components),
        Dijkstra must return found=False, node_path=[], pipe_ids=[].
        """
        g = Graph(directed=False)
        # Component 1
        g.add_node(1)
        g.add_node(2)
        g.add_edge(1, 2, cost=1.0, status="active")

        # Component 2
        g.add_node(3)
        g.add_node(4)
        g.add_edge(3, 4, cost=2.0, status="active")

        res = dijkstra(g, 1, 4)
        assert res.found is False
        assert res.node_path == []
        assert res.pipe_ids == []
        assert res.total_cost == float("inf")
        assert "not reachable" in res.message.lower()

    def test_invalid_nodes(self):
        """Non-existent source or destination IDs are handled gracefully."""
        g = Graph(directed=False)
        g.add_node(1)
        g.add_node(2)
        g.add_edge(1, 2, cost=1.0, status="active")

        # Missing source
        res_src = dijkstra(g, 999, 2)
        assert res_src.found is False
        assert res_src.node_path == []
        assert "source node #999" in res_src.message.lower()

        # Missing destination
        res_dst = dijkstra(g, 1, 888)
        assert res_dst.found is False
        assert res_dst.node_path == []
        assert "destination node #888" in res_dst.message.lower()

    def test_source_equals_destination(self):
        """
        When source == destination, calculation is rejected per Requirement 1.
        Returns found=False, total_cost=None, and message 'Source and destination must be different.'
        """
        g = Graph(directed=False)
        g.add_node(1, name="Tank 1", status="active")
        g.add_node(2, name="Junction 2", status="active")
        g.add_edge(1, 2, cost=5.0, status="active")

        res = dijkstra(g, 1, 1)
        assert res.found is False
        assert res.total_cost is None
        assert res.node_path == []
        assert res.pipe_ids == []
        assert res.pipe_names == []
        assert res.message == "Source and destination must be different."

    def test_multiple_path_demonstration_network(self):
        """
        Test Requirement 3:
        Demonstration network with 4 nodes:
          A (1) -> B (2): cost 10
          A (1) -> C (3): cost 4
          C (3) -> B (2): cost 2
          B (2) -> D (4): cost 5
          C (3) -> D (4): cost 12

        Expected shortest route from A to D:
          A -> C -> B -> D
          Total cost: 4 + 2 + 5 = 11

        Verifies that Dijkstra chooses the lowest-cost path (11) rather than
        fewer hops:
          - Direct A -> B -> D (2 hops, cost 15)
          - Alternative A -> C -> D (2 hops, cost 16)
        """
        from dsa.dijkstra import create_demo_four_node_network
        g = create_demo_four_node_network(directed=False)

        res = dijkstra(g, 1, 4)
        assert res.found is True
        assert res.total_cost == pytest.approx(11.0)
        assert res.node_path == [1, 3, 2, 4]
        assert res.pipe_ids == [102, 103, 104]
        assert res.pipe_names == ["Pipe A-C", "Pipe C-B", "Pipe B-D"]

    def test_directionality_demo_network(self):
        """
        Test Requirement 4:
        In directed mode:
          A (1) -> D (4) traverses A -> C -> B -> D with cost 11.
          D (4) -> A (1) has no reverse directed path and returns No Path Available.
        In bidirectional mode:
          D (4) -> A (1) traverses D -> B -> C -> A with cost 11.
        """
        from dsa.dijkstra import create_demo_four_node_network

        # 1. Directed Mode
        g_dir = create_demo_four_node_network(directed=True)
        res_dir_forward = dijkstra(g_dir, 1, 4)
        assert res_dir_forward.found is True
        assert res_dir_forward.total_cost == pytest.approx(11.0)
        assert res_dir_forward.node_path == [1, 3, 2, 4]

        res_dir_backward = dijkstra(g_dir, 4, 1)
        assert res_dir_backward.found is False
        assert "not reachable" in res_dir_backward.message.lower()

        # 2. Bidirectional Mode
        g_bi = create_demo_four_node_network(directed=False)
        res_bi_backward = dijkstra(g_bi, 4, 1)
        assert res_bi_backward.found is True
        assert res_bi_backward.total_cost == pytest.approx(11.0)
        assert res_bi_backward.node_path == [4, 2, 3, 1]

    def test_negative_weight_rejected(self):
        """Negative routing costs must be rejected with ValueError."""
        g = Graph(directed=False)
        g.add_node(1)
        g.add_node(2)

        with pytest.raises(ValueError, match="Routing cost cannot be negative"):
            g.add_edge(1, 2, cost=-1.0, status="active")

    def test_inactive_endpoint_node(self):
        """An inactive endpoint node cannot be routed to or through."""
        g = Graph(directed=False)
        g.add_node(1, status="active")
        g.add_node(2, status="inactive")  # Inactive junction
        g.add_node(3, status="active")

        g.add_edge(1, 2, cost=1.0, status="active")
        g.add_edge(2, 3, cost=1.0, status="active")

        # Routing to inactive destination
        res_inactive_dest = dijkstra(g, 1, 2)
        assert res_inactive_dest.found is False
        assert "inactive" in res_inactive_dest.message.lower()

        # Routing through inactive intermediate node
        res_thru_inactive = dijkstra(g, 1, 3)
        assert res_thru_inactive.found is False
        assert res_thru_inactive.node_path == []

    def test_directed_vs_bidirectional_mode(self):
        """
        In directed mode (directed=True), pipe from 1 -> 2 cannot be traversed from 2 -> 1.
        In bidirectional mode (directed=False), pipe can be traversed in both directions.
        """
        # 1. Directed graph
        g_dir = Graph(directed=True)
        g_dir.add_node(1)
        g_dir.add_node(2)
        g_dir.add_edge(1, 2, pipe_id=501, cost=3.0, status="active")

        res_forward = dijkstra(g_dir, 1, 2)
        assert res_forward.found is True
        assert res_forward.total_cost == 3.0

        res_backward = dijkstra(g_dir, 2, 1)
        assert res_backward.found is False

        # 2. Bidirectional graph
        g_bi = Graph(directed=False)
        g_bi.add_node(1)
        g_bi.add_node(2)
        g_bi.add_edge(1, 2, pipe_id=502, cost=3.0, status="active")

        res_bi_back = dijkstra(g_bi, 2, 1)
        assert res_bi_back.found is True
        assert res_bi_back.total_cost == 3.0
        assert res_bi_back.node_path == [2, 1]

    def test_dijkstra_result_dict_interface(self):
        """DijkstraResult must support dictionary indexing and to_dict()."""
        g = Graph(directed=False)
        g.add_node(1)
        g.add_node(2)
        g.add_edge(1, 2, pipe_id=601, cost=4.2, status="active")

        res = dijkstra(g, 1, 2)
        assert res["total_cost"] == pytest.approx(4.2)
        assert res["node_path"] == [1, 2]
        assert res["pipe_ids"] == [601]

        as_dict = res.to_dict()
        assert isinstance(as_dict, dict)
        assert as_dict["source"] == 1
        assert as_dict["destination"] == 2
        assert as_dict["found"] is True


# ═════════════════════════════════════════════════════════════════════════════
#  INTEGRATION TEST WITH SQLITE DATABASE
# ═════════════════════════════════════════════════════════════════════════════

class TestDijkstraDatabaseIntegration:
    """Test Dijkstra loaded from SQLite DB schema."""

    def test_database_loaded_dijkstra_route(self, tmp_db):
        """
        Set up a real network in SQLite:
        - Tank Alpha (#1)
        - Junction Beta (#2)
        - Junction Gamma (#3)
        - Valve Delta (#4)
        Route A: Alpha -> Beta -> Delta (cost 5.0 + 5.0 = 10.0)
        Route B: Alpha -> Gamma -> Delta (cost 2.0 + 3.0 = 5.0)  <-- Optimal
        Parallel pipe on Gamma -> Delta (cost 1.5)               <-- Even more optimal!
        Inactive pipe Alpha -> Delta (cost 1.0, inactive)        <-- Must be ignored
        """
        id_alpha = tmp_db.create_node("Tank Alpha", "tank", "Zone 1", "active")
        id_beta = tmp_db.create_node("Junction Beta", "junction", "Zone 2", "active")
        id_gamma = tmp_db.create_node("Junction Gamma", "junction", "Zone 3", "active")
        id_delta = tmp_db.create_node("Valve Delta", "valve", "Zone 4", "active")

        # Inactive direct pipe
        tmp_db.create_pipe("Direct-Shut", id_alpha, id_delta, "100.0", "100.0", "1.0", "inactive")

        # Route A pipes
        pipe_ab = tmp_db.create_pipe("Alpha-Beta", id_alpha, id_beta, "50.0", "200.0", "5.0", "active")
        pipe_bd = tmp_db.create_pipe("Beta-Delta", id_beta, id_delta, "50.0", "200.0", "5.0", "active")

        # Route B pipes
        pipe_ag = tmp_db.create_pipe("Alpha-Gamma", id_alpha, id_gamma, "20.0", "300.0", "2.0", "active")
        pipe_gd1 = tmp_db.create_pipe("Gamma-Delta-1", id_gamma, id_delta, "30.0", "250.0", "3.0", "active")
        pipe_gd2 = tmp_db.create_pipe("Gamma-Delta-Fast", id_gamma, id_delta, "30.0", "350.0", "1.5", "active")

        graph = load_graph_from_db(db_module=tmp_db, directed=False)
        result = dijkstra(graph, id_alpha, id_delta)

        assert result.found is True
        assert result.total_cost == pytest.approx(3.5)  # 2.0 + 1.5
        assert result.node_path == [id_alpha, id_gamma, id_delta]
        assert result.pipe_ids == [pipe_ag, pipe_gd2]
        assert result.pipe_names == ["Alpha-Gamma", "Gamma-Delta-Fast"]

