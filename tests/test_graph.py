"""
tests/test_graph.py – Unit test suite for the AquaGrid Graph Engine (DSA).

Covers:
  - A simple connected graph (2 nodes, 1 active pipe)
  - A graph with three connected nodes (linear & cycle)
  - An isolated node (no incident pipes)
  - Two disconnected components
  - An inactive pipe (ignored for reachability)
  - An empty graph (0 nodes, 0 edges)
  - An invalid starting node (non-existent ID)
  - A graph loaded from the database (SQLite integration via tmp_db)
  - Directed vs undirected reachability behavior
  - BFS queue & visited set behavior (no duplicate visits)
"""

import os
import sys
import pytest

# ── Ensure project root is on sys.path ────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dsa.graph import Graph, breadth_first_search, load_graph_from_db


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture()
def tmp_db(monkeypatch, tmp_path):
    """
    Isolate database tests with a temporary SQLite database.
    """
    import database as db_module

    test_db = str(tmp_path / "test_graph_aquagrid.db")
    monkeypatch.setattr(db_module, "INSTANCE_DIR", str(tmp_path))
    monkeypatch.setattr(db_module, "DB_PATH", test_db)
    db_module.init_db()
    return db_module


# ═════════════════════════════════════════════════════════════════════════════
#  UNIT TESTS FOR GRAPH REPRESENTATION & BREADTH-FIRST SEARCH
# ═════════════════════════════════════════════════════════════════════════════

class TestGraphEngine:
    """Test pure in-memory Graph construction and BFS algorithms."""

    def test_simple_connected_graph(self):
        """
        A simple connected graph with two nodes and one active edge:
        (Node 1) <---> (Node 2)
        BFS from Node 1 must return [1, 2].
        BFS from Node 2 must return [2, 1].
        """
        g = Graph(directed=False)
        g.add_node(1, name="Tank A")
        g.add_node(2, name="Junction B")
        g.add_edge(1, 2, status="active")

        assert g.node_count == 2
        assert g.edge_count == 1

        result_from_1 = g.breadth_first_search(1)
        assert result_from_1 == [1, 2]

        result_from_2 = g.breadth_first_search(2)
        assert result_from_2 == [2, 1]

    def test_three_connected_nodes(self):
        """
        A graph with three connected nodes:
        (1) <---> (2) <---> (3)
        BFS from 1 visits 1, then 2, then 3.
        BFS from 2 visits 2, then both neighbors 1 and 3.
        Each reachable node must be visited at most once.
        """
        g = Graph(directed=False)
        g.add_node(1, name="Tank 1")
        g.add_node(2, name="Junction 2")
        g.add_node(3, name="Valve 3")

        g.add_edge(1, 2, status="active")
        g.add_edge(2, 3, status="active")

        # Traversal from node 1
        res_1 = g.breadth_first_search(1)
        assert res_1 == [1, 2, 3]
        assert len(res_1) == len(set(res_1))  # No duplicates

        # Traversal from middle node 2
        res_2 = g.breadth_first_search(2)
        assert res_2[0] == 2
        assert set(res_2) == {1, 2, 3}

        # Traversal from end node 3
        res_3 = g.breadth_first_search(3)
        assert res_3 == [3, 2, 1]

    def test_isolated_node(self):
        """
        An isolated node has no incident pipes.
        It must be present in the graph with an empty adjacency list.
        BFS starting from an isolated node returns only [start_node_id].
        """
        g = Graph(directed=False)
        g.add_node(10, name="Isolated Tank")

        assert g.has_node(10)
        assert g.adjacency_list[10] == []
        assert g.get_neighbors(10) == []

        res = g.breadth_first_search(10)
        assert res == [10]

    def test_isolated_node_alongside_connected_nodes(self):
        """
        An isolated node alongside connected nodes should neither be reached
        by BFS from other nodes, nor reach other nodes itself.
        """
        g = Graph(directed=False)
        g.add_node(1)
        g.add_node(2)
        g.add_node(99, name="Isolated")
        g.add_edge(1, 2, status="active")

        # From connected node 1: node 99 must NOT be in reachable list
        res_1 = g.breadth_first_search(1)
        assert res_1 == [1, 2]
        assert 99 not in res_1

        # From isolated node 99: only 99 is reachable
        res_99 = g.breadth_first_search(99)
        assert res_99 == [99]

    def test_two_disconnected_components(self):
        """
        Component 1: (1) <---> (2)
        Component 2: (3) <---> (4)
        BFS starting in Component 1 must never reach Component 2, and vice versa.
        """
        g = Graph(directed=False)
        g.add_node(1)
        g.add_node(2)
        g.add_node(3)
        g.add_node(4)

        g.add_edge(1, 2, status="active")
        g.add_edge(3, 4, status="active")

        # Component 1
        res_1 = g.breadth_first_search(1)
        assert set(res_1) == {1, 2}
        assert 3 not in res_1
        assert 4 not in res_1

        # Component 2
        res_3 = g.breadth_first_search(3)
        assert set(res_3) == {3, 4}
        assert 1 not in res_3
        assert 2 not in res_3

    def test_inactive_pipe_ignored_for_reachability(self):
        """
        Inactive pipes must be ignored for reachability.
        If node 1 and node 2 are connected ONLY by an inactive pipe,
        node 2 is not reachable from node 1.
        """
        g = Graph(directed=False)
        g.add_node(1)
        g.add_node(2)
        g.add_node(3)

        # Pipe 1-2 is inactive (e.g. shut down for maintenance)
        added_1_2 = g.add_edge(1, 2, pipe_id=101, status="inactive")
        assert not added_1_2
        assert len(g.inactive_pipes) == 1

        # Pipe 1-3 is active
        added_1_3 = g.add_edge(1, 3, pipe_id=102, status="active")
        assert added_1_3
        assert len(g.edges) == 1

        # Node 2 is not in 1's adjacency list
        assert 2 not in g.adjacency_list[1]
        assert 3 in g.adjacency_list[1]

        # BFS from 1 reaches 1 and 3, but NOT 2
        res = g.breadth_first_search(1)
        assert res == [1, 3]
        assert 2 not in res

        # BFS from 2 only reaches 2 (it is effectively isolated from reachability)
        res_2 = g.breadth_first_search(2)
        assert res_2 == [2]

    def test_empty_graph(self):
        """
        An empty graph has 0 nodes and 0 edges.
        BFS on any start ID must return an empty list [] gracefully.
        """
        g = Graph()
        assert g.node_count == 0
        assert g.edge_count == 0

        res = g.breadth_first_search(1)
        assert res == []

        res_none = g.breadth_first_search(999)
        assert res_none == []

    def test_invalid_starting_node(self):
        """
        If the starting node ID is missing from the graph, BFS must return [].
        """
        g = Graph()
        g.add_node(1)
        g.add_node(2)
        g.add_edge(1, 2, status="active")

        assert g.breadth_first_search(999) == []
        assert g.breadth_first_search(-1) == []
        assert g.breadth_first_search(0) == []

    def test_directed_graph_reachability(self):
        """
        When directed=True:
        Pipe 1 -> 2: 1 can reach 2, but 2 cannot reach 1.
        """
        g = Graph(directed=True)
        g.add_node(1)
        g.add_node(2)
        g.add_edge(1, 2, status="active")

        # 1 reaches 2
        assert g.breadth_first_search(1) == [1, 2]
        # 2 has no outgoing edges, only reaches itself
        assert g.breadth_first_search(2) == [2]

    def test_cycle_graph_visits_each_node_once(self):
        """
        In a cycle graph (1-2, 2-3, 3-1), BFS must visit every node exactly once
        without infinite loops.
        """
        g = Graph(directed=False)
        g.add_edge(1, 2, status="active")
        g.add_edge(2, 3, status="active")
        g.add_edge(3, 1, status="active")

        res = g.breadth_first_search(1)
        assert set(res) == {1, 2, 3}
        assert len(res) == 3

    def test_module_level_bfs_function(self):
        """Test module-level breadth_first_search(graph, start_id)."""
        g = Graph(directed=False)
        g.add_node(1)
        g.add_node(2)
        g.add_edge(1, 2, status="active")

        res = breadth_first_search(g, 1)
        assert res == [1, 2]

    def test_bfs_detailed_diagnostics(self):
        """Test bfs_detailed provides step order, depth, and connections used."""
        g = Graph(directed=False)
        g.add_node(1, name="Tank A", node_type="tank")
        g.add_node(2, name="Junction B", node_type="junction")
        g.add_node(3, name="Valve C", node_type="valve")
        g.add_edge(1, 2, pipe_id=10, name="Pipe-10", status="active")
        g.add_edge(2, 3, pipe_id=20, name="Pipe-20", status="active")

        detailed = g.bfs_detailed(1)
        assert detailed["found"] is True
        assert detailed["reachable_node_ids"] == [1, 2, 3]
        assert detailed["total_reachable"] == 3
        assert detailed["is_isolated"] is False
        assert len(detailed["connections_used"]) == 2

        # Check depth
        assert detailed["nodes"][0]["depth"] == 0
        assert detailed["nodes"][1]["depth"] == 1
        assert detailed["nodes"][2]["depth"] == 2


# ═════════════════════════════════════════════════════════════════════════════
#  INTEGRATION TESTS: GRAPH LOADED FROM SQLITE DATABASE
# ═════════════════════════════════════════════════════════════════════════════

class TestGraphDatabaseIntegration:
    """Test loading Graph instances from SQLite database schema."""

    def test_graph_loaded_from_database(self, tmp_db):
        """
        Create actual nodes and pipes in SQLite database and load the graph:
        - 3 active connected nodes (Tank Alpha -> Junction Beta -> Valve Gamma)
        - 1 isolated node (Junction Delta with no pipes)
        - 1 inactive pipe (Junction Beta -> Valve Epsilon)
        - 1 node attached only by an inactive pipe (Valve Epsilon)
        """
        # Create nodes
        id_tank = tmp_db.create_node("Tank Alpha", "tank", "Zone 1", "active")
        id_junc = tmp_db.create_node("Junction Beta", "junction", "Zone 2", "active")
        id_valve = tmp_db.create_node("Valve Gamma", "valve", "Zone 3", "active")
        id_isolated = tmp_db.create_node("Junction Delta", "junction", "Zone 4", "active")
        id_inactive_target = tmp_db.create_node("Valve Epsilon", "valve", "Zone 5", "active")

        # Create active pipes
        pipe_1_id = tmp_db.create_pipe(
            "Main-Trunk", id_tank, id_junc, "150.0", "500.0", "1.5", "active"
        )
        pipe_2_id = tmp_db.create_pipe(
            "Branch-Line", id_junc, id_valve, "80.0", "250.0", "2.0", "active"
        )

        # Create inactive pipe connecting Junction Beta to Valve Epsilon
        pipe_inactive_id = tmp_db.create_pipe(
            "Closed-Pipe", id_junc, id_inactive_target, "100.0", "300.0", "5.0", "inactive"
        )

        # Load graph from the temporary database
        g = load_graph_from_db(db_module=tmp_db, directed=False)

        # Verification 1: All 5 nodes exist as vertices in the graph
        assert g.node_count == 5
        assert g.has_node(id_tank)
        assert g.has_node(id_junc)
        assert g.has_node(id_valve)
        assert g.has_node(id_isolated)
        assert g.has_node(id_inactive_target)

        # Verification 2: Only 2 active pipes are loaded into active edges
        assert g.edge_count == 2
        active_pipe_ids = {e["pipe_id"] for e in g.edges}
        assert active_pipe_ids == {pipe_1_id, pipe_2_id}

        # Verification 3: Inactive pipe was ignored for reachability
        assert pipe_inactive_id not in active_pipe_ids
        assert len(g.inactive_pipes) == 1
        assert g.inactive_pipes[0]["pipe_id"] == pipe_inactive_id

        # Verification 4: BFS starting from Tank Alpha
        reachable_from_tank = g.breadth_first_search(id_tank)
        # Should reach Tank, Junction, and Valve
        assert reachable_from_tank[0] == id_tank
        assert set(reachable_from_tank) == {id_tank, id_junc, id_valve}
        # Neither isolated node nor inactive target should be reachable
        assert id_isolated not in reachable_from_tank
        assert id_inactive_target not in reachable_from_tank

        # Verification 5: BFS from isolated node
        reachable_from_isolated = g.breadth_first_search(id_isolated)
        assert reachable_from_isolated == [id_isolated]

        # Verification 6: BFS from node connected only by inactive pipe
        reachable_from_inactive_target = g.breadth_first_search(id_inactive_target)
        assert reachable_from_inactive_target == [id_inactive_target]

        # Verification 7: BFS with invalid ID returns []
        assert g.breadth_first_search(99999) == []

    def test_empty_database_graph(self, tmp_db):
        """When database has no nodes, graph loads cleanly with 0 nodes."""
        g = load_graph_from_db(db_module=tmp_db)
        assert g.node_count == 0
        assert g.edge_count == 0
        assert g.breadth_first_search(1) == []

