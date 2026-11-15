"""
tests/test_widest_path.py – Test suite for Maximum Bottleneck Path (Widest Path) algorithm.

Validates that:
1. Widest path maximizes the minimum pipe capacity along a route.
2. Shortest path (Dijkstra) vs Widest Path trade-offs are accurately demonstrated:
   - Path A: Cost 2.0, Bottleneck 50 L/min
   - Path B: Cost 5.0, Bottleneck 200 L/min
   Dijkstra selects Path A, while Widest Path selects Path B.
3. Deterministic tie-breaking prefers lower routing cost, then fewer hops.
4. Parallel pipes between the same endpoints are evaluated individually.
5. Inactive pipes and inactive endpoint nodes are strictly ignored.
6. Edge cases: empty network, missing endpoints, source == destination, unreachable nodes,
   and invalid capacities/costs.
"""

import math
import pytest
from dsa.graph import Graph
from dsa.dijkstra import dijkstra
from dsa.widest_path import widest_path, find_widest_path, WidestPathResult


class TestWidestPathBasics:
    """Basic routing and path reconstruction validation."""

    def test_single_pipe_route(self):
        """A simple route with one active pipe."""
        g = Graph(directed=False)
        g.add_node(1, name="Tank 1", status="active")
        g.add_node(2, name="Junction 2", status="active")
        g.add_edge(1, 2, pipe_id=10, name="Pipe-10", capacity_lpm=500.0, cost=3.0, status="active")

        res = widest_path(g, 1, 2)
        assert res.found is True
        assert res.bottleneck_capacity == 500.0
        assert res.total_cost == 3.0
        assert res.node_path == [1, 2]
        assert res.pipe_ids == [10]
        assert res.pipe_names == ["Pipe-10"]
        assert len(res.pipe_path) == 1

    def test_multi_pipe_linear_route(self):
        """Bottleneck capacity along a multi-hop path is the minimum of edge capacities."""
        g = Graph(directed=False)
        for i in range(1, 5):
            g.add_node(i, status="active")

        g.add_edge(1, 2, pipe_id=1, capacity_lpm=1000.0, cost=1.0, status="active")
        g.add_edge(2, 3, pipe_id=2, capacity_lpm=250.0, cost=1.0, status="active")  # Bottleneck!
        g.add_edge(3, 4, pipe_id=3, capacity_lpm=800.0, cost=1.0, status="active")

        res = widest_path(g, 1, 4)
        assert res.found is True
        assert res.bottleneck_capacity == 250.0
        assert res.total_cost == 3.0
        assert res.node_path == [1, 2, 3, 4]
        assert res.pipe_ids == [1, 2, 3]

    def test_alias_find_widest_path(self):
        """`find_widest_path` is an exact alias of `widest_path`."""
        g = Graph(directed=False)
        g.add_node(1, status="active")
        g.add_node(2, status="active")
        g.add_edge(1, 2, pipe_id=1, capacity_lpm=300.0, cost=1.0, status="active")

        r1 = widest_path(g, 1, 2)
        r2 = find_widest_path(g, 1, 2)
        assert r1.node_path == r2.node_path
        assert r1.bottleneck_capacity == r2.bottleneck_capacity

    def test_graph_method_widest_path(self):
        """Graph instance method `graph.widest_path(source, dest)` operates identically."""
        g = Graph(directed=False)
        g.add_node(1, status="active")
        g.add_node(2, status="active")
        g.add_edge(1, 2, pipe_id=1, capacity_lpm=450.0, cost=2.0, status="active")

        res = g.widest_path(1, 2)
        assert res.found is True
        assert res.bottleneck_capacity == 450.0


class TestWidestPathVsDijkstraComparison:
    """
    Core algorithmic requirement:
    Path A: Total cost = 2.0, Bottleneck capacity = 50 L/min
    Path B: Total cost = 5.0, Bottleneck capacity = 200 L/min
    """

    @pytest.fixture
    def comparison_network(self) -> Graph:
        g = Graph(directed=False)
        g.add_node(1, name="Source Tank", status="active")
        g.add_node(2, name="Inter A", status="active")
        g.add_node(3, name="Inter B", status="active")
        g.add_node(4, name="Target Zone", status="active")

        # Path A: 1 -> 2 -> 4 (Cost = 1.0 + 1.0 = 2.0, Bottleneck = min(50, 100) = 50 L/min)
        g.add_edge(1, 2, pipe_id=101, name="Pipe A1", capacity_lpm=50.0, cost=1.0, status="active")
        g.add_edge(2, 4, pipe_id=102, name="Pipe A2", capacity_lpm=100.0, cost=1.0, status="active")

        # Path B: 1 -> 3 -> 4 (Cost = 2.5 + 2.5 = 5.0, Bottleneck = min(200, 300) = 200 L/min)
        g.add_edge(1, 3, pipe_id=201, name="Pipe B1", capacity_lpm=200.0, cost=2.5, status="active")
        g.add_edge(3, 4, pipe_id=202, name="Pipe B2", capacity_lpm=300.0, cost=2.5, status="active")

        return g

    def test_dijkstra_selects_path_a_for_minimum_cost(self, comparison_network):
        """Dijkstra strictly minimizes routing cost and must select Path A."""
        res = dijkstra(comparison_network, 1, 4)
        assert res.found is True
        assert res.total_cost == pytest.approx(2.0)
        assert res.node_path == [1, 2, 4]
        assert res.pipe_ids == [101, 102]

        # Verify that Path A has a bottleneck of only 50 L/min
        bottleneck = min(p["capacity_lpm"] for p in res.pipe_path)
        assert bottleneck == 50.0

    def test_widest_path_selects_path_b_for_maximum_capacity(self, comparison_network):
        """Widest Path strictly maximizes bottleneck capacity and must select Path B."""
        res = widest_path(comparison_network, 1, 4)
        assert res.found is True
        assert res.bottleneck_capacity == 200.0
        assert res.total_cost == pytest.approx(5.0)
        assert res.node_path == [1, 3, 4]
        assert res.pipe_ids == [201, 202]


class TestTieBreakingAndParallelPipes:
    """Tie-breaking and parallel edge behaviors."""

    def test_tie_breaking_by_routing_cost(self):
        """When two paths have equal bottleneck capacity, choose lower routing cost."""
        g = Graph(directed=False)
        g.add_node(1, status="active")
        g.add_node(2, status="active")
        g.add_node(3, status="active")
        g.add_node(4, status="active")

        # Path 1: 1 -> 2 -> 4, Bottleneck 300, Cost 10.0
        g.add_edge(1, 2, pipe_id=1, capacity_lpm=300.0, cost=5.0, status="active")
        g.add_edge(2, 4, pipe_id=2, capacity_lpm=500.0, cost=5.0, status="active")

        # Path 2: 1 -> 3 -> 4, Bottleneck 300, Cost 3.0 (Better!)
        g.add_edge(1, 3, pipe_id=3, capacity_lpm=300.0, cost=1.5, status="active")
        g.add_edge(3, 4, pipe_id=4, capacity_lpm=600.0, cost=1.5, status="active")

        res = widest_path(g, 1, 4)
        assert res.found is True
        assert res.bottleneck_capacity == 300.0
        assert res.total_cost == pytest.approx(3.0)
        assert res.node_path == [1, 3, 4]
        assert res.pipe_ids == [3, 4]

    def test_tie_breaking_by_hop_count(self):
        """When bottleneck and cost are identical, choose fewer pipe hops."""
        g = Graph(directed=False)
        g.add_node(1, status="active")
        g.add_node(2, status="active")
        g.add_node(3, status="active")
        g.add_node(4, status="active")

        # Direct 1-hop path: 1 -> 4 (Bottleneck 500, Cost 4.0, 1 hop)
        g.add_edge(1, 4, pipe_id=10, capacity_lpm=500.0, cost=4.0, status="active")

        # Indirect 2-hop path: 1 -> 2 -> 4 (Bottleneck 500, Cost 4.0, 2 hops)
        g.add_edge(1, 2, pipe_id=11, capacity_lpm=500.0, cost=2.0, status="active")
        g.add_edge(2, 4, pipe_id=12, capacity_lpm=500.0, cost=2.0, status="active")

        res = widest_path(g, 1, 4)
        assert res.found is True
        assert res.bottleneck_capacity == 500.0
        assert res.total_cost == 4.0
        assert res.node_path == [1, 4]
        assert res.pipe_ids == [10]

    def test_parallel_pipes_prefers_higher_capacity(self):
        """Between the same two nodes, evaluate parallel pipes and choose the wider conduit."""
        g = Graph(directed=False)
        g.add_node(1, status="active")
        g.add_node(2, status="active")

        # Parallel pipes:
        # Pipe 1: Capacity 150, Cost 1.0
        # Pipe 2: Capacity 600, Cost 4.0
        g.add_edge(1, 2, pipe_id=1, name="Narrow-Cheap", capacity_lpm=150.0, cost=1.0, status="active")
        g.add_edge(1, 2, pipe_id=2, name="Wide-Trunk", capacity_lpm=600.0, cost=4.0, status="active")

        res = widest_path(g, 1, 2)
        assert res.found is True
        assert res.bottleneck_capacity == 600.0
        assert res.total_cost == 4.0
        assert res.pipe_ids == [2]
        assert res.pipe_names == ["Wide-Trunk"]


class TestInactivePipesAndNodes:
    """Inactive infrastructure filtering."""

    def test_ignores_inactive_high_capacity_pipe(self):
        """A pipe with massive capacity must be bypassed if it is inactive."""
        g = Graph(directed=False)
        g.add_node(1, status="active")
        g.add_node(2, status="active")
        g.add_node(3, status="active")

        # Inactive high-capacity conduit:
        g.add_edge(1, 3, pipe_id=1, capacity_lpm=10000.0, cost=1.0, status="inactive")

        # Active path through intermediate junction:
        g.add_edge(1, 2, pipe_id=2, capacity_lpm=350.0, cost=2.0, status="active")
        g.add_edge(2, 3, pipe_id=3, capacity_lpm=400.0, cost=2.0, status="active")

        res = widest_path(g, 1, 3)
        assert res.found is True
        assert res.bottleneck_capacity == 350.0
        assert res.node_path == [1, 2, 3]
        assert 1 not in res.pipe_ids

    def test_ignores_inactive_intermediate_node(self):
        """Cannot route through an inactive intermediate vertex."""
        g = Graph(directed=False)
        g.add_node(1, status="active")
        g.add_node(2, status="inactive")  # Inactive node!
        g.add_node(3, status="active")
        g.add_node(4, status="active")

        # Path via inactive node 2 has high capacity:
        g.add_edge(1, 2, pipe_id=1, capacity_lpm=1000.0, cost=1.0, status="active")
        g.add_edge(2, 4, pipe_id=2, capacity_lpm=1000.0, cost=1.0, status="active")

        # Active path via node 3 has lower capacity:
        g.add_edge(1, 3, pipe_id=3, capacity_lpm=220.0, cost=3.0, status="active")
        g.add_edge(3, 4, pipe_id=4, capacity_lpm=250.0, cost=3.0, status="active")

        res = widest_path(g, 1, 4)
        assert res.found is True
        assert res.bottleneck_capacity == 220.0
        assert res.node_path == [1, 3, 4]


class TestEdgeCasesAndValidation:
    """Boundary conditions, disconnections, and invalid values."""

    def test_empty_graph(self):
        g = Graph()
        res = widest_path(g, 1, 2)
        assert res.found is False
        assert "empty" in res.message.lower()

    def test_missing_endpoints(self):
        g = Graph()
        g.add_node(1, status="active")
        res1 = widest_path(g, 99, 1)
        assert res1.found is False
        assert "Source node #99" in res1.message

        res2 = widest_path(g, 1, 88)
        assert res2.found is False
        assert "Destination node #88" in res2.message

    def test_inactive_endpoints(self):
        g = Graph()
        g.add_node(1, status="inactive")
        g.add_node(2, status="active")
        res = widest_path(g, 1, 2)
        assert res.found is False
        assert "inactive" in res.message.lower()

    def test_source_equals_destination(self):
        g = Graph()
        g.add_node(1, status="active")
        res = widest_path(g, 1, 1)
        assert res.found is False
        assert res.bottleneck_capacity is None
        assert res.total_cost is None
        assert res.node_path == []
        assert res.message == "Source and destination must be different."

    def test_unreachable_destination(self):
        g = Graph()
        g.add_node(1, status="active")
        g.add_node(2, status="active")
        g.add_node(3, status="active")
        g.add_edge(1, 2, capacity_lpm=100.0, cost=1.0, status="active")
        # Node 3 is isolated
        res = widest_path(g, 1, 3)
        assert res.found is False
        assert res.bottleneck_capacity is None
        assert "not reachable" in res.message.lower()

    def test_invalid_pipe_capacity_raises_error(self):
        g = Graph()
        g.add_node(1, status="active")
        g.add_node(2, status="active")

        # Zero capacity
        g.add_edge(1, 2, capacity_lpm=0.0, cost=1.0, status="active")
        with pytest.raises(ValueError, match="must be strictly positive"):
            widest_path(g, 1, 2)

    def test_nan_pipe_capacity_raises_error(self):
        g = Graph()
        g.add_node(1, status="active")
        g.add_node(2, status="active")
        g.add_edge(1, 2, capacity_lpm=float("nan"), cost=1.0, status="active")
        with pytest.raises(ValueError, match="strictly positive and finite"):
            widest_path(g, 1, 2)

    def test_negative_cost_raises_error_on_edge_addition(self):
        g = Graph()
        with pytest.raises(ValueError, match="cannot be negative"):
            g.add_edge(1, 2, cost=-5.0, status="active")

    def test_result_dict_access(self):
        g = Graph()
        g.add_node(1, status="active")
        g.add_node(2, status="active")
        g.add_edge(1, 2, pipe_id=1, capacity_lpm=300.0, cost=2.0, status="active")

        res = widest_path(g, 1, 2)
        assert res["found"] is True
        assert res["bottleneck_capacity"] == 300.0
        assert res.get("total_cost") == 2.0
        d = res.to_dict()
        assert isinstance(d, dict)
        assert d["pipe_ids"] == [1]

