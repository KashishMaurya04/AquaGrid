"""
tests/test_distribution_engine.py – Test suite for Network-Aware Water Distribution Optimization.

Validates that:
1. Core mathematical calculations:
   - Q_required = V_allocated / T_duration
   - V_max = C_path * T_duration
   - V_deliverable = min(V_allocated, V_max)
   - V_undelivered = V_allocated - V_deliverable
2. Two routing strategies:
   - SHORTEST_PATH (Dijkstra)
   - WIDEST_PATH (Maximum Bottleneck Path)
3. The benchmark scenario:
   - Path A: cost = 2.0, bottleneck = 50 L/min
   - Path B: cost = 5.0, bottleneck = 200 L/min
   - Allocation requires 100 L/min:
     * Shortest Path -> CAPACITY_CONSTRAINED (50% delivery)
     * Widest Path -> FEASIBLE (100% delivery)
4. Five Feasibility Statuses:
   - FEASIBLE, CAPACITY_CONSTRAINED, UNREACHABLE, UNMAPPED, ZERO_ALLOCATION
5. Shared-Pipe Contention Analysis:
   - Identifies aggregate flow overload on pipes shared across multiple zones.
6. Edge cases and validations:
   - Zero allocation, unmapped destination, co-located source/destination,
     empty network, invalid duration, negative allocation, non-finite values.
"""

import math
import pytest
from dsa.graph import Graph
from distribution.engine import (
    DistributionPlanResult,
    ZoneDeliveryInput,
    ZoneDeliveryResult,
    assess_distribution,
    STATUS_FEASIBLE,
    STATUS_CAPACITY_CONSTRAINED,
    STATUS_UNREACHABLE,
    STATUS_UNMAPPED,
    STATUS_ZERO_ALLOCATION,
    STRATEGY_SHORTEST_PATH,
    STRATEGY_WIDEST_PATH,
)


@pytest.fixture
def dual_route_network() -> Graph:
    """
    Constructs the canonical dual-route benchmark network:
        Source: Node 1 (Central Reservoir)
        Intermediate A: Node 2
        Intermediate B: Node 3
        Target: Node 4 (South Sector Zone)

        Path A: 1 -> 2 -> 4
            Pipe 101: capacity 50 L/min, cost 1.0
            Pipe 102: capacity 100 L/min, cost 1.0
            Total Cost = 2.0, Bottleneck = 50 L/min

        Path B: 1 -> 3 -> 4
            Pipe 201: capacity 200 L/min, cost 2.5
            Pipe 202: capacity 300 L/min, cost 2.5
            Total Cost = 5.0, Bottleneck = 200 L/min
    """
    g = Graph(directed=False)
    g.add_node(1, name="Central Reservoir", node_type="tank", status="active")
    g.add_node(2, name="Junction Alpha", node_type="junction", status="active")
    g.add_node(3, name="Junction Beta", node_type="junction", status="active")
    g.add_node(4, name="South Sector Node", node_type="junction", status="active")

    g.add_edge(1, 2, pipe_id=101, name="Pipe A-1", capacity_lpm=50.0, cost=1.0, status="active")
    g.add_edge(2, 4, pipe_id=102, name="Pipe A-2", capacity_lpm=100.0, cost=1.0, status="active")

    g.add_edge(1, 3, pipe_id=201, name="Pipe B-1", capacity_lpm=200.0, cost=2.5, status="active")
    g.add_edge(3, 4, pipe_id=202, name="Pipe B-2", capacity_lpm=300.0, cost=2.5, status="active")
    return g


class TestDistributionStrategyComparison:
    """Validates the core trade-off between Shortest-Path and Widest-Path routing."""

    def test_shortest_path_constrained_vs_widest_path_feasible(self, dual_route_network):
        """
        Required Flow = 6,000 Litres in 60 minutes = 100 L/min.
        - Shortest Path selects Path A (Cost 2.0, Bottleneck 50 L/min):
          Can only deliver 50 * 60 = 3,000 L -> CAPACITY_CONSTRAINED (50%).
        - Widest Path selects Path B (Cost 5.0, Bottleneck 200 L/min):
          Can deliver all 6,000 L -> FEASIBLE (100%).
        """
        zones = [
            ZoneDeliveryInput(
                zone_id=1,
                zone_name="South Sector District",
                destination_node_id=4,
                allocated_volume_l=6000.0,
            )
        ]

        # 1. Evaluate with SHORTEST_PATH
        plan_sp = assess_distribution(
            graph=dual_route_network,
            source_node_id=1,
            zones=zones,
            duration_minutes=60.0,
            routing_strategy=STRATEGY_SHORTEST_PATH,
        )

        assert plan_sp.routing_strategy == STRATEGY_SHORTEST_PATH
        assert plan_sp.total_allocated_l == 6000.0
        assert plan_sp.total_deliverable_l == 3000.0
        assert plan_sp.total_undelivered_l == 3000.0
        assert plan_sp.overall_delivery_efficiency_pct == 50.0

        z_sp = plan_sp.zone_results[0]
        assert z_sp.feasibility_status == STATUS_CAPACITY_CONSTRAINED
        assert z_sp.route_node_ids == [1, 2, 4]
        assert z_sp.route_pipe_ids == [101, 102]
        assert z_sp.route_cost == 2.0
        assert z_sp.bottleneck_pipe_id == 101
        assert z_sp.bottleneck_capacity_lpm == 50.0
        assert z_sp.required_rate_lpm == 100.0
        assert z_sp.maximum_deliverable_l == 3000.0
        assert z_sp.deliverable_volume_l == 3000.0
        assert z_sp.undelivered_volume_l == 3000.0
        assert z_sp.delivery_efficiency_pct == 50.0

        # 2. Evaluate with WIDEST_PATH
        plan_wp = assess_distribution(
            graph=dual_route_network,
            source_node_id=1,
            zones=zones,
            duration_minutes=60.0,
            routing_strategy=STRATEGY_WIDEST_PATH,
        )

        assert plan_wp.routing_strategy == STRATEGY_WIDEST_PATH
        assert plan_wp.total_allocated_l == 6000.0
        assert plan_wp.total_deliverable_l == 6000.0
        assert plan_wp.total_undelivered_l == 0.0
        assert plan_wp.overall_delivery_efficiency_pct == 100.0

        z_wp = plan_wp.zone_results[0]
        assert z_wp.feasibility_status == STATUS_FEASIBLE
        assert z_wp.route_node_ids == [1, 3, 4]
        assert z_wp.route_pipe_ids == [201, 202]
        assert z_wp.route_cost == 5.0
        assert z_wp.bottleneck_pipe_id == 201
        assert z_wp.bottleneck_capacity_lpm == 200.0
        assert z_wp.required_rate_lpm == 100.0
        assert z_wp.maximum_deliverable_l == 12000.0
        assert z_wp.deliverable_volume_l == 6000.0
        assert z_wp.undelivered_volume_l == 0.0
        assert z_wp.delivery_efficiency_pct == 100.0


class TestFeasibilityStatuses:
    """Validates all five feasibility classifications."""

    def test_zero_allocation_status(self, dual_route_network):
        """A zone with 0.0 L allocated receives ZERO_ALLOCATION status."""
        zones = [
            ZoneDeliveryInput(
                zone_id=10,
                zone_name="Zero Demand Zone",
                destination_node_id=4,
                allocated_volume_l=0.0,
            )
        ]
        plan = assess_distribution(dual_route_network, 1, zones, duration_minutes=60.0)
        res = plan.zone_results[0]
        assert res.feasibility_status == STATUS_ZERO_ALLOCATION
        assert res.required_rate_lpm == 0.0
        assert res.deliverable_volume_l == 0.0
        assert res.undelivered_volume_l == 0.0
        assert res.delivery_efficiency_pct == 100.0
        assert "0.0 L allocated" in res.explanation

    def test_unmapped_zone_status(self, dual_route_network):
        """A zone with destination_node_id=None receives UNMAPPED status."""
        zones = [
            ZoneDeliveryInput(
                zone_id=11,
                zone_name="Unmapped Rural Zone",
                destination_node_id=None,
                allocated_volume_l=2400.0,
            )
        ]
        plan = assess_distribution(dual_route_network, 1, zones, duration_minutes=60.0)
        res = plan.zone_results[0]
        assert res.feasibility_status == STATUS_UNMAPPED
        assert res.required_rate_lpm == 40.0
        assert res.deliverable_volume_l == 0.0
        assert res.undelivered_volume_l == 2400.0
        assert res.delivery_efficiency_pct == 0.0
        assert "not mapped" in res.explanation
        assert plan.has_unmapped_zones is True

    def test_co_located_source_destination_status(self, dual_route_network):
        """A zone co-located at the reservoir (dest == source) is 100% FEASIBLE without pipe constraints."""
        zones = [
            ZoneDeliveryInput(
                zone_id=12,
                zone_name="Reservoir Gate Tap",
                destination_node_id=1,  # Co-located with source!
                allocated_volume_l=5000.0,
            )
        ]
        plan = assess_distribution(dual_route_network, 1, zones, duration_minutes=60.0)
        res = plan.zone_results[0]
        assert res.feasibility_status == STATUS_FEASIBLE
        assert res.route_node_ids == [1]
        assert res.route_pipe_ids == []
        assert res.route_cost == 0.0
        assert res.bottleneck_pipe_id is None
        assert res.deliverable_volume_l == 5000.0
        assert res.undelivered_volume_l == 0.0
        assert res.delivery_efficiency_pct == 100.0
        assert "co-located" in res.explanation.lower()

    def test_unreachable_zone_status(self):
        """A zone located on an isolated, disconnected component receives UNREACHABLE status."""
        g = Graph(directed=False)
        g.add_node(1, status="active")  # Source
        g.add_node(2, status="active")
        g.add_node(99, status="active")  # Disconnected destination
        g.add_edge(1, 2, capacity_lpm=500.0, cost=1.0, status="active")

        zones = [
            ZoneDeliveryInput(
                zone_id=13,
                zone_name="Isolated Island Zone",
                destination_node_id=99,
                allocated_volume_l=1500.0,
            )
        ]
        plan = assess_distribution(g, 1, zones, duration_minutes=30.0)
        res = plan.zone_results[0]
        assert res.feasibility_status == STATUS_UNREACHABLE
        assert res.deliverable_volume_l == 0.0
        assert res.undelivered_volume_l == 1500.0
        assert res.delivery_efficiency_pct == 0.0
        assert "unreachable" in res.explanation.lower()
        assert plan.has_unreachable_zones is True


class TestSharedPipeContentionAnalysis:
    """Validates the diagnostic for concurrent delivery over shared conduits."""

    def test_shared_pipe_contention_detection(self):
        """
        Two zones share Trunk Pipe #1:
            Source (Node 1) -> Trunk (Node 2) via Pipe 1 (Capacity 100 L/min)
            From Node 2:
                -> Zone A (Node 3) via Pipe 2 (Capacity 200 L/min)
                -> Zone B (Node 4) via Pipe 3 (Capacity 200 L/min)

        Zone A: 4,200 L in 60 min -> 70 L/min required (individually < 100 L/min).
        Zone B: 3,000 L in 60 min -> 50 L/min required (individually < 100 L/min).

        Aggregate flow on Pipe 1 = 70 + 50 = 120 L/min > 100 L/min!
        Must trigger a shared-pipe contention warning!
        """
        g = Graph(directed=False)
        for i in range(1, 5):
            g.add_node(i, status="active")

        g.add_edge(1, 2, pipe_id=1, name="Main-Trunk-Line", capacity_lpm=100.0, cost=1.0, status="active")
        g.add_edge(2, 3, pipe_id=2, name="Branch-North", capacity_lpm=200.0, cost=1.0, status="active")
        g.add_edge(2, 4, pipe_id=3, name="Branch-South", capacity_lpm=200.0, cost=1.0, status="active")

        zones = [
            ZoneDeliveryInput(zone_id=101, zone_name="North District", destination_node_id=3, allocated_volume_l=4200.0),
            ZoneDeliveryInput(zone_id=102, zone_name="South District", destination_node_id=4, allocated_volume_l=3000.0),
        ]

        plan = assess_distribution(g, 1, zones, duration_minutes=60.0)

        # Both individual zone paths are assessed as FEASIBLE individually
        assert plan.zone_results[0].feasibility_status == STATUS_FEASIBLE
        assert plan.zone_results[1].feasibility_status == STATUS_FEASIBLE

        # Contention warning MUST be created for shared Pipe 1
        assert len(plan.shared_pipe_warnings) == 1
        warn = plan.shared_pipe_warnings[0]
        assert warn["pipe_id"] == 1
        assert warn["pipe_name"] == "Main-Trunk-Line"
        assert warn["pipe_capacity_lpm"] == 100.0
        assert warn["aggregate_required_lpm"] == pytest.approx(120.0)
        assert warn["excess_demand_lpm"] == pytest.approx(20.0)
        assert set(warn["affected_zone_ids"]) == {101, 102}
        assert "Contention detected" in warn["warning_message"]
        assert "Notice: 1 shared pipe(s)" in plan.notes

    def test_no_contention_when_within_capacity(self):
        """When aggregate flow is within capacity, no warning is generated."""
        g = Graph(directed=False)
        for i in range(1, 4):
            g.add_node(i, status="active")

        g.add_edge(1, 2, pipe_id=1, name="Big-Trunk", capacity_lpm=500.0, cost=1.0, status="active")
        g.add_edge(2, 3, pipe_id=2, name="Branch", capacity_lpm=500.0, cost=1.0, status="active")

        zones = [
            ZoneDeliveryInput(zone_id=1, zone_name="Zone A", destination_node_id=3, allocated_volume_l=3000.0),
            ZoneDeliveryInput(zone_id=2, zone_name="Zone B", destination_node_id=3, allocated_volume_l=3000.0),
        ]
        # Aggregate flow = 50 + 50 = 100 L/min < 500 L/min
        plan = assess_distribution(g, 1, zones, duration_minutes=60.0)
        assert len(plan.shared_pipe_warnings) == 0


class TestEdgeCasesAndValidation:
    """Robust error checking and boundary protection."""

    def test_invalid_duration_rejected(self, dual_route_network):
        zones = [ZoneDeliveryInput(1, "Z", 4, 100.0)]
        with pytest.raises(ValueError, match="strictly positive"):
            assess_distribution(dual_route_network, 1, zones, duration_minutes=0.0)

        with pytest.raises(ValueError, match="strictly positive"):
            assess_distribution(dual_route_network, 1, zones, duration_minutes=-15.0)

        with pytest.raises(ValueError, match="finite"):
            assess_distribution(dual_route_network, 1, zones, duration_minutes=float("nan"))

    def test_negative_allocated_volume_rejected(self):
        with pytest.raises(ValueError, match="cannot be negative"):
            ZoneDeliveryInput(zone_id=1, zone_name="Bad Zone", destination_node_id=2, allocated_volume_l=-100.0)

    def test_non_finite_allocated_volume_rejected(self):
        with pytest.raises(ValueError, match="finite"):
            ZoneDeliveryInput(zone_id=1, zone_name="Bad Zone", destination_node_id=2, allocated_volume_l=float("inf"))

    def test_unknown_routing_strategy_rejected(self, dual_route_network):
        zones = [ZoneDeliveryInput(1, "Z", 4, 100.0)]
        with pytest.raises(ValueError, match="Unknown routing strategy"):
            assess_distribution(dual_route_network, 1, zones, routing_strategy="QUANTUM_TUNNELING")

    def test_missing_source_node_rejected(self, dual_route_network):
        zones = [ZoneDeliveryInput(1, "Z", 4, 100.0)]
        with pytest.raises(ValueError, match="does not exist"):
            assess_distribution(dual_route_network, source_node_id=999, zones=zones)

    def test_inactive_source_node_rejected(self):
        g = Graph()
        g.add_node(1, status="inactive")
        g.add_node(2, status="active")
        zones = [ZoneDeliveryInput(1, "Z", 2, 100.0)]
        with pytest.raises(ValueError, match="is inactive"):
            assess_distribution(g, source_node_id=1, zones=zones)

    def test_empty_graph_rejected(self):
        g = Graph()
        zones = [ZoneDeliveryInput(1, "Z", 2, 100.0)]
        with pytest.raises(ValueError, match="graph is empty"):
            assess_distribution(g, source_node_id=1, zones=zones)

    def test_empty_zones_returns_zero_plan(self, dual_route_network):
        plan = assess_distribution(dual_route_network, 1, zones=[])
        assert plan.total_allocated_l == 0.0
        assert plan.total_deliverable_l == 0.0
        assert plan.overall_delivery_efficiency_pct == 100.0
        assert plan.zone_results == []
        assert plan.shared_pipe_warnings == []

    def test_coercion_from_dict_inputs(self, dual_route_network):
        """Accepts dict inputs transparently matching database and JSON payloads."""
        raw_zones = [
            {"zone_id": 1, "zone_name": "Dict Zone", "destination_node_id": 4, "allocated_volume_l": 1200.0}
        ]
        plan = assess_distribution(dual_route_network, 1, raw_zones, duration_minutes=30.0)
        assert len(plan.zone_results) == 1
        assert plan.zone_results[0].zone_name == "Dict Zone"
        assert plan.zone_results[0].allocated_volume_l == 1200.0

    def test_to_dict_serialization(self, dual_route_network):
        """Ensures complete JSON-serializable to_dict implementation for UI and APIs."""
        zones = [ZoneDeliveryInput(1, "Z", 4, 600.0)]
        plan = assess_distribution(dual_route_network, 1, zones, duration_minutes=60.0)
        d = plan.to_dict()
        assert isinstance(d, dict)
        assert d["source_node_id"] == 1
        assert isinstance(d["zone_results"], list)
        assert d["zone_results"][0]["zone_name"] == "Z"

