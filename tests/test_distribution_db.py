"""
tests/test_distribution_db.py – Comprehensive SQLite persistence and integration tests for Network-Aware Water Distribution.

Validates that:
1. Delivery assessment summaries and zone details persist cleanly and accurately.
2. Full transactional atomicity: rollback occurs when any zone insert or validation fails.
3. Historical snapshot integrity:
   - Preserves source and destination node names, zone names, and pipe details.
   - Deleting a node or pipe sets foreign key to NULL without corrupting the historical assessment.
   - Renaming a zone after assessment does not rewrite historical records.
4. Foreign-key protection:
   - Allocation runs with attached delivery assessments cannot be deleted (ON DELETE RESTRICT).
   - Deleting a delivery assessment cascades to its own zone-detail rows.
5. Integration service:
   - `evaluate_and_record_run_distribution` seamlessly loads an allocation run, assesses it
     across the physical network, and stores the results atomically.
6. Diagnostic snapshots:
   - Shared-pipe contention warnings persist and deserialize accurately.
7. Totals reconciliation:
   - Prevents persistence of corrupt plans where zone sums do not reconcile with totals.
"""

from __future__ import annotations

import json
import math
import os
import sys
import pytest
import sqlite3

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import database as db_module
from allocation.engine import ZoneDemandInput, allocate_water
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
from distribution.service import evaluate_and_record_run_distribution
from dsa.graph import Graph


@pytest.fixture()
def isolated_db(monkeypatch, tmp_path):
    """Isolated SQLite database fixture for distribution persistence tests."""
    test_db = str(tmp_path / "test_distribution.db")
    monkeypatch.setattr(db_module, "INSTANCE_DIR", str(tmp_path))
    monkeypatch.setattr(db_module, "DB_PATH", test_db)
    db_module.init_db()
    return db_module


@pytest.fixture()
def populated_network(isolated_db):
    """
    Sets up a complete network with nodes, pipes, allocation zones, and a recorded allocation run:
        Source: Node 1 (Central Reservoir)
        Intermediates: Node 2 (Junction A), Node 3 (Junction B)
        Destinations: Node 4 (Zone 1 Dest), Node 5 (Zone 2 Dest)
        Pipes connecting 1 -> 2, 2 -> 4, 1 -> 3, 3 -> 5
    """
    db = isolated_db
    # Create Nodes
    src_id = db.create_node("Central Reservoir", "tank", "North Hill", "active")
    junc_a = db.create_node("Junction Alpha", "junction", "Sector A", "active")
    junc_b = db.create_node("Junction Beta", "junction", "Sector B", "active")
    dest_1 = db.create_node("South Gate Node", "valve", "Sector South", "active")
    dest_2 = db.create_node("East Gate Node", "valve", "Sector East", "active")

    # Create Pipes
    p1 = db.create_pipe("Trunk-Alpha", src_id, junc_a, 200.0, 150.0, 1.0, "active")
    p2 = db.create_pipe("Branch-South", junc_a, dest_1, 300.0, 80.0, 2.0, "active")
    p3 = db.create_pipe("Trunk-Beta", src_id, junc_b, 250.0, 300.0, 2.0, "active")
    p4 = db.create_pipe("Branch-East", junc_b, dest_2, 400.0, 250.0, 2.0, "active")

    # Create Allocation Zones
    z1 = db.create_allocation_zone(
        zone_name="South Residential District",
        requested_demand_l=5000.0,
        essential_min_l=2000.0,
        fairness_weight=1.0,
        priority_category="standard",
        status="active",
        node_id=dest_1,
    )
    z2 = db.create_allocation_zone(
        zone_name="East Commercial Park",
        requested_demand_l=6000.0,
        essential_min_l=3000.0,
        fairness_weight=1.5,
        priority_category="commercial",
        status="active",
        node_id=dest_2,
    )

    # Execute and record a Fair Water Allocation run (Day 8 engine)
    active_zones = db.get_active_allocation_zones()
    zone_inputs = [
        ZoneDemandInput(
            zone_id=int(z["zone_id"]),
            zone_name=str(z["zone_name"]),
            requested_demand=float(z["requested_demand_l"]),
            essential_min=float(z["essential_min_l"]),
            fairness_weight=float(z["fairness_weight"]),
            node_id=z["node_id"],
        )
        for z in active_zones
    ]
    alloc_run_res = allocate_water(total_supply=10000.0, zones=zone_inputs)
    run_id = db.record_allocation_run(run_result=alloc_run_res, notes="Initial test allocation run")

    return {
        "db": db,
        "src_id": src_id,
        "junc_a": junc_a,
        "junc_b": junc_b,
        "dest_1": dest_1,
        "dest_2": dest_2,
        "pipes": [p1, p2, p3, p4],
        "zones": [z1, z2],
        "run_id": run_id,
        "alloc_run_res": alloc_run_res,
    }


class TestDeliveryAssessmentPersistence:
    """Tests basic storage and retrieval of delivery assessments."""

    def test_record_distribution_assessment_success(self, populated_network):
        """Successfully record a plan and verify all summary and detail fields."""
        data = populated_network
        db = data["db"]
        run_id = data["run_id"]
        src_id = data["src_id"]

        # Build a valid DistributionPlanResult
        z1 = ZoneDeliveryResult(
            zone_id=data["zones"][0],
            zone_name="South Residential District",
            destination_node_id=data["dest_1"],
            allocated_volume_l=4500.0,
            route_node_ids=[src_id, data["junc_a"], data["dest_1"]],
            route_pipe_ids=[data["pipes"][0], data["pipes"][1]],
            route_cost=3.0,
            bottleneck_pipe_id=data["pipes"][1],
            bottleneck_pipe_name="Branch-South",
            bottleneck_capacity_lpm=80.0,
            required_rate_lpm=75.0,
            maximum_deliverable_l=4800.0,
            deliverable_volume_l=4500.0,
            undelivered_volume_l=0.0,
            delivery_efficiency_pct=100.0,
            feasibility_status=STATUS_FEASIBLE,
            explanation="100% deliverable over active conduits.",
        )
        z2 = ZoneDeliveryResult(
            zone_id=data["zones"][1],
            zone_name="East Commercial Park",
            destination_node_id=data["dest_2"],
            allocated_volume_l=5500.0,
            route_node_ids=[src_id, data["junc_b"], data["dest_2"]],
            route_pipe_ids=[data["pipes"][2], data["pipes"][3]],
            route_cost=4.0,
            bottleneck_pipe_id=data["pipes"][3],
            bottleneck_pipe_name="Branch-East",
            bottleneck_capacity_lpm=250.0,
            required_rate_lpm=91.67,
            maximum_deliverable_l=15000.0,
            deliverable_volume_l=5500.0,
            undelivered_volume_l=0.0,
            delivery_efficiency_pct=100.0,
            feasibility_status=STATUS_FEASIBLE,
            explanation="100% deliverable over high-capacity trunk.",
        )

        plan = DistributionPlanResult(
            source_node_id=src_id,
            source_node_name="Central Reservoir",
            duration_minutes=60.0,
            routing_strategy=STRATEGY_SHORTEST_PATH,
            total_allocated_l=10000.0,
            total_deliverable_l=10000.0,
            total_undelivered_l=0.0,
            overall_delivery_efficiency_pct=100.0,
            zone_results=[z1, z2],
            shared_pipe_warnings=[],
            notes="Nominal network conditions",
        )

        assess_id = db.record_distribution_assessment(plan, run_id=run_id)
        assert assess_id > 0

        # Verify summary row
        row = db.get_delivery_assessment_by_id(assess_id)
        assert row is not None
        assert row["assessment_id"] == assess_id
        assert row["run_id"] == run_id
        assert row["source_node_id"] == src_id
        assert row["source_name_snapshot"] == "Central Reservoir"
        assert row["duration_minutes"] == 60.0
        assert row["routing_strategy"] == STRATEGY_SHORTEST_PATH
        assert row["total_allocated_l"] == 10000.0
        assert row["total_deliverable_l"] == 10000.0
        assert row["total_undelivered_l"] == 0.0
        assert row["delivery_efficiency_pct"] == 100.0
        assert row["has_contention"] == 0
        assert row["contention_warnings_json"] is None
        assert row["notes"] == "Nominal network conditions"

        # Verify zone details
        zone_rows = db.get_delivery_zone_assessments(assess_id)
        assert len(zone_rows) == 2

        zr1 = zone_rows[0]
        assert zr1["zone_id"] == data["zones"][0]
        assert zr1["zone_name_snapshot"] == "South Residential District"
        assert zr1["destination_node_id"] == data["dest_1"]
        assert zr1["destination_name_snapshot"] == "South Gate Node"
        assert zr1["allocated_amount_l"] == 4500.0
        assert zr1["deliverable_amount_l"] == 4500.0
        assert zr1["bottleneck_capacity_lpm"] == 80.0
        assert zr1["bottleneck_pipe_id"] == data["pipes"][1]
        assert zr1["bottleneck_pipe_name_snapshot"] == "Branch-South"
        assert zr1["feasibility_status"] == STATUS_FEASIBLE
        assert json.loads(zr1["route_node_ids"]) == [src_id, data["junc_a"], data["dest_1"]]
        assert json.loads(zr1["route_pipe_ids"]) == [data["pipes"][0], data["pipes"][1]]

    def test_record_contention_warnings_persistence(self, populated_network):
        """Shared pipe contention warnings are persisted in contention_warnings_json."""
        data = populated_network
        db = data["db"]
        run_id = data["run_id"]
        src_id = data["src_id"]

        warning = {
            "pipe_id": 99,
            "pipe_name": "Shared-Main",
            "pipe_capacity_lpm": 100.0,
            "aggregate_required_lpm": 140.0,
            "excess_demand_lpm": 40.0,
            "affected_zone_ids": [1, 2],
            "affected_zone_names": ["Zone 1", "Zone 2"],
            "warning_message": "Flow overload detected",
        }

        plan = DistributionPlanResult(
            source_node_id=src_id,
            source_node_name="Central Reservoir",
            duration_minutes=60.0,
            routing_strategy=STRATEGY_SHORTEST_PATH,
            total_allocated_l=100.0,
            total_deliverable_l=100.0,
            total_undelivered_l=0.0,
            overall_delivery_efficiency_pct=100.0,
            zone_results=[
                ZoneDeliveryResult(
                    zone_id=data["zones"][0],
                    zone_name="South Residential District",
                    destination_node_id=data["dest_1"],
                    allocated_volume_l=100.0,
                    deliverable_volume_l=100.0,
                    undelivered_volume_l=0.0,
                    delivery_efficiency_pct=100.0,
                    feasibility_status=STATUS_FEASIBLE,
                )
            ],
            shared_pipe_warnings=[warning],
            notes="Contention observed on trunk",
        )

        assess_id = db.record_distribution_assessment(plan, run_id=run_id)
        row = db.get_delivery_assessment_by_id(assess_id)
        assert row["has_contention"] == 1
        assert row["contention_warnings_json"] is not None

        deserialized = json.loads(row["contention_warnings_json"])
        assert len(deserialized) == 1
        assert deserialized[0]["pipe_id"] == 99
        assert deserialized[0]["excess_demand_lpm"] == 40.0


class TestTransactionAtomicityAndRollback:
    """Verifies that failures during assessment recording roll back cleanly."""

    def test_atomic_rollback_on_zone_failure(self, populated_network):
        """If a zone insert fails (e.g. simulated failure hook), everything rolls back."""
        data = populated_network
        db = data["db"]
        run_id = data["run_id"]
        src_id = data["src_id"]

        z1 = ZoneDeliveryResult(
            zone_id=data["zones"][0],
            zone_name="Zone 1",
            destination_node_id=data["dest_1"],
            allocated_volume_l=1000.0,
            deliverable_volume_l=1000.0,
            undelivered_volume_l=0.0,
            feasibility_status=STATUS_FEASIBLE,
        )
        z2 = ZoneDeliveryResult(
            zone_id=data["zones"][1],
            zone_name="Zone 2",
            destination_node_id=data["dest_2"],
            allocated_volume_l=2000.0,
            deliverable_volume_l=2000.0,
            undelivered_volume_l=0.0,
            feasibility_status=STATUS_FEASIBLE,
        )

        plan = DistributionPlanResult(
            source_node_id=src_id,
            duration_minutes=30.0,
            routing_strategy=STRATEGY_SHORTEST_PATH,
            total_allocated_l=3000.0,
            total_deliverable_l=3000.0,
            total_undelivered_l=0.0,
            overall_delivery_efficiency_pct=100.0,
            zone_results=[z1, z2],
        )

        initial_assess_count = db.count_delivery_assessments()
        initial_detail_count = db.count_delivery_zone_assessments()

        # Trigger simulated failure on second zone
        with pytest.raises(RuntimeError, match="Simulated transaction failure"):
            db.record_distribution_assessment(
                plan, run_id=run_id, fail_on_zone_id=data["zones"][1]
            )

        # Assert complete rollback: no summary row and no detail rows were left
        assert db.count_delivery_assessments() == initial_assess_count
        assert db.count_delivery_zone_assessments() == initial_detail_count

    def test_reconciliation_mismatch_rejected(self, populated_network):
        """If summary totals do not match the sum of zone results, reject before writing."""
        data = populated_network
        db = data["db"]
        run_id = data["run_id"]
        src_id = data["src_id"]

        plan = DistributionPlanResult(
            source_node_id=src_id,
            duration_minutes=60.0,
            routing_strategy=STRATEGY_SHORTEST_PATH,
            total_allocated_l=5000.0,  # Corrupt total! (Zone has only 1000.0)
            total_deliverable_l=1000.0,
            total_undelivered_l=0.0,
            overall_delivery_efficiency_pct=100.0,
            zone_results=[
                ZoneDeliveryResult(
                    zone_id=data["zones"][0],
                    zone_name="Zone 1",
                    destination_node_id=data["dest_1"],
                    allocated_volume_l=1000.0,
                    deliverable_volume_l=1000.0,
                    undelivered_volume_l=0.0,
                    feasibility_status=STATUS_FEASIBLE,
                )
            ],
        )

        with pytest.raises(ValueError, match="Totals reconciliation error"):
            db.record_distribution_assessment(plan, run_id=run_id)


class TestForeignKeysAndSnapshotPreservation:
    """Verifies foreign key constraints and historical snapshot durability."""

    def test_cannot_delete_allocation_run_with_assessments(self, populated_network):
        """An allocation run referenced by delivery assessments cannot be deleted (RESTRICT)."""
        data = populated_network
        db = data["db"]
        run_id = data["run_id"]
        src_id = data["src_id"]

        plan = DistributionPlanResult(
            source_node_id=src_id,
            duration_minutes=60.0,
            routing_strategy=STRATEGY_SHORTEST_PATH,
            total_allocated_l=0.0,
            total_deliverable_l=0.0,
            total_undelivered_l=0.0,
            overall_delivery_efficiency_pct=100.0,
            zone_results=[],
        )
        db.record_distribution_assessment(plan, run_id=run_id)

        # Attempting to delete the allocation run must raise RuntimeError
        with pytest.raises(RuntimeError, match="delivery assessment records attached"):
            db.delete_allocation_run(run_id)

    def test_delete_delivery_assessment_cascades_to_zone_details(self, populated_network):
        """Deleting an assessment removes its zone details via ON DELETE CASCADE."""
        data = populated_network
        db = data["db"]
        run_id = data["run_id"]
        src_id = data["src_id"]

        plan = DistributionPlanResult(
            source_node_id=src_id,
            duration_minutes=60.0,
            routing_strategy=STRATEGY_SHORTEST_PATH,
            total_allocated_l=100.0,
            total_deliverable_l=100.0,
            total_undelivered_l=0.0,
            overall_delivery_efficiency_pct=100.0,
            zone_results=[
                ZoneDeliveryResult(
                    zone_id=data["zones"][0],
                    zone_name="Zone 1",
                    destination_node_id=data["dest_1"],
                    allocated_volume_l=100.0,
                    deliverable_volume_l=100.0,
                    undelivered_volume_l=0.0,
                    feasibility_status=STATUS_FEASIBLE,
                )
            ],
        )
        assess_id = db.record_distribution_assessment(plan, run_id=run_id)
        assert db.count_delivery_zone_assessments(assess_id) == 1

        db.delete_delivery_assessment(assess_id)
        assert db.get_delivery_assessment_by_id(assess_id) is None
        assert db.count_delivery_zone_assessments(assess_id) == 0

    def test_zone_name_snapshot_retained_after_operational_renaming(self, populated_network):
        """If an operational zone is renamed later, the historical delivery assessment retains the snapshot."""
        data = populated_network
        db = data["db"]
        run_id = data["run_id"]
        src_id = data["src_id"]
        z1_id = data["zones"][0]

        plan = DistributionPlanResult(
            source_node_id=src_id,
            duration_minutes=60.0,
            routing_strategy=STRATEGY_SHORTEST_PATH,
            total_allocated_l=100.0,
            total_deliverable_l=100.0,
            total_undelivered_l=0.0,
            overall_delivery_efficiency_pct=100.0,
            zone_results=[
                ZoneDeliveryResult(
                    zone_id=z1_id,
                    zone_name="Original District Name",
                    destination_node_id=data["dest_1"],
                    allocated_volume_l=100.0,
                    deliverable_volume_l=100.0,
                    undelivered_volume_l=0.0,
                    feasibility_status=STATUS_FEASIBLE,
                )
            ],
        )
        assess_id = db.record_distribution_assessment(plan, run_id=run_id)

        # Rename operational zone
        db.update_allocation_zone(
            zone_id=z1_id,
            zone_name="New Renovated District Name",
            requested_demand_l=5000.0,
            essential_min_l=2000.0,
            fairness_weight=1.0,
            priority_category="standard",
            status="active",
        )

        # Historical assessment zone detail MUST still show the original snapshot name
        zone_rows = db.get_delivery_zone_assessments(assess_id)
        assert zone_rows[0]["zone_name_snapshot"] == "Original District Name"

    def test_node_deletion_preserves_historical_snapshots(self, populated_network):
        """Deleting an unreferenced endpoint node sets FK to NULL while preserving name snapshots."""
        data = populated_network
        db = data["db"]
        run_id = data["run_id"]

        # Create isolated temporary source and destination node without pipe FK restrictions
        src_temp = db.create_node("Temp Source Reservoir", "tank", "Site 1", "active")
        dest_temp = db.create_node("Temp Destination Tap", "valve", "Site 2", "active")

        plan = DistributionPlanResult(
            source_node_id=src_temp,
            source_node_name="Temp Source Reservoir",
            duration_minutes=60.0,
            routing_strategy=STRATEGY_SHORTEST_PATH,
            total_allocated_l=50.0,
            total_deliverable_l=50.0,
            total_undelivered_l=0.0,
            overall_delivery_efficiency_pct=100.0,
            zone_results=[
                ZoneDeliveryResult(
                    zone_id=None,
                    zone_name="Temporary Zone",
                    destination_node_id=dest_temp,
                    allocated_volume_l=50.0,
                    deliverable_volume_l=50.0,
                    undelivered_volume_l=0.0,
                    feasibility_status=STATUS_FEASIBLE,
                )
            ],
        )
        assess_id = db.record_distribution_assessment(plan, run_id=run_id)

        # Delete the temporary nodes
        db.delete_node(src_temp)
        db.delete_node(dest_temp)

        # Assessment row must still exist with source_node_id set to NULL and snapshot preserved
        assess_row = db.get_delivery_assessment_by_id(assess_id)
        assert assess_row is not None
        assert assess_row["source_node_id"] is None
        assert assess_row["source_name_snapshot"] == "Temp Source Reservoir"

        # Zone row destination_node_id must be NULL and snapshot preserved
        z_rows = db.get_delivery_zone_assessments(assess_id)
        assert z_rows[0]["destination_node_id"] is None
        assert z_rows[0]["destination_name_snapshot"] == "Temp Destination Tap"


class TestIntegrationService:
    """Tests the evaluate_and_record_run_distribution integration pipeline."""

    def test_evaluate_and_record_run_distribution_e2e(self, populated_network):
        """End-to-end service execution: run -> zones -> graph -> engine -> persistence."""
        data = populated_network
        db = data["db"]
        run_id = data["run_id"]
        src_id = data["src_id"]

        assess_id, plan = evaluate_and_record_run_distribution(
            run_id=run_id,
            source_node_id=src_id,
            duration_minutes=60.0,
            routing_strategy=STRATEGY_WIDEST_PATH,
            db_module=db,
        )

        assert assess_id > 0
        assert isinstance(plan, DistributionPlanResult)
        assert plan.source_node_id == src_id
        assert plan.routing_strategy == STRATEGY_WIDEST_PATH
        assert len(plan.zone_results) == 2

        # Verify persisted record in SQLite
        row = db.get_delivery_assessment_by_id(assess_id)
        assert row["assessment_id"] == assess_id
        assert row["run_id"] == run_id
        assert row["routing_strategy"] == STRATEGY_WIDEST_PATH
        assert row["total_allocated_l"] == pytest.approx(plan.total_allocated_l)
        assert row["total_deliverable_l"] == pytest.approx(plan.total_deliverable_l)

        # Verify multiple independent assessments can be recorded for the same allocation run
        assess_id_2, plan_2 = evaluate_and_record_run_distribution(
            run_id=run_id,
            source_node_id=src_id,
            duration_minutes=60.0,
            routing_strategy=STRATEGY_SHORTEST_PATH,
            db_module=db,
        )
        assert assess_id_2 != assess_id

        run_assessments = db.get_delivery_assessments_for_run(run_id)
        assert len(run_assessments) == 2
        strategies = {r["routing_strategy"] for r in run_assessments}
        assert strategies == {STRATEGY_SHORTEST_PATH, STRATEGY_WIDEST_PATH}

    def test_evaluate_nonexistent_run_raises_key_error(self, populated_network):
        data = populated_network
        with pytest.raises(KeyError, match="does not exist"):
            evaluate_and_record_run_distribution(
                run_id=99999,
                source_node_id=data["src_id"],
                db_module=data["db"],
            )

    def test_evaluate_inactive_source_node_raises_value_error(self, populated_network):
        data = populated_network
        db = data["db"]
        inactive_src = db.create_node("Inactive Tank", "tank", "Site 3", "inactive")

        with pytest.raises(ValueError, match="is inactive"):
            evaluate_and_record_run_distribution(
                run_id=data["run_id"],
                source_node_id=inactive_src,
                db_module=db,
            )


class TestQueriesAndRetrieval:
    """Tests getter and count query functions."""

    def test_get_latest_delivery_assessment(self, populated_network):
        data = populated_network
        db = data["db"]
        run_id = data["run_id"]
        src_id = data["src_id"]

        assert db.get_latest_delivery_assessment() is None

        plan1 = DistributionPlanResult(
            source_node_id=src_id,
            duration_minutes=30.0,
            routing_strategy=STRATEGY_SHORTEST_PATH,
            total_allocated_l=0.0,
            total_deliverable_l=0.0,
            total_undelivered_l=0.0,
            overall_delivery_efficiency_pct=100.0,
            zone_results=[],
        )
        id1 = db.record_distribution_assessment(plan1, run_id=run_id)

        plan2 = DistributionPlanResult(
            source_node_id=src_id,
            duration_minutes=60.0,
            routing_strategy=STRATEGY_WIDEST_PATH,
            total_allocated_l=0.0,
            total_deliverable_l=0.0,
            total_undelivered_l=0.0,
            overall_delivery_efficiency_pct=100.0,
            zone_results=[],
        )
        id2 = db.record_distribution_assessment(plan2, run_id=run_id)

        latest = db.get_latest_delivery_assessment()
        assert latest is not None
        assert latest["assessment_id"] == id2

        latest_for_run = db.get_latest_delivery_assessment(run_id=run_id)
        assert latest_for_run["assessment_id"] == id2


class TestHistoricalDataIntegrityAudit:
    """
    Dedicated tests for Historical Data Integrity and UI Readiness Audit:
    - Source node renaming and deletion after delivery assessment recorded.
    - Historical allocation mapping with inactive, renamed, or unmapped zones.
    - End-to-end 6-step lifecycle workflow test on an isolated database.
    """

    def test_source_node_renaming_and_deletion_preserves_historical_snapshots(self, populated_network):
        """
        Verify that get_delivery_assessment_by_id returns the historical source name
        even when the operational source node is renamed or deleted.
        """
        data = populated_network
        db = data["db"]
        run_id = data["run_id"]

        # Create a dedicated source node and destination node for this test
        src_temp = db.create_node("Original Reservoir Tank", "tank", "North Gate", "active")
        dest_temp = db.create_node("Consumer Tap Alpha", "valve", "South Gate", "active")
        db.create_pipe("Temp-Feed", src_temp, dest_temp, 50.0, 100.0, 1.0, "active")

        plan = DistributionPlanResult(
            source_node_id=src_temp,
            source_node_name="Original Reservoir Tank",
            duration_minutes=60.0,
            routing_strategy=STRATEGY_SHORTEST_PATH,
            total_allocated_l=100.0,
            total_deliverable_l=100.0,
            total_undelivered_l=0.0,
            overall_delivery_efficiency_pct=100.0,
            zone_results=[
                ZoneDeliveryResult(
                    zone_id=None,
                    zone_name="Test Historical Zone",
                    destination_node_id=dest_temp,
                    allocated_volume_l=100.0,
                    deliverable_volume_l=100.0,
                    undelivered_volume_l=0.0,
                    feasibility_status=STATUS_FEASIBLE,
                )
            ],
        )
        assess_id = db.record_distribution_assessment(plan, run_id=run_id)

        # 1. Rename the source node in network_nodes
        db.update_node(
            src_temp,
            name="Renamed Mega Reservoir 9000",
            node_type="tank",
            location="North Gate",
            status="active",
        )

        # Query assessment: source_name_snapshot and source_node_name MUST return original snapshot
        assess_row = db.get_delivery_assessment_by_id(assess_id)
        assert assess_row is not None
        assert assess_row["source_name_snapshot"] == "Original Reservoir Tank"
        assert assess_row["source_node_name"] == "Original Reservoir Tank"

        # 2. Delete the pipes and then the source node (FK ON DELETE SET NULL)
        pipes = db.get_all_pipes()
        for p in pipes:
            if p["from_node_id"] == src_temp or p["to_node_id"] == src_temp:
                db.delete_pipe(p["id"])
        db.delete_node(src_temp)

        # Query assessment: source_node_id is NULL, snapshot still preserved
        assess_after_delete = db.get_delivery_assessment_by_id(assess_id)
        assert assess_after_delete is not None
        assert assess_after_delete["source_node_id"] is None
        assert assess_after_delete["source_name_snapshot"] == "Original Reservoir Tank"
        assert assess_after_delete["source_node_name"] == "Original Reservoir Tank"

    def test_evaluate_historical_run_with_inactive_and_unmapped_zones(self, populated_network):
        """
        Verify that service.evaluate_and_record_run_distribution correctly evaluates
        historical allocation runs when zones are currently inactive, and that unmapped zones
        at allocation time are explicitly classified as UNMAPPED.
        """
        data = populated_network
        db = data["db"]
        src_id = data["src_id"]
        z1_id = data["zones"][0]

        # Deactivate Zone 1 in operational table
        db.update_allocation_zone(
            zone_id=z1_id,
            zone_name="South Residential District",
            requested_demand_l=5000.0,
            essential_min_l=2000.0,
            fairness_weight=1.0,
            priority_category="standard",
            status="inactive",
            node_id=data["dest_1"],
        )

        # Create an explicitly unmapped zone at allocation time
        z_unmapped = db.create_allocation_zone(
            zone_name="Unmapped Fringe District",
            requested_demand_l=2500.0,
            essential_min_l=1000.0,
            fairness_weight=1.0,
            priority_category="standard",
            status="active",
            node_id=None,
        )

        # Run fresh allocation with the unmapped zone included
        from allocation.engine import ZoneDemandInput, allocate_water
        inputs = [
            ZoneDemandInput(
                zone_id=z1_id,
                zone_name="South Residential District",
                requested_demand=5000.0,
                essential_min=2000.0,
                fairness_weight=1.0,
            ),
            ZoneDemandInput(
                zone_id=z_unmapped,
                zone_name="Unmapped Fringe District",
                requested_demand=2500.0,
                essential_min=1000.0,
                fairness_weight=1.0,
            ),
        ]
        alloc_res = allocate_water(total_supply=7500.0, zones=inputs)
        run_id_2 = db.record_allocation_run(alloc_res)

        # Evaluate distribution for this run
        assess_id, plan = evaluate_and_record_run_distribution(
            run_id=run_id_2,
            source_node_id=src_id,
            duration_minutes=60.0,
            routing_strategy=STRATEGY_SHORTEST_PATH,
            db_module=db,
        )

        assert assess_id > 0
        zone_map = {z.zone_id: z for z in plan.zone_results}

        # Inactive zone (Zone 1) retains historical allocation and routes to its destination
        assert zone_map[z1_id].allocated_volume_l == 5000.0
        assert zone_map[z1_id].feasibility_status in (STATUS_FEASIBLE, STATUS_CAPACITY_CONSTRAINED)

        # Unmapped zone is explicitly classified as UNMAPPED (no guessing)
        assert zone_map[z_unmapped].allocated_volume_l == 2500.0
        assert zone_map[z_unmapped].feasibility_status == STATUS_UNMAPPED
        assert zone_map[z_unmapped].deliverable_volume_l == 0.0
        assert zone_map[z_unmapped].undelivered_volume_l == 2500.0
        assert "not mapped" in zone_map[z_unmapped].explanation.lower()

    def test_historical_mapping_before_first_assessment_scenario(self, isolated_db):
        """
        Phase 3.1 Gap Scenario:
        1. Create a source node and two destination nodes, Node A and Node B.
        2. Create a zone mapped to Node A.
        3. Create an allocation run and persist an allocation for that zone.
        4. Do NOT create a delivery assessment yet.
        5. Change the zone's current mapping from Node A to Node B.
        6. Now assess the original allocation run for the first time.
        7. Inspect the persisted delivery assessment:
           - The assessment MUST use Node A (the allocation-time mapping).
           - It MUST NEVER silently use Node B just because that is the current mapping.
           - The original allocated amount and zone-name snapshot must remain unchanged.
        """
        db = isolated_db
        from allocation.engine import ZoneDemandInput, allocate_water

        # 1. Create a source node and two destination nodes, Node A and Node B
        src_id = db.create_node("Central Reservoir", "tank", "North Sector", "active")
        node_a = db.create_node("Terminal Node Alpha", "valve", "Sector Alpha", "active")
        node_b = db.create_node("Terminal Node Beta", "valve", "Sector Beta", "active")

        pipe_a = db.create_pipe("Feed-To-Alpha", src_id, node_a, 100.0, 50.0, 1.0, "active")
        pipe_b = db.create_pipe("Feed-To-Beta", src_id, node_b, 200.0, 50.0, 2.0, "active")

        # 2. Create a zone mapped to Node A
        z_id = db.create_allocation_zone(
            zone_name="Alpha Priority District",
            requested_demand_l=1500.0,
            essential_min_l=500.0,
            fairness_weight=1.0,
            priority_category="standard",
            status="active",
            node_id=node_a,
        )

        # 3. Create an allocation run and persist an allocation for that zone
        alloc_res = allocate_water(
            total_supply=1500.0,
            zones=[
                ZoneDemandInput(
                    zone_id=z_id,
                    zone_name="Alpha Priority District",
                    requested_demand=1500.0,
                    essential_min=500.0,
                    fairness_weight=1.0,
                )
            ],
        )
        run_id = db.record_allocation_run(alloc_res, notes="Run prior to remapping")

        # 4. Verify that no delivery assessment exists yet
        assert db.count_delivery_assessments(run_id=run_id) == 0

        # 5. Change the zone's current mapping from Node A to Node B in allocation_zones
        db.update_allocation_zone(
            zone_id=z_id,
            zone_name="Alpha Priority District",
            requested_demand_l=1500.0,
            essential_min_l=500.0,
            fairness_weight=1.0,
            priority_category="standard",
            status="active",
            node_id=node_b,  # Operational remapping to Node B!
        )

        # Verify operational zone now points to Node B
        current_zone = db.get_allocation_zone_by_id(z_id)
        assert current_zone["node_id"] == node_b

        # 6. Now assess the original allocation run for the first time
        assess_id, plan = evaluate_and_record_run_distribution(
            run_id=run_id,
            source_node_id=src_id,
            duration_minutes=60.0,
            routing_strategy=STRATEGY_SHORTEST_PATH,
            db_module=db,
        )

        # 7. Inspect the persisted delivery assessment
        assert assess_id > 0
        zone_details = db.get_delivery_zone_assessments(assess_id)
        assert len(zone_details) == 1

        zd = zone_details[0]

        # MUST use Node A (original allocation-time mapping), NEVER Node B (current mapping)
        assert zd["destination_node_id"] == node_a
        assert zd["destination_name_snapshot"] == "Terminal Node Alpha"
        assert zd["destination_node_name"] == "Terminal Node Alpha"
        assert zd["destination_node_id"] != node_b

        # Route MUST go to Node A
        assert json.loads(zd["route_node_ids"]) == [src_id, node_a]
        assert json.loads(zd["route_pipe_ids"]) == [pipe_a]

        # Original allocated amount and zone name snapshot must remain unchanged
        assert zd["zone_name_snapshot"] == "Alpha Priority District"
        assert zd["zone_name"] == "Alpha Priority District"
        assert zd["allocated_amount_l"] == 1500.0

    def test_legacy_unpreserved_destination_node_evaluates_as_unmapped_without_guessing(self, isolated_db):
        """
        Verify that if an older allocation record lacks a destination node snapshot (NULL),
        the assessment explicitly marks the zone UNMAPPED and NEVER guesses from current allocation_zones.
        """
        db = isolated_db
        from allocation.engine import ZoneDemandInput, allocate_water

        src_id = db.create_node("Main Reservoir", "tank", "North Sector", "active")
        node_b = db.create_node("Operational Node Beta", "valve", "Sector Beta", "active")
        db.create_pipe("Main-Feed", src_id, node_b, 100.0, 50.0, 1.0, "active")

        # Create zone mapped to Node B in operational table
        z_id = db.create_allocation_zone(
            zone_name="Legacy Zone",
            requested_demand_l=1000.0,
            essential_min_l=500.0,
            fairness_weight=1.0,
            priority_category="standard",
            status="active",
            node_id=node_b,
        )

        alloc_res = allocate_water(
            total_supply=1000.0,
            zones=[
                ZoneDemandInput(
                    zone_id=z_id,
                    zone_name="Legacy Zone",
                    requested_demand=1000.0,
                    essential_min=500.0,
                    fairness_weight=1.0,
                )
            ],
        )
        run_id = db.record_allocation_run(alloc_res)

        # Simulate legacy record: manually set destination_node_id to NULL in water_allocations
        with db._connect() as conn:
            conn.execute(
                "UPDATE water_allocations SET destination_node_id = NULL WHERE run_id = ?",
                (run_id,),
            )
            conn.commit()

        # Assess the run
        assess_id, plan = evaluate_and_record_run_distribution(
            run_id=run_id,
            source_node_id=src_id,
            duration_minutes=60.0,
            routing_strategy=STRATEGY_SHORTEST_PATH,
            db_module=db,
        )

        assert assess_id > 0
        zone_details = db.get_delivery_zone_assessments(assess_id)
        assert len(zone_details) == 1
        zd = zone_details[0]

        # Must be UNMAPPED: must NOT guess Node B even though current zone points to Node B!
        assert zd["destination_node_id"] is None
        assert zd["feasibility_status"] == STATUS_UNMAPPED
        assert zd["deliverable_amount_l"] == 0.0
        assert zd["undelivered_amount_l"] == 1000.0

    def test_e2e_historical_workflow_mutations_and_retrieval(self, isolated_db):
        """
        Step 1 through 6 lifecycle test on isolated database:
        1. Record a delivery assessment.
        2. Rename one zone.
        3. Change one zone's mapping.
        4. Deactivate another zone.
        5. Delete an operational node where permitted.
        6. Retrieve the original delivery assessment and verify snapshots remain 100% accurate.
        """
        db = isolated_db

        # Setup network
        src_id = db.create_node("Master Reservoir", "tank", "Sector North", "active")
        node_a = db.create_node("District Node A", "valve", "Sector West", "active")
        node_b = db.create_node("District Node B", "valve", "Sector East", "active")
        isolated_tap = db.create_node("Isolated Tap Node", "valve", "Sector Far", "active")

        pipe_a = db.create_pipe("Main-A", src_id, node_a, 100.0, 50.0, 1.0, "active")
        pipe_b = db.create_pipe("Main-B", src_id, node_b, 150.0, 80.0, 1.0, "active")

        # Setup zones
        z1 = db.create_allocation_zone(
            zone_name="Alpha Residential Zone",
            requested_demand_l=1200.0,
            essential_min_l=600.0,
            fairness_weight=1.0,
            priority_category="standard",
            status="active",
            node_id=node_a,
        )
        z2 = db.create_allocation_zone(
            zone_name="Beta Commercial Zone",
            requested_demand_l=1800.0,
            essential_min_l=900.0,
            fairness_weight=1.5,
            priority_category="commercial",
            status="active",
            node_id=node_b,
        )

        # Allocate water (Day 8 engine)
        active_zones = db.get_active_allocation_zones()
        from allocation.engine import ZoneDemandInput, allocate_water
        inputs = [
            ZoneDemandInput(
                zone_id=z["zone_id"],
                zone_name=z["zone_name"],
                requested_demand=z["requested_demand_l"],
                essential_min=z["essential_min_l"],
                fairness_weight=z["fairness_weight"],
            )
            for z in active_zones
        ]
        alloc_res = allocate_water(total_supply=3000.0, zones=inputs)
        run_id = db.record_allocation_run(alloc_res)

        # Step 1: Record a delivery assessment
        assess_id, plan = evaluate_and_record_run_distribution(
            run_id=run_id,
            source_node_id=src_id,
            duration_minutes=60.0,
            routing_strategy=STRATEGY_SHORTEST_PATH,
            db_module=db,
        )
        assert assess_id > 0

        # Step 2: Rename one zone in operational table
        db.update_allocation_zone(
            zone_id=z1,
            zone_name="Renamed Alpha Sector Deluxe",
            requested_demand_l=2000.0,  # Changed demand too!
            essential_min_l=1000.0,
            fairness_weight=2.0,
            priority_category="critical",
            status="active",
            node_id=node_a,
        )

        # Step 3: Change one zone's mapping in operational table
        db.update_allocation_zone(
            zone_id=z2,
            zone_name="Beta Commercial Zone",
            requested_demand_l=1800.0,
            essential_min_l=900.0,
            fairness_weight=1.5,
            priority_category="commercial",
            status="active",
            node_id=node_a,  # Remapped to Node A!
        )

        # Step 4: Deactivate another zone
        db.update_allocation_zone(
            zone_id=z1,
            zone_name="Renamed Alpha Sector Deluxe",
            requested_demand_l=2000.0,
            essential_min_l=1000.0,
            fairness_weight=2.0,
            priority_category="critical",
            status="inactive",  # Deactivated!
            node_id=node_a,
        )

        # Step 5: Delete an operational node where permitted (isolated node)
        db.delete_node(isolated_tap)

        # Step 6: Retrieve the original delivery assessment and verify 100% snapshot accuracy
        assess_row = db.get_delivery_assessment_by_id(assess_id)
        assert assess_row is not None
        assert assess_row["assessment_id"] == assess_id
        assert assess_row["source_name_snapshot"] == "Master Reservoir"
        assert assess_row["source_node_name"] == "Master Reservoir"
        assert assess_row["total_allocated_l"] == 3000.0
        assert assess_row["routing_strategy"] == STRATEGY_SHORTEST_PATH

        zone_details = db.get_delivery_zone_assessments(assess_id)
        assert len(zone_details) == 2

        zd_map = {zd["zone_id"]: zd for zd in zone_details}

        # Zone 1 checks: original snapshot name, original allocated amount, original destination
        z1_hist = zd_map[z1]
        assert z1_hist["zone_name_snapshot"] == "Alpha Residential Zone"
        assert z1_hist["zone_name"] == "Alpha Residential Zone"
        assert z1_hist["destination_node_id"] == node_a
        assert z1_hist["destination_name_snapshot"] == "District Node A"
        assert z1_hist["destination_node_name"] == "District Node A"
        assert z1_hist["allocated_amount_l"] == 1200.0  # NOT 2000.0 (current demand)
        assert json.loads(z1_hist["route_node_ids"]) == [src_id, node_a]
        assert json.loads(z1_hist["route_pipe_ids"]) == [pipe_a]

        # Zone 2 checks: original snapshot name, original allocated amount, original destination B
        z2_hist = zd_map[z2]
        assert z2_hist["zone_name_snapshot"] == "Beta Commercial Zone"
        assert z2_hist["zone_name"] == "Beta Commercial Zone"
        assert z2_hist["destination_node_id"] == node_b  # Remapping to Node A did NOT alter history!
        assert z2_hist["destination_name_snapshot"] == "District Node B"
        assert z2_hist["destination_node_name"] == "District Node B"
        assert z2_hist["allocated_amount_l"] == 1800.0
        assert json.loads(z2_hist["route_node_ids"]) == [src_id, node_b]
        assert json.loads(z2_hist["route_pipe_ids"]) == [pipe_b]
