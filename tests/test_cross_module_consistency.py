"""
tests/test_cross_module_consistency.py – Cross-Module Data Consistency and Historical Integrity Audit Test Suite.

Validates Day 10 Phase 3 requirements:
1. Network data consistency (inactive pipes excluded, parallel pipes evaluated independently, missing nodes safe, read-only routing).
2. Sensor and leak data consistency (scenario provenance, baseline purity, duplicate prevention, stale telemetry protection).
3. Leak and repair lifecycle consistency (task generation, Max Heap priorities, audit trail, transaction atomicity).
4. Allocation and distribution consistency (supply/volume reconciliation, historical destination snapshots, contention warnings).
5. Historical mutation testing (10 operational mutations verifying immutable snapshots).
6. Transaction failure and rollback consistency (allocation, distribution, audit, and sensor reading batch rollbacks).
"""

from __future__ import annotations

import os
import shutil
import tempfile
import pytest

import database as db
import distribution.service
from allocation.engine import allocate_water, ZoneDemandInput
from app import app
from detection.engine import LeakDetector
from dsa.graph import load_graph_from_db
from repair_queue.manager import PriorityQueueManager


@pytest.fixture
def tmp_audit_db():
    """Create an isolated temporary SQLite database for cross-module consistency tests."""
    temp_dir = tempfile.mkdtemp()
    temp_db_path = os.path.join(temp_dir, "test_consistency.db")

    orig_db_path = db.DB_PATH
    orig_instance_dir = db.INSTANCE_DIR

    db.DB_PATH = temp_db_path
    db.INSTANCE_DIR = temp_dir
    db.init_db()

    yield {
        "db_path": temp_db_path,
        "temp_dir": temp_dir,
    }

    db.DB_PATH = orig_db_path
    db.INSTANCE_DIR = orig_instance_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def consistency_network(tmp_audit_db):
    """Seed a representative multi-pipe network with parallel conduits, sensors, and zones."""
    tank_id = db.create_node("Main Reservoir", "tank", "North Sector", "active")
    mid_id = db.create_node("Central Hub", "junction", "Central Sector", "active")
    dest_id = db.create_node("East Consumer Node", "junction", "East Sector", "active")

    # Twin parallel pipes connecting Main Reservoir to Central Hub
    # Pipe 1: Low cost (1.0), lower capacity (2000 LPM)
    # Pipe 2: Higher cost (3.0), higher capacity (6000 LPM)
    pipe_low_cost = db.create_pipe("Trunk Alpha (Low Cost)", tank_id, mid_id, "1000", "2000", "1.0", "active")
    pipe_high_cap = db.create_pipe("Trunk Beta (High Cap)", tank_id, mid_id, "1000", "6000", "3.0", "active")

    # Branch from Central Hub to East Consumer
    pipe_branch = db.create_pipe("Branch East", mid_id, dest_id, "500", "4000", "1.0", "active")

    # Inactive bypass pipe
    pipe_inactive = db.create_pipe("Bypass Direct (Inactive)", tank_id, dest_id, "300", "8000", "0.5", "inactive")

    # Sensors installed on Trunk Alpha
    sensor_flow = db.create_sensor("FLOW-TRUNK-A", pipe_low_cost, "flow_meter", "L/min", "active")
    sensor_press = db.create_sensor("PRESS-TRUNK-A", pipe_low_cost, "pressure_sensor", "bar", "active")

    # Consumer zone mapped to East Consumer Node
    zone_id = db.create_allocation_zone(
        zone_name="Sector East Zone",
        requested_demand_l=1500.0,
        essential_min_l=300.0,
        fairness_weight=1.0,
        priority_category="standard",
        status="active",
        node_id=dest_id,
    )

    return {
        "tank_id": tank_id,
        "mid_id": mid_id,
        "dest_id": dest_id,
        "pipe_low_cost": pipe_low_cost,
        "pipe_high_cap": pipe_high_cap,
        "pipe_branch": pipe_branch,
        "pipe_inactive": pipe_inactive,
        "sensor_flow": sensor_flow,
        "sensor_press": sensor_press,
        "zone_id": zone_id,
    }


# ── PART 1: NETWORK DATA CONSISTENCY ──────────────────────────────────────────


class TestNetworkDataConsistency:
    """Validate graph loading, inactive pipe exclusion, parallel edge handling, and read-only routing."""

    def test_inactive_pipes_excluded_from_routing(self, tmp_audit_db, consistency_network):
        """Verify inactive bypass pipe is excluded from both Dijkstra and Widest Path."""
        graph = load_graph_from_db(db_module=db, directed=False)
        tank = consistency_network["tank_id"]
        dest = consistency_network["dest_id"]

        # 1. Dijkstra shortest path should traverse via Central Hub, NOT the inactive direct bypass
        dijkstra_res = graph.dijkstra(tank, dest)
        assert dijkstra_res.found is True
        assert dijkstra_res.node_path == [tank, consistency_network["mid_id"], dest]
        assert consistency_network["pipe_inactive"] not in dijkstra_res.pipe_ids

        # 2. Widest path should also traverse via Central Hub, ignoring the inactive 8000 LPM bypass
        widest_res = graph.widest_path(tank, dest)
        assert widest_res.found is True
        assert widest_res.node_path == [tank, consistency_network["mid_id"], dest]
        assert consistency_network["pipe_inactive"] not in widest_res.pipe_ids

    def test_parallel_pipes_independently_represented_and_evaluated(self, tmp_audit_db, consistency_network):
        """
        Verify parallel pipes between same nodes are independently evaluated:
        - Dijkstra selects Trunk Alpha (cost 1.0, capacity 2000).
        - Widest Path selects Trunk Beta (capacity 6000, cost 3.0).
        """
        graph = load_graph_from_db(db_module=db, directed=False)
        tank = consistency_network["tank_id"]
        mid = consistency_network["mid_id"]

        dijkstra_res = graph.dijkstra(tank, mid)
        widest_res = graph.widest_path(tank, mid)

        # Dijkstra chooses minimum cost edge (Trunk Alpha)
        assert dijkstra_res.pipe_ids == [consistency_network["pipe_low_cost"]]
        assert dijkstra_res.total_cost == 1.0

        # Widest Path chooses maximum bottleneck capacity edge (Trunk Beta)
        assert widest_res.pipe_ids == [consistency_network["pipe_high_cap"]]
        assert widest_res.bottleneck_capacity == 6000.0

    def test_deleted_or_missing_nodes_handled_safely(self, tmp_audit_db, consistency_network):
        """Verify missing node IDs return empty traversal or unreachable results without throwing exceptions."""
        graph = load_graph_from_db(db_module=db, directed=False)
        missing_node = 99999

        # BFS returns empty list
        assert graph.breadth_first_search(missing_node) == []

        # Dijkstra returns found=False
        dijkstra_res = graph.dijkstra(missing_node, consistency_network["dest_id"])
        assert dijkstra_res.found is False
        assert dijkstra_res.node_path == []

        # Widest Path returns found=False
        widest_res = graph.widest_path(consistency_network["tank_id"], missing_node)
        assert widest_res.found is False
        assert widest_res.bottleneck_capacity is None or widest_res.bottleneck_capacity == 0.0

    def test_routing_algorithms_are_strictly_readonly(self, tmp_audit_db, consistency_network):
        """Verify executing routing algorithms does not insert, update, or delete any database rows."""
        graph = load_graph_from_db(db_module=db, directed=False)
        tank = consistency_network["tank_id"]
        dest = consistency_network["dest_id"]

        with db._connect() as conn:
            pre_pipes = conn.execute("SELECT COUNT(*) FROM pipes").fetchone()[0]
            pre_nodes = conn.execute("SELECT COUNT(*) FROM network_nodes").fetchone()[0]

        _ = graph.breadth_first_search(tank)
        _ = graph.dijkstra(tank, dest)
        _ = graph.widest_path(tank, dest)

        with db._connect() as conn:
            post_pipes = conn.execute("SELECT COUNT(*) FROM pipes").fetchone()[0]
            post_nodes = conn.execute("SELECT COUNT(*) FROM network_nodes").fetchone()[0]

        assert pre_pipes == post_pipes
        assert pre_nodes == post_nodes


# ── PART 2: SENSOR AND LEAK DATA CONSISTENCY ──────────────────────────────────


class TestSensorAndLeakDataConsistency:
    """Validate scenario labeling, baseline isolation, deduplication, and stale telemetry protection."""

    def test_simulated_readings_retain_scenario_labels(self, tmp_audit_db, consistency_network):
        """Verify that sensor readings retain their exact simulation scenario provenance."""
        s_flow = consistency_network["sensor_flow"]
        r1 = db.record_sensor_reading(s_flow, 100.0, scenario="normal")
        r2 = db.record_sensor_reading(s_flow, 450.0, scenario="leak")

        with db._connect() as conn:
            row1 = conn.execute("SELECT scenario FROM sensor_readings WHERE reading_id = ?", (r1,)).fetchone()
            row2 = conn.execute("SELECT scenario FROM sensor_readings WHERE reading_id = ?", (r2,)).fetchone()

        assert row1["scenario"] == "normal"
        assert row2["scenario"] == "leak"

    def test_leak_readings_do_not_contaminate_normal_baselines(self, tmp_audit_db, consistency_network):
        """Verify that leak scenario telemetry is strictly excluded from normal baseline statistical queries."""
        s_flow = consistency_network["sensor_flow"]

        # Insert 10 normal readings with mean 100.0
        for _ in range(10):
            db.record_sensor_reading(s_flow, 100.0, scenario="normal")

        # Insert 5 massive leak readings (1000.0)
        for _ in range(5):
            db.record_sensor_reading(s_flow, 1000.0, scenario="leak")

        # Detection engine computes normal baseline
        detector = LeakDetector()
        sensor_dict = dict(db.get_sensor_by_id(s_flow))
        readings = [dict(r) for r in db.get_readings_for_sensor(s_flow, limit=50)]
        analysis = detector.analyze_sensor(sensor_dict, readings)

        # Baseline mean must strictly reflect normal readings (~100.0), NOT shifted by the 1000.0 leak spikes
        assert analysis.baseline_mean is not None
        assert abs(analysis.baseline_mean - 100.0) < 0.001
        assert analysis.baseline_samples == 10
        assert analysis.is_abnormal is True
        assert analysis.classification == "elevated_flow"

    def test_resolved_leak_event_not_recreated_from_stale_telemetry(self, tmp_audit_db, consistency_network):
        """Verify that an already resolved leak event cannot be re-opened by stale readings."""
        p_alpha = consistency_network["pipe_low_cost"]

        ev_id, is_new = db.record_leak_event(p_alpha, "high", "Active Rupture", detected_at="2026-11-14T10:00:00Z")
        assert is_new is True

        # Resolve the leak event at 10:30:00Z
        db.update_leak_event_status(ev_id, "resolved")

        # Try to record an event using stale detected_at (10:15:00Z, before resolved_at)
        ev_id2, is_new2 = db.record_leak_event(
            p_alpha, "critical", "Stale Detection Call", detected_at="2026-11-14T10:15:00Z"
        )
        assert is_new2 is False
        assert ev_id2 == ev_id

        # Leak event status must remain 'resolved'
        ev_row = db.get_leak_event_by_id(ev_id)
        assert ev_row["status"] == "resolved"

    def test_duplicate_active_leak_events_prevented(self, tmp_audit_db, consistency_network):
        """Verify that multiple detection triggers on the same active pipe escalate rather than duplicate."""
        p_alpha = consistency_network["pipe_low_cost"]

        ev1, is_new1 = db.record_leak_event(p_alpha, "low", "Minor seepage", detected_at="2026-11-14T11:00:00Z")
        assert is_new1 is True

        # Second detection triggers on same pipe with higher severity
        ev2, is_new2 = db.record_leak_event(p_alpha, "critical", "Major rupture", detected_at="2026-11-14T11:05:00Z")
        assert is_new2 is False
        assert ev2 == ev1

        ev_row = db.get_leak_event_by_id(ev1)
        assert ev_row["severity"] == "critical"


# ── PART 3: LEAK AND REPAIR LIFECYCLE CONSISTENCY ─────────────────────────────


class TestLeakAndRepairLifecycleConsistency:
    """Validate task synchronization, Max Heap ordering, duplicate prevention, and audit lineage."""

    def test_leak_event_generates_correct_repair_task_and_heap_order(self, tmp_audit_db, consistency_network):
        """Verify that recording leak events synchronizes into repair_tasks and heap extracts in priority order."""
        p_alpha = consistency_network["pipe_low_cost"]
        p_beta = consistency_network["pipe_high_cap"]

        # Ev1 on Alpha: Medium severity
        ev1, _ = db.record_leak_event(p_alpha, "medium", "Seepage on Alpha")
        # Ev2 on Beta: Critical severity
        ev2, _ = db.record_leak_event(p_beta, "critical", "Burst on Beta")

        queue_mgr = PriorityQueueManager()
        heap = queue_mgr.sync_and_build_heap(db)

        # Max Heap should have 2 active tasks
        assert heap.size() == 2

        # Extract max priority task -> Critical event on Beta must be top priority
        top_task = heap.extract_max()
        assert top_task.event_id == ev2
        assert top_task.pipe_id == p_beta

        # Second task is Alpha (medium severity)
        second_task = heap.extract_max()
        assert second_task.event_id == ev1
        assert second_task.pipe_id == p_alpha

        # Max Heap invariant: higher severity event scores higher priority
        assert top_task.priority_score > second_task.priority_score

    def test_duplicate_repair_tasks_prevented_by_unique_constraint(self, tmp_audit_db, consistency_network):
        """Verify repeated queue synchronization creates exactly 1 repair task per leak event."""
        p_alpha = consistency_network["pipe_low_cost"]
        ev_id, _ = db.record_leak_event(p_alpha, "high", "High anomaly")

        queue_mgr = PriorityQueueManager()
        _ = queue_mgr.sync_and_build_heap(db)
        _ = queue_mgr.sync_and_build_heap(db)
        _ = queue_mgr.sync_and_build_heap(db)

        with db._connect() as conn:
            cnt = conn.execute("SELECT COUNT(*) FROM repair_tasks WHERE event_id = ?", (ev_id,)).fetchone()[0]
            assert cnt == 1

    def test_lifecycle_audit_records_state_transitions(self, tmp_audit_db, consistency_network):
        """Verify state transitions write immutable audit logs with correct action descriptions."""
        p_alpha = consistency_network["pipe_low_cost"]
        ev_id, _ = db.record_leak_event(p_alpha, "high", "High anomaly")
        task_id, _ = db.create_or_update_repair_task(ev_id, p_alpha, 75.0, "HIGH")

        # Transition task: Pending -> In Progress -> Completed
        db.update_repair_task_status(task_id, "In Progress", action_description="Crew A dispatched")
        db.update_repair_task_status(task_id, "Completed", action_description="Fixed completely")

        audit_logs = db.get_lifecycle_audit_logs(limit=20)
        task_audits = [a for a in audit_logs if a["entity_type"] == "repair_task" and a["entity_id"] == task_id]

        assert len(task_audits) >= 2
        statuses = [a["new_status"] for a in task_audits]
        assert "In Progress" in statuses
        assert "Completed" in statuses


# ── PART 4: ALLOCATION AND DISTRIBUTION CONSISTENCY ───────────────────────────


class TestAllocationAndDistributionConsistency:
    """Validate mathematical reconciliation, snapshot preservation, contention detection, and read-only views."""

    def test_allocation_totals_and_volume_reconciliation(self, tmp_audit_db, consistency_network):
        """
        Verify:
        1. Allocated volume reconciles with macro run total.
        2. Deliverable + Undelivered = Allocated volume for every zone.
        """
        dest_id = consistency_network["dest_id"]
        zone_id = consistency_network["zone_id"]
        tank_id = consistency_network["tank_id"]

        inputs = [
            ZoneDemandInput(
                zone_id=zone_id,
                zone_name="Sector East Zone",
                requested_demand=1500.0,
                essential_min=300.0,
                fairness_weight=1.0,
                node_id=dest_id,
            )
        ]
        alloc_plan = allocate_water(1200.0, inputs)
        run_id = db.record_allocation_run(alloc_plan, notes="Reconciliation Test")

        # Persisted run total matches
        run_row = db.get_allocation_run_by_id(run_id)
        assert run_row["total_allocated_l"] == 1200.0

        # Distribution evaluation
        aid, plan = distribution.service.evaluate_and_record_run_distribution(
            run_id=run_id,
            source_node_id=tank_id,
            duration_minutes=60.0,
            routing_strategy="SHORTEST_PATH",
        )

        for z in plan.zone_results:
            # Mathematical conservation: deliverable + undelivered == allocated
            assert abs((z.deliverable_volume_l + z.undelivered_volume_l) - z.allocated_volume_l) < 0.001

    def test_distribution_uses_historical_destination_snapshot(self, tmp_audit_db, consistency_network):
        """
        Verify an allocation run uses the destination node captured at allocation time,
        even if the zone is later operationalized to a different network node.
        """
        dest_id = consistency_network["dest_id"]
        zone_id = consistency_network["zone_id"]
        tank_id = consistency_network["tank_id"]

        inputs = [
            ZoneDemandInput(
                zone_id=zone_id,
                zone_name="Sector East Zone",
                requested_demand=1000.0,
                essential_min=200.0,
                fairness_weight=1.0,
                node_id=dest_id,
            )
        ]
        alloc_plan = allocate_water(800.0, inputs)
        run_id = db.record_allocation_run(alloc_plan, notes="Snapshot Integrity Test")

        # Change operational mapping of the zone to Central Hub (mid_id)
        mid_id = consistency_network["mid_id"]
        db.update_allocation_zone(
            zone_id=zone_id,
            zone_name="Sector East Zone",
            requested_demand_l=1000.0,
            essential_min_l=200.0,
            fairness_weight=1.0,
            priority_category="standard",
            status="active",
            node_id=mid_id,
        )

        # Evaluate distribution for the older run
        aid, plan = distribution.service.evaluate_and_record_run_distribution(
            run_id=run_id,
            source_node_id=tank_id,
            duration_minutes=60.0,
            routing_strategy="SHORTEST_PATH",
        )

        # Must route to original destination node dest_id, NOT the remapped mid_id
        z_res = plan.zone_results[0]
        assert z_res.destination_node_id == dest_id
        assert z_res.route_node_ids[-1] == dest_id

    def test_shared_pipe_contention_detection(self, tmp_audit_db, consistency_network):
        """Verify that multiple zones competing for capacity on a common conduit trigger contention warnings."""
        tank_id = consistency_network["tank_id"]
        dest_id = consistency_network["dest_id"]

        # Create a second zone also mapped to dest_id
        z2_id = db.create_allocation_zone(
            zone_name="Zone East Commercial",
            requested_demand_l=500000.0,
            essential_min_l=100000.0,
            fairness_weight=2.0,
            priority_category="commercial",
            status="active",
            node_id=dest_id,
        )

        inputs = [
            ZoneDemandInput(
                zone_id=consistency_network["zone_id"],
                zone_name="Sector East Zone",
                requested_demand=300000.0,
                essential_min=50000.0,
                fairness_weight=1.0,
                node_id=dest_id,
            ),
            ZoneDemandInput(
                zone_id=z2_id,
                zone_name="Zone East Commercial",
                requested_demand=500000.0,
                essential_min=100000.0,
                fairness_weight=2.0,
                node_id=dest_id,
            ),
        ]
        alloc_plan = allocate_water(800000.0, inputs)
        run_id = db.record_allocation_run(alloc_plan, notes="Contention Test Run")

        # 800,000 L over 60 min requires ~13,333 LPM, but Branch East only supports 4,000 LPM
        aid, plan = distribution.service.evaluate_and_record_run_distribution(
            run_id=run_id,
            source_node_id=tank_id,
            duration_minutes=60.0,
            routing_strategy="SHORTEST_PATH",
        )

        assert plan.shared_pipe_warnings is not None
        assert len(plan.shared_pipe_warnings) > 0
        contended_pipes = [w["pipe_id"] for w in plan.shared_pipe_warnings]
        assert consistency_network["pipe_branch"] in contended_pipes

    def test_viewing_historical_assessments_does_not_modify_database(self, tmp_audit_db, consistency_network):
        """Verify that viewing historical assessment detail routes is strictly read-only."""
        dest_id = consistency_network["dest_id"]
        zone_id = consistency_network["zone_id"]
        tank_id = consistency_network["tank_id"]

        inputs = [
            ZoneDemandInput(
                zone_id=zone_id,
                zone_name="Sector East Zone",
                requested_demand=1000.0,
                essential_min=200.0,
                fairness_weight=1.0,
                node_id=dest_id,
            )
        ]
        alloc_plan = allocate_water(800.0, inputs)
        run_id = db.record_allocation_run(alloc_plan, notes="Read Only View Test")
        aid, _ = distribution.service.evaluate_and_record_run_distribution(
            run_id=run_id,
            source_node_id=tank_id,
            duration_minutes=60.0,
            routing_strategy="SHORTEST_PATH",
        )

        with db._connect() as conn:
            pre_assess_count = conn.execute("SELECT COUNT(*) FROM delivery_assessments").fetchone()[0]
            pre_zone_count = conn.execute("SELECT COUNT(*) FROM delivery_zone_assessments").fetchone()[0]

        app.config["TESTING"] = True
        client = app.test_client()

        resp = client.get(f"/distribution/assessments/{aid}")
        assert resp.status_code == 200

        with db._connect() as conn:
            post_assess_count = conn.execute("SELECT COUNT(*) FROM delivery_assessments").fetchone()[0]
            post_zone_count = conn.execute("SELECT COUNT(*) FROM delivery_zone_assessments").fetchone()[0]

        assert pre_assess_count == post_assess_count
        assert pre_zone_count == post_zone_count


# ── PART 5: HISTORICAL MUTATION TESTING (THE 10 OPERATIONAL MUTATIONS) ────────


class TestHistoricalMutationInvariants:
    """Execute all 10 operational mutations sequentially and verify immutable snapshot preservation."""

    def test_all_ten_operational_mutations_preserve_historical_records(
        self, tmp_audit_db, consistency_network
    ):
        tank_id = consistency_network["tank_id"]
        mid_id = consistency_network["mid_id"]
        dest_id = consistency_network["dest_id"]
        p_alpha = consistency_network["pipe_low_cost"]
        p_branch = consistency_network["pipe_branch"]
        s_flow = consistency_network["sensor_flow"]
        zone_id = consistency_network["zone_id"]

        # Initial baseline records
        _ = db.record_sensor_reading(s_flow, 120.0, scenario="normal")
        ev_id, _ = db.record_leak_event(p_alpha, "high", "Burst detected on Alpha")
        task_id, _ = db.create_or_update_repair_task(ev_id, p_alpha, 80.0, "HIGH")

        inputs = [
            ZoneDemandInput(
                zone_id=zone_id,
                zone_name="Sector East Zone",
                requested_demand=1000.0,
                essential_min=200.0,
                fairness_weight=1.0,
                node_id=dest_id,
            )
        ]
        alloc_plan = allocate_water(750.0, inputs)
        run_id = db.record_allocation_run(alloc_plan, notes="Initial Allocation")
        aid, plan = distribution.service.evaluate_and_record_run_distribution(
            run_id=run_id,
            source_node_id=tank_id,
            duration_minutes=60.0,
            routing_strategy="SHORTEST_PATH",
        )

        pre_alloc = dict(db.get_allocations_for_run(run_id)[0])
        pre_assess = dict(db.get_delivery_assessment_by_id(aid))
        pre_zone_assess = dict(db.get_delivery_zone_assessments(aid)[0])

        # ── EXECUTE 10 OPERATIONAL MUTATIONS ──────────────────────────────────
        # 1. Rename network node
        db.update_node(tank_id, "Main Reservoir RENAMED", "tank", "North Sector", "active")
        db.update_node(dest_id, "East Consumer RENAMED", "junction", "East Sector", "active")

        # 2. Rename pipe
        db.update_pipe(p_alpha, "Trunk Alpha RENAMED", tank_id, mid_id, "1000", "2000", "1.0", "active")

        # 3. Deactivate pipe
        db.update_pipe(p_branch, "Branch East", mid_id, dest_id, "500", "4000", "1.0", "inactive")

        # 4. Deactivate sensor
        db.update_sensor(s_flow, "FLOW-TRUNK-A", p_alpha, "flow_meter", "L/min", "inactive")

        # 5. Rename allocation zone
        db.update_allocation_zone(
            zone_id=zone_id,
            zone_name="Sector East Zone RENAMED",
            requested_demand_l=1000.0,
            essential_min_l=200.0,
            fairness_weight=1.0,
            priority_category="standard",
            status="active",
            node_id=dest_id,
        )

        # 6. Change zone operational node mapping
        new_node = db.create_node("New Outpost", "junction", "South", "active")
        db.update_allocation_zone(
            zone_id=zone_id,
            zone_name="Sector East Zone RENAMED",
            requested_demand_l=1000.0,
            essential_min_l=200.0,
            fairness_weight=1.0,
            priority_category="standard",
            status="active",
            node_id=new_node,
        )

        # 7. Change zone demand
        db.update_allocation_zone(
            zone_id=zone_id,
            zone_name="Sector East Zone RENAMED",
            requested_demand_l=8888.0,
            essential_min_l=1000.0,
            fairness_weight=3.0,
            priority_category="critical",
            status="active",
            node_id=new_node,
        )

        # 8. Resolve leak event
        db.update_leak_event_status(ev_id, "resolved")

        # 9. Complete repair task
        db.update_repair_task_status(task_id, "Completed", action_description="Fixed completely")

        # 10. Retrieve previously recorded historical records
        post_alloc = dict(db.get_allocations_for_run(run_id)[0])
        post_assess = dict(db.get_delivery_assessment_by_id(aid))
        post_zone_assess = dict(db.get_delivery_zone_assessments(aid)[0])
        post_ev = dict(db.get_leak_event_by_id(ev_id))
        post_task = dict(db.get_repair_task_by_id(task_id))

        # ── VERIFY INVARIANT IMMUTABILITY ─────────────────────────────────────
        # Allocation snapshot preserved
        assert post_alloc["zone_name_snapshot"] == "Sector East Zone"
        assert post_alloc["allocated_amount_l"] == pre_alloc["allocated_amount_l"]
        assert post_alloc["destination_node_id"] == dest_id

        # Assessment snapshot preserved
        assert post_assess["source_name_snapshot"] == "Main Reservoir"
        assert post_assess["total_allocated_l"] == pre_assess["total_allocated_l"]
        assert post_assess["total_deliverable_l"] == pre_assess["total_deliverable_l"]

        # Zone assessment snapshot preserved
        assert post_zone_assess["zone_name_snapshot"] == "Sector East Zone"
        assert post_zone_assess["destination_name_snapshot"] == "East Consumer Node"

        # Lifecycle transitions verified
        assert post_ev["status"] == "resolved"
        assert post_task["status"] == "Completed"


# ── PART 6: TRANSACTION AND FAILURE CONSISTENCY ───────────────────────────────


class TestTransactionAndFailureConsistency:
    """Validate atomic rollback on simulated database failures across all critical workflows."""

    def test_allocation_run_creation_rollback_on_failure(self, tmp_audit_db, consistency_network):
        """Verify allocation run recording rolls back completely if any zone insert fails."""
        zone_id = consistency_network["zone_id"]
        inputs = [
            ZoneDemandInput(
                zone_id=zone_id,
                zone_name="Sector East Zone",
                requested_demand=1000.0,
                essential_min=200.0,
                fairness_weight=1.0,
                node_id=consistency_network["dest_id"],
            )
        ]
        alloc_plan = allocate_water(800.0, inputs)

        with db._connect() as conn:
            pre_runs = conn.execute("SELECT COUNT(*) FROM allocation_runs").fetchone()[0]
            pre_allocs = conn.execute("SELECT COUNT(*) FROM water_allocations").fetchone()[0]

        # Trigger failure using intentional fail_on_zone_id hook
        with pytest.raises(RuntimeError, match="Simulated.*failure"):
            db.record_allocation_run(alloc_plan, fail_on_zone_id=zone_id)

        with db._connect() as conn:
            post_runs = conn.execute("SELECT COUNT(*) FROM allocation_runs").fetchone()[0]
            post_allocs = conn.execute("SELECT COUNT(*) FROM water_allocations").fetchone()[0]

        assert pre_runs == post_runs
        assert pre_allocs == post_allocs

    def test_distribution_assessment_rollback_on_failure(self, tmp_audit_db, consistency_network):
        """Verify distribution assessment persistence rolls back completely if any zone insert fails."""
        zone_id = consistency_network["zone_id"]
        tank_id = consistency_network["tank_id"]
        dest_id = consistency_network["dest_id"]

        inputs = [
            ZoneDemandInput(
                zone_id=zone_id,
                zone_name="Sector East Zone",
                requested_demand=1000.0,
                essential_min=200.0,
                fairness_weight=1.0,
                node_id=dest_id,
            )
        ]
        alloc_plan = allocate_water(800.0, inputs)
        run_id = db.record_allocation_run(alloc_plan, notes="Rollback Test Run")

        graph = load_graph_from_db(db_module=db, directed=False)
        plan = distribution.engine.assess_distribution(
            graph=graph,
            source_node_id=tank_id,
            zones=[
                distribution.engine.ZoneDeliveryInput(
                    zone_id=zone_id,
                    zone_name="Sector East Zone",
                    destination_node_id=dest_id,
                    allocated_volume_l=800.0,
                )
            ],
            duration_minutes=60.0,
        )

        with db._connect() as conn:
            pre_assess = conn.execute("SELECT COUNT(*) FROM delivery_assessments").fetchone()[0]
            pre_zones = conn.execute("SELECT COUNT(*) FROM delivery_zone_assessments").fetchone()[0]

        with pytest.raises(RuntimeError, match="Simulated.*failure"):
            db.record_distribution_assessment(plan, run_id=run_id, fail_on_zone_id=zone_id)

        with db._connect() as conn:
            post_assess = conn.execute("SELECT COUNT(*) FROM delivery_assessments").fetchone()[0]
            post_zones = conn.execute("SELECT COUNT(*) FROM delivery_zone_assessments").fetchone()[0]

        assert pre_assess == post_assess
        assert pre_zones == post_zones

    def test_sensor_reading_batch_rollback_on_failure(self, tmp_audit_db, consistency_network):
        """Verify record_sensor_readings_batch rolls back all readings if an error occurs mid-batch."""
        s_flow = consistency_network["sensor_flow"]
        batch = [
            {"sensor_id": s_flow, "reading_value": 100.0, "scenario": "normal"},
            {"sensor_id": s_flow, "reading_value": 110.0, "scenario": "normal"},
            {"sensor_id": s_flow, "reading_value": 120.0, "scenario": "normal"},
        ]

        with db._connect() as conn:
            pre_readings = conn.execute("SELECT COUNT(*) FROM sensor_readings").fetchone()[0]

        # Fail at index 1
        with pytest.raises(RuntimeError, match="Simulated failure at batch index 1"):
            db.record_sensor_readings_batch(batch, fail_on_index=1)

        with db._connect() as conn:
            post_readings = conn.execute("SELECT COUNT(*) FROM sensor_readings").fetchone()[0]

        # Zero partial rows left behind
        assert pre_readings == post_readings
