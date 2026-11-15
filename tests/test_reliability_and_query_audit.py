"""
tests/test_reliability_and_query_audit.py – Targeted Reliability, Concurrency & Query Audit Test Suite.

Validates Day 10 Phase 2 requirements:
1. SQLite connection timeout (30.0s) and busy-lock resilience under concurrent writes.
2. WAL journal mode configuration and persistence without redundant per-connection execution.
3. Distribution run history query count audit (verifies O(1) query count invariance as assessment count grows).
4. Authoritative historical source name snapshot preservation after operational node rename and node deletion.
5. Legacy allocation handling: zones without destination_node_id strictly remain UNMAPPED with clear UI notice.
"""

from __future__ import annotations

import concurrent.futures
import os
import random
import shutil
import tempfile
import threading
import time
import pytest

import database as db
import distribution.service
from allocation.engine import allocate_water, ZoneDemandInput
from app import app


@pytest.fixture
def tmp_audit_db():
    """Create an isolated temporary SQLite database for reliability and audit tests."""
    temp_dir = tempfile.mkdtemp()
    temp_db_path = os.path.join(temp_dir, "test_audit.db")

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
def sample_network(tmp_audit_db):
    """Seed a basic network with source, junctions, pipe, and allocation zone."""
    tank_id = db.create_node("North Reservoir", "tank", "North Sector", "active")
    junc_id = db.create_node("Central Junction", "junction", "Central Sector", "active")
    dest_id = db.create_node("East Outpost", "junction", "East Sector", "active")

    pipe1 = db.create_pipe("Main Trunk", tank_id, junc_id, "1000", "5000", "1.0", "active")
    pipe2 = db.create_pipe("Branch East", junc_id, dest_id, "800", "3500", "1.0", "active")

    sensor_flow = db.create_sensor("FLOW-NORTH", pipe1, "flow_meter", "L/min", "active")
    sensor_press = db.create_sensor("PRESS-NORTH", pipe1, "pressure_sensor", "bar", "active")

    zone_id = db.create_allocation_zone(
        zone_name="Zone East",
        requested_demand_l=1000.0,
        essential_min_l=200.0,
        fairness_weight=1.0,
        priority_category="standard",
        status="active",
        node_id=dest_id,
    )

    return {
        "tank_id": tank_id,
        "junc_id": junc_id,
        "dest_id": dest_id,
        "pipe1": pipe1,
        "pipe2": pipe2,
        "sensor_flow": sensor_flow,
        "sensor_press": sensor_press,
        "zone_id": zone_id,
    }


# ── PART 1: SQLITE CONNECTION & CONCURRENCY TESTS ─────────────────────────────


class TestSQLiteConcurrencyReliability:
    """Validate timeout configuration, busy-lock handling, and concurrent execution."""

    def test_sqlite_connect_timeout_configured_to_30s(self, tmp_audit_db):
        """Verify _connect() uses a 30-second timeout for lock resilience."""
        conn = db._connect()
        try:
            # Under SQLite's Python API, busy_timeout is set when timeout=30.0 is passed
            # We can verify the connection is active and can query PRAGMA busy_timeout
            cur = conn.execute("PRAGMA busy_timeout")
            timeout_ms = cur.fetchone()[0]
            assert timeout_ms == 30000, f"Expected 30000ms (30s) timeout, got {timeout_ms}ms"
        finally:
            conn.close()

    def test_wal_journal_mode_persists_without_per_connection_pragma(self, tmp_audit_db):
        """Verify that journal_mode remains 'wal' on new connections without reapplying the pragma."""
        conn1 = db._connect()
        try:
            cur1 = conn1.execute("PRAGMA journal_mode")
            mode1 = cur1.fetchone()[0].lower()
            assert mode1 == "wal"
        finally:
            conn1.close()

        # Connect again, verifying WAL mode is persistent
        conn2 = db._connect()
        try:
            cur2 = conn2.execute("PRAGMA journal_mode")
            mode2 = cur2.fetchone()[0].lower()
            assert mode2 == "wal"
        finally:
            conn2.close()

    def test_concurrent_read_write_operations(self, tmp_audit_db, sample_network):
        """
        Execute concurrent sensor ingestion, allocation runs, distribution assessments,
        and read queries across multiple threads without database-locked failures.
        """
        s_flow = sample_network["sensor_flow"]
        s_press = sample_network["sensor_press"]
        tank_id = sample_network["tank_id"]
        dest_id = sample_network["dest_id"]
        zone_id = sample_network["zone_id"]

        errors: list[tuple[str, str]] = []
        stop_event = threading.Event()

        def sensor_worker(worker_id: int):
            for i in range(50):
                if stop_event.is_set():
                    break
                try:
                    sid = s_flow if i % 2 == 0 else s_press
                    val = random.uniform(100.0, 500.0) if sid == s_flow else random.uniform(30.0, 60.0)
                    stype = "flow_meter" if sid == s_flow else "pressure_sensor"
                    unit = "L/min" if sid == s_flow else "bar"
                    db.record_sensor_reading(
                        sensor_id=sid,
                        reading_value=val,
                        reading_type=stype,
                        unit=unit,
                        is_simulated=True,
                        scenario="normal",
                    )
                except Exception as exc:
                    errors.append((f"sensor_{worker_id}", f"{type(exc).__name__}: {exc}"))

        def allocation_distribution_worker(worker_id: int):
            for i in range(8):
                if stop_event.is_set():
                    break
                try:
                    inputs = [
                        ZoneDemandInput(
                            zone_id=zone_id,
                            zone_name="Zone East",
                            requested_demand=1000.0,
                            essential_min=200.0,
                            fairness_weight=1.0,
                            node_id=dest_id,
                        )
                    ]
                    alloc_plan = allocate_water(800.0, inputs)
                    run_id = db.record_allocation_run(alloc_plan, notes=f"Worker {worker_id} Run {i}")

                    strat = "SHORTEST_PATH" if i % 2 == 0 else "WIDEST_PATH"
                    aid, plan = distribution.service.evaluate_and_record_run_distribution(
                        run_id=run_id,
                        source_node_id=tank_id,
                        duration_minutes=60.0,
                        routing_strategy=strat,
                        notes=f"Worker {worker_id} Assessment {i}",
                    )
                    assert aid > 0
                    assert plan.total_deliverable_l > 0
                except Exception as exc:
                    errors.append((f"alloc_dist_{worker_id}", f"{type(exc).__name__}: {exc}"))

        def reader_worker(worker_id: int):
            for _ in range(30):
                if stop_event.is_set():
                    break
                try:
                    _ = db.get_all_nodes()
                    _ = db.get_all_pipes()
                    _ = db.get_all_allocation_runs(limit=10)
                    _ = db.get_all_delivery_assessments(limit=10)
                except Exception as exc:
                    errors.append((f"reader_{worker_id}", f"{type(exc).__name__}: {exc}"))

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = [
                executor.submit(sensor_worker, 1),
                executor.submit(sensor_worker, 2),
                executor.submit(allocation_distribution_worker, 1),
                executor.submit(allocation_distribution_worker, 2),
                executor.submit(reader_worker, 1),
                executor.submit(reader_worker, 2),
            ]
            for f in concurrent.futures.as_completed(futures):
                f.result()

        assert len(errors) == 0, f"Encountered concurrency errors: {errors}"

        # Verify database integrity after concurrency test
        with db._connect() as conn:
            integrity = conn.execute("PRAGMA integrity_check").fetchall()
            assert [row[0] for row in integrity] == ["ok"]
            fk_check = conn.execute("PRAGMA foreign_key_check").fetchall()
            assert len(fk_check) == 0


# ── PART 2: DISTRIBUTION HISTORY QUERY OPTIMIZATION AUDIT ─────────────────────


class TestDistributionHistoryQueryAudit:
    """Validate query counts and historical snapshot immutability for distribution history."""

    def test_query_count_does_not_grow_linearly_with_assessments(self, tmp_audit_db, sample_network):
        """
        Verify that rendering /distribution/runs/<run_id> does NOT exhibit an N+1 query pattern.
        The query count must remain invariant (O(1)) as the number of recorded assessments increases.
        """
        tank_id = sample_network["tank_id"]
        dest_id = sample_network["dest_id"]
        zone_id = sample_network["zone_id"]

        inputs = [
            ZoneDemandInput(
                zone_id=zone_id,
                zone_name="Zone East",
                requested_demand=1000.0,
                essential_min=200.0,
                fairness_weight=1.0,
                node_id=dest_id,
            )
        ]
        alloc_res = allocate_water(800.0, inputs)
        run_id = db.record_allocation_run(alloc_res, notes="Benchmark Run")

        query_log: list[str] = []
        orig_connect = db._connect

        def instrumented_connect():
            conn = orig_connect()
            conn.set_trace_callback(lambda stmt: query_log.append(stmt.strip()))
            return conn

        db._connect = instrumented_connect

        app.config["TESTING"] = True
        client = app.test_client()

        try:
            # Measure query counts at 1, 5, 10, and 20 assessments
            assessment_checkpoints = [1, 5, 10, 20]
            measured_user_queries: dict[int, int] = {}

            for target_count in assessment_checkpoints:
                current_count = len(db.get_delivery_assessments_for_run(run_id))
                for i in range(current_count, target_count):
                    strat = "SHORTEST_PATH" if i % 2 == 0 else "WIDEST_PATH"
                    distribution.service.evaluate_and_record_run_distribution(
                        run_id=run_id,
                        source_node_id=tank_id,
                        duration_minutes=60.0,
                        routing_strategy=strat,
                        notes=f"Benchmark Assessment #{i}",
                    )

                query_log.clear()
                resp = client.get(f"/distribution/runs/{run_id}")
                assert resp.status_code == 200

                user_queries = [
                    q for q in query_log
                    if not q.startswith("PRAGMA") and q not in ("BEGIN", "COMMIT", "ROLLBACK")
                ]
                measured_user_queries[target_count] = len(user_queries)

            # Invariant: Query count with 20 assessments must equal query count with 1 assessment
            assert measured_user_queries[20] == measured_user_queries[1], (
                f"Query count scaled with assessments: {measured_user_queries}"
            )
            # Exactly 4 user queries: run lookup, assessments list, allocations list, active nodes list
            assert measured_user_queries[20] <= 5
        finally:
            db._connect = orig_connect

    def test_historical_source_name_remains_unchanged_after_operational_renaming(
        self, tmp_audit_db, sample_network
    ):
        """
        Verify that renaming the operational source node does NOT alter the historical
        source_name_snapshot displayed on run history and assessment details.
        """
        tank_id = sample_network["tank_id"]
        dest_id = sample_network["dest_id"]
        zone_id = sample_network["zone_id"]

        inputs = [
            ZoneDemandInput(
                zone_id=zone_id,
                zone_name="Zone East",
                requested_demand=1000.0,
                essential_min=200.0,
                fairness_weight=1.0,
                node_id=dest_id,
            )
        ]
        alloc_res = allocate_water(800.0, inputs)
        run_id = db.record_allocation_run(alloc_res, notes="Snapshot Invariance Test")

        aid, _ = distribution.service.evaluate_and_record_run_distribution(
            run_id=run_id,
            source_node_id=tank_id,
            duration_minutes=60.0,
            routing_strategy="SHORTEST_PATH",
        )

        # Operational rename of the source tank
        db.update_node(tank_id, "Renamed Mega Reservoir", "tank", "North Sector", "active")

        app.config["TESTING"] = True
        client = app.test_client()

        # 1. Check /distribution/runs/<run_id>
        resp_hist = client.get(f"/distribution/runs/{run_id}")
        assert resp_hist.status_code == 200
        # Historical snapshot 'North Reservoir' must be preserved in assessment row
        assert "North Reservoir" in resp_hist.get_data(as_text=True)

        # 2. Check /distribution/assessments/<aid>
        resp_det = client.get(f"/distribution/assessments/{aid}")
        assert resp_det.status_code == 200
        assert "North Reservoir" in resp_det.get_data(as_text=True)

    def test_historical_source_name_remains_available_after_source_node_deletion(
        self, tmp_audit_db, sample_network
    ):
        """
        Verify that deleting the source node sets foreign key to NULL but preserves
        the historical source_name_snapshot on assessments without errors.
        """
        tank_id = sample_network["tank_id"]
        junc_id = sample_network["junc_id"]
        dest_id = sample_network["dest_id"]
        zone_id = sample_network["zone_id"]

        inputs = [
            ZoneDemandInput(
                zone_id=zone_id,
                zone_name="Zone East",
                requested_demand=1000.0,
                essential_min=200.0,
                fairness_weight=1.0,
                node_id=dest_id,
            )
        ]
        alloc_res = allocate_water(800.0, inputs)
        run_id = db.record_allocation_run(alloc_res, notes="Node Deletion Snapshot Test")

        aid, _ = distribution.service.evaluate_and_record_run_distribution(
            run_id=run_id,
            source_node_id=tank_id,
            duration_minutes=60.0,
            routing_strategy="WIDEST_PATH",
        )

        # To delete the tank node, first delete connecting pipe and sensor
        with db._connect() as conn:
            conn.execute("DELETE FROM sensor_readings")
            conn.execute("DELETE FROM sensors WHERE pipe_id = ?", (sample_network["pipe1"],))
            conn.execute("DELETE FROM pipes WHERE id = ?", (sample_network["pipe1"],))
            conn.commit()

        db.delete_node(tank_id)

        # Assessment record must now have source_node_id = NULL but source_name_snapshot = 'North Reservoir'
        assessment = db.get_delivery_assessment_by_id(aid)
        assert assessment is not None
        assert assessment["source_node_id"] is None
        assert assessment["source_name_snapshot"] == "North Reservoir"

        # Verify UI rendering does not crash
        app.config["TESTING"] = True
        client = app.test_client()

        resp_hist = client.get(f"/distribution/runs/{run_id}")
        assert resp_hist.status_code == 200
        assert "North Reservoir" in resp_hist.get_data(as_text=True)

        resp_det = client.get(f"/distribution/assessments/{aid}")
        assert resp_det.status_code == 200
        assert "North Reservoir" in resp_det.get_data(as_text=True)

    def test_existing_assessment_details_remain_unchanged(self, tmp_audit_db, sample_network):
        """Verify that stored distribution assessment details match exactly and are immutable."""
        tank_id = sample_network["tank_id"]
        dest_id = sample_network["dest_id"]
        zone_id = sample_network["zone_id"]

        inputs = [
            ZoneDemandInput(
                zone_id=zone_id,
                zone_name="Zone East",
                requested_demand=1000.0,
                essential_min=200.0,
                fairness_weight=1.0,
                node_id=dest_id,
            )
        ]
        alloc_res = allocate_water(800.0, inputs)
        run_id = db.record_allocation_run(alloc_res, notes="Immutability Detail Check")

        aid, plan = distribution.service.evaluate_and_record_run_distribution(
            run_id=run_id,
            source_node_id=tank_id,
            duration_minutes=60.0,
            routing_strategy="SHORTEST_PATH",
        )

        assessment = db.get_delivery_assessment_by_id(aid)
        assert assessment["total_allocated_l"] == float(plan.total_allocated_l)
        assert assessment["total_deliverable_l"] == float(plan.total_deliverable_l)
        assert assessment["delivery_efficiency_pct"] == float(plan.overall_delivery_efficiency_pct)
        assert assessment["routing_strategy"] == "SHORTEST_PATH"

        zones = db.get_delivery_zone_assessments(aid)
        assert len(zones) == 1
        assert zones[0]["zone_name"] == "Zone East"
        assert zones[0]["destination_node_name"] == "East Outpost"


# ── PART 3: LEGACY ALLOCATION UNMAPPED HANDLING TESTS ─────────────────────────


class TestLegacyAllocationUnmappedHandling:
    """Validate explicit UNMAPPED handling and UI notices for legacy allocation runs."""

    def test_legacy_allocation_without_destination_remains_unmapped(self, tmp_audit_db, sample_network):
        """
        Verify that an allocation run created without destination_node_id (simulating legacy pre-Day 9 data)
        strictly evaluates as UNMAPPED and does not guess current operational mappings.
        """
        tank_id = sample_network["tank_id"]
        zone_id = sample_network["zone_id"]

        # Create allocation run and manually set destination_node_id = NULL in water_allocations
        inputs = [
            ZoneDemandInput(
                zone_id=zone_id,
                zone_name="Legacy Unmapped Zone",
                requested_demand=1000.0,
                essential_min=200.0,
                fairness_weight=1.0,
                node_id=None,
            )
        ]
        alloc_res = allocate_water(600.0, inputs)
        run_id = db.record_allocation_run(alloc_res, notes="Pre-Day 9 Legacy Run")

        with db._connect() as conn:
            conn.execute(
                "UPDATE water_allocations SET destination_node_id = NULL WHERE run_id = ?",
                (run_id,),
            )
            conn.commit()

        # Execute distribution assessment
        aid, plan = distribution.service.evaluate_and_record_run_distribution(
            run_id=run_id,
            source_node_id=tank_id,
            duration_minutes=60.0,
            routing_strategy="SHORTEST_PATH",
        )

        zone_outcomes = plan.zone_results
        assert len(zone_outcomes) == 1
        assert zone_outcomes[0].feasibility_status == "UNMAPPED"
        assert zone_outcomes[0].deliverable_volume_l == 0.0

        # Verify UI renders the unmapped informational notice
        app.config["TESTING"] = True
        client = app.test_client()

        resp_det = client.get(f"/distribution/assessments/{aid}")
        assert resp_det.status_code == 200
        html = resp_det.get_data(as_text=True)
        assert "Historical Destination Mapping Notice" in html
        assert "UNMAPPED" in html

        resp_hist = client.get(f"/distribution/runs/{run_id}")
        assert resp_hist.status_code == 200
        html_hist = resp_hist.get_data(as_text=True)
        assert "Historical Allocation Mapping Notice" in html_hist
