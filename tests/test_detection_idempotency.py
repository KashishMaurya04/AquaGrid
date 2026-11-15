"""
tests/test_detection_idempotency.py – Tests for Leak Detection Idempotency and Post-Resolution Telemetry Protection.

Validates all 10 architectural protection scenarios:
1. GET /leak-detection is strictly read-only (creates 0 records).
2. Multiple refreshes of GET /leak-detection create 0 duplicate events.
3. Explicit POST /leak-detection/run identifies and records genuine anomalies.
4. Repeated POST /leak-detection/run while an event is open updates in place without duplicating.
5. Resolving an event and immediately rerunning detection against stale telemetry returns
   insufficient_data and creates NO new event.
6. Adding fewer than 5 post-resolution readings returns insufficient_data and creates NO new event.
7. Adding 5 or more fresh post-resolution anomaly readings creates a legitimate Leak Event #2.
8. That new event receives exactly one associated repair task.
9. Lifecycle audit entries remain intact and reflect true transitions without duplicates.
10. Priority queue Max Heap synchronization properly orders active tasks without duplicates.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone, timedelta
import pytest

from repair_queue.max_heap import MaxHeap, RepairTask
from repair_queue.manager import PriorityQueueManager
from detection.engine import LeakDetector, DEFAULT_DETECTOR_CONFIG
import database as db_module
import app as app_module


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture()
def tmp_db(monkeypatch, tmp_path):
    """Isolate database tests with a fresh SQLite DB in a temporary directory."""
    test_db = str(tmp_path / "test_idempotency_aquagrid.db")
    monkeypatch.setattr(db_module, "INSTANCE_DIR", str(tmp_path))
    monkeypatch.setattr(db_module, "DB_PATH", test_db)
    monkeypatch.setattr(app_module, "db", db_module)
    db_module.init_db()
    return db_module


@pytest.fixture()
def app_client(tmp_db):
    """Flask test client wired to isolated database."""
    app_module.app.config["TESTING"] = True
    app_module.app.config["WTF_CSRF_ENABLED"] = False
    app_module.app.config["SECRET_KEY"] = "test-idempotency-key"

    with app_module.app.test_client() as client:
        yield client


@pytest.fixture()
def setup_telemetry_network(tmp_db):
    """
    Sets up a distribution pipe with dual active sensors (flow meter + pressure sensor),
    10 verified normal baseline readings, and 5 leak-scenario anomaly readings.
    """
    n1 = tmp_db.create_node("Reservoir Central", "tank", "Central Zone", "active")
    n2 = tmp_db.create_node("District Junction 1", "junction", "Zone 1", "active")
    pipe_id = tmp_db.create_pipe("Main-Trunk-1", n1, n2, "250.0", "1500.0", "1.0", "active")

    fm_id = tmp_db.create_sensor("FM-Main-1", pipe_id, "flow_meter", "L/min", "active")
    ps_id = tmp_db.create_sensor("PS-Main-1", pipe_id, "pressure_sensor", "psi", "active")

    # 10 verified normal baseline readings: flow ~ 350 L/min, pressure ~ 52 psi
    base_time = datetime(2026, 11, 9, 10, 0, 0, tzinfo=timezone.utc)
    for i in range(10):
        t = (base_time + timedelta(minutes=i)).isoformat()
        tmp_db.record_sensor_reading(
            sensor_id=fm_id,
            reading_value=350.0 + (i % 3) * 2.0,
            recorded_at=t,
            reading_type="flow_meter",
            unit="L/min",
            is_simulated=True,
            scenario="normal",
        )
        tmp_db.record_sensor_reading(
            sensor_id=ps_id,
            reading_value=52.0 - (i % 2) * 0.5,
            recorded_at=t,
            reading_type="pressure_sensor",
            unit="psi",
            is_simulated=True,
            scenario="normal",
        )

    # 5 leak anomaly readings: flow surge ~ 600 L/min (+71%), pressure drop ~ 25 psi (-52%)
    for i in range(5):
        t = (base_time + timedelta(minutes=10 + i)).isoformat()
        tmp_db.record_sensor_reading(
            sensor_id=fm_id,
            reading_value=600.0 + i * 5.0,
            recorded_at=t,
            reading_type="flow_meter",
            unit="L/min",
            is_simulated=True,
            scenario="leak",
        )
        tmp_db.record_sensor_reading(
            sensor_id=ps_id,
            reading_value=25.0 - i * 0.5,
            recorded_at=t,
            reading_type="pressure_sensor",
            unit="psi",
            is_simulated=True,
            scenario="leak",
        )

    return {
        "nodes": [n1, n2],
        "pipe_id": pipe_id,
        "sensors": {"flow": fm_id, "pressure": ps_id},
        "base_time": base_time,
    }


# ==============================================================================
# 10 Phased Protection Tests
# ==============================================================================

class TestLeakDetectionIdempotency:
    """Comprehensive test suite for Phase 1 - Phase 5 requirements."""

    def test_01_get_leak_detection_is_strictly_read_only(self, tmp_db, app_client, setup_telemetry_network):
        """Scenario 1: Visiting GET /leak-detection creates 0 records in leak_events, repair_tasks, and audit logs."""
        assert tmp_db.count_total_leak_events() == 0
        assert tmp_db.count_total_repair_tasks() == 0
        assert len(tmp_db.get_lifecycle_audit_logs()) == 0

        res = app_client.get("/leak-detection")
        assert res.status_code == 200

        # Must NOT create any records
        assert tmp_db.count_total_leak_events() == 0
        assert tmp_db.count_total_repair_tasks() == 0
        assert len(tmp_db.get_lifecycle_audit_logs()) == 0

    def test_02_multiple_refreshes_of_get_create_zero_events(self, tmp_db, app_client, setup_telemetry_network):
        """Scenario 2: Refreshing GET /leak-detection multiple times creates 0 duplicate events."""
        for _ in range(5):
            res = app_client.get("/leak-detection")
            assert res.status_code == 200

        assert tmp_db.count_total_leak_events() == 0
        assert tmp_db.count_total_repair_tasks() == 0
        assert len(tmp_db.get_lifecycle_audit_logs()) == 0

    def test_03_explicit_post_run_records_anomaly_and_redirects(self, tmp_db, app_client, setup_telemetry_network):
        """Scenario 3: Explicit POST /leak-detection/run identifies genuine anomalies and records them."""
        res = app_client.post("/leak-detection/run")
        # PRG redirect
        assert res.status_code == 302
        assert res.headers["Location"].endswith("/leak-detection")

        # Now an event is recorded
        assert tmp_db.count_total_leak_events() == 1
        open_events = tmp_db.get_open_leak_events()
        assert len(open_events) == 1
        ev = open_events[0]
        assert ev["severity"] in ("high", "critical")
        assert ev["status"] == "open"

        # Repair queue synchronized
        assert tmp_db.count_active_repair_tasks() == 1

    def test_04_repeated_post_run_updates_existing_open_event(self, tmp_db, app_client, setup_telemetry_network):
        """Scenario 4: Repeated POST /leak-detection/run while event is open updates in place without duplication."""
        app_client.post("/leak-detection/run")
        assert tmp_db.count_total_leak_events() == 1
        assert tmp_db.count_total_repair_tasks() == 1

        first_audit_count = len(tmp_db.get_lifecycle_audit_logs())

        # Repeated POST
        app_client.post("/leak-detection/run")
        assert tmp_db.count_total_leak_events() == 1
        assert tmp_db.count_total_repair_tasks() == 1

        # No duplicate audit log entries created when status did not change
        second_audit_count = len(tmp_db.get_lifecycle_audit_logs())
        assert second_audit_count == first_audit_count

    def test_05_resolving_event_with_stale_telemetry_creates_no_new_event(self, tmp_db, app_client, setup_telemetry_network):
        """
        Scenario 5: Resolving an event and immediately rerunning detection against only old telemetry
        returns insufficient_data and does NOT create a new event.
        """
        pipe_id = setup_telemetry_network["pipe_id"]

        # 1. Run detection to create Event #1
        app_client.post("/leak-detection/run")
        events = tmp_db.get_all_leak_events()
        assert len(events) == 1
        ev_1 = events[0]
        ev_1_id = ev_1["event_id"]

        # 2. Mark Event #1 as resolved
        res_time = "2026-11-09T10:30:00+00:00"
        tmp_db.update_leak_event_status(ev_1_id, "resolved")
        ev_1_updated = tmp_db.get_leak_event_by_id(ev_1_id)
        assert ev_1_updated["status"] == "resolved"
        assert ev_1_updated["resolved_at"] is not None

        # 3. Re-run detection against stale telemetry (no readings recorded after 10:30)
        app_client.post("/leak-detection/run")

        # 4. Must NOT create Event #2!
        all_events_after = tmp_db.get_all_leak_events()
        assert len(all_events_after) == 1
        assert all_events_after[0]["event_id"] == ev_1_id

        # 5. Diagnostic must report insufficient_data for this pipe
        detector = LeakDetector()
        diags = detector.run_detection_network(tmp_db, auto_record=False)
        p_diag = next(d for d in diags if d.pipe_id == pipe_id)
        assert p_diag.status == "insufficient_data"
        assert "post-resolution" in p_diag.evidence.lower() or "insufficient" in p_diag.evidence.lower()

    def test_06_fewer_than_five_post_resolution_readings_creates_no_event(self, tmp_db, app_client, setup_telemetry_network):
        """Scenario 6: Adding fewer than 5 post-resolution readings returns insufficient_data and creates NO event."""
        pipe_id = setup_telemetry_network["pipe_id"]
        fm_id = setup_telemetry_network["sensors"]["flow"]
        ps_id = setup_telemetry_network["sensors"]["pressure"]

        # Create and resolve Event #1
        app_client.post("/leak-detection/run")
        ev_1_id = tmp_db.get_all_leak_events()[0]["event_id"]
        tmp_db.update_leak_event_status(ev_1_id, "resolved")
        resolved_ev = tmp_db.get_leak_event_by_id(ev_1_id)
        resolved_at = resolved_ev["resolved_at"]

        # Add only 3 fresh post-resolution readings (less than window_size of 5)
        res_dt = datetime.fromisoformat(resolved_at.replace("Z", "+00:00"))
        for i in range(1, 4):
            t = (res_dt + timedelta(minutes=i)).isoformat()
            tmp_db.record_sensor_reading(
                sensor_id=fm_id,
                reading_value=610.0,
                recorded_at=t,
                scenario="leak",
            )
            tmp_db.record_sensor_reading(
                sensor_id=ps_id,
                reading_value=24.0,
                recorded_at=t,
                scenario="leak",
            )

        # Run detection
        app_client.post("/leak-detection/run")

        # Must NOT create a new event
        assert tmp_db.count_total_leak_events() == 1

        # Check diagnostic
        detector = LeakDetector()
        diags = detector.run_detection_network(tmp_db, auto_record=False)
        p_diag = next(d for d in diags if d.pipe_id == pipe_id)
        assert p_diag.status == "insufficient_data"

    def test_07_five_or_more_fresh_anomaly_readings_creates_event_two(self, tmp_db, app_client, setup_telemetry_network):
        """Scenario 7: Adding 5 or more fresh post-resolution anomaly readings creates a legitimate Leak Event #2."""
        pipe_id = setup_telemetry_network["pipe_id"]
        fm_id = setup_telemetry_network["sensors"]["flow"]
        ps_id = setup_telemetry_network["sensors"]["pressure"]

        # 1. Create and resolve Event #1
        app_client.post("/leak-detection/run")
        ev_1_id = tmp_db.get_all_leak_events()[0]["event_id"]
        tmp_db.update_leak_event_status(ev_1_id, "resolved")
        resolved_ev = tmp_db.get_leak_event_by_id(ev_1_id)
        resolved_at = resolved_ev["resolved_at"]
        res_dt = datetime.fromisoformat(resolved_at.replace("Z", "+00:00"))

        # 2. Add 5 fresh post-resolution anomaly readings (>= window_size, >= 3 consecutive)
        for i in range(1, 6):
            t = (res_dt + timedelta(minutes=i)).isoformat()
            tmp_db.record_sensor_reading(
                sensor_id=fm_id,
                reading_value=620.0 + i,
                recorded_at=t,
                scenario="leak",
            )
            tmp_db.record_sensor_reading(
                sensor_id=ps_id,
                reading_value=24.0 - i * 0.2,
                recorded_at=t,
                scenario="leak",
            )

        # 3. Explicit POST to run detection
        app_client.post("/leak-detection/run")

        # 4. Now a fresh event #2 is legitimately recorded
        all_events = tmp_db.get_all_leak_events()
        assert len(all_events) == 2
        newest = all_events[0]
        oldest = all_events[1]

        assert oldest["event_id"] == ev_1_id
        assert oldest["status"] == "resolved"
        assert newest["event_id"] != ev_1_id
        assert newest["status"] == "open"
        assert newest["pipe_id"] == pipe_id

    def test_08_new_event_receives_exactly_one_repair_task(self, tmp_db, app_client, setup_telemetry_network):
        """Scenario 8: That new event receives exactly one associated repair task."""
        pipe_id = setup_telemetry_network["pipe_id"]
        fm_id = setup_telemetry_network["sensors"]["flow"]
        ps_id = setup_telemetry_network["sensors"]["pressure"]

        # Create and resolve Event #1
        app_client.post("/leak-detection/run")
        ev_1_id = tmp_db.get_all_leak_events()[0]["event_id"]
        tmp_db.update_leak_event_status(ev_1_id, "resolved")
        resolved_at = tmp_db.get_leak_event_by_id(ev_1_id)["resolved_at"]
        res_dt = datetime.fromisoformat(resolved_at.replace("Z", "+00:00"))

        # Mark Task #1 Completed
        task_1 = tmp_db.get_repair_task_by_id(1)
        tmp_db.update_repair_task_status(1, "Completed")

        # Add 5 fresh readings
        for i in range(1, 6):
            t = (res_dt + timedelta(minutes=i)).isoformat()
            tmp_db.record_sensor_reading(sensor_id=fm_id, reading_value=615.0, recorded_at=t, scenario="leak")
            tmp_db.record_sensor_reading(sensor_id=ps_id, reading_value=23.0, recorded_at=t, scenario="leak")

        # Run detection -> triggers Event #2 + Task #2
        app_client.post("/leak-detection/run")

        all_tasks = tmp_db.get_all_repair_tasks()
        assert len(all_tasks) == 2

        task_1_row = next(t for t in all_tasks if t["task_id"] == 1)
        task_2_row = next(t for t in all_tasks if t["task_id"] == 2)

        assert task_1_row["event_id"] == ev_1_id
        assert task_1_row["status"] == "Completed"

        assert task_2_row["status"] == "Pending"
        assert task_2_row["event_id"] != ev_1_id

        # Repeated queue sync does not create duplicate repair tasks
        mgr = PriorityQueueManager()
        mgr.sync_and_build_heap(tmp_db)
        assert tmp_db.count_total_repair_tasks() == 2

    def test_09_lifecycle_audit_entries_accurate_and_intact(self, tmp_db, app_client, setup_telemetry_network):
        """Scenario 9: Lifecycle audit entries remain intact and reflect true transitions without duplicates."""
        pipe_id = setup_telemetry_network["pipe_id"]

        # 1. Detection runs -> Event #1 (open) & Task #1 (Pending)
        app_client.post("/leak-detection/run")
        ev_1_id = tmp_db.get_all_leak_events()[0]["event_id"]
        task_1_id = tmp_db.get_all_repair_tasks()[0]["task_id"]

        # 2. Begin investigation via repair task
        app_client.post(f"/repairs/{task_1_id}/investigate")

        # 3. Complete repair task
        app_client.post(f"/repairs/{task_1_id}/status", data={"status": "Completed"})

        # 4. Resolve leak event
        app_client.post(f"/leak-detection/events/{ev_1_id}/status", data={"status": "resolved"})

        # Check audit log
        logs = tmp_db.get_lifecycle_audit_logs()

        # Audit transitions must include:
        # - leak_event #1: NULL -> open
        # - repair_task #1: NULL -> Pending
        # - repair_task #1: Pending -> In Progress
        # - leak_event #1: open -> investigating
        # - repair_task #1: In Progress -> Completed
        # - leak_event #1: investigating -> resolved
        leak_logs = [log for log in logs if log["entity_type"] == "leak_event"]
        task_logs = [log for log in logs if log["entity_type"] == "repair_task"]

        assert len(leak_logs) == 3
        assert len(task_logs) == 3

        # Transitions for leak_event:
        assert (leak_logs[2]["previous_status"], leak_logs[2]["new_status"]) == (None, "open")
        assert (leak_logs[1]["previous_status"], leak_logs[1]["new_status"]) == ("open", "investigating")
        assert (leak_logs[0]["previous_status"], leak_logs[0]["new_status"]) == ("investigating", "resolved")

        # Transitions for repair_task:
        assert (task_logs[2]["previous_status"], task_logs[2]["new_status"]) == (None, "Pending")
        assert (task_logs[1]["previous_status"], task_logs[1]["new_status"]) == ("Pending", "In Progress")
        assert (task_logs[0]["previous_status"], task_logs[0]["new_status"]) == ("In Progress", "Completed")

    def test_10_priority_queue_max_heap_orders_active_tasks_without_duplicates(self, tmp_db, app_client):
        """Scenario 10: Priority queue Max Heap properly orders active tasks without duplicates."""
        n1 = tmp_db.create_node("Node A", "tank", "Zone A", "active")
        n2 = tmp_db.create_node("Node B", "junction", "Zone B", "active")
        n3 = tmp_db.create_node("Node C", "valve", "Zone C", "active")

        p1 = tmp_db.create_pipe("Pipe-Alpha", n1, n2, "100.0", "500.0", "1.0", "active")
        p2 = tmp_db.create_pipe("Pipe-Beta", n2, n3, "200.0", "800.0", "1.0", "active")
        p3 = tmp_db.create_pipe("Pipe-Gamma", n1, n3, "300.0", "1200.0", "1.0", "active")

        # Record leak events with varying severities and scores
        ev1, _ = tmp_db.record_leak_event(p1, "low", "Minor drop", flow_anomaly_score=1.2, pressure_anomaly_score=-1.1)
        ev2, _ = tmp_db.record_leak_event(p2, "critical", "Major rupture", flow_anomaly_score=4.5, pressure_anomaly_score=-3.9)
        ev3, _ = tmp_db.record_leak_event(p3, "high", "High flow", flow_anomaly_score=2.8, pressure_anomaly_score=-2.2)

        # Sync queue
        mgr = PriorityQueueManager()
        heap = mgr.sync_and_build_heap(tmp_db)

        assert heap.size() == 3

        # Extract all from heap
        ordered = heap.extract_all()
        assert len(ordered) == 3

        # Strict non-increasing priority score order
        assert ordered[0].priority_score >= ordered[1].priority_score
        assert ordered[1].priority_score >= ordered[2].priority_score

        # Critical pipe should be root
        assert ordered[0].pipe_id == p2
        assert ordered[0].priority_category == "CRITICAL"

        # Mark Pipe-Beta Completed
        beta_task_id = ordered[0].task_id
        tmp_db.update_repair_task_status(beta_task_id, "Completed")

        # Re-sync heap: completed task must be excluded from active heap
        new_heap = mgr.sync_and_build_heap(tmp_db)
        assert new_heap.size() == 2
        active_ordered = new_heap.extract_all()
        assert beta_task_id not in [t.task_id for t in active_ordered]
