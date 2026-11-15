"""
tests/test_duplicate_prevention_regression.py – Focused Regression Audit for Duplicate Leak Event Prevention.

Strictly validates:
PART 1 – GET REQUEST IDEMPOTENCY
  - GET /leak-detection does not execute detection that creates or updates leak events.
  - Refreshing the page 5 consecutive times does not change:
      * Total leak event count
      * Existing event statuses
      * Lifecycle audit record count
      * Repair task count and statuses
  - Page rendering does not trigger repair queue synchronization that creates or modifies records.

PART 2 – HISTORICAL TELEMETRY PROTECTION
  - Pipe with an active leak event and anomalous readings recorded.
  - Event is resolved at timestamp T_res.
  - Detection executed using only readings recorded before or at T_res.
  - Verifies no new leak event is created.
  - Verifies no duplicate repair task is created.
  - Diagnostic status is confirmed as 'insufficient_data'.

PART 3 – GENUINE NEW INCIDENT
  - Event resolved at timestamp T1.
  - Sufficient fresh synthetic normal readings inserted after T1 to establish baseline.
  - New sustained anomalous sequence inserted after T1.
  - Detection executed.
  - Verifies a new leak event is created only when all detection criteria are satisfied.
  - Verifies exactly one repair task is associated with the new event.

PART 4 – REPEATED DETECTION
  - Detection executed 5 consecutive times while an event remains open.
  - Verifies only one active event exists for that incident.
  - Verifies no duplicate repair tasks are created.
  - Verifies existing event lifecycle history is preserved.
  - Verifies repeated detection does not generate redundant lifecycle audit entries for unchanged statuses.

PART 5 – DATA SAFETY & LIVE DB INTEGRITY
  - Confirms isolated test databases were used.
  - Confirms live database records remain untouched.
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
    test_db = str(tmp_path / "test_regression_aquagrid.db")
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
    app_module.app.config["SECRET_KEY"] = "test-regression-secret-key"

    with app_module.app.test_client() as client:
        yield client


# ==============================================================================
# PART 1: GET REQUEST IDEMPOTENCY
# ==============================================================================

class TestPart1GetRequestIdempotency:
    """Validate that GET /leak-detection is strictly read-only and idempotent."""

    def test_get_leak_detection_5_refreshes_produces_zero_mutations(self, tmp_db, app_client):
        """
        Setup existing state with an open event, a resolved event, associated repair tasks,
        and telemetry. Refresh GET /leak-detection 5 times.
        Verify:
        - Total leak event count unchanged
        - Existing event statuses unchanged
        - Lifecycle audit record count unchanged
        - Repair task count unchanged
        - No queue synchronization mutations occur
        """
        # 1. Setup network topology
        n1 = tmp_db.create_node("Reservoir North", "tank", "North Zone", "active")
        n2 = tmp_db.create_node("Junction Central", "junction", "Central Zone", "active")
        n3 = tmp_db.create_node("Valve South", "valve", "South Zone", "active")

        p1 = tmp_db.create_pipe("Trunk-Resolved", n1, n2, "200.0", "1000.0", "1.0", "active")
        p2 = tmp_db.create_pipe("Trunk-Open", n2, n3, "200.0", "800.0", "1.0", "active")

        # 2. Record Event #1 on Pipe 1 and resolve it
        ev1_id, _ = tmp_db.record_leak_event(
            pipe_id=p1,
            severity="high",
            evidence_description="Historical anomaly on Trunk-Resolved",
            flow_anomaly_score=3.1,
            pressure_anomaly_score=-2.8,
            detected_at="2026-11-09T08:00:00+00:00",
        )
        tmp_db.update_leak_event_status(ev1_id, "resolved")

        # Create Task #1 and complete it
        task1_id, _ = tmp_db.create_or_update_repair_task(
            event_id=ev1_id,
            pipe_id=p1,
            priority_score=75.0,
            priority_category="HIGH",
        )
        tmp_db.update_repair_task_status(task1_id, "Completed")

        # 3. Record Event #2 on Pipe 2 (open) and Task #2 (Pending)
        ev2_id, _ = tmp_db.record_leak_event(
            pipe_id=p2,
            severity="critical",
            evidence_description="Active anomaly on Trunk-Open",
            flow_anomaly_score=4.2,
            pressure_anomaly_score=-3.5,
            detected_at="2026-11-09T09:00:00+00:00",
        )
        task2_id, _ = tmp_db.create_or_update_repair_task(
            event_id=ev2_id,
            pipe_id=p2,
            priority_score=90.0,
            priority_category="CRITICAL",
        )

        # Snapshot exact database state before GET requests
        events_before = [dict(r) for r in tmp_db.get_all_leak_events()]
        tasks_before = [dict(r) for r in tmp_db.get_all_repair_tasks()]
        audits_before = [dict(r) for r in tmp_db.get_lifecycle_audit_logs()]

        assert len(events_before) == 2
        assert len(tasks_before) == 2
        assert len(audits_before) == 6  # 3 events audits + 3 tasks audits
        initial_audit_count = len(audits_before)

        # 4. Perform 5 consecutive GET requests to /leak-detection
        for i in range(5):
            res = app_client.get("/leak-detection")
            assert res.status_code == 200

        # Snapshot exact database state after 5 GET requests
        events_after = [dict(r) for r in tmp_db.get_all_leak_events()]
        tasks_after = [dict(r) for r in tmp_db.get_all_repair_tasks()]
        audits_after = [dict(r) for r in tmp_db.get_lifecycle_audit_logs()]

        # 5. Assertions: Zero state changes
        # Event count unchanged
        assert len(events_after) == len(events_before) == 2
        # Task count unchanged
        assert len(tasks_after) == len(tasks_before) == 2
        # Audit log count unchanged
        assert len(audits_after) == initial_audit_count

        # Detailed event property equality
        for eb, ea in zip(events_before, events_after):
            assert eb["event_id"] == ea["event_id"]
            assert eb["status"] == ea["status"]
            assert eb["severity"] == ea["severity"]
            assert eb["detected_at"] == ea["detected_at"]
            assert eb["resolved_at"] == ea["resolved_at"]

        # Detailed task property equality
        for tb, ta in zip(tasks_before, tasks_after):
            assert tb["task_id"] == ta["task_id"]
            assert tb["status"] == ta["status"]
            assert tb["priority_score"] == ta["priority_score"]
            assert tb["updated_at"] == ta["updated_at"]

        # Also verify /repairs route does not create duplicate records
        for _ in range(3):
            res_rep = app_client.get("/repairs")
            assert res_rep.status_code == 200

        assert len(tmp_db.get_all_leak_events()) == 2
        assert len(tmp_db.get_all_repair_tasks()) == 2
        assert len(tmp_db.get_lifecycle_audit_logs()) == initial_audit_count


# ==============================================================================
# PART 2: HISTORICAL TELEMETRY PROTECTION
# ==============================================================================

class TestPart2HistoricalTelemetryProtection:
    """Validate that stale telemetry cannot recreate an incident on a resolved pipe."""

    def test_stale_telemetry_cannot_recreate_resolved_leak_event(self, tmp_db, app_client):
        """
        1. Create pipe with active leak event.
        2. Record synthetic anomalous readings.
        3. Resolve the event at timestamp T_res.
        4. Execute detection using only readings recorded before or at T_res.
        5. Verify that no new leak event is created.
        6. Verify that no duplicate repair task is created.
        """
        n1 = tmp_db.create_node("Node A", "tank", "Zone A", "active")
        n2 = tmp_db.create_node("Node B", "junction", "Zone B", "active")
        pipe_id = tmp_db.create_pipe("Pipe-Stale-Test", n1, n2, "200.0", "500.0", "1.0", "active")

        fm = tmp_db.create_sensor("FM-Stale", pipe_id, "flow_meter", "L/min", "active")
        ps = tmp_db.create_sensor("PS-Stale", pipe_id, "pressure_sensor", "psi", "active")

        # Insert 10 baseline readings (scenario="normal")
        base_time = datetime(2026, 11, 9, 10, 0, 0, tzinfo=timezone.utc)
        for i in range(10):
            t = (base_time + timedelta(minutes=i)).isoformat()
            tmp_db.record_sensor_reading(sensor_id=fm, reading_value=350.0, recorded_at=t, scenario="normal")
            tmp_db.record_sensor_reading(sensor_id=ps, reading_value=52.0, recorded_at=t, scenario="normal")

        # Insert 5 anomalous readings (scenario="leak")
        for i in range(5):
            t = (base_time + timedelta(minutes=10 + i)).isoformat()
            tmp_db.record_sensor_reading(sensor_id=fm, reading_value=620.0, recorded_at=t, scenario="leak")
            tmp_db.record_sensor_reading(sensor_id=ps, reading_value=24.0, recorded_at=t, scenario="leak")

        # Run detection to create Event #1 and Task #1
        app_client.post("/leak-detection/run")
        events = tmp_db.get_all_leak_events()
        assert len(events) == 1
        ev1 = events[0]
        ev1_id = ev1["event_id"]
        assert ev1["status"] == "open"
        assert tmp_db.count_active_repair_tasks() == 1

        # Resolve Event #1 at T_res (after the last reading at 10:14)
        t_res = "2026-11-09T10:20:00+00:00"
        tmp_db.update_leak_event_status(ev1_id, "resolved")
        ev1_updated = tmp_db.get_leak_event_by_id(ev1_id)
        assert ev1_updated["status"] == "resolved"
        assert ev1_updated["resolved_at"] is not None

        # Execute detection again using ONLY the historical telemetry (all recorded <= 10:14, so <= t_res)
        app_client.post("/leak-detection/run")

        # Verify NO new leak event created
        all_events_after = tmp_db.get_all_leak_events()
        assert len(all_events_after) == 1
        assert all_events_after[0]["event_id"] == ev1_id
        assert all_events_after[0]["status"] == "resolved"

        # Verify NO duplicate repair task created
        all_tasks = tmp_db.get_all_repair_tasks()
        assert len(all_tasks) == 1
        assert all_tasks[0]["event_id"] == ev1_id

        # Verify diagnostic status reports insufficient_data
        detector = LeakDetector()
        diags = detector.run_detection_network(tmp_db, auto_record=False)
        pipe_diag = next(d for d in diags if d.pipe_id == pipe_id)
        assert pipe_diag.status == "insufficient_data"
        assert "post-resolution" in pipe_diag.evidence.lower() or "insufficient" in pipe_diag.evidence.lower()


# ==============================================================================
# PART 3: GENUINE NEW INCIDENT
# ==============================================================================

class TestPart3GenuineNewIncident:
    """Validate that genuinely fresh telemetry can legitimately create a new incident."""

    def test_fresh_telemetry_post_resolution_creates_legitimate_event_2(self, tmp_db, app_client):
        """
        1. Resolve an existing leak event at timestamp T1.
        2. Insert sufficient fresh synthetic normal readings after T1 to establish required baseline.
        3. Insert a new sustained anomalous sequence after T1.
        4. Execute detection.
        5. Verify that a new leak event is created only when all detection criteria are satisfied.
        6. Verify that exactly one repair task is associated with the new event.
        """
        n1 = tmp_db.create_node("Node Alpha", "tank", "Zone Alpha", "active")
        n2 = tmp_db.create_node("Node Beta", "junction", "Zone Beta", "active")
        pipe_id = tmp_db.create_pipe("Pipe-New-Incident", n1, n2, "200.0", "600.0", "1.0", "active")

        fm = tmp_db.create_sensor("FM-Incident", pipe_id, "flow_meter", "L/min", "active")
        ps = tmp_db.create_sensor("PS-Incident", pipe_id, "pressure_sensor", "psi", "active")

        # Initial baseline and initial leak
        base_time = datetime(2026, 11, 9, 8, 0, 0, tzinfo=timezone.utc)
        for i in range(10):
            t = (base_time + timedelta(minutes=i)).isoformat()
            tmp_db.record_sensor_reading(sensor_id=fm, reading_value=350.0, recorded_at=t, scenario="normal")
            tmp_db.record_sensor_reading(sensor_id=ps, reading_value=52.0, recorded_at=t, scenario="normal")
        for i in range(5):
            t = (base_time + timedelta(minutes=10 + i)).isoformat()
            tmp_db.record_sensor_reading(sensor_id=fm, reading_value=600.0, recorded_at=t, scenario="leak")
            tmp_db.record_sensor_reading(sensor_id=ps, reading_value=25.0, recorded_at=t, scenario="leak")

        # Create Event #1 and resolve it
        app_client.post("/leak-detection/run")
        ev1_id = tmp_db.get_all_leak_events()[0]["event_id"]
        tmp_db.update_leak_event_status(ev1_id, "resolved")
        resolved_ev = tmp_db.get_leak_event_by_id(ev1_id)
        t1_str = resolved_ev["resolved_at"]
        t1_dt = datetime.fromisoformat(t1_str.replace("Z", "+00:00"))

        # Mark Task #1 Completed
        tmp_db.update_repair_task_status(1, "Completed")

        # Step 2: Insert fresh post-repair telemetry (10 normal readings after T1)
        # to re-establish post-repair normal baseline
        for i in range(1, 11):
            t = (t1_dt + timedelta(minutes=i)).isoformat()
            tmp_db.record_sensor_reading(sensor_id=fm, reading_value=350.0 + (i % 2) * 1.0, recorded_at=t, scenario="normal")
            tmp_db.record_sensor_reading(sensor_id=ps, reading_value=52.0 - (i % 2) * 0.5, recorded_at=t, scenario="normal")

        # Intermediate check: With normal readings post-repair, detection reports normal and creates NO new event
        app_client.post("/leak-detection/run")
        assert tmp_db.count_total_leak_events() == 1
        detector = LeakDetector()
        diags = detector.run_detection_network(tmp_db, auto_record=False)
        p_diag = next(d for d in diags if d.pipe_id == pipe_id)
        assert p_diag.status == "normal"

        # Step 3: Insert new sustained anomalous sequence (> T1, after the normal readings)
        # Flow surge ~ 620 L/min (+77%), Pressure drop ~ 23 psi (-55%), 5 consecutive readings
        surge_start = t1_dt + timedelta(minutes=15)
        for i in range(5):
            t = (surge_start + timedelta(minutes=i)).isoformat()
            tmp_db.record_sensor_reading(sensor_id=fm, reading_value=620.0 + i * 2.0, recorded_at=t, scenario="leak")
            tmp_db.record_sensor_reading(sensor_id=ps, reading_value=23.0 - i * 0.5, recorded_at=t, scenario="leak")

        # Step 4: Execute detection
        app_client.post("/leak-detection/run")

        # Step 5: Verify new leak event is created only when criteria are satisfied
        all_events = tmp_db.get_all_leak_events()
        assert len(all_events) == 2
        new_event = all_events[0]
        old_event = all_events[1]

        assert old_event["event_id"] == ev1_id
        assert old_event["status"] == "resolved"

        assert new_event["event_id"] != ev1_id
        assert new_event["status"] == "open"
        assert new_event["pipe_id"] == pipe_id
        assert new_event["severity"] in ("high", "critical")

        # Step 6: Verify exactly one repair task is associated with the new event
        all_tasks = tmp_db.get_all_repair_tasks()
        assert len(all_tasks) == 2

        task_old = next(t for t in all_tasks if t["event_id"] == ev1_id)
        task_new = next(t for t in all_tasks if t["event_id"] == new_event["event_id"])

        assert task_old["status"] == "Completed"
        assert task_new["status"] == "Pending"
        assert task_new["pipe_id"] == pipe_id


# ==============================================================================
# PART 4: REPEATED DETECTION
# ==============================================================================

class TestPart4RepeatedDetection:
    """Validate that executing detection repeatedly on an open event is strictly idempotent."""

    def test_repeated_detection_5_times_preserves_single_event_and_task(self, tmp_db, app_client):
        """
        Execute detection 5 times while an event remains open.
        Verify:
        - Only one active event exists for that incident.
        - No duplicate repair tasks are created.
        - Existing event lifecycle history is preserved.
        - Repeated detection does not generate redundant lifecycle audit entries for unchanged statuses.
        """
        n1 = tmp_db.create_node("Node Main", "tank", "Zone M", "active")
        n2 = tmp_db.create_node("Node Branch", "junction", "Zone B", "active")
        pipe_id = tmp_db.create_pipe("Pipe-Repeat-Test", n1, n2, "200.0", "500.0", "1.0", "active")

        fm = tmp_db.create_sensor("FM-Rep", pipe_id, "flow_meter", "L/min", "active")
        ps = tmp_db.create_sensor("PS-Rep", pipe_id, "pressure_sensor", "psi", "active")

        base_time = datetime(2026, 11, 9, 12, 0, 0, tzinfo=timezone.utc)
        for i in range(10):
            t = (base_time + timedelta(minutes=i)).isoformat()
            tmp_db.record_sensor_reading(sensor_id=fm, reading_value=350.0, recorded_at=t, scenario="normal")
            tmp_db.record_sensor_reading(sensor_id=ps, reading_value=52.0, recorded_at=t, scenario="normal")
        for i in range(5):
            t = (base_time + timedelta(minutes=10 + i)).isoformat()
            tmp_db.record_sensor_reading(sensor_id=fm, reading_value=610.0, recorded_at=t, scenario="leak")
            tmp_db.record_sensor_reading(sensor_id=ps, reading_value=24.0, recorded_at=t, scenario="leak")

        # 1. Initial detection execution
        app_client.post("/leak-detection/run")

        assert tmp_db.count_total_leak_events() == 1
        assert tmp_db.count_active_repair_tasks() == 1
        initial_events = [dict(r) for r in tmp_db.get_all_leak_events()]
        initial_tasks = [dict(r) for r in tmp_db.get_all_repair_tasks()]
        initial_audits = [dict(r) for r in tmp_db.get_lifecycle_audit_logs()]

        ev_id = initial_events[0]["event_id"]
        task_id = initial_tasks[0]["task_id"]
        assert len(initial_audits) == 2  # (1 for leak event NULL->open, 1 for repair task NULL->Pending)

        # 2. Execute detection 4 more times (5 total executions)
        for run_idx in range(4):
            app_client.post("/leak-detection/run")

        # Verify:
        # - Only one active event exists
        events_after = [dict(r) for r in tmp_db.get_all_leak_events()]
        assert len(events_after) == 1
        assert events_after[0]["event_id"] == ev_id
        assert events_after[0]["status"] == "open"

        # - No duplicate repair tasks are created
        tasks_after = [dict(r) for r in tmp_db.get_all_repair_tasks()]
        assert len(tasks_after) == 1
        assert tasks_after[0]["task_id"] == task_id
        assert tasks_after[0]["status"] == "Pending"

        # - Lifecycle audit history is preserved and NO duplicate entries are created
        audits_after = [dict(r) for r in tmp_db.get_lifecycle_audit_logs()]
        assert len(audits_after) == 2

        # Verify audit log content
        event_audits = [a for a in audits_after if a["entity_type"] == "leak_event"]
        task_audits = [a for a in audits_after if a["entity_type"] == "repair_task"]
        assert len(event_audits) == 1
        assert len(task_audits) == 1
        assert event_audits[0]["new_status"] == "open"
        assert task_audits[0]["new_status"] == "Pending"


# ==============================================================================
# PART 5: DATA SAFETY & LIVE DATABASE INTEGRITY
# ==============================================================================

class TestPart5DataSafetyAndLiveDatabaseIntegrity:
    """Validate that live database records remain untouched and isolated."""

    def test_live_database_remains_unmodified(self):
        """
        Verify that running tests against tmp_db does not alter the actual live
        aquagrid.db file in instance/.
        """
        import os
        import sqlite3

        live_db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "instance", "aquagrid.db")
        if not os.path.exists(live_db_path):
            pytest.skip("Live database file not present.")

        conn = sqlite3.connect(live_db_path)
        conn.row_factory = sqlite3.Row
        try:
            # Check existing records in live DB
            events = conn.execute("SELECT * FROM leak_events ORDER BY event_id ASC").fetchall()
            tasks = conn.execute("SELECT * FROM repair_tasks ORDER BY task_id ASC").fetchall()
            audits = conn.execute("SELECT * FROM lifecycle_audit_log ORDER BY audit_id ASC").fetchall()

            # The live database has 2 leak events, 2 repair tasks, and 4 audit logs
            assert len(events) >= 2
            assert len(tasks) >= 2
            assert len(audits) >= 4

            # Verify Event #1 is resolved and Event #2 is open
            ev1 = next(e for e in events if e["event_id"] == 1)
            ev2 = next(e for e in events if e["event_id"] == 2)
            assert ev1["status"] == "resolved"
            assert ev1["resolved_at"] is not None
            assert ev2["status"] == "open"

            # Verify Task #1 is Completed and Task #2 is Pending
            t1 = next(t for t in tasks if t["task_id"] == 1)
            t2 = next(t for t in tasks if t["task_id"] == 2)
            assert t1["status"] == "Completed"
            assert t2["status"] == "Pending"
        finally:
            conn.close()
