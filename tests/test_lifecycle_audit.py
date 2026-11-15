"""
tests/test_lifecycle_audit.py – Automated tests for AquaGrid Lifecycle Audit Logging.

Covers:
  1. Initial task creation records NULL -> Pending with entity_type='repair_task'.
  2. Initial leak event creation records NULL -> open with entity_type='leak_event'.
  3. Task transition: Pending -> In Progress.
  4. Task transition: In Progress -> Completed.
  5. Leak event transition: open -> investigating -> resolved.
  6. Begin Investigation action updates both repair task and leak event statuses
     with exactly one distinct audit entry each.
  7. Repeated submission with identical status creates no additional audit entries (idempotence).
  8. GET page refresh (/repairs) triggers zero additional audit entries.
  9. Preservation of audit history: deleting a parent repair task or leak event
     preserves the audit log row intact (no cascade deletion).
 10. Transaction safety & rollback: if audit insertion fails, status update is rolled back.
"""

import os
import sys
import sqlite3
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import database


@pytest.fixture()
def tmp_db(monkeypatch, tmp_path):
    """Isolate database tests with a fresh SQLite DB in a temporary directory."""
    import database as db_module

    test_db = str(tmp_path / "test_audit_aquagrid.db")
    monkeypatch.setattr(db_module, "INSTANCE_DIR", str(tmp_path))
    monkeypatch.setattr(db_module, "DB_PATH", test_db)
    db_module.init_db()
    return db_module


@pytest.fixture()
def audit_network(tmp_db):
    """Sets up a minimal network with 2 nodes, 1 pipe, 1 sensor."""
    node_a = tmp_db.create_node("Reservoir Alpha", "tank", "North", "active")
    node_b = tmp_db.create_node("Junction Beta", "junction", "South", "active")
    pipe_1 = tmp_db.create_pipe("Main-Trunk-1", node_a, node_b, "100.0", "500.0", "1.0", "active")
    s_flow = tmp_db.create_sensor("FM-1", pipe_1, "flow_meter", "L/min", "active")

    return {
        "nodes": [node_a, node_b],
        "pipe_id": pipe_1,
        "sensor_id": s_flow,
    }


@pytest.fixture()
def app_client(tmp_db, monkeypatch):
    """Flask test client wired to the isolated database."""
    import database as db_module
    import app as app_module

    monkeypatch.setattr(app_module, "db", db_module)

    app_module.app.config["TESTING"] = True
    app_module.app.config["WTF_CSRF_ENABLED"] = False
    app_module.app.config["SECRET_KEY"] = "test-audit-secret-key"

    with app_module.app.test_client() as client:
        yield client


class TestLifecycleAuditLog:
    """Test suite verifying immutable lifecycle audit logging across AquaGrid."""

    def test_initial_leak_event_creation_audit(self, tmp_db, audit_network):
        """Initial leak event creation records NULL -> open."""
        pipe_id = audit_network["pipe_id"]
        ev_id, is_new = tmp_db.record_leak_event(
            pipe_id=pipe_id,
            severity="critical",
            evidence_description="Surge detected",
            flow_anomaly_score=3.5,
        )
        assert is_new is True

        logs = tmp_db.get_lifecycle_audit_logs(entity_type="leak_event", entity_id=ev_id)
        assert len(logs) == 1
        log = logs[0]
        assert log["entity_type"] == "leak_event"
        assert log["entity_id"] == ev_id
        assert log["previous_status"] is None
        assert log["new_status"] == "open"
        assert "Leak anomaly detected" in log["action_description"]

    def test_initial_task_creation_audit(self, tmp_db, audit_network):
        """Initial repair task creation records NULL -> Pending."""
        pipe_id = audit_network["pipe_id"]
        ev_id, _ = tmp_db.record_leak_event(pipe_id, "critical", "Surge detected")

        task_id, is_new = tmp_db.create_or_update_repair_task(
            event_id=ev_id,
            pipe_id=pipe_id,
            priority_score=85.0,
            priority_category="CRITICAL",
        )
        assert is_new is True

        logs = tmp_db.get_lifecycle_audit_logs(entity_type="repair_task", entity_id=task_id)
        assert len(logs) == 1
        log = logs[0]
        assert log["entity_type"] == "repair_task"
        assert log["entity_id"] == task_id
        assert log["previous_status"] is None
        assert log["new_status"] == "Pending"
        assert f"Repair task #{task_id} initialized" in log["action_description"]

    def test_task_transitions_pending_to_in_progress_to_completed(self, tmp_db, audit_network):
        """Sequential task transitions record previous and new status distinctly."""
        pipe_id = audit_network["pipe_id"]
        ev_id, _ = tmp_db.record_leak_event(pipe_id, "high", "High flow")
        task_id, _ = tmp_db.create_or_update_repair_task(ev_id, pipe_id, 75.0, "HIGH")

        # Pending -> In Progress
        ok1 = tmp_db.update_repair_task_status(task_id, "In Progress")
        assert ok1 is True

        # In Progress -> Completed
        ok2 = tmp_db.update_repair_task_status(task_id, "Completed")
        assert ok2 is True

        logs = tmp_db.get_lifecycle_audit_logs(entity_type="repair_task", entity_id=task_id)
        # Order is newest first: Completed, In Progress, Initial (Pending)
        assert len(logs) == 3

        # 1. Newest: In Progress -> Completed
        assert logs[0]["previous_status"] == "In Progress"
        assert logs[0]["new_status"] == "Completed"

        # 2. Middle: Pending -> In Progress
        assert logs[1]["previous_status"] == "Pending"
        assert logs[1]["new_status"] == "In Progress"

        # 3. Oldest: NULL -> Pending
        assert logs[2]["previous_status"] is None
        assert logs[2]["new_status"] == "Pending"

    def test_leak_event_transitions(self, tmp_db, audit_network):
        """Leak event transitions open -> investigating -> resolved."""
        pipe_id = audit_network["pipe_id"]
        ev_id, _ = tmp_db.record_leak_event(pipe_id, "medium", "Pressure drop")

        tmp_db.update_leak_event_status(ev_id, "investigating")
        tmp_db.update_leak_event_status(ev_id, "resolved")

        logs = tmp_db.get_lifecycle_audit_logs(entity_type="leak_event", entity_id=ev_id)
        assert len(logs) == 3

        assert logs[0]["previous_status"] == "investigating"
        assert logs[0]["new_status"] == "resolved"

        assert logs[1]["previous_status"] == "open"
        assert logs[1]["new_status"] == "investigating"

        assert logs[2]["previous_status"] is None
        assert logs[2]["new_status"] == "open"

    def test_begin_investigation_action(self, app_client, tmp_db, audit_network):
        """Begin Investigation sets task to In Progress and event to investigating with 1 audit entry each."""
        pipe_id = audit_network["pipe_id"]
        ev_id, _ = tmp_db.record_leak_event(pipe_id, "critical", "Dual failure")
        task_id, _ = tmp_db.create_or_update_repair_task(ev_id, pipe_id, 90.0, "CRITICAL")

        # Initial logs count: 1 for event, 1 for task
        assert tmp_db.count_lifecycle_audit_logs() == 2

        # Post to investigate
        rv = app_client.post(f"/repairs/{task_id}/investigate", follow_redirects=True)
        assert rv.status_code == 200

        # After investigate: exactly 2 new audit entries (1 for task, 1 for event)
        assert tmp_db.count_lifecycle_audit_logs() == 4

        task_logs = tmp_db.get_lifecycle_audit_logs(entity_type="repair_task", entity_id=task_id)
        assert task_logs[0]["previous_status"] == "Pending"
        assert task_logs[0]["new_status"] == "In Progress"
        assert "Investigation begun" in task_logs[0]["action_description"]

        event_logs = tmp_db.get_lifecycle_audit_logs(entity_type="leak_event", entity_id=ev_id)
        assert event_logs[0]["previous_status"] == "open"
        assert event_logs[0]["new_status"] == "investigating"
        assert "Field investigation initiated" in event_logs[0]["action_description"]

    def test_repeated_submission_creates_no_duplicate_audit_records(self, tmp_db, audit_network):
        """Submitting an identical status must NOT create redundant audit entries."""
        pipe_id = audit_network["pipe_id"]
        ev_id, _ = tmp_db.record_leak_event(pipe_id, "high", "Fluctuation")
        task_id, _ = tmp_db.create_or_update_repair_task(ev_id, pipe_id, 70.0, "HIGH")

        # Currently Pending. Update to In Progress once
        tmp_db.update_repair_task_status(task_id, "In Progress")
        count_before = tmp_db.count_lifecycle_audit_logs()

        # Update to In Progress again (identical state)
        tmp_db.update_repair_task_status(task_id, "In Progress")
        count_after = tmp_db.count_lifecycle_audit_logs()

        assert count_before == count_after, "Duplicate status update must not create new audit log!"

        # Same for leak events
        tmp_db.update_leak_event_status(ev_id, "investigating")
        ev_count_before = tmp_db.count_lifecycle_audit_logs()
        tmp_db.update_leak_event_status(ev_id, "investigating")
        ev_count_after = tmp_db.count_lifecycle_audit_logs()

        assert ev_count_before == ev_count_after, "Duplicate leak event status must not create new audit log!"

    def test_page_refresh_creates_no_additional_audit_records(self, app_client, tmp_db, audit_network):
        """GET /repairs page refresh must be idempotent and produce 0 audit records."""
        pipe_id = audit_network["pipe_id"]
        ev_id, _ = tmp_db.record_leak_event(pipe_id, "critical", "Burst")
        tmp_db.create_or_update_repair_task(ev_id, pipe_id, 88.0, "CRITICAL")

        count_initial = tmp_db.count_lifecycle_audit_logs()

        # Refresh page multiple times via GET
        app_client.get("/repairs")
        app_client.get("/repairs")
        app_client.get("/repairs")

        count_final = tmp_db.count_lifecycle_audit_logs()
        assert count_initial == count_final, "GET page refresh must never generate audit logs!"

    def test_preservation_of_audit_history_on_parent_deletion(self, tmp_db, audit_network):
        """Audit history is preserved without cascade deletion if task/event is deleted."""
        pipe_id = audit_network["pipe_id"]
        ev_id, _ = tmp_db.record_leak_event(pipe_id, "critical", "Pipe burst")
        task_id, _ = tmp_db.create_or_update_repair_task(ev_id, pipe_id, 92.0, "CRITICAL")

        tmp_db.update_repair_task_status(task_id, "In Progress")

        # Verify audit logs exist
        logs_before = tmp_db.get_lifecycle_audit_logs(entity_type="repair_task", entity_id=task_id)
        assert len(logs_before) == 2

        # Manually delete repair_tasks row
        with tmp_db._connect() as conn:
            conn.execute("DELETE FROM repair_tasks WHERE task_id = ?", (task_id,))
            conn.commit()

        # Verify task is deleted
        assert tmp_db.get_repair_task_by_id(task_id) is None

        # CRITICAL: Verify audit logs STILL exist!
        logs_after = tmp_db.get_lifecycle_audit_logs(entity_type="repair_task", entity_id=task_id)
        assert len(logs_after) == 2, "Audit logs must be preserved when parent task is deleted!"
        assert logs_after[0]["action_description"] is not None

    def test_transaction_rollback_if_audit_fails(self, tmp_db, audit_network, monkeypatch):
        """If audit insertion encounters an error, the status update rolls back cleanly."""
        pipe_id = audit_network["pipe_id"]
        ev_id, _ = tmp_db.record_leak_event(pipe_id, "medium", "Transient")
        task_id, _ = tmp_db.create_or_update_repair_task(ev_id, pipe_id, 45.0, "MEDIUM")

        task = tmp_db.get_repair_task_by_id(task_id)
        assert task["status"] == "Pending"

        # Monkeypatch record_lifecycle_audit to raise an error
        def failing_audit(*args, **kwargs):
            raise sqlite3.OperationalError("Simulated disk error or audit failure")

        monkeypatch.setattr(tmp_db, "record_lifecycle_audit", failing_audit)

        # Attempt status update; should fail
        with pytest.raises(sqlite3.OperationalError):
            tmp_db.update_repair_task_status(task_id, "Completed")

        # Verify status did NOT change to Completed
        task_after = tmp_db.get_repair_task_by_id(task_id)
        assert task_after["status"] == "Pending", "Status must rollback if audit insertion fails!"

