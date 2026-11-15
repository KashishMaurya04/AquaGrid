"""
tests/test_database_integrity_audit.py – Day 4 Database Integrity, Lifecycle Consistency,
and Transaction Atomicity Test Suite for AquaGrid.

Covers:
  - SQLite PRAGMA integrity_check and foreign_key_check.
  - Foreign key RESTRICT enforcement across network_nodes, pipes, sensors, sensor_readings, leak_events, repair_tasks.
  - Survival of lifecycle_audit_log upon deletion of parent entities.
  - Leak event lifecycle transitions (NULL -> open -> investigating -> resolved), idempotency, and invalid status rejection.
  - Repair task lifecycle transitions (NULL -> Pending -> In Progress -> Completed), duplicate prevention, and active queue filtering.
  - Transaction atomicity & rollback safety under simulated audit failure.
  - Concurrent task creation race condition safety.
  - Live database integrity and read-only preservation.
"""

import os
import sys
import pytest
import sqlite3
import concurrent.futures
from datetime import datetime, timezone

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import database as db_module


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture()
def tmp_db(monkeypatch, tmp_path):
    """Isolate database tests with a fresh SQLite DB in a temporary directory."""
    test_db = str(tmp_path / "test_db_integrity_aquagrid.db")
    monkeypatch.setattr(db_module, "INSTANCE_DIR", str(tmp_path))
    monkeypatch.setattr(db_module, "DB_PATH", test_db)
    db_module.init_db()
    return db_module


@pytest.fixture()
def sample_network(tmp_db):
    """Creates a basic network with 2 nodes and 1 pipe."""
    n1 = tmp_db.create_node("Reservoir-Alpha", "tank", "North Zone", "active")
    n2 = tmp_db.create_node("Junction-Beta", "junction", "Central Zone", "active")
    p1 = tmp_db.create_pipe("MainTrunk-1", n1, n2, "200.0", "5000.0", "1.0", "active")
    return {"n1": n1, "n2": n2, "p1": p1}


# ═════════════════════════════════════════════════════════════════════════════
#  1. DATABASE INTEGRITY & FOREIGN KEY ENFORCEMENT
# ═════════════════════════════════════════════════════════════════════════════

class TestDatabaseIntegrity:
    def test_pragma_integrity_and_foreign_key_checks(self, tmp_db):
        """PRAGMA integrity_check returns 'ok' and PRAGMA foreign_key_check returns zero violations."""
        with tmp_db._connect() as conn:
            integrity = conn.execute("PRAGMA integrity_check").fetchall()
            fk_violations = conn.execute("PRAGMA foreign_key_check").fetchall()
            assert integrity[0][0] == "ok"
            assert len(fk_violations) == 0

    def test_referential_integrity_restricted_deletions(self, tmp_db, sample_network):
        """Foreign key RESTRICT prevents deleting parents when dependent child records exist."""
        n1 = sample_network["n1"]
        p1 = sample_network["p1"]

        # 1. Deleting node referenced by pipe is blocked
        with pytest.raises(RuntimeError, match="still used by"):
            tmp_db.delete_node(n1)

        # 2. Deleting pipe referenced by sensor is blocked
        s1 = tmp_db.create_sensor("FM-1", p1, "flow_meter", "L/min", "active")
        with pytest.raises(RuntimeError, match="still monitored by"):
            tmp_db.delete_pipe(p1)

        # 3. Deleting sensor referenced by readings is blocked
        tmp_db.record_sensor_reading(s1, 350.0)
        with pytest.raises(RuntimeError, match="recorded reading"):
            tmp_db.delete_sensor(s1)

        # 4. Deleting leak event referenced by repair task is blocked by SQLite FK RESTRICT
        tmp_db.clear_sensor_readings()
        tmp_db.delete_sensor(s1)
        ev_id, _ = tmp_db.record_leak_event(p1, "critical", "Severe pipe rupture")
        task_id, _ = tmp_db.create_or_update_repair_task(ev_id, p1, 100.0, "CRITICAL")

        with tmp_db._connect() as conn:
            with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY constraint failed"):
                conn.execute("DELETE FROM leak_events WHERE event_id = ?", (ev_id,))

    def test_lifecycle_audit_log_survives_parent_record_deletion(self, tmp_db, sample_network):
        """Audit history must NOT be deleted via CASCADE if a repair task or leak event is deleted."""
        p1 = sample_network["p1"]
        ev_id, _ = tmp_db.record_leak_event(p1, "high", "High anomaly")
        task_id, _ = tmp_db.create_or_update_repair_task(ev_id, p1, 80.0, "HIGH")

        # Manually delete the repair task
        with tmp_db._connect() as conn:
            conn.execute("DELETE FROM repair_tasks WHERE task_id = ?", (task_id,))
            conn.commit()

            # Verify audit records for task_id survived
            audits = conn.execute("SELECT * FROM lifecycle_audit_log WHERE repair_task_id = ?", (task_id,)).fetchall()
            assert len(audits) > 0
            assert audits[0]["entity_type"] == "repair_task"
            assert audits[0]["new_status"] == "Pending"


# ═════════════════════════════════════════════════════════════════════════════
#  2. LEAK EVENT & REPAIR TASK LIFECYCLE VALIDATION
# ═════════════════════════════════════════════════════════════════════════════

class TestLifecycleTransitions:
    def test_leak_event_lifecycle_transitions_and_idempotency(self, tmp_db, sample_network):
        """
        Verify:
        - NULL -> open (on creation)
        - open -> investigating
        - investigating -> resolved
        - Repeated update with same status produces ZERO duplicate audit records.
        """
        p1 = sample_network["p1"]
        ev_id, _ = tmp_db.record_leak_event(p1, "medium", "Medium anomaly")
        ev = dict(tmp_db.get_leak_event_by_id(ev_id))
        assert ev["status"] == "open"

        # open -> investigating
        tmp_db.update_leak_event_status(ev_id, "investigating")
        ev = dict(tmp_db.get_leak_event_by_id(ev_id))
        assert ev["status"] == "investigating"

        # investigating -> resolved
        tmp_db.update_leak_event_status(ev_id, "resolved")
        ev = dict(tmp_db.get_leak_event_by_id(ev_id))
        assert ev["status"] == "resolved"
        assert ev["resolved_at"] is not None

        # Repeated resolution does not produce duplicate audit logs
        audits_before = len(tmp_db.get_lifecycle_audit_logs(entity_type="leak_event", entity_id=ev_id))
        tmp_db.update_leak_event_status(ev_id, "resolved")
        audits_after = len(tmp_db.get_lifecycle_audit_logs(entity_type="leak_event", entity_id=ev_id))
        assert audits_before == audits_after

    def test_repair_task_lifecycle_transitions_and_queue_isolation(self, tmp_db, sample_network):
        """
        Verify:
        - NULL -> Pending
        - Pending -> In Progress
        - In Progress -> Completed
        - Completed tasks do not appear in active repair queue.
        - Queue synchronization does NOT overwrite In Progress or Completed tasks.
        """
        p1 = sample_network["p1"]
        ev_id, _ = tmp_db.record_leak_event(p1, "high", "High burst")
        task_id, is_new = tmp_db.create_or_update_repair_task(ev_id, p1, 85.0, "HIGH")
        assert is_new is True
        t = dict(tmp_db.get_repair_task_by_id(task_id))
        assert t["status"] == "Pending"

        # Pending -> In Progress
        tmp_db.update_repair_task_status(task_id, "In Progress")
        t = dict(tmp_db.get_repair_task_by_id(task_id))
        assert t["status"] == "In Progress"

        # In Progress -> Completed
        tmp_db.update_repair_task_status(task_id, "Completed")
        t = dict(tmp_db.get_repair_task_by_id(task_id))
        assert t["status"] == "Completed"

        # Completed tasks excluded from active queue
        active = tmp_db.get_active_repair_tasks()
        assert all(task["task_id"] != task_id for task in active)

        # Re-running synchronization updates priority without overwriting status
        task_dup_id, is_new_dup = tmp_db.create_or_update_repair_task(ev_id, p1, 95.0, "CRITICAL")
        assert is_new_dup is False
        assert task_dup_id == task_id
        t_after = dict(tmp_db.get_repair_task_by_id(task_id))
        assert t_after["status"] == "Completed"
        assert t_after["priority_score"] == 95.0


# ═════════════════════════════════════════════════════════════════════════════
#  3. TRANSACTION ATOMICITY AND ROLLBACK SAFETY
# ═════════════════════════════════════════════════════════════════════════════

class TestTransactionAtomicity:
    def test_leak_event_rollback_on_audit_failure(self, tmp_db, sample_network, monkeypatch):
        """If audit logging fails during update_leak_event_status, the event UPDATE is rolled back."""
        p1 = sample_network["p1"]
        ev_id, _ = tmp_db.record_leak_event(p1, "low", "Low anomaly")
        ev = dict(tmp_db.get_leak_event_by_id(ev_id))
        assert ev["status"] == "open"

        def failing_audit(*args, **kwargs):
            raise RuntimeError("Simulated Database I/O Failure during audit logging!")

        monkeypatch.setattr(tmp_db, "record_lifecycle_audit", failing_audit)

        with pytest.raises(RuntimeError, match="Simulated Database I/O Failure"):
            tmp_db.update_leak_event_status(ev_id, "investigating")

        # Status must have rolled back to 'open'
        ev_after = dict(tmp_db.get_leak_event_by_id(ev_id))
        assert ev_after["status"] == "open"

    def test_repair_task_rollback_on_audit_failure(self, tmp_db, sample_network, monkeypatch):
        """If audit logging fails during update_repair_task_status, the task UPDATE is rolled back."""
        p1 = sample_network["p1"]
        ev_id, _ = tmp_db.record_leak_event(p1, "low", "Low anomaly")
        task_id, _ = tmp_db.create_or_update_repair_task(ev_id, p1, 50.0, "LOW")
        t = dict(tmp_db.get_repair_task_by_id(task_id))
        assert t["status"] == "Pending"

        def failing_audit(*args, **kwargs):
            raise RuntimeError("Simulated Database I/O Failure during audit logging!")

        monkeypatch.setattr(tmp_db, "record_lifecycle_audit", failing_audit)

        with pytest.raises(RuntimeError, match="Simulated Database I/O Failure"):
            tmp_db.update_repair_task_status(task_id, "In Progress")

        # Status must have rolled back to 'Pending'
        t_after = dict(tmp_db.get_repair_task_by_id(task_id))
        assert t_after["status"] == "Pending"


# ═════════════════════════════════════════════════════════════════════════════
#  4. CONCURRENCY SAFETY
# ═════════════════════════════════════════════════════════════════════════════

class TestConcurrencySafety:
    def test_concurrent_task_creation_for_same_event_produces_exactly_one_task(self, tmp_db, sample_network):
        """
        Under concurrent execution, UNIQUE(event_id) constraint and transaction handling
        ensure that exactly one task is inserted and zero duplicates are created.
        """
        p1 = sample_network["p1"]
        ev_id, _ = tmp_db.record_leak_event(p1, "critical", "Rupture")

        results = []
        def try_create(thread_id):
            return tmp_db.create_or_update_repair_task(ev_id, p1, 70.0 + thread_id, "MEDIUM")

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(try_create, i) for i in range(5)]
            for f in concurrent.futures.as_completed(futures):
                results.append(f.result())

        # All workers received the exact same task_id
        task_ids = {r[0] for r in results}
        assert len(task_ids) == 1

        # Exactly 1 row exists in repair_tasks
        with tmp_db._connect() as conn:
            cnt = conn.execute("SELECT COUNT(*) FROM repair_tasks WHERE event_id = ?", (ev_id,)).fetchone()[0]
            assert cnt == 1


# ═════════════════════════════════════════════════════════════════════════════
#  5. LIVE DATABASE READ-ONLY SAFETY CHECK
# ═════════════════════════════════════════════════════════════════════════════

def test_live_database_remains_unmodified():
    """Verify live database instance/aquagrid.db records were untouched during the audit."""
    live_db_path = os.path.join(PROJECT_ROOT, "instance", "aquagrid.db")
    assert os.path.isfile(live_db_path), f"Live DB not found at {live_db_path}"

    conn = sqlite3.connect(f"file:{live_db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        # Integrity checks
        integ = conn.execute("PRAGMA integrity_check").fetchall()
        assert integ[0][0] == "ok"
        fk_check = conn.execute("PRAGMA foreign_key_check").fetchall()
        assert len(fk_check) == 0

        # Exact historical record counts
        events = conn.execute("SELECT * FROM leak_events ORDER BY event_id ASC").fetchall()
        assert len(events) == 2
        assert events[0]["event_id"] == 1 and events[0]["status"] == "resolved"
        assert events[1]["event_id"] == 2 and events[1]["status"] == "open"

        tasks = conn.execute("SELECT * FROM repair_tasks ORDER BY task_id ASC").fetchall()
        assert len(tasks) == 2
        assert tasks[0]["task_id"] == 1 and tasks[0]["status"] == "Completed"
        assert tasks[1]["task_id"] == 2 and tasks[1]["status"] == "Pending"

        audits = conn.execute("SELECT * FROM lifecycle_audit_log").fetchall()
        assert len(audits) == 4
    finally:
        conn.close()

