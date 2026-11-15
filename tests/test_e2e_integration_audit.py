"""
tests/test_e2e_integration_audit.py – Day 7: End-to-End System Integration and Regression Audit.

Validates end-to-end multi-module integration across:
- Scenario A: Normal Network (provenance, baseline, zero false alarms, BFS/Dijkstra)
- Scenario B: Persistent Suspected Leak (detection, scoring, Max Heap ranking, audit logging)
- Scenario C: Investigation and Resolution (decoupled lifecycle, active exclusion, GET idempotency)
- Scenario D: Fresh Incident After Resolution (fresh baseline, new event/task, historical isolation)
- Scenario E: Network Connectivity Changes (pipe deactivation, bypass routing, dynamic cost updates)
- Cross-module referential and state consistency
- HTTP route read/write safety and user workflow validation
"""

from __future__ import annotations

import json
from datetime import datetime, timezone, timedelta
import pytest

import database as db_module
import app as app_module
from simulation.engine import SensorSimulator
from detection.engine import LeakDetector
from repair_queue.manager import PriorityQueueManager
from dsa.graph import load_graph_from_db
from dsa.dijkstra import dijkstra


@pytest.fixture()
def isolated_db(monkeypatch, tmp_path):
    """Isolated SQLite database in a temporary directory for integration tests."""
    test_db = str(tmp_path / "test_e2e_audit.db")
    monkeypatch.setattr(db_module, "INSTANCE_DIR", str(tmp_path))
    monkeypatch.setattr(db_module, "DB_PATH", test_db)
    db_module.init_db()
    return db_module


@pytest.fixture()
def app_client(isolated_db, monkeypatch):
    """Flask test client wired to the isolated database."""
    monkeypatch.setattr(app_module, "db", isolated_db)
    app_module.app.config["TESTING"] = True
    app_module.app.config["WTF_CSRF_ENABLED"] = False
    with app_module.app.test_client() as client:
        yield client


# ==============================================================================
# End-to-End Scenario Tests (A - E)
# ==============================================================================

class TestEndToEndScenarios:
    """End-to-End validation across the 5 primary operational scenarios."""

    def test_e2e_scenario_a_normal_network(self, isolated_db, app_client):
        """Scenario A: Normal network produces zero false alarms and valid graph reachability."""
        n1 = isolated_db.create_node("Res-North", "tank", "North Zone", "active")
        n2 = isolated_db.create_node("Junc-Mid", "junction", "Central Zone", "active")
        n3 = isolated_db.create_node("Valve-South", "valve", "South Zone", "active")

        p1 = isolated_db.create_pipe("Trunk-Main", n1, n2, "1000.0", "3000.0", "10.0", "active")
        p2 = isolated_db.create_pipe("Branch-Sub", n2, n3, "500.0", "1500.0", "5.0", "active")

        s1_flow = isolated_db.create_sensor("FM-Trunk", p1, "flow_meter", "L/min", "active")
        s1_press = isolated_db.create_sensor("PS-Trunk", p1, "pressure_sensor", "psi", "active")
        s2_flow = isolated_db.create_sensor("FM-Branch", p2, "flow_meter", "L/min", "active")

        sim = SensorSimulator(seed=100)
        active_sensors = isolated_db.get_active_sensors()
        start_ref = datetime.now(timezone.utc) - timedelta(hours=2)
        normal_readings = sim.generate_batch(active_sensors, mode="normal", iterations=25, base_time=start_ref)
        for r in normal_readings:
            isolated_db.record_sensor_reading(
                sensor_id=r.sensor_id,
                reading_value=r.reading_value,
                recorded_at=r.recorded_at,
                reading_type=r.sensor_type,
                unit=r.unit,
                is_simulated=True,
                scenario="normal",
            )

        # 1. Readings and provenance
        readings_flow = isolated_db.get_readings_for_sensor(s1_flow, limit=50)
        assert len(readings_flow) == 25
        assert all(r["is_simulated"] == 1 and r["scenario"] == "normal" for r in readings_flow)

        # 2. Detection engine
        detector = LeakDetector()
        diagnostics = detector.run_detection_network(isolated_db, auto_record=True)
        diag_p1 = next(d for d in diagnostics if d.pipe_id == p1)
        assert diag_p1.status == "normal"
        assert diag_p1.baseline_flow_samples == 20
        assert diag_p1.window_flow_samples == 5

        # 3. Zero false alarms
        assert len(isolated_db.get_open_leak_events()) == 0
        assert len(isolated_db.get_all_repair_tasks()) == 0

        # 4. Graph exploration & routing
        g = load_graph_from_db(db_module=isolated_db, directed=False)
        assert g.breadth_first_search(n1) == [n1, n2, n3]
        route_res = dijkstra(g, n1, n3)
        assert route_res.found is True
        assert route_res.total_cost == pytest.approx(15.0)

    def test_e2e_scenarios_b_and_c_leak_investigation_and_resolution(self, isolated_db, app_client):
        """Scenarios B & C: Persistent leak detection, priority queue, investigation, resolution, and GET idempotency."""
        n1 = isolated_db.create_node("Res-1", "tank", "North Zone", "active")
        n2 = isolated_db.create_node("Junc-2", "junction", "Central Zone", "active")
        p1 = isolated_db.create_pipe("Trunk-P1", n1, n2, "1000.0", "3000.0", "10.0", "active")
        isolated_db.create_sensor("FM-1", p1, "flow_meter", "L/min", "active")
        isolated_db.create_sensor("PS-1", p1, "pressure_sensor", "psi", "active")

        sim = SensorSimulator(seed=200)
        active_sensors = isolated_db.get_active_sensors()

        # Step 1: Normal readings
        t_base = datetime.now(timezone.utc) - timedelta(hours=2)
        normal_readings = sim.generate_batch(active_sensors, mode="normal", iterations=20, base_time=t_base)
        for r in normal_readings:
            isolated_db.record_sensor_reading(r.sensor_id, r.reading_value, r.recorded_at, r.sensor_type, r.unit, True, "normal")

        # Step 2: Leak readings (Scenario B)
        t_leak = t_base + timedelta(minutes=15)
        leak_readings = sim.generate_batch(active_sensors, mode="leak", leak_pipe_id=p1, iterations=10, base_time=t_leak)
        for r in leak_readings:
            isolated_db.record_sensor_reading(r.sensor_id, r.reading_value, r.recorded_at, r.sensor_type, r.unit, True, "leak")

        # Run detection via POST
        rv_run = app_client.post("/leak-detection/run", follow_redirects=True)
        assert rv_run.status_code == 200

        # Verify event and task creation
        open_events = isolated_db.get_open_leak_events()
        assert len(open_events) == 1
        ev = dict(open_events[0])
        assert ev["pipe_id"] == p1
        assert ev["severity"] == "critical"

        active_tasks = isolated_db.get_active_repair_tasks()
        assert len(active_tasks) == 1
        task = dict(active_tasks[0])
        assert task["event_id"] == ev["event_id"]
        assert task["priority_category"] == "CRITICAL"
        assert task["status"] == "Pending"

        # Max Heap orders it at top
        mgr = PriorityQueueManager()
        ordered = mgr.get_ordered_tasks(isolated_db)
        assert len(ordered) == 1
        assert ordered[0].task_id == task["task_id"]

        # Audit history verified
        audit_logs = isolated_db.get_lifecycle_audit_logs()
        assert len(audit_logs) == 2

        # Step 3: Investigation (Scenario C)
        task_id = task["task_id"]
        ev_id = ev["event_id"]
        rv_inv = app_client.post(f"/repairs/{task_id}/investigate", follow_redirects=True)
        assert rv_inv.status_code == 200
        assert isolated_db.get_repair_task_by_id(task_id)["status"] == "In Progress"
        assert isolated_db.get_leak_event_by_id(ev_id)["status"] == "investigating"

        # Complete repair task
        rv_comp = app_client.post(f"/repairs/{task_id}/status", data={"status": "Completed"}, follow_redirects=True)
        assert rv_comp.status_code == 200
        assert isolated_db.get_repair_task_by_id(task_id)["status"] == "Completed"
        assert len(mgr.get_ordered_tasks(isolated_db)) == 0 # Excluded from active queue
        assert isolated_db.get_leak_event_by_id(ev_id)["status"] == "investigating" # Decoupled lifecycle

        # Resolve leak event
        rv_res = app_client.post(f"/leak-detection/events/{ev_id}/status", data={"status": "resolved"}, follow_redirects=True)
        assert rv_res.status_code == 200
        assert isolated_db.get_leak_event_by_id(ev_id)["status"] == "resolved"

        # Repeated GET requests are idempotent
        for _ in range(5):
            rv_get = app_client.get("/leak-detection")
            assert rv_get.status_code == 200
        assert len(isolated_db.get_all_leak_events()) == 1

    def test_e2e_scenario_d_fresh_incident_after_resolution(self, isolated_db, app_client):
        """Scenario D: Genuinely fresh anomaly after resolution forms a new event and task without contaminating baseline."""
        n1 = isolated_db.create_node("Res-1", "tank", "North Zone", "active")
        n2 = isolated_db.create_node("Junc-2", "junction", "Central Zone", "active")
        p1 = isolated_db.create_pipe("Trunk-P1", n1, n2, "1000.0", "3000.0", "10.0", "active")
        isolated_db.create_sensor("FM-1", p1, "flow_meter", "L/min", "active")
        isolated_db.create_sensor("PS-1", p1, "pressure_sensor", "psi", "active")

        sim = SensorSimulator(seed=300)
        active_sensors = isolated_db.get_active_sensors()

        # Initial incident and resolution in the past
        t_past = datetime.now(timezone.utc) - timedelta(hours=2)
        norm1 = sim.generate_batch(active_sensors, mode="normal", iterations=20, base_time=t_past)
        for r in norm1:
            isolated_db.record_sensor_reading(r.sensor_id, r.reading_value, r.recorded_at, r.sensor_type, r.unit, True, "normal")

        leak1 = sim.generate_batch(active_sensors, mode="leak", leak_pipe_id=p1, iterations=10, base_time=t_past + timedelta(minutes=15))
        for r in leak1:
            isolated_db.record_sensor_reading(r.sensor_id, r.reading_value, r.recorded_at, r.sensor_type, r.unit, True, "leak")

        app_client.post("/leak-detection/run")
        ev1 = isolated_db.get_open_leak_events()[0]["event_id"]
        app_client.post(f"/leak-detection/events/{ev1}/status", data={"status": "resolved"})
        assert isolated_db.get_leak_event_by_id(ev1)["status"] == "resolved"

        # Fresh incident strictly after resolution
        now_utc = datetime.now(timezone.utc)
        fresh_norm = sim.generate_batch(active_sensors, mode="normal", iterations=10, base_time=now_utc + timedelta(seconds=10))
        for r in fresh_norm:
            isolated_db.record_sensor_reading(r.sensor_id, r.reading_value, r.recorded_at, r.sensor_type, r.unit, True, "normal")

        fresh_leak = sim.generate_batch(active_sensors, mode="leak", leak_pipe_id=p1, iterations=10, base_time=now_utc + timedelta(minutes=10))
        for r in fresh_leak:
            isolated_db.record_sensor_reading(r.sensor_id, r.reading_value, r.recorded_at, r.sensor_type, r.unit, True, "leak")

        app_client.post("/leak-detection/run")

        all_events = isolated_db.get_all_leak_events()
        assert len(all_events) == 2
        e_old = next(e for e in all_events if e["event_id"] == ev1)
        e_new = next(e for e in all_events if e["event_id"] != ev1)
        assert e_old["status"] == "resolved"
        assert e_new["status"] == "open"

        all_tasks = isolated_db.get_all_repair_tasks()
        assert len(all_tasks) == 2
        t_new = next(t for t in all_tasks if t["event_id"] == e_new["event_id"])
        assert t_new["status"] == "Pending"

    def test_e2e_scenario_e_network_connectivity_changes(self, isolated_db):
        """Scenario E: Pipe deactivation, dynamic bypass routing, and cost updates."""
        n1 = isolated_db.create_node("Res-1", "tank", "North", "active")
        n2 = isolated_db.create_node("Junc-2", "junction", "Central", "active")
        n3 = isolated_db.create_node("Valve-3", "valve", "South", "active")

        p1 = isolated_db.create_pipe("P1", n1, n2, "1000.0", "3000.0", "10.0", "active")
        p2 = isolated_db.create_pipe("P2", n2, n3, "500.0", "1500.0", "5.0", "active")

        g1 = load_graph_from_db(db_module=isolated_db, directed=False)
        assert dijkstra(g1, n1, n3).total_cost == pytest.approx(15.0)

        # Deactivate P1
        isolated_db.update_pipe(p1, "P1", n1, n2, "1000.0", "3000.0", "10.0", "inactive")
        g2 = load_graph_from_db(db_module=isolated_db, directed=False)
        assert dijkstra(g2, n1, n3).found is False

        # Add bypass P3
        p3 = isolated_db.create_pipe("P3-Bypass", n1, n3, "800.0", "2000.0", "8.0", "active")
        g3 = load_graph_from_db(db_module=isolated_db, directed=False)
        assert dijkstra(g3, n1, n3).total_cost == pytest.approx(8.0)


# ==============================================================================
# Cross-Module Data Consistency & HTTP Workflow
# ==============================================================================

class TestCrossModuleConsistencyAndHTTPWorkflow:
    """Verifies relational invariants across tables, KPI counts, and HTTP safety."""

    def test_cross_module_referential_consistency(self, isolated_db):
        """Ensures integrity across nodes, pipes, sensors, readings, leak events, and repair tasks."""
        n1 = isolated_db.create_node("N1", "tank", "North", "active")
        n2 = isolated_db.create_node("N2", "junction", "Central", "active")
        p1 = isolated_db.create_pipe("P1", n1, n2, "100.0", "1000.0", "1.0", "active")
        s1 = isolated_db.create_sensor("S1", p1, "flow_meter", "L/min", "active")

        isolated_db.record_sensor_reading(s1, 100.0, datetime.now(timezone.utc).isoformat(), "flow_meter", "L/min", True, "normal")
        ev_id, _ = isolated_db.record_leak_event(p1, "high", "Test anomaly", flow_anomaly_score=2.5, is_simulated=True)
        task_id, _ = isolated_db.create_or_update_repair_task(ev_id, p1, 75.0, "HIGH")

        # Foreign keys check
        with isolated_db._connect() as conn:
            fk_violations = conn.execute("PRAGMA foreign_key_check").fetchall()
            assert len(fk_violations) == 0

        # Score breakdown matches priority score
        task = isolated_db.get_repair_task_by_id(task_id)
        assert task["priority_score"] == pytest.approx(75.0)

    def test_http_get_requests_are_strictly_read_only(self, isolated_db, app_client):
        """Validates that visiting dashboard, repairs, graph, and leak-detection via GET performs 0 DB mutations."""
        n1 = isolated_db.create_node("N1", "tank", "North", "active")
        n2 = isolated_db.create_node("N2", "junction", "Central", "active")
        isolated_db.create_pipe("P1", n1, n2, "100.0", "1000.0", "1.0", "active")

        # Baseline counts
        init_events = isolated_db.count_total_leak_events()
        init_tasks = isolated_db.count_active_repair_tasks()
        init_audits = len(isolated_db.get_lifecycle_audit_logs())

        # Repeated GET requests across views
        for url in ["/", "/network", "/pipes", "/sensors", "/sensor-simulator", "/leak-detection", "/repairs", "/graph", "/route"]:
            rv = app_client.get(url)
            assert rv.status_code == 200

        # Post-request counts must be completely identical
        assert isolated_db.count_total_leak_events() == init_events
        assert isolated_db.count_active_repair_tasks() == init_tasks
        assert len(isolated_db.get_lifecycle_audit_logs()) == init_audits

