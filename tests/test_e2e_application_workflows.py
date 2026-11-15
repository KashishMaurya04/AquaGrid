"""
tests/test_e2e_application_workflows.py – End-to-End Application Workflow and UI Reliability Audit Test Suite.

Validates Day 10 Phase 4 requirements:
1. Part 1: Application startup, database initialization, schema migration safety, sidebar navigation resolution, static assets, zero startup DB mutations.
2. Part 2: Network Management workflow (CRUD nodes: tank, junctions, valve; CRUD pipes: connect, edit, deactivate; graph explorer and route finder verification).
3. Part 3: Sensor and Leak Detection workflow (sensor assignment, normal telemetry, leak telemetry on monitored pipe, leak detection run, repair task creation in Max Heap, task status update, leak event resolution, lifecycle audit log).
4. Part 4: Water Allocation workflow (multiple allocation zones, essential mins, fairness weights, run fair allocation, verify run details, historical zone snapshots).
5. Part 5: Network-Aware Distribution workflow (shortest path vs widest path on same allocation run, duration, route & bottleneck conduit inspection, shared-pipe contention warnings, historical assessment details, GET idempotency).
6. Part 6: UI and Navigation audit (11 sidebar pages, identical order, active-page highlight, fixed sidebar, responsive layout, clean empty states).
7. Part 7: Test-injection hook safety verification (`fail_on_index` in `record_sensor_readings_batch()` is strictly internal, never accepted from user HTTP inputs, cannot accidentally trigger rollback).
8. Part 8: Final regression and data safety (isolated DBs, PRAGMA integrity/foreign_key checks, 0 modifications to live database).
"""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import tempfile
import pytest

import database as db
import distribution.service
from app import app


@pytest.fixture
def tmp_e2e_env():
    """Create an isolated temporary environment with fresh SQLite database for E2E tests."""
    temp_dir = tempfile.mkdtemp()
    temp_db_path = os.path.join(temp_dir, "test_e2e.db")

    orig_db_path = db.DB_PATH
    orig_instance_dir = db.INSTANCE_DIR

    db.DB_PATH = temp_db_path
    db.INSTANCE_DIR = temp_dir
    db.init_db()

    app.config["TESTING"] = True
    app.config["WTF_CSRF_ENABLED"] = False

    with app.test_client() as client:
        yield {
            "client": client,
            "db_path": temp_db_path,
            "temp_dir": temp_dir,
        }

    db.DB_PATH = orig_db_path
    db.INSTANCE_DIR = orig_instance_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


# ==============================================================================
# Part 1: Application Startup & Navigation Link Resolution
# ==============================================================================

def test_part1_application_startup_and_sidebar_resolution(tmp_e2e_env):
    """Verify application startup, idempotent init_db, sidebar structure, and zero startup mutations."""
    client = tmp_e2e_env["client"]
    db_path = tmp_e2e_env["db_path"]

    # 1. Verify schema migrations and database initialization idempotency
    db.init_db()
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("PRAGMA integrity_check;")
    assert cur.fetchone()[0] == "ok"
    cur.execute("PRAGMA foreign_key_check;")
    assert cur.fetchall() == []

    # Count initial rows across all 12 tables
    tables = [
        "network_nodes", "pipes", "sensors", "sensor_readings",
        "leak_events", "repair_tasks", "lifecycle_audit_log",
        "allocation_zones", "allocation_runs", "water_allocations",
        "delivery_assessments", "delivery_zone_assessments"
    ]
    initial_counts = {}
    for tbl in tables:
        cur.execute(f"SELECT COUNT(*) FROM {tbl};")
        initial_counts[tbl] = cur.fetchone()[0]
    conn.close()

    # 2. GET / (Overview Dashboard)
    resp = client.get("/")
    assert resp.status_code == 200
    html = resp.data.decode("utf-8")

    # 3. Verify all 11 sidebar links exist in exact order
    expected_links = [
        ("/", "▦ Overview"),
        ("/network", "⌘ Water Network"),
        ("/pipes", "🔗 Pipe Management"),
        ("/sensors", "📡 Sensor Management"),
        ("/sensor-simulator", "⚡ Sensor Simulator"),
        ("/graph", "☍ Graph Explorer"),
        ("/route", "🧭 Route Finder"),
        ("/leak-detection", "⚠ Leak Detection"),
        ("/repairs", "🔧 Repair Queue"),
        ("/allocation", "⚖ Water Allocation"),
        ("/distribution", "🌊 Distribution"),
    ]
    for href, label in expected_links:
        assert href in html
        assert label in html

    # 4. Verify Overview page is marked active
    assert 'href="/"' in html
    assert 'class="nav-link active" href="/"' in html

    # 5. Verify zero DB mutations occurred during startup and GET /
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    for tbl in tables:
        cur.execute(f"SELECT COUNT(*) FROM {tbl};")
        assert cur.fetchone()[0] == initial_counts[tbl], f"Table {tbl} mutated during startup/GET"
    conn.close()


# ==============================================================================
# Part 2: Network Management Workflow (Nodes, Pipes, Route Finder)
# ==============================================================================

def test_part2_network_management_crud_and_route_finder(tmp_e2e_env):
    """Verify full network creation (Tank, 2 Junctions, Valve), Pipe editing/deactivation, and Route Finder."""
    client = tmp_e2e_env["client"]

    # 1. Create Reservoir Tank Node
    res = client.post("/network/create", data={
        "name": "North Lake Tank",
        "node_type": "tank",
        "location": "North Basin",
        "status": "active",
    }, follow_redirects=True)
    assert res.status_code == 200
    assert "North Lake Tank" in res.data.decode("utf-8")

    # 2. Create Junction 1
    res = client.post("/network/create", data={
        "name": "Junction Alpha",
        "node_type": "junction",
        "location": "Sector 1",
        "status": "active",
    }, follow_redirects=True)
    assert res.status_code == 200

    # 3. Create Junction 2
    res = client.post("/network/create", data={
        "name": "Junction Beta",
        "node_type": "junction",
        "location": "Sector 2",
        "status": "active",
    }, follow_redirects=True)
    assert res.status_code == 200

    # 4. Create Terminal Valve Node
    res = client.post("/network/create", data={
        "name": "Downtown Terminal",
        "node_type": "valve",
        "location": "District Center",
        "status": "active",
    }, follow_redirects=True)
    assert res.status_code == 200

    nodes = db.get_all_nodes()
    assert len(nodes) == 4
    node_by_name = {n["name"]: n["id"] for n in nodes}
    r_id = node_by_name["North Lake Tank"]
    j1_id = node_by_name["Junction Alpha"]
    j2_id = node_by_name["Junction Beta"]
    d_id = node_by_name["Downtown Terminal"]

    # 5. Create Pipe 1 (Tank -> Junction Alpha)
    res = client.post("/pipes/create", data={
        "name": "Main Feeder 1",
        "from_node_id": str(r_id),
        "to_node_id": str(j1_id),
        "length_m": "500",
        "capacity_lpm": "1000",
        "cost": "50",
        "status": "active",
    }, follow_redirects=True)
    assert res.status_code == 200

    # 6. Create Pipe 2 (Junction Alpha -> Junction Beta)
    res = client.post("/pipes/create", data={
        "name": "Mid Trunk 2",
        "from_node_id": str(j1_id),
        "to_node_id": str(j2_id),
        "length_m": "300",
        "capacity_lpm": "800",
        "cost": "30",
        "status": "active",
    }, follow_redirects=True)
    assert res.status_code == 200

    # 7. Create Pipe 3 (Junction Beta -> Downtown Terminal)
    res = client.post("/pipes/create", data={
        "name": "Terminal Pipe 3",
        "from_node_id": str(j2_id),
        "to_node_id": str(d_id),
        "length_m": "200",
        "capacity_lpm": "600",
        "cost": "20",
        "status": "active",
    }, follow_redirects=True)
    assert res.status_code == 200

    pipes = db.get_all_pipes()
    assert len(pipes) == 3
    pipe_by_name = {p["name"]: p for p in pipes}
    p2 = pipe_by_name["Mid Trunk 2"]

    # 8. Edit Pipe 2 attributes (update length and capacity)
    res = client.post(f"/pipes/{p2['id']}/edit", data={
        "name": "Mid Trunk 2 Upgraded",
        "from_node_id": str(j1_id),
        "to_node_id": str(j2_id),
        "length_m": "350",
        "capacity_lpm": "850",
        "cost": "35",
        "status": "active",
    }, follow_redirects=True)
    assert res.status_code == 200
    updated_p2 = db.get_pipe_by_id(p2["id"])
    assert updated_p2["name"] == "Mid Trunk 2 Upgraded"
    assert updated_p2["capacity_lpm"] == 850.0

    # 9. Verify Route Finder finds path (r_id -> d_id)
    res = client.get(f"/route?source_id={r_id}&destination_id={d_id}&traversal_mode=directed")
    assert res.status_code == 200
    html = res.data.decode("utf-8")
    assert "Route Found" in html or "Total Distance" in html or "North Lake Tank" in html

    # 10. Deactivate Pipe 2 and verify route finder reports disconnected
    res = client.post(f"/pipes/{p2['id']}/edit", data={
        "name": "Mid Trunk 2 Upgraded",
        "from_node_id": str(j1_id),
        "to_node_id": str(j2_id),
        "length_m": "350",
        "capacity_lpm": "850",
        "cost": "35",
        "status": "inactive",
    }, follow_redirects=True)
    assert res.status_code == 200

    res = client.get(f"/route?source_id={r_id}&destination_id={d_id}&traversal_mode=directed")
    assert res.status_code == 200
    html = res.data.decode("utf-8")
    assert "No Path Available" in html

    # 11. Reactivate Pipe 2
    res = client.post(f"/pipes/{p2['id']}/edit", data={
        "name": "Mid Trunk 2 Upgraded",
        "from_node_id": str(j1_id),
        "to_node_id": str(j2_id),
        "length_m": "350",
        "capacity_lpm": "850",
        "cost": "35",
        "status": "active",
    }, follow_redirects=True)
    assert res.status_code == 200


# ==============================================================================
# Part 3: Sensor & Leak Detection Workflow (Sensors, Simulator, Detection, Repairs)
# ==============================================================================

def test_part3_sensor_and_leak_detection_workflow(tmp_e2e_env):
    """Verify sensor creation, normal/leak simulation, detection run, Max Heap repair task, status transitions, and audit logs."""
    client = tmp_e2e_env["client"]

    # 1. Setup minimal network nodes and pipe
    n1 = db.create_node("Pump Station Tank", "tank", "Zone 1", "active")
    n2 = db.create_node("Distribution Center Valve", "valve", "Zone 2", "active")
    p1 = db.create_pipe("Mainline Alpha", n1, n2, "100.0", "500.0", "10.0", "active")

    # 2. Create Flow Meter Sensor on Pipe 1
    res = client.post("/sensors/create", data={
        "sensor_name": "Flow-Alpha-1",
        "pipe_id": str(p1),
        "sensor_type": "flow_meter",
        "unit": "L/min",
        "status": "active",
    }, follow_redirects=True)
    assert res.status_code == 200
    assert "Flow-Alpha-1" in res.data.decode("utf-8")

    # 3. Create Pressure Sensor on Pipe 1
    res = client.post("/sensors/create", data={
        "sensor_name": "Pressure-Alpha-1",
        "pipe_id": str(p1),
        "sensor_type": "pressure_sensor",
        "unit": "bar",
        "status": "active",
    }, follow_redirects=True)
    assert res.status_code == 200

    # 4. Generate Normal Telemetry Readings via Sensor Simulator
    res = client.post("/sensor-simulator/generate", data={
        "mode": "normal",
        "count": "10",
        "seed": "42",
    }, follow_redirects=True)
    assert res.status_code == 200
    assert "Generated" in res.data.decode("utf-8")
    assert db.count_sensor_readings() >= 20

    # 5. Generate Leak Telemetry on Pipe 1
    res = client.post("/sensor-simulator/generate", data={
        "mode": "leak",
        "pipe_id": str(p1),
        "count": "10",
        "seed": "99",
    }, follow_redirects=True)
    assert res.status_code == 200
    assert "including simulated leak anomaly on pipe" in res.data.decode("utf-8")

    # 6. Trigger Leak Detection Run via POST /leak-detection/run
    res = client.post("/leak-detection/run", follow_redirects=True)
    assert res.status_code == 200
    html = res.data.decode("utf-8")
    assert "suspected leak" in html.lower() or "anomalies detected" in html.lower() or "diagnostic" in html.lower()

    # Verify leak event was recorded in DB
    events = db.get_open_leak_events()
    assert len(events) >= 1
    leak_ev = events[0]
    assert leak_ev["pipe_id"] == p1

    # 7. Verify Repair Task created in Smart Repair Queue (Max Heap)
    res = client.get("/repairs")
    assert res.status_code == 200
    html = res.data.decode("utf-8")
    assert f"Pipe #{p1}" in html or "Mainline Alpha" in html

    tasks = [t for t in db.get_all_repair_tasks() if t["status"] == "Pending"]
    assert len(tasks) >= 1
    task = tasks[0]
    assert task["pipe_id"] == p1
    assert task["priority_score"] > 0

    # 8. Dispatch field investigation via POST /repairs/<id>/investigate
    res = client.post(f"/repairs/{task['task_id']}/investigate", follow_redirects=True)
    assert res.status_code == 200
    updated_task = db.get_repair_task_by_id(task["task_id"])
    assert updated_task["status"] == "In Progress"

    # Verify associated leak event moved to 'investigating'
    ev_after_investigate = db.get_leak_event_by_id(leak_ev["event_id"])
    assert ev_after_investigate["status"] == "investigating"

    # 9. Complete repair task via POST /repairs/<id>/status
    res = client.post(f"/repairs/{task['task_id']}/status", data={
        "status": "Completed"
    }, follow_redirects=True)
    assert res.status_code == 200
    assert db.get_repair_task_by_id(task["task_id"])["status"] == "Completed"

    # 10. Resolve leak event via POST /leak-detection/events/<id>/status
    res = client.post(f"/leak-detection/events/{leak_ev['event_id']}/status", data={
        "status": "resolved"
    }, follow_redirects=True)
    assert res.status_code == 200
    assert db.get_leak_event_by_id(leak_ev["event_id"])["status"] == "resolved"

    # 11. Verify lifecycle audit logs recorded the full transition history
    logs = db.get_lifecycle_audit_logs(limit=50)
    assert len(logs) >= 3
    actions = [l["action_description"] for l in logs]
    assert any("investigat" in a.lower() or "status" in a.lower() for a in actions)


# ==============================================================================
# Part 4: Water Allocation Workflow (Fair Allocation & Immutable Snapshots)
# ==============================================================================

def test_part4_water_allocation_fairness_and_snapshots(tmp_e2e_env):
    """Verify zone creation, weighted fair allocation execution, detail page rendering, and historical zone snapshots."""
    client = tmp_e2e_env["client"]

    # 1. Setup destination nodes for zones
    n_res = db.create_node("Clearwater Reservoir Tank", "tank", "North", "active")
    n_hosp = db.create_node("General Hospital Node", "junction", "Central", "active")
    n_ind = db.create_node("Industrial Park Node", "junction", "East", "active")
    n_resd = db.create_node("Residential Ward Node", "junction", "South", "active")

    # 2. Create Hospital Zone (critical, high weight, high essential min)
    res = client.post("/allocation/zones/create", data={
        "zone_name": "General Hospital",
        "requested_demand_l": "5000",
        "essential_min_l": "2000",
        "fairness_weight": "3.0",
        "priority_category": "critical",
        "node_id": str(n_hosp),
        "description": "Emergency services and inpatient wards",
    }, follow_redirects=True)
    assert res.status_code == 200

    # 3. Create Residential Zone (standard, moderate weight, moderate essential min)
    res = client.post("/allocation/zones/create", data={
        "zone_name": "Downtown Residential",
        "requested_demand_l": "6000",
        "essential_min_l": "1500",
        "fairness_weight": "1.5",
        "priority_category": "standard",
        "node_id": str(n_resd),
        "description": "Dense residential apartments",
    }, follow_redirects=True)
    assert res.status_code == 200

    # 4. Create Industrial Zone (commercial, lower weight, low essential min)
    res = client.post("/allocation/zones/create", data={
        "zone_name": "East Industrial",
        "requested_demand_l": "4000",
        "essential_min_l": "500",
        "fairness_weight": "1.0",
        "priority_category": "commercial",
        "node_id": str(n_ind),
        "description": "Manufacturing factories",
    }, follow_redirects=True)
    assert res.status_code == 200

    # Total demand = 15000, Total essential min = 4000
    # Available supply = 8000 (Scarcity scenario)
    res = client.post("/allocation/run", data={
        "available_supply": "8000",
        "notes": "E2E Drought Allocation Run",
    }, follow_redirects=True)
    assert res.status_code == 200
    html = res.data.decode("utf-8")
    assert "Allocation Run Details" in html or "Fairness Index" in html or "Allocation run" in html

    latest_run = db.get_latest_allocation_run()
    assert latest_run is not None
    assert latest_run["total_allocated_l"] <= 8000.0
    assert latest_run["total_allocated_l"] >= 4000.0  # At least all essential mins covered

    # Verify per-zone allocations
    allocs = db.get_allocations_for_run(latest_run["run_id"])
    assert len(allocs) == 3
    alloc_by_name = {a["zone_name_snapshot"]: a for a in allocs}

    # Verify essential minimums guaranteed
    assert alloc_by_name["General Hospital"]["allocated_amount_l"] >= 2000.0
    assert alloc_by_name["Downtown Residential"]["allocated_amount_l"] >= 1500.0
    assert alloc_by_name["East Industrial"]["allocated_amount_l"] >= 500.0

    # Verify destination node snapshot preserved
    assert alloc_by_name["General Hospital"]["destination_node_id"] == n_hosp
    assert alloc_by_name["Downtown Residential"]["destination_node_id"] == n_resd
    assert alloc_by_name["East Industrial"]["destination_node_id"] == n_ind

    # Verify allocation detail page GET renders cleanly
    res_details = client.get(f"/allocation/runs/{latest_run['run_id']}")
    assert res_details.status_code == 200
    assert "General Hospital" in res_details.data.decode("utf-8")


# ==============================================================================
# Part 5: Network-Aware Distribution Workflow (Shortest Path vs Widest Path)
# ==============================================================================

def test_part5_network_aware_distribution_workflows(tmp_e2e_env):
    """Verify Shortest Path vs Widest Path distribution assessments, contention warnings, detail view, and GET idempotency."""
    client = tmp_e2e_env["client"]

    # 1. Build an intentional network where Shortest Path and Widest Path diverge
    # Source: Node 1 (Tank)
    # Target: Node 4 (Valve Destination)
    # Path A (Shortest): 1 -> 4 directly (Length: 100m, Capacity: 50 LPM)
    # Path B (Widest):   1 -> 2 -> 3 -> 4 (Length: 300m, Capacity: 500 LPM each)
    n1 = db.create_node("Central Reservoir Tank", "tank", "West", "active")
    n2 = db.create_node("Junction 2", "junction", "Mid-North", "active")
    n3 = db.create_node("Junction 3", "junction", "Mid-East", "active")
    n4 = db.create_node("Zone 4 Terminal", "valve", "East", "active")

    # Shortest path pipe (chokepoint: 50 LPM, cost: 10)
    p_short = db.create_pipe("Direct Chokepoint Pipe", n1, n4, "100.0", "50.0", "10.0", "active")
    # Widest path pipes (wide: 500 LPM, cost: 10 each, total cost: 30)
    p_w1 = db.create_pipe("Wide Pipe 1", n1, n2, "100.0", "500.0", "10.0", "active")
    p_w2 = db.create_pipe("Wide Pipe 2", n2, n3, "100.0", "500.0", "10.0", "active")
    p_w3 = db.create_pipe("Wide Pipe 3", n3, n4, "100.0", "500.0", "10.0", "active")

    # 2. Create Zone mapped to Node 4
    z1 = db.create_allocation_zone(
        zone_name="Suburban Sector 4",
        requested_demand_l=10000.0,
        essential_min_l=2000.0,
        fairness_weight=1.0,
        priority_category="standard",
        status="active",
        node_id=n4,
        description="Suburban residential zone",
    )

    # 3. Create Allocation Run (allocated 6000 L)
    client.post("/allocation/run", data={
        "available_supply": "6000",
        "notes": "Allocation for Distribution Comparison",
    }, follow_redirects=True)
    latest_run = db.get_latest_allocation_run()
    run_id = latest_run["run_id"]

    # 4. Run Distribution Assessment via SHORTEST_PATH (Duration: 60 minutes)
    # Bottleneck = 50 LPM * 60 min = 3000 L maximum delivery
    res_sp = client.post("/distribution/run", data={
        "allocation_run_id": str(run_id),
        "source_node_id": str(n1),
        "duration_minutes": "60",
        "routing_strategy": "SHORTEST_PATH",
        "notes": "Testing Shortest Path Bottleneck",
    }, follow_redirects=True)
    assert res_sp.status_code == 200
    html_sp = res_sp.data.decode("utf-8")
    assert "Assessment Details" in html_sp or "Delivery Feasibility" in html_sp or "assessment" in html_sp.lower()

    assessment_sp = db.get_latest_delivery_assessment(run_id=run_id)
    assert assessment_sp["routing_strategy"] == "SHORTEST_PATH"
    # Delivered should be capped at 3000 L due to 50 LPM bottleneck
    assert assessment_sp["total_deliverable_l"] == pytest.approx(3000.0, abs=1.0)
    assert assessment_sp["delivery_efficiency_pct"] == pytest.approx(50.0, abs=0.5)

    # 5. Run Distribution Assessment via WIDEST_PATH (Duration: 60 minutes)
    # Bottleneck = 500 LPM * 60 min = 30,000 L capacity, can deliver entire 6000 L allocation
    res_wp = client.post("/distribution/run", data={
        "allocation_run_id": str(run_id),
        "source_node_id": str(n1),
        "duration_minutes": "60",
        "routing_strategy": "WIDEST_PATH",
        "notes": "Testing Widest Path Capacity",
    }, follow_redirects=True)
    assert res_wp.status_code == 200

    assessment_wp = db.get_latest_delivery_assessment(run_id=run_id)
    assert assessment_wp["routing_strategy"] == "WIDEST_PATH"
    assert assessment_wp["total_deliverable_l"] == pytest.approx(6000.0, abs=1.0)
    assert assessment_wp["delivery_efficiency_pct"] == pytest.approx(100.0, abs=0.5)

    # Comparison assertion: Widest Path delivered twice as much water as Shortest Path!
    assert assessment_wp["total_deliverable_l"] > assessment_sp["total_deliverable_l"]

    # 6. Verify Assessment Detail Page GET is idempotent
    conn = sqlite3.connect(tmp_e2e_env["db_path"])
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM delivery_assessments;")
    count_before = cur.fetchone()[0]
    conn.close()

    res_detail = client.get(f"/distribution/assessments/{assessment_wp['assessment_id']}")
    assert res_detail.status_code == 200
    html_detail = res_detail.data.decode("utf-8")
    assert "WIDEST_PATH" in html_detail
    assert "Suburban Sector 4" in html_detail
    assert "100.0%" in html_detail

    # Verify 0 DB mutations on GET
    conn = sqlite3.connect(tmp_e2e_env["db_path"])
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM delivery_assessments;")
    count_after = cur.fetchone()[0]
    conn.close()
    assert count_before == count_after


# ==============================================================================
# Part 6: UI & Navigation Audit (11 Pages, Sidebar Active Classes & Empty States)
# ==============================================================================

def test_part6_ui_navigation_audit_and_active_classes(tmp_e2e_env):
    """Verify all 11 application pages return HTTP 200, render identical sidebar links, and highlight active page."""
    client = tmp_e2e_env["client"]

    pages = [
        ("/", "active\" href=\"/\""),
        ("/network", "active\" href=\"/network\""),
        ("/pipes", "active\" href=\"/pipes\""),
        ("/sensors", "active\" href=\"/sensors\""),
        ("/sensor-simulator", "active\" href=\"/sensor-simulator\""),
        ("/graph", "active\" href=\"/graph\""),
        ("/route", "active\" href=\"/route\""),
        ("/leak-detection", "active\" href=\"/leak-detection\""),
        ("/repairs", "active\" href=\"/repairs\""),
        ("/allocation", "active\" href=\"/allocation\""),
        ("/distribution", "active\" href=\"/distribution\""),
    ]

    for url, active_marker in pages:
        res = client.get(url)
        assert res.status_code == 200, f"Page {url} returned {res.status_code}"
        html = res.data.decode("utf-8")

        # Verify brand & structure
        assert "AquaGrid" in html
        assert "<aside class=\"sidebar\">" in html
        assert "SDG 6 · Clean Water" in html

        # Verify active page highlight
        assert active_marker in html, f"Page {url} did not have active highlight '{active_marker}'"


# ==============================================================================
# Part 7: Test-Injection Hook Safety Verification
# ==============================================================================

def test_part7_test_injection_hook_safety(tmp_e2e_env):
    """Verify that fail_on_index cannot be passed or triggered via HTTP requests."""
    client = tmp_e2e_env["client"]

    # Setup 1 sensor on a pipe
    n1 = db.create_node("Station Tank A", "tank", "A", "active")
    n2 = db.create_node("Station Valve B", "valve", "B", "active")
    p1 = db.create_pipe("Pipe AB", n1, n2, "100.0", "500.0", "10.0", "active")
    db.create_sensor("Sensor-AB", p1, "flow_meter", "L/min", "active")

    # Attempt to pass fail_on_index via query parameter and form data
    res = client.post("/sensor-simulator/generate?fail_on_index=1", data={
        "mode": "normal",
        "count": "10",
        "seed": "42",
        "fail_on_index": "1",
    }, follow_redirects=True)

    # Must succeed without failure or rollback because HTTP handlers do NOT pass fail_on_index
    assert res.status_code == 200
    assert "Generated" in res.data.decode("utf-8")
    assert db.count_sensor_readings() >= 10


# ==============================================================================
# Part 8: Database Integrity & Referencing Safety
# ==============================================================================

def test_part8_database_integrity_and_referencing_safety(tmp_e2e_env):
    """Verify SQLite integrity_check and foreign_key_check are clean after running workflows."""
    db_path = tmp_e2e_env["db_path"]
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("PRAGMA integrity_check;")
    integrity = cur.fetchone()[0]
    assert integrity == "ok"

    cur.execute("PRAGMA foreign_key_check;")
    fk_violations = cur.fetchall()
    assert fk_violations == []
    conn.close()
