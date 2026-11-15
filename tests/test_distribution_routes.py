"""
tests/test_distribution_routes.py – Comprehensive Integration and Route Tests for Water Distribution.

Validates Day 9 Phase 4 UI and Flask integration:
1. Distribution dashboard route (GET /distribution).
2. Successful assessment POST (POST /distribution/run).
3. Invalid form submissions (missing required parameters).
4. Invalid run/source selections (non-existent or inactive IDs).
5. Invalid duration and routing strategy.
6. Assessment detail page (GET /distribution/assessments/<id>).
7. Historical assessment list and run-specific history (GET /distribution/runs/<run_id>).
8. Unmapped destination display (explaining unmapped status without guessing).
9. Shared-pipe contention warning display.
10. Snapshot-based historical display after operational renaming.
11. Read-only GET behaviour (no mutations on GET).
12. No accidental duplicate assessment creation on repeated GET requests.
"""

from __future__ import annotations

import os
import sys
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import database as db_module
import app as app_module
from allocation.engine import ZoneDemandInput, allocate_water
from distribution.engine import STRATEGY_SHORTEST_PATH, STRATEGY_WIDEST_PATH


@pytest.fixture()
def isolated_db(monkeypatch, tmp_path):
    """Isolated SQLite database for distribution route integration tests."""
    test_db = str(tmp_path / "test_distribution_routes.db")
    monkeypatch.setattr(db_module, "INSTANCE_DIR", str(tmp_path))
    monkeypatch.setattr(db_module, "DB_PATH", test_db)
    db_module.init_db()
    return db_module


@pytest.fixture()
def client(isolated_db, monkeypatch):
    """Flask test client wired to the isolated database with CSRF disabled for test submissions."""
    monkeypatch.setattr(app_module, "db", isolated_db)
    app_module.app.config["TESTING"] = True
    app_module.app.config["WTF_CSRF_ENABLED"] = False
    with app_module.app.test_client() as c:
        yield c


@pytest.fixture()
def seeded_network(isolated_db):
    """
    Sets up a small distribution network with nodes, pipes, zones, and an allocation run:
        Source: Node 1 (Central Reservoir)
        Destinations: Node 2 (District Alpha), Node 3 (District Beta)
        Pipes: 1 -> 2 (cap 100 L/min), 1 -> 3 (cap 50 L/min)
        Zone 1: mapped to Node 2 (Allocated 1200 L)
        Zone 2: mapped to Node 3 (Allocated 1800 L)
    """
    db = isolated_db
    src_id = db.create_node("Central Reservoir", "tank", "North Sector", "active")
    node_a = db.create_node("District Alpha Node", "valve", "Sector Alpha", "active")
    node_b = db.create_node("District Beta Node", "valve", "Sector Beta", "active")

    pipe_a = db.create_pipe("Pipe-1-2", src_id, node_a, 100.0, 100.0, 1.0, "active")
    pipe_b = db.create_pipe("Pipe-1-3", src_id, node_b, 150.0, 50.0, 2.0, "active")

    z1 = db.create_allocation_zone(
        zone_name="Zone Alpha",
        requested_demand_l=1200.0,
        essential_min_l=600.0,
        fairness_weight=1.0,
        priority_category="standard",
        status="active",
        node_id=node_a,
    )
    z2 = db.create_allocation_zone(
        zone_name="Zone Beta",
        requested_demand_l=1800.0,
        essential_min_l=900.0,
        fairness_weight=1.5,
        priority_category="commercial",
        status="active",
        node_id=node_b,
    )

    inputs = [
        ZoneDemandInput(
            zone_id=z1,
            zone_name="Zone Alpha",
            requested_demand=1200.0,
            essential_min=600.0,
            fairness_weight=1.0,
        ),
        ZoneDemandInput(
            zone_id=z2,
            zone_name="Zone Beta",
            requested_demand=1800.0,
            essential_min=900.0,
            fairness_weight=1.5,
        ),
    ]
    alloc_res = allocate_water(total_supply=3000.0, zones=inputs)
    run_id = db.record_allocation_run(alloc_res, notes="Seeded run for route tests")

    return {
        "db": db,
        "src_id": src_id,
        "node_a": node_a,
        "node_b": node_b,
        "pipe_a": pipe_a,
        "pipe_b": pipe_b,
        "z1": z1,
        "z2": z2,
        "run_id": run_id,
    }


class TestDistributionDashboardRoute:
    """Tests GET /distribution."""

    def test_dashboard_renders_empty_state_cleanly(self, client, isolated_db):
        """GET /distribution renders with 200 and helpful empty state when no runs exist."""
        resp = client.get("/distribution")
        assert resp.status_code == 200
        html = resp.get_data(as_text=True)
        assert "Distribution Dashboard" in html
        assert "No allocation runs exist yet to evaluate" in html

    def test_dashboard_renders_with_seeded_runs(self, client, seeded_network):
        """GET /distribution renders configuration form with available runs and source nodes."""
        data = seeded_network
        resp = client.get("/distribution")
        assert resp.status_code == 200
        html = resp.get_data(as_text=True)
        assert "Distribution Dashboard" in html
        assert f"Run #{data['run_id']}" in html
        assert "Central Reservoir" in html
        assert "SHORTEST_PATH" in html
        assert "WIDEST_PATH" in html

    def test_dashboard_read_only_and_idempotent(self, client, seeded_network):
        """Repeated GET /distribution requests create no assessment records (read-only guarantee)."""
        db = seeded_network["db"]
        initial_count = db.count_delivery_assessments()

        for _ in range(3):
            resp = client.get("/distribution")
            assert resp.status_code == 200

        assert db.count_delivery_assessments() == initial_count


class TestDistributionRunExecutionRoutes:
    """Tests POST /distribution/run."""

    def test_successful_assessment_creation_and_redirect(self, client, seeded_network):
        """POST /distribution/run executes feasibility evaluation and redirects to assessment details."""
        data = seeded_network
        db = data["db"]

        post_data = {
            "allocation_run_id": str(data["run_id"]),
            "source_node_id": str(data["src_id"]),
            "duration_minutes": "60.0",
            "routing_strategy": "SHORTEST_PATH",
            "notes": "Automated route test run",
        }
        resp = client.post("/distribution/run", data=post_data, follow_redirects=False)
        assert resp.status_code == 302
        assert "/distribution/assessments/" in resp.location

        # Follow redirect and verify assessment page content
        follow_resp = client.get(resp.location)
        assert follow_resp.status_code == 200
        html = follow_resp.get_data(as_text=True)
        assert "Delivery Feasibility Assessment Report" in html
        assert "Central Reservoir" in html
        assert "Zone Alpha" in html
        assert "Zone Beta" in html

        # Verify DB row created
        assert db.count_delivery_assessments() == 1

    def test_missing_allocation_run_rejected(self, client, seeded_network):
        """Missing allocation run yields error and redirects."""
        post_data = {
            "source_node_id": str(seeded_network["src_id"]),
            "duration_minutes": "60",
            "routing_strategy": "SHORTEST_PATH",
        }
        resp = client.post("/distribution/run", data=post_data, follow_redirects=True)
        assert resp.status_code == 200
        assert "Allocation run selection is required" in resp.get_data(as_text=True)

    def test_nonexistent_allocation_run_rejected(self, client, seeded_network):
        """Non-existent allocation run ID is safely rejected."""
        post_data = {
            "allocation_run_id": "99999",
            "source_node_id": str(seeded_network["src_id"]),
            "duration_minutes": "60",
            "routing_strategy": "SHORTEST_PATH",
        }
        resp = client.post("/distribution/run", data=post_data, follow_redirects=True)
        assert resp.status_code == 200
        assert "does not exist" in resp.get_data(as_text=True)

    def test_missing_or_inactive_source_node_rejected(self, client, seeded_network):
        """Inactive source node is rejected with validation message."""
        data = seeded_network
        db = data["db"]
        inactive_src = db.create_node("Inactive Well", "tank", "West Sector", "inactive")

        post_data = {
            "allocation_run_id": str(data["run_id"]),
            "source_node_id": str(inactive_src),
            "duration_minutes": "60",
            "routing_strategy": "SHORTEST_PATH",
        }
        resp = client.post("/distribution/run", data=post_data, follow_redirects=True)
        assert resp.status_code == 200
        assert "is inactive" in resp.get_data(as_text=True)

    def test_invalid_duration_rejected(self, client, seeded_network):
        """Negative, zero, or non-numeric duration is rejected."""
        data = seeded_network
        for bad_dur in ["0", "-15", "abc", "NaN"]:
            post_data = {
                "allocation_run_id": str(data["run_id"]),
                "source_node_id": str(data["src_id"]),
                "duration_minutes": bad_dur,
                "routing_strategy": "SHORTEST_PATH",
            }
            resp = client.post("/distribution/run", data=post_data, follow_redirects=True)
            assert resp.status_code == 200
            assert "Delivery duration must be a" in resp.get_data(as_text=True)

    def test_invalid_routing_strategy_rejected(self, client, seeded_network):
        """Unsupported routing strategy name is rejected."""
        data = seeded_network
        post_data = {
            "allocation_run_id": str(data["run_id"]),
            "source_node_id": str(data["src_id"]),
            "duration_minutes": "60",
            "routing_strategy": "MAGIC_HYDRAULIC_PATH",
        }
        resp = client.post("/distribution/run", data=post_data, follow_redirects=True)
        assert resp.status_code == 200
        assert "Invalid routing strategy" in resp.get_data(as_text=True)


class TestHistoricalAssessmentDetailsAndSnapshots:
    """Tests GET /distribution/assessments/<assessment_id> and GET /distribution/runs/<run_id>."""

    def test_assessment_details_displays_persisted_snapshots(self, client, seeded_network):
        """Assessment details page renders snapshot fields and never recomputes."""
        data = seeded_network
        client.post(
            "/distribution/run",
            data={
                "allocation_run_id": str(data["run_id"]),
                "source_node_id": str(data["src_id"]),
                "duration_minutes": "60.0",
                "routing_strategy": "WIDEST_PATH",
                "notes": "Snapshot fidelity check",
            },
        )
        assess = data["db"].get_latest_delivery_assessment()
        assert assess is not None

        resp = client.get(f"/distribution/assessments/{assess['assessment_id']}")
        assert resp.status_code == 200
        html = resp.get_data(as_text=True)

        assert f"ASSESSMENT #{assess['assessment_id']}" in html
        assert "Central Reservoir" in html
        assert "Zone Alpha" in html
        assert "Zone Beta" in html
        assert "District Alpha Node" in html
        assert "District Beta Node" in html
        assert "WIDEST_PATH" in html

    def test_snapshots_preserved_after_operational_renaming(self, client, seeded_network):
        """
        Renaming source node, zone name, or destination node in live tables does NOT alter
        the displayed historical assessment details page.
        """
        data = seeded_network
        db = data["db"]

        # 1. Record assessment
        client.post(
            "/distribution/run",
            data={
                "allocation_run_id": str(data["run_id"]),
                "source_node_id": str(data["src_id"]),
                "duration_minutes": "60.0",
                "routing_strategy": "SHORTEST_PATH",
            },
        )
        assess = db.get_latest_delivery_assessment()
        assess_id = assess["assessment_id"]

        # 2. Mutate operational records in live tables
        db.update_node(data["src_id"], "Renamed Future Tank", "tank", "North Sector", "active")
        db.update_allocation_zone(
            data["z1"],
            "Future Renamed District Alpha",
            requested_demand_l=9999.0,
            essential_min_l=100.0,
            fairness_weight=1.0,
            priority_category="critical",
            status="active",
            node_id=data["node_a"],
        )
        db.update_node(data["node_a"], "Renamed Future Node A", "valve", "Sector Alpha", "active")

        # 3. View historical assessment details
        resp = client.get(f"/distribution/assessments/{assess_id}")
        assert resp.status_code == 200
        html = resp.get_data(as_text=True)

        # MUST show original snapshots
        assert "Central Reservoir" in html
        assert "Zone Alpha" in html
        assert "District Alpha Node" in html

        # MUST NOT show modified operational names
        assert "Renamed Future Tank" not in html
        assert "Future Renamed District Alpha" not in html
        assert "Renamed Future Node A" not in html

    def test_unmapped_destination_display_without_guessing(self, client, isolated_db):
        """A zone unmapped at allocation time explicitly displays UNMAPPED and explanation."""
        db = isolated_db
        src_id = db.create_node("Supply Tank", "tank", "North", "active")
        z_unmapped = db.create_allocation_zone(
            zone_name="Wilderness Camp",
            requested_demand_l=500.0,
            essential_min_l=200.0,
            fairness_weight=1.0,
            priority_category="standard",
            status="active",
            node_id=None,  # Explicitly unmapped
        )
        alloc_res = allocate_water(
            total_supply=500.0,
            zones=[
                ZoneDemandInput(
                    zone_id=z_unmapped,
                    zone_name="Wilderness Camp",
                    requested_demand=500.0,
                    essential_min=200.0,
                    fairness_weight=1.0,
                )
            ],
        )
        run_id = db.record_allocation_run(alloc_res)

        client.post(
            "/distribution/run",
            data={
                "allocation_run_id": str(run_id),
                "source_node_id": str(src_id),
                "duration_minutes": "60.0",
                "routing_strategy": "SHORTEST_PATH",
            },
        )
        assess = db.get_latest_delivery_assessment()
        resp = client.get(f"/distribution/assessments/{assess['assessment_id']}")
        assert resp.status_code == 200
        html = resp.get_data(as_text=True)

        assert "Wilderness Camp" in html
        assert "UNMAPPED" in html
        assert "not mapped to any physical network node" in html

    def test_shared_pipe_contention_warning_rendered(self, client, isolated_db):
        """
        When multiple zones share a bottleneck pipe that exceeds capacity,
        the shared-pipe contention banner is prominently displayed.
        """
        db = isolated_db
        src_id = db.create_node("Master Tank", "tank", "North", "active")
        junc_id = db.create_node("Junction Hub", "junction", "Center", "active")
        dest_1 = db.create_node("East Consumer", "valve", "East", "active")
        dest_2 = db.create_node("West Consumer", "valve", "West", "active")

        # Shared conduit: capacity is only 20 L/min!
        db.create_pipe("Shared-Trunk", src_id, junc_id, 100.0, 20.0, 1.0, "active")
        db.create_pipe("Branch-1", junc_id, dest_1, 50.0, 100.0, 1.0, "active")
        db.create_pipe("Branch-2", junc_id, dest_2, 50.0, 100.0, 1.0, "active")

        # Demand: each zone requires 600 L in 30 min (20 L/min each -> aggregate 40 L/min > 20 L/min cap)
        z1 = db.create_allocation_zone(
            zone_name="Zone East",
            requested_demand_l=600.0,
            essential_min_l=300.0,
            fairness_weight=1.0,
            priority_category="standard",
            status="active",
            node_id=dest_1,
        )
        z2 = db.create_allocation_zone(
            zone_name="Zone West",
            requested_demand_l=600.0,
            essential_min_l=300.0,
            fairness_weight=1.0,
            priority_category="standard",
            status="active",
            node_id=dest_2,
        )

        alloc_res = allocate_water(
            total_supply=1200.0,
            zones=[
                ZoneDemandInput(zone_id=z1, zone_name="Zone East", requested_demand=600.0, essential_min=300.0),
                ZoneDemandInput(zone_id=z2, zone_name="Zone West", requested_demand=600.0, essential_min=300.0),
            ],
        )
        run_id = db.record_allocation_run(alloc_res)

        client.post(
            "/distribution/run",
            data={
                "allocation_run_id": str(run_id),
                "source_node_id": str(src_id),
                "duration_minutes": "30.0",
                "routing_strategy": "SHORTEST_PATH",
            },
        )
        assess = db.get_latest_delivery_assessment()
        resp = client.get(f"/distribution/assessments/{assess['assessment_id']}")
        assert resp.status_code == 200
        html = resp.get_data(as_text=True)

        assert "Shared-Pipe Capacity Contention Detected" in html
        assert "Do not present independently calculated zone deliveries as jointly feasible" in html
        assert "Shared-Trunk" in html

    def test_run_history_view_displays_comparison(self, client, seeded_network):
        """GET /distribution/runs/<run_id> displays run summary and assessment history."""
        data = seeded_network
        run_id = data["run_id"]

        # Run 2 assessments on same run with different strategies
        client.post(
            "/distribution/run",
            data={
                "allocation_run_id": str(run_id),
                "source_node_id": str(data["src_id"]),
                "duration_minutes": "60.0",
                "routing_strategy": "SHORTEST_PATH",
            },
        )
        client.post(
            "/distribution/run",
            data={
                "allocation_run_id": str(run_id),
                "source_node_id": str(data["src_id"]),
                "duration_minutes": "60.0",
                "routing_strategy": "WIDEST_PATH",
            },
        )

        resp = client.get(f"/distribution/runs/{run_id}")
        assert resp.status_code == 200
        html = resp.get_data(as_text=True)

        assert f"ALLOCATION RUN #{run_id}" in html
        assert "Run Feasibility History &amp; Strategy Comparison" in html
        assert "SHORTEST_PATH" in html
        assert "WIDEST_PATH" in html
        assert "2 assessment(s)" in html

    def test_nonexistent_assessment_details_redirects(self, client, seeded_network):
        """Accessing a non-existent assessment ID flashes error and redirects to dashboard."""
        resp = client.post if False else client.get("/distribution/assessments/99999", follow_redirects=True)
        assert resp.status_code == 200
        assert "does not exist" in resp.get_data(as_text=True)
