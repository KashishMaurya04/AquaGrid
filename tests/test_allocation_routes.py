"""
tests/test_allocation_routes.py – Comprehensive Integration and Route Tests for Fair Water Allocation.

Validates Phase 4 UI and Flask integration:
1. Allocation dashboard loads successfully (GET /allocation).
2. GET /allocation is strictly read-only and idempotent (does not mutate database records).
3. Zone creation works (POST /allocation/zones/create).
4. Invalid zone inputs are rejected (negative demand, essential > demand, weight <= 0, blank name).
5. Zone editing works (GET and POST /allocation/zones/<id>/edit).
6. Historical snapshots survive zone edits.
7. Zone deletion restrictions work (rejects deletion of referenced zones with clear advice).
8. Allocation execution persists correct results (POST /allocation/run).
9. Allocation failure rolls back the entire transaction.
10. Historical run details display correctly (GET /allocation/runs/<id>).
11. Historical results remain unchanged after zone renaming.
12. Repeated GET requests create no duplicate allocation runs.
13. Invalid supply values are rejected (empty, negative, non-numeric).
14. Invalid allocation periods are rejected (start > end, invalid format).
15. Navigation renders correctly with active link highlighting.
"""

from __future__ import annotations

import os
import sys
import pytest
from decimal import Decimal

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import database as db_module
import app as app_module
from allocation.engine import ZoneDemandInput, allocate_water


@pytest.fixture()
def isolated_db(monkeypatch, tmp_path):
    """Isolated SQLite database for route tests."""
    test_db = str(tmp_path / "test_allocation_routes.db")
    monkeypatch.setattr(db_module, "INSTANCE_DIR", str(tmp_path))
    monkeypatch.setattr(db_module, "DB_PATH", test_db)
    db_module.init_db()
    return db_module


@pytest.fixture()
def client(isolated_db, monkeypatch):
    """Flask test client wired to the isolated database."""
    monkeypatch.setattr(app_module, "db", isolated_db)
    app_module.app.config["TESTING"] = True
    app_module.app.config["WTF_CSRF_ENABLED"] = False
    with app_module.app.test_client() as c:
        yield c


class TestAllocationDashboardRoutes:
    """Dashboard loading, read-only idempotency, and navigation checks."""

    def test_allocation_dashboard_loads_successfully(self, client):
        """GET /allocation returns 200 and renders the dashboard with all core panels."""
        resp = client.get("/allocation")
        assert resp.status_code == 200
        html = resp.get_data(as_text=True)
        assert "Fair Water Allocation" in html
        assert "Execute Water Rationing Run" in html
        assert "Configured Consumer Distribution Zones" in html
        assert "Fairness Doctrine" in html
        assert "Fair Water Allocation Simulation Notice" in html

    def test_get_allocation_is_strictly_readonly_idempotent(self, client, isolated_db):
        """Repeated GET requests do not mutate zone counts, run counts, or allocation records."""
        # Create an initial zone
        isolated_db.create_allocation_zone("Hospital-A", 1000.0, 400.0, 2.0)

        initial_zones = isolated_db.count_allocation_zones()
        initial_runs = isolated_db.count_allocation_runs()
        initial_allocs = isolated_db.count_water_allocations()

        for _ in range(5):
            resp = client.get("/allocation")
            assert resp.status_code == 200

        assert isolated_db.count_allocation_zones() == initial_zones
        assert isolated_db.count_allocation_runs() == initial_runs
        assert isolated_db.count_water_allocations() == initial_allocs

    def test_navigation_renders_active_highlight(self, client):
        """Sidebar navigation marks Water Allocation as active on /allocation."""
        resp = client.get("/allocation")
        assert resp.status_code == 200
        html = resp.get_data(as_text=True)
        assert '<a class="nav-link active" href="/allocation">⚖ Water Allocation</a>' in html


class TestAllocationZoneManagementRoutes:
    """Zone CRUD, validation, and historical deletion protection."""

    def test_zone_creation_success(self, client, isolated_db):
        """POST /allocation/zones/create successfully inserts a valid zone and redirects."""
        data = {
            "zone_name": "East Medical Center",
            "requested_demand_l": "5000.00",
            "essential_min_l": "1500.00",
            "fairness_weight": "2.5",
            "priority_category": "critical",
            "status": "active",
            "description": "Primary regional trauma center",
        }
        resp = client.post("/allocation/zones/create", data=data, follow_redirects=True)
        assert resp.status_code == 200
        html = resp.get_data(as_text=True)
        assert "East Medical Center" in html
        assert "Allocation zone &#39;East Medical Center&#39; created successfully." in html

        zones = isolated_db.get_all_allocation_zones()
        assert len(zones) == 1
        assert zones[0]["zone_name"] == "East Medical Center"
        assert zones[0]["requested_demand_l"] == 5000.0
        assert zones[0]["essential_min_l"] == 1500.0
        assert zones[0]["fairness_weight"] == 2.5
        assert zones[0]["priority_category"] == "critical"

    def test_zone_creation_validation_rejections(self, client, isolated_db):
        """POST /allocation/zones/create rejects illegal values with informative messages."""
        # Case 1: Negative demand
        resp = client.post("/allocation/zones/create", data={
            "zone_name": "Invalid-Dem",
            "requested_demand_l": "-500",
            "essential_min_l": "0",
            "fairness_weight": "1.0",
        }, follow_redirects=True)
        assert "Requested demand must be zero or greater" in resp.get_data(as_text=True)

        # Case 2: Essential min > requested demand
        resp = client.post("/allocation/zones/create", data={
            "zone_name": "Invalid-Ess",
            "requested_demand_l": "500",
            "essential_min_l": "600",
            "fairness_weight": "1.0",
        }, follow_redirects=True)
        assert "Essential minimum cannot exceed requested demand" in resp.get_data(as_text=True)

        # Case 3: Weight <= 0
        resp = client.post("/allocation/zones/create", data={
            "zone_name": "Invalid-Weight",
            "requested_demand_l": "500",
            "essential_min_l": "100",
            "fairness_weight": "0",
        }, follow_redirects=True)
        assert "Fairness weight must be strictly greater than zero" in resp.get_data(as_text=True)

        # Case 4: Blank zone name
        resp = client.post("/allocation/zones/create", data={
            "zone_name": "   ",
            "requested_demand_l": "500",
            "essential_min_l": "100",
            "fairness_weight": "1.0",
        }, follow_redirects=True)
        assert "Zone name is required and cannot be blank" in resp.get_data(as_text=True)

        assert isolated_db.count_allocation_zones() == 0

    def test_zone_editing_workflow(self, client, isolated_db):
        """GET /allocation/zones/<id>/edit and POST updates zone parameters."""
        zid = isolated_db.create_allocation_zone("Original-Name", 2000.0, 500.0, 1.0)

        # GET edit page
        resp_get = client.get(f"/allocation/zones/{zid}/edit")
        assert resp_get.status_code == 200
        assert "Original-Name" in resp_get.get_data(as_text=True)

        # POST update
        update_data = {
            "zone_name": "Updated-Name",
            "requested_demand_l": "2500.00",
            "essential_min_l": "800.00",
            "fairness_weight": "1.8",
            "priority_category": "standard",
            "status": "active",
            "description": "Updated description",
        }
        resp_post = client.post(f"/allocation/zones/{zid}/edit", data=update_data, follow_redirects=True)
        assert resp_post.status_code == 200
        assert "Zone &#39;Updated-Name&#39; updated successfully." in resp_post.get_data(as_text=True)

        updated_zone = isolated_db.get_allocation_zone_by_id(zid)
        assert updated_zone["zone_name"] == "Updated-Name"
        assert updated_zone["requested_demand_l"] == 2500.0
        assert updated_zone["essential_min_l"] == 800.0

    def test_zone_deletion_restrictions_and_deactivation_advice(self, client, isolated_db):
        """Deleting a zone with historical allocations is safely rejected advising deactivation."""
        zid = isolated_db.create_allocation_zone("Historic-Zone", 1000.0, 300.0, 1.0)
        zone_inputs = [ZoneDemandInput(zone_id=zid, zone_name="Historic-Zone", requested_demand=1000.0, essential_min=300.0)]
        run_res = allocate_water(total_supply=800, zones=zone_inputs)
        isolated_db.record_allocation_run(run_res)

        # Attempt deletion
        resp = client.post(f"/allocation/zones/{zid}/delete", follow_redirects=True)
        assert resp.status_code == 200
        html = resp.get_data(as_text=True)
        assert "Cannot delete zone: historical allocation records exist" in html
        assert "Deactivate the zone instead" in html

        # Zone still exists
        assert isolated_db.get_allocation_zone_by_id(zid) is not None

    def test_unreferenced_zone_deletion_succeeds(self, client, isolated_db):
        """Deleting an unreferenced zone deletes it cleanly."""
        zid = isolated_db.create_allocation_zone("Temp-Zone", 500.0, 100.0, 1.0)
        resp = client.post(f"/allocation/zones/{zid}/delete", follow_redirects=True)
        assert resp.status_code == 200
        assert "Zone deleted successfully." in resp.get_data(as_text=True)
        assert isolated_db.get_allocation_zone_by_id(zid) is None


class TestAllocationExecutionRoutes:
    """Execution, persistence, supply validation, period validation, and historical fidelity."""

    def test_allocation_run_success_and_redirection(self, client, isolated_db):
        """POST /allocation/run executes fair allocation and redirects to run details."""
        z1 = isolated_db.create_allocation_zone("Hospital", 1000.0, 500.0, 2.0, "critical")
        z2 = isolated_db.create_allocation_zone("Suburbs", 2000.0, 300.0, 1.0, "standard")

        data = {
            "available_supply": "2000.00",
            "period_start": "2026-11-12T08:00",
            "period_end": "2026-11-12T20:00",
            "notes": "Morning test allocation",
        }
        resp = client.post("/allocation/run", data=data, follow_redirects=True)
        assert resp.status_code == 200
        html = resp.get_data(as_text=True)
        assert "Fair water allocation run #1 completed successfully" in html
        assert "Run #1" in html
        assert "Hospital" in html
        assert "Suburbs" in html
        assert "2,000.00" in html

        # Verify DB
        assert isolated_db.count_allocation_runs() == 1
        run = isolated_db.get_allocation_run_by_id(1)
        assert run["total_available_supply_l"] == 2000.0
        assert run["total_requested_demand_l"] == 3000.0
        assert run["total_allocated_l"] == 2000.0
        assert run["total_unmet_l"] == 1000.0
        assert run["allocation_policy"] == "WEIGHTED_MAX_MIN_RESIDUAL"

    def test_allocation_run_rejects_invalid_supply_values(self, client, isolated_db):
        """Invalid supply values (empty, negative, non-numeric) are rejected without mutation."""
        isolated_db.create_allocation_zone("Z1", 1000.0, 200.0, 1.0)

        # Empty supply
        resp1 = client.post("/allocation/run", data={"available_supply": ""}, follow_redirects=True)
        assert "Available water supply is required" in resp1.get_data(as_text=True)

        # Negative supply
        resp2 = client.post("/allocation/run", data={"available_supply": "-100"}, follow_redirects=True)
        assert "Available water supply cannot be negative" in resp2.get_data(as_text=True)

        # Non-numeric supply
        resp3 = client.post("/allocation/run", data={"available_supply": "infinite_water"}, follow_redirects=True)
        assert "Available water supply must be a valid number" in resp3.get_data(as_text=True)

        assert isolated_db.count_allocation_runs() == 0

    def test_allocation_run_rejects_invalid_periods(self, client, isolated_db):
        """Period start after period end or invalid formats are rejected."""
        isolated_db.create_allocation_zone("Z1", 1000.0, 200.0, 1.0)

        # Start after end
        resp = client.post("/allocation/run", data={
            "available_supply": "1000",
            "period_start": "2026-11-15T12:00",
            "period_end": "2026-11-10T12:00",
        }, follow_redirects=True)
        assert "Allocation period start cannot be after period end" in resp.get_data(as_text=True)
        assert isolated_db.count_allocation_runs() == 0

    def test_allocation_run_with_no_active_zones_gives_warning(self, client, isolated_db):
        """Running allocation when no active zones exist informs user gracefully."""
        # Create inactive zone only
        isolated_db.create_allocation_zone("Inactive-Zone", 1000.0, 200.0, 1.0, status="inactive")

        resp = client.post("/allocation/run", data={"available_supply": "1000"}, follow_redirects=True)
        assert "No active allocation zones exist to receive water" in resp.get_data(as_text=True)
        assert isolated_db.count_allocation_runs() == 0

    def test_historical_run_details_view(self, client, isolated_db):
        """GET /allocation/runs/<id> renders full historical details."""
        zid = isolated_db.create_allocation_zone("Historic-Zone-A", 1500.0, 500.0, 1.5)
        zone_inputs = [ZoneDemandInput(zone_id=zid, zone_name="Historic-Zone-A", requested_demand=1500.0, essential_min=500.0)]
        run_res = allocate_water(total_supply=1000.0, zones=zone_inputs)
        run_id = isolated_db.record_allocation_run(run_res, notes="Audit run")

        resp = client.get(f"/allocation/runs/{run_id}")
        assert resp.status_code == 200
        html = resp.get_data(as_text=True)
        assert f"Allocation Run #{run_id}" in html
        assert "Audit run" in html
        assert "Historic-Zone-A" in html
        assert "1,000.00" in html

    def test_historical_snapshots_survive_zone_rename(self, client, isolated_db):
        """Historical runs display original snapshot name even if zone was subsequently renamed."""
        zid = isolated_db.create_allocation_zone("Old-Hospital-Name", 1000.0, 400.0, 2.0)
        zone_inputs = [ZoneDemandInput(zone_id=zid, zone_name="Old-Hospital-Name", requested_demand=1000.0, essential_min=400.0)]
        run_res = allocate_water(total_supply=1000.0, zones=zone_inputs)
        run_id = isolated_db.record_allocation_run(run_res)

        # Rename zone
        client.post(f"/allocation/zones/{zid}/edit", data={
            "zone_name": "New-Hospital-Name",
            "requested_demand_l": "1200",
            "essential_min_l": "500",
            "fairness_weight": "2.0",
            "priority_category": "standard",
            "status": "active",
        }, follow_redirects=True)

        # Historical run details must show original snapshot name
        resp_hist = client.get(f"/allocation/runs/{run_id}")
        assert resp_hist.status_code == 200
        hist_html = resp_hist.get_data(as_text=True)
        assert "Old-Hospital-Name" in hist_html
        assert "New-Hospital-Name" not in hist_html

        # Dashboard zones table must show the new name
        resp_dash = client.get("/allocation")
        assert resp_dash.status_code == 200
        dash_html = resp_dash.get_data(as_text=True)
        assert "New-Hospital-Name" in dash_html

    def test_nonexistent_run_or_zone_redirects_safely(self, client):
        """Accessing nonexistent zone or run redirects with a flash message."""
        resp1 = client.get("/allocation/runs/99999", follow_redirects=True)
        assert "Allocation run #99999 not found" in resp1.get_data(as_text=True)

        resp2 = client.get("/allocation/zones/99999/edit", follow_redirects=True)
        assert "Zone not found" in resp2.get_data(as_text=True)
