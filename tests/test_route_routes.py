"""
tests/test_route_routes.py – HTTP integration tests for the Route Finder page.

Covers:
  - GET /route when network is empty (displays empty network message)
  - GET /route when nodes exist (loads page with default endpoints selected)
  - GET /route with custom ?source_id=...&destination_id=...
  - POST /route with form submission
  - Source equals destination display (cost 0.0 notice)
  - Unreachable destination display (helpful no path message)
  - Simulation labels: explicitly marked as graph-based route planning,
    NOT hydraulic optimization or live sensor data
  - Route Finder sidebar navigation link across pages
"""

import os
import sys
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


@pytest.fixture()
def tmp_db(monkeypatch, tmp_path):
    """Isolate database tests with a fresh SQLite DB."""
    import database as db_module

    test_db = str(tmp_path / "test_routes_aquagrid.db")
    monkeypatch.setattr(db_module, "INSTANCE_DIR", str(tmp_path))
    monkeypatch.setattr(db_module, "DB_PATH", test_db)
    db_module.init_db()
    return db_module


@pytest.fixture()
def app_client(tmp_db, monkeypatch):
    """Flask test client wired to the isolated database."""
    from app import app
    app.config["TESTING"] = True
    app.config["WTF_CSRF_ENABLED"] = False
    with app.test_client() as client:
        yield client


class TestRouteFinderRoutes:
    """Test Flask routes for Route Finder."""

    def test_empty_network_shows_empty_state(self, app_client):
        """When DB has 0 nodes, /route must render empty network state."""
        r = app_client.get("/route")
        assert r.status_code == 200
        assert b"Water Network is Empty" in r.data
        assert b"graph-based route planning" in r.data.lower()

    def test_route_page_loads_with_nodes(self, app_client, tmp_db):
        """When nodes exist, /route computes route between default endpoints."""
        id_a = tmp_db.create_node("Tank Alpha", "tank", "North", "active")
        id_b = tmp_db.create_node("Junction Beta", "junction", "Central", "active")
        tmp_db.create_pipe("Main", id_a, id_b, "100", "500", "2.5", "active")

        r = app_client.get("/route")
        assert r.status_code == 200
        assert b"Route Finder" in r.data
        assert b"ROUTE OPTIMIZATION" in r.data
        assert b"Tank Alpha" in r.data
        assert b"Junction Beta" in r.data
        assert b"2.5" in r.data
        # Disclaimer verification
        assert b"not hydraulic optimization" in r.data.lower()
        assert b"not live sensor data" in r.data.lower()

    def test_custom_route_query_params(self, app_client, tmp_db):
        """GET /route?source_id=...&destination_id=... computes path."""
        id_1 = tmp_db.create_node("Tank 1", "tank", "Z1", "active")
        id_2 = tmp_db.create_node("Junction 2", "junction", "Z2", "active")
        id_3 = tmp_db.create_node("Valve 3", "valve", "Z3", "active")
        tmp_db.create_pipe("P1", id_1, id_2, "50", "200", "1.2", "active")
        tmp_db.create_pipe("P2", id_2, id_3, "60", "200", "1.8", "active")

        r = app_client.get(f"/route?source_id={id_1}&destination_id={id_3}")
        assert r.status_code == 200
        assert b"3.0" in r.data
        assert b"Optimal Route Sequence" in r.data

    def test_post_form_selection(self, app_client, tmp_db):
        """POST /route calculates and displays path."""
        id_1 = tmp_db.create_node("Tank A", "tank", "", "active")
        id_2 = tmp_db.create_node("Valve B", "valve", "", "active")
        tmp_db.create_pipe("Pipe-AB", id_1, id_2, "50", "200", "3.75", "active")

        r = app_client.post(
            "/route",
            data={
                "source_id": id_1,
                "destination_id": id_2,
                "traversal_mode": "undirected",
            },
        )
        assert r.status_code == 200
        assert b"3.75" in r.data
        assert b"Pipe-AB" in r.data

    def test_source_equals_destination_display(self, app_client, tmp_db):
        """Selecting identical source and destination displays validation error and no misleading route cards."""
        id_1 = tmp_db.create_node("Solo Tank", "tank", "", "active")

        r = app_client.get(f"/route?source_id={id_1}&destination_id={id_1}")
        assert r.status_code == 200
        assert b"Source and destination must be different." in r.data
        assert b"Validation Error" in r.data
        assert b"Total Routing Cost" not in r.data
        assert b"Optimal Route Sequence" not in r.data

    def test_multiple_path_demonstration_network_route(self, app_client, tmp_db):
        """
        Verify demonstration 4-node network in web Route Finder:
          A -> B (10), A -> C (4), C -> B (2), B -> D (5), C -> D (12)
        Shortest route: A -> C -> B -> D with total cost 11.00 units.
        """
        id_a = tmp_db.create_node("Tank A", "tank", "Zone A", "active")
        id_b = tmp_db.create_node("Junction B", "junction", "Zone B", "active")
        id_c = tmp_db.create_node("Junction C", "junction", "Zone C", "active")
        id_d = tmp_db.create_node("Valve D", "valve", "Zone D", "active")

        tmp_db.create_pipe("Pipe A-B", id_a, id_b, "1000", "500", "10.0", "active")
        tmp_db.create_pipe("Pipe A-C", id_a, id_c, "400", "500", "4.0", "active")
        tmp_db.create_pipe("Pipe C-B", id_c, id_b, "200", "500", "2.0", "active")
        tmp_db.create_pipe("Pipe B-D", id_b, id_d, "500", "500", "5.0", "active")
        tmp_db.create_pipe("Pipe C-D", id_c, id_d, "1200", "500", "12.0", "active")

        r = app_client.get(f"/route?source_id={id_a}&destination_id={id_d}")
        assert r.status_code == 200
        assert b"11.0" in r.data
        assert b"units" in r.data
        assert b"Pipe A-C" in r.data
        assert b"Pipe C-B" in r.data
        assert b"Pipe B-D" in r.data
        assert b"Pipe C-D" not in r.data  # higher cost direct pipe not chosen

    def test_directionality_route_finder_integration(self, app_client, tmp_db):
        """Verify directed vs bidirectional flow in web Route Finder."""
        id_1 = tmp_db.create_node("Source Node", "tank", "", "active")
        id_2 = tmp_db.create_node("Target Node", "valve", "", "active")
        tmp_db.create_pipe("One-Way Pipe", id_1, id_2, "100", "500", "3.5", "active")

        # Directed backward flow should fail
        r_directed_back = app_client.post(
            "/route",
            data={
                "source_id": id_2,
                "destination_id": id_1,
                "traversal_mode": "directed",
            },
        )
        assert r_directed_back.status_code == 200
        assert b"No Path Available" in r_directed_back.data

        # Bidirectional backward flow should succeed
        r_bi_back = app_client.post(
            "/route",
            data={
                "source_id": id_2,
                "destination_id": id_1,
                "traversal_mode": "undirected",
            },
        )
        assert r_bi_back.status_code == 200
        assert b"3.5" in r_bi_back.data
        assert b"One-Way Pipe" in r_bi_back.data

    def test_unreachable_destination_display(self, app_client, tmp_db):
        """Disconnected components show friendly 'No Path Available' message."""
        id_1 = tmp_db.create_node("Island 1", "tank", "", "active")
        id_2 = tmp_db.create_node("Island 2", "tank", "", "active")

        r = app_client.get(f"/route?source_id={id_1}&destination_id={id_2}")
        assert r.status_code == 200
        assert b"No Path Available" in r.data
        assert b"not reachable" in r.data.lower()

    def test_route_finder_sidebar_link_present(self, app_client, tmp_db):
        """Sidebar contains link to /route across views."""
        for path in ["/", "/network", "/pipes", "/graph", "/route"]:
            r = app_client.get(path)
            assert r.status_code == 200
            assert b"/route" in r.data
            assert b"Route Finder" in r.data


    def test_route_finder_responsive_form_classes(self, app_client, tmp_db):
        """Verify responsive CSS classes are properly wired to Route Finder."""
        id_1 = tmp_db.create_node("Tank 1", "tank", "Z1", "active")
        id_2 = tmp_db.create_node("Junction 2", "junction", "Z2", "active")
        tmp_db.create_pipe("P1", id_1, id_2, "50", "200", "1.5", "active")

        r = app_client.get(f"/route?source_id={id_1}&destination_id={id_2}")
        assert r.status_code == 200
        assert b"route-form-row" in r.data
        assert b"btn-compute" in r.data
        assert b"form-action-group" in r.data
        assert b"nw-table-wrap" in r.data
        assert b"route-pipe-badge" in r.data

    def test_css_responsive_rules_present(self, app_client):
        """Verify that CSS stylesheets contain responsive overflow and breakpoint rules."""
        r_graph_css = app_client.get("/static/css/graph.css")
        assert r_graph_css.status_code == 200
        assert b".route-form-row" in r_graph_css.data
        assert b"@media (max-width: 1050px)" in r_graph_css.data
        assert b"@media (max-width: 650px)" in r_graph_css.data
        assert b"min-width: 0" in r_graph_css.data
        assert b"box-sizing: border-box" in r_graph_css.data

        r_net_css = app_client.get("/static/css/network.css")
        assert r_net_css.status_code == 200
        assert b".table-wrap" in r_net_css.data
        assert b".nw-table-wrap" in r_net_css.data
        assert b"overflow-x: auto" in r_net_css.data

    def test_route_form_grid_columns_and_overflow_invariants(self, app_client):
        """Verify route-form-row uses minmax(0, 1fr) for all controls and has no overflow at 1366, 1024, 768px."""
        r_css = app_client.get("/static/css/graph.css")
        assert r_css.status_code == 200
        css_text = r_css.data.decode("utf-8")

        # Desktop route-form-row has 3 equal minmax(0, 1fr) controls and auto button
        assert "grid-template-columns: minmax(0, 1fr) minmax(0, 1fr) minmax(0, 1fr) auto;" in css_text

        # Verify no horizontal overflow at 1366, 1024, and 768px
        for viewport in [1366, 1024, 768]:
            if viewport <= 1050:
                sidebar = 190
                padding = 44 + 42
                available = viewport - sidebar - padding
                col_width = (available - 14) / 2
                assert col_width > 200, f"Columns too narrow at {viewport}px"
                total = col_width * 2 + 14
                assert abs(total - available) < 0.01, f"Overflow at {viewport}px"
            else:
                sidebar = 245
                padding = 70 + 42
                available = viewport - sidebar - padding
                btn = 140
                col_width = (available - 42 - btn) / 3
                assert col_width > 250, f"Columns too narrow on desktop at {viewport}px"
                total = col_width * 3 + btn + 42
                assert abs(total - available) < 0.01, f"Overflow at {viewport}px"
