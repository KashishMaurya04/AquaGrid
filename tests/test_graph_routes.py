"""
tests/test_graph_routes.py – HTTP integration tests for the Graph Explorer page.

Covers:
  - GET /graph when network is empty (displays helpful empty network message)
  - GET /graph when nodes exist (loads graph, displays root node simulation)
  - GET /graph with custom ?start_node_id=<id>
  - POST /graph with form selection
  - Simulation labels: explicitly marked as graph-based simulation, NOT real
    hydraulic simulation or live sensor data
  - Isolated node display: displays helpful isolated notice
  - Non-existent start node ID: displays graceful not-found message
  - Navigation bar link to Graph Explorer exists across views
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


class TestGraphRoutes:
    """Test Flask routes for Graph Explorer."""

    def test_empty_network_shows_empty_state(self, app_client):
        """When DB has 0 nodes, /graph must render a friendly empty network message."""
        r = app_client.get("/graph")
        assert r.status_code == 200
        assert b"Water Network is Empty" in r.data
        assert b"Go to Water Network" in r.data
        # Must still show simulation disclaimer
        assert b"graph-based simulation" in r.data.lower()

    def test_graph_page_loads_with_nodes(self, app_client, tmp_db):
        """When nodes exist, /graph renders with root node selected."""
        id_a = tmp_db.create_node("Tank 1", "tank", "North", "active")
        id_b = tmp_db.create_node("Junction 2", "junction", "Central", "active")
        tmp_db.create_pipe("Main", id_a, id_b, "100", "500", "1", "active")

        r = app_client.get("/graph")
        assert r.status_code == 200
        assert b"Graph Explorer" in r.data
        assert b"GRAPH SIMULATION" in r.data
        assert b"Tank 1" in r.data
        assert b"Junction 2" in r.data
        # Disclaimer check
        assert b"not real hydraulic simulation" in r.data.lower()
        assert b"not live sensor data" in r.data.lower()

    def test_select_start_node_via_query_param(self, app_client, tmp_db):
        """Selecting start node via GET query parameter ?start_node_id=..."""
        id_a = tmp_db.create_node("Tank A", "tank", "North", "active")
        id_b = tmp_db.create_node("Junction B", "junction", "Central", "active")
        id_c = tmp_db.create_node("Valve C", "valve", "South", "active")
        tmp_db.create_pipe("P1", id_a, id_b, "50", "200", "1", "active")
        tmp_db.create_pipe("P2", id_b, id_c, "60", "200", "1", "active")

        r = app_client.get(f"/graph?start_node_id={id_a}")
        assert r.status_code == 200
        assert b"Tank A" in r.data
        assert b"Reachable Network Nodes" in r.data
        assert b"3 reachable" in r.data.lower() or b"3 / 3" in r.data

    def test_post_form_selection(self, app_client, tmp_db):
        """Submitting start node via POST form."""
        id_a = tmp_db.create_node("Tank X", "tank", "", "active")
        id_b = tmp_db.create_node("Junction Y", "junction", "", "active")
        tmp_db.create_pipe("P1", id_a, id_b, "50", "200", "1", "active")

        r = app_client.post("/graph", data={"start_node_id": id_b, "traversal_mode": "undirected"})
        assert r.status_code == 200
        assert b"Tank X" in r.data
        assert b"Junction Y" in r.data

    def test_isolated_node_shows_helpful_notice(self, app_client, tmp_db):
        """An isolated node shows an explicit message that no other nodes are reachable."""
        id_iso = tmp_db.create_node("Isolated Tank", "tank", "Remote", "active")

        r = app_client.get(f"/graph?start_node_id={id_iso}")
        assert r.status_code == 200
        assert b"Isolated Node Notice" in r.data
        assert b"No other nodes in the network are reachable" in r.data

    def test_invalid_start_node_id_handled_gracefully(self, app_client, tmp_db):
        """Invalid or non-existent start node ID displays a clear error state."""
        tmp_db.create_node("Node 1", "junction", "", "active")

        r = app_client.get("/graph?start_node_id=99999")
        assert r.status_code == 200
        assert b"Node Not Found" in r.data

    def test_sidebar_has_graph_explorer_link(self, app_client, tmp_db):
        """The sidebar across pages contains the Graph Explorer link."""
        for path in ["/", "/network", "/pipes", "/graph"]:
            r = app_client.get(path)
            assert r.status_code == 200
            assert b"/graph" in r.data
            assert b"Graph Explorer" in r.data

