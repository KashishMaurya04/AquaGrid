"""
Tests for AquaGrid Smart Repair Priority Queue using Custom Max Heap.

Covers:
- Custom MaxHeap implementation:
    - insert, extract_max, peek, heapify, sift_up, sift_down, is_empty, size
    - Max-heap property maintained at all steps
    - Deterministic tie-breaking (equal scores -> created_at FIFO -> task_id)
    - Empty heap safety (no crashes on empty peek/extract)
    - Verification that heapq is NOT used
- Priority Scoring Engine:
    - Normalized 0-100 score computation
    - 5 sub-scores: flow surge, pressure drop, severity, persistence, pipe capacity
    - Transparent handling of missing sensor telemetry
    - Priority category classification: CRITICAL, HIGH, MEDIUM, LOW
    - Audit breakdown and explanation strings
- Database Integration:
    - repair_tasks table creation and uniqueness constraint on event_id
    - Deduplication: updating existing task without resetting status
    - Status transitions (Pending -> In Progress -> Completed)
    - Decoupled lifecycle: task completion does not alter leak event status
    - Aggregation and counting helpers
- PriorityQueueManager:
    - Synchronizing leak events to repair tasks
    - Active vs completed heap filtering
    - Root-first extraction order
- Flask Routes and Dashboard:
    - GET /repairs, GET /repair-queue
    - POST /repairs/<task_id>/status
    - POST /repairs/<task_id>/investigate
    - POST /repairs/sync
    - 405 on GET for POST endpoints
    - Live dashboard metrics and top tasks in GET /
"""

import ast
import os
import sys
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from repair_queue.max_heap import MaxHeap, RepairTask
from repair_queue.scoring import (
    calculate_priority_score,
    PriorityScore,
    PRIORITY_THRESHOLDS
)
from repair_queue.manager import PriorityQueueManager
import database


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture()
def tmp_db(monkeypatch, tmp_path):
    """Isolate database tests with a fresh SQLite DB in a temporary directory."""
    import database as db_module

    test_db = str(tmp_path / "test_repair_aquagrid.db")
    monkeypatch.setattr(db_module, "INSTANCE_DIR", str(tmp_path))
    monkeypatch.setattr(db_module, "DB_PATH", test_db)
    db_module.init_db()
    return db_module


@pytest.fixture()
def repair_network(tmp_db):
    """
    Sets up a small network with nodes, pipes, sensors, and open leak events
    to test repair task creation, synchronization, and route actions.
    """
    id_a = tmp_db.create_node("Reservoir North", "tank", "North Zone", "active")
    id_b = tmp_db.create_node("Junction Central", "junction", "Central Zone", "active")
    id_c = tmp_db.create_node("Valve West", "valve", "West Zone", "active")

    pipe_1 = tmp_db.create_pipe("Trunk-North", id_a, id_b, "200.0", "1200.0", "1.0", "active")
    pipe_2 = tmp_db.create_pipe("Branch-West", id_b, id_c, "150.0", "400.0", "1.5", "active")

    # Add sensors
    s1_flow = tmp_db.create_sensor("FM-Trunk-1", pipe_1, "flow_meter", "L/min", "active")
    s1_press = tmp_db.create_sensor("PS-Trunk-1", pipe_1, "pressure_sensor", "psi", "active")
    s2_flow = tmp_db.create_sensor("FM-Branch-2", pipe_2, "flow_meter", "L/min", "active")

    # Add leak events using database.record_leak_event
    event_1_id, _ = tmp_db.record_leak_event(
        pipe_id=pipe_1,
        severity="critical",
        evidence_description="Extreme flow surge and pressure drop on Trunk-North (6 consecutive readings)",
        flow_anomaly_score=3.8,
        pressure_anomaly_score=-3.2,
        is_simulated=True,
    )

    event_2_id, _ = tmp_db.record_leak_event(
        pipe_id=pipe_2,
        severity="medium",
        evidence_description="Moderate flow anomaly on Branch-West (3 consecutive readings)",
        flow_anomaly_score=1.8,
        pressure_anomaly_score=None,
        is_simulated=True,
    )

    return {
        "nodes": [id_a, id_b, id_c],
        "pipes": [pipe_1, pipe_2],
        "events": [event_1_id, event_2_id]
    }


@pytest.fixture()
def app_client(tmp_db, monkeypatch):
    """Flask test client wired to the isolated database."""
    import database as db_module
    import app as app_module

    monkeypatch.setattr(app_module, "db", db_module)

    app_module.app.config["TESTING"] = True
    app_module.app.config["WTF_CSRF_ENABLED"] = False
    app_module.app.config["SECRET_KEY"] = "test-repair-secret-key"

    with app_module.app.test_client() as client:
        yield client


# ==============================================================================
# 1. Custom MaxHeap Unit Tests
# ==============================================================================

class TestMaxHeap:
    """Unit tests for pure Python MaxHeap."""

    def test_heap_does_not_import_heapq(self):
        """Strict requirement: MaxHeap must NOT import or use python's heapq."""
        import repair_queue.max_heap as mh_module
        import inspect
        source = inspect.getsource(mh_module)
        tree = ast.parse(source)

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name != "heapq", "Forbidden import of heapq detected!"
            elif isinstance(node, ast.ImportFrom):
                assert node.module != "heapq", "Forbidden import from heapq detected!"

    def test_empty_heap_operations(self):
        """Empty heap must return None on peek/extract_max and report empty."""
        heap = MaxHeap()
        assert heap.is_empty() is True
        assert heap.size() == 0
        assert heap.peek() is None
        assert heap.extract_max() is None
        assert heap.to_list() == []
        assert heap.extract_all() == []

    def test_insert_and_peek(self):
        """Single insert makes the item root and updates size."""
        heap = MaxHeap()
        task = RepairTask(task_id=1, event_id=10, priority_score=75.0)
        heap.insert(task)
        assert heap.size() == 1
        assert heap.is_empty() is False
        assert heap.peek() is task
        # Peek does not remove
        assert heap.size() == 1

    def test_insert_maintains_max_property(self):
        """Inserting items in any order keeps the highest priority at index 0."""
        heap = MaxHeap()
        tasks = [
            RepairTask(task_id=1, event_id=1, priority_score=30.0),
            RepairTask(task_id=2, event_id=2, priority_score=90.0),
            RepairTask(task_id=3, event_id=3, priority_score=60.0),
            RepairTask(task_id=4, event_id=4, priority_score=85.0),
            RepairTask(task_id=5, event_id=5, priority_score=45.0),
        ]
        for t in tasks:
            heap.insert(t)

        assert heap.size() == 5
        assert heap.peek().task_id == 2
        assert heap.peek().priority_score == 90.0

        # Verify max-heap property for every node
        nodes = heap._heap
        for i in range(1, len(nodes)):
            parent = (i - 1) // 2
            assert nodes[parent] >= nodes[i], f"Parent {nodes[parent]} < child {nodes[i]} at index {i}"

    def test_extract_max_descending_order(self):
        """extract_max must yield elements in non-increasing priority score."""
        heap = MaxHeap()
        scores = [12.5, 95.0, 44.0, 78.0, 3.0, 88.0, 62.0]
        for idx, s in enumerate(scores):
            heap.insert(RepairTask(task_id=idx + 1, event_id=idx + 10, priority_score=s))

        extracted = []
        while not heap.is_empty():
            extracted.append(heap.extract_max().priority_score)

        assert extracted == sorted(scores, reverse=True)
        assert heap.is_empty() is True
        assert heap.size() == 0
        assert heap.peek() is None

    def test_heapify_unordered_tasks(self):
        """heapify() converts an arbitrary list into a valid max heap in O(N)."""
        tasks = [
            RepairTask(task_id=1, event_id=1, priority_score=15.0),
            RepairTask(task_id=2, event_id=2, priority_score=75.0),
            RepairTask(task_id=3, event_id=3, priority_score=99.0),
            RepairTask(task_id=4, event_id=4, priority_score=50.0),
            RepairTask(task_id=5, event_id=5, priority_score=35.0),
            RepairTask(task_id=6, event_id=6, priority_score=88.0),
        ]
        heap = MaxHeap()
        heap.heapify(tasks)

        assert heap.size() == 6
        assert heap.peek().priority_score == 99.0

        # Verify max property
        nodes = heap._heap
        for i in range(1, len(nodes)):
            parent = (i - 1) // 2
            assert nodes[parent] >= nodes[i]

        # Extract all must be sorted descending
        extracted_scores = [t.priority_score for t in heap.extract_all()]
        assert extracted_scores == [99.0, 88.0, 75.0, 50.0, 35.0, 15.0]

    def test_deterministic_tie_breaking(self):
        """When scores are equal, earlier created_at wins. When created_at also matches, smaller task_id wins."""
        t1 = "2026-11-01T10:00:00"
        t2 = "2026-11-01T11:00:00"

        # Equal score (80.0), different created_at
        task_older = RepairTask(task_id=10, event_id=1, priority_score=80.0, created_at=t1)
        task_newer = RepairTask(task_id=5, event_id=2, priority_score=80.0, created_at=t2)

        # older should be > newer (FIFO urgency)
        assert task_older > task_newer
        assert not (task_newer > task_older)

        heap = MaxHeap()
        heap.insert(task_newer)
        heap.insert(task_older)
        assert heap.peek() is task_older
        assert heap.extract_max() is task_older
        assert heap.extract_max() is task_newer

        # Equal score AND equal timestamp -> smaller task_id wins
        task_id_1 = RepairTask(task_id=1, event_id=3, priority_score=50.0, created_at=t1)
        task_id_2 = RepairTask(task_id=2, event_id=4, priority_score=50.0, created_at=t1)

        assert task_id_1 > task_id_2
        heap2 = MaxHeap()
        heap2.insert(task_id_2)
        heap2.insert(task_id_1)
        assert heap2.extract_max() is task_id_1
        assert heap2.extract_max() is task_id_2

    def test_extract_all_helper(self):
        """extract_all empties the heap and returns tasks in priority order."""
        heap = MaxHeap()
        for score in [20, 80, 50]:
            heap.insert(RepairTask(task_id=score, event_id=score, priority_score=float(score)))

        result = heap.extract_all()
        assert len(result) == 3
        assert [r.priority_score for r in result] == [80.0, 50.0, 20.0]
        assert heap.is_empty() is True


# ==============================================================================
# 2. Priority Scoring Engine Unit Tests
# ==============================================================================

class TestPriorityScoring:
    """Unit tests for hydraulic anomaly priority scoring formula."""

    def test_max_score_components(self):
        """Extreme anomaly scenario achieves maximum score of 100 with CRITICAL category."""
        res = calculate_priority_score(
            flow_z_score=4.5,          # >= 3.0 -> 30 pts
            pressure_z_score=4.0,      # >= 3.0 -> 30 pts
            event_severity='critical', # 20 pts
            consecutive_abnormal=6,    # min(10, 6*2) -> 10 pts
            pipe_capacity_lpm=1500.0   # >= 1000 -> 10 pts
        )
        assert res.total_score == 100.0
        assert res.category == "CRITICAL"
        assert res.breakdown['flow_surge'] == 30.0
        assert res.breakdown['pressure_drop'] == 30.0
        assert res.breakdown['event_severity'] == 20.0
        assert res.breakdown['persistence'] == 10.0
        assert res.breakdown['pipe_capacity'] == 10.0
        assert "CRITICAL" in res.explanation
        assert "Audit Disclaimer" in res.explanation

    def test_zero_or_negative_z_scores(self):
        """Zero or negative z-scores yield 0 points for hydraulic surges."""
        res = calculate_priority_score(
            flow_z_score=-1.2,
            pressure_z_score=0.0,
            event_severity='low',     # 5 pts
            consecutive_abnormal=1,   # 2 pts
            pipe_capacity_lpm=200.0   # min(10, 200/100) -> 2 pts
        )
        assert res.breakdown['flow_surge'] == 0.0
        assert res.breakdown['pressure_drop'] == 0.0
        assert res.breakdown['event_severity'] == 5.0
        assert res.breakdown['persistence'] == 2.0
        assert res.breakdown['pipe_capacity'] == 2.0
        assert res.total_score == 9.0
        assert res.category == "LOW"

    def test_category_thresholds(self):
        """Validate threshold borders: CRITICAL (>=80), HIGH (>=60), MEDIUM (>=35), LOW (<35)."""
        # Critical border (80.0)
        r80 = calculate_priority_score(
            flow_z_score=2.0, pressure_z_score=2.0,  # 20 + 20 = 40
            event_severity='critical',               # 20 -> 60
            consecutive_abnormal=5,                  # 10 -> 70
            pipe_capacity_lpm=1000.0                 # 10 -> 80
        )
        assert r80.category == "CRITICAL"
        assert r80.total_score == 80.0

        # HIGH (65.0)
        r65 = calculate_priority_score(
            flow_z_score=1.5, pressure_z_score=1.5,  # 15 + 15 = 30
            event_severity='high',                   # 15 -> 45
            consecutive_abnormal=5,                  # 10 -> 55
            pipe_capacity_lpm=1000.0                 # 10 -> 65
        )
        assert r65.category == "HIGH"
        assert r65.total_score == 65.0

        # MEDIUM (45.0)
        r45 = calculate_priority_score(
            flow_z_score=1.0, pressure_z_score=1.0,  # 10 + 10 = 20
            event_severity='medium',                 # 10 -> 30
            consecutive_abnormal=5,                  # 10 -> 40
            pipe_capacity_lpm=500.0                  # 5  -> 45
        )
        assert r45.category == "MEDIUM"
        assert r45.total_score == 45.0

        # LOW (20.0)
        r20 = calculate_priority_score(
            flow_z_score=0.5, pressure_z_score=0.0,  # 5 + 0 = 5
            event_severity='low',                    # 5 -> 10
            consecutive_abnormal=2,                  # 4 -> 14
            pipe_capacity_lpm=600.0                  # 6 -> 20
        )
        assert r20.category == "LOW"
        assert r20.total_score == 20.0

    def test_missing_telemetry_graceful_handling(self):
        """Pipes with missing sensors (None) should yield 0 for that metric and transparently log in explanation."""
        res = calculate_priority_score(
            flow_z_score=None,
            pressure_z_score=2.5,     # 25.0
            event_severity='high',    # 15.0
            consecutive_abnormal=None,# 0
            pipe_capacity_lpm=None    # 0
        )
        assert res.breakdown['flow_surge'] == 0.0
        assert res.breakdown['pressure_drop'] == 25.0
        assert res.breakdown['event_severity'] == 15.0
        assert res.breakdown['persistence'] == 0.0
        assert res.breakdown['pipe_capacity'] == 0.0
        assert res.total_score == 40.0
        assert res.category == "MEDIUM"
        assert "no flow meter" in res.explanation.lower()


# ==============================================================================
# 3. Database Integration Tests
# ==============================================================================

class TestDatabaseIntegration:
    """Tests for repair_tasks table, deduplication, status transitions, and lifecycle decoupling."""

    def test_repair_task_crud_and_deduplication(self, tmp_db, repair_network):
        """Create or update repair task cleanly; subsequent calls for same event update without duplicates."""
        event_id = repair_network['events'][0]
        pipe_id = repair_network['pipes'][0]

        # 1. Create task
        task_id, is_new = tmp_db.create_or_update_repair_task(
            event_id=event_id,
            pipe_id=pipe_id,
            priority_score=72.5,
            priority_category='HIGH',
            score_breakdown='{"flow_surge": 25.0, "pressure_drop": 20.0}',
        )
        assert task_id is not None
        assert is_new is True

        # Retrieve task
        task = tmp_db.get_repair_task_by_id(task_id)
        assert task is not None
        assert task['event_id'] == event_id
        assert task['pipe_id'] == pipe_id
        assert task['priority_score'] == 72.5
        assert task['priority_category'] == 'HIGH'
        assert task['status'] == 'Pending'

        # 2. Update task status to In Progress
        ok = tmp_db.update_repair_task_status(task_id, 'In Progress')
        assert ok is True
        task_updated = tmp_db.get_repair_task_by_id(task_id)
        assert task_updated['status'] == 'In Progress'

        # 3. Deduplication: Calling create_or_update again for the SAME event_id
        # must NOT create a new row, and must preserve the 'In Progress' status!
        task_id2, is_new2 = tmp_db.create_or_update_repair_task(
            event_id=event_id,
            pipe_id=pipe_id,
            priority_score=85.0,
            priority_category='CRITICAL',
        )
        assert task_id2 == task_id
        assert is_new2 is False
        task_dedup = tmp_db.get_repair_task_by_id(task_id)
        assert task_dedup['priority_score'] == 85.0
        assert task_dedup['priority_category'] == 'CRITICAL'
        # Preserves 'In Progress'
        assert task_dedup['status'] == 'In Progress'

        # 4. Invalid status raises ValueError
        with pytest.raises(ValueError):
            tmp_db.update_repair_task_status(task_id, 'InvalidStatus')

    def test_completing_repair_task_does_not_resolve_leak_event(self, tmp_db, repair_network):
        """Lifecycle decoupling: Completing a repair task must NEVER alter leak_events status."""
        event_id = repair_network['events'][0]
        pipe_id = repair_network['pipes'][0]

        # Ensure leak event is open
        orig_event = tmp_db.get_leak_event_by_id(event_id)
        assert orig_event['status'] == 'open'

        # Create repair task and mark Completed
        task_id, _ = tmp_db.create_or_update_repair_task(
            event_id=event_id,
            pipe_id=pipe_id,
            priority_score=90.0,
            priority_category='CRITICAL',
        )

        tmp_db.update_repair_task_status(task_id, 'Completed')

        # Verify task is Completed
        task_row = tmp_db.get_repair_task_by_id(task_id)
        assert task_row['status'] == 'Completed'

        # CRITICAL CHECK: Leak event status must remain 'open'!
        current_event = tmp_db.get_leak_event_by_id(event_id)
        assert current_event['status'] == 'open', "Completing repair task must not modify leak event status!"

    def test_count_and_aggregation_helpers(self, tmp_db, repair_network):
        """Verify count_pending_repair_tasks, count_active_repair_tasks, and count_repair_tasks_by_category."""
        ev1 = repair_network['events'][0]
        ev2 = repair_network['events'][1]
        p1 = repair_network['pipes'][0]
        p2 = repair_network['pipes'][1]

        t1, _ = tmp_db.create_or_update_repair_task(ev1, p1, 85.0, 'CRITICAL')
        t2, _ = tmp_db.create_or_update_repair_task(ev2, p2, 45.0, 'MEDIUM')
        tmp_db.update_repair_task_status(t2, 'In Progress')

        assert tmp_db.count_pending_repair_tasks() == 1
        assert tmp_db.count_active_repair_tasks() == 2

        counts = tmp_db.count_repair_tasks_by_category()
        assert counts.get('CRITICAL', 0) == 1
        assert counts.get('MEDIUM', 0) == 1
        assert counts.get('HIGH', 0) == 0
        assert counts.get('LOW', 0) == 0

        # Mark t1 Completed
        tmp_db.update_repair_task_status(t1, 'Completed')
        assert tmp_db.count_pending_repair_tasks() == 0
        assert tmp_db.count_active_repair_tasks() == 1


# ==============================================================================
# 4. PriorityQueueManager Tests
# ==============================================================================

class TestPriorityQueueManager:
    """Tests for PriorityQueueManager orchestrating heap construction and sorting."""

    def test_manager_builds_heap_and_orders_tasks(self, tmp_db, repair_network):
        """PriorityQueueManager builds heap of active tasks and returns root-first descending list."""
        manager = PriorityQueueManager()

        # Sync and build heap from the network's leak events
        heap = manager.sync_and_build_heap(tmp_db)
        assert isinstance(heap, MaxHeap)
        assert heap.size() == 2

        # Root must be the critical event (event 1 on pipe 1)
        root = heap.peek()
        assert root.pipe_name == "Trunk-North"
        assert root.priority_category == "CRITICAL"

        # Get ordered tasks (extracted descending)
        ordered = manager.get_ordered_tasks(tmp_db)
        assert len(ordered) == 2
        assert ordered[0].priority_score >= ordered[1].priority_score
        assert ordered[0].pipe_name == "Trunk-North"
        assert ordered[1].pipe_name == "Branch-West"

    def test_completed_tasks_excluded_from_active_heap(self, tmp_db, repair_network):
        """Completed tasks are excluded from the active Max Heap but retained in history."""
        manager = PriorityQueueManager()
        heap = manager.sync_and_build_heap(tmp_db)
        assert heap.size() == 2

        # Complete the top task
        top_task_id = heap.peek().task_id
        tmp_db.update_repair_task_status(top_task_id, 'Completed')

        # Re-build heap
        active_heap = manager.sync_and_build_heap(tmp_db)
        assert active_heap.size() == 1
        assert active_heap.peek().task_id != top_task_id

        # Archival history retains all tasks
        history = manager.get_all_tasks_for_history(tmp_db)
        assert len(history) == 2


# ==============================================================================
# 5. Flask Routes & Dashboard Tests
# ==============================================================================

class TestRepairRoutes:
    """Integration tests for /repairs endpoints and dashboard integration."""

    def test_get_repairs_page_empty(self, app_client):
        """GET /repairs renders cleanly even when no tasks exist."""
        rv = app_client.get('/repairs')
        assert rv.status_code == 200
        html = rv.data.decode('utf-8')
        assert "Smart Repair Priority Queue" in html
        assert "Algorithmic Priority Ranking Notice" in html
        assert "No active repair tasks in the queue" in html

    def test_get_repairs_page_with_data(self, app_client, repair_network):
        """GET /repairs renders summary metrics, category pills, and task cards."""
        # Trigger sync first
        app_client.post('/repairs/sync')

        rv = app_client.get('/repairs')
        assert rv.status_code == 200
        html = rv.data.decode('utf-8')
        assert "Trunk-North" in html
        assert "Branch-West" in html
        assert "CRITICAL" in html

    def test_get_repair_queue_alias(self, app_client):
        """GET /repair-queue serves the same repairs page."""
        rv = app_client.get('/repair-queue')
        assert rv.status_code == 200
        assert "Smart Repair Priority Queue" in rv.data.decode('utf-8')

    def test_post_status_update(self, app_client, tmp_db, repair_network):
        """POST /repairs/<task_id>/status updates task status and redirects."""
        manager = PriorityQueueManager()
        manager.sync_and_build_heap(tmp_db)
        tasks = tmp_db.get_all_repair_tasks()
        assert len(tasks) > 0
        task_id = tasks[0]['task_id']

        rv = app_client.post(f'/repairs/{task_id}/status', data={'status': 'In Progress'}, follow_redirects=True)
        assert rv.status_code == 200

        updated = tmp_db.get_repair_task_by_id(task_id)
        assert updated['status'] == 'In Progress'

    def test_post_status_method_not_allowed(self, app_client):
        """GET on POST /repairs/<task_id>/status returns 405 Method Not Allowed."""
        rv = app_client.get('/repairs/1/status')
        assert rv.status_code == 405

    def test_post_investigate_action(self, app_client, tmp_db, repair_network):
        """POST /repairs/<task_id>/investigate sets task to In Progress and leak event to investigating."""
        manager = PriorityQueueManager()
        manager.sync_and_build_heap(tmp_db)
        tasks = tmp_db.get_all_repair_tasks()
        task_id = tasks[0]['task_id']
        event_id = tasks[0]['event_id']

        rv = app_client.post(f'/repairs/{task_id}/investigate', follow_redirects=True)
        assert rv.status_code == 200

        updated_task = tmp_db.get_repair_task_by_id(task_id)
        assert updated_task['status'] == 'In Progress'

        # Check leak event
        event = tmp_db.get_leak_event_by_id(event_id)
        assert event['status'] == 'investigating'

    def test_post_sync(self, app_client, repair_network):
        """POST /repairs/sync synchronizes tasks and redirects to /repairs."""
        rv = app_client.post('/repairs/sync', follow_redirects=True)
        assert rv.status_code == 200
        assert "synchronized" in rv.data.decode('utf-8').lower()

    def test_post_sync_method_not_allowed(self, app_client):
        """GET on /repairs/sync returns 405."""
        rv = app_client.get('/repairs/sync')
        assert rv.status_code == 405

    def test_dashboard_home_contains_repair_queue_card(self, app_client, tmp_db, repair_network):
        """GET / includes repair priority queue section and live task count."""
        manager = PriorityQueueManager()
        manager.sync_and_build_heap(tmp_db)

        rv = app_client.get('/')
        assert rv.status_code == 200
        html = rv.data.decode('utf-8')
        assert "Repair Priority Queue" in html
        assert "Pending Repairs" in html
        assert "/repairs" in html
        assert "Trunk-North" in html
