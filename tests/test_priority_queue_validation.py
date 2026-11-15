"""
tests/test_priority_queue_validation.py – Day 5: Smart Repair Priority Queue Multi-Task Validation.

Comprehensive validation of AquaGrid's Custom Max Heap, ranking correctness,
deterministic tie-breaking, priority scoring formula, database synchronization,
and computational complexity bounds.
"""

from __future__ import annotations

import math
import random
from datetime import datetime, timezone, timedelta
from typing import Any
import pytest

from repair_queue.max_heap import MaxHeap, RepairTask
from repair_queue.scoring import calculate_priority_score, PRIORITY_THRESHOLDS
from repair_queue.manager import PriorityQueueManager
import database as db_module


def verify_max_heap_invariant(heap: MaxHeap[RepairTask]) -> bool:
    """Verifies that for every node in the heap, parent >= child."""
    nodes = heap._heap
    for i in range(1, len(nodes)):
        parent_idx = (i - 1) // 2
        parent = nodes[parent_idx]
        child = nodes[i]
        if not (parent >= child):
            return False
    return True


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture()
def isolated_db(monkeypatch, tmp_path):
    """Isolated SQLite database in a temporary directory for integration tests."""
    test_db = str(tmp_path / "test_pq_val.db")
    monkeypatch.setattr(db_module, "INSTANCE_DIR", str(tmp_path))
    monkeypatch.setattr(db_module, "DB_PATH", test_db)
    db_module.init_db()
    return db_module


# ==============================================================================
# 1. Max Heap Correctness & Invariant Validation
# ==============================================================================

class TestMaxHeapValidation:
    """Comprehensive multi-task correctness tests for pure Python MaxHeap."""

    def test_heap_invariant_after_every_insertion_and_extraction(self):
        """Validates that parent >= child invariant holds after every single insert and extract."""
        random.seed(999)
        scores = [round(random.uniform(0.0, 100.0), 1) for _ in range(40)]
        heap = MaxHeap[RepairTask]()

        # Insertions
        for idx, score in enumerate(scores):
            task = RepairTask(task_id=idx + 1, event_id=idx + 1, priority_score=score)
            heap.insert(task)
            assert verify_max_heap_invariant(heap), f"Invariant violated after inserting score {score}"

        assert heap.size() == 40

        # Extractions
        extracted_scores = []
        while not heap.is_empty():
            task = heap.extract_max()
            assert task is not None
            extracted_scores.append(task.priority_score)
            assert verify_max_heap_invariant(heap), "Invariant violated after extract_max"

        assert extracted_scores == sorted(scores, reverse=True)
        assert heap.is_empty() is True

    def test_priority_ranking_exact_scores(self):
        """
        Validates exact priority scores extraction order:
        [100, 95, 85, 75, 60, 40, 20, 0] across ascending, descending, and shuffled inputs.
        """
        target_scores = [100.0, 95.0, 85.0, 75.0, 60.0, 40.0, 20.0, 0.0]

        # Case 1: Ascending insertion
        h_asc = MaxHeap[RepairTask]()
        for i, s in enumerate(reversed(target_scores)):
            h_asc.insert(RepairTask(task_id=i + 1, event_id=i + 1, priority_score=s))
            assert verify_max_heap_invariant(h_asc)
        assert [t.priority_score for t in h_asc.extract_all()] == target_scores

        # Case 2: Descending insertion
        h_desc = MaxHeap[RepairTask]()
        for i, s in enumerate(target_scores):
            h_desc.insert(RepairTask(task_id=i + 1, event_id=i + 1, priority_score=s))
            assert verify_max_heap_invariant(h_desc)
        assert [t.priority_score for t in h_desc.extract_all()] == target_scores

        # Case 3: Random shuffle
        shuffled = list(target_scores)
        random.seed(123)
        random.shuffle(shuffled)
        h_shuf = MaxHeap[RepairTask]()
        for i, s in enumerate(shuffled):
            h_shuf.insert(RepairTask(task_id=i + 1, event_id=i + 1, priority_score=s))
            assert verify_max_heap_invariant(h_shuf)
        assert [t.priority_score for t in h_shuf.extract_all()] == target_scores

    def test_deterministic_tie_breaking_three_levels(self):
        """
        Validates the 3 deterministic tie-breaking levels:
        1. Higher priority score first.
        2. Earlier created_at first when scores are equal.
        3. Lower task_id first when scores and timestamps are equal.
        """
        t_base = datetime(2026, 11, 1, 10, 0, 0, tzinfo=timezone.utc)
        t_early = t_base.isoformat()
        t_late = (t_base + timedelta(hours=1)).isoformat()

        # Level 1: Priority score difference
        t_high = RepairTask(task_id=100, event_id=1, priority_score=90.0, created_at=t_late)
        t_low = RepairTask(task_id=1, event_id=2, priority_score=80.0, created_at=t_early)
        assert t_high > t_low

        # Level 2: Equal score, earlier timestamp wins
        t_older = RepairTask(task_id=50, event_id=3, priority_score=75.0, created_at=t_early)
        t_newer = RepairTask(task_id=2, event_id=4, priority_score=75.0, created_at=t_late)
        assert t_older > t_newer
        h_lvl2 = MaxHeap[RepairTask]()
        h_lvl2.insert(t_newer)
        h_lvl2.insert(t_older)
        assert h_lvl2.extract_max() is t_older
        assert h_lvl2.extract_max() is t_newer

        # Level 3: Equal score and equal timestamp, lower task_id wins
        t_id_low = RepairTask(task_id=10, event_id=5, priority_score=75.0, created_at=t_early)
        t_id_high = RepairTask(task_id=20, event_id=6, priority_score=75.0, created_at=t_early)
        assert t_id_low > t_id_high
        h_lvl3 = MaxHeap[RepairTask]()
        h_lvl3.insert(t_id_high)
        h_lvl3.insert(t_id_low)
        assert h_lvl3.extract_max() is t_id_low
        assert h_lvl3.extract_max() is t_id_high

    def test_heap_operations_interleaved(self):
        """Tests peek, insert, and extraction interleaved in arbitrary order."""
        heap = MaxHeap[RepairTask]()
        assert heap.peek() is None

        t1 = RepairTask(task_id=1, event_id=1, priority_score=50.0)
        heap.insert(t1)
        assert heap.peek() is t1
        assert heap.size() == 1

        t2 = RepairTask(task_id=2, event_id=2, priority_score=85.0)
        heap.insert(t2)
        assert heap.peek() is t2

        t3 = RepairTask(task_id=3, event_id=3, priority_score=65.0)
        heap.insert(t3)
        assert heap.peek() is t2

        # Extract top (t2)
        extracted = heap.extract_max()
        assert extracted is t2
        assert heap.peek() is t3

        # Insert new higher task
        t4 = RepairTask(task_id=4, event_id=4, priority_score=95.0)
        heap.insert(t4)
        assert heap.peek() is t4

        # Drain
        remaining = heap.extract_all()
        assert [t.priority_score for t in remaining] == [95.0, 65.0, 50.0]
        assert heap.is_empty() is True

    def test_heapify_linear_time_construction(self):
        """heapify() converts arbitrary list into a valid max heap in-place."""
        tasks = [
            RepairTask(task_id=i, event_id=i, priority_score=score)
            for i, score in enumerate([10.0, 60.0, 40.0, 95.0, 80.0, 30.0, 70.0])
        ]
        heap = MaxHeap[RepairTask]()
        heap.heapify(tasks)
        assert verify_max_heap_invariant(heap)
        assert heap.peek().priority_score == 95.0
        assert [t.priority_score for t in heap.extract_all()] == [95.0, 80.0, 70.0, 60.0, 40.0, 30.0, 10.0]


# ==============================================================================
# 2. Priority Scoring Formula Validation
# ==============================================================================

class TestPriorityScoringValidation:
    """Validation of multi-factor hydraulic anomaly priority scoring formula."""

    def test_scoring_components_breakdown_and_bounds(self):
        """Validates maximum 100 score decomposition and [0.0, 100.0] clamping."""
        res_max = calculate_priority_score(
            flow_z_score=5.0,          # max 30.0
            pressure_z_score=5.0,      # max 30.0
            event_severity="critical", # 20.0
            consecutive_abnormal=6,    # max 10.0
            pipe_capacity_lpm=2500.0,  # max 10.0
        )
        assert res_max.total_score == 100.0
        assert res_max.category == "CRITICAL"
        assert res_max.flow_component == 30.0
        assert res_max.pressure_component == 30.0
        assert res_max.severity_component == 20.0
        assert res_max.persistence_component == 10.0
        assert res_max.pipe_importance_component == 10.0

    def test_exact_category_thresholds(self):
        """Validates threshold borders: CRITICAL (>=80), HIGH (60-79.9), MEDIUM (35-59.9), LOW (<35)."""
        # 80.0 CRITICAL
        s80 = calculate_priority_score(flow_z_score=2.0, pressure_z_score=2.0, event_severity="critical", consecutive_abnormal=5, pipe_capacity_lpm=1000.0)
        assert s80.total_score == 80.0
        assert s80.category == "CRITICAL"

        # 79.9 HIGH
        s79_9 = calculate_priority_score(flow_z_score=2.0, pressure_z_score=2.0, event_severity="critical", consecutive_abnormal=5, pipe_capacity_lpm=990.0)
        assert s79_9.total_score == 79.9
        assert s79_9.category == "HIGH"

        # 60.0 HIGH
        s60 = calculate_priority_score(flow_z_score=1.5, pressure_z_score=1.5, event_severity="high", consecutive_abnormal=5, pipe_capacity_lpm=500.0)
        assert s60.total_score == 60.0
        assert s60.category == "HIGH"

        # 59.9 MEDIUM
        s59_9 = calculate_priority_score(flow_z_score=1.5, pressure_z_score=1.5, event_severity="high", consecutive_abnormal=5, pipe_capacity_lpm=490.0)
        assert s59_9.total_score == 59.9
        assert s59_9.category == "MEDIUM"

        # 35.0 MEDIUM
        s35 = calculate_priority_score(flow_z_score=1.0, pressure_z_score=0.5, event_severity="medium", consecutive_abnormal=3, pipe_capacity_lpm=400.0)
        assert s35.total_score == 35.0
        assert s35.category == "MEDIUM"

        # 34.9 LOW
        s34_9 = calculate_priority_score(flow_z_score=1.0, pressure_z_score=0.5, event_severity="medium", consecutive_abnormal=3, pipe_capacity_lpm=390.0)
        assert s34_9.total_score == 34.9
        assert s34_9.category == "LOW"

    def test_missing_telemetry_transparent_handling(self):
        """Pipes with missing sensor readings yield 0 for that metric and log transparent explanation."""
        res = calculate_priority_score(
            flow_z_score=None,
            pressure_z_score=2.0,
            event_severity="medium",
            consecutive_abnormal=None,
            pipe_capacity_lpm=None,
        )
        assert res.flow_component == 0.0
        assert res.pressure_component == 20.0
        assert res.severity_component == 10.0
        assert res.persistence_component == 0.0
        assert res.pipe_importance_component == 0.0
        assert res.total_score == 30.0
        assert res.category == "LOW"
        assert "no flow meter" in res.explanation.lower()

    def test_scoring_determinism_and_audit_disclaimer(self):
        """Confirms 50 repeated evaluations produce identical outputs and audit disclaimer is present."""
        scores = [
            calculate_priority_score(
                flow_z_score=3.1,
                pressure_z_score=-2.8,
                event_severity="critical",
                consecutive_abnormal=5,
                pipe_capacity_lpm=1200.0,
            ).total_score
            for _ in range(50)
        ]
        assert len(set(scores)) == 1
        assert "Audit Disclaimer" in calculate_priority_score(flow_z_score=1.0).explanation


# ==============================================================================
# 3. Database & Priority Queue Integration
# ==============================================================================

class TestDatabaseQueueIntegration:
    """Integration of database leak events with Smart Repair Priority Queue."""

    def test_multi_task_sync_preserves_priority_order(self, isolated_db):
        """Creates multiple leak events and confirms queue extraction preserves calculated priorities."""
        n1 = isolated_db.create_node("Node-A", "tank", "North", "active")
        n2 = isolated_db.create_node("Node-B", "junction", "Central", "active")
        n3 = isolated_db.create_node("Node-C", "valve", "South", "active")

        p1 = isolated_db.create_pipe("Trunk-High", n1, n2, "100.0", "1500.0", "1.0", "active")
        p2 = isolated_db.create_pipe("Branch-Med", n2, n3, "150.0", "500.0", "1.0", "active")
        p3 = isolated_db.create_pipe("Service-Low", n1, n3, "200.0", "200.0", "1.2", "active")

        ev1, _ = isolated_db.record_leak_event(p1, "critical", "Coincident surge 5 consecutive", flow_anomaly_score=4.0, pressure_anomaly_score=-3.5, is_simulated=True)
        ev2, _ = isolated_db.record_leak_event(p2, "high", "Moderate surge 4 consecutive", flow_anomaly_score=2.0, pressure_anomaly_score=-1.5, is_simulated=True)
        ev3, _ = isolated_db.record_leak_event(p3, "low", "Minor anomaly 3 consecutive", flow_anomaly_score=0.5, pressure_anomaly_score=None, is_simulated=True)

        manager = PriorityQueueManager()
        ordered = manager.get_ordered_tasks(isolated_db)
        assert len(ordered) == 3
        assert ordered[0].event_id == ev1
        assert ordered[1].event_id == ev2
        assert ordered[2].event_id == ev3
        assert ordered[0].priority_score > ordered[1].priority_score > ordered[2].priority_score

    def test_one_to_one_task_event_and_deduplication(self, isolated_db):
        """Repeated queue synchronizations do not duplicate repair tasks."""
        n1 = isolated_db.create_node("N1", "tank", "North", "active")
        n2 = isolated_db.create_node("N2", "junction", "Central", "active")
        p1 = isolated_db.create_pipe("P1", n1, n2, "100.0", "1000.0", "1.0", "active")
        ev1, _ = isolated_db.record_leak_event(p1, "high", "Surge 4 consecutive", flow_anomaly_score=2.5, pressure_anomaly_score=None, is_simulated=True)

        manager = PriorityQueueManager()
        manager.sync_and_build_heap(isolated_db)
        manager.sync_and_build_heap(isolated_db)
        manager.sync_and_build_heap(isolated_db)

        all_tasks = isolated_db.get_all_repair_tasks()
        assert len(all_tasks) == 1
        assert all_tasks[0]["event_id"] == ev1

    def test_completed_tasks_excluded_from_active_queue_preserved_in_history(self, isolated_db):
        """Completed repair tasks are excluded from active dispatch queue but preserved in all_tasks."""
        n1 = isolated_db.create_node("N1", "tank", "North", "active")
        n2 = isolated_db.create_node("N2", "junction", "Central", "active")
        p1 = isolated_db.create_pipe("P1", n1, n2, "100.0", "1000.0", "1.0", "active")
        p2 = isolated_db.create_pipe("P2", n1, n2, "200.0", "500.0", "1.0", "active")

        ev1, _ = isolated_db.record_leak_event(p1, "critical", "Surge 5 consecutive", flow_anomaly_score=4.0, pressure_anomaly_score=-3.0, is_simulated=True)
        ev2, _ = isolated_db.record_leak_event(p2, "medium", "Surge 3 consecutive", flow_anomaly_score=1.5, pressure_anomaly_score=None, is_simulated=True)

        manager = PriorityQueueManager()
        heap = manager.sync_and_build_heap(isolated_db)
        assert heap.size() == 2

        # Complete top task
        top_task_id = heap.peek().task_id
        isolated_db.update_repair_task_status(top_task_id, "Completed")

        # Re-build active queue
        active_tasks = manager.get_ordered_tasks(isolated_db)
        assert len(active_tasks) == 1
        assert active_tasks[0].task_id != top_task_id

        # Preserved in history
        history = manager.get_all_tasks_for_history(isolated_db)
        assert len(history) == 2


# ==============================================================================
# 4. Complexity & Edge Cases Validation
# ==============================================================================

class TestComplexityAndEdgeCases:
    """Empirical complexity confirmation and extreme edge cases."""

    def test_heap_operational_complexity_bounds(self):
        """Confirms comparisons during insert and extract are bounded by tree height O(log N)."""
        class CountingMaxHeap(MaxHeap[RepairTask]):
            def __init__(self):
                self.comparisons = 0
                super().__init__()

            def sift_up(self, index: int) -> None:
                child_idx = index
                while child_idx > 0:
                    parent_idx = (child_idx - 1) // 2
                    self.comparisons += 1
                    if self._heap[child_idx] > self._heap[parent_idx]:
                        self._heap[child_idx], self._heap[parent_idx] = self._heap[parent_idx], self._heap[child_idx]
                        child_idx = parent_idx
                    else:
                        break

            def sift_down(self, index: int) -> None:
                parent_idx = index
                heap_len = len(self._heap)
                while True:
                    left_idx = 2 * parent_idx + 1
                    right_idx = 2 * parent_idx + 2
                    largest_idx = parent_idx

                    if left_idx < heap_len:
                        self.comparisons += 1
                        if self._heap[left_idx] > self._heap[largest_idx]:
                            largest_idx = left_idx

                    if right_idx < heap_len:
                        self.comparisons += 1
                        if self._heap[right_idx] > self._heap[largest_idx]:
                            largest_idx = right_idx

                    if largest_idx != parent_idx:
                        self._heap[parent_idx], self._heap[largest_idx] = self._heap[largest_idx], self._heap[parent_idx]
                        parent_idx = largest_idx
                    else:
                        break

        sizes = [15, 31, 63, 127, 255]
        for n in sizes:
            h = CountingMaxHeap()
            tree_height = math.floor(math.log2(n))

            # Worst-case ascending insertions
            for i in range(n):
                before = h.comparisons
                h.insert(RepairTask(task_id=i + 1, event_id=i + 1, priority_score=float(i)))
                comps = h.comparisons - before
                assert comps <= tree_height + 1, f"Insert comparisons {comps} exceeded tree height {tree_height} + 1"

            # Worst-case extractions
            while not h.is_empty():
                before = h.comparisons
                h.extract_max()
                comps = h.comparisons - before
                assert comps <= 2 * (tree_height + 1), f"Extract comparisons {comps} exceeded 2 * height"

    def test_edge_cases_negative_scores_and_duplicate_ids(self):
        """Confirms negative scores and duplicate task IDs are handled safely without crash."""
        t_neg = RepairTask(task_id=1, event_id=1, priority_score=-5.0)
        t_zero = RepairTask(task_id=2, event_id=2, priority_score=0.0)
        assert t_zero > t_neg

        heap = MaxHeap[RepairTask]([t_neg, t_zero])
        assert heap.extract_max() is t_zero
        assert heap.extract_max() is t_neg

        # Duplicate IDs with same score
        t_dup1 = RepairTask(task_id=99, event_id=1, priority_score=50.0, created_at="2026-11-01T12:00:00")
        t_dup2 = RepairTask(task_id=99, event_id=2, priority_score=50.0, created_at="2026-11-01T12:00:00")
        heap_dup = MaxHeap[RepairTask]([t_dup1, t_dup2])
        assert heap_dup.size() == 2
        e1 = heap_dup.extract_max()
        e2 = heap_dup.extract_max()
        assert e1.task_id == 99 and e2.task_id == 99

