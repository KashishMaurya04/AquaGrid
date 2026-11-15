"""
repair_queue/manager.py – Queue orchestrator bridging database leak events and the custom MaxHeap.
"""

from __future__ import annotations

import json
from typing import Any
from repair_queue.max_heap import MaxHeap, RepairTask
from repair_queue.scoring import calculate_priority_score


class PriorityQueueManager:
    """
    High-level manager for the Smart Repair Priority Queue.
    Orchestrates task synchronization, priority calculation, and MaxHeap sorting.
    """

    def __init__(self) -> None:
        pass

    def sync_and_build_heap(self, db_module: Any) -> MaxHeap[RepairTask]:
        """
        Synchronize open/investigating leak events into repair_tasks,
        build a MaxHeap of active (Pending, In Progress) tasks, and return the heap.
        """
        # 1. Fetch unresolved leak events
        open_events = db_module.get_open_leak_events()

        # 2. Synchronize each event into repair_tasks table
        for ev in open_events:
            ev_dict = dict(ev)
            pipe_id = int(ev_dict["pipe_id"])
            pipe = db_module.get_pipe_by_id(pipe_id)

            score = calculate_priority_score(ev_dict, pipe)

            db_module.create_or_update_repair_task(
                event_id=int(ev_dict["event_id"]),
                pipe_id=pipe_id,
                priority_score=score.total_score,
                priority_category=score.category,
                score_breakdown=json.dumps(score.breakdown),
            )

        # 3. Load all active repair tasks from database
        active_task_rows = db_module.get_active_repair_tasks()
        tasks: list[RepairTask] = []

        for row in active_task_rows:
            r = dict(row)
            breakdown = {}
            if r.get("score_breakdown"):
                try:
                    breakdown = json.loads(r["score_breakdown"])
                except Exception:
                    breakdown = {}

            task = RepairTask(
                task_id=int(r["task_id"]),
                event_id=int(r["event_id"]),
                pipe_id=int(r["pipe_id"]),
                pipe_name=str(r.get("pipe_name", f"Pipe #{r['pipe_id']}")),
                from_node_name=str(r.get("from_node_name", "From")),
                to_node_name=str(r.get("to_node_name", "To")),
                priority_score=float(r["priority_score"]),
                priority_category=str(r["priority_category"]),
                status=str(r["status"]),
                detected_at=str(r.get("detected_at", r.get("created_at", ""))),
                created_at=str(r.get("created_at", "")),
                updated_at=str(r.get("updated_at", "")),
                severity=str(r.get("severity", "medium")),
                flow_anomaly_score=r.get("flow_anomaly_score"),
                pressure_anomaly_score=r.get("pressure_anomaly_score"),
                observed_values=r.get("observed_values"),
                baseline_values=r.get("baseline_values"),
                evidence_description=str(r.get("evidence_description", "")),
                score_breakdown=breakdown,
                assigned_to=r.get("assigned_to"),
                notes=r.get("notes"),
            )
            tasks.append(task)

        # 4. Construct the custom MaxHeap
        heap: MaxHeap[RepairTask] = MaxHeap()
        heap.heapify(tasks)
        return heap

    def get_ordered_tasks(self, db_module: Any) -> list[RepairTask]:
        """
        Return all active repair tasks extracted from the MaxHeap in strict priority order.
        Highest-priority task appears first at index 0.
        """
        heap = self.sync_and_build_heap(db_module)
        return heap.extract_all()

    def get_all_tasks_for_history(self, db_module: Any) -> list[RepairTask]:
        """
        Return all repair tasks (including Completed) joined with pipe details.
        """
        rows = db_module.get_all_repair_tasks()
        tasks: list[RepairTask] = []
        for row in rows:
            r = dict(row)
            breakdown = {}
            if r.get("score_breakdown"):
                try:
                    breakdown = json.loads(r["score_breakdown"])
                except Exception:
                    breakdown = {}

            task = RepairTask(
                task_id=int(r["task_id"]),
                event_id=int(r["event_id"]),
                pipe_id=int(r["pipe_id"]),
                pipe_name=str(r.get("pipe_name", f"Pipe #{r['pipe_id']}")),
                from_node_name=str(r.get("from_node_name", "From")),
                to_node_name=str(r.get("to_node_name", "To")),
                priority_score=float(r["priority_score"]),
                priority_category=str(r["priority_category"]),
                status=str(r["status"]),
                detected_at=str(r.get("detected_at", r.get("created_at", ""))),
                created_at=str(r.get("created_at", "")),
                updated_at=str(r.get("updated_at", "")),
                severity=str(r.get("severity", "medium")),
                flow_anomaly_score=r.get("flow_anomaly_score"),
                pressure_anomaly_score=r.get("pressure_anomaly_score"),
                observed_values=r.get("observed_values"),
                baseline_values=r.get("baseline_values"),
                evidence_description=str(r.get("evidence_description", "")),
                score_breakdown=breakdown,
                assigned_to=r.get("assigned_to"),
                notes=r.get("notes"),
            )
            tasks.append(task)
        return tasks

