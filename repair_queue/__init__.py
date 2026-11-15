"""
repair_queue/__init__.py – Smart Repair Priority Queue package for AquaGrid.

Exports:
  - MaxHeap: Custom list-based max-heap (no heapq or external libraries).
  - RepairTask: Comparable task model with deterministic tie-breaking.
  - PriorityScore, calculate_priority_score: Normalized 0-100 scoring engine.
  - PriorityQueueManager: High-level queue orchestrator.
"""

from repair_queue.max_heap import MaxHeap, RepairTask
from repair_queue.scoring import PriorityScore, calculate_priority_score, PRIORITY_THRESHOLDS
from repair_queue.manager import PriorityQueueManager

__all__ = [
    "MaxHeap",
    "RepairTask",
    "PriorityScore",
    "calculate_priority_score",
    "PRIORITY_THRESHOLDS",
    "PriorityQueueManager",
]

