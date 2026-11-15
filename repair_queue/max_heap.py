"""
repair_queue/max_heap.py – Custom Max Heap implementation for AquaGrid.

Design & Constraints:
  - Pure Python list-based binary heap.
  - Custom index-based array representation:
      parent(i) = (i - 1) // 2
      left(i)   = 2 * i + 1
      right(i)  = 2 * i + 2
  - Zero dependencies on standard priority queue libraries.
  - Complete operations:
      * insert(task)
      * extract_max() -> RepairTask | None
      * peek() -> RepairTask | None
      * heapify(tasks)
      * sift_up(index)
      * sift_down(index)
      * is_empty() -> bool
      * size() -> int
      * to_list() -> list[RepairTask]
      * extract_all() -> list[RepairTask]

Deterministic Comparison Logic:
  1. Priority Score: Higher score has higher priority.
  2. FIFO Timestamp: If scores match, older task (earlier created_at) has higher priority.
  3. Unique Task ID: If timestamp also matches, smaller task_id breaks tie deterministically.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Generic, TypeVar

T = TypeVar("T")


@dataclass
class RepairTask:
    """
    Represents an actionable water network repair task derived from a leak event.
    Implements total ordering with deterministic tie-breaking for priority queue management.
    """
    task_id: int
    event_id: int
    pipe_id: int = 0
    pipe_name: str = ""
    from_node_name: str = ""
    to_node_name: str = ""
    priority_score: float = 0.0
    priority_category: str = "LOW"            # 'CRITICAL', 'HIGH', 'MEDIUM', 'LOW'
    status: str = "Pending"                   # 'Pending', 'In Progress', 'Completed'
    detected_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    severity: str = "medium"
    flow_anomaly_score: float | None = None
    pressure_anomaly_score: float | None = None
    observed_values: str | None = None
    baseline_values: str | None = None
    evidence_description: str = ""
    score_breakdown: dict[str, Any] = field(default_factory=dict)
    assigned_to: str | None = None
    notes: str | None = None

    # ── Deterministic Comparison Logic ────────────────────────────────────────
    # In a Max-Heap, a > b indicates that 'a' has strictly higher priority and
    # should be placed closer to the root (extracted first).

    def __gt__(self, other: Any) -> bool:
        if not isinstance(other, RepairTask):
            return NotImplemented
        # 1. Primary key: priority_score (higher score has higher priority)
        if self.priority_score != other.priority_score:
            return self.priority_score > other.priority_score
        # 2. Secondary key: created_at timestamp (FIFO: earlier timestamp has higher priority)
        if self.created_at != other.created_at:
            return self.created_at < other.created_at
        # 3. Tertiary tie-breaker: task_id (smaller id wins)
        return self.task_id < other.task_id

    def __ge__(self, other: Any) -> bool:
        if not isinstance(other, RepairTask):
            return NotImplemented
        return self == other or self > other

    def __lt__(self, other: Any) -> bool:
        if not isinstance(other, RepairTask):
            return NotImplemented
        return not (self >= other)

    def __le__(self, other: Any) -> bool:
        if not isinstance(other, RepairTask):
            return NotImplemented
        return not (self > other)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, RepairTask):
            return NotImplemented
        return (
            self.task_id == other.task_id
            and self.priority_score == other.priority_score
            and self.created_at == other.created_at
        )


class MaxHeap(Generic[T]):
    """
    Pure Python binary Max-Heap implemented using a contiguous dynamic list.
    Guarantees the max-heap property: parent >= child for all nodes.
    """

    def __init__(self, items: list[T] | None = None) -> None:
        self._heap: list[T] = []
        if items:
            self.heapify(items)

    def is_empty(self) -> bool:
        """Return True if heap contains no elements."""
        return len(self._heap) == 0

    def size(self) -> int:
        """Return the number of elements in the heap."""
        return len(self._heap)

    def peek(self) -> T | None:
        """Return the highest priority task (root) without removing it. Returns None if empty."""
        if not self._heap:
            return None
        return self._heap[0]

    def insert(self, task: T) -> None:
        """
        Insert a new task into the heap while maintaining the max-heap property.
        Appends the element to the end of the array and sifts it up to its correct position.
        Complexity: O(log N).
        """
        self._heap.append(task)
        self.sift_up(len(self._heap) - 1)

    def extract_max(self) -> T | None:
        """
        Remove and return the highest priority task (root).
        Returns None if heap is empty.
        Complexity: O(log N).
        """
        if not self._heap:
            return None
        if len(self._heap) == 1:
            return self._heap.pop()

        max_val = self._heap[0]
        # Move the last element to the root and sift down
        self._heap[0] = self._heap.pop()
        self.sift_down(0)
        return max_val

    def sift_up(self, index: int) -> None:
        """
        Move the element at `index` upward until heap invariant parent >= child holds.
        Complexity: O(log N).
        """
        child_idx = index
        while child_idx > 0:
            parent_idx = (child_idx - 1) // 2
            # MaxHeap: if child > parent, swap them
            if self._heap[child_idx] > self._heap[parent_idx]:
                self._heap[child_idx], self._heap[parent_idx] = self._heap[parent_idx], self._heap[child_idx]
                child_idx = parent_idx
            else:
                break

    def sift_down(self, index: int) -> None:
        """
        Move the element at `index` downward until heap invariant holds.
        Complexity: O(log N).
        """
        parent_idx = index
        heap_len = len(self._heap)

        while True:
            left_idx = 2 * parent_idx + 1
            right_idx = 2 * parent_idx + 2
            largest_idx = parent_idx

            if left_idx < heap_len and self._heap[left_idx] > self._heap[largest_idx]:
                largest_idx = left_idx

            if right_idx < heap_len and self._heap[right_idx] > self._heap[largest_idx]:
                largest_idx = right_idx

            if largest_idx != parent_idx:
                self._heap[parent_idx], self._heap[largest_idx] = self._heap[largest_idx], self._heap[parent_idx]
                parent_idx = largest_idx
            else:
                break

    def heapify(self, tasks: list[T]) -> None:
        """
        Construct a valid max-heap from an arbitrary list of tasks in-place in O(N) time
        using bottom-up heap construction (sifting down from the last internal node).
        """
        self._heap = list(tasks)
        heap_len = len(self._heap)
        if heap_len <= 1:
            return

        # Start from the last non-leaf parent node down to 0
        start_idx = (heap_len - 2) // 2
        for idx in range(start_idx, -1, -1):
            self.sift_down(idx)

    def to_list(self) -> list[T]:
        """Return a shallow copy of the underlying array in current heap order."""
        return list(self._heap)

    def extract_all(self) -> list[T]:
        """
        Extract all items sequentially from highest to lowest priority.
        Empties the heap and returns tasks in strictly descending order.
        Complexity: O(N log N).
        """
        result: list[T] = []
        while not self.is_empty():
            item = self.extract_max()
            if item is not None:
                result.append(item)
        return result
