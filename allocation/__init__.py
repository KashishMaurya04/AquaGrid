"""
allocation/__init__.py – Fair Water Allocation module for AquaGrid.
"""

from allocation.engine import (
    ZoneDemandInput,
    ZoneAllocationResult,
    AllocationRunResult,
    allocate_water,
)

__all__ = [
    "ZoneDemandInput",
    "ZoneAllocationResult",
    "AllocationRunResult",
    "allocate_water",
]

