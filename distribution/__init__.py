"""
distribution – Network-Aware Water Distribution Optimization package for AquaGrid.

Provides path capacity analysis, feasibility evaluation, and delivery simulation
for consumer zones following fair water allocation.
"""

from distribution.engine import (
    DistributionPlanResult,
    ZoneDeliveryInput,
    ZoneDeliveryResult,
    assess_distribution,
    FEASIBILITY_STATUSES,
    STATUS_FEASIBLE,
    STATUS_CAPACITY_CONSTRAINED,
    STATUS_UNREACHABLE,
    STATUS_UNMAPPED,
    STATUS_ZERO_ALLOCATION,
    STRATEGY_SHORTEST_PATH,
    STRATEGY_WIDEST_PATH,
    SUPPORTED_ROUTING_STRATEGIES,
)
from distribution.service import evaluate_and_record_run_distribution

__all__ = [
    "DistributionPlanResult",
    "ZoneDeliveryInput",
    "ZoneDeliveryResult",
    "assess_distribution",
    "evaluate_and_record_run_distribution",
    "FEASIBILITY_STATUSES",
    "STATUS_FEASIBLE",
    "STATUS_CAPACITY_CONSTRAINED",
    "STATUS_UNREACHABLE",
    "STATUS_UNMAPPED",
    "STATUS_ZERO_ALLOCATION",
    "STRATEGY_SHORTEST_PATH",
    "STRATEGY_WIDEST_PATH",
    "SUPPORTED_ROUTING_STRATEGIES",
]

