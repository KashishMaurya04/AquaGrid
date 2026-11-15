"""
simulation – AquaGrid Synthetic Telemetry Simulation Engine.

Provides synthetic telemetry generation for active water distribution sensors
in Normal monitoring mode and simulated Leak scenario mode.
"""

from .engine import (
    SensorSimulator,
    SimulatorConfig,
    GeneratedReading,
    DEFAULT_CONFIG,
)

__all__ = [
    "SensorSimulator",
    "SimulatorConfig",
    "GeneratedReading",
    "DEFAULT_CONFIG",
]

