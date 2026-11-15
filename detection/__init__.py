"""
detection – AquaGrid Sliding-Window Leak Detection Engine.

Provides statistical sliding-window anomaly detection across municipal water distribution pipes.
"""

from .engine import (
    LeakDetector,
    DetectorConfig,
    PipeDiagnostic,
    SensorAnalysisResult,
    DEFAULT_DETECTOR_CONFIG,
    build_detection_result_message,
)

__all__ = [
    "LeakDetector",
    "DetectorConfig",
    "PipeDiagnostic",
    "SensorAnalysisResult",
    "DEFAULT_DETECTOR_CONFIG",
    "build_detection_result_message",
]

