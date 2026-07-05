"""Pluggable, modality-agnostic drift detectors.

    from driftguard.detectors import PSIDetector, DomainClassifierDetector, CompositeDetector

Each detector implements the :class:`DriftDetector` protocol (``fit`` + ``detect`` →
``DetectionResult``) and is adapted to a modality by its extractor/estimator, not by new
detector code. See ``docs/DETECTORS.md``.
"""

from driftguard.detectors.base import (
    CompositeDetector,
    DetectionResult,
    DriftDetector,
)
from driftguard.detectors.domain import DomainClassifierDetector
from driftguard.detectors.ks import DescriptorKSDetector
from driftguard.detectors.mmd import MMDDetector
from driftguard.detectors.psi import PSIDetector

__all__ = [
    "DriftDetector",
    "DetectionResult",
    "CompositeDetector",
    "PSIDetector",
    "DomainClassifierDetector",
    "DescriptorKSDetector",
    "MMDDetector",
]
