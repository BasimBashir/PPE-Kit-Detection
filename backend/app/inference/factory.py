"""Build the configured detector / classifier (adapters are replaceable - NFR-MAINT-001)."""

from __future__ import annotations

import logging
from pathlib import Path

from app.config.settings import ModelsConfig
from app.inference.types import Detector, HeadClassifier

log = logging.getLogger(__name__)


def build_detector(cfg: ModelsConfig, models_dir: Path) -> Detector:
    if cfg.backend == "mock":
        from app.inference.mock_detector import MockDetector

        return MockDetector()
    from app.inference.ultralytics_detector import UltralyticsCascadeDetector

    return UltralyticsCascadeDetector(cfg, models_dir)


def build_classifier(cfg: ModelsConfig, models_dir: Path) -> HeadClassifier | None:
    if cfg.backend == "mock" or cfg.classifier is None:
        return None
    if not (models_dir / cfg.classifier.weights).exists():
        log.warning("head classifier weights %s missing - classifier disabled", cfg.classifier.weights)
        return None
    from app.inference.classifier import UltralyticsHeadClassifier

    return UltralyticsHeadClassifier(cfg.classifier, models_dir, cfg.classifier_uncertain_below)
