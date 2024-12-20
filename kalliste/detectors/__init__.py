"""Detection framework for identifying and cropping regions in images."""
from .base import BaseDetector, DetectionConfig, Region
from .yolo_detector import YOLODetector

__all__ = ['BaseDetector', 'DetectionConfig', 'Region', 'YOLODetector']