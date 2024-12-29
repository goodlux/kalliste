"""Detection framework for identifying and cropping regions in images."""
from .base import BaseDetector,  Region
from .yolo_detector import YOLODetector

__all__ = ['BaseDetector', 'Region', 'YOLODetector']