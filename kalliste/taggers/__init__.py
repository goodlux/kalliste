"""Kalliste image tagging system."""

from .base_tagger import BaseTagger, TagResult, get_default_device
from .caption_tagger import CaptionTagger
from .wd14_tagger import WD14Tagger
from .orientation_tagger import OrientationTagger
from .tagger_pipeline import TaggerPipeline

__all__ = [
    'BaseTagger',
    'TagResult',
    'get_default_device',
    'CaptionTagger',
    'WD14Tagger',
    'OrientationTagger',
    'TaggerPipeline'
]