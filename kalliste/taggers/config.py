"""Configuration settings for Kalliste tagger pipeline."""

from typing import Dict, List, Set, Optional
from dataclasses import dataclass, field

# TODO: Make more robust blocklist handling
def get_default_wd14_blacklist() -> Set[str]:
    return {
        'questionable', 'explicit', 'nude', 'nudity', 'nipples', 'pussy',
        'penis', 'sex', 'cum', 'penetration', 'penetrated'
    }

def get_default_taggers() -> Set[str]:
    return {'caption', 'wd14', 'orientation'}

@dataclass
class TaggingDefaults:
    """Default settings for taggers."""
    # WD14 settings
    WD14_CONFIDENCE_THRESHOLD: float = 0.35
    WD14_CATEGORY_FILTERS: Optional[List[str]] = None
    WD14_DEFAULT_BLACKLIST: Set[str] = field(default_factory=get_default_wd14_blacklist)
    
    # BLIP2 caption settings
    CAPTION_MAX_LENGTH: int = 100
    CAPTION_TEMPERATURE: float = 1.0
    CAPTION_LENGTH_PENALTY: float = 1.0
    CAPTION_REPETITION_PENALTY: float = 1.5

    # Pipeline settings
    DEFAULT_ENABLED_TAGGERS: Set[str] = field(default_factory=get_default_taggers)

@dataclass
class PipelineConfig:
    """Runtime configuration for tagger pipeline."""
    device: Optional[str] = None
    enable_caption: bool = True
    enable_wd14: bool = True
    enable_orientation: bool = True
    wd14_confidence: float = TaggingDefaults.WD14_CONFIDENCE_THRESHOLD
    wd14_categories: Optional[List[str]] = None
    wd14_blacklist: Optional[Set[str]] = field(default_factory=get_default_wd14_blacklist)
    caption_max_length: int = TaggingDefaults.CAPTION_MAX_LENGTH
    caption_temperature: float = TaggingDefaults.CAPTION_TEMPERATURE
    caption_length_penalty: float = TaggingDefaults.CAPTION_LENGTH_PENALTY
    caption_repetition_penalty: float = TaggingDefaults.CAPTION_REPETITION_PENALTY

    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'PipelineConfig':
        """Create config from dictionary of settings."""
        return cls(**{
            k: v for k, v in config_dict.items()
            if k in cls.__dataclass_fields__
        })