"""Defines the KallisteTag system for metadata handling."""
from typing import Any, Set, Optional, Union, Dict,  Sequence
from enum import Enum
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

class KallisteTagType(Enum):
    STRING = 'string'
    INTEGER = 'integer'
    REAL = 'real'
    BOOLEAN = 'boolean'
    BAG = 'bag'
    SEQ = 'seq'
    ALT = 'alt'

@dataclass
class KallisteTagDefinition:
    """Definition of a tag type"""
    name: str
    type: KallisteTagType
    description: str = ""
    source: str = ""  # e.g. 'lightroom', 'face_detector', etc.
    confidence: Optional[float] = None

    def __post_init__(self):
        """Validate tag attributes after initialization"""
        if self.confidence is not None and not 0 <= self.confidence <= 1:
            raise ValueError("Confidence must be between 0 and 1")

KallisteTagValue = Union[str, int, float, bool, Set[str]]

class KallisteTagBase:
    """Base class for all Kalliste tag types."""
    def __init__(self, name: str, description: str = "", source: str = ""):
        if not name.startswith("Kalliste"):
            raise ValueError("Tag name must start with 'Kalliste'")
        self.name = name
        self.description = description
        self.source = source
        
    def validate(self, value: Any) -> bool:
        """Validate a value for this tag type."""
        raise NotImplementedError
        
    def to_xmp(self, value: Any) -> str:
        """Convert value to XMP format."""
        raise NotImplementedError

    def get_exiftool_type(self) -> str:
        """Get the exiftool type name for this tag type."""
        type_mapping = {
            KallisteStringTag: 'string',
            KallisteIntegerTag: 'integer',
            KallisteRealTag: 'real',
            KallisteBooleanTag: 'boolean',
            KallisteBagTag: 'bag',
            KallisteSeqTag: 'seq',
            KallisteAltTag: 'alt'
        }
        return type_mapping[self.__class__]

class KallisteStringTag(KallisteTagBase):
    """String-valued tag."""
    def validate(self, value: Any) -> bool:
        return isinstance(value, str)
        
    def to_xmp(self, value: str) -> str:
        return str(value)

class KallisteIntegerTag(KallisteTagBase):
    """Integer-valued tag."""
    def validate(self, value: Any) -> bool:
        return isinstance(value, int)
        
    def to_xmp(self, value: int) -> str:
        return str(value)

class KallisteRealTag(KallisteTagBase):
    """Float-valued tag."""
    def validate(self, value: Any) -> bool:
        return isinstance(value, (int, float))
        
    def to_xmp(self, value: Union[int, float]) -> str:
        return str(float(value))

class KallisteBooleanTag(KallisteTagBase):
    """Boolean-valued tag."""
    def validate(self, value: Any) -> bool:
        return isinstance(value, bool)
        
    def to_xmp(self, value: bool) -> str:
        return str(value).lower()

class KallisteBagTag(KallisteTagBase):
    """Tag containing an unordered set of strings."""
    def validate(self, value: Any) -> bool:
        return isinstance(value, (set, list))
        
    def to_xmp(self, value: Set[str]) -> str:
        return "{" + ",".join(str(v) for v in value) + "}"

class KallisteSeqTag(KallisteTagBase):
    """Tag containing an ordered sequence of strings."""
    def validate(self, value: Any) -> bool:
        return isinstance(value, (list, tuple))
        
    def to_xmp(self, value: Sequence[str]) -> str:
        return "(" + ",".join(str(v) for v in value) + ")"

class KallisteAltTag(KallisteTagBase):
    """Tag containing alternative versions (like multi-language text)."""
    def validate(self, value: Any) -> bool:
        return isinstance(value, dict)  # e.g., {'en': 'text', 'fr': 'texte'}
        
    def to_xmp(self, value: Dict[str, str]) -> str:
        parts = [f"{lang}:{text}" for lang, text in value.items()]
        return "[" + ",".join(parts) + "]"

class KallisteTag:
    """Standard Kalliste tags and tag creation utilities."""
    
    # Known standard tags
    PersonName = KallisteStringTag(
        name="KallistePersonName",
        description="Name of detected person",
        source="lightroom"
    )
    
    RegionType = KallisteStringTag(
        name="KallisteRegionType",
        description="Type of detected region (face, person, hand, etc)",
        source="detector"
    )
    
    PhotoshootId = KallisteStringTag(
        name="KallistePhotoshootId",
        description="ID of the photoshoot (folder name)",
        source="system"
    )
    
    OriginalPath = KallisteStringTag(
        name="KallisteOriginalPath",
        description="Full path to original image",
        source="system"
    )
    
    @classmethod
    def create_wd14_category_tag(cls, category: str) -> KallisteBagTag:
        """Create a new WD14 category tag."""
        return KallisteBagTag(
            name=f"KallisteWd14{category}",
            description=f"WD14 tags for {category}",
            source="wd14_tagger"
        )
        
    @classmethod
    def create_tag(cls, type_class: type, name: str, description: str = "", source: str = ""):
        """Create a new tag of specified type."""
        if not name.startswith("Kalliste"):
            name = f"Kalliste{name}"
        return type_class(name=name, description=description, source=source)

    @classmethod
    def get_tag_def(cls, tag_name: str) -> KallisteTagBase:
        """Get the tag definition for a given tag name."""
        # Try standard tags first
        for attr in dir(cls):
            if attr == tag_name:
                return getattr(cls, attr)
        
        # Fall back to creating a string tag
        return cls.create_tag(KallisteStringTag, tag_name)