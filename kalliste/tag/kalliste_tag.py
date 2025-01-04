"""Simple tag classes for Kalliste metadata."""
from typing import Set, Any, List, Dict, Sequence
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class KallisteBaseTag:
    """Base class for all Kalliste tags."""
    def __init__(self, name: str):
        if not name.startswith("Kalliste"):  # Case-sensitive check
            raise ValueError("Tag name must start with 'Kalliste' (case-sensitive)")
        self.name = name

    def validate_value(self, value: Any) -> bool:
        """Validate value type - override in subclasses."""
        raise NotImplementedError

    def to_xmp(self) -> str:
        """Format value for XMP - override if needed."""
        return str(self.value)

class KallisteStringTag(KallisteBaseTag):
    """A tag containing a string value."""
    def __init__(self, name: str, value: str):
        super().__init__(name)
        if not self.validate_value(value):
            raise ValueError(f"Value must be string, got {type(value)}")
        self.value = value

    def validate_value(self, value: Any) -> bool:
        return isinstance(value, str)

class KallisteBagTag(KallisteBaseTag):
    """A tag containing an unordered set of strings.
    XMP format: comma-separated list that exiftool will format as a bag"""
    def __init__(self, name: str, value: Set[str]):
        super().__init__(name)
        if not self.validate_value(value):
            raise ValueError(f"Value must be set or list, got {type(value)}")
        self.value = set(value)  # Convert to set if it was a list

    def validate_value(self, value: Any) -> bool:
        return isinstance(value, (set, list))

    def to_xmp(self) -> str:
        """Format as comma-separated list for exiftool."""
        return ", ".join(sorted(self.value))  # Sort for consistent output

class KallisteSeqTag(KallisteBaseTag):
    """A tag containing an ordered sequence of strings.
    XMP format: (value1,value2,value3)"""
    def __init__(self, name: str, value: Sequence[str]):
        super().__init__(name)
        if not self.validate_value(value):
            raise ValueError(f"Value must be list or tuple, got {type(value)}")
        self.value = list(value)  # Convert to list to ensure order

    def validate_value(self, value: Any) -> bool:
        return isinstance(value, (list, tuple))

    def to_xmp(self) -> str:
        """Format as comma-separated values in parentheses."""
        return "(" + ",".join(self.value) + ")"

class KallisteAltTag(KallisteBaseTag):
    """A tag containing alternative versions (like multi-language text).
    XMP format: [lang1:text1,lang2:text2]"""
    def __init__(self, name: str, value: Dict[str, str]):
        super().__init__(name)
        if not self.validate_value(value):
            raise ValueError(f"Value must be dict of lang:text pairs, got {type(value)}")
        self.value = value

    def validate_value(self, value: Any) -> bool:
        return isinstance(value, dict) and all(isinstance(k, str) and isinstance(v, str) 
                                             for k, v in value.items())

    def to_xmp(self) -> str:
        """Format as language-text pairs in square brackets."""
        pairs = [f"{lang}:{text}" for lang, text in self.value.items()]
        return "[" + ",".join(pairs) + "]"

class KallisteIntegerTag(KallisteBaseTag):
    """A tag containing an integer value."""
    def __init__(self, name: str, value: int):
        super().__init__(name)
        if not self.validate_value(value):
            raise ValueError(f"Value must be integer, got {type(value)}")
        self.value = value

    def validate_value(self, value: Any) -> bool:
        return isinstance(value, int)

class KallisteRealTag(KallisteBaseTag):
    """A tag containing a floating point value."""
    def __init__(self, name: str, value: float):
        super().__init__(name)
        if not self.validate_value(value):
            raise ValueError(f"Value must be number, got {type(value)}")
        self.value = float(value)

    def validate_value(self, value: Any) -> bool:
        return isinstance(value, (int, float))

class KallisteDateTag(KallisteBaseTag):
    """A tag containing a datetime value.
    XMP format: YYYY-MM-DDThh:mm:ss±hh:mm"""
    def __init__(self, name: str, value: datetime):
        super().__init__(name)
        if not self.validate_value(value):
            raise ValueError(f"Value must be datetime, got {type(value)}")
        self.value = value

    def validate_value(self, value: Any) -> bool:
        return isinstance(value, datetime)

    def to_xmp(self) -> str:
        """Format as ISO 8601 date string with timezone."""
        # Format with timezone offset (exiftool expects this format)
        return self.value.astimezone().isoformat()