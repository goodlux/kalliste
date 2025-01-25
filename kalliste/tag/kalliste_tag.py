"""Simple tag classes for Kalliste metadata."""
from typing import Set, Any, List, Dict, Sequence, Union
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
    
    def to_chroma(self) -> Any:
        """Convert tag value to ChromaDB-compatible format."""
        return self.value

class KallisteStringTag(KallisteBaseTag):
    """A tag containing a string value."""
    def __init__(self, name: str, value: str):
        super().__init__(name)
        if not self.validate_value(value):
            raise ValueError(f"Value must be string, got {type(value)}")
        self.value = value

    def validate_value(self, value: Any) -> bool:
        return isinstance(value, str)
    
    def to_chroma(self) -> str:
        return str(self.value)

class KallisteBagTag(KallisteBaseTag):
    """A tag containing an unordered set of strings."""
    def __init__(self, name: str, value: Set[str]):
        super().__init__(name)
        if not self.validate_value(value):
            raise ValueError(f"Value must be set or list, got {type(value)}")
        self.value = set(value)  # Convert to set if it was a list

    def validate_value(self, value: Any) -> bool:
        return isinstance(value, (set, list))
    
    def to_xmp(self) -> str:
        """Format as comma-separated list for exiftool."""
        # Clean up any Python formatting artifacts and normalize
        cleaned_values = []
        for val in self.value:
            # Remove Python set/dict formatting chars
            val = str(val).strip("'{}")
            # Remove any extra whitespace
            val = val.strip()
            cleaned_values.append(val)
        return ", ".join(sorted(cleaned_values))
    
    def to_chroma(self) -> List[str]:
        return sorted(str(x).strip() for x in self.value if x and str(x).strip())

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
        """Format as comma-separated values for exiftool."""
        return ",".join(self.value)
    
    def to_chroma(self) -> List[str]:
        return [str(x) for x in self.value]

class KallisteBooleanTag(KallisteBaseTag):
    """A tag containing a boolean value."""
    def __init__(self, name: str, value: bool):
        super().__init__(name)
        if not self.validate_value(value):
            raise ValueError(f"Value must be boolean, got {type(value)}")
        self.value = value

    def validate_value(self, value: Any) -> bool:
        return isinstance(value, bool)

    def to_xmp(self) -> str:
        """Format as 'True' or 'False' for exiftool."""
        return str(self.value).lower()  # exiftool expects lowercase true/false

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
        """Format as comma-separated lang:text pairs for exiftool."""
        return ",".join(f"{lang}:{text}" for lang, text in self.value.items())

class KallisteIntegerTag(KallisteBaseTag):
    """A tag containing an integer value."""
    def __init__(self, name: str, value: int):
        super().__init__(name)
        if not self.validate_value(value):
            raise ValueError(f"Value must be integer, got {type(value)}")
        self.value = value

    def validate_value(self, value: Any) -> bool:
        return isinstance(value, int)
    
    def to_chroma(self) -> int:
        return int(self.value)

class KallisteRealTag(KallisteBaseTag):
    """A tag containing a floating point value."""
    def __init__(self, name: str, value: float):
        super().__init__(name)
        if not self.validate_value(value):
            raise ValueError(f"Value must be number, got {type(value)}")
        self.value = float(value)

    def validate_value(self, value: Any) -> bool:
        return isinstance(value, (int, float))
    
    def to_chroma(self) -> float:
        return float(self.value)

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
    
    def to_chroma(self) -> str:
        return self.value.isoformat()
    
class KallisteStructureTag(KallisteBaseTag):
    """
    A tag containing structured data following exiftool's XMP structure format.
    
    Examples:
        Simple structure:
            KallisteStructureTag("KallisteFaceLocation", {
                "x": 100,
                "y": 200,
                "width": 50,
                "height": 75,
                "confidence": 0.95
            })
            
        Nested structure:
            KallisteStructureTag("KallisteDetection", {
                "detector": "yolov8",
                "location": {
                    "x": 100,
                    "y": 200,
                    "width": 50,
                    "height": 75
                },
                "confidence": 0.95
            })
            
        Array of structures:
            KallisteStructureTag("KallisteDetections", [
                {
                    "detector": "yolov8",
                    "class": "person",
                    "confidence": 0.95
                },
                {
                    "detector": "yolov8", 
                    "class": "face",
                    "confidence": 0.87
                }
            ])
    """
    def __init__(self, name: str, value: Union[Dict[str, Any], List[Dict[str, Any]]]):
        super().__init__(name)
        if not self.validate_value(value):
            raise ValueError(f"Value must be dict or list of dicts, got {type(value)}")
        self.value = value

    def validate_value(self, value: Any) -> bool:
        """Validate structure format recursively."""
        if isinstance(value, dict):
            return all(isinstance(k, str) for k in value.keys())
        elif isinstance(value, list):
            return all(isinstance(v, dict) and self.validate_value(v) for v in value)
        return False

    def _format_value(self, v: Any) -> str:
        """Format a value according to exiftool's structure syntax."""
        if isinstance(v, dict):
            return "{" + ",".join(f"{k}={self._format_value(val)}" for k, val in v.items()) + "}"
        elif isinstance(v, list):
            return "[" + ",".join(self._format_value(x) for x in v) + "]"
        elif isinstance(v, bool):
            return str(v).lower()
        elif isinstance(v, (int, float)):
            return str(v)
        else:
            # Strings and everything else - escape commas and wrap in quotes if needed
            val_str = str(v)
            if ',' in val_str or ' ' in val_str:
                return f'"{val_str}"'
            return val_str

    def to_xmp(self) -> str:
        """
        Format according to exiftool's XMP structure syntax.
        Single structure: {field1=value1,field2=value2}
        Array of structures: [{field1=value1},{field1=value2}]
        """
        return self._format_value(self.value)
    
    def to_chroma(self) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        def convert_value(v: Any) -> Any:
            if isinstance(v, dict):
                return {k: convert_value(val) for k, val in v.items()}
            elif isinstance(v, list):
                return [convert_value(x) for x in v]
            elif isinstance(v, (int, float, str, bool)):
                return v
            return str(v)
        return convert_value(self.value)