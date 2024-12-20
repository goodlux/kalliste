# kalliste/utils/metadata.py
from pathlib import Path
from typing import Dict, List, Optional
import json
import subprocess

def get_exif_metadata(image_path: Path) -> List[Optional[Dict]]:
    """Extract metadata from image using exiftool."""
    try:
        # Run exiftool and get JSON output
        result = subprocess.run(
            ['exiftool', '-j', '-G', str(image_path)], 
            capture_output=True, 
            text=True, 
            check=True
        )
        
        metadata = json.loads(result.stdout)[0]
        face_regions = []
        
        # Look for region list indicators in metadata
        region_count = 1
        while True:
            region_type_key = f'XMP:RegionType[{region_count}]'
            
            # Handle single region case
            if region_type_key not in metadata:
                if region_count == 1 and 'XMP:RegionType' in metadata:
                    if metadata['XMP:RegionType'] == 'Face':
                        face_regions.append(_extract_face_metadata(metadata))
                break
            
            # Handle multiple regions case    
            if metadata[region_type_key] == 'Face':
                face_regions.append(_extract_face_metadata(metadata, region_count))
            
            region_count += 1
                
        return face_regions
        
    except subprocess.CalledProcessError as e:
        print(f"Error running exiftool: {e}")
        return []
    except json.JSONDecodeError as e:
        print(f"Error parsing exiftool output: {e}")
        return []
    except Exception as e:
        print(f"Unexpected error extracting metadata: {e}")
        return []

def _extract_face_metadata(metadata: Dict, region_idx: Optional[int] = None) -> Dict:
    """Extract face metadata for a given region index."""
    idx_suffix = f'[{region_idx}]' if region_idx else ''
    
    return {
        'Person In Image': metadata.get(f'XMP:PersonInImage{idx_suffix}', ''),
        'Region Applied To Dimensions W': metadata.get('XMP:RegionAppliedToDimensionsW', '0'),
        'Region Applied To Dimensions H': metadata.get('XMP:RegionAppliedToDimensionsH', '0'),
        'Region Rotation': metadata.get(f'XMP:RegionRotation{idx_suffix}', '0'),
        'Region Name': metadata.get(f'XMP:RegionName{idx_suffix}', ''),
        'Region Type': metadata.get(f'XMP:RegionType{idx_suffix}', ''),
        'Region Area H': metadata.get(f'XMP:RegionAreaH{idx_suffix}', '0'),
        'Region Area W': metadata.get(f'XMP:RegionAreaW{idx_suffix}', '0'),
        'Region Area X': metadata.get(f'XMP:RegionAreaX{idx_suffix}', '0'),
        'Region Area Y': metadata.get(f'XMP:RegionAreaY{idx_suffix}', '0')
    }