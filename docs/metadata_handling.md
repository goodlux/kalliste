# Kalliste Metadata Handling Strategy

## Overview
This document outlines the metadata handling strategy for the Kalliste project, including how we preserve original metadata, add Kalliste-specific tags, and handle special cases like people identification and source media types.

## Tag Prefixing Conventions
- `@FnLnLux` - Identifies people (e.g., @JoDoLux)
- `#source` - Identifies source media type (e.g., #photo, #video, #drawing)
  - Can use Lightroom synonyms for unprefixed versions (e.g., "photo") for more subtle training signals

## XMP-kalliste Namespace Fields
- photoshoot_id (from folder name)
- photoshoot_date
- caption 
- wd_tags
- lr_tags (including location/setting)
- orientation_tag (front/back/side)
- other_tags
- all_tags
- source_media
- kalliste_process_version
- crop_type
- lr_rating
- lr_label

## People Identification Logic
Different strategies based on crop type:

1. Face Crops:
   - Use direct face detection results

2. Person Crops:
   - Use direct face detection results
   - If no face detection, use @FnLn tags from "Weighted Flat Subject"

3. Full Crops:
   - Use "Person in Image" field first
   - If empty, scan "Weighted Flat Subject" for @FnLnLux format

## Metadata Copying Strategy
Basic exiftool command structure:
```bash
exiftool -TagsFromFile original.jpg -all:all -ImageSize= -PixelDimensions= \
-XMP-kalliste:PhotoshootId="20131220_EventName" \
-XMP-kalliste:PhotoshootDate="2013:12:20" \
-XMP-kalliste:People="Person1,Person2" \
-XMP-kalliste:Caption="Description here" \
-XMP-kalliste:WDTags="tag1,tag2,tag3" \
-XMP-kalliste:LRTags="tag1,tag2,tag3" \
-XMP-kalliste:OrientationTag="side" \
-XMP-kalliste:OtherTags="future1,future2" \
-XMP-kalliste:AllTags="combined,tags,here" \
-XMP-kalliste:SourceMedia="photo" \
-XMP-kalliste:CropType="face" \
-XMP-kalliste:ProcessVersion="1.0" \
-XMP-kalliste:LRRating=3 \
-XMP-kalliste:LRLabel="Red" \
cropped.jpg
```

## ChromaDB Structure
```python
metadata = {
    "kalliste_fast": {
        "photoshoot_id": "20131220_EventName",
        "photoshoot_date": "2013-12-20",
        "people": ["Person1", "Person2"],
        "caption": "Description here",
        "wd_tags": ["tag1", "tag2", "tag3"],
        "lr_tags": ["tag1", "tag2", "tag3"],
        "orientation_tag": "side",
        "other_tags": ["future1", "future2"],
        "all_tags": ["combined", "tags", "here"],
        "source_media": "photo",
        "crop_type": "face",
        "process_version": "1.0",
        "lr_rating": 3,
        "lr_label": "Red"
    },
    "original_metadata": {
        # All other preserved EXIF/IPTC/XMP data
    }
}
```

## Workflow
1. Run image processing/cropping
2. Resize images
3. Run exiftool to:
   - Copy all metadata except size fields
   - Add XMP-kalliste namespace tags
   - Update correct size metadata for new dimensions
4. Import to ChromaDB with structured metadata
5. ChromaDB search for image selection
6. Copy selected images to LoRA folder
7. Generate sidecar files by extracting relevant XMP-kalliste tags
8. Run LoRA process

## Notes
- Original metadata is preserved through the `-all:all` flag
- Lightroom synonyms can be used for training variations
- Face detection and person identification use multiple fallback strategies
- Source media types use #prefix for clear categorization
- All original file identifiers (Document ID, Raw File Name) are preserved in original metadata