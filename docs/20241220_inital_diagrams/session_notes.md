# Kalliste Project Session Notes

## Current Setup
- Project location: `/Users/rob/repos/kalliste`
- Data location: `/Volumes/g2/kalliste_db`
- Using conda environment: `kalliste` with Python 3.11

## Pixeltable Testing Findings
- Successfully ingests DNG files but currently only accesses thumbnail/preview images (171x256, 256x171)
- Uses PIL's TiffImagePlugin to read DNGs
- No EXIF data currently being extracted from TIFF container
- Metadata handling needs improvement for DNGs

## DNG Handling Insights
- DNG files contain multiple components:
  - Raw camera data
  - Embedded JPEG previews (when enabled in Lightroom)
  - Metadata container with Lightroom edits
  - TIFF container structure

## Potential Pixeltable Contributions
1. Enhanced DNG Support
   - Full resolution image access
   - Complete metadata extraction
   - Access to embedded previews
   - Integration with Adobe DNG SDK

2. Metadata Handling
   - Better support for DNG-specific metadata
   - Access to Lightroom edits
   - EXIF data extraction

## Adobe DNG SDK Integration
- Would provide access to:
  - DNG metadata container
  - Different preview images
  - Lightroom edit data
  - Color profiles and DNG-specific data
- Could be contributed back to Pixeltable as an enhancement

## Database Organization
- Using clear directory structure:
```
/Volumes/g2/kalliste_db/
├── pixeltable/     # All Pixeltable files
│   ├── data/       # Pixeltable data
│   └── postgres/   # PostgreSQL data
└── img/
    └── test/       # Test image directories
        └── 01_test/  # Initial test images
```

## Key Decisions & Notes
1. Using configuration files instead of environment variables for settings
2. Test files should stay in Lightroom catalog while testing
3. Need to verify DNG preview settings in test files
4. Current focus on direct DNG ingestion before adding preprocessing steps

## Next Steps
1. Test DNG SDK integration
2. Look into accessing full-resolution images
3. Investigate metadata extraction
4. Consider contributing improvements back to Pixeltable project

## Development Workflow
- Configuration managed in `config/settings.py`
- Tests and scripts in project root
- Data and images kept on G2 drive for scalability