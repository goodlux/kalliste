"""Writes batch processing statistics to a summary file."""
from pathlib import Path
import logging
from typing import TextIO
from .batch_statistics import BatchStatistics

logger = logging.getLogger(__name__)

class BatchResultsWriter:
    """Writes batch processing results to a summary file."""
    
    def __init__(self, output_dir: Path):
        """Initialize with output directory."""
        self.output_dir = output_dir
        
    def write_results(self, stats: BatchStatistics):
        """Write batch statistics to a results file."""
        try:
            output_path = self.output_dir / "batch_results.txt"
            with output_path.open("w") as f:
                self._write_summary(f, stats)
            logger.info(f"Wrote batch results to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to write batch results: {e}")
            raise
            
    def _write_summary(self, f: TextIO, stats: BatchStatistics):
        """Write formatted statistics summary."""
        # Basic counts
        f.write("=== Batch Processing Summary ===\n\n")
        f.write(f"Original images processed: {stats.original_images_processed}\n\n")
        
        # Region detection summary
        f.write("=== Regions Detected ===\n")
        for region_type, count in stats.regions_by_type.items():
            f.write(f"{region_type}: {count}\n")
        f.write("\n")
        
        # Small regions summary
        f.write("=== Regions Rejected (Too Small) ===\n")
        for region_type, count in stats.small_regions_by_type.items():
            f.write(f"{region_type}: {count}\n")
        f.write("\n")
        
        # Assessments by region type
        for region_type in stats.regions_by_type:
            f.write(f"=== Assessment Results for {region_type} ===\n")
            
            f.write("\nTechnical Assessments:\n")
            for assessment, count in stats.technical_assessments[region_type].items():
                f.write(f"  {assessment}: {count}\n")
                
            f.write("\nAesthetic Assessments:\n")
            for assessment, count in stats.aesthetic_assessments[region_type].items():
                f.write(f"  {assessment}: {count}\n")
                
            f.write("\nOverall NIMA Assessments:\n")
            for assessment, count in stats.overall_assessments[region_type].items():
                f.write(f"  {assessment}: {count}\n")
                
            f.write("\nKalliste Final Assessments:\n")
            for assessment, count in stats.kalliste_assessments[region_type].items():
                f.write(f"  {assessment}: {count}\n")
            
            f.write("\n")
