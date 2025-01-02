"""Writes kalliste tags to XMP metadata."""
import subprocess
from pathlib import Path
from typing import Dict, Any, List
import logging
from io import StringIO
import tempfile
import os
from ..tag.kalliste_tag import KallisteBagTag, KallisteSeqTag, KallisteAltTag, KallisteRealTag, KallisteIntegerTag, KallisteDateTag

logger = logging.getLogger(__name__)

class ExifWriter:
    """Writes kalliste tags to XMP metadata."""
    
    def __init__(self, source_path: Path, dest_path: Path):
        self.source_path = source_path
        self.dest_path = dest_path
        
    def write_tags(self, kalliste_tags: Dict[str, Any]) -> bool:
        """Write kalliste tags to XMP metadata."""
        temp_config = None
        try:
            if not self._validate_paths():
                return False

            # Create temp config file
            temp_config = tempfile.NamedTemporaryFile(mode='w', suffix='.config', delete=False)
            config_content = self._generate_config(kalliste_tags)
            temp_config.write(config_content)
            temp_config.flush()
            temp_config.close()  # Close but don't delete yet
            
            logger.debug(f"Created temp config at: {temp_config.name}")
            logger.debug(f"Config content:\n{config_content}")
            
            # Build and execute exiftool command using temp config
            cmd = self._build_exiftool_command(kalliste_tags, temp_config.name)
            success = self._execute_command(cmd)
            
            # Clean up temp file
            os.unlink(temp_config.name)
            return success
            
        except Exception as e:
            logger.error(f"Failed to write XMP tags: {e}")
            return False
        finally:
            # Ensure temp file cleanup in case of errors
            if temp_config and os.path.exists(temp_config.name):
                try:
                    os.unlink(temp_config.name)
                except Exception as e:
                    logger.error(f"Failed to cleanup temp file: {e}")

    def _validate_paths(self) -> bool:
        """Ensure source exists and destination path's parent exists."""
        try:
            if not self.source_path.exists():
                logger.error(f"Source file does not exist: {self.source_path}")
                return False
                
            dest_dir = self.dest_path.parent
            if not dest_dir.exists():
                logger.error(f"Destination directory does not exist: {dest_dir}")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error validating paths: {e}")
            return False

    def _generate_config(self, kalliste_tags: Dict[str, Any]) -> str:
        """Generate exiftool config content using StringIO."""
        config = StringIO()
        config.write("# Generated Kalliste XMP namespace configuration\n\n")
        config.write("%Image::ExifTool::UserDefined = (\n")
        config.write(" 'Image::ExifTool::XMP::Main' => {\n")
        config.write(" Kalliste => {\n")
        config.write(" SubDirectory => {\n")
        config.write(" TagTable => 'Image::ExifTool::UserDefined::Kalliste',\n")
        config.write(" },\n")
        config.write(" },\n")
        config.write(" },\n")
        config.write(");\n\n")
        config.write("%Image::ExifTool::UserDefined::Kalliste = (\n")
        config.write(" GROUPS => { 0 => 'XMP', 1 => 'XMP-Kalliste', 2 => 'Image' },\n")
        config.write(" NAMESPACE => { 'Kalliste' => 'http://kalliste.ai/1.0/' },\n")
        config.write(" WRITABLE => 'string',\n")
        # Add tag definitions based on tag type
        for tag_name, tag in kalliste_tags.items():
            if isinstance(tag, KallisteBagTag):
                config.write(f" '{tag_name}' => {{ List => 'Bag', Writable => 'string' }},\n")
            elif isinstance(tag, KallisteSeqTag):
                config.write(f" '{tag_name}' => {{ List => 'Seq', Writable => 'string' }},\n")
            elif isinstance(tag, KallisteAltTag):
                config.write(f" '{tag_name}' => {{ List => 'Alt', Writable => 'lang-alt' }},\n")
            elif isinstance(tag, KallisteRealTag):
                config.write(f" '{tag_name}' => {{ Writable => 'real' }},\n")
            elif isinstance(tag, KallisteIntegerTag):
                config.write(f" '{tag_name}' => {{ Writable => 'integer' }},\n")
            elif isinstance(tag, KallisteDateTag):
                config.write(f" '{tag_name}' => {{ Writable => 'date' }},\n")
            else:  # KallisteStringTag and any others default to string
                config.write(f" '{tag_name}' => {{ Writable => 'string' }},\n")
        config.write(");\n")
        config.write("1;\n")
        return config.getvalue()
        
    def _build_exiftool_command(self, kalliste_tags: Dict[str, Any], config_path: str) -> List[str]:
        """Build exiftool command with proper XMP tags."""
        # Start with basic command to copy all metadata
        cmd = [
            "exiftool",
            "-config", config_path,
            "-TagsFromFile", str(self.source_path),
            "-all:all",
        ]
        
        # Add each kalliste tag as XMP using the tag's own formatting
        for tag_name, tag in kalliste_tags.items():
            xmp_value = tag.to_xmp()
            cmd.extend([f"-XMP-Kalliste:{tag_name}={xmp_value}"])
        
        # Add destination and overwrite flag
        cmd.extend([str(self.dest_path), "-overwrite_original"])
        
        logger.debug(f"Built exiftool command: {' '.join(cmd)}")
        return cmd
        
    def _execute_command(self, cmd: List[str]) -> bool:
        """Execute exiftool command and handle results."""
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                logger.error(f"Exiftool error: {result.stderr}")
                return False
                
            # Log warnings but don't fail
            if result.stderr:
                logger.warning(f"Exiftool warnings: {result.stderr}")
                
            logger.debug(f"Exiftool stdout: {result.stdout}")
            logger.info(f"Successfully wrote XMP metadata to: {self.dest_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error executing exiftool command: {e}")
            return False