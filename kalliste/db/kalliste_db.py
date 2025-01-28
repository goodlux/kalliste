"""Database operations for Kalliste."""
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime

from ..config import KALLISTE_DB_PATH

logger = logging.getLogger(__name__)

class KallisteDB:
    def __init__(self):
        self.db_path = KALLISTE_DB_PATH
        
    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection with row factory."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
        
    def add_image(self, file_path: str, kalliste_tags: Dict[str, Any]) -> int:
        """Add image record and associated tags. Returns image_id."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                # Extract values correctly from kalliste_tags
                image_data = {
                    'file_path': file_path,
                    'photoshoot': kalliste_tags['KallistePhotoshoot'].value if 'KallistePhotoshoot' in kalliste_tags else '',
                    'photoshoot_date': kalliste_tags['KallistePhotoshootDate'].value if 'KallistePhotoshootDate' in kalliste_tags else None,
                    'photoshoot_location': kalliste_tags['KallistePhotoshootLocation'].value if 'KallistePhotoshootLocation' in kalliste_tags else '',
                    'person_name': kalliste_tags['KallistePersonName'].value if 'KallistePersonName' in kalliste_tags else '',
                    'source_type': kalliste_tags['KallisteSourceType'].value if 'KallisteSourceType' in kalliste_tags else '',
                    'lr_rating': kalliste_tags['KallisteLRRating'].value if 'KallisteLRRating' in kalliste_tags else None,
                    'creation_date': datetime.now().isoformat(),
                    'region_type': kalliste_tags['KallisteRegionType'].value if 'KallisteRegionType' in kalliste_tags else '',
                    'nima_technical_score': kalliste_tags['KallisteNimaTechnicalScore'].value if 'KallisteNimaTechnicalScore' in kalliste_tags else None,
                    'nima_assessment_technical': kalliste_tags['KallisteNimaAssessmentTechnical'].value if 'KallisteNimaAssessmentTechnical' in kalliste_tags else '',
                    'nima_aesthetic_score': kalliste_tags['KallisteNimaAestheticScore'].value if 'KallisteNimaAestheticScore' in kalliste_tags else None,
                    'nima_assessment_aesthetic': kalliste_tags['KallisteNimaAssessmentAesthetic'].value if 'KallisteNimaAssessmentAesthetic' in kalliste_tags else '',
                    'nima_overall_score': kalliste_tags['KallisteNimaOverallScore'].value if 'KallisteNimaOverallScore' in kalliste_tags else None,
                    'nima_assessment_overall': kalliste_tags['KallisteNimaAssessmentOverall'].value if 'KallisteNimaAssessmentOverall' in kalliste_tags else '',
                    'nima_calc_average': kalliste_tags['KallisteNimaCalcAverage'].value if 'KallisteNimaCalcAverage' in kalliste_tags else None,
                    'assessment': kalliste_tags['KallisteAssessment'].value if 'KallisteAssessment' in kalliste_tags else ''
                }

                # Insert image record
                cursor.execute("""
                    INSERT INTO image (
                        file_path, photoshoot, photoshoot_date, photoshoot_location,
                        person_name, source_type, lr_rating, creation_date,
                        region_type, nima_technical_score, nima_technical_assessment,
                        nima_aesthetic_score, nima_aesthetic_assessment,
                        nima_overall_score, nima_overall_assessment,
                        nima_calc_average_score, kalliste_assessment
                    ) VALUES (
                        :file_path, :photoshoot, :photoshoot_date, :photoshoot_location,
                        :person_name, :source_type, :lr_rating, :creation_date,
                        :region_type, :nima_technical_score, :nima_assessment_technical,
                        :nima_aesthetic_score, :nima_assessment_aesthetic,
                        :nima_overall_score, :nima_assessment_overall,
                        :nima_calc_average, :assessment
                    )
                """, image_data)

                image_id = cursor.lastrowid

                # Deduplicate all tags before inserting
                all_tags = set()
                for key in ('KallisteTags', 'KallisteLrTags', 'KallisteWd14Tags'):
                    if key in kalliste_tags and kalliste_tags[key].value:
                        all_tags.update(kalliste_tags[key].value)

                # Insert tags
                self._add_tags_to_image(cursor, image_id, all_tags)

                conn.commit()
                return image_id

        except Exception as e:
            logger.error(f"Error adding image to database: {e}")
            raise

            
    def _add_tags_to_image(self, cursor: sqlite3.Cursor, image_id: int, tags: set):
        """Helper method to add tags and create image-tag associations."""
        for tag in tags:
            # Add tag if it doesn't exist
            cursor.execute(
                "INSERT OR IGNORE INTO tag (tag) VALUES (?)",
                (tag,)
            )
            
            # Get tag_id
            cursor.execute(
                "SELECT tag_id FROM tag WHERE tag = ?",
                (tag,)
            )
            tag_id = cursor.fetchone()[0]
            
            # Create image-tag association
            cursor.execute("""
                INSERT INTO image_tag (image_id, tag_id)
                VALUES (?, ?)
            """, (image_id, tag_id))