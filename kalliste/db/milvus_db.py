from pymilvus import MilvusClient
from typing import Dict, Any, List
from datetime import datetime
from ..tag.kalliste_tag import (
    KallisteBaseTag,
    KallisteStringTag,
    KallisteIntegerTag,
    KallisteRealTag,
    KallisteDateTag
)
from .embedding_generator import EmbeddingGenerator
import logging

logger = logging.getLogger(__name__)

class MilvusDB:

    def __init__(self, uri: str = "http://localhost:19530"):
        self.client = MilvusClient(uri=uri)
        self.collection_name = "kalliste_images"

    def _map_kalliste_tags_to_schema(self, kalliste_tags: Dict[str, KallisteBaseTag]) -> Dict[str, Any]:
        """
        Maps KallisteTags to Milvus schema fields, handling missing tags gracefully.
        All fields are optional except image_file_path which is handled separately.
        """
        # Define field types for proper type conversion
        field_types = {
            "photoshoot": "varchar",
            "photoshoot_date": "varchar",
            "photoshoot_location": "varchar",
            "person_name": "varchar",
            "source_type": "varchar",
            "lr_rating": "int64",
            "lr_label": "varchar",
            "image_date": "varchar",
            "region_type": "varchar",
            "nima_score_technical": "double",
            "nima_score_aesthetic": "double",
            "nima_score_calc_average": "double",
            "nima_assessment_technical": "varchar",
            "nima_assessment_aesthetic": "varchar",
            "nima_assessment_overall": "varchar",
            "kalliste_assessment": "varchar",
            "record_creation_date": "varchar"
        }
        
        # Initialize all fields with empty strings for varchar and 0 for numeric types
        schema_data = {}
        for field, field_type in field_types.items():
            if field_type == "varchar":
                schema_data[field] = ""
            elif field_type == "int64":
                schema_data[field] = 0
            elif field_type == "double":
                schema_data[field] = 0.0
                
        # Set record creation date
        schema_data["record_creation_date"] = datetime.now().isoformat()
        
        # Process each tag if it exists
        for tag_name, tag in kalliste_tags.items():
            if isinstance(tag, KallisteBaseTag):
                # Extract the field name from the tag name by removing "Kalliste" prefix
                # e.g., "KallistePhotoshoot" -> "photoshoot"
                field_name = tag_name.replace("Kalliste", "").lower()
                
                # Only process if field exists in our schema
                if field_name in schema_data:
                    # Get the expected type for this field
                    field_type = field_types.get(field_name)
                    
                    if tag.value is not None:
                        if field_type == "varchar":
                            schema_data[field_name] = str(tag.value)
                        elif field_type == "int64":
                            try:
                                schema_data[field_name] = int(tag.value)
                            except (ValueError, TypeError):
                                schema_data[field_name] = 0
                        elif field_type == "double":
                            try:
                                schema_data[field_name] = float(tag.value)
                            except (ValueError, TypeError):
                                schema_data[field_name] = 0.0
        
        return schema_data

    def insert(self, image_path: str, kalliste_tags: Dict[str, KallisteBaseTag]) -> bool:
        """
        Insert a single image with KallisteTags into Milvus.
        Now includes both DINOv2 and OpenCLIP embeddings.
        """
        try:

            # Generate embeddings
            # dinov2_embedding = EmbeddingGenerator.generate_dinov2_embedding(image_path)
            openclip_embedding = EmbeddingGenerator.generate_openclip_embedding(image_path)


            # Initialize data dictionary with empty/zero values
            data = {
                # Required fields
                "image_file_path": str(image_path),
                # "dinov2_vector": dinov2_embedding,  # For image similarity/diversity
                "openclip_vector": openclip_embedding,  # For text-image search
                
                # Initialize optional varchar fields with empty strings
                "photoshoot": "",
                "photoshoot_date": "",
                "photoshoot_location": "",
                "person_name": "",
                "source_type": "",
                "lr_label": "",
                "image_date": "",
                "region_type": "",
                "nima_assessment_technical": "",
                "nima_assessment_aesthetic": "",
                "nima_assessment_overall": "",
                "kalliste_assessment": "",
                
                # Initialize optional numeric fields with zeros
                "lr_rating": 0,
                "nima_score_technical": 0.0,
                "nima_score_aesthetic": 0.0,
                "nima_score_calc_average": 0.0,
                
                # Timestamp field
                "record_creation_date": datetime.now().isoformat(),
                
                # All tags combined field
                "all_tags": ""
            }
            
            # Explicit mapping of each possible tag
            if "KallistePhotoshoot" in kalliste_tags:
                data["photoshoot"] = str(kalliste_tags["KallistePhotoshoot"].value or "")
                
            if "KallistePhotoshootDate" in kalliste_tags:
                data["photoshoot_date"] = str(kalliste_tags["KallistePhotoshootDate"].value or "")
                
            if "KallistePhotoshootLocation" in kalliste_tags:
                data["photoshoot_location"] = str(kalliste_tags["KallistePhotoshootLocation"].value or "")
                
            if "KallistePersonName" in kalliste_tags:
                data["person_name"] = str(kalliste_tags["KallistePersonName"].value or "")
                
            if "KallisteSourceType" in kalliste_tags:
                data["source_type"] = str(kalliste_tags["KallisteSourceType"].value or "")
                
            if "KallisteLrRating" in kalliste_tags:
                try:
                    data["lr_rating"] = int(kalliste_tags["KallisteLrRating"].value or 0)
                except (ValueError, TypeError):
                    data["lr_rating"] = 0
                    
            if "KallisteLrLabel" in kalliste_tags:
                data["lr_label"] = str(kalliste_tags["KallisteLrLabel"].value or "")
                
            if "KallisteImageDate" in kalliste_tags:
                data["image_date"] = str(kalliste_tags["KallisteImageDate"].value or "")
                
            if "KallisteRegionType" in kalliste_tags:
                data["region_type"] = str(kalliste_tags["KallisteRegionType"].value or "")
                
            if "KallisteNimaScoreTechnical" in kalliste_tags:
                try:
                    data["nima_score_technical"] = float(kalliste_tags["KallisteNimaScoreTechnical"].value or 0.0)
                except (ValueError, TypeError):
                    data["nima_score_technical"] = 0.0
                    
            if "KallisteNimaScoreAesthetic" in kalliste_tags:
                try:
                    data["nima_score_aesthetic"] = float(kalliste_tags["KallisteNimaScoreAesthetic"].value or 0.0)
                except (ValueError, TypeError):
                    data["nima_score_aesthetic"] = 0.0
                    
            if "KallisteNimaScoreCalcAverage" in kalliste_tags:
                try:
                    data["nima_score_calc_average"] = float(kalliste_tags["KallisteNimaScoreCalcAverage"].value or 0.0)
                except (ValueError, TypeError):
                    data["nima_score_calc_average"] = 0.0
                    
            if "KallisteNimaAssessmentTechnical" in kalliste_tags:
                data["nima_assessment_technical"] = str(kalliste_tags["KallisteNimaAssessmentTechnical"].value or "")
                
            if "KallisteNimaAssessmentAesthetic" in kalliste_tags:
                data["nima_assessment_aesthetic"] = str(kalliste_tags["KallisteNimaAssessmentAesthetic"].value or "")
                
            if "KallisteNimaAssessmentOverall" in kalliste_tags:
                data["nima_assessment_overall"] = str(kalliste_tags["KallisteNimaAssessmentOverall"].value or "")
                
            if "KallisteAssessment" in kalliste_tags:
                data["kalliste_assessment"] = str(kalliste_tags["KallisteAssessment"].value or "")
                
            # Collect all tags from the three main tag collections
            all_tags = []
            
            # Add LR tags if available
            if "KallisteLrTags" in kalliste_tags and hasattr(kalliste_tags["KallisteLrTags"], "value"):
                if isinstance(kalliste_tags["KallisteLrTags"].value, set):
                    all_tags.extend(kalliste_tags["KallisteLrTags"].value)
                
            # Add WD14 tags if available
            if "KallisteWd14Tags" in kalliste_tags and hasattr(kalliste_tags["KallisteWd14Tags"], "value"):
                if isinstance(kalliste_tags["KallisteWd14Tags"].value, set):
                    all_tags.extend(kalliste_tags["KallisteWd14Tags"].value)
                
            # Add regular Kalliste tags if available
            if "KallisteTags" in kalliste_tags and hasattr(kalliste_tags["KallisteTags"], "value"):
                if isinstance(kalliste_tags["KallisteTags"].value, set):
                    all_tags.extend(kalliste_tags["KallisteTags"].value)
            
            # Join tags with spaces for better text search
            if all_tags:
                data["all_tags"] = " ".join(str(tag) for tag in all_tags)
            
            logger.debug(f"Inserting into Milvus with data: {data}")
            
            # Insert the data
            result = self.client.insert(
                collection_name=self.collection_name,
                data=[data]
            )
            
            success = result["insert_count"] == 1
            if success:
                logger.info(f"Successfully inserted image into Milvus: {image_path}")
            else:
                logger.error(f"Failed to insert image into Milvus: {image_path}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error inserting into Milvus: {e}")
            return False
    
    def query(self, filter_expr: str, limit: int = None, output_fields: List[str] = None) -> List[Dict[str, Any]]:
        """
        Query images using Milvus filter expressions.
        
        Args:
            filter_expr: Milvus filter expression (e.g., "person_name == 'ArMcLux' && nima_assessment_overall == 'acceptable'")
            limit: Maximum number of results to return (None = no limit)
            output_fields: List of fields to return, if None returns all fields
            
        Returns:
            List of dictionaries containing the query results
        """
        try:
            # Default output fields if none provided
            if output_fields is None:
                output_fields = ["id", "image_file_path", "nima_score_calc_average", "lr_rating"]
            
            # Prepare query parameters
            query_params = {
                "collection_name": self.collection_name,
                "filter": filter_expr,
                "output_fields": output_fields
            }
            
            # Add limit if specified (otherwise Milvus will return all matches)
            if limit is not None:
                query_params["limit"] = limit
                
            # Execute the query
            results = self.client.query(**query_params)
            
            logger.info(f"Query returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Error querying Milvus: {e}")
            return []
    
    def export_images(self, query_results: List[Dict[str, Any]], destination_folder: str, 
                      create_folder: bool = True) -> List[str]:
        """
        Export images from query results to a destination folder.
        Also copies associated sidecar text files with the same base name.
        
        Args:
            query_results: Results from a query() call
            destination_folder: Folder to copy images to
            create_folder: Whether to create the destination folder if it doesn't exist
            
        Returns:
            List of paths to successfully copied images
        """
        from pathlib import Path
        import shutil
        import os
        from ..config import KALLISTE_DATA_DIR
        
        # Ensure destination folder exists
        dest_path = Path(destination_folder)
        if create_folder:
            dest_path.mkdir(parents=True, exist_ok=True)
        elif not dest_path.exists():
            logger.error(f"Destination folder does not exist: {destination_folder}")
            return []
        
        copied_files = []
        failed_files = []
        
        # Process each image
        for result in query_results:
            if "image_file_path" not in result:
                logger.warning("Query result missing image_file_path field")
                continue
                
            # Get source file paths
            image_path = Path(result["image_file_path"])
            txt_path = image_path.with_suffix(".txt")
            
            # Get destination file paths
            dest_image_path = dest_path / image_path.name
            dest_txt_path = dest_path / txt_path.name
            
            try:
                # Copy image file
                if image_path.exists():
                    shutil.copy2(image_path, dest_image_path)
                    copied_files.append(str(dest_image_path))
                else:
                    logger.warning(f"Image file not found: {image_path}")
                    failed_files.append(str(image_path))
                    continue
                
                # Copy text file if it exists
                if txt_path.exists():
                    shutil.copy2(txt_path, dest_txt_path)
                    logger.debug(f"Copied sidecar file: {txt_path}")
                
            except Exception as e:
                logger.error(f"Error copying file {image_path}: {e}")
                failed_files.append(str(image_path))
        
        logger.info(f"Successfully copied {len(copied_files)} files to {destination_folder}")
        if failed_files:
            logger.warning(f"Failed to copy {len(failed_files)} files")
            
        return copied_files
            
    def __del__(self):
        """Ensure connection is closed when object is destroyed"""
        if hasattr(self, 'client'):
            self.client.close()