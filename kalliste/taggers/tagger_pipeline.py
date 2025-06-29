"""Pipeline for managing and running multiple image taggers."""
from typing import Dict, List, Optional
from pathlib import Path
import logging
from PIL import Image

from .AbstractTagger import AbstractTagger
from .wd14_tagger import WD14Tagger
from .caption_tagger import CaptionTagger
from .orientation_tagger import OrientationTagger
from .NIMA_tagger import NIMATagger
from ..types import TagResult
from ..tag.kalliste_tag import KallisteRealTag, KallisteStringTag, KallisteBagTag, KallisteStructureTag
from ..region import Region

# TODO:LMLooking at this structure of the pipeline, it's totally un maintainable, too ... 
# like, I get that we couldn't pass the region down to each tagger ... 
# i think there was a reason for this .... 
# but now we have this long file that would just grow and grow if we added different taggers. 
# It feels a bit brutalistic. I don't understand why we aren't just creating the KallisteTags in the taggers themselves, 
# instead of passing this all back to the tagger pipeline. Like we take these simple results, put them into TagResults, 
# pass the tag results and an arbitrary array, then just make KallisteTags. 
# Why don't we make the KallisteTags in the taggers, pass them back to TaggerPipeline, 
# tagger pipeline just takes them and adds them to the region?
# 
# You're right - the current structure adds unnecessary complexity by:

# Converting tagger outputs to TagResults
# Passing them to pipeline
# Converting TagResults back to KallisteTags

# A simpler approach would be:

# Each tagger creates and returns its own KallisteTags
# Pipeline just collects and adds tags to the region
# Remove TagResults entirely since they're just an intermediate step



logger = logging.getLogger(__name__)

class TaggerPipeline:
    """Pipeline for managing and running multiple image taggers."""
    
    # Map of tagger name to tagger class
    TAGGER_CLASSES = {
        'wd14': WD14Tagger,
        'caption': CaptionTagger,
        'orientation': OrientationTagger,
        'nima': NIMATagger
    }
    
    def __init__(self, config: Dict):
        """Initialize tagger pipeline with configuration."""
        self.config = config
        self.taggers: Dict[str, AbstractTagger] = {}
        
    def _ensure_tagger_initialized(self, tagger_name: str) -> None:
        """Initialize a specific tagger if not already initialized."""
        if tagger_name not in self.taggers:
            logger.info(f"🆕 Tagger '{tagger_name}' not in cache, initializing...")
            
            if tagger_name not in self.TAGGER_CLASSES:
                raise ValueError(f"Unsupported tagger: {tagger_name}")
                
            try:
                # Get tagger-specific config if it exists
                tagger_config = self.config.get('tagger', {}).get(tagger_name, {})
                logger.debug(f"🔧 Initializing {tagger_name} with config: {tagger_config}")
                
                # Initialize tagger with config
                tagger_class = self.TAGGER_CLASSES[tagger_name]
                self.taggers[tagger_name] = tagger_class(config=tagger_config)
                logger.info(f"✅ Successfully initialized {tagger_name} tagger")
                
            except Exception as e:
                logger.error(f"Failed to initialize {tagger_name} tagger: {e}", exc_info=True)
                raise RuntimeError(f"Tagger initialization failed: {e}")
        else:
            logger.debug(f"♾️ Tagger '{tagger_name}' already initialized, reusing cached instance")

    def _process_orientation_results(self, region: Region, orientation_results: List[TagResult]) -> None:
        """Process orientation results and add them to region's kalliste_tags."""
        try:
            if orientation_results:
                logger.info("Processing orientation tags:")
                logger.info(f"  All orientations: {orientation_results}")
                
                # Get highest confidence orientation
                highest_conf = max(orientation_results, key=lambda x: x.confidence)
                orientation_tag = KallisteStringTag(
                    "KallisteOrientationTag",
                    highest_conf.label.lower()
                )
                region.add_tag(orientation_tag)
                
                # Save raw orientation data as bag
                orientation_data = [
                    {
                        "orientation": tag.label,
                        "confidence": tag.confidence
                    }
                    for tag in orientation_results
                ]
                raw_tag = KallisteStructureTag(
                    "KallisteOrientationDataRaw",
                    orientation_data
                )
                region.add_tag(raw_tag)
                
                logger.debug(f"Added orientation tags: {highest_conf.label} with raw data")
                
        except Exception as e:
            logger.error(f"Failed to process orientation results: {e}")
            raise RuntimeError(f"Orientation tag processing failed: {e}")

    def _process_wd14_results(self, region: Region, wd14_results: List[TagResult]) -> None:
        """Process WD14 results and add them to region's kalliste_tags."""
        try:
            if wd14_results:
                logger.info("Processing wd14 tags:")
                
                # Map numeric categories to meaningful names
                category_map = {
                    '0': 'Attributes',
                    '4': 'Character',
                    '9': 'Content'
                }
                
                # Group tags by category
                category_tags = {}
                all_tags = set()  # For storing all tags
                
                for tag in wd14_results:
                    # Add to all tags
                    all_tags.add(tag.label)
                    
                    # Group by category using friendly names
                    category_name = category_map.get(tag.category, 'Other')
                    if category_name not in category_tags:
                        category_tags[category_name] = set()
                    category_tags[category_name].add(tag.label)
                
                logger.info(f"  Raw wd14 tags by category: {category_tags}")
                
                # Add category-specific tags
                for category, tags in category_tags.items():
                    category_tag_name = f"KallisteWd14{category}"
                    region.add_tag(KallisteBagTag(category_tag_name, tags))

                # Add combined tags collection (for compatibility)
                region.add_tag(KallisteBagTag("KallisteWd14Tags", all_tags))
                
                # Save raw WD14 data with confidences as bag
                wd14_data = [
                    {
                        "tag": tag.label,
                        "confidence": tag.confidence,
                        "category": category_map.get(tag.category, 'Other')
                    }
                    for tag in wd14_results
                ]
                raw_wd14_tag = KallisteStructureTag(
                    "KallisteWd14TagsRaw",
                    wd14_data
                )
                region.add_tag(raw_wd14_tag)
                
                logger.debug(f"Added WD14 tags: {len(all_tags)} total tags across {len(category_tags)} categories")
                
        except Exception as e:
            logger.error(f"Failed to process WD14 results: {e}")
            raise RuntimeError(f"WD14 tag processing failed: {e}")

    def _process_caption_results(self, region: Region, caption_results: List[TagResult]) -> None:
        """Process caption results and add them to region's kalliste_tags."""
        try:
            if caption_results:
                logger.info("Processing caption:")
                caption = caption_results[0].label
                logger.info(f"  Caption: {caption}")
                caption_tag = KallisteStringTag("KallisteCaption", caption)
                region.add_tag(caption_tag)
                
                logger.debug(f"Added caption: {caption}")
                
        except Exception as e:
            logger.error(f"Failed to process caption results: {e}")
            raise RuntimeError(f"Caption tag processing failed: {e}")

    def _process_nima_results(self, region: Region, nima_results: Dict[str, List[TagResult]]) -> None:
        try:
            # Add normalized scores
            if 'technical_score' in nima_results:
                region.add_tag(KallisteRealTag(
                    "KallisteNimaScoreTechnical",
                    float(nima_results['technical_score'][0].label)
                ))
                
            if 'aesthetic_score' in nima_results:
                region.add_tag(KallisteRealTag(
                    "KallisteNimaScoreAesthetic",
                    float(nima_results['aesthetic_score'][0].label)
                ))
                
            # Add assessments
            if 'technical_assessment' in nima_results:
                region.add_tag(KallisteStringTag(
                    "KallisteNimaAssessmentTechnical",
                    nima_results['technical_assessment'][0].label
                ))
                
            if 'aesthetic_assessment' in nima_results:
                region.add_tag(KallisteStringTag(
                    "KallisteNimaAssessmentAesthetic",
                    nima_results['aesthetic_assessment'][0].label
                ))
                
            # Add calculated values
            if 'overall_calculations' in nima_results:
                for calc in nima_results['overall_calculations']:
                    if calc.category == 'calc_average':
                        region.add_tag(KallisteRealTag(
                            "KallisteNimaScoreCalcAverage",
                            calc.confidence
                        )
                    )
                        
            # Add overall assessment
            if 'overall_assessment' in nima_results:
                region.add_tag(KallisteStringTag(
                    "KallisteNimaAssessmentOverall",
                    nima_results['overall_assessment'][0].label
                ))

        except Exception as e:
            logger.error(f"Failed to process NIMA results: {e}")
            raise RuntimeError(f"NIMA tag processing failed: {e}")

    def _process_tagger_results(self, region: Region, tagger_name: str, results: Dict[str, List[TagResult]]) -> None:
        """Process results from a specific tagger and add them to region's kalliste_tags."""
        try:
            if tagger_name == 'orientation' and 'orientation' in results:
                self._process_orientation_results(region, results['orientation'])
            elif tagger_name == 'wd14':  # Process all WD14 categories
                # Combine all WD14 results into a single list
                all_wd14_results = []
                for category_results in results.values():
                    all_wd14_results.extend(category_results)
                self._process_wd14_results(region, all_wd14_results)
            elif tagger_name == 'caption' and 'caption' in results:
                self._process_caption_results(region, results['caption'])
            elif tagger_name == 'nima':
                self._process_nima_results(region, results)
                
        except Exception as e:
            logger.error(f"Failed to process {tagger_name} results: {e}")
            raise RuntimeError(f"Tag processing failed for {tagger_name}: {e}")
        
    async def tag_pillow_image(self, image: Image.Image, region_type: str, region: Optional[Region] = None) -> Dict[str, List[TagResult]]:
        """Run configured taggers for this region type using a PIL Image."""
        # Get list of taggers to run for this region type
        taggers_to_run = self.config.get(region_type)
        if not taggers_to_run:
            raise ValueError(f"No taggers configured for region type: {region_type}")
        
        results = {}
        try:
            # Initialize and run each configured tagger
            for tagger_name in taggers_to_run:
                if tagger_name in self.TAGGER_CLASSES:
                    # Ensure tagger is initialized
                    self._ensure_tagger_initialized(tagger_name)
                    
                    # Run tagger
                    logger.debug(f"Running {tagger_name} tagger")
                    tagger_results = await self.taggers[tagger_name].tag_pillow_image(image)
                    
                    # Log individual tagger results
                    for category, tags in tagger_results.items():
                        if tags:
                            tag_details = [
                                f"{tag.label}({tag.confidence:.2f})" 
                                for tag in tags
                            ]
                            logger.debug(f"{tagger_name} {category} results: {tag_details}")
                    
                    # Process results into kalliste_tags if we have a region
                    if region is not None:
                        self._process_tagger_results(region, tagger_name, tagger_results)
                    
                    results.update(tagger_results)
                else:
                    logger.warning(f"Unsupported tagger requested: {tagger_name}")
            
            # Log final combined results
            combined_results = []
            for category, tags in results.items():
                if tags:
                    if category == 'caption':
                        # Special formatting for captions
                        formatted = f"{category}: \"{tags[0].label}\""
                    else:
                        # Format other tags with confidence
                        formatted = f"{category}: " + ", ".join(
                            f"{tag.label}({tag.confidence:.2f})"
                            for tag in tags
                        )
                    combined_results.append(formatted)
                    
            if combined_results:
                logger.info("Tagger results:")
                for result in combined_results:
                    logger.info(f"  {result}")
            else:
                logger.warning("No tag results generated")
                    
            return results
            
        except Exception as e:
            logger.error(f"Tagging failed: {str(e)}", exc_info=True)
            raise RuntimeError(f"Tagging failed: {str(e)}") from e
            
    async def tag_image(self, image_path: Path, region_type: str) -> Dict[str, List[TagResult]]:
        """Run configured taggers for this region type using an image file."""
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
            
        with Image.open(image_path) as img:
            return await self.tag_pillow_image(img, region_type)