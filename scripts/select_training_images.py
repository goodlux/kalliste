"""Select diverse, high-quality images for training based on person and assessment."""
import sys
from pathlib import Path
import numpy as np
from sklearn.cluster import KMeans
from typing import List, Dict, Any
import logging
from dataclasses import dataclass
import os
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent))

from kalliste.db.kalliste_db import KallisteDB
from kalliste.db.chroma_db import ChromaDB

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ImageRecord:
    """Container for image data."""
    image_id: int
    file_path: str
    nima_calc_average: float
    lr_rating: int
    assessment: str

def get_candidate_images(db: KallisteDB, person_name: str, assessment: str = None) -> List[ImageRecord]:
    """Get all suitable images for a person with optional assessment filter."""
    with db._get_connection() as conn:
        cursor = conn.cursor()
        
        query = """
            SELECT image_id, file_path, nima_calc_average, lr_rating, assessment
            FROM image 
            WHERE person_name = ?
        """
        params = [person_name]
        
        if assessment:
            query += " AND assessment = ?"
            params.append(assessment)
            
        query += " ORDER BY nima_calc_average DESC NULLS LAST, lr_rating DESC NULLS LAST"
        
        cursor.execute(query, params)
        
        return [ImageRecord(
            image_id=row['image_id'],
            file_path=row['file_path'],
            nima_calc_average=row['nima_calc_average'] or 0.0,
            lr_rating=row['lr_rating'] or 0,
            assessment=row['assessment'] or ''
        ) for row in cursor.fetchall()]

def cluster_and_select(chroma_db: ChromaDB, image_records: List[ImageRecord], 
                      target_images: int) -> List[ImageRecord]:
    """Cluster images and select best representatives to reach target count."""
    if not image_records:
        return []
        
    # Get embeddings for all images
    image_ids = [str(record.image_id) for record in image_records]
    
    try:
        embeddings_result = chroma_db.collection.get(
            ids=image_ids,
            include=['embeddings']
        )
        embeddings = embeddings_result['embeddings']
    except Exception as e:
        logger.error(f"Error getting embeddings: {e}")
        return []
    
    # Calculate number of clusters
    # We use more clusters than target images to allow for filtering
    n_clusters = min(len(image_records), int(target_images * 1.5))
    
    # Perform clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(embeddings)
    
    # Select best representatives from each cluster
    selected_images = []
    for cluster_id in range(kmeans.n_clusters):
        # Get images in this cluster
        cluster_indices = np.where(clusters == cluster_id)[0]
        cluster_records = [image_records[i] for i in cluster_indices]
        
        # Sort by quality metrics
        sorted_records = sorted(
            cluster_records,
            key=lambda x: (x.nima_calc_average, x.lr_rating),
            reverse=True
        )
        
        # Take the best image from this cluster
        if sorted_records:
            selected_images.append(sorted_records[0])
    
    # Final sort and limit to target number
    final_selection = sorted(
        selected_images,
        key=lambda x: (x.nima_calc_average, x.lr_rating),
        reverse=True
    )[:target_images]
    
    return final_selection

def generate_html_report(selected_images: List[ImageRecord], person_name: str, 
                        target_images: int, assessment: str = None) -> str:
    """Generate HTML report of selected images."""
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Selected Images for {person_name}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .header {{ margin-bottom: 20px; }}
            .image-grid {{ 
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
                gap: 20px;
                margin-top: 20px;
            }}
            .image-card {{
                border: 1px solid #ddd;
                padding: 10px;
                border-radius: 5px;
            }}
            .image-card img {{
                max-width: 100%;
                height: auto;
            }}
            .metadata {{
                margin-top: 10px;
                font-size: 0.9em;
            }}
            .score {{
                font-weight: bold;
                color: #2a6496;
            }}
            .stats {{
                background: #f8f9fa;
                padding: 15px;
                border-radius: 5px;
                margin-bottom: 20px;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Selected Images for {person_name}</h1>
            <div class="stats">
                <p>Target Images: {target_images}</p>
                <p>Selected: {len(selected_images)}</p>
                {f'<p>Assessment Filter: {assessment}</p>' if assessment else ''}
                <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
        </div>
        <div class="image-grid">
    """
    
    for img in selected_images:
        # Convert absolute path to relative path for browser
        rel_path = os.path.relpath(img.file_path, '/Users/rob/repos/kalliste')
        
        html += f"""
            <div class="image-card">
                <a href="{rel_path}" target="_blank">
                    <img src="{rel_path}" alt="ID: {img.image_id}">
                </a>
                <div class="metadata">
                    <p>Image ID: {img.image_id}</p>
                    <p>NIMA Score: <span class="score">{img.nima_calc_average:.2f}</span></p>
                    <p>LR Rating: <span class="score">{img.lr_rating}</span></p>
                    <p>Assessment: {img.assessment}</p>
                    <p>Path: <a href="{rel_path}" target="_blank">{os.path.basename(img.file_path)}</a></p>
                </div>
            </div>
        """
    
    html += """
        </div>
    </body>
    </html>
    """
    
    return html

def main(person_name: str, target_images: int = 100, assessment: str = None):
    """Main function to select diverse, high-quality images."""
    db = KallisteDB()
    chroma_db = ChromaDB()
    
    # Get candidate images
    logger.info(f"Fetching images for {person_name}")
    image_records = get_candidate_images(db, person_name, assessment)
    
    if not image_records:
        logger.error(f"No images found for {person_name}")
        return
    
    logger.info(f"Found {len(image_records)} candidate images")
    
    # Perform clustering and selection
    logger.info(f"Clustering and selecting {target_images} representative images")
    selected_images = cluster_and_select(chroma_db, image_records, target_images)
    
    # Generate and save HTML report
    logger.info("Generating HTML report")
    html_content = generate_html_report(selected_images, person_name, target_images, assessment)
    
    # Create reports directory if it doesn't exist
    reports_dir = Path('/Users/rob/repos/kalliste/reports')
    reports_dir.mkdir(exist_ok=True)
    
    # Save HTML report
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = reports_dir / f'selected_images_{person_name.replace(" ", "_")}_{timestamp}.html'
    with open(report_path, 'w') as f:
        f.write(html_content)
    
    logger.info(f"Report saved to: {report_path}")
    
    # Output results to console as well
    logger.info(f"Selected {len(selected_images)} images:")
    for img in selected_images:
        logger.info(
            f"Image ID: {img.image_id}, "
            f"NIMA Score: {img.nima_calc_average:.2f}, "
            f"LR Rating: {img.lr_rating}, "
            f"Path: {img.file_path}"
        )
    
    return selected_images

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Select diverse, high-quality images for training')
    parser.add_argument('person_name', help='Name of the person to select images for')
    parser.add_argument('--target', type=int, default=100, help='Target number of images')
    parser.add_argument('--assessment', help='Filter by assessment value')
    
    args = parser.parse_args()
    main(args.person_name, args.target, args.assessment)
