import asyncio
from chromadb import Client  # Assuming you're using ChromaDB
from chromadb.config import Settings  # Adjust based on your setup
from chromadb.utils.data_loaders import ImageLoader
from chromadb.utils.embedding_functions.open_clip_embedding_function import OpenCLIPEmbeddingFunction

import json
import asyncio
from typing import Dict, Any, List

import subprocess
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def run_exiftool(cmd):
    """Runs ExifTool and returns success flag, stdout, and stderr."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        success = result.returncode == 0
        return success, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

def extract_kalliste_metadata(file_path: str):
    """Extracts Kalliste-specific XMP metadata using ExifTool."""
    
    cmd = [
        'exiftool',
        '-XMP:KallisteAssessment',
        '-XMP:KallisteCaption',
        '-XMP:KallisteLrLabel',
        '-XMP:KallisteLrRating',
        '-XMP:KallisteLrTags',
        '-XMP:KallisteNimaAestheticAssessment',
        '-XMP:KallisteNimaCalcAverage',
        '-XMP:KallisteNimaOverallAssessment',
        '-XMP:KallisteNimaScoreAesthetic',
        '-XMP:KallisteNimaScoreTechnical',
        '-XMP:KallisteNimaTechnicalAssessment',
        '-XMP:KallisteWd14Content',
        '-XMP:KallisteWd14Tags',
        '-j',  # JSON output
        str(file_path)
    ]

    success, stdout, stderr = run_exiftool(cmd)

    if not success:
        print(f"ExifTool failed for {file_path}: {stderr}")
        return {}
    print(json.loads(stdout)[0])
    # Parse JSON response
    try:
        metadata = json.loads(stdout)[0]  # ExifTool returns a list
    except (IndexError, json.JSONDecodeError):
        print(f"Failed to parse metadata for {file_path}")
        return {}

    # Extract fields, handling missing data
    def get_value(key, default=""):
        return metadata.get(key, default)

    def get_list_value(key):
        """Ensures lists/sets are converted to comma-separated strings."""
        val = metadata.get(key, [])
        return ", ".join(val) if isinstance(val, list) else str(val)

    extracted_metadata = {
        "KallisteAssessment": get_value("KallisteAssessment"),
        "KallisteCaption": get_value("KallisteCaption"),
        "KallisteLrLabel": get_value("KallisteLrLabel"),
        "KallisteLrRating": get_value("KallisteLrRating"),
        # "KallisteNimaAestheticAssessment": get_value("KallisteNimaAestheticAssessment"),
        # "KallisteNimaCalcAverage": float(get_value("KallisteNimaCalcAverage", 0.0)),
        # "KallisteNimaOverallAssessment": get_value("KallisteNimaOverallAssessment"),
        # "KallisteNimaScoreAesthetic": float(get_value("KallisteNimaScoreAesthetic", 0.0)),
        # "KallisteNimaScoreTechnical": float(get_value("KallisteNimaScoreTechnical", 0.0)),
        # "KallisteNimaTechnicalAssessment": get_value("KallisteNimaTechnicalAssessment"),
        # "KallisteTags": get_list_value("KallisteLrTags") + ", " + get_list_value("KallisteWd14Tags"),
        # "KallisteWd14Content": get_list_value("KallisteWd14Content"),
    }
    print("Extracted metadata -------------------------------------")
    print(extracted_metadata)
    return extracted_metadata


def process_file_and_add_to_chroma(file_path: str, collection):
    """Extract metadata and add file entry to Chroma."""
    
    # Extract metadata using ExifTool
    metadata = extract_kalliste_metadata(file_path)
    print("Metadata------------------------------------")
    print(metadata)
    # Create a unique document ID (use filename or another identifier)
    document_id = file_path  # You might want a more unique identifier
    print(file_path)
    print("--------------")
    # Add to Chroma
    collection.add(
        ids=file_path,
        uris=file_path
    )
  


    print(f"Added {file_path} to Chroma with metadata.")

def main(directory_path: str):
    """Scans a directory for files and adds them to Chroma with metadata."""
    

    embedding_function = OpenCLIPEmbeddingFunction()
    data_loader = ImageLoader()

    chroma_client = Client(Settings(persist_directory="/Volumes/m01/kalliste_data/chromadb"))  # Adjust storage path
    collection = chroma_client.get_or_create_collection(name="kalliste_metadata",
                                                        embedding_function=embedding_function,
                                                        data_loader=data_loader)

    # Gather all png files in the directory
    import os
    files = [
    os.path.join(directory_path, f) 
    for f in os.listdir(directory_path) 
    if f.lower().endswith(".png") and os.path.isfile(os.path.join(directory_path, f))
    ]

    # Process files asynchronously
    for file in files:
        process_file_and_add_to_chroma(file, collection)

    print("Finished processing all files.")

# Run script (replace with actual directory)
if __name__ == "__main__":
    directory = "/Volumes/m01/kalliste_data/images/20191214_NaGiLux_LinesS4"
    main(directory)  # Call `main` directly without `asyncio.run
