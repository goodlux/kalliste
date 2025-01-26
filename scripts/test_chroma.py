import chromadb
from chromadb.utils.data_loaders import ImageLoader

import json
import subprocess
from chromadb.utils.embedding_functions.open_clip_embedding_function import OpenCLIPEmbeddingFunction
import os

def find_png_files(rootdir):
    png_files = []
    for dirpath, _, filenames in os.walk(rootdir):
        for filename in filenames:
            if filename.lower().endswith(".png"):
                png_files.append(os.path.join(dirpath, filename))
    return png_files

    # Extract fields, handling missing data
def get_value(key, default=""):
    return metadata.get(key, default)

def get_list_value(key):
    """Ensures lists/sets are converted to comma-separated strings."""
    val = metadata.get(key, [])
    return ", ".join(val) if isinstance(val, list) else str(val)

def run_exiftool(cmd):
    """Runs ExifTool and returns success flag, stdout, and stderr."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        success = result.returncode == 0
        return success, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

# Set your root directory here
rootdir = "/Volumes/m01/kalliste_data/images"

# Get the list of PNG files
png_files = find_png_files(rootdir)

# Print the results





client = chromadb.PersistentClient(
            path='/Volumes/m01/kalliste_data/chromadb'
        )

embedding_function = OpenCLIPEmbeddingFunction()

data_loader = ImageLoader()

collection = client.get_or_create_collection(
    name="kalliste_metadata",
    embedding_function=embedding_function,
    data_loader=data_loader
    )

for file in png_files:

    cmd = [
        'exiftool',
        '-XMP:KallisteAssessment',
        '-XMP:KallisteCaption',
        '-XMP:KallisteLrLabel',
        '-XMP:KallisteLrRating',
        # '-XMP:KallisteLrTags',
        # '-XMP:KallisteNimaAestheticAssessment',
        # '-XMP:KallisteNimaCalcAverage',
        # '-XMP:KallisteNimaOverallAssessment',
        # '-XMP:KallisteNimaScoreAesthetic',
        # '-XMP:KallisteNimaScoreTechnical',
        # '-XMP:KallisteNimaTechnicalAssessment',
        # '-XMP:KallisteWd14Content',
        # '-XMP:KallisteWd14Tags',
        '-j',  # JSON output
        str(file)
    ]

    success, stdout, stderr = run_exiftool(cmd)


    metadata = json.loads(stdout)[0]  # ExifTool returns a list



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
    collection.add(
        ids=file,
        metadatas=extracted_metadata,
        uris=file
    )

    print(file)

