import chromadb
from chromadb.utils.data_loaders import ImageLoader

from chromadb.utils.embedding_functions.open_clip_embedding_function import OpenCLIPEmbeddingFunction
import os

def find_png_files(rootdir):
    png_files = []
    for dirpath, _, filenames in os.walk(rootdir):
        for filename in filenames:
            if filename.lower().endswith(".png"):
                png_files.append(os.path.join(dirpath, filename))
    return png_files


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
    name="kalliste2",
    embedding_function=embedding_function,
    data_loader=data_loader
    )

for file in png_files:
    collection.add(
        ids=file,
        uris=file
    )
    print(file)

