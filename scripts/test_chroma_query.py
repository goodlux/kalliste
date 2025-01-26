from xml.etree.ElementInclude import include
import chromadb
from chromadb.utils.data_loaders import ImageLoader

from chromadb.utils.embedding_functions.open_clip_embedding_function import OpenCLIPEmbeddingFunction
import os

client = chromadb.HttpClient(
            host='localhost', port=8000
        )

data_loader = ImageLoader()
embedding_function = OpenCLIPEmbeddingFunction()
# Print the results
collection = client.get_or_create_collection(
    name="kalliste_metadata",
    embedding_function=embedding_function,
    data_loader=data_loader,

    )

results = collection.query(
    
    includes=['metadata']
)

# results=collection.query(
#     where={"KallisteLrRating": "1_star"},
#     n_results=100,
#     )

print(results)
