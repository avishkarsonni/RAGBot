import os
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Read connection parameters from environment variables
milvus_uri = os.getenv("URI")
milvus_api_key = os.getenv("API_KEY")
collection_name = "pdf_context"

# Establish connection to Milvus
try:
    print("Connecting to Milvus...")
    connections.connect(
        alias="default",
        uri=milvus_uri,
        secure=True,
        token=milvus_api_key
    )
    print("Connected to Milvus successfully!")
except Exception as e:
    print(f"Failed to connect to Milvus: {e}")
    exit(1)

# Define schema for the collection
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=512),
    FieldSchema(name="metadata", dtype=DataType.VARCHAR, max_length=255),
    FieldSchema(name="type", dtype=DataType.VARCHAR, max_length=50)  # 'text' or 'image'
]
schema = CollectionSchema(fields, description="Schema for RAG context embeddings")

# Check if the collection exists
if not utility.has_collection(collection_name):
    print(f"Collection '{collection_name}' does not exist. Creating it...")
    collection = Collection(name=collection_name, schema=schema)
    print(f"Collection '{collection_name}' created successfully!")
else:
    print(f"Collection '{collection_name}' already exists. Loading it...")
    collection = Collection(name=collection_name)

# Ensure the index is created before loading the collection
if not collection.has_index():
    print("Creating index for the collection...")
    collection.create_index(
        field_name="embedding",
        index_params={"index_type": "IVF_FLAT", "metric_type": "IP", "params": {"nlist": 128}}
    )
    print("Index created successfully!")

# Load the collection for search
try:
    print("Loading the collection...")
    collection.load()
    print(f"Collection '{collection_name}' loaded successfully!")
except Exception as e:
    print(f"Failed to load the collection: {e}")
    exit(1)

def insert_embedding(embedding, metadata, data_type):
    """Insert embeddings into Milvus."""
    try:
        collection.insert([
            [embedding],
            [metadata],
            [data_type]
        ])
        print("Embedding inserted successfully!")
    except Exception as e:
        print(f"Error inserting embedding: {e}")

def search_embedding(query_embedding, top_k=5):
    """Search embeddings in Milvus."""
    try:
        collection.load()
        results = collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param={"metric_type": "IP", "params": {"ef": 50}},
            limit=top_k
        )
        return results
    except Exception as e:
        print(f"Error searching embeddings: {e}")
        return None


