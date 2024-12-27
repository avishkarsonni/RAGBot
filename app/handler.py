from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility

# Connect to Milvus
connections.connect(
    alias="default",
    uri="your_milvus_cloud_uri",  # Replace with your Milvus Cloud URI
    secure=True,
    token="your_api_key"          # Replace with your API key
)

# Define schema and create collection if it doesn't exist
collection_name = "pdf_context"
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=512),
    FieldSchema(name="metadata", dtype=DataType.VARCHAR, max_length=255),
    FieldSchema(name="type", dtype=DataType.VARCHAR, max_length=50)  # 'text' or 'image'
]
schema = CollectionSchema(fields, description="Schema for RAG context embeddings")

if not utility.has_collection(collection_name):
    collection = Collection(name=collection_name, schema=schema)

def insert_embedding(embedding, metadata, data_type):
    """Insert embeddings into Milvus."""
    collection.insert([
        [embedding],
        [metadata],
        [data_type]
    ])

def search_embedding(query_embedding, top_k=5):
    """Search embeddings in Milvus."""
    collection.load()
    results = collection.search(
        data=[query_embedding],
        anns_field="embedding",
        param={"metric_type": "IP", "params": {"ef": 50}},
        limit=top_k
    )
    return results
