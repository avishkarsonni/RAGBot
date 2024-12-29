from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType

# Connect to Zilliz server
connections.connect(
    alias="default",
    uri="https://<your-zilliz-uri>",  # Replace with your Zilliz URI
    token="<your-token>",  # Replace with your Zilliz authentication token
)

# Define collection schema
def create_collection(collection_name="embeddings"):
    if collection_name not in [col.name for col in Collection.list()]:
        schema = CollectionSchema(
            fields=[
                FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=768),
                FieldSchema(name="metadata", dtype=DataType.JSON),
                FieldSchema(name="data_type", dtype=DataType.STRING),
            ],
            description="Stores embeddings with metadata and data type.",
        )
        collection = Collection(name=collection_name, schema=schema)
        collection.create_index("vector", {"index_type": "IVF_FLAT", "metric_type": "IP", "params": {"nlist": 128}})
        collection.load()
    else:
        collection = Collection(name=collection_name)
        collection.load()
    return collection

# Collection initialization
collection = create_collection()

# Insert embedding
def insert_embedding(vector, metadata, data_type):
    collection.insert([vector, metadata, data_type])

# Search embedding
def search_embedding(query_vector, top_k=5):
    search_params = {"metric_type": "IP", "params": {"nprobe": 10}}
    results = collection.search([query_vector], "vector", search_params, top_k=top_k, output_fields=["metadata", "data_type"])
    return results
