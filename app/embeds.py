from sentence_transformers import SentenceTransformer
import numpy as np

text_model = SentenceTransformer("all-MiniLM-L6-v2")

def generate_text_embedding(text):
    return text_model.encode(text).tolist()

def generate_image_embedding(image):
    """Placeholder for image embedding (use CLIP or similar models)."""
    return np.random.rand(512).tolist()
