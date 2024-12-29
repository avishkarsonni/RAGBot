from transformers import AutoTokenizer, AutoModel
import torch

# Load text embedding model
text_model_name = "sentence-transformers/all-MiniLM-L6-v2"
text_tokenizer = AutoTokenizer.from_pretrained(text_model_name)
text_model = AutoModel.from_pretrained(text_model_name)

def generate_text_embedding(text):
    inputs = text_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = text_model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().tolist()
    return embedding

# Image embedding placeholder (update as per your image embedding model)
def generate_image_embedding(image):
    # Replace this with actual image embedding logic
    return [0.0] * 768  # Dummy vector for demonstration
