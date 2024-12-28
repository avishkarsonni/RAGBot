from transformers import GPT2Tokenizer, GPT2LMHeadModel

def load_gpt2_model():
    tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
    model = GPT2LMHeadModel.from_pretrained("distilgpt2")
    return model, tokenizer

gpt2_model, gpt2_tokenizer = load_gpt2_model()

def generate_response(query, context):
    """Generate a response using the distilled GPT-2 model."""
    input_text = f"Context: {context}\nQuery: {query}"
    inputs = gpt2_tokenizer(input_text, return_tensors="pt")
    outputs = gpt2_model.generate(**inputs, max_length=200, num_return_sequences=1)
    return gpt2_tokenizer.decode(outputs[0], skip_special_tokens=True)
