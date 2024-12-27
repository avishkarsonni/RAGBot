from transformers import LlamaTokenizer, LlamaForCausalLM

def load_llama_model():
    tokenizer = LlamaTokenizer.from_pretrained("your-llama-model-path")
    model = LlamaForCausalLM.from_pretrained("your-llama-model-path")
    return model, tokenizer

llama_model, llama_tokenizer = load_llama_model()

def generate_response(query, context):
    """Generate a response using the Llama model."""
    input_text = f"Context: {context}\nQuery: {query}"
    inputs = llama_tokenizer(input_text, return_tensors="pt")
    outputs = llama_model.generate(**inputs, max_length=200, num_return_sequences=1)
    return llama_tokenizer.decode(outputs[0], skip_special_tokens=True)
