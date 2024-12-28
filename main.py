import streamlit as st
from app.scraper import scrape_web_page, extract_text_and_images_from_pdf
from app.handler import insert_embedding, search_embedding
from app.embeds import generate_text_embedding, generate_image_embedding
from app.util import save_uploaded_file
from transformers import GPT2Tokenizer, GPT2LMHeadModel

st.title("RAG Chatbot with Llama")

# Load the distilgpt2 model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
model = GPT2LMHeadModel.from_pretrained("distilgpt2")

def generate_response(query, context):
    input_text = f"Context: {context}\nQuery: {query}\nResponse:"
    inputs = tokenizer.encode(input_text, return_tensors="pt")
    outputs = model.generate(inputs, max_length=150, num_return_sequences=1)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.split("Response:")[-1].strip()

# Sidebar options
option = st.sidebar.selectbox("Choose an option", ["Upload PDF", "Scrape Website", "Ask Query"])

if option == "Upload PDF":
    st.header("Upload a PDF")
    uploaded_file = st.file_uploader("Upload your PDF", type="pdf")
    if uploaded_file:
        pdf_path = save_uploaded_file(uploaded_file)
        st.write("Processing PDF...")
        texts, images = extract_text_and_images_from_pdf(pdf_path)
        st.write(f"Extracted {len(texts)} text blocks and {len(images)} images.")
        for text in texts:
            embedding = generate_text_embedding(text)
            insert_embedding(embedding, metadata="PDF text", data_type="text")
        for image in images:
            embedding = generate_image_embedding(image)
            insert_embedding(embedding, metadata="PDF image", data_type="image")
        st.success("PDF content stored in Milvus.")

elif option == "Scrape Website":
    st.header("Scrape a Website")
    url = st.text_input("Enter the URL:")
    if st.button("Scrape"):
        content = scrape_web_page(url)
        st.text_area("Extracted Content", content, height=300)
        embedding = generate_text_embedding(content)
        insert_embedding(embedding, metadata="Web content", data_type="text")
        st.success("Web content stored in Milvus.")

elif option == "Ask Query":
    st.header("Ask the Chatbot")
    query = st.text_input("Enter your query:")
    if st.button("Submit"):
        query_embedding = generate_text_embedding(query)
        results = search_embedding(query_embedding, top_k=5)
        context = "\n".join([res.entity.get("metadata") for res in results if res is not None and res.entity is not None])
        response = generate_response(query, context)
        st.write("Context Used:")
        st.text_area("Context", context, height=200)
        st.write("Response:")
        st.text_area("Response", response, height=100)
        insert_embedding(generate_text_embedding(f"Query: {query}\nAnswer: {response}"), metadata="user_interaction", data_type="text")
        st.success("Query and response stored in Milvus.")
