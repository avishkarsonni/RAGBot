import requests
from bs4 import BeautifulSoup
import fitz  # PyMuPDF

def scrape_web_page(url):
    """Scrape the main content from a web page."""
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    return soup.get_text()

def extract_text_and_images_from_pdf(pdf_path):
    """Extract text and images from a PDF."""
    pdf_document = fitz.open(pdf_path)
    texts, images = [], []
    for page in pdf_document:
        texts.append(page.get_text())
        for image_index, img in enumerate(page.get_images(full=True)):
            image = pdf_document.extract_image(img[0])
            images.append(image["image"])
    return texts, images
