import requests
from bs4 import BeautifulSoup
from pdf2image import convert_from_path
from PIL import Image
import pytesseract

# Scrape a web page
def scrape_web_page(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    text = soup.get_text()
    return text.strip()

# Extract text and images from a PDF
def extract_text_and_images_from_pdf(pdf_path):
    # Extract text
    images = convert_from_path(pdf_path)
    text_blocks = []
    extracted_images = []
    
    for page_image in images:
        text_blocks.append(pytesseract.image_to_string(page_image))
        extracted_images.append(page_image)
    
    return text_blocks, extracted_images
