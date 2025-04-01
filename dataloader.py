import requests
from bs4 import BeautifulSoup
import json
import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
import time
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
from urllib.parse import urljoin, urlparse


# Replace with your website URL
BASE_URL = "https://certifymyhealth.com"
visited_urls = set()  # Track visited pages
all_texts = []  # Store extracted text

# Function to scrape a webpage
def scrape_page(url):
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")

            # Extract text from paragraphs
            texts = [p.get_text().strip() for p in soup.find_all("p")]
            full_text = " ".join(texts)

            return full_text, soup
    except requests.RequestException as e:
        print(f"Error fetching {url}: {e}")
    return "", None

# Function to find all internal links
def get_internal_links(soup, base_url):
    links = set()
    for link in soup.find_all("a", href=True):
        href = link["href"]
        full_url = urljoin(base_url, href)
        
        # Ensure link is within the same domain
        if urlparse(full_url).netloc == urlparse(BASE_URL).netloc and full_url not in visited_urls:
            links.add(full_url)
    return links

# Recursive function to crawl and scrape the whole website
def crawl_website(url):
    if url in visited_urls:
        return
    print(f"Scraping: {url}")

    visited_urls.add(url)
    text, soup = scrape_page(url)
    
    if text:
        all_texts.append({"url": url, "content": text})
    
    if soup:
        links = get_internal_links(soup, url)
        for link in links:
            time.sleep(1)  # Be polite, avoid hitting the server too fast
            crawl_website(link)

# Start crawling
crawl_website(BASE_URL)

# Save scraped data to a JSON file
with open("website_data.json", "w", encoding="utf-8") as file:
    json.dump(all_texts, file, indent=4)

print("✅ Website crawling complete. Data saved in 'website_data.json'")

# Load the data for embeddings
documents = [Document(page_content=item["content"]) for item in all_texts]

# Initialize a Hugging Face embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create FAISS index and store embeddings
db = FAISS.from_documents(documents, embedding_model)
db.save_local("faiss_index")

print("✅ Embeddings generated and saved in FAISS database.")
