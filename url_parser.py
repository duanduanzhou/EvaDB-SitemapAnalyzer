# url_parser.py
# fetch and parse the sitemap from a given URL

import xml.etree.ElementTree as ET  # for XML parser
import requests
from transformers import DistilBertTokenizer, DistilBertModel
import torch
import numpy as np

# parse XML content from sitemaps
def parse_sitemap(xml_content):
    root = ET.fromstring(xml_content) # get root elem from XML content 
    sitemap_data = [] # store 
    
    for url in root:  # loop over each URL element in XML
        url_data = {}
        for child in url:
            tag = child.tag.split("}")[-1]  # extract the tag name, remove namespace
            url_data[tag] = child.text  # store url data
        sitemap_data.append(url_data)
    return sitemap_data

def fetch_sitemap(url):
    response = requests.get(url, timeout=30)  # Set a 30-second timeout
    if response.status_code != 200:
        raise ValueError(f"Failed to fetch sitemap from {url}")

    content_type = response.headers.get("content-type", "")

    # Check if it's an XSD format
    if "xml" in content_type and url.endswith(".xsd"):
        print("This is an XSD sitemap format")
        # Handle XSD content
        return None
    return response.text

def get_sitemap_data(url):
    try:
        xml_content = fetch_sitemap(url)
        return parse_sitemap(xml_content)
    except requests.RequestException as e:
        print(f"Error fetching sitemap: {e}")
        return []

def chunk_sitemap_data(sitemap_data, chunk_size=5):
    # Break sitemap data into smaller chunks
    return [sitemap_data[i:i + chunk_size] for i in range(0, len(sitemap_data), chunk_size)]

# Initialize DistilBERT model and tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained('distilbert-base-uncased')

def embed_data(data_chunk):
    # convert data chunks into vectors using DistilBERT
    if not data_chunk:
        return np.array([]).reshape(0, 768)
    
    # if the data chunk is a list of dictionaries, extract urls
    # Otherwise use the list of strings directly
    data_to_embed = [entry['loc'] if isinstance(entry, dict) else entry for entry in data_chunk]
    
    # Tokenize and get embeddings
    inputs = tokenizer(data_to_embed, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    
    embeddings = outputs.last_hidden_state.mean(dim=1).numpy()
    return embeddings
