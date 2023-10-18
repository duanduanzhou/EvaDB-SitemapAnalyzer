# sitemap_analysis.py
import evadb
from url_parser import get_sitemap_data, chunk_sitemap_data, embed_data
import csv
import pandas as pd
import requests
from bs4 import BeautifulSoup
import numpy as np
import faiss

index = None
sitemap_urls = []

def initialize_faiss_index(dimension):
    # Initialize FAISS index
    global index
    index = faiss.IndexFlatL2(dimension)
    
def add_to_faiss_index(dt_chunk):
    # Add data embeddings to FAISS
    embeddings = embed_data(dt_chunk)
    index.add(embeddings)
    
    # Update the sitemap_urls list
    for data in dt_chunk:
        sitemap_urls.append(data['loc'])

def search_faiss_index(query_embedding, k=10):
   # Search FAISS for similar embeddings
    _, indices = index.search(query_embedding.reshape(1, -1), k)
    return indices[0]

def connect_to_evadb():
    # Connect to evadb
    connection = evadb.connect()
    if not connection:
        print("Connection to EvaDB failed.")
        exit()
    return connection.cursor()

def create_sitemap_table():
    # Create sitemap table in EvaDB
    conn = evadb.connect()
    cursor = conn.cursor()
    cursor.query("""
        CREATE TABLE IF NOT EXISTS sitemaps (
            id INTEGER UNIQUE,
            loc TEXT(255),
            lastmod TEXT(255),
            changefreq TEXT(255),
            priority FLOAT(2,4),
            embedding TEXT(255)
        );
    """).df()

def create_text_classifier():
    # Create a text classifer function using HuggingFace
    cursor = connect_to_evadb()
    cursor.query("""
        CREATE FUNCTION IF NOT EXISTS TextClassifier
        TYPE HuggingFace
        TASK "text-classification"
        MODEL "bert-base-uncased";
    """).df()

    
def create_sentiment_analysis_function():
    # Creates a sentiment analysis function in EvaDB using HuggingFace
    cursor = connect_to_evadb()
    cursor.query("""
        CREATE FUNCTION IF NOT EXISTS SentimentAnalysis
        TYPE HuggingFace
        TASK "text-classification"
        MODEL "nlptown/bert-base-multilingual-uncased-sentiment";
    """).df()
    
def fetch_metadata(url):
    # Fetches the metadata from a given URL
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        # Extract meta description and keywords
        description = soup.find('meta', attrs={'name': 'description'})
        keywords = soup.find('meta', attrs={'name': 'keywords'})

        meta_content = []
        if description and description.get('content'):
            meta_content.append(description.get('content'))
        if keywords and keywords.get('content'):
            meta_content.append(keywords.get('content'))

        # Clean up the meta content
        cleaned_content = ' '.join(meta_content).replace("''", "'")
        return cleaned_content

    except requests.RequestException:
        print(f"Failed to fetch metadata from {url}")
        return None

def get_sentiment(content):
    # Gets the sentiment for metadata
    cursor = connect_to_evadb()
    # remove contractions to prevent parsing error
    contraction_list = ["'re", "'ll", "'ve", "'s", "'d", "'m", "'t"]
    for elem in contraction_list:
        content = content.replace(elem, "")
        
    # Content escaping to prevent parsing error
    content_processed = content.replace("\\", "\\\\").replace("'", "''")

    # Ensure the content doesn't start or end with a single quote
    if content_processed.startswith("'") or content_processed.endswith("'"):
        content_processed = " " + content_processed.strip("'") + " "

    response = cursor.query(f"""
    SELECT SentimentAnalysis('{content_processed}');""").df()

    sentiment = response.iloc[0, 0] if not response.empty else None
    return sentiment

def store_embeddings_in_faiss(sitemap_data):
    # Store embeddings in the vector database
    all_embeddings = []
    for chunk in chunk_sitemap_data(sitemap_data):
        embeddings = embed_data(chunk)
        # print(f"Embeddings shape for current chunk: {np.array(embeddings).shape}")
        all_embeddings.append(embeddings)

def enhanced_search_using_embeddings(user_query, top_k=5):
    # Combine string match with embeddings to enhance the query
    initial_results = string_match_search(user_query)
    
    # Remove duplicates from initial results
    initial_results = list(dict.fromkeys(initial_results))

    # rerank the initial results using embeddings
    reranked_urls = rerank_results(user_query, index, initial_results, min(top_k, len(initial_results)))
    
    # Get the top_k URLs without duplicates
    unique_reranked_urls = list(dict.fromkeys(reranked_urls))
    return unique_reranked_urls

def rerank_results(user_query, faiss_db, initial_results, top_k=5):
    """Reranks search results using embeddings."""
    user_query_embedding = embed_data([user_query])[0]  # Directly embed the user query
    user_query_embedding = np.array([user_query_embedding], dtype='float32')
    # print(f"User Query Embedding Shape: {user_query_embedding.shape}")
    
    # Filter only include initial results
    filtered_sitemap_data = [entry for entry in sitemap_data if entry['loc'] in initial_results]
    
    # Create an index for these filtered results
    filtered_index = faiss.IndexFlatL2(user_query_embedding.shape[1])
    filtered_embeddings = embed_data([entry['loc'] for entry in filtered_sitemap_data])
    filtered_index.add(np.array(filtered_embeddings, dtype='float32'))
    
    # Search the filtered index
    _, indices = filtered_index.search(user_query_embedding, top_k)
    reranked_urls = [filtered_sitemap_data[idx]['loc'] for idx in indices[0]]
 
    return reranked_urls

def string_match_search(user_query):
    cursor = connect_to_evadb()
    
    # Classify the user query
    response = cursor.query(f"""
    SELECT TextClassifier('{user_query}');
    """).df()
    
    result = cursor.query(f"""
        SELECT loc
        FROM sitemaps
    """).df()
    
    rows = result[result['sitemaps.loc'].str.contains(user_query)]
    result_loc = rows['sitemaps.loc'].tolist()
    return result_loc

def search_using_text_classifier(user_query):
    cursor = connect_to_evadb()
    
    # Classify the user query
    response = cursor.query(f"""SELECT TextClassifier('{user_query}');""").df()
    
    # Assuming the classifier returns a column named 'category'
    category = response.iloc[0, 0] if not response.empty else None
    if not category:
        return None
    
    # Get embeddings for the user query
    query_embedding = embed_data([user_query])[0]  # Directly embed the user query
    
    # Use FAISS to find the most similar URLs
    similar_indices (query_embedding)
    
    # Retrieve the URLs based on the indices
    result_urls = [sitemap_data[idx]['loc'] for idx in similar_indices]
    
    return result_urls

def save_sitemap_data_to_csv(sitemap_data, filename="sitemap_data.csv"):
    with open(filename, mode="w", newline="", encoding="utf-8") as file:
        fieldnames = ["id", "loc", "lastmod", "changefreq", "priority", "embedding"]  # Added embedding
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        
        id_counter = 1  # initialize an id counter
        
        for chunk in chunk_sitemap_data(sitemap_data):
            embeddings = embed_data(chunk)
            for idx, data in enumerate(chunk):
                if data.get("loc"):
                    # Add the id field and embedding to each row
                    data["id"] = id_counter
                    data["embedding"] = embeddings[idx].tolist()
                    id_counter += 1
                    
                    # Exclude any field not in fieldnames before writing the row to CSV
                    filtered_data = {key: value for key, value in data.items() if key in fieldnames}
                    writer.writerow(filtered_data)

def load_csv(csv_filename, table_name):
    cursor = evadb.connect().cursor()
    cursor.query(f"""LOAD CSV '{csv_filename}' INTO {table_name};""").df()
    
def clear_sitemap_table():
    cursor = evadb.connect().cursor()
    cursor.query("DROP TABLE IF EXISTS sitemaps;").df()
    create_sitemap_table()

if __name__ == "__main__":
    # sample user query URL
    # sitemap_url = "https://xml-sitemaps.com/download/yoast.com-9d3a344a/sitemap.xml"
    print("\nWelcome! Enter a sitemap URL ending with 'sitemap.xml':")
    sitemap_url = input().strip()  # Get user input for the sitemap URL

    if not sitemap_url.endswith("sitemap.xml"):
        print("\nOops! The URL should end with sitemap.xml. Please check and try again.")
        exit()

    while True: 
        print("\nNext, provide a keyword for your query, or input 'quit' to exit:")
        user_query = input().strip()  # Get user input for the search query
        
        if user_query.lower() == 'quit':
            print("\nGoodbye!")
            break

        sitemap_data = get_sitemap_data(sitemap_url)
        # Save sitemap data to CSV
        save_sitemap_data_to_csv(sitemap_data)
        # Connect to EvaDB and set up tables and functions
        cursor = connect_to_evadb()
        
        # Check if the table exists 
        all_tables = cursor.query("SHOW TABLES;").df()
        if "sitemaps" not in all_tables.values:
            create_sitemap_table()

        create_text_classifier()
        create_sentiment_analysis_function()

        # Load the CSV data to EvaDB
        load_csv("sitemap_data.csv", "sitemaps")

        # Initialize FAISS index, add sitemap data for efficient similarity search
        sample_embedding = embed_data([sitemap_data[0]])  # take a sample to determine the dimension
        initialize_faiss_index(sample_embedding.shape[1])
        for elem in chunk_sitemap_data(sitemap_data):
            add_to_faiss_index(elem)

        # Use enhanced search with embeddings
        search_results = enhanced_search_using_embeddings(user_query, top_k=10)
        print(f"\nSearch results for '{user_query}':")

        for idx, url in enumerate(search_results, start=1):
            metadata = fetch_metadata(url)
            if metadata: # Get the metadata sentiment data
                sentiment = get_sentiment(metadata)
                if sentiment:
                    print(f"{idx}. {url}, Metadata Sentiment Analysis: {sentiment}")
                else:
                    print(f"{idx}. {url}, Metadata Sentiment Analysis: Not available")
            else:
                print(f"{idx}. {url}, Metadata: Not available")
