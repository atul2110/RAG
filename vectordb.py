import requests
from bs4 import BeautifulSoup
import networkx as nx
import re
import json
import lancedb
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from fastapi import FastAPI
from pydantic import BaseModel

def get_links(url, depth=1):
    G = nx.DiGraph()
    G.add_node(url, depth=0)
    visited = set([url])
    queue = [url]

    while queue:
        current_url = queue.pop(0)
        current_depth = G.nodes[current_url]['depth']
        print(current_url,current_depth)
        if current_depth < depth:
            try:
                response = requests.get(current_url)
                soup = BeautifulSoup(response.content, 'html.parser')
                for a_tag in soup.find_all('a', href=True):
                    link = a_tag['href']
                    if re.match(r'^https?://', link):
                        if link not in visited:
                            G.add_node(link, depth=current_depth + 1)
                            G.add_edge(current_url, link)
                            queue.append(link)
                            visited.add(link)
            except Exception as e:
                print(f"Failed to retrieve {current_url}: {e}")

    return G

def scrape_content(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        paragraphs = soup.find_all('p')
        content = ' '.join([para.get_text() for para in paragraphs])
        return content
    except Exception as e:
        print(f"Failed to scrape {url}: {e}")
        return ""

def chunk_data(data):
    model = SentenceTransformer("model")#'all-MiniLM-L6-v2')
    sentences = data.split(".")

    if len(sentences) < 5:
        sentences = data.split(',')

    print(f"Length of chuncked sentences : {len(sentences)}")

    if len(sentences) >= 5:
        embeddings = model.encode(sentences)

        num_clusters = 5  # Adjust this based on your needs
        clustering_model = KMeans(n_clusters=num_clusters)
        clustering_model.fit(embeddings)
        cluster_assignment = clustering_model.labels_

        clustered_data = [[] for i in range(num_clusters)]
        for sentence_id, cluster_id in enumerate(cluster_assignment):
            clustered_data[cluster_id-1].append(sentences[sentence_id])

        clustered_data_new = []
        for i in clustered_data:
            clustered_data_new.append(".".join(i))

        embed_data = []
        for i in clustered_data_new:
            emb = model.encode(i)
            embed_data.append(emb)

        return clustered_data_new, embed_data #clustering_model.cluster_centers_
    else:
        print("Empty data returning")
        return [],[]
    

def call(url):
    base_url = url
    web_graph = get_links(base_url)

    content_dict = {}

    for url in web_graph.nodes:
        content_dict[url] = scrape_content(url)

    with open('scraped_data.json', 'w') as f:
        json.dump(content_dict, f)

    all_chunks = []
    all_embeddings = []

    for url, content in list(content_dict.items()):
        print(f"URL during Chunking : {url}")
        chunks, embeddings = chunk_data(content)
        for c,e in zip(chunks,embeddings):
            chk = c
            chk.replace(" ","")
            chk.replace('\n',"")
            if len(chk) < 10 :
                continue
            all_chunks.append(c)
            all_embeddings.append(e)

    df = pd.DataFrame({"vector":all_embeddings,"docs":all_chunks})

    uri = "data/url-scrap"
    db = lancedb.connect(uri)

    db.create_table("url_rag",data=df, overwrite = True)

class Request(BaseModel):
    url: str

app = FastAPI()

@app.post("/create/db")
def create_db(req:Request):
    input_url = req.url
    call(input_url)
    return True
