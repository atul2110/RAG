import requests
from bs4 import BeautifulSoup
import networkx as nx
import re
import json


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


def main():
    base_url = "https://docs.nvidia.com/cuda/"
    web_graph = get_links(base_url)
    print(web_graph)
    content_dict = {}

    for url in web_graph.nodes:
        print(url)
        content_dict[url] = scrape_content(url)

    with open('scraped_data.json', 'w') as f:
        json.dump(content_dict, f)

    print("Scraping completed and data saved to scraped_data.json")


if __name__ == "__main__":
    main()
