from rank_bm25 import BM25Okapi
import numpy as np
from sentence_transformers import SentenceTransformer
from pymilvus import Collection, connections
import os
from openai import OpenAI
import json

def hybrid_retrieval(query, collection):
    model = SentenceTransformer('./model/')  #all-MiniLM-L6-v2
    query_embedding = model.encode(query)


    search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
    milvus_results = collection.search(data=[query_embedding], anns_field="embeddings", param = search_params, limit=20,
                                       output_fields=["id", "chunkdata" ])

    milvus_scores = np.array([result.distance for result in milvus_results[0]])
    milvus_chunks = [result.entity.chunkdata for result in milvus_results[0]]

    print(milvus_scores)
    milvus_scores_updated = np.array([1-((i-min(milvus_scores))/(max(milvus_scores)-min(milvus_scores))) for i in milvus_scores])
    print(milvus_scores_updated)

    bm25chunks = [i.split(" ") for i in milvus_chunks]
    bm25 = BM25Okapi(bm25chunks)
    bm25_scores = bm25.get_scores(query.split(" "))

    print(bm25_scores)
    bm25_scores_updated = np.array([(i-min(bm25_scores))/(max(bm25_scores)-min(bm25_scores)) for i in bm25_scores])
    print(bm25_scores_updated)

    combined_scores = (bm25_scores_updated + milvus_scores_updated) / 2
    print(combined_scores)
    sorted_indices = np.argsort(combined_scores)[::-1]
    print(sorted_indices)

    sorted_results = [milvus_chunks[i] for i in sorted_indices[:10]]
    return sorted_results


def main():
    query = "What is Cuda ?"

    connections.connect("default", host="localhost", port="19530")

    collection = Collection("cuda_docs")

    results = hybrid_retrieval(query, collection)

    with open('retrieved_data.json', 'w') as f:
        json.dump(results, f)

    print("Retrieved and re-ranked data saved to retrieved_data.json")

    # os.environ["OPENAI_API_KEY"] = "sk- Your OPENAI Keyr"

    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY")
    )

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": f"You have been given a query {query}. Based on the given top 10 document chunks {results} return the answer.",
            }
        ],
        model="gpt-3.5-turbo",
    )

    print("Output Result to query :- ",query)
    print(chat_completion.choices[0].message.content)

if __name__ == "__main__":
    main()
