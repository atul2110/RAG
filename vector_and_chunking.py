from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import json
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection,utility
# from pymilvus import MilvusClient

def chunk_data(data):
    model = SentenceTransformer('./model/')#'all-MiniLM-L6-v2')
    # model.save('./model/')
    sentences = data.split(".")
    print(f"Length of chuncked sentences : {len(sentences)}")
    if len(sentences) >= 5:
        embeddings = model.encode(sentences)

        num_clusters = 5  # Adjust this based on your needs
        clustering_model = KMeans(n_clusters=num_clusters)
        clustering_model.fit(embeddings)
        cluster_assignment = clustering_model.labels_
        print(clustering_model.cluster_centers_)
        print(len(clustering_model.cluster_centers_))

        # clustered_data = {i: [] for i in range(num_clusters)}
        clustered_data = [[] for i in range(num_clusters)]
        for sentence_id, cluster_id in enumerate(cluster_assignment):
            clustered_data[cluster_id-1].append(sentences[sentence_id])

        clustered_data_new = []
        for i in clustered_data:
            clustered_data_new.append(".".join(i))

        return clustered_data_new, clustering_model.cluster_centers_   #embeddings
    else:
        print("Empty data returning")
        return [],[]


def create_vector_db(chunked_data, embeddingss, metadataa):
    connections.connect("default", host="localhost", port="19530")
    # client = MilvusClient("milvus_step.db")

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=len(embeddingss[0])),
        FieldSchema(name="chunkdata",dtype=DataType.VARCHAR,max_length=50000),
        FieldSchema(name="metadata", dtype=DataType.VARCHAR, max_length=500)
    ]
    # collection = client.insert(collection_name="cuda_docs", data=data)

    if utility.has_collection("cuda_docs"):
        collection = Collection("cuda_docs")
        collection.drop()

    schema = CollectionSchema(fields, "cuda_docs schema")
    # client.create_collection(collection_name="cuda_docs", schema=schemaa)
    collection = Collection("cuda_docs", schema)

    collection.insert([embeddingss, chunked_data, metadataa])
    collection.create_index(field_name="embeddings", index_params={"index_type": "IVF_FLAT", "metric_type": "L2", "params": {"nlist": 100}})
    collection.load()


def main():
    with open('scraped_data.json', 'r') as f:
        content_dict = json.load(f)

    all_chunks = [] #{}
    all_embeddings = []
    metadata = []
    print(len(content_dict.keys()))
    count = 0
    for url, content in list(content_dict.items()):
        count += 1
        print(f"URL during Chunking : {url}")
        chunks, embeddings = chunk_data(content)
        all_chunks.extend(chunks)  #update(chunks)
        all_embeddings.extend(embeddings)
        if len(embeddings) != 0:
            metadata.extend([url] * len(chunks))  # Store the URL as metadata for each chunk
        print(count,len(metadata))

    print("EMB META",len(all_embeddings),len(metadata),len(all_chunks))

    create_vector_db(all_chunks, all_embeddings, metadata)
    print("Data chunked and stored in Milvus")


if __name__ == "__main__":
    main()
