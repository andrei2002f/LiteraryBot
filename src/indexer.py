from src.utils import parse_bulk_json, clean_text, extract_keywords
from src.embedder import embed
from elasticsearch import Elasticsearch, helpers
from src.config import ES_HOST, VECTOR_DIM, INDEX_ALL

es = Elasticsearch(ES_HOST)

def create_unified_index(name):
    if es.indices.exists(index=name):
        es.indices.delete(index=name)
    mapping = {
        "mappings": {
            "properties": {
                "type": {"type": "keyword"},  # "author" or "publication"
                "name": {"type": "text"},
                "description": {"type": "text"},
                "professions": {"type": "text"},  # New field for authors
                "writings": {"type": "text"},  # New field for authors
                "category": {"type": "text"},  # New field for publications
                "vector": {
                    "type": "dense_vector",
                    "dims": VECTOR_DIM,
                    "index": True,
                    "similarity": "cosine"
                }
            }
        }
    }
    es.indices.create(index=name, body=mapping)

def index_unified_documents(authors_json, publications_json, index_name):
    # Index authors
    author_docs = parse_bulk_json(authors_json)
    author_texts = [clean_text(doc, "description") for doc in author_docs]
    author_vectors = embed(author_texts)
    author_actions = []
    for doc, vec, text in zip(author_docs, author_vectors, author_texts):
        keywords = extract_keywords(text, language="romanian")  # Dynamically extract keywords
        author_actions.append({
            "_index": index_name,
            "_id": f"author_{doc.get('search_name', doc.get('name'))}",
            "_source": {
                "type": "author",
                "name": doc.get("name").replace(",", ""),  # Escape commas
                "description": text,
                "keywords": keywords,  # Add keywords field
                "professions": doc.get("professions", []),
                "writings": clean_text(doc, "writings"),
                "vector": vec
            }
        })

    # Index publications
    publication_docs = parse_bulk_json(publications_json)
    pub_texts = [clean_text(doc, "description") for doc in publication_docs]
    pub_vectors = embed(pub_texts)
    pub_actions = []
    for doc, vec, text in zip(publication_docs, pub_vectors, pub_texts):
        keywords = extract_keywords(text, language="romanian")  # Dynamically extract keywords
        pub_actions.append({
            "_index": index_name,
            "_id": f"publication_{doc.get('name', doc.get('search_name'))}",
            "_source": {
                "type": "publication",
                "name": doc.get("name"),
                "description": text,
                "keywords": keywords,  # Add keywords field
                "category": clean_text(doc, "broad_category"),
                "vector": vec
            }
        })

    helpers.bulk(es, author_actions + pub_actions)

if __name__ == "__main__":
    create_unified_index(INDEX_ALL)
    index_unified_documents("data/authors_bulk.json", "data/publications_bulk.json", INDEX_ALL)
