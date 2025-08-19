from elasticsearch import Elasticsearch
from src.embedder import embed
from src.config import ES_HOST

es = Elasticsearch(ES_HOST)

def hybrid_search(query, index_name, k):
    """
    Hybrid search: focuses on matching the query with document fields.
    """
    # Embed the full query for vector similarity
    query_vector = embed([query])[0]

    body = {
        "size": k,
        "query": {
            "script_score": {
                "query": {
                    "bool": {
                        "should": [
                            {
                                "multi_match": {
                                    "query": query,
                                    "fields": ["name^3", "description^1.5"],
                                    "type": "best_fields"
                                }
                            },
                            {
                                "multi_match": {
                                    "query": query,
                                    "fields": ["name", "description"],
                                    "type": "most_fields"
                                }
                            }
                        ]
                    }
                },
                "script": {
                    "source": (
                        "0.7 * cosineSimilarity(params.query_vector, 'vector') + "
                        "0.3 * _score"
                    ),
                    "params": {
                        "query_vector": query_vector
                    }
                }
            }
        },
        "highlight": {
            "fields": {
                "description": {
                    "fragment_size": 700,
                    "number_of_fragments": 3,
                    "pre_tags": [""],
                    "post_tags": [""]
                }
            }
        }
    }

    response = es.search(index=index_name, body=body)
    hits = response["hits"]["hits"]
    results = []
    for hit in hits:
        source = hit["_source"]
        highlight = hit.get("highlight", {}).get("description", [])
        score = hit["_score"]
        doc_type = source.get("type", "unknown")
        enriched_result = {
            "type": doc_type,
            "name": source.get("name", ""),
            "description": source.get("description", ""),
            "keywords": source.get("keywords", []),  # Include keywords in results
            "professions": source.get("professions", []) if doc_type == "author" else None,
            "writings": source.get("writings", []) if doc_type == "author" else None,
            "category": source.get("category", "") if doc_type == "publication" else None,
            "score": score,
            "highlight": highlight
        }
        results.append(enriched_result)

    return results
