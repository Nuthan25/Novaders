import json
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def search_relevant_node(index, model_name, question_embedding, client, question_text=None):
    """
    Enhanced search function using hybrid search approach (BM25 + kNN)

    Args:
        index: The OpenSearch index name
        model_name: Model identifier
        question_embedding: Vector embedding of the question
        client: OpenSearch client
        question_text: Original question text for keyword matching (optional)

    Returns:
        List of relevant nodes
    """
    logger.info(f"Searching for relevant nodes in index {index} for model {model_name}")

    # Base query with kNN component
    query_components = []

    # Add kNN component
    knn_query = {
        "knn": {
            "embedding": {
                "vector": question_embedding,
                "k": 50
            }
        }
    }
    query_components.append(knn_query)

    # Add BM25 text search if question_text is provided
    if question_text:
        # Use specific text fields rather than dynamic properties for fuzzy matching
        text_query = {
            "bool": {
                "should": [
                    {
                        "match": {
                            "note": {
                                "query": question_text,
                                "boost": 2.0,
                                "fuzziness": "AUTO"
                            }
                        }
                    },
                    {
                        "match": {
                            "label": {
                                "query": question_text,
                                "boost": 2.0,
                                "fuzziness": "AUTO"
                            }
                        }
                    },
                    {
                        "match": {
                            "node_id": {
                                "query": question_text
                            }
                        }
                    },
                    {
                        "match": {
                            "model_id": {
                                "query": model_name,
                                "boost": 2.0,
                                "fuzziness": "AUTO"
                            }
                        }
                    },
                    {
                        "match": {
                            "type": {
                                "query": "node"
                            }
                        }
                    },
                ]
            }
        }
        query_components.append(text_query)

    # Build the full query without collapse - we'll handle deduplication in post-processing
    query_body = {
        "size": 10,  # Request more results to account for potential duplicates
        "query": {
            "bool": {
                "should": query_components,
                "filter": [
                    {"term": {"model_id": model_name}},
                    {"term": {"type": "node"}}
                ],
                "minimum_should_match": 1
            }
        },
        "_source": ["node_id", "label", "properties", "note", "model_id"]
    }

    try:
        response = client.search(index=index, body=query_body)

        if "hits" in response and "hits" in response["hits"]:
            results = response["hits"]["hits"]
            logger.info(f"Found {len(results)} raw results before deduplication")

            # Post-process and deduplicate results
            processed_results = []
            seen_node_ids = set()

            for hit in results:
                if "_source" in hit:
                    node_id = hit["_source"].get("node_id", "")

                    # Skip if we've already seen this node_id
                    if node_id in seen_node_ids:
                        continue

                    seen_node_ids.add(node_id)

                    node_data = {
                        "node_id": node_id,
                        "label": hit["_source"].get("label", ""),
                        "note": hit["_source"].get("note", ""),
                        "properties": hit["_source"].get("properties", {}),
                        "model_id": hit["_source"].get("model_id", ""),
                        "score": hit["_score"]
                    }
                    processed_results.append(node_data)

            logger.info(f"Found {len(processed_results)} nodes after deduplication")

            # If we found less than 3 results, try again without the model_id filter
            if len(processed_results) < 3:
                logger.info(f"Found only {len(processed_results)} results, trying without model_id filter")
                query_body["query"]["bool"]["filter"] = [{"term": {"type": "node"}}]
                response = client.search(index=index, body=query_body)

                if "hits" in response and "hits" in response["hits"]:
                    additional_results = response["hits"]["hits"]
                    for hit in additional_results:
                        if "_source" in hit and len(processed_results) < 10:
                            # Check if we already have this node
                            node_id = hit["_source"].get("node_id", "")
                            if node_id not in seen_node_ids:
                                seen_node_ids.add(node_id)
                                node_data = {
                                    "node_id": node_id,
                                    "label": hit["_source"].get("label", ""),
                                    "note": hit["_source"].get("note", ""),
                                    "properties": hit["_source"].get("properties", {}),
                                    "model_id": hit["_source"].get("model_id", ""),
                                    "score": hit["_score"]
                                }
                                processed_results.append(node_data)

            return processed_results
    except Exception as e:
        logger.error(f"Error searching for relevant nodes: {str(e)}")

    return []


def query_shared_node_index(index, model_name, question_embedding, client):
    query_body = {
            "size": 10,
            "query": {
                "script_score": {
                    "query": {
                        "bool": {
                            "must": [
                                {"match": {"model_id": model_name}},
                                {"match": {"type": "node"}}
                            ]
                        }
                    },
                    "script": {
                        "source": "knn_score",
                        "lang": "knn",
                        "params": {
                            "field": "embedding",
                            "query_value": question_embedding,
                            "space_type": "cosinesimil"
                        }
                    } 
                }
            },
            "_source": ["node_id", "label", "properties", "note"]
        }


    response = client.search(index=index, body=query_body)
    if "hits" in response:
        results = response["hits"]["hits"]
        # print(results)
        return [
            {"node_id": hit["_source"]["node_id"], "label": hit["_source"]["label"], "note": hit["_source"]["note"],
             "properties": hit["_source"]["properties"], "score": hit["_score"]} for hit in results]

    return []
    
def search_similar_edges(index, model_name, question_embedding, edge, client):
    query_body = {
        "size": 2,
        "query": {
            "script_score": {
                "query": {
                    "bool": {
                        "must": [
                            {"match": {"model_id": model_name}},
                            {"match": {"type": "edge"}},
                            {
                                "bool": {
                                    "should": [
                                        {"match": {"from_node": edge}},
                                        {"match": {"to_node": edge}}
                                    ],
                                    "minimum_should_match": 1
                                }
                            }
                        ]
                    }
                },
                "script": {
                    "source": "knn_score",
                    "lang": "knn",
                    "params": {
                        "field": "embedding",
                        "query_value": question_embedding,
                        "space_type": "cosinesimil"
                    }
                }
            }
        },
        "_source": ["from_node", "to_node", "relationship"]
    }


    response = client.search(index=index, body=query_body)
    results = []
    if "hits" in response and "hits" in response["hits"]:
        for hit in response["hits"]["hits"]:
            results.append({
                "from": hit["_source"]["from_node"],
                "to": hit["_source"]["to_node"],
                "relationship": hit["_source"]["relationship"],
                "score": hit["_score"]
            })

    return results

def query_shared_index(index,model_name, doc_type, question_embedding, client, size):    
    query_body = {
            "size": size,
            "query": {
                "script_score": {
                    "query": {
                        "bool": {
                            "must": [
                                {"match": {"model_id": model_name}},
                                {"match": {"type": doc_type}}
                            ]
                        }
                    },
                    "script": {
                        "source": "knn_score",
                        "lang": "knn",
                        "params": {
                            "field": "embedding",
                            "query_value": question_embedding,
                            "space_type": "cosinesimil"
                        }
                    } 
                }
            },
            "_source": ["content"]
        }


    response = client.search(index=index, body=query_body)
    results = []
    if "hits" in response and "hits" in response["hits"]:
        for hit in response["hits"]["hits"]:
                results.append({
                    "Note": hit["_source"]["content"]
                })
        all_notes = " ".join(item["Note"] for item in results)
        return all_notes

def query_shared_index_like(index, model_name, doc_type, question, client, size):    
    query_body = {
            "size": size,
            "query": {
                "bool": {
                    "must": [
                        {"term": {"model_id": model_name}},
                        {"term": {"type": doc_type}},
                        {"term": {"question": question}}
                    ]
                }
            },
            "_source": ["question","query"]
        }

    response = client.search(index=index, body=query_body)
    results = []
    if "hits" in response and "hits" in response["hits"]:
        for hit in response["hits"]["hits"]:
                results.append({
                    "question": hit["_source"]["question"],
                    "query": hit["_source"]["query"]
                })

        return results
    
def check_query_index(index, model_name, doc_type, client) -> bool:
    query_body = {
        "size": 1,
        "query": {
            "bool": {
                "must": [
                    {"term": {"model_id": model_name}},
                    {"term": {"type": doc_type}}
                ]
            }
        }
    }

    response = client.search(index=index, body=query_body)
    if "hits" in response:
        result = response["hits"]["hits"]
        return bool(result)  # Will return True if there are hits, else False
