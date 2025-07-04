import json

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
