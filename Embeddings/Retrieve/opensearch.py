import json
import boto3
import logging
from requests_aws4auth import AWS4Auth
from opensearchpy import OpenSearch, RequestsHttpConnection, helpers

logger = logging.getLogger()
logger.setLevel(logging.INFO)


class OpenSearchClient:
    def __init__(self, host: str, port: str, index: str, region: str, model_id: str):
        self.host = host
        self.port = port
        self.index = index
        self.model_id = model_id
        self.region = region
        self.service = 'aoss'
        self.use_ssl = True
        self.verify_certs = True
        self.client = None

    def connect(self):
        try:
            session = boto3.Session()
            credentials = session.get_credentials()
            awsauth = AWS4Auth(credentials.access_key, credentials.secret_key, self.region, self.service,
                               session_token=credentials.token)
            self.client = OpenSearch(
                hosts=[{'host': self.host.replace('https://', ''), 'port': self.port}],
                http_auth=awsauth,
                use_ssl=self.use_ssl,
                verify_certs=self.verify_certs,
                connection_class=RequestsHttpConnection,
                timeout=30,
                max_retries=3,
                retry_on_timeout=True
            )
            logger.info(f"Connected to OpenSearch")
        except Exception as e:
            raise Exception(f"Error connecting to OpenSearch: {str(e)}")

    def disconnect(self):
        try:
            self.client.close()
            logger.info(f"Disconnected from OpenSearch")
        except Exception as e:
            raise Exception(f"Error disconnecting from OpenSearch: {str(e)}")

    def create_embedding(self, text: str, dimensions: int = 512):
        """
        Generate text embeddings using Amazon Bedrock's Titan Text Embeddings V2 model.

        :param text: The input text to be embedded.
        :param dimensions: The dimensionality of the output embedding (256, 512, or 1024). Default is 512.
        :return: List representing the text embedding vector.
        """
        bedrock_runtime = boto3.client(service_name='bedrock-runtime', region_name=self.region)
        try:
            payload = {
                "inputText": text,
                "dimensions": dimensions
            }

            response = bedrock_runtime.invoke_model(
                body=json.dumps(payload),
                modelId='amazon.titan-embed-text-v2:0',
                accept='application/json',
                contentType='application/json'
            )

            response_body = json.loads(response.get('body').read())
            return response_body.get("embedding", [])
        except Exception as e:
            raise Exception(f"Error creating Embedding: {str(e)}")

    def query_shared_index(self, doc_type, embedded_question):
        try:
            size_map = {"like": 1, "note": 10, "file": 10}
            size = size_map.get(doc_type, 10)
            if not doc_type == "like":
                query_body = {
                    "size": size,
                    "query": {
                        "script_score": {
                            "query": {
                                "bool": {
                                    "must": [
                                        {"match": {"model_id": self.model_id}},
                                        {"match": {"type": doc_type}}
                                    ]
                                }
                            },
                            "script": {
                                "source": "knn_score",
                                "lang": "knn",
                                "params": {
                                    "field": "embedding",
                                    "query_value": embedded_question,
                                    "space_type": "cosinesimil"
                                }
                            }
                        }
                    },
                    "_source": ["content"]
                }

                response = self.client.search(index=self.index, body=query_body)
                results = []
                all_notes = ""
                if "hits" in response and "hits" in response["hits"]:
                    for hit in response["hits"]["hits"]:
                        results.append({
                            "Note": hit["_source"]["content"]
                        })
                    all_notes = "\n".join(item["Note"] for item in results)
                return all_notes
            else:
                return self.query_shared_index_like(doc_type, embedded_question, size)
        except Exception as e:
            raise Exception(f"Error getting shared index: {str(e)}")

    def query_shared_index_like(self, doc_type, question, size):
        try:
            logger.info(f"Querying shared index for {doc_type}")
            query_body = {
                "size": size,
                "query": {
                    "bool": {
                        "must": [
                            {"term": {"model_id": self.model_id}},
                            {"term": {"type": doc_type}},
                            {"term": {"question": question}}
                        ]
                    }
                },
                "_source": ["question", "query"]
            }

            response = self.client.search(index=self.index, body=query_body)

            if "hits" in response and "hits" in response["hits"]:
                for hit in response["hits"]["hits"]:
                    return {
                        "question": hit["_source"]["question"],
                        "query": hit["_source"]["query"]
                    }

            return {}
        except Exception as e:
            raise Exception(f"Error getting liked query: {str(e)}")

    def check_query_index(self, doc_type) -> bool:
        try:
            query_body = {
                "size": 1,
                "query": {
                    "bool": {
                        "must": [
                            {"term": {"model_id": self.model_id}},
                            {"term": {"type": doc_type}}
                        ]
                    }
                }
            }

            response = self.client.search(index=self.index, body=query_body)
            if "hits" in response:
                result = response["hits"]["hits"]
                return bool(result)
            else:
                return False
        except Exception as e:
            raise Exception(f"Error checking query index: {str(e)}")
