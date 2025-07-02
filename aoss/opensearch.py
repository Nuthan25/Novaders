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

    def check_and_create_indices_index(self):
        try:
            body = {
                "settings": {
                    "index": {
                        "knn": True,
                        "number_of_shards": 3,
                        "number_of_replicas": 1
                    }
                },
                "mappings": {
                    "properties": {
                        "model_id": {"type": "keyword"},  # Added field to identify the model
                        "type": {"type": "keyword"},  # Identifies document type: node, edge, chunk, liked_query

                        "node_id": {"type": "keyword"},
                        "label": {"type": "keyword"},
                        "properties": {
                            "type": "object",
                            "dynamic": False  # stored but not indexed
                        },
                        "properties_flat": {
                            "type": "flat_object"
                        },
                        "note": {"type": "keyword"},

                        "from_node": {"type": "keyword"},
                        "to_node": {"type": "keyword"},
                        "relationship": {"type": "keyword"},

                        "chunk_id": {"type": "integer"},
                        "content": {"type": "text"},

                        "question": {"type": "keyword"},
                        "query": {"type": "keyword"},

                        "embedding": {
                            "type": "knn_vector",
                            "dimension": 512,
                            "method": {"name": "hnsw", "space_type": "cosinesimil"}
                        },
                    }
                }
            }
            if not self.client.indices.exists(index=self.index):
                self.client.indices.create(index=self.index, body=body)
                logger.info(f"Created indices index: {self.index}")
            else:
                logger.info(f"Indices index already exists: {self.index}")
        except Exception as e:
            raise Exception(f"Error checking and creating indices index: {str(e)}")

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

    def check_query_index(self, document_type: str):
        try:
            result = False
            body = {
                "size": 1,
                "query": {
                    "bool": {
                        "must": [
                            {"term": {"model_id": self.model_id}},
                            {"term": {"type": document_type}}
                        ]
                    }
                }
            }
            response = self.client.search(index=self.index, body=body)
            if "hits" in response:
                result = response["hits"]["hits"]

            return bool(result)
        except Exception as e:
            raise Exception(f"Error checking query index: {str(e)}")

    def create_index(self, document_type: str, data):
        try:
#             if document_type in ['node', 'edge']:
#                 for data_item in data:
#                     doc = {
#                         'model_id': self.model_id, 'type': document_type,
#                         'label': data_item['label'],
#                         'properties': data_item['properties'],
#                         'note': data_item['note'] if data_item.get('note') else 'No Notes',
#                         'embedding': self.create_embedding(json.dumps(data_item)),
#                     }

#                     self.client.index(index=self.index, body=doc)
            if document_type in ['note', 'file', 'catalogue']:
                docs = []
                for i, data_item in enumerate(data):
                    if data_item:
                        doc = {
                            '_index': self.index,
                            '_source': {
                                'model_id': self.model_id,
                                'type': document_type,
                                'chunk_id': i,
                                'content': data_item,
                                'embedding': self.create_embedding(data_item),
                            }
                        }
                        docs.append(doc)

                if docs:
                    helpers.bulk(self.client, docs)

            logger.info(f"Created index")
        except Exception as e:
            raise Exception(f"Error creating index: {str(e)}")

    def delete_index(self, document_type: str):
        try:
            body = {
                "query": {
                    "bool": {
                        "must": [
                            {"term": {"model_id": self.model_id}},
                            {"term": {"type": document_type}}
                        ]
                    }
                },
            }
            response = self.client.count(index=self.index, body=body)

            body["size"] = response["count"]
            body["_source"] = ["_id"]

            response = self.client.search(index=self.index, body=body)

            if "hits" in response and "hits" in response["hits"]:
                for hit in response["hits"]["hits"]:
                    self.client.delete(index=self.index, id=hit["_id"])

        except Exception as e:
            logger.error(f"Error deleting index: {str(e)}")
            # raise Exception(f"Error deleting index: {str(e)}")
