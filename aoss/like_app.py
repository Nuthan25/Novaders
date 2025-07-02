import json
import logging
from opensearchpy import OpenSearch, RequestsHttpConnection, ConnectionError, TransportError, helpers
# from sentence_transformers import SentenceTransformer
from requests_aws4auth import AWS4Auth
from opensearchpy import NotFoundError
from opensearchpy import helpers, OpenSearch
import boto3
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
import io
import os
# Initialize S3 client

# from error_helper import sqs_helper

logger = logging.getLogger()
logger.setLevel(logging.INFO)

s3 = boto3.client("s3")

# S3 Bucket and Base Folder
BUCKET = os.getenv('BUCKET', 'preprod-cymonix-internal')
DB_HOST_AOSS = os.getenv('DB_HOST_AOSS', 'https://ft9umsemmpbp0anq2s6e.us-west-2.aoss.amazonaws.com')
DB_PORT_AOSS = os.getenv('DB_PORT_AOSS', '443')
INDEX = os.getenv('INDEX', 'preprod_index')
AWSREGION = os.getenv('AWSREGION', 'us-west-2')

def get_text_embedding(text: str, dimensions, region: str = AWSREGION):
    """
    Generate text embeddings using Amazon Bedrock's Titan Text Embeddings V2 model.

    :param text: The input text to be embedded.
    :param dimensions: The dimensionality of the output embedding (256, 512, or 1024). Default is 1024.
    :param region: AWS region where Bedrock service is deployed.
    :return: List representing the text embedding vector.
    """
    bedrock_runtime = boto3.client(service_name='bedrock-runtime', region_name=AWSREGION)
    try:
        if dimensions not in [256, 512, 1024]:
            raise ValueError("Invalid dimensions. Must be one of 256, 512, or 1024.")

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
        logger.info(f"Error creating Embedding: {e}")
        return "error"


def chunk_text(text):
    """Splits the text into smaller chunks based on periods ('.') while preserving sentence boundaries."""
    chunks = []
    words = text.split()
    current_chunk = []

    try:
        for word in words:
            current_chunk.append(word)  # Add word to current chunk

            if '.' in word:  # If a period is present, finalize the chunk
                chunks.append(" ".join(current_chunk))
                current_chunk = []  # Reset chunk

        # Add any remaining words as the last chunk
        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks
    except Exception as e:
        logger.info(f"Error creating chunks: {e}")
        return "error"


def get_delete_id(index, model_name, doc_type, client):
    query_body = {
        "query": {
            "bool": {
                "must": [
                    {"term": {"model_id": model_name}},
                    {"term": {"type": doc_type}}
                ]
            }
        },
        "_source": ["_id"]
    }
    try:
        results = []
        response = client.search(index=index, body=query_body)
        if "hits" in response and "hits" in response["hits"]:
            for hit in response["hits"]["hits"]:
                results.append({"id": hit["_id"]})
        print(f"ID's of doc_type {doc_type}: {results}")
        del_response = delete_query_by_id(index, results, model_name, doc_type, client)
        print(f"Deleted model: {model_name}_{doc_type}, response: {del_response}")
        return del_response
    except NotFoundError as e:
        # Log full context for debugging
        logger.info(f"Error in fetching doc id: {e}")


def delete_query_by_id(index, result, model_name, doc_type, client):
    print("result:--", result)
    all_responses = []
    try:
        for id_type in result:
            id_value = id_type.get('id')
            print(f"deleting id {id_value}, for type {doc_type}")
            response = client.delete(index=index, id=id_value)

            if response.get('result') == 'deleted':
                logger.info(f"Document with ID {id_value} deleted, for model {model_name}_{doc_type}")
            elif response.get('result') == 'not_found':
                logger.info(f"Document with ID {id_value} not found, for model {model_name}_{doc_type}")
                response = {"status": "not_found", "id": id_value}
            else:
                logger.info(f"Unexpected result: {response.get('result')}")

            all_responses.append(response)

        return all_responses

    except Exception as e:
        logger.info("Error deleting the index.")
        logger.info(e)
        return {"status": "error", "error": str(e)}


def create_opensearch_client():
    """Initialize OpenSearch client with exception handling."""
    region = AWSREGION
    service = 'aoss'
    try:
        host = DB_HOST_AOSS
        session = boto3.Session()
        credentials = session.get_credentials()

        awsauth = AWS4Auth(
            credentials.access_key,
            credentials.secret_key,
            region,
            service,
            session_token=credentials.token
        )
        return OpenSearch(
            hosts=[{'host': host.replace('https://', ''), 'port': DB_PORT_AOSS}],
            http_auth=awsauth,
            use_ssl=True,
            verify_certs=True,
            connection_class=RequestsHttpConnection,
            timeout=30,
            max_retries=3,
            retry_on_timeout=True
        )
    except Exception as e:
        logger.info(f"Error initializing OpenSearch client: {e}")
        return e


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
        return bool(result)


import boto3
import json


def get_text_schema_doc(org_id, job_id, only_json=True):
    """
    Retrieves all JSON data from a given S3 bucket and prefix.

    Args:
        bucket_name (str): Name of the S3 bucket.
        prefix (str): Prefix (folder path) in the S3 bucket.
        only_json (bool): If True, only process files ending in .json.

    Returns:
        list: A list of JSON-parsed Python dictionaries.
    """
    s3 = boto3.client('s3')
    json_data_list = []
    continuation_token = None
    bucket_name = BUCKET

    while True:
        list_kwargs = {'Bucket': bucket_name, 'Prefix': f'import/izzy_liked/{org_id}/{job_id}/'}
        if continuation_token:
            list_kwargs['ContinuationToken'] = continuation_token

        response = s3.list_objects_v2(**list_kwargs)

        for obj in response.get('Contents', []):
            key = obj['Key']
            if not only_json or key.endswith('.json'):
                file_obj = s3.get_object(Bucket=bucket_name, Key=key)
                file_content = file_obj['Body'].read().decode('utf-8')
                try:
                    json_data = json.loads(file_content)
                    json_data_list.append(json_data)
                except json.JSONDecodeError:
                    logger.info(f"Warning: Could not decode JSON from {key}")

        # Handle pagination
        if response.get('IsTruncated'):
            continuation_token = response.get('NextContinuationToken')
        else:
            break

    return json_data_list


def index_likes(client, json_data, LIKED_INDEX, model_name):
    """Process and index liked queries with error handling."""
    if client is None:
        return "OpenSearch client is not initialized."

    if not json_data or not isinstance(json_data, list):
        logger.info("No valid data found. Skipping indexing.")
        return "No data to index."

    error = ""  # Store errors if any

    for item in json_data:  # item is a dict like {'data': [...]}
        data_entries = item.get("data", [])
        for liked in data_entries:
            try:
                question = liked.get("Question")
                query = liked.get("Query")

                if not question or not query:
                    logger.info(f"Skipping invalid entry: {liked}")
                    continue  # Skip invalid entries

                liked_question = f"model_id: {model_name}_like 'question': {question}, 'query': {query}"
                embeddings = get_text_embedding(liked_question, dimensions=512)
                like_doc = {
                    "model_id": model_name,
                    "type": "like",
                    "question": question,
                    "query": query,
                    "embedding": embeddings,
                }
                client.index(index=LIKED_INDEX, body=like_doc)
                logger.info(f"Indexed: {question} -> {query}")

            except TransportError as e:
                logger.info(f"Error indexing like: {e}")
                error = f"Error indexing like {e}"  # Mark failure

    return error if error else "Indexing completed successfully."


def lambda_handler_like(event, context):
    try:
        # records = event['Records']
        # messages = event

        # logger.info(f"Messages: {messages}")

        # for message in messages:
        # data = json.loads(message)
        key = event

        job_id = key.split("/")[-2]
        org_id = key.split("/")[-3]
        graph_index = INDEX
        model_name = f"model_{org_id}_{job_id}"
        schema_data = get_text_schema_doc(org_id, job_id)
        logger.info(f"question_query:-- {schema_data}")
        client = create_opensearch_client()
        if client.indices.exists(index=graph_index):
            if check_query_index(graph_index, model_name, "like", client):
                get_delete_id(graph_index, model_name, "like", client)
                result = index_likes(client, schema_data, graph_index, model_name)
                logger.info(result)
                if result:
                    logger.info(f"Mapped Liked queries for model {model_name}, after deletion of previous data")
                else:
                    logger.info(f"Error mapping Liked queries for model {model_name}")
            else:
                result = index_likes(client, schema_data, graph_index, model_name)
                logger.info(result)
                if result:
                    logger.info(f"Mapped Liked queries for model {model_name}")
                else:
                    logger.info(f"Error mapping Liked queries for model {model_name}")
        else:
            logger.info(f"No index name {graph_index}")

        logger.info('Event processed')
        return "like data loaded"

    except Exception as e:
        logger.info(f"Error processing event: {e}")
        logger.error(traceback.format_exc())
        trace_back = traceback.format_exc()
        # sqs_helper.queue_message(context.function_name, e, trace_back)