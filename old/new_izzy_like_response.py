import json
import os
import boto3
import logging
import traceback
import uuid
import psycopg2
from typing import List
from botocore.exceptions import ClientError
from requests_aws4auth import AWS4Auth
from opensearchpy import OpenSearch, RequestsHttpConnection, ConnectionError, TransportError, helpers

from error_helper import sqs_helper

logger = logging.getLogger()
logger.setLevel(logging.INFO)

s3 = boto3.client('s3')
s3_resource = boto3.resource('s3')

DB_HOST = os.getenv('DB_HOST')
BUCKET = os.getenv('BUCKET')

bedrock_runtime = boto3.client(service_name='bedrock-runtime', region_name='us-west-2')

def get_text_embedding(text: str, dimensions, region: str = "us-west-2"):
    """
    Generate text embeddings using Amazon Bedrock's Titan Text Embeddings V2 model.

    :param text: The input text to be embedded.
    :param dimensions: The dimensionality of the output embedding (256, 512, or 1024). Default is 1024.
    :param region: AWS region where Bedrock service is deployed.
    :return: List representing the text embedding vector.
    """
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
        print(f"Error creating Embedding: {e}")
        return "error"
    

def create_opensearch_client():
    """Initialize OpenSearch client with exception handling."""
    region = "us-west-2"
    service = 'aoss'
    try:
        host = "https://59m13mwwnv15mukrewt8.us-west-2.aoss.amazonaws.com"
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
            hosts=[{'host': host.replace('https://', ''), 'port': 443}],
            http_auth=awsauth,
            use_ssl=True,
            verify_certs=True,
            connection_class=RequestsHttpConnection,
            timeout=30,
            max_retries=3,
            retry_on_timeout=True
        )
    except Exception as e:
        print(f"Error initializing OpenSearch client: {e}")
        return e
    
def create_indices(client, LIKED_INDEX):
    """Create OpenSearch indices with error handling."""
    if client is None:
        print("OpenSearch client is not initialized.")
        return "OpenSearch client is not initialized."

    liked_mapping = {
        "settings": {"index": {"knn": True}},
        "mappings": {"properties": {"embedding": {"type": "knn_vector", "dimension": 512, "method": {"name": "hnsw", "space_type": "cosinesimil"}},
                                       "question": {"type": "keyword"}, "query": {"type": "keyword"}}}
    }
    result = "Index Created Successfully"
    for index_name, mapping in [(LIKED_INDEX, liked_mapping)]:
        try:
            if not client.indices.exists(index=index_name):
                client.indices.create(index=index_name, body=mapping)
                print(f"Index '{index_name}' created.")
        except Exception as e:
            result = f"Error creating index {index_name}: {e}"
    
    return result

def index_likes(client, json_data, LIKED_INDEX):
    """Process and index edges with error handling."""
    if client is None:
        return "OpenSearch client is not initialized."  # Return False instead of None

    error = "Data Mapped Successfully" 
    for liked in json_data.get("data", {}):
        try:
            question = liked.get("Question")
            query = liked.get("Query")
            
            if not (question and query):
                print(f"Skipping invalid edge: {question}")
                continue  # Skip invalid edges
            
            liked_question = f"'question':{question},'query':{query}"
            embeddings = get_text_embedding(liked_question, dimensions=512)
            edge_doc = {
                "question": question,
                "query": query,
                "embedding": embeddings,
            }
            client.index(index=LIKED_INDEX, body=edge_doc)
            print(f"Indexed : {question} and {query}")
        except TransportError as e:
            print(f"Error indexing edge: {e}")
            error = f"Error indexing edge {e}"  # Mark failure

    return error  

def get_text_schema_doc(org_id, job_id):
    # Define bucket name and initialize an output string for the resul
    output_data = ''

    # List all JSON files in the bucket
    response = s3.list_objects_v2(Bucket=BUCKET, Prefix=f'import/izzy_liked/{org_id}/{job_id}/')

    for obj in response.get('Contents', []):
        if obj['Key'].endswith('.json'):
            # Read the JSON file
            file_content = s3.get_object(Bucket=BUCKET, Key=obj['Key'])['Body'].read()
            json_data = json.loads(file_content)
            
    return json_data

def lambda_handler(event, context):
    try:
        records = event['Records']
        messages = [record['body'] for record in records]

        logger.info(f"Messages: {messages}")

        for message in messages:
            data = json.loads(message)
            key = data['key']

            job_id = key.split("/")[-2]
            org_id = key.split("/")[-3]
            LIKED_INDEX = f'liked_index_{org_id}_{job_id}'

            schema_data = get_text_schema_doc(org_id, job_id)

            if create_opensearch_client():
                client = create_opensearch_client()
                logger.info("Connected to DB")
            else:
                logger.info(f"Error creating Connection, {client}")
            
            if client.indices.exists(index=LIKED_INDEX):
                client.indices.delete(index=LIKED_INDEX)
                logger.info("Deleted previous Indices")
                if create_indices(client, LIKED_INDEX):
                    logger.info("Created Index")
                else:
                    logger.info(f"Error creating Index, {create_indices(client, LIKED_INDEX)}")
            else:
                if create_indices(client, LIKED_INDEX):
                    logger.info("Created new Index")
                else:
                    logger.info(f"Error creating Index, {create_indices(client, LIKED_INDEX)}")
            
            if index_likes(client, schema_data, LIKED_INDEX):
                logger.info("Mapped data to index")
            else:
                logger.info(f"Error mapping to Index, {index_likes(client, schema_data, LIKED_INDEX)}")

        logger.info('Event processed')

    except Exception as e:
        logger.info(f"Error processing event: {e}")
        logger.error(traceback.format_exc())
        trace_back = traceback.format_exc()
        sqs_helper.queue_message(context.function_name, e, trace_back)

