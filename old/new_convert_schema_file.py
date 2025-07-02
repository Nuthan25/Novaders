import json
import boto3
import requests
import logging
import uuid
import os
import traceback
from requests_aws4auth import AWS4Auth
from opensearchpy import OpenSearch, RequestsHttpConnection, ConnectionError, TransportError, helpers
import pdfplumber
from botocore.exceptions import ClientError
import io
import re

from error_helper import sqs_helper

logger = logging.getLogger()
logger.setLevel(logging.INFO)

s3 = boto3.client('s3')
s3_resource = boto3.resource('s3')

BUCKET = os.getenv('BUCKET')
DB_HOST = os.getenv('DB_HOST')

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
    
def chunk_text(text):
    """Splits the text into smaller chunks based on periods ('.') while preserving sentence boundaries."""
    chunks = []
    text = get_pdf_from_s3(text)
    if any("Error" in word.lower() for word in text):
        logger.info(f"Error While extrating data from S3, {text}")
        return f'Error, {text}'
    else:
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
            print(f"Error creating chunks: {e}")
            return f"Error creating chunks: {e}"

def note_chunk_text(text):
    """Splits the text into smaller chunks based on periods ('.') while preserving sentence boundaries."""
    chunks = []
    if any("Error" in word.lower() for word in text):
        logger.info(f"Error While extrating data from S3, {text}")
        return f'Error, {text}'
    else:
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
            print(f"Error creating chunks: {e}")
            return f"Error creating chunks: {e}"

def load_json_data(filepath):
    """Load JSON schema from file."""
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading JSON data: {e}")
        return "Error"

def get_pdf_from_s3(files):
    # Extract resourceList from the event
    resource_list = files
    results = []

    # Specify the S3 bucket name (you need to know this beforehand)
    bucket_name = BUCKET
    logger.info(f"Bucket:- {bucket_name}")
    logger.info(f"Resource:- {resource_list}")
    try:
        for resource_path in resource_list:
            logger.info(f"inside resource for-loop getting file from {resource_path}")
            try:
                # Get the S3 object path
                s3_key = resource_path

                # Download the file content from S3
                pdf_content = s3.get_object(Bucket=bucket_name, Key=s3_key)['Body'].read()

                # Process the PDF content
                with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
                    text = ''
                    for page in pdf.pages:
                        text += page.extract_text() + '\n'
                # Append the extracted text to results
                results.append(text)

            except Exception as e:
                # Log any error encountered
                logger.info(f"error while reading file from s3: {e}")
                results.append(f"Error while reading file from s3: {e}")
        # Return the extracted content or errors
    except Exception as e:
        logger.info(f"failed to get file from s3:{e}")
        results.append(f"failed to get file from s3:{e}")

    return results

def create_opensearch_client():
    """Initialize OpenSearch client with exception handling."""
    region = "us-west-2"
    service = 'aoss'
    try:
        host = DB_HOST
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

def create_indices(client, NODE_INDEX, EDGE_INDEX):
    """Create OpenSearch indices with error handling."""
    if client is None:
        print("OpenSearch client is not initialized.")
        return "OpenSearch client is not initialized."

    node_mapping = {
        "settings": {"index": {"knn": True}},
        "mappings": {"properties": {"embedding": {"type": "knn_vector", "dimension": 512, "method": {"name": "hnsw", "space_type": "cosinesimil"}},
                                       "node_id": {"type": "keyword"}, "label": {"type": "keyword"}, "note": {"type": "keyword"}, "properties": {"type": "object"}}}
    }

    edge_mapping = {
        "settings": {"index": {"knn": True}},
        "mappings": {"properties": {"embedding": {"type": "knn_vector", "dimension": 512, "method": {"name": "hnsw", "space_type": "cosinesimil"}},
                                       "from_node": {"type": "keyword"}, "to_node": {"type": "keyword"}, "note": {"type": "keyword"}, "properties": {"type": "object"}, "relationship": {"type": "keyword"}}}
    }
    result = " "
    for index_name, mapping in [(NODE_INDEX, node_mapping), (EDGE_INDEX, edge_mapping)]:
        try:
            if not client.indices.exists(index=index_name):
                client.indices.create(index=index_name, body=mapping)
                print(f"Index '{index_name}' created.")
        except Exception as e:
            result = f"Error creating index {index_name}: {e}"
    
    return result

def create_index(client, index_name):
    """Creates an OpenSearch index with k-NN enabled for vector search."""
    index_body = {
        "settings": {
            "index": {
                "knn": True
            }
        },
        "mappings": {
            "properties": {
                "chunk_id": {"type": "integer"},
                "content": {"type": "text"},
                "embedding": {"type": "knn_vector", "dimension": 512, "method": {"name": "hnsw", "space_type": "cosinesimil"}}
            }
        }
    }
    try:
        if not client.indices.exists(index=index_name):
            client.indices.create(index=index_name, body=index_body)
            logging.info(f"Index '{index_name}' created.")
    except TransportError as e:
        print(f"Error creating index {index_name}: {e}")
        return e

def update_model(message, org_id):
    logger.info(f'Message: {message}')
    try:
        sqs_client = boto3.client('sqs')
        sqs_client.send_message(
            QueueUrl=os.environ['INPUT_TRIGGER_URL'],
            MessageBody=json.dumps(message, default=str),
            MessageDeduplicationId=uuid.uuid4().__str__(),
            MessageGroupId=uuid.uuid4().__str__()
        )
        logger.info(json.dumps({"log_type": "sqs", "value": org_id}))
    except Exception as e:
        raise e    

def get_json_schema(key, bucket):
    s3_response = s3.get_object(
        Bucket=bucket,
        Key=key
    )
    s3_object_body = s3_response.get('Body')
    content = s3_object_body.read()
    return json.loads(content)

def extract_nodes(json_data):
    """Extract nodes and properties from JSON schema with validation."""
    node_map = {}
    try:
        if not json_data:
            logging.warning("Empty JSON data provided.")
            return node_map

        types = {1: 'STRING', 2: 'INTEGER', 3: 'FLOAT', 4: 'DATE', 5: 'BOOLEAN', 6: 'DATE', 7: 'DATE', 8: 'DATE', 9: 'DATE',}

        for node in json_data.get("schema", {}).get("vertex", []):
            node_id = node.get("label")
            label = node.get("label")
            note = node.get("notes", "No data")
            properties_list = node.get("properties", [])
            properties = {prop["key"]: types.get(prop["type"], "UNKNOWN") for prop in properties_list}

            if node_id and label:
                node_map[node_id] = {"label": label, "properties": properties, "note": note}
            else:
                logging.warning(f"Skipping node with missing label: {node}")

        return node_map
    except Exception as e:
        print(f"Error Extracting Nodes: {e}")
        return "error, {e}"
    
def index_nodes(client, node_map, NODE_INDEX):
    """Index nodes with exception handling."""
    if client is None:
        return "Error: OpenSearch client is not initialized."
    
    error = " " 
    for node_id, node_data in node_map.items():
        try:
            properties_text = " ".join([f"{k}:{v}" for k, v in node_data["properties"].items()])
            text_data = f"{node_id} {node_data['label']} {node_data['note']} {properties_text}"
            embedding = get_text_embedding(text_data, dimensions=512)
            
            opensearch_doc = {"node_id": node_id, "label": node_data["label"], "properties": node_data["properties"], "note": node_data["note"], "embedding": embedding}
            
            client.index(index=NODE_INDEX, body=opensearch_doc)
            print(f"Indexed Node: {node_id}")
        except TransportError as e:
            error = f"Error indexing node {node_id}: {e}"
    return error
            
def index_edges(client, json_data, EDGE_INDEX):
    """Process and index edges with error handling."""
    if client is None:
        return "OpenSearch client is not initialized."  # Return False instead of None

    error = " " 
    for edge in json_data.get("schema", {}).get("edge", []):
        try:
            from_node = edge.get("fromLabel")
            to_node = edge.get("toLabel")
            relationship = edge.get("label")
            properties = edge.get("properties", {})
            note = edge.get("notes")
            
            if not (from_node and to_node and relationship):
                print(f"Skipping invalid edge: {edge}")
                continue  # Skip invalid edges
            
            relation = f"{from_node}-{relationship}->{to_node}"
            embeddings = get_text_embedding(relation, dimensions=512)
            edge_doc = {
                "from_node": from_node,
                "to_node": to_node,
                "relationship": relationship,
                "embedding": embeddings,
                "properties": properties,
                "note": note
            }
            client.index(index=EDGE_INDEX, body=edge_doc)
            print(f"Indexed Edge: {from_node} -> {relationship} -> {to_node}")
        except TransportError as e:
            print(f"Error indexing edge: {e}")
            error = f"Error indexing edge {e}"  # Mark failure

    return error  

def index_chunks(client, index_name, text):
    """Splits text into chunks, generates embeddings, and indexes them into OpenSearch."""
    chunks = chunk_text(text)
    if "Error" in chunks:
        logger.info(f"Error creating chunks, {chunks}")
        return f"Error, {chunks}"
    else:
        actions = []
        error = ''
        for i, chunk in enumerate(chunks):
            try:
                embedding = get_text_embedding(chunk, 512)
                actions.append({
                    "_index": index_name,
                    "_source": {
                        "chunk_id": i,
                        "content": chunk,
                        "embedding": embedding
                    }
                })
            except Exception as e:
                print(f"Skipping chunk {i} due to error: {e}")
                error = f"Error in chunked data{e}"
        if actions:
            try:
                helpers.bulk(client, actions)
                print(f"Indexed {len(actions)} chunks into {index_name}")
            except helpers.BulkIndexError as bulk_error:
                for error in bulk_error.errors:
                    print(f"Error indexing document: {error}")
                    error = f"Error indexing document: {error}"
        else:
            print("No valid chunks to index.")
            error = "No valid chunks to index."
        return error

def note_index_chunk(client, index_name, text):
    """Splits text into chunks, generates embeddings, and indexes them into OpenSearch."""
    chunks = note_chunk_text(text)
    if "Error" in chunks:
        logger.info(f"Error creating chunks, {chunks}")
        return f"Error, {chunks}"
    else:
        actions = []
        error = ''
        for i, chunk in enumerate(chunks):
            try:
                embedding = get_text_embedding(chunk, 512)
                actions.append({
                    "_index": index_name,
                    "_source": {
                        "chunk_id": i,
                        "content": chunk,
                        "embedding": embedding
                    }
                })
            except Exception as e:
                print(f"Skipping chunk {i} due to error: {e}")
                error = f"Error in chunked data{e}"
        if actions:
            try:
                helpers.bulk(client, actions)
                print(f"Indexed {len(actions)} chunks into {index_name}")
            except helpers.BulkIndexError as bulk_error:
                for error in bulk_error.errors:
                    print(f"Error indexing document: {error}")
                    error = f"Error indexing document: {error}"
        else:
            print("No valid chunks to index.")
            error = "No valid chunks to index."
        return error

def read_json_from_s3(bucket, key):
    s3 = boto3.client("s3")
    try:
        response = s3.get_object(Bucket=bucket, Key=key)
        json_content = response["Body"].read().decode("utf-8")
        data = json.loads(json_content)
        return data
    except Exception as e:
        print(f"Error reading JSON from S3: {e}")
        return None

def lambda_handler(event, context):
    # --- AWS OpenSearch Configuration ---
    try:
        tmp_file = f'/tmp/{uuid.uuid4()}.txt'
        for record in event['Records']:
            model_id = None
            key = record['s3']['object']['key']
            bucket_name = record['s3']['bucket']['name']

            # Extract job_id and org_id from the key
            job_id = key.split("/")[-2]
            org_id = key.split("/")[-3]

            logger.info(json.dumps({"log_type": "org_id", "value": org_id}))

            # Get the schema from S3 (assuming this function is implemented)
            json_data = get_json_schema(key, bucket_name)
            logger.info(f"json_data:-{json_data}")
            model_id = json_data['schema'].get('fromModel')
            files = json_data['schema'].get('resourceList')

            # OpenSearch indices
            NODE_INDEX = f'node_index_{org_id}_{job_id}'
            EDGE_INDEX = f'edge_index_{org_id}_{job_id}'
            NOTE_INDEX = f'note_index_{org_id}_{job_id}'
            FILES_INDEX = f'file_index_{org_id}_{job_id}'

            try:
                if create_opensearch_client():
                    client = create_opensearch_client()
                    logger.info("Connected to DB")
                else:
                    logger.info(f"Error creating Connection, {client}")
                if client.indices.exists(index=NODE_INDEX) and client.indices.exists(index=EDGE_INDEX):
                    client.indices.delete(index=NODE_INDEX)
                    client.indices.delete(index=EDGE_INDEX)
                    logger.info("Deleted previous Indices")
                    if create_indices(client, NODE_INDEX, EDGE_INDEX):
                        logger.info("Created indices for Nodes and Edges")
                    else:
                        print("Error for indices creation", create_indices(client, NODE_INDEX, EDGE_INDEX))
                else:
                    if create_indices(client, NODE_INDEX, EDGE_INDEX):
                        logger.info("Created new indices for Nodes and Edges")
                    else:
                        print("Error for indices creation", create_indices(client, NODE_INDEX, EDGE_INDEX))
                if client.indices.exists(index=NOTE_INDEX):
                    client.indices.delete(index=NOTE_INDEX)
                    logger.info("Deleted previous Indices")
                    if create_index(client, NOTE_INDEX):
                        logger.info('Created Note Index')
                    else:
                        logger.info(f"Error creating Note Index, {create_index(client, NOTE_INDEX)}")
                else:
                    if create_index(client, NOTE_INDEX):
                        logger.info('Created new Note Index')
                    else:
                        logger.info(f"Error creating Note Index, {create_index(client, NOTE_INDEX)}")
                if client.indices.exists(index=FILES_INDEX):
                    client.indices.delete(index=FILES_INDEX)
                    logger.info("Deleted previous Indices")
                    if create_index(client, FILES_INDEX):
                        logger.info('Created File Index')
                    else:
                        logger.info(f"Error creating File Index, {create_index(client, FILES_INDEX)}")
                else:
                    if create_index(client, FILES_INDEX):
                        logger.info('Created new File Index')
                    else:
                        logger.info(f"Error creating File Index, {create_index(client, FILES_INDEX)}")

            except Exception as e:
                logger.info(f"Error Creating Index: {e}")

            try:
                if files:
                    logger.info("read and load the FILE to opensearch db.")
                    if index_chunks(client, FILES_INDEX, files):
                        logger.info('File mapped')
                    else:
                        logger.info(f'Error mapping Files{index_chunks(client,FILES_INDEX,files)}')
                else:
                    try:
                        if client.indices.exists(index=FILES_INDEX):
                            client.indices.delete(index=FILES_INDEX)
                            logger.info("deleted previous FILE")
                    except Exception as e:
                        logger.info(f"Error deleting FILE DB Instance: {e}")
            except Exception as e:
                logger.info(f"Error creating FILE DB Instance: {e}")
    
            schema_notes = json_data.get("schema", {}).get("notes", "No data")

            try:
                if schema_notes:
                    logger.info("read and load the Model Additional info to db")
                    if note_index_chunk(client, FILES_INDEX, schema_notes):
                        logger.info('File mapped')
                    else:
                        logger.info(f'Error mapping Files{note_index_chunk(client,FILES_INDEX,schema_notes)}')
            except Exception as e:
                logger.info(f"Error creating FILE DB Instance: {e}")

            if extract_nodes(json_data):
                node_map = extract_nodes(json_data)
                if 'error' in node_map:
                    logger.info(f"Error creating Node and Edge Map")
                else:
                    try:
                        if index_nodes(client, node_map, NODE_INDEX):
                            logger.info("mapped node data")
                        else:
                            logger.info(f"Error for node data, {index_nodes(client, node_map, NODE_INDEX)}")

                        if index_edges(client, json_data, EDGE_INDEX):
                            logger.info("mapped edge data")
                        else:
                            logger.info(f"Error for edge data, {index_edges(client, json_data, EDGE_INDEX)}")
                    except Exception as e:
                        logger.info(f"Error in mapping: {e}")

                    logger.info("mapped Node and Edge index")
                    logger.info("Processing completed!")
            else:
                logger.info(f"Error Extract Node and Edge")
            
            if model_id:
                update_model({
                    "path": "api/graph-model/",
                    "data": {
                        "id": model_id,
                        "notes_updating": False
                    }
                }, org_id)
                logger.info('Model updated')
        
    except Exception as e:
        logger.info(f"Error processing event: {e}")
        trace_back = traceback.format_exc()
        sqs_helper.queue_message(context.function_name, e, trace_back)

    



