import json
import logging
from opensearchpy import OpenSearch, RequestsHttpConnection, ConnectionError, TransportError, helpers
# from sentence_transformers import SentenceTransformer
from requests_aws4auth import AWS4Auth
from opensearchpy import NotFoundError
from opensearchpy import helpers, OpenSearch
import uuid
import boto3
# import numpy as np
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
import io
import os
import pdfplumber
import re

# Initialize S3 client

from error_helper import sqs_helper

logger = logging.getLogger()
logger.setLevel(logging.INFO)

s3 = boto3.client("s3")

# S3 Bucket and Base Folder
BUCKET_NAME = os.getenv('BUCKET')
DB_HOST_AOSS = os.getenv('DB_HOST_AOSS')
DB_PORT_AOSS = os.getenv('DB_PORT_AOSS')
INDEX = os.getenv('INDEX')


def get_text_embedding(text: str, dimensions, region: str = "us-west-2"):
    """
    Generate text embeddings using Amazon Bedrock's Titan Text Embeddings V2 model.

    :param text: The input text to be embedded.
    :param dimensions: The dimensionality of the output embedding (256, 512, or 1024). Default is 1024.
    :param region: AWS region where Bedrock service is deployed.
    :return: List representing the text embedding vector.
    """
    bedrock_runtime = boto3.client(service_name='bedrock-runtime', region_name='us-west-2')
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
        logger.info(e)


def delete_query_by_id(index, result, model_name, doc_type, client):
    all_responses = []
    try:
        for id_type in result:
            id_value = id_type.get('id')
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


def get_text_schema_doc(org_id, job_id, bucket_name, only_json=True):
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


def load_json_data(filepath):
    """Load JSON schema from file."""
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.info(f"Error loading JSON data: {e}")
        return "error"


def create_opensearch_client():
    """Initialize OpenSearch client with exception handling."""
    region = "us-west-2"
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


def create_unified_graph_index(client, INDEX_NAME):
    """Create a single OpenSearch index for nodes, edges, text chunks, and liked queries."""
    if client is None:
        logger.info("OpenSearch client is not initialized.")
        return "OpenSearch client is not initialized."

    unified_mapping = {
        "settings": {
            # Enable KNN if using vector search; adjust additional settings (like number_of_shards) if needed.
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

                # 🔹 Node Fields
                "node_id": {"type": "keyword"},
                "label": {"type": "keyword"},
                # Store ALL raw properties as a JSON blob (not indexed)
                "properties": {
                    "type": "object",
                    "dynamic": False  # stored but not indexed
                },
                # Store selected keys as flat fields so they can be searched/indexed
                "properties_flat": {
                    "type": "flat_object"
                },
                "note": {"type": "keyword"},

                # 🔹 Edge Fields
                "from_node": {"type": "keyword"},
                "to_node": {"type": "keyword"},
                "relationship": {"type": "keyword"},

                # 🔹 Text Chunk Fields
                "chunk_id": {"type": "integer"},
                "content": {"type": "text"},

                # 🔹 Liked Query Fields
                "question": {"type": "keyword"},
                "query": {"type": "keyword"},

                # 🔹 Common Fields
                "embedding": {
                    "type": "knn_vector",
                    "dimension": 512,
                    "method": {"name": "hnsw", "space_type": "cosinesimil"}
                },
            }
        }
    }

    try:
        if not client.indices.exists(index=INDEX_NAME):
            client.indices.create(index=INDEX_NAME, body=unified_mapping)
            logger.info(f"Index '{INDEX_NAME}' created.")
    except Exception as e:
        return f"Error creating index {INDEX_NAME}: {e}"

    return f"Index '{INDEX_NAME}' is ready."


def remove_none_from_text(text):
    if not text:
        return ""
        # Split the text into lines
    lines = text.split('\n')

    # Filter out lines that contain only "None" (considering possible whitespace)
    filtered_lines = [line for line in lines if line.strip() != "None"]

    # Join the filtered lines back into a single string
    return '\n'.join(filtered_lines)


def extract_nodes(json_data, job_id, org_id):
    """Extract nodes and properties from JSON schema with validation."""
    info_result = get_addition_info_text(job_id, org_id)
    if info_result == "No file data":
        info_json = ""
        nodes_list = []
        print("No addition info from note")
    else:
        info_after_remove_none = remove_none_from_text(info_result)
        info_json, nodes_list = get_addition_info(info_after_remove_none)
        print("Nodes addition info:--", nodes_list)

    node_map = {}
    info_note = {}

    try:
        if not json_data:
            logging.warning("Empty JSON data provided.")
            return node_map, info_note

        types = {
            1: 'STRING', 2: 'INTEGER', 3: 'FLOAT', 4: 'DATE',
            5: 'BOOLEAN', 6: 'DATE', 7: 'DATE', 8: 'DATE', 9: 'DATE',
        }

        for node in json_data.get("schema", {}).get("vertex", []):
            node_id = node.get("label")
            label = node.get("label")

            # Default note value
            note = node.get("notes", "No data")
            # note = "No data"

            # Update note with node_data if label is found in nodes_list
            node_data = get_node_data_by_label(nodes_list, label)
            if node_data:
                note = node_data

            properties_list = node.get("properties", [])
            properties = {}

            for prop in properties_list:
                raw_key = prop.get("key")
                prop_name = prop.get("replaceHeader") or raw_key
                properties[prop_name] = types.get(prop.get("type"), "UNKNOWN")

            if node_id and label:
                node_map[node_id] = {
                    "label": label,
                    "properties": properties,
                    "note": note if note is not None else "No data"
                }
            else:
                logging.warning(f"Skipping node with missing label: {node}")

        # Return the info_json as info_note for additional information
        info_note = info_json
        if info_note:
            print("Model addition info:--", info_note)
        return node_map, info_note

    except Exception as e:
        logging.error(f"Error Extracting Nodes: {e}")
        return "error", {}


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


def index_nodes(client, node_map, INDEX_NAME, model_name):
    """Index nodes with type 'node' and debug logging."""
    if client is None:
        logger.info("OpenSearch client is not initialized ❌")
        return "Error: OpenSearch client is not initialized."
    model_id = f"{model_name}_node"
    for node_id, node_data in node_map.items():
        try:
            logger.info(f"Indexing Node: {node_id} 🚀")
            if "label" not in node_data or "properties" not in node_data:
                logger.info(f"Skipping node {node_id} due to missing required fields ❌")
                continue
            text_data = model_id + str(node_data)
            embedding = get_text_embedding(text_data, dimensions=512)
            if not embedding:
                logger.info(f"Failed to generate embedding for node {node_id} ❌")
                continue
            opensearch_doc = {
                "model_id": model_name,
                "type": "node",
                # "node_id": node_id,
                "label": node_data["label"],
                "properties": node_data["properties"],
                # "properties_flat": node_data["properties"],
                "note": node_data.get("note", "No note"),
                "embedding": embedding
            }
            # logger.info(opensearch_doc)
            response = client.index(index=INDEX_NAME, body=opensearch_doc)
            # logger.info(response)

        except Exception as e:
            logger.info(f"Error indexing node {node_id}: {e} ❌")
            return f"Error indexing node {node_id}: {e}"

    return "Nodes indexed successfully ✅"


def index_edges(client, json_data, INDEX_NAME, model_name):
    """Index edges with debug logs."""
    if client is None:
        logger.info("OpenSearch client is not initialized ❌")
        return "Error: OpenSearch client is not initialized."

    error = " "
    model_id = f"{model_name}_edge"
    for edge in json_data.get("schema", {}).get("edge", []):
        try:
            from_node = edge.get("fromLabel")
            to_node = edge.get("toLabel")
            relationship = edge.get("label")
            properties = edge.get("properties", {})
            note = edge.get("notes", "No note")

            if not from_node or not to_node or not relationship:
                logger.info(f"Skipping invalid edge: {edge} ❌")
                continue

            logger.info(f"Indexing Edge: {from_node} -> {relationship} -> {to_node} 🚀")

            relation_text = f"{model_id} + {from_node}-{relationship}->{to_node}"
            embedding = get_text_embedding(relation_text, dimensions=512)
            if not embedding:
                logger.info(f"Failed to generate embedding for edge {from_node} -> {relationship} -> {to_node} ❌")
                continue

            edge_doc = {
                "model_id": model_name,
                "type": "edge",
                "label": relationship,
                "embedding": embedding,
                "properties": properties,
                "note": note
            }

            response = client.index(index=INDEX_NAME, body=edge_doc)
            logger.info(f"Indexed Edge {from_node} -> {relationship} -> {to_node} ✅ Response: {response}")

        except Exception as e:
            logger.info(f"Error indexing edge {from_node} -> {relationship} -> {to_node}: {e} ❌")
            error = f"Error indexing edge: {e}"

    return "Edges indexed successfully ✅" if not error.strip() else error


def index_chunks(client, index_name, text, Name, model_name, chunk_size=100):
    """Splits text into chunks, generates embeddings, and indexes them into OpenSearch."""
    chunks = chunk_text(text)
    actions = []
    model_id = f"{model_name}_{Name}"
    for i, chunk in enumerate(chunks):
        chunks = chunk
        try:
            embedding = get_text_embedding(chunks, 512)
            actions.append({
                "_index": index_name,
                "_source": {
                    "model_id": model_name,
                    "type": Name,
                    "chunk_id": i,
                    "content": chunk,
                    "embedding": embedding
                }
            })
        except Exception as e:
            logger.info(f"Skipping chunk {i} due to error: {e}")

    if actions:
        try:
            helpers.bulk(client, actions)
            logger.info(f"Indexed {len(actions)} chunks into {index_name}")
            return "Data chunked"
        except helpers.BulkIndexError as bulk_error:
            for error in bulk_error.errors:
                logger.info(f"Error indexing document: {error}")
            return None
    else:
        logger.info("No valid chunks to index.")


def get_addition_info_text(job_id, org_id):
    important_notes_path = f"import/important_notes/{org_id}/{job_id}/important_notes.txt"
    s3 = boto3.client("s3")
    try:
        s3.head_object(Bucket=BUCKET_NAME, Key=important_notes_path)
        response = s3.get_object(
            Bucket=BUCKET_NAME,
            Key=important_notes_path
        )
        file_data = response['Body'].read().decode('utf-8')
        file_content = file_data.replace("\n", "")
        return file_content
    except Exception as e:
        # If a 404 error is raised, the object does not exist
        print(f"Error getting important note file from s3: {e}")
        return "No file data"


def extract_node_info(text):
    """
    Extract node information from text.
    Format: Node Name (node): node details
    Returns a list of dictionaries with node_name and node_data keys.
    """
    # Define a pattern to match node sections - more flexible to handle various formats
    node_pattern = r"([A-Za-z0-9_\s]+)\s+\(node\):\s*(.*?)(?=\s*(?:[A-Za-z0-9_\s]+\s+\((?:node|'property')\)|\Z))"

    # Find all matches in the text
    matches = re.findall(node_pattern, text, re.DOTALL)

    # Process the matches into a list of dictionaries
    nodes_list = []
    for node_name, node_details in matches:
        node_name = node_name.strip()
        node_details = node_details.strip()

        # Skip if node_name is empty
        if not node_name:
            continue

        # Clean up node details
        node_details = clean_content(node_details)

        # Add to list as a dictionary with node_name and node_data keys
        nodes_list.append({
            "node_name": node_name,
            "node_data": node_details
        })

    return nodes_list


def check_label_in_nodes(nodes_list, label):
    """
    Check if a specific label is present in any of the node names
    Returns True if found, False otherwise
    """
    try:
        for node in nodes_list:
            if label.lower() in node["node_name"].lower():
                return True
        return False
    except Exception as e:
        print(f"Error checking label {label} in nodes: {e}")


def extract_all_non_node_data(text, nodes_list):
    """
    Extract all data that is not part of node definitions, including properties and other notes.
    Returns a dictionary with extracted data.
    """
    # First, create a clean copy of the text
    modified_text = text

    # Remove node sections from the text
    for node_obj in nodes_list:
        node_name = node_obj["node_name"]
        pattern = rf"{re.escape(node_name)}\s+\(node\):.*?(?=\s*(?:[A-Za-z0-9_\s]+\s+\((?:node|'property')\)|\Z))"
        for match in re.findall(pattern, modified_text, re.DOTALL):
            modified_text = modified_text.replace(match, "")

    # Clean the text
    modified_text = re.sub(r'\s*None\s*', ' ', modified_text).strip()

    notes_dict = {}

    # Extract properties with a more flexible pattern
    # This should handle property definitions regardless of placement and format
    property_pattern = r"([A-Za-z0-9_\s]+)\s+\('property'\):\s*(.*?)(?=\s*(?:[A-Za-z0-9_\s]+\s+\((?:node|'property')\)|\Z))"
    property_matches = re.findall(property_pattern, modified_text, re.DOTALL)

    # Process property matches
    for property_name, property_details in property_matches:
        property_name = property_name.strip()
        if not property_name:
            continue

        property_details = clean_content(property_details)
        notes_dict[f"PROPERTY_{property_name}"] = property_details

        # Remove this property from the modified text
        pattern = rf"{re.escape(property_name)}\s+\('property'\):.*?(?=\s*(?:[A-Za-z0-9_\s]+\s+\((?:node|'property')\)|\Z))"
        for match in re.findall(pattern, modified_text, re.DOTALL):
            modified_text = modified_text.replace(match, "")

    # Extract remaining structured notes (KEY: value format)
    note_pattern = r"([A-Z][A-Z0-9_\s]+):\s*(.*?)(?=\s*(?:[A-Z][A-Z0-9_\s]+:|\Z))"
    note_matches = re.findall(note_pattern, modified_text, re.DOTALL)

    # Process note matches
    for note_name, note_details in note_matches:
        note_name = note_name.strip()
        if not note_name:
            continue

        note_details = clean_content(note_details)
        if note_details:
            notes_dict[note_name] = note_details

        # Remove this note from the modified text
        pattern = rf"{re.escape(note_name)}:.*?(?=\s*(?:[A-Z][A-Z0-9_\s]+:|\Z))"
        for match in re.findall(pattern, modified_text, re.DOTALL):
            modified_text = modified_text.replace(match, "")

    # Any remaining text that doesn't fit the patterns above
    remaining_text = clean_content(modified_text)
    if remaining_text:
        notes_dict["ADDITIONAL_INFO"] = remaining_text

    return notes_dict


def clean_content(text):
    """Helper function to clean and normalize content."""
    if not text:
        return ""

    # Remove standalone "None" lines and clean whitespace
    lines = [line.strip() for line in text.split('\n') if line.strip() and line.strip() != "None"]
    result = '\n'.join(lines)

    # Remove trailing commas
    if result.endswith(','):
        result = result[:-1].strip()

    return result


def get_addition_info(info_result):
    """
    Process additional information from API result
    """
    try:
        # Handle different input types
        if isinstance(info_result, str):
            file_content = info_result
        elif isinstance(info_result, dict):
            # Try to extract string content from different typical keys
            for key in ['content', 'body', 'text', 'data']:
                if key in info_result and isinstance(info_result[key], str):
                    file_content = info_result[key]
                    break
            else:
                # If no string found, convert whole dict to string
                file_content = json.dumps(info_result)
        else:
            # Default to string conversion for other types
            file_content = str(info_result)

        # Process the text
        processed_text = fix_node_extraction(file_content)

        # Extract node information
        nodes_list = extract_node_info(processed_text)

        # Extract additional info/notes
        info_json = extract_all_non_node_data(processed_text, nodes_list)

        return info_json, nodes_list

    except Exception as e:
        logging.error(f"Error in get_addition_info: {str(e)}")
        return {}, []


def get_node_data_by_label(nodes_list, label):
    """
    Get node_data for a specific label from nodes_list
    Returns the node_data if found, None otherwise
    """
    if not label or not nodes_list:
        return None

    for node in nodes_list:
        if label.lower() in node["node_name"].lower():
            return node["node_data"]
    return None


def fix_node_extraction(text):
    """
    Pre-process the text to handle edge cases and ensure consistent formatting.
    """
    if not isinstance(text, str):
        logging.warning(f"Expected string in fix_node_extraction, got {type(text)}")
        try:
            text = str(text)
        except Exception as e:
            logging.error(f"Could not convert to string: {e}")
            return ""

    # Normalize line endings
    text = text.replace('\r\n', '\n').replace('\r', '\n')

    # Fix potential issues with 'None' values and inconsistent spacing
    fixed_text = re.sub(r'(node\):.*?)\n\s*None\s*\n', r'\1\n', text, flags=re.DOTALL)
    fixed_text = re.sub(r'\n\s*None\s*(?=\n)', '\n', fixed_text)

    # Ensure proper spacing around node and property declarations
    fixed_text = re.sub(r',\s*([A-Za-z0-9_\s]+)\s+\(', r',\n\1 (', fixed_text)

    return fixed_text


def get_pdf_from_s3(files):
    # Extract resourceList from the event
    resource_list = files
    results = []

    # Specify the S3 bucket name (you need to know this beforehand)
    bucket_name = 'preprod-cymonix-internal'
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

        # Return the extracted content or errors
        return results
    except Exception as e:
        logger.info(f"failed to get file from s3:{e}")


def read_json_from_s3(bucket, key):
    s3 = boto3.client("s3")
    try:
        response = s3.get_object(Bucket=bucket, Key=key)
        json_content = response["Body"].read().decode("utf-8")
        data = json.loads(json_content)
        return data
    except Exception as e:
        logger.info(f"Error reading JSON from S3: {e}")
        return None


def list_folders(prefix):
    """List folders inside a given prefix (only immediate subdirectories)."""
    result = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=prefix, Delimiter='/')
    return [content["Prefix"] for content in result.get("CommonPrefixes", [])]


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


def get_json_from_s3(org_id, job_id):
    """Retrieve and load JSON content from an S3 object."""
    file_key = f"import/multifile_schema/{org_id}/{job_id}/schema.json"
    try:
        response = s3.get_object(Bucket=BUCKET_NAME, Key=file_key)
        json_content = response["Body"].read().decode("utf-8")
        logger.info(f"json_data:-- {json_content}")
        return json.loads(json_content)
    except Exception as e:
        raise RuntimeError(f"Error reading JSON from {file_key}: {e}")


def lambda_handler(event, context):
    """
    Process a single model:
      - Load the JSON file from S3,
      - Index nodes, edges, schema note chunks, into the shared OpenSearch index.
    """
    try:
        for record in event['Records']:
            model_id = None
            key = record['s3']['object']['key']
            bucket_name = record['s3']['bucket']['name']

            # Extract job_id and org_id from the key
            job_id = key.split("/")[-2]
            org_id = key.split("/")[-3]

            logger.info(json.dumps({"log_type": "org_id", "value": org_id}))

            # OpenSearch index and model identifier
            graph_index = INDEX
            model_name = f"model_{org_id}_{job_id}"
            client = create_opensearch_client()

            # Create unified index if it doesn't exist
            if create_unified_graph_index(client, graph_index):
                logger.info(
                    f"Created unified index (or confirmed it exists), {create_unified_graph_index(client, graph_index)}")
            else:
                logger.info(f"Error creating unified index, {create_unified_graph_index(client, graph_index)}")

            json_data = get_json_from_s3(org_id, job_id)
            model_id = json_data['schema'].get('fromModel')
            files = json_data['schema'].get('resourceList')
            file_notes = get_pdf_from_s3(files)
            if file_notes:
                print(f"files data:-- {file_notes}")
            # Extract schema notes and node map from the JSON data
            schema_notes = json_data.get("schema", {}).get("notes", "No data")
            node_map, info_note = extract_nodes(json_data, job_id, org_id)
            if node_map:
                logger.info(f"Extracted node data for model {model_name}")
            else:
                logger.info(f"Error extracting data for model {model_name},")

            if info_note:
                schema_notes = str(info_note)
            else:
                schema_notes = json_data.get("schema", {}).get("notes", "No data")

            try:
                if check_query_index(graph_index, model_name, "node", client):
                    get_delete_id(graph_index, model_name, "node", client)
                    if index_nodes(client, node_map, graph_index, model_name):
                        logger.info(f"Mapped node data for model {model_name}")
                    else:
                        logger.info(f"Error mapping node data for model {model_name}")
                else:
                    if index_nodes(client, node_map, graph_index, model_name):
                        logger.info(f"Mapped node data for model {model_name}")
                    else:
                        logger.info(f"Error mapping node data for model {model_name}")
                if check_query_index(graph_index, model_name, "edge", client):
                    get_delete_id(graph_index, model_name, "edge", client)
                    if index_edges(client, json_data, graph_index, model_name):
                        logger.info(f"Mapped edge data for model {model_name}")
                    else:
                        logger.info(f"Error mapping edge data for model {model_name}")
                else:
                    if index_edges(client, json_data, graph_index, model_name):
                        logger.info(f"Mapped edge data for model {model_name}")
                    else:
                        logger.info(f"Error mapping edge data for model {model_name}")
            except Exception as e:
                logger.info(f"Error mapping nodes/edges for model {model_name}: {e}")

            # Index note chunks (assuming index_chunks accepts an additional 'model_name' argument)  # field name for note documents
            if schema_notes:
                if check_query_index(graph_index, model_name, "note", client):
                    get_delete_id(graph_index, model_name, "note", client)
                    if index_chunks(client, graph_index, schema_notes, "note", model_name):
                        logger.info(f"Mapped note data for model {model_name}")
                    else:
                        logger.info("No additional data to map")
                else:
                    if index_chunks(client, graph_index, schema_notes, "note", model_name):
                        logger.info(f"Mapped note data for model {model_name}")
                    else:
                        logger.info("No additional data to map")
            else:
                if check_query_index(graph_index, model_name, "note", client):
                    get_delete_id(graph_index, model_name, "note", client)
                    print("Deleted old Model and Property additional info")
                print("*No additional info Data*")

            if files:
                if check_query_index(graph_index, model_name, "file", client):
                    get_delete_id(graph_index, model_name, "file", client)
                else:
                    print("No file data to delete")
                if file_notes:
                    if check_query_index(graph_index, model_name, "file", client):
                        get_delete_id(graph_index, model_name, "file", client)
                        if index_chunks(client, graph_index, file_notes[0], "file", model_name):
                            logger.info(f"Mapped file data for model {model_name}")
                        else:
                            logger.info("No filse data to map")
                    else:
                        if index_chunks(client, graph_index, file_notes[0], "file", model_name):
                            logger.info(f"Mapped file data for model {model_name}")
                        else:
                            logger.info("No filse data to map")
                else:
                    print("No file note to load")
            else:
                if check_query_index(graph_index, model_name, "file", client):
                    get_delete_id(graph_index, model_name, "file", client)
                    print("Deleted old files")
                print("*No File Data*")

            logger.info(f"Processing completed for model {model_name}!")

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
        update_model({
            "path": "api/graph-model/",
            "data": {
                "id": model_id,
                "notes_updating": False
            }
        }, org_id)
        logger.info('Model updated')

