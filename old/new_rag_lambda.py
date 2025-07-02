import json
import boto3
import os
import re
import traceback
import logging
import time as t
import uuid
from error_helper import sqs_helper
import tiktoken
from datetime import datetime
from requests_aws4auth import AWS4Auth
from opensearchpy import OpenSearch, RequestsHttpConnection, ConnectionError, TransportError, helpers

logger = logging.getLogger()
logger.setLevel(logging.INFO)

env = os.getenv('ENV')
DB_HOST = os.getenv('DB_HOST')
DB_PORT = os.getenv('DB_PORT')
MODEL_ID = os.getenv('MODEL_ID')
MODEL_ID_HQ = os.getenv('MODEL_ID_HQ')
REGION = os.getenv('AWS_REGION')

bedrock_runtime = boto3.client(service_name='bedrock-runtime', region_name='us-west-2')

def transform_event(event):  # Read event from event-bridge
    try:
        event_params = event['Records'][0]["body"]
        event_params = json.loads(event_params)
        return event_params
    except Exception as e:
        raise e

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

# --- Query OpenSearch with a Question ---
def search_relevant_node(question, NODE_INDEX):

    try:
        embedding = get_text_embedding(question, dimensions=512)
        # print(embedding)
        client = create_opensearch_client()
        query = {
            "size": 4,  # Get top 3 most relevant nodes
            "query": {
                "knn": {
                    "embedding": {
                        "vector": embedding,
                        "k": 5
                    }
                }
            }
        }

        response = client.search(index=NODE_INDEX, body=query)

        if "hits" in response:
            results = response["hits"]["hits"]
            # print(results)
            return [{"node_id": hit["_source"]["node_id"], "label": hit["_source"]["label"], "note": hit["_source"]["note"] ,"properties": hit["_source"]["properties"], "score": hit["_score"]} for hit in results]

        return []
    except Exception as e:
        print(f"Error in searchng relevant nodes: {e}")
        return "error"
    
def search_similar_edges(query_text, EDGE_INDEX, top_k=2):
    """Retrieve edges with similar embeddings from OpenSearch."""
    try:
        # Generate embedding for the query text
        query_embedding = get_text_embedding(query_text, dimensions=512)
        client = create_opensearch_client()
        # OpenSearch k-NN query (remove num_candidates)
        query = {
            "size": top_k,
            "query": {
                "knn": {
                    "embedding": {
                        "vector": query_embedding,
                        "k": top_k
                    }
                }
            }
        }

        response = client.search(index=EDGE_INDEX, body=query)
        # print(response)
        # Extract results
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
    
    except Exception as e:
        print(f"Error in searchng relevant edges: {e}")
        return "error"
    
def get_prompt(result, prompt_file):
    with open(prompt_file, 'r') as file:
        prompt = file.read().replace("$$QUERY$$", result)
    return prompt

def get_prompt_query(result, value, prompt_file):
    with open(prompt_file, 'r') as file:
        prompt = file.read().replace("$$QUERY$$", result).replace("$$VALUE$$", value)
    return prompt

def search_similar_chunks(client, index_name, query, top_k=9):
    """Finds the most relevant schema chunks for a given query using k-NN search."""
    try:
        query_embedding = get_text_embedding(query, dimensions=512)

        search_body = {
            "size": top_k,
            "query": {
                "knn": {
                    "embedding": {
                        "vector": query_embedding,
                        "k": top_k
                   }
                }
            }
        }

        response = client.search(index=index_name, body=search_body)
        results = [(hit["_source"]["content"], hit["_score"]) for hit in response["hits"]["hits"]]

        return results
    except Exception as e:
        print(f"Error in searchng relevant chunks: {e}")
        return "error"
    
def search_relevant_query(question, LIKED_INDEX):

    try:
        embedding = get_text_embedding(question, dimensions=512)
        # print(embedding)
        client = create_opensearch_client()
        query = {
            "size": 1,  # Get top 3 most relevant nodes
            "query": {
                "knn": {
                    "embedding": {
                        "vector": embedding,
                        "k": 1
                    }
                }
            }
        }

        response = client.search(index=LIKED_INDEX, body=query)

        if "hits" in response:
            results = response["hits"]["hits"]
            # print(results)
            return [{"question": hit["_source"]["question"], "query": hit["_source"]["query"], "score": hit["_score"]} for hit in results]

        return []
    except Exception as e:
        print(f"Error in searchng relevant nodes: {e}")
        return "error"

def generate_response(prompt):
    client = boto3.client(service_name="bedrock-runtime", region_name=REGION)
    model_id = MODEL_ID_HQ

    response = client.invoke_model(
        modelId=model_id,
        body=json.dumps(
            {
                "anthropic_version": "bedrock-2023-05-31", "max_tokens": 1500, "top_k": 250, "temperature": 1,
                "top_p": 0.999,
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": prompt}],
                    }
                ],
            }
        ),
    )

    response_body = json.loads(response["body"].read())
    completion = response_body['content'][0]['text']
    return completion

def parse_json_nodes(json_data):
    # Initialize text parts
    ID = 'job_id'
    node_labels = "Following are the Nodes Label:\n"
    properties = ""
    
    try:
        # Extract nodes and edges
        nodes = [item for item in json_data if 'node_id' in item]

        # Extract node names and properties
        node_dict = {}
        for node in nodes:
            node_name = node.get("label")
            node_labels += f"`{node_name}`,\n"

            # Extract properties for each node
            node_properties = node.get("properties", {})
            property_text = f"Properties for Node `{node_name}` are:\n"

            # Always include creationDate and Name properties
            property_text += "$$Name$$ as type STRING"

            if node_properties:
                for key, value in node_properties.items():
                    # Convert key to lowercase and replace spaces with underscores
                    key = key.lower().replace(" ", "_")
                    property_text += f",\n{key} as type {value}"

            property_text += ".\n\n"
            properties += property_text
            node_dict[node.get("node_id")] = node_name

        # Combine all parts into final output
        output_text = f"\n{node_labels}\n{properties}"

        return output_text
    except Exception as e:
        print(f"Error in creating Text Schema Nodes: {e}")
        return "error"
    
def parse_json_edges(json_data):
    # Initialize text parts
    edge_labels = "Following are the Edges Label:\n"
    relationships = "Relationships are defined as follows:\n"

    # Extract unique edges
    edge_dict = set()  
    
    try:
        for edge in json_data:
            edge_name = edge.get("relationship")
            if edge_name:
                edge_dict.add(edge_name)

        # Format edge labels
        for edge in sorted(edge_dict):
            edge_labels += f"`{edge}`,\n"

        # Extract relationships correctly
        for edge in json_data:
            edge_name = edge.get("relationship")
            from_node = edge.get("from")
            to_node = edge.get("to")
            relationships += f"Node `{from_node}` is connected to Node `{to_node}` via Edge `{edge_name}`,\n"

        # Combine all parts into final output
        output_text = f"{edge_labels}\n{relationships}"

        return output_text
    except Exception as e:
        print(f"Error in creating Text Schema Edges: {e}")
        return "error"
    
def generate_claude_response(prompt):
    client = boto3.client(service_name="bedrock-runtime", region_name='us-west-2')
    model_id = MODEL_ID
    response = client.invoke_model(modelId=model_id, body=json.dumps(
        {
            "anthropic_version": "bedrock-2023-05-31", "max_tokens": 8196,
            "messages": [
                {
                    "role": "user", "content": [{"type": "text", "text": prompt}],
                }
            ],
        }), )
    result = json.loads(response.get("body").read())
    answer = str(result['content'][0]['text'])
    return answer

def check_question(question, common_query):
    try:
        if isinstance(common_query, list) and all(isinstance(item, dict) for item in common_query):
            return any(question.strip().lower() == item.get("question", "").strip().lower() for item in common_query)
        return False
    except Exception as e:
        print(f"Error: {e}")
        return None

def get_edges(org_id, job_id, similar_data_nodes):
    relationships = []
    EDGE_INDEX = f'edge_index_{org_id}_{job_id}'
    if similar_data_nodes:
        for nodes in similar_data_nodes:
            node_id = nodes["node_id"]  # Get the most relevant node
            node_relationships = search_similar_edges(node_id, EDGE_INDEX)
            relationships.extend(node_relationships)  # Add to the main list
            logger.info(f"Relationships for Node {node_id}:", node_relationships)
    return relationships

def get_files_data(results):
    similar_data = []
    # print("\nTop Relevant Chunks:")
    for content, score in results:
        # print(f"Score: {score:.4f}\n{content}\n")
        similar_data.append(content)
    return similar_data

def query_response(job_id, files_data, matched_query, lmtn_prompt, question, Nodes, relationships):
    try:
        # Read and format important notes
        with open("imp_note.txt", "r") as file:
            important_note = file.read().replace("$$ID$$", job_id)
        
        # Concatenate schema elements
        context = Nodes + relationships + important_note
        
        # Read and format prompt
        if matched_query:
            with open("prompt.txt", "r") as file:
                prompt = file.read()
                prompt = prompt.replace("$$ID$$", job_id)
                prompt = prompt.replace("$$Limitation$$", lmtn_prompt)
                prompt = prompt.replace("{context}", context)
                prompt = prompt.replace("{question}", question)
                prompt = prompt.replace("$$common_query$$", matched_query) 

        elif files_data:
            with open("prompt.txt", "r") as file:
                prompt = file.read()
                prompt = prompt.replace("$$ID$$", job_id)
                prompt = prompt.replace("$$Limitation$$", lmtn_prompt)
                prompt = prompt.replace("{context}", context)
                prompt = prompt.replace("{question}", question)
                prompt = prompt.replace("$$file_imp_data$$", str(files_data))
        
        else:
            with open("prompt.txt", "r") as file:
                prompt = file.read()
                prompt = prompt.replace("$$ID$$", job_id)
                prompt = prompt.replace("$$Limitation$$", lmtn_prompt)
                prompt = prompt.replace("{context}", context)
                prompt = prompt.replace("{question}", question)

        
        # Generate response
        return generate_claude_response(prompt)
    
    except FileNotFoundError as e:
        print(f"File not found: {e}")
    except IOError as e:
        print(f"File read/write error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
    
    return None

def send_message_to_event_bridge(message):
    client = boto3.client('events')
    try:
        # Send event to EventBridge
        response = client.put_events(
            Entries=[
                {
                    'Source': f'{env}-bedrock',
                    'DetailType': 'event trigger for bedrock',
                    'Detail': json.dumps(message),
                    'EventBusName': f'{env}-bedrock-bus'
                },
            ]
        )
        # Check if the event was successfully sent
        if response['FailedEntryCount'] > 0:
            # Log or handle failure
            return {
                'statusCode': 500,
                'body': 'Failed to send message to EventBridge',
                'details': response
            }

        return {
            'statusCode': 200,
            'body': 'Message sent to EventBridge successfully',
            'details': response
        }

    except Exception as e:
        # Handle and return the exception message
        return {
            'statusCode': 500,
            'body': f'Error sending message to EventBridge: {str(e)}'
        }

def send_message_to_websocket_dispatcher(message_body):
    sqs_client = boto3.client('sqs')
    queue_url = os.environ.get('DISPATCHER_QUEUE_URL')
    try:
        message = {
            "target": [message_body['ConnectionID']],
            "data": {
                "message": {'q_id': message_body['q_id'], 'answer': message_body['query'], 'statuscode': 200},
                'type': 6
            }
        }
        sqs_client.send_message(
            QueueUrl=queue_url,
            MessageBody=json.dumps(message, default=str),
            MessageDeduplicationId=str(uuid.uuid4()),
            MessageGroupId=str(uuid.uuid4())
        )
    except Exception as e:
        logger.info(f"An error occurred on dispatcher: {str(e)}")
        raise e

def get_message(event_params, start_time, queries):
    connection_id = event_params['target'][0]
    data = event_params['data']['message']
    try:
        if "q_id" in data:
            q_id = data['q_id']
        else:
            q_id = str(uuid.uuid4())
        types = '1'
        if "type" in data.keys():
            types = data['type']

        query = 'None'
        if types == '1':
            query = queries
        # logger.info("queries", query)
        # Build the message
        message = {
            'question': data['question'],
            'query': query,
            'ConnectionID': connection_id,
            'q_id': q_id,
            'type': types,
            'job_id': data['job_id'],
            'org_id': data['org_id'],
            'start_time': start_time
        }

        return message

    except KeyError as e:
        logger.info(f"KeyError: Missing key {str(e)} in input data")
        raise
    except Exception as e:
        logger.info(f"An error occurred: {str(e)}")
        raise

def lambda_handler(event, context):
    """Lambda handler function for processing the request and returning the response."""
    start_time = t.time()
    event_params = transform_event(event)
    logger.info(f"event:- {event_params}")
    data = event_params['data']['message']
    job_id = data['job_id']
    org_id = data['org_id']
    question = data['question']
    # OpenSearch indices
    NODE_INDEX = f'node_index_{org_id}_{job_id}'
    EDGE_INDEX = f'edge_index_{org_id}_{job_id}'
    NOTE_INDEX = f'note_index_{org_id}_{job_id}'
    FILES_INDEX = f'file_index_{org_id}_{job_id}'
    LIKED_INDEX = f'liked_index_{org_id}_{job_id}'
    embed_data_question = None
    embed_data_file = None
    query_rag = None
    input_tokens = 0
    output_tokens = 0
    client = create_opensearch_client()
    with open("AWS_Neptune_OpenCypher_Restrictions.txt", 'r') as file:
        lmtn_prompt = file.read()

    json_nodes = search_relevant_node(question, NODE_INDEX)
    json_edges = get_edges(org_id, job_id, json_nodes)
    text_nodes = parse_json_nodes(json_nodes)
    text_edges = parse_json_edges(json_edges)
    try:
        if not client.indices.exists(index=LIKED_INDEX):
            logger.info("common question not exist, checking files")
            # Fallback to file data if question collection does not exist
            if client.indices.exists(index=FILES_INDEX):
                logger.info("Using File data")
                results = search_similar_chunks(client, FILES_INDEX , question)
                files_data = get_files_data(results)
                query = query_response(job_id, files_data, None, lmtn_prompt, question, text_nodes, text_edges)
                logger.info("Got DATA from Files")
            else:
                logger.info("No Like Query or files data")
                query = query_response(job_id, None, None, lmtn_prompt, question, text_nodes, text_edges)
                logger.info("**No Liked query or files data**")
        else:
            # Question collection exists
            logger.info("Question collection exists")
            similar_query = search_relevant_node(question, LIKED_INDEX)
            if check_question(question, embed_data_question):
                if similar_query:
                    query_rag = similar_query[0]['query']
                    logger.info(f"got Liked query query")
            
            elif client.indices.exists(index=FILES_INDEX):
                logger.info("Using File data")
                results = search_similar_chunks(client, FILES_INDEX , question)
                files_data = get_files_data(results)
                if similar_query:
                    query_rag = similar_query[0]['query']
                    logger.info(f"got Liked query query")
                query = query_response(job_id, files_data, query_rag, lmtn_prompt, question, text_nodes, text_edges)
                logger.info("**Got DATA from Files**")

            else:
                logger.info("No Like Query or files data")
                if similar_query:
                    query_rag = similar_query[0]['query']
                    logger.info(f"got Liked query query")
                query = query_response(job_id, None, query_rag, lmtn_prompt, question, text_nodes, text_edges)
                logger.info("**No Liked query or files data**")

        if 'response' in locals() and query and 'result' in query:
            logger.info("got response")
            # output_tokens = calculate_tokens(response)
            match_queries = re.findall(r'```cypher(.*?)```', query['result'], re.DOTALL)
            query_rag = query_rag or "".join(match_queries)
        else:
            logger.warning("Response object is missing or malformed.")
        if not query_rag:
            return {
                'statusCode': 400,
                'body': 'No Cypher queries found in the response.'
            }
        if input_tokens:
            logger.info(f"input_token:- {input_tokens}")

        logger.info(f"cypher_query:- {query_rag}")

        backtick_prompt = get_prompt(query_rag, prompt_file='check_backtick_prompt.txt')
        # input_1 = calculate_tokens(backtick_prompt)
        backtick_query = generate_response(backtick_prompt)
        # output_1 = calculate_tokens(backtick_query)
        logger.info(f"backtick_query:- {backtick_query}")
        # checking edge variable
        variable_prompt = get_prompt(backtick_query, prompt_file='check_edge_variable.txt')
        # input_2 = calculate_tokens(variable_prompt)
        variable_query = generate_response(variable_prompt)
        # output_2 = calculate_tokens(variable_query)
        logger.info(f"variable_query:- {variable_query}")
        # checking job_id
        job_id_prompt = get_prompt_query(variable_query, job_id, prompt_file='check_job_id.txt')
        # input_3 = calculate_tokens(job_id_prompt)
        job_id_query = generate_response(job_id_prompt)
        # output_3 = calculate_tokens(job_id_query)
        logger.info(f"job_id_query:- {job_id_query}")
        #  adding org_id
        org_id_prompt = get_prompt_query(job_id_query, org_id, prompt_file='check_org_id.txt')
        # input_4 = calculate_tokens(org_id_prompt)
        query = generate_response(org_id_prompt)
        # output_4 = calculate_tokens(query)
        logger.info(f"org_id_query:- {query}")

        logger.info('query sent to Neptune lambda')
        change_job_id_prompt = get_prompt(query, prompt_file='change_jobid.txt')
        change_job_id_query = generate_response(change_job_id_prompt)
        logger.info(f"change_job_id:- {change_job_id_query}")
        job_message = get_message(event_params, start_time, change_job_id_query)
        send_message_to_websocket_dispatcher(job_message)
        message = get_message(event_params, start_time, query)
        event_bridge_response = send_message_to_event_bridge(message)

        logger.info(json.dumps({"log_type": "sqs", "value": org_id}))
        logger.info(json.dumps({"log_type": "org_id", "value": org_id}))

        # Check if the message was successfully sent
        if event_bridge_response.get('statusCode') == 200:
            logger.info("Message sent to EventBridge successfully!")
        else:
            logger.info(f"Failed to send message to EventBridge: {event_bridge_response}")

        return {
            'statusCode': 200,
            'body': 'Message Sent'
        }

    except Exception as e:
        trace_back = traceback.format_exc()
        sqs_helper.queue_message(context.function_name, e, trace_back)
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': str(e),
                'traceback': trace_back
            })
        }


        