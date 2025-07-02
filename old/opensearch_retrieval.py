import json
import logging
from opensearchpy import OpenSearch, RequestsHttpConnection, ConnectionError, TransportError, helpers
# from sentence_transformers import SentenceTransformer
from requests_aws4auth import AWS4Auth
from opensearchpy import helpers, OpenSearch
import query_retrieve
import uuid
import boto3
import numpy as np
import importlib

importlib.reload(query_retrieve)

from query_retrieve import query_shared_node_index, search_similar_edges, query_shared_index, query_shared_index_like, check_query_index


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
        host = "https://3tv6hfs4qp9pwxybq9fa.us-west-2.aoss.amazonaws.com"
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
    
def generate_claude_response(prompt):
    client = boto3.client(service_name="bedrock-runtime", region_name='us-west-2')
    model_id = "anthropic.claude-3-haiku-20240307-v1:0"
    try:
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
    except Exception as e:
        print(f"Error generating claude response: {e}")
        return e

def get_edges(index, model_name, question_embedding, labels):
    relationships = []
    client = create_opensearch_client()
    try:
        if labels:
            for nodes in labels:
                node_relationships = search_similar_edges(index, model_name, question_embedding, nodes, client)
                relationships.extend(node_relationships)  # Add to the main list
                # logger.info(f"Relationships for Node {node_id},\n {node_relationships}")  # Corrected logging
        return relationships
    except Exception as e:
        print(f"Error getting edges: {e}")
        return e

def parse_json_nodes(json_data):
    # Initialize text parts
    node_labels = "Following are the Nodes Label:\n"
    properties = ""

    try:
        # Extract nodes
        nodes = [item for item in json_data if 'label' in item]

        for node in nodes:
            node_name = node.get("label")
            node_labels += f"`{node_name}`,\n"

            # Extract properties for each node
            node_properties = node.get("properties", {})
            node_note = node.get("note", "No description available.")

            property_text = f"Properties for Node `{node_name}` are:\n"
            property_text += f"**Note:** {node_note}\n"
            property_text += "$$Name$$ as type STRING"

            if node_properties:
                for key, value in node_properties.items():
                    key = key.lower().replace(" ", "_")
                    property_text += f",\n{key} as type {value}"

            property_text += ".\n\n"
            properties += property_text

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
        logger.info(f"Error in creating Text Schema Edges: {e}")
        return "error"
    
def embedd_question(question, model_name, type):
    final_question = f"{question} + {model_name}_{type}"
    question_embedding = get_text_embedding(final_question, 512)
    return question_embedding

def check_query_index(index, model_name, doc_type, client) -> bool:
    try:
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
    except Exception as e:
        logger.info(f"Error checking data in index for {doc_type}: {e}")
        return "error"

def retrieve_similer_nodes(question, json_nodes):
    with open("node_retrieve_prompt.txt", "r") as file:
            prompt = file.read().replace("$$question$$", question).replace("$$nodes$$",str(json_nodes))
    similer_nodes = generate_claude_response(prompt)
    return similer_nodes

def check_null(similer_nodes):
    labels = []
    if "None" in similer_nodes:
            similer_nodes = similer_nodes.replace("None", "null")
    data = json.loads(similer_nodes)
    for label in data:
        labels.append(label.get("label"))
    return labels

def generate_claude_response_aoss(prompt):
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

def query_response(job_id, files_data, matched_query, lmtn_prompt, question, Nodes, relationships, note_response):
    try:
        # Read and format important notes
        with open("default_notes.txt", "r") as file:
            important_note = file.read().replace("$$ID$$", job_id)

        # Concatenate schema elements
        context = Nodes + relationships + important_note + note_response

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
        return generate_claude_response_aoss(prompt)

    except FileNotFoundError as e:
        logger.info(f"File not found: {e}")
    except IOError as e:
        logger.info(f"File read/write error: {e}")
    except Exception as e:
        logger.info(f"Unexpected error: {e}")

    return None

def check_question_aoss(question, common_query)->bool:
    logger.info("Checking common question")
    try:
        # Extract question texts from common_query list
        question_texts = [item['question'] for item in common_query if 'question' in item]

        # Check if the given question exists in the extracted questions
        return bool(question in question_texts)
    except Exception as e:
        logger.info(f"Error: {e}")
        return None

def lambda_handler(event, context):

    start_time = t.time()
    event_params = transform_event(event)
    logger.info(f"event:- {event_params}")
    data = event_params['data']['message']
    job_id = data['job_id']
    org_id = data['org_id']

    if job_id == JOB_ID:
        index = "preprod_index"
        model_name = f"model_{org_id}_{job_id}"
        question = data['question']
        query_rag = None
        query = None
        input_tokens = 0
        output_tokens = 0

        client = create_opensearch_client()
        question_embed_node = embedd_question(question, model_name, "node")
        json_nodes = query_shared_node_index(index, model_name, question_embed_node, client)
        similer_nodes = retrieve_similer_nodes(question, json_nodes)
        labels = check_null(similer_nodes)
        question_embed_edge = embedd_question(question, model_name, "edge")
        json_edges = get_edges(index, model_name, question_embed_edge, labels)
        question_embed_note = embedd_question(question, model_name, "note")
        note_response = query_shared_index(index, model_name, "note", question_embed_note, client, 10)
        print(json.dumps(note_response, indent=2))
        schema = parse_json_nodes(data)
        relation = parse_json_edges(json_edges)
        schema_relation = schema + relation 
        print(schema_relation)

        try:
            with open("AWS_Neptune_OpenCypher_Restrictions.txt", 'r') as file:
                lmtn_prompt = file.read()
            if not check_query_index(index, model_name, 'like', client):
                logger.info("common question not exist, checking files")
                if check_query_index(index, model_name, 'file', client):
                    logger.info("Using File data")
                    question_embed_file = embedd_question(question, model_name, "file")
                    file_response = query_shared_index(index, model_name, "file", question_embed_file, client, 5)
                    print(json.dumps(file_response, indent=2))
                    query = query_response(job_id, file_response, None, lmtn_prompt, question, schema, relation, note_response)
                    logger.info("Got DATA from Files")
                else:
                    logger.info("No Like Query or files data")
                    query = query_response(job_id, None, None, lmtn_prompt, question, schema, relation, note_response)
                    logger.info("**No Liked query or files data**")

            else:
                # Question collection exists
                logger.info("Question collection exists")
                like_response = query_shared_index_like(index, model_name, "like", question, client, 1)
                print("got similar query:", json.dumps(like_response, indent=2))
                if check_question_aoss(question, like_response):
                    print("got query from common question")
                    query_rag = like_response[0]['query']
                    logger.info(f"got Liked query query")

                elif check_query_index(index, model_name, 'file', client):
                    logger.info("Using File data")
                    question_embed_file = embedd_question(question, model_name, "file")
                    file_response = query_shared_index(index, model_name, "file", question_embed_file, client, 5)
                    print(json.dumps(file_response, indent=2))
                    query = query_response(job_id, file_response, None, lmtn_prompt, question, schema, relation, note_response)
                    logger.info("**Got DATA from Files**")
                else:
                    logger.info("No Like Query or files data")
                    query = query_response(job_id, None, None, lmtn_prompt, question, schema, relation, note_response)
                    logger.info("**No Liked query or files data**")

            if "MATCH" or "RETURN" in query:
                logger.info("got query")
                # output_tokens = calculate_tokens(response)
                match_queries = re.findall(r'```cypher(.*?)```', query, re.DOTALL)
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
            logger.info(f"variable_query:- {variable_query}")
            # output_2 = calculate_tokens(variable_query)
            slash_prompt = get_prompt(variable_query, prompt_file='check_back_slash.txt')
            slash_query = generate_response(slash_prompt)
            logger.info(f"slash_query:- {slash_query}")
            # checking job_id
            job_id_prompt = get_prompt_query(slash_query, job_id, prompt_file='check_job_id.txt')
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