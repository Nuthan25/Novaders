import json
import boto3
import os
import re
import traceback
import logging
from langchain.embeddings import BedrockEmbeddings
from langchain.vectorstores.pgvector import PGVector
from langchain.docstore.document import Document
from langchain_community.chat_models import BedrockChat
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_aws import ChatBedrockConverse
import psycopg2
import time as t
import uuid
from html import escape
from error_helper import sqs_helper
import tiktoken
from datetime import datetime
from requests_aws4auth import AWS4Auth
from opensearchpy import OpenSearch, RequestsHttpConnection, ConnectionError, TransportError, helpers

from datetime import datetime

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Constants
env = os.getenv('ENV')
DB_HOST = os.getenv('DB_HOST')
DB_USER = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASSWORD')
DB_PORT = os.getenv('DB_PORT')
MODEL_ID = os.getenv('MODEL_ID')
MODEL_ID_HQ = os.getenv('MODEL_ID_HQ')
REGION = os.getenv('AWS_REGION')
JOB_ID = os.getenv('JOB_ID')
DB_HOST_AOSS = os.getenv('DB_HOST_AOSS')
DB_PORT_AOSS = os.getenv('DB_PORT_AOSS')


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
    bedrock_runtime = boto3.client(service_name='bedrock-runtime', region_name=REGION)
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
            return [
                {"node_id": hit["_source"]["node_id"], "label": hit["_source"]["label"], "note": hit["_source"]["note"],
                 "properties": hit["_source"]["properties"], "score": hit["_score"]} for hit in results]

        return []
    except Exception as e:
        logger.info(f"Error in searchng relevant nodes: {e}")
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
        logger.info(f"Error in searchng relevant edges: {e}")
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
        logger.info(f"Error in searchng relevant chunks: {e}")
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
            return [{"question": hit["_source"]["question"], "query": hit["_source"]["query"], "score": hit["_score"]}
                    for hit in results]

        return []
    except Exception as e:
        logger.info(f"Error in searchng relevant nodes: {e}")
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
    node_labels = "Following are the Nodes Label:\n"
    properties = ""

    try:
        # Extract nodes
        nodes = [item for item in json_data if 'node_id' in item]

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


def check_question(question, common_query):
    try:
        if isinstance(common_query, list) and all(isinstance(item, dict) for item in common_query):
            return any(question.strip().lower() == item.get("question", "").strip().lower() for item in common_query)
        return False
    except Exception as e:
        logger.info(f"Error: {e}")
        return None


def get_edges(similar_data_nodes, EDGE_INDEX):
    relationships = []
    if similar_data_nodes:
        for nodes in similar_data_nodes:
            node_id = nodes["node_id"]  # Get the most relevant node
            node_relationships = search_similar_edges(node_id, EDGE_INDEX)
            relationships.extend(node_relationships)  # Add to the main list
            logger.info(f"Relationships for Node {node_id},\n {node_relationships}")  # Corrected logging
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
        with open("default_notes.txt", "r") as file:
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
        return generate_claude_response_aoss(prompt)

    except FileNotFoundError as e:
        logger.info(f"File not found: {e}")
    except IOError as e:
        logger.info(f"File read/write error: {e}")
    except Exception as e:
        logger.info(f"Unexpected error: {e}")

    return None


# def send_message_to_event_bridge(message):
#     client = boto3.client('events')
#     try:
#         # Send event to EventBridge
#         response = client.put_events(
#             Entries=[
#                 {
#                     'Source': f'{env}-bedrock',
#                     'DetailType': 'event trigger for bedrock',
#                     'Detail': json.dumps(message),
#                     'EventBusName': f'{env}-bedrock-bus'
#                 },
#             ]
#         )
#         # Check if the event was successfully sent
#         if response['FailedEntryCount'] > 0:
#             # Log or handle failure
#             return {
#                 'statusCode': 500,
#                 'body': 'Failed to send message to EventBridge',
#                 'details': response
#             }
#
#         return {
#             'statusCode': 200,
#             'body': 'Message sent to EventBridge successfully',
#             'details': response
#         }
#
#     except Exception as e:
#         # Handle and return the exception message
#         return {
#             'statusCode': 500,
#             'body': f'Error sending message to EventBridge: {str(e)}'
#         }


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

def calculate_tokens(text):
    try:
        # encoder = tiktoken.get_encoding("gpt2")
        # token_count = len(encoder.encode(str(text)))
        return 0
    except Exception as e:
        logger.info(f"error while calculating tokens {e}")
        return None

# def transform_event(event):  # Read event from event-bridge
#     try:
#         event_params = event['Records'][0]["body"]
#         event_params = json.loads(event_params)
#         return event_params
#     except Exception as e:
#         raise e


def cred(db_name):
    """Generates a PostgreSQL connection string."""
    try:
        connection_str = PGVector.connection_string_from_db_params(
            driver='psycopg2',
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT,
            database=db_name,
        )
        logger.info(f"Connection string generated: {connection_str}")
        return connection_str
    except Exception as e:
        logger.info(f"Error generating connection string: {str(e)}")
        raise e


def check_collection_exists(database, collection_name: str) -> bool:
    # Establish connection to the PostgreSQL database
    connection = psycopg2.connect(
        host=DB_HOST,
        database=database,  # Connect to the specified database
        user=DB_USER,
        password=DB_PASSWORD,
        port=DB_PORT,
    )
    try:
        with connection:
            with connection.cursor() as cursor:
                # Example query to check collection existence
                query = """
                SELECT EXISTS(
                    SELECT 1
                    FROM langchain_pg_collection
                    WHERE name = %s
                );
                """
                cursor.execute(query, (collection_name,))
                exists = cursor.fetchone()[0]
        return exists
    except Exception as e:
        logger.info(f"Error checking collection existence: {e}")
        return False


def create_or_get_db_inst(database, collection_name, schema_data, use_existing=True):
    # Initialize the text embedding model
    embeddings = BedrockEmbeddings()
    CONNECTION_STRING = cred(database)

    if use_existing:
        # Check if the collection exists
        if check_collection_exists(database, collection_name):
            # Get an existing PGVector instance
            vector_store = PGVector.from_existing_index(
                embedding=embeddings,
                collection_name=collection_name,
                connection_string=CONNECTION_STRING
            )
        else:
            # Collection does not exist; create a new vector store
            vector_store = PGVector.from_documents(
                documents=schema_data,
                embedding=embeddings,
                collection_name=collection_name,
                connection_string=CONNECTION_STRING
            )
    else:
        # Create a new PGVector instance and populate it with document data and embeddings
        vector_store = PGVector.from_documents(
            documents=schema_data,
            embedding=embeddings,
            collection_name=collection_name,
            connection_string=CONNECTION_STRING
        )

    return vector_store

def calculate_prompt_token(embedings, question, job_id, matched_query, imp_data, lmtn_prompt, prompt_file='prompt.txt'):
    try:
        with open(prompt_file, 'r') as file:
            prompt = file.read()

        # Replace common placeholders
        prompt = prompt.replace("$$ID$$", 'job_id').replace("$$VALUE$$", str(job_id)).replace("$$Limitation$$", str(lmtn_prompt))
        prompt = prompt.replace("{context}", str(embedings)).replace("{question}", str(question))

        # Replace matched_query if available
        if matched_query:
            prompt = prompt.replace("$$common_query$$", str(matched_query))
        else:
            prompt = prompt.replace("$$common_query$$", "")

        # Replace imp_data if available
        if imp_data:
            imp_data_str = ", ".join(map(str, imp_data)) if isinstance(imp_data, list) else str(imp_data)
            prompt = prompt.replace("$$file_imp_data$$", str(imp_data_str))
        else:
            prompt = prompt.replace("$$file_imp_data$$", "")

        input_token = calculate_tokens(str(prompt))
        return input_token
    except Exception as e:
        logger.info(f"error while calculating prompt token {e}")
        return None  # Return None to indicate failure


def get_db_inst(database, collection_name, use_existing=True):
    # Initialize the text embedding model
    embeddings = BedrockEmbeddings()
    CONNECTION_STRING = cred(database)
    try:
        if use_existing:
            # Check if the collection exists
            vector = PGVector.from_existing_index(
                embedding=embeddings,
                collection_name=collection_name,
                connection_string=CONNECTION_STRING
            )
            logger.info(f"Collection exist:- {collection_name}")
            return vector
    except Exception as e:
        logger.info(f"Collection does not exist:- {collection_name}")


def create_chain(db_instance, question, job_id, matched_query, imp_data, lmtn_prompt, prompt_file='prompt.txt'):
    """Sets up the RetrievalQA chain with the given prompt and database instance."""
    client = boto3.client(service_name="bedrock-runtime", region_name=REGION)
    input_token = 0

    matched_query_safe = str(matched_query).replace("{", "{{").replace("}", "}}")
    imp_data_safe = str(imp_data).replace("{", "{{").replace("}", "}}")

    try:
        if matched_query_safe:
            with open(prompt_file, 'r') as file:
                prompt = file.read().replace("$$ID$$", 'job_id').replace("$$VALUE$$", str(job_id)).replace(
                    "$$common_query$$", str(matched_query_safe)).replace("$$Limitation$$", str(lmtn_prompt))
                # input_token = calculate_tokens(prompt)
        elif imp_data:
            with open(prompt_file, 'r') as file:
                prompt = file.read().replace("$$ID$$", 'job_id').replace("$$VALUE$$", str(job_id)).replace(
                    "$$common_query$$", str(matched_query_safe)).replace("$$file_imp_data$$", str(imp_data_safe)).replace(
                    "$$Limitation$$", str(lmtn_prompt))
                # input_token = calculate_tokens(prompt)
        else:
            with open(prompt_file, 'r') as file:
                prompt = file.read().replace("$$ID$$", 'job_id').replace("$$VALUE$$", str(job_id)).replace("$$Limitation$$",
                                                                                                      str(lmtn_prompt))
                # input_token = calculate_tokens(prompt)

        # Initialize LLM
        llm = ChatBedrockConverse(
            client=client,
            model_id= MODEL_ID
        )

        # Set up prompt template
        # prompt_template = PromptTemplate(
        #     template=prompt,
        #     input_variables=["context", "question"]
        # )
        prompt_template = PromptTemplate.from_template(prompt)
        prompt_template.input_variables = ["context", "question"]  # force override

        # Create and return the retrieval chain
        return RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            chain_type_kwargs={"prompt": prompt_template},
            retriever=db_instance.as_retriever(search_type="mmr", search_kwargs={'k': 9, 'lambda_mult': 0.5}),
        )

    except FileNotFoundError:
        raise Exception(f"Prompt file {prompt_file} not found.")
    except Exception as e:
        raise Exception(f"An error occurred while creating the RetrievalQA chain: {e}")


# def get_message(event_params, start_time, queries):
#     connection_id = event_params['target'][0]
#     data = event_params['data']['message']
#     try:
#         if "q_id" in data:
#             q_id = data['q_id']
#         else:
#             q_id = str(uuid.uuid4())
#         types = '1'
#         if "type" in data.keys():
#             types = data['type']
#
#         query = 'None'
#         if types == '1':
#             query = queries
#         # logger.info("queries", query)
#         # Build the message
#         message = {
#             'question': data['question'],
#             'query': query,
#             'ConnectionID': connection_id,
#             'q_id': q_id,
#             'type': types,
#             'job_id': data['job_id'],
#             'org_id': data['org_id'],
#             'start_time': start_time
#         }
#
#         return message
#
#     except KeyError as e:
#         logger.info(f"KeyError: Missing key {str(e)} in input data")
#         raise
#     except Exception as e:
#         logger.info(f"An error occurred: {str(e)}")
#         raise


def get_embedding_data(question, db_instance):
    try:
        if db_instance:
            retriever = db_instance.as_retriever(search_type="mmr", search_kwargs={'k': 1, 'lambda_mult': 0.9})
            matched_docs = retriever.get_relevant_documents(query=question)
            common_query = [doc.page_content for doc in matched_docs]
            return common_query
        else:
            return None
    except Exception as e:
        logger.info(f"An error occurred: {str(e)}")
        raise


def get_embedding_schema_data(question, db_instance):
    try:
        if db_instance:
            retriever = db_instance.as_retriever(search_type="mmr", search_kwargs={'k': 9, 'lambda_mult': 0.9})
            matched_docs = retriever.get_relevant_documents(query=question)
            common_query = [doc.page_content for doc in matched_docs]
            print("common_query:--", common_query)
            return common_query
        else:
            return None
    except Exception as e:
        logger.info(f"An error occurred: {str(e)}")
        raise


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


# def send_message_to_websocket_dispatcher(message_body):
#     sqs_client = boto3.client('sqs')
#     queue_url = os.environ.get('DISPATCHER_QUEUE_URL')
#     try:
#         message = {
#             "target": [message_body['ConnectionID']],
#             "data": {
#                 "message": {'q_id': message_body['q_id'], 'answer': message_body['query'], 'statuscode': 200},
#                 'type': 6
#             }
#         }
#         sqs_client.send_message(
#             QueueUrl=queue_url,
#             MessageBody=json.dumps(message, default=str),
#             MessageDeduplicationId=str(uuid.uuid4()),
#             MessageGroupId=str(uuid.uuid4())
#         )
#     except Exception as e:
#         logger.info(f"An error occurred on dispatcher: {str(e)}")
#         raise e


def get_common_query(question, common_query):
    try:
        # Combine the common_query into a single string
        result = "".join(common_query) if isinstance(common_query, list) else common_query
        logger.info(f"Combined query data (repr): {repr(result)}")

        # Normalize line endings and strip whitespace
        result = result.replace('\r\n', '\n').strip()

        # Locate the question and query markers
        question_marker = f"QUESTION: {question}"
        query_marker = "QUERY:"

        if question_marker in result and query_marker in result:
            # Find start and end indices
            question_start = result.index(question_marker) + len(question_marker)
            query_start = result.index(query_marker, question_start) + len(query_marker)

            # Extract the query until the next "Good Query:" or end of string
            query_end = result.find("Good Query:", query_start)
            query_end = query_end if query_end != -1 else len(result)

            # Extract and clean the query
            query = result[query_start:query_end].strip()
            logger.info(f"Extracted query: {query}")
            return query
        else:
            logger.warning("Markers not found in the provided data.")
            return None
    except Exception as e:
        logger.error(f"Error in get_common_query: {e}")
        return None


def check_question_aoss(question, common_query):
    logger.info("Checking common question")
    try:
        # Extract question texts from common_query list
        question_texts = [item['question'] for item in common_query if 'question' in item]

        # Check if the given question exists in the extracted questions
        return question in question_texts
    except Exception as e:
        logger.info(f"Error: {e}")
        return None


# def generate_response(prompt):
#     client = boto3.client(service_name="bedrock-runtime", region_name=REGION)
#     model_id = MODEL_ID_HQ
#
#     response = client.invoke_model(
#         modelId=model_id,
#         body=json.dumps(
#             {
#                 "anthropic_version": "bedrock-2023-05-31", "max_tokens": 1500, "top_k": 250, "temperature": 1,
#                 "top_p": 0.999,
#                 "messages": [
#                     {
#                         "role": "user",
#                         "content": [{"type": "text", "text": prompt}],
#                     }
#                 ],
#             }
#         ),
#     )
#
#     response_body = json.loads(response["body"].read())
#     completion = response_body['content'][0]['text']
#     return completion


# def get_prompt_query(result, value, prompt_file):
#     with open(prompt_file, 'r') as file:
#         prompt = file.read().replace("$$QUERY$$", result).replace("$$VALUE$$", value)
#     return prompt


# def get_prompt(result, prompt_file):
#     with open(prompt_file, 'r') as file:
#         prompt = file.read().replace("$$QUERY$$", result)
#     return prompt


# def get_prompt_condition(result, question):
#     with open('remove_condition_prompt.txt', 'r') as file:
#         prompt = file.read().replace("$$QUERY$$", result).replace("$$QUESTION$$", question)
#     return prompt

def lambda_handler(event, context):
    """Lambda handler function for processing the request and returning the response."""

    start_time = t.time()
    event_params = transform_event(event)
    logger.info(f"event:- {event_params}")
    data = event_params['data']['message']
    job_id = data['job_id']
    org_id = data['org_id']

    if job_id == JOB_ID:
        """Lambda handler function for processing the request and returning the response."""
        start_time = t.time()
        event_params = transform_event(event)
        logger.info("Checking in OpenSearch Serverless")
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
        query_rag = None
        query = None
        input_tokens = 0
        output_tokens = 0
        client = create_opensearch_client()
        with open("AWS_Neptune_OpenCypher_Restrictions.txt", 'r') as file:
            lmtn_prompt = file.read()

        json_nodes = search_relevant_node(question, NODE_INDEX)
        json_edges = get_edges(json_nodes, EDGE_INDEX)
        text_nodes = parse_json_nodes(json_nodes)
        text_edges = parse_json_edges(json_edges)
        logger.info(f"text_schema:- {text_nodes}\n{text_edges}")
        try:
            if not client.indices.exists(index=LIKED_INDEX):
                logger.info("common question not exist, checking files")
                # Fallback to file data if question collection does not exist
                if client.indices.exists(index=FILES_INDEX):
                    logger.info("Using File data")
                    results = search_similar_chunks(client, FILES_INDEX, question)
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
                similar_query = search_relevant_query(question, LIKED_INDEX)
                print("got similar query:", similar_query)
                if check_question_aoss(question, similar_query):
                    print("got query from common question")
                    if similar_query:
                        query_rag = similar_query[0]['query']
                        logger.info(f"got Liked query query")
                    else:
                        logger.info("No Liked query")
                        query_rag = None

                elif client.indices.exists(index=FILES_INDEX):
                    logger.info("Using File data")
                    results = search_similar_chunks(client, FILES_INDEX, question)
                    files_data = get_files_data(results)
                    if similar_query:
                        query_rag = similar_query[0]['query']
                        logger.info(f"got Liked query query")
                    query = query_response(job_id, files_data, query_rag, lmtn_prompt, question, text_nodes, text_edges)
                    logger.info("**Got DATA from Files**")

                else:
                    logger.info("No Like Query or files data")
                    query = query_response(job_id, None, None, lmtn_prompt, question, text_nodes, text_edges)
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
    else:
        database = "vec-db"
        collection_id = data['job_id']
        collection_id_question = job_id + "_Question"
        collection_id_files = job_id + "_Files"
        embed_data_question = None
        embed_data_file = None
        query_rag = None
        response = None
        input_tokens = 0
        output_tokens = 0

        with open("AWS_Neptune_OpenCypher_Restrictions.txt", 'r') as file:
            lmtn_prompt = file.read()

        try:
            if not check_collection_exists(database, collection_id_question):
                logger.info("common question not exist, checking files")
                # Fallback to file data if question collection does not exist
                if check_collection_exists(database, collection_id_files):
                    logger.info("Using File data")
                    db_instance_schema = get_db_inst(database, collection_id, use_existing=True)
                    embed_schema_data = get_embedding_schema_data(data['question'], db_instance_schema)
                    logger.info(f"embed_schema_data:- {embed_schema_data}")
                    db_instance_file = get_db_inst(database, collection_id_files, use_existing=True)
                    embed_data_file = get_embedding_data(data['question'], db_instance_file)
                    logger.info(f"embed_data_files:- {embed_data_file}")
                    chain = create_chain(
                        get_db_inst(database, collection_id, use_existing=True),
                        data['question'], job_id, None, embed_data_file, lmtn_prompt
                    )
                    # input_tokens = calculate_prompt_token(embed_schema_data, str(data['question']), job_id, None, embed_data_file, lmtn_prompt)
                    # response = chain.invoke(data['question'])
                    logger.info("**Got DATA from Files**")
                else:
                    logger.info("No Like Query or files data")
                    db_instance_schema = get_db_inst(database, collection_id, use_existing=True)
                    embed_schema_data = get_embedding_schema_data(data['question'], db_instance_schema)
                    logger.info(f"embed_schema_data:- {embed_schema_data}")
                    # input_token_embed = calculate_tokens(embed_data_file)
                    chain = create_chain(
                        get_db_inst(database, collection_id, use_existing=True),
                        data['question'], job_id, None, None, lmtn_prompt
                    )
                    # input_tokens = calculate_prompt_token(embed_schema_data, str(data['question']), job_id, None,
                    #                                       None, lmtn_prompt)
                    response = chain.invoke(data['question'])
                    logger.info("**No Liked query or files data**")
            else:
                # Question collection exists
                logger.info("Question collection exists")
                db_instance_question = get_db_inst(database, collection_id_question, use_existing=True)
                embed_data_question = get_embedding_data(data['question'], db_instance_question)
                logger.info(f"embed_data_question:- {embed_data_question}")

                if check_question(data['question'], embed_data_question):
                    logger.info("Using Like Query")
                    query_rag = get_common_query(data['question'], embed_data_question)
                    logger.info("**Created LIKED Query**")
                elif check_collection_exists(database, collection_id_files):
                    logger.info("Using File data")
                    db_instance_schema = get_db_inst(database, collection_id, use_existing=True)
                    embed_schema_data = get_embedding_schema_data(data['question'], db_instance_schema)
                    logger.info(f"embed_schema_data:- {embed_schema_data}")
                    db_instance_file = get_db_inst(database, collection_id_files, use_existing=True)
                    embed_data_file = get_embedding_data(data['question'], db_instance_file)
                    logger.info(f"embed_data_files:- {embed_data_file}")
                    chain = create_chain(
                        get_db_inst(database, collection_id, use_existing=True),
                        data['question'], job_id, embed_data_question, embed_data_file, lmtn_prompt
                    )
                    # input_tokens = calculate_prompt_token(embed_schema_data, str(data['question']), job_id, embed_data_question,
                    #                                       embed_data_file, lmtn_prompt)
                    response = chain.invoke(data['question'])
                    logger.info("**Got DATA from Files**")
                else:
                    logger.info("No Like Query or files data")
                    db_instance_schema = get_db_inst(database, collection_id, use_existing=True)
                    embed_schema_data = get_embedding_schema_data(data['question'], db_instance_schema)
                    logger.info(f"embed_schema_data:- {embed_schema_data}")
                    # input_token_embed = calculate_tokens(embed_data_file)
                    chain = create_chain(
                        get_db_inst(database, collection_id, use_existing=True),
                        data['question'], job_id, embed_data_question, None, lmtn_prompt
                    )
                    # input_tokens = calculate_prompt_token(embed_schema_data, str(data['question']), job_id, embed_data_question,
                    #                                       None, lmtn_prompt)
                    response = chain.invoke(data['question'])
                    logger.info("**No Liked query or files data**")

            # Extract Cypher queries from the response
            if 'response' in locals() and response and 'result' in response:
                logger.info("got response")
                # output_tokens = calculate_tokens(response)
                match_queries = re.findall(r'```cypher(.*?)```', response['result'], re.DOTALL)
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
            # logger.info(f"output_token:- {output_tokens}")
            logger.info(f"query_rag:- {query_rag}")
            # checking backtick
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

            # total_input_tokens = input_tokens + input_1 + input_2 + input_3 + input_4
            # total_output_tokens = output_tokens + output_1 + output_2 + output_3 + output_4
            # logger.info(f"total_input_tokens:-{total_input_tokens}")
            # logger.info(f"total_output_tokens:-{total_output_tokens}")
            # timestamp = datetime.utcnow().isoformat()
            # logger.info(
            #     json.dumps(
            #         {
            #             "timestamp": timestamp,
            #             "function_name": context.function_name,
            #             "org-id": org_id,
            #             "total_input_token": total_input_tokens,
            #             "total_output_token": total_output_tokens,
            #         }
            #     )
            # )
            # checking toLower()
            # lower_prompt = get_prompt(job_id_query, prompt_file ='check_toLower_prompt.txt' )
            # query = generate_response(lower_prompt)
            # logger.info(f"lower_query:- {query}")

            # property_lower = get_prompt_property(lower_query)
            # query = generate_response(property_lower)
            # checking group_by
            # groupBy_prompt = get_prompt_groupby(lower_query)
            # # final query
            # query = generate_response(groupBy_prompt)
            # logger.info(f"property_lower query:- {query}")

            logger.info('query sent to Neptune lambda')
            change_job_id_prompt = get_prompt(query, prompt_file='change_jobid.txt')
            change_job_id_query = generate_response(change_job_id_prompt)
            logger.info(f"change_job_id:- {change_job_id_query}")
            job_message = get_message(event_params, start_time, change_job_id_query)
            send_message_to_websocket_dispatcher(job_message)
            logger.info(f"query sent")
            message = get_message(event_params, start_time, query)
            event_bridge_response = send_message_to_event_bridge(message)
            logger.info(f"message sent from event_bridge")
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
