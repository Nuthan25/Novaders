import json
import boto3
import os
import re
import traceback
from langchain.embeddings import BedrockEmbeddings
from langchain.vectorstores.pgvector import PGVector
from langchain.docstore.document import Document
from langchain_community.chat_models import BedrockChat
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from db import DB
import psycopg2
import time as t
import uuid

# Constants
env = os.environ.get('Env')
DB_HOST = os.getenv('DB_HOST')
DB_USER = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASSWORD')
DB_PORT = os.getenv('DB_PORT')
MODEL_ID = os.getenv('MODEL_ID')
REGION = os.getenv('AWS_REGION')

def transform_event(event):  # Read event from event-bridge
    try:
        event_params = event['Records'][0]["body"]
        event_params = json.loads(event_params)
        return event_params
    except Exception as e:
        raise e

def get_schema(job_id, schema_file='schema_as_txt.txt'):
    """Reads and processes the schema text."""
    try:
        with open(schema_file, 'r') as file:
            schema_text = file.read().replace("$$ID$$", 'job_id').replace("$$VALUE$$", job_id)
    except FileNotFoundError:
        raise Exception(f"Schema file {schema_file} not found.")

    sections = schema_text.strip().split("\n\n")
    documents = [Document(metadata={"source": schema_file}, page_content=section.strip()) for section in sections]

    return documents


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
        # exe = test_db_connection(connection_str)
        # print(exe)
        print(f"Connection string generated: {connection_str}")
        return connection_str
    except Exception as e:
        print(f"Error generating connection string: {str(e)}")
        raise e



def create_db_instance(database, collection_name, data):
    """Creates a vector store instance and populates it with embeddings."""
    embeddings = BedrockEmbeddings()
    connection_string = cred(database)

    try:
        print(f"Attempting to create PGVector instance with collection: {collection_name}")
        embedding = PGVector.from_documents(
            documents=data,
            embedding=embeddings,
            collection_name=collection_name,
            connection_string=connection_string
        )
        print("Successfully connected to the Aurora DB and created embeddings.")
        return embedding
    except Exception as e:
        print(f"Error while connecting to the database or creating embeddings: {str(e)}")
        raise e



def create_chain(db_instance, question, job_id, prompt_file='prompt.txt'):
    """Sets up the RetrievalQA chain with the given prompt and database instance."""
    client = boto3.client(service_name="bedrock-runtime", region_name=REGION)

    try:
        with open(prompt_file, 'r') as file:
            prompt = file.read().replace("$$ID$$", 'job_id').replace("$$VALUE$$", job_id)
    except FileNotFoundError:
        raise Exception(f"Prompt file {prompt_file} not found.")

    # Initialize LLM
    llm = BedrockChat(
        client=client,
        model_id=MODEL_ID
    )

    # Set up prompt template
    prompt_template = PromptTemplate(
        template=prompt,
        input_variables=["context", "question"]
    )

    # Create and return the retrieval chain
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt_template},
        retriever=db_instance.as_retriever(search_type="mmr", search_kwargs={'k': 9, 'lambda_mult': 0.5}),
    )


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
        print("queries type:", type(query))
        print("queries", query)
        # Build the message
        message = {
            'question': data['question'],
            'query': query,
            'ConnectionID': connection_id,
            'q_id': q_id,
            'type': types,
            'start_time': start_time
        }
        return message

    except KeyError as e:
        print(f"KeyError: Missing key {str(e)} in input data")
        raise
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise


def send_message_to_event_bridge(message):
    client = boto3.client('events')
    try:
        # Send event to EventBridge
        print("inside try-------")
        response = client.put_events(
            Entries=[
                {
                    'Source': 'preprod-bedrock',
                    'DetailType': 'event trigger for bedrock',
                    'Detail': json.dumps(message),
                    'EventBusName': 'preprod-bedrock-bus'
                },
            ]
        )
        print("response:", response)
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


def lambda_handler(event, context):
    """Lambda handler function for processing the request and returning the response."""
    start_time = t.time()
    event_params = transform_event(event)
    print(event_params)
    data = event_params['data']['message']
    job_id = data['job_id']

    print("job id:",job_id)
    database = "vec-db"
    collection_name = "text-schema"

    try:
        # Retrieve schema and initialize database
        schema_data = get_schema(job_id)
        print("data:", schema_data)
        db_instance = create_db_instance(database, collection_name, schema_data)
        # Process question
        question = data['question']
        # Create retrieval chain
        chain = create_chain(db_instance, question, job_id)
        response = chain.invoke(question)
        # Extract Cypher queries from the response
        match_queries = re.findall(r'```cypher(.*?)```', response['result'], re.DOTALL)
        queries = "".join(match_queries)
        if not queries:
            return {
                'statusCode': 400,
                'body': 'No Cypher queries found in the response.'
            }

        print("queries type:", type(queries))
        print("queries", queries)
        message = get_message(event_params, start_time, queries)
        event_bridge_response = send_message_to_event_bridge(message)
        print("event_bridge_response:",event_bridge_response)
        # Check if the message was successfully sent
        if event_bridge_response.get('statusCode') == 200:
            print("Message sent to EventBridge successfully!")
        else:
            print(f"Failed to send message to EventBridge: {event_bridge_response}")

        return {
            'statusCode': 200,
            'body': 'Message Sent'
        }

    except Exception as e:
        trace_back = traceback.format_exc()
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': str(e),
                'traceback': trace_back
            })
        }
