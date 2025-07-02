import json
import os
import boto3
import logging
import traceback
import uuid
import psycopg2
from typing import List
from langchain.vectorstores.pgvector import PGVector
from langchain.embeddings import BedrockEmbeddings
from langchain.docstore.document import Document
from botocore.exceptions import ClientError

from error_helper import sqs_helper

logger = logging.getLogger()
logger.setLevel(logging.INFO)

s3 = boto3.client('s3')
s3_resource = boto3.resource('s3')

DB_HOST = os.getenv('DB_HOST')
DB_USER = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASSWORD')
DB_PORT = os.getenv('DB_PORT')
BUCKET = os.getenv('BUCKET')
DATABASE = os.getenv('DATABASE')


# def update_model(message):
#     logger.info(f'Message: {message}')
#     try:
#         sqs_client = boto3.client('sqs')
#         sqs_client.send_message(
#             QueueUrl=os.environ['INPUT_TRIGGER_URL'],
#             MessageBody=json.dumps(message, default=str),
#             MessageDeduplicationId=uuid.uuid4().__str__(),
#             MessageGroupId=uuid.uuid4().__str__()
#         )
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


def delete_collection(database, name):
    # Database connection parameters

    # Establish connection to the PostgreSQL database
    connection = psycopg2.connect(
        host=DB_HOST,
        database=database,  # Connect to the specified database
        user=DB_USER,
        password=DB_PASSWORD,
        port=DB_PORT,
    )

    try:
        # Create a session or connection to the PostgreSQL database
        with connection:
            with connection.cursor() as cursor:
                # SQL query to delete the entry by name and UUID
                query = """
                DELETE FROM langchain_pg_collection
                WHERE name = %s;
                """
                cursor.execute(query, (name,))

                # Commit the changes
                connection.commit()
                logger.info(f"Collection '{name}' deleted successfully.")

    except Exception as e:
        logger.info(f"Error occurred: {e}")

    finally:
        connection.close()


def create_db_inst(database, collection_name, schema_data):
    try:
        # Initialize the text embedding model
        embeddings = BedrockEmbeddings()
        CONNECTION_STRING = cred(database)
        # Check if the collection exists
        logger.info(f"schema_data:, {schema_data}")
        logger.info(f"embeddings:, {embeddings}")
        logger.info(f"collection_name:, {collection_name}")
        logger.info(f"CONNECTION_STRING:, {CONNECTION_STRING}")
        if check_collection_exists(database, collection_name):
            # Delete the existing collection and create a new vector store
            delete_collection(database, collection_name)
            vector_store = PGVector.from_documents(
                documents=schema_data,
                embedding=embeddings,
                collection_name=collection_name,
                connection_string=CONNECTION_STRING
            )
            logger.info("previous Data deleted and successfully loaded new Data.")
        else:
            # Collection does not exist; create a new vector store
            vector_store = PGVector.from_documents(
                documents=schema_data,
                embedding=embeddings,
                collection_name=collection_name,
                connection_string=CONNECTION_STRING
            )
            logger.info("Data loaded successfully.")
        # If everything is successful, print success message
    except Exception as e:
        # Handle any exceptions that occur
        logger.info(f"An error occurred: {e}")


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

            # Check if json_data is structured as expected
            if isinstance(json_data.get('data'), list):
                # Filter and process only records with non-null 'Query'
                for item in json_data['data']:
                    question = item.get('Question')
                    query = item.get('Query')

                    if question and query:  # Include only if both question and query are non-empty
                        output_data += f"Good Query:\nQUESTION: {question}\nQUERY: {query}\n\n"
            else:
                logger.info(f"Unexpected JSON structure in file: {obj['Key']}")
    # Create chunks for each "Good Query" section
    sections = output_data.strip().split("\n\n")
    documents = []

    for section in sections:
        if section.strip():  # Ensure no empty sections are added
            documents.append(
                Document(metadata={"source": "query_as_txt.txt"}, page_content=section.strip())
            )

    # Return the documents as JSON for inspection or further processing
    return documents


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
            collection_name = job_id + "_Question"

            schema_data = get_text_schema_doc(org_id, job_id)
            create_db_inst(DATABASE, collection_name, schema_data)

        logger.info('Event processed')

    except Exception as e:
        logger.info(f"Error processing event: {e}")
        logger.error(traceback.format_exc())
        trace_back = traceback.format_exc()
        sqs_helper.queue_message(context.function_name, e, trace_back)