import json
import os
import boto3
import logging
import traceback
import uuid
import psycopg2
from langchain.vectorstores.pgvector import PGVector
from langchain.embeddings import BedrockEmbeddings
from langchain.docstore.document import Document
from botocore.exceptions import ClientError
import pdfplumber
import io
import re

from error_helper import sqs_helper

logger = logging.getLogger()
logger.setLevel(logging.INFO)

s3 = boto3.client('s3')
s3_resource = boto3.resource('s3')

BUCKET = os.getenv('BUCKET')
DB_HOST = os.getenv('DB_HOST')
DB_USER = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASSWORD')
DB_PORT = os.getenv('DB_PORT')

types = {
    1: 'STRING',
    2: 'INTEGER',
    3: 'FLOAT',
    4: 'DATE',
    5: 'BOOLEAN',
    6: 'DATE',
    7: 'DATE',
    8: 'DATE',
    9: 'DATE',
}


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


def get_total_data_size(database, collection_name):
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
                # SQL query to get the total size of all data stored for the collection
                query = """
                    SELECT SUM(pg_column_size(t.*)) AS total_size_bytes
                    FROM langchain_pg_embedding t
                    WHERE collection_id = (
                        SELECT uuid FROM langchain_pg_collection
                        WHERE name = %s
                    )
                """
                cursor.execute(query, (collection_name,))
                result = cursor.fetchone()
                print(f"Total data size (bytes): {result[0]}")
                return result[0]

    except Exception as e:
        print(f"Error occurred: {e}")

    finally:
        connection.close()

def create_db_inst(database, collection_name, schema_data):
    try:
        # Initialize the text embedding model
        embeddings = BedrockEmbeddings()
        CONNECTION_STRING = cred(database)
        # Check if the collection exists
        logger.info(f"collection_name:-, {collection_name}")
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


def parse_json(json_data, org_id, job_id):
    # Initialize text parts
    ID = 'job_id'
    VALUE = job_id
    node_labels = "Following are the Nodes Label:\n"
    node_names = "Following are the Node Names:\n"
    edge_labels = "Following are the Edges Label:\n"
    relationships = "Relationships are defined as follows:\n"
    properties = ""

    # Extract vertices (nodes)
    nodes = json_data.get("schema", {}).get("vertex", [])
    edges = json_data.get("schema", {}).get("edge", [])

    # Extract node names and properties
    node_dict = {}
    for node in nodes:
        node_name = node.get("label")
        # node_name = node.get("replaceHeader") or raw_label
        node_labels += f"`{node_name}`,\n"
        node_names += f"Node Name of `{node_name}` is `{node_name}`,\n"

        # Extract properties for each node
        node_properties = node.get("properties", [])
        property_text = f"Properties for Node `{node_name}` are:\n"

        # Always include creationDate and Name properties
        property_text += "$$Name$$ as type STRING"

        if node_properties:
            for prop in node_properties:
                # Convert key to lowercase and replace spaces with underscores
                raw_key = prop.get("key")
                prop_name = prop.get("replaceHeader") or raw_key

                key = prop_name.lower().replace(" ", "_")
                # Use the provided types dictionary for the type conversion
                prop_type_code = prop.get("type")
                prop_type = types.get(prop_type_code, "STRING")  # Default to STRING if type is not found

                # Append other properties, ensuring $$ only for creationDate and Name
                property_text += f",\n{key} as type {prop_type}"
        property_text += ".\n\n"
        properties += property_text
        node_dict[node.get("label")] = node_name

    # Extract edge names and relationships
    for edge in edges:
        edge_name = edge.get("label")
        edge_labels += f"`{edge_name}`,\n"

        from_node = node_dict.get(edge.get("fromLabel"))
        to_node = node_dict.get(edge.get("toLabel"))
        relationships += f"Node `{from_node}` is connected to Node `{to_node}` via Edge `{edge_name}`,\n"

    important_notes = get_prompt('default_notes.txt').replace('$$ID$$', VALUE)

    important_notes_path = f"import/important_notes/{org_id}/{job_id}/important_notes.txt"
    try:
        s3.head_object(Bucket=BUCKET, Key=important_notes_path)
        response = s3.get_object(
            Bucket=BUCKET,
            Key=important_notes_path
        )
        file_data = response['Body'].read().decode('utf-8')
        file_content = file_data.replace("\n", "")
        important_notes += file_content

        # Combine all the parts
        final_output = (
                node_labels + "\n" + edge_labels + "\n" + node_names + "\n" + relationships + "\n" + properties + important_notes)
        return final_output
    except ClientError as e:
        # If a 404 error is raised, the object does not exist
        if e.response['Error']['Code'] == '404':
            # important_notes = (
            #     "Important Note:\n"
            #     f"JOB_ID: always add {ID} = '{VALUE}' in WHERE clause for every query generated, "
            #     "job_id and value should not be missed in query."
            # )
            final_output = (
                    node_labels + "\n" + edge_labels + "\n" + node_names + "\n" + relationships + "\n" + properties + important_notes)
            return final_output


def convert_to_doc(schema_text):
    """Reads and processes the schema text."""
    schema_file = 'schema_as_txt.txt'

    try:
        # Ensure schema_text is valid
        if not schema_text:
            raise ValueError("Parsed schema text is empty or invalid.")

        # Process and split the schema into sections
        sections = schema_text.strip().split("\n\n")
        documents = [Document(metadata={"source": schema_file}, page_content=section.strip()) for section in sections]

        return documents

    except Exception as e:
        # Catch any other general exceptions
        raise Exception(f"An unexpected error occurred: {str(e)}")


def get_json_schema(key, bucket):
    s3_response = s3.get_object(
        Bucket=bucket,
        Key=key
    )
    s3_object_body = s3_response.get('Body')
    content = s3_object_body.read()
    return json.loads(content)


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


def get_prompt(file_name):
    with open(file_name, 'r') as file:
        prompt = file.read()
    return prompt


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

        # Return the extracted content or errors
        return results
    except Exception as e:
        logger.info(f"failed to get file from s3:{e}")


def chunk_text(files, chunk_size=10):
    try:
        text = get_pdf_from_s3(files)
        data = "\n\n".join(text)
        # Split the text by period, strip any whitespace
        sentences = [sentence.strip() + '.' for sentence in data.split('.') if sentence.strip()]

        # Group sentences into chunks
        chunks = []
        for i in range(0, len(sentences), chunk_size):
            chunk = ' '.join(sentences[i:i + chunk_size])
            chunks.append(chunk)

        return chunks
    except Exception as e:
        logger.info(f"failed to create chunks:{e}")


def get_doc(schema_text):
    # Create chunks as Document objects
    try:
        sections = schema_text
        source = "Model_Information.txt"
        documents = []
        for section in sections:
            documents.append(
                Document(metadata={"source": source}, page_content=section.strip())
            )

        return documents
    except Exception as e:
        logger.info(f"failed to create document from chunks:{e}")


def lambda_handler(event, context):
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
            database = os.getenv('DATABASE')
            model_id = json_data['schema'].get('fromModel')
            files = json_data['schema'].get('resourceList')
            collection_id = job_id + "_Files"
            try:
                if files:
                    logger.info("read and load the FILE to aurora db.")
                    chunks = chunk_text(files)
                    doc_chunk = get_doc(chunks)
                    logger.info(f"chunk documents:-{doc_chunk}")
                    create_db_inst(database, collection_id, doc_chunk)
                    logger.info("FILE db instance created")
                else:
                    try:
                        if check_collection_exists(database, collection_id):
                            delete_collection(database, collection_id)
                            logger.info("deleted previous FILE")
                    except Exception as e:
                        logger.info(f"Error deleting FILE DB Instance: {e}")
            except Exception as e:
                logger.info(f"Error creating FILE DB Instance: {e}")

            logger.info("read and load the ADDITIONAL INFO to aurora db.")
            txt_schema = parse_json(json_data, org_id, job_id)
            result = convert_to_doc(txt_schema)

            # Prepare for database operations
            collection_name = job_id
            create_db_inst(database, collection_name, result)

            # calculating the size based on collection name
            schema_data_size = get_total_data_size(database, collection_name)
            file_data_size = get_total_data_size(database, collection_id)
            logger.info(f"schema_data_size:-{schema_data_size}")
            logger.info(f"file_data_size:-{file_data_size}")

            # Construct the folder path based on org_id and job_id
            schema_folder = f"import/embedding_schema/{org_id}/{job_id}/"

            # Check if any files exist in the import/embedding_schema/{org_id}/{job_id} folder
            existing_files = s3.list_objects_v2(Bucket=bucket_name, Prefix=schema_folder)

            # If there are existing files, delete them
            if 'Contents' in existing_files:
                for file in existing_files['Contents']:
                    s3.delete_object(Bucket=bucket_name, Key=file['Key'])
                    logger.info(f"Deleted existing file: {file['Key']}")

            # Construct the new schema path
            if 'schemapath' in event:
                new_schema = f"{schema_folder}{event['schemapath'].split('/')[-1].split('.')[0]}.txt"
            else:
                new_schema = f"{schema_folder}schema.txt"

            # Save the output to a temporary file
            with open(tmp_file, 'w') as output_file:
                output_file.write(txt_schema)

            # Upload the temporary file to S3
            try:
                s3.upload_file(tmp_file, bucket_name, new_schema)
                #logger.info(f"File uploaded successfully to {bucket_name}/{new_schema}")
            except Exception as e:
                raise f"Error uploading file to S3: {e}"

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
