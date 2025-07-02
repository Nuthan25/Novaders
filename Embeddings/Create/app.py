import io
import os
import json
import uuid
import boto3
import logging
import asyncio
import traceback
import opensearch
import pdfplumber
from typing import Dict, List, Any, Optional
from botocore.exceptions import ClientError

logger = logging.getLogger()
logger.setLevel(logging.INFO)

INDEX_NAME = os.getenv('INDEX_NAME')
REGION = os.getenv('REGION')
HOST = os.getenv('HOST')
PORT = os.getenv('PORT')
QUEUE_URL = os.getenv('QUEUE_URL')


def update_model(message: dict, org_id: str):
    try:
        sqs_client = boto3.client('sqs')
        sqs_client.send_message(
            QueueUrl=QUEUE_URL,
            MessageBody=json.dumps(message, default=str),
            MessageDeduplicationId=str(uuid.uuid4()),
            MessageGroupId=str(uuid.uuid4())
        )
        logger.info(json.dumps({"log_type": "sqs", "value": org_id}))
    except Exception as e:
        raise Exception(f"Error sending message to SQS: {str(e)}")


async def get_nodes_and_edges_from_schema(schema):
    """
    Extract nodes and edges from the schema.
    """
    try:
        nodes = await get_data_from_array(schema['vertex'])
        edges = await get_data_from_array(schema['edge'])
        return {
            "nodes": nodes,
            "edges": edges
        }
    except Exception as e:
        raise Exception(f"Error getting nodes and edges from schema: {str(e)}")


async def get_data_from_array(data: List[Dict[str, Any]]):
    """
    Get data from a list of data.
    """
    try:
        types = {
            1: 'STRING', 2: 'INTEGER', 3: 'FLOAT', 4: 'DATE',
            5: 'BOOLEAN', 6: 'DATE', 7: 'DATE', 8: 'DATE', 9: 'DATE',
        }

        results = []
        for data_item in data:
            label = data_item['label']
            properties = data_item.get('properties', [])
            note = data_item.get('note')

            updated_properties = {}

            for prop in properties:
                raw_key = prop.get("key")
                prop_name = prop.get("replaceHeader") or raw_key
                updated_properties[prop_name] = types.get(prop.get("type"), "STRING")

            res = {'label': label, 'properties': updated_properties}
            if note:
                res['note'] = note

            results.append(res)

        return results
    except Exception as e:
        raise Exception(f"Error getting data from array: {str(e)}")


async def get_important_notes(bucket: str, org_id: str, job_id: str):
    """
    Get important notes related to the data.
    """
    try:
        key = f'import/important_notes/{org_id}/{job_id}/important_notes.txt'

        try:
            res = boto3.client('s3').get_object(Bucket=bucket, Key=key)
            text = res['Body'].read().decode('utf-8')
        except Exception as e:
            logger.info(e)
            return []

        text_list = text.split('.,')
        chunk_list = []
        for text_item in text_list:
            for item in text_item.split('\n'):
                chunk_list.append(item)

        return chunk_list
    except Exception as e:
        raise Exception(f"Error getting important notes: {str(e)}")


async def get_user_files(bucket: str, keys: list):
    """
    Get user files from S3 or other storage.
    """
    try:
        results = []
        chunk_list = []

        for key in keys:
            res = boto3.client('s3').get_object(Bucket=bucket, Key=key)
            content = res['Body'].read()

            with pdfplumber.open(io.BytesIO(content)) as pdf:
                text = ''
                for page in pdf.pages:
                    text += page.extract_text()

            results.append(text)

        for result in results:
            text_list = result.split('\n')
            for text_item in text_list:
                chunk_list.append(text_item)

        return chunk_list
    except Exception as e:
        raise Exception(f"Error getting user files: {str(e)}")


async def get_catalogue(schema: dict):
    try:
        documents = []

        for vertex in schema['vertex']:
            if vertex.get('catalog_data'):
                documents.append(vertex['catalog_data'])

        for edge in schema['edge']:
            if edge.get('catalog_data'):
                documents.append(edge['catalog_data'])

        return documents
    except Exception as e:
        raise Exception(f"Error getting catalogue: {str(e)}")


async def get_data_for_embedding(bucket: str, key: str, job_id: str, org_id: str):
    """
    Asynchronous function to get data for embedding.
    This function is triggered by an S3 event.
    """
    try:
        res = boto3.client('s3').get_object(Bucket=bucket, Key=key)
        schema = json.loads(res['Body'].read())['schema']
        pdf_files = schema.get('resourceList', [])
        model_id = schema.get('fromModel')

        notes, files, catalogue = await asyncio.gather(
            get_important_notes(bucket, org_id, job_id),
            get_user_files(bucket, pdf_files),
            get_catalogue(schema)
        )

        return {
            "note": notes,
            "file": files,
            "catalogue": catalogue,
            "model": model_id,
        }
    except Exception as e:
        raise Exception(f"Error getting data for embedding: {str(e)}")


async def handler(record):
    try:
        bucket = record['s3']['bucket']['name']
        key = record['s3']['object']['key']

        job_id = key.split('/')[-2]
        org_id = key.split('/')[-3]
        model_id = f'model_{org_id}_{job_id}'

        logger.info(json.dumps({"log_type": "org_id", "value": org_id}))

        result = await get_data_for_embedding(bucket, key, job_id, org_id)
        result['model_id'] = model_id
        result['job_id'] = job_id
        result['org_id'] = org_id

        return result
    except Exception as e:
        raise Exception(f"Error parsing record: {str(e)}")


def lambda_handler(event, context):
    """
    Main function triggered by S3 events.
    """
    logger.info(f"Received event: {json.dumps(event)}")
    try:
        for record in event['Records']:
            result = asyncio.run(handler(record))
            org_id = result['org_id']
            model_id = result['model_id']

            opensearch_client = opensearch.OpenSearchClient(HOST, PORT, INDEX_NAME, REGION, model_id)
            opensearch_client.connect()
            opensearch_client.check_and_create_indices_index()

            for doc_type in ['note', 'file', 'catalogue']:
                if result[doc_type]:
                    if opensearch_client.check_query_index(doc_type):
                        opensearch_client.delete_index(doc_type)

                    opensearch_client.create_index(doc_type, result[doc_type])

            opensearch_client.disconnect()

            if result.get('model'):
                update_model({
                    "path": "api/graph-model/",
                    "data": {
                        "id": result['model'],
                        "notes_updating": False
                    }
                }, org_id)

        return {'statusCode': 200}
    except Exception as e:
        logger.error(e)
        logger.error(traceback.format_exc())
        return {'statusCode': 500}
