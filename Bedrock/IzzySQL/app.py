import json
import boto3
import psycopg2
from psycopg2 import sql
import os
import logging
import uuid
import time as t
import traceback
from error_helper import sqs_helper

logger = logging.getLogger()
logger.setLevel(logging.INFO)
error_message = 'Could not process the input. Please try to phrase it differently'


def transform_event(event):  # Read event from event-bridge
    try:
        event_params = event['detail']
        return event_params
    except Exception as e:
        logger.error(e, exc_info=True)
        raise e


def create_sql_connection():  # Create SQL connection
    env = os.environ.get('Env')
    ssm_client = boto3.client('ssm')

    db_host = ssm_client.get_parameter(Name=f'/{env}/rds/address/endpoint')
    db_host = db_host['Parameter']['Value']

    db_name = ssm_client.get_parameter(Name=f'/{env}/rds/dbname')
    db_name = db_name['Parameter']['Value']

    db_user = ssm_client.get_parameter(Name=f'/{env}/rds/username')
    db_user = db_user['Parameter']['Value']

    db_password = ssm_client.get_parameter(Name=f'/{env}/rds/masteruser/password')
    db_password = db_password['Parameter']['Value']

    db_params = {
        'host': db_host,
        'database': db_name,
        'user': db_user,
        'password': db_password,
    }
    conn = psycopg2.connect(**db_params)
    cursor = conn.cursor()
    return conn, cursor


def close_sql_connection(conn, cursor):  # Close SQL connection
    cursor.close()
    conn.close()


def check_query(query):  # Check if query has create, update, delete or drop operations
    keywords = ["CREATE", "UPDATE", "DELETE", "DROP"]
    words = query.split()
    found_keyword = False
    for word in words:
        if word.strip('\'') in keywords and '\'' not in word:
            found_keyword = True
            break
    return found_keyword


def execute_query(query, connection_id, message_to_queue):  # Execute the query
    found_keyword = check_query(query)  # Returns True if CREATE, UPDATE, UPDATE, DELETE is present in query
    if found_keyword:
        return "error"
    else:
        try:
            conn, cursor = create_sql_connection()
            cursor.execute(query)
            result = str(cursor.fetchall())
            close_sql_connection(conn, cursor)
            return result
        except Exception as e:
            logger.error(str(e))
            start_time = message_to_queue['time_taken']
            exec_time = t.time() - start_time
            message_to_queue['statuscode'] = 500
            message_to_queue['time_taken'] = exec_time
            send_message_to_queue(connection_id, message_to_queue)
            return {
                'statuscode': 200,
                'body': 'Message Sent'
            }


def send_message_to_queue(connection_id, message_to_queue):  # Send message to dispatcher queue
    sqs_client = boto3.client('sqs')
    queue_url = os.environ.get('QUEUE_URL')
    try:
        logger.info(f'Queue Message{message_to_queue}')
        message = {
            "target": [connection_id],
            "data": {
                "message": message_to_queue,
                'type': 4
            }
        }
        response = sqs_client.send_message(QueueUrl=queue_url,
                                           MessageBody=json.dumps(message, default=str),
                                           MessageDeduplicationId=str(uuid.uuid4()),
                                           MessageGroupId=str(uuid.uuid4())
                                           )
        logger.debug("Message pushed")
    except Exception as e:
        logger.error(e, exc_info=True)
        raise e


def error_message_to_queue(log_message, connection_id, message_to_queue):  # Send error message to dispatcher queue
    logger.info(log_message)
    start_time = message_to_queue['time_taken']
    message_to_queue['time_taken'] = t.time() - start_time
    send_message_to_queue(connection_id, message_to_queue)


def lambda_handler(event, context):
    logger.info(event)
    event_params = transform_event(event)
    connection_id = event_params['ConnectionID']
    question = event_params['question']
    query = event_params['query']
    type = event_params['type']
    message_to_queue = {'q_id': event_params['q_id'],
                        'statuscode': 400,
                        'question': question,
                        'answer': error_message,
                        'type': type,
                        'ConnectionID': connection_id,
                        'time_taken': float(event_params['start_time'])}
    try:
        if (len(question.split())) <= 1:
            error_message_to_queue('Invalid Question', connection_id, message_to_queue)
        else:
            query_result = execute_query(query, connection_id, message_to_queue)
            logger.info(f'Query:{query}\nQuery Result:{query_result}')
            if query_result == 'error':
                error_message_to_queue('Invalid Operation', connection_id, message_to_queue)
            else:
                message_to_queue['answer'] = query_result
                send_message_to_queue(connection_id, message_to_queue)
        return {
            'statuscode': 200,
            'body': 'Message Sent'
        }
    except Exception as e:
        trace_back = traceback.format_exc()
        error_query = f'question{question}\nquery:{query}\nerror:{e}'
        logger.error(f'{trace_back}\n{error_query}')
        sqs_helper.queue_message(context.function_name, error_query, trace_back)
        return {
            'statuscode': 500,
            'message': str(e)
        }
