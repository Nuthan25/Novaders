import json
import uuid
import boto3
import logging
import traceback
from error_helper import sqs_helper
import re
import gzip
import base64
# file imports
import claude
import constants
import generateQuery
import generateAnswer
import getNeptuneResponse
import generateQueryWithHistory

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def send_to_websocket(connection_id, q_id, message, message_type=6):
    """Send a message to websocket dispatcher"""
    try:
        sqs_client = boto3.client('sqs')
        compressed_data = gzip.compress(json.dumps(message).encode('utf-8'))
        data = base64.b64encode(compressed_data).decode('utf-8')
        print(f"Base64 encoded size (bytes):- {len(data)}")
        message_body = {
            "target": [connection_id],
            "data": {
                "message": {
                    'q_id': q_id,
                    'answer': data,
                    'statuscode': 200,
                    'type': message_type,
                },
                'type': message_type
            }
        }

        response = sqs_client.send_message(
            QueueUrl=constants.dispatcher_queue_url,
            MessageBody=json.dumps(message_body, default=str),
            MessageDeduplicationId=str(uuid.uuid4()),
            MessageGroupId=str(uuid.uuid4())
        )
        return response
    except Exception as e:
        raise e


def generate_query(question, org_id, job_id):
    try:
        return generateQuery.handler(question, org_id, job_id)
    except Exception as e:
        raise e


def generate_query_with_history(question, org_id, job_id, chat_id):
    try:
        return generateQueryWithHistory.handler(question, org_id, job_id, chat_id)
    except Exception as e:
        raise e


def execute_query(query):
    try:
        return getNeptuneResponse.run_query(query)
    except Exception as e:
        raise e


def generate_answer(question, query, result, note, org_id, job_id):
    try:
        return generateAnswer.handler(question, query, result, note, org_id, job_id)
    except Exception as e:
        raise e


def respond_to_empty_or_error_result(question, result, job_id, org_id):
    try:
        processed_results = None
        if isinstance(result, dict):
            processed_results = getNeptuneResponse.process_response(result)
        graph_schema = generateQuery.get_graph_schema(job_id, org_id)
        error_prompt = generateQuery.format_prompt("prompt_file/query_response_prompt.txt", {
            "$$RESULT$$": str(processed_results if processed_results else result),
            "$$QUESTION$$": question,
            "$$SCHEMA$$": graph_schema
        })

        friendly_response, input_token, output_token = claude.generate_response(error_prompt, max_tokens=1500)
        return friendly_response
    except Exception as e:
        raise e


def replace_job_id(query, job_id):
    """Replace job id in a query with get_job_id()"""
    query = query.replace(f"'{job_id}'", 'get_job_id()')
    return query


def add_org_id(query, org_id):
    """adding org id to query"""
    if re.search(r'\borg_id\b', query):
        return query

    # Match something like "alias.job_id = get_job_id()"
    pattern = r'(\b\w+\.)?job_id\s*=\s*get_job_id\(\)'

    def replacer(match):
        prefix = match.group(1) or ''  # alias. if present, else ''
        return f"{prefix}job_id = get_job_id() AND {prefix}org_id = \"{org_id}\""

    modified_query = re.sub(pattern, replacer, query, count=1)
    return modified_query


def get_config(processed_results, message):
    """
    Extract config data from message and add it to processed_results.

    Args:
        processed_results (dict): Dictionary containing processed query results
        message (list of dict): Message potentially containing config information

    Returns:
        dict: Updated processed_results with config data if available
    """
    if processed_results.get('type') == "table":
        try:
            config = message[1].get('config') or message[0].get('config')
            if config:
                processed_results['config'] = config
            else:
                logger.info("Config not found in message[0] or message[1]")
        except (IndexError, AttributeError, TypeError) as e:
            logger.info(f"Error accessing config: {e}")
    else:
        logger.info("Skipped config extraction due to non-table type")

    return processed_results


def lambda_handler(event, context):
    event = json.loads(event['Records'][0]['body'])
    message = event.get('data', {}).get('message', {})
    logger.info(f"event:- {message}")
    job_id = message.get('job_id')
    org_id = message.get('org_id')
    question = message.get('question')
    chat_id = message.get('chat_id')
    connection_id = event.get('target', [''])[0]
    q_id = message.get('q_id')
    cypher_query = None

    try:
        if not chat_id:
            cypher_query, generated_query, note = generate_query(question, org_id, job_id)
        else:
            cypher_query, generated_query, note = generate_query_with_history(question, org_id, job_id, chat_id)
            logger.info(f"got query from history")

        if cypher_query and generated_query:
            status_code, result = execute_query(cypher_query)
            logger.info(f"got cypher query in FIRST trial")
            logger.info(f"cypher_query:- {cypher_query}")
            logger.info(f"generated_query_result:- {result}")
            if status_code != 200 or not len(result['results']) > 0:
                if not chat_id:
                    cypher_query, generated_query, note = generate_query(question, org_id, job_id)
                else:
                    cypher_query, generated_query, note = generate_query_with_history(question, org_id, job_id, chat_id)
                    logger.info(f"got query from history")
                status_code, result = execute_query(cypher_query)
                logger.info(f"got cypher query in SECOND trial")
                logger.info(f"cypher_query:- {cypher_query}")
                logger.info(f"generated_query_result:- {result}")
                if status_code != 200:
                    logger.info("Query returned empty or error")
                    message = respond_to_empty_or_error_result(question, result, job_id, org_id)
                    send_to_websocket(connection_id, q_id, add_org_id(replace_job_id(cypher_query, job_id), org_id))
                    send_to_websocket(connection_id, q_id, message, message_type=4)
                    logger.info(f"Query NLU response:- {message}")
                    return {'statusCode': 400}
                else:
                    send_to_websocket(connection_id, q_id, add_org_id(replace_job_id(cypher_query, job_id), org_id),
                                      message_type=6)
                    processed_results = getNeptuneResponse.process_response(result)
                    message = generate_answer(question, cypher_query, processed_results, note, org_id, job_id)
                    processed_results = get_config(processed_results, message)
                    send_to_websocket(connection_id, q_id, processed_results, message_type=7)
                    send_to_websocket(connection_id, q_id, message, message_type=4)
                    print(f"processed result with config:- {processed_results}")
                    logger.info(f"Query NLU response:- {message}")
            else:
                send_to_websocket(connection_id, q_id, add_org_id(replace_job_id(cypher_query, job_id), org_id),
                                  message_type=6)
                processed_results = getNeptuneResponse.process_response(result)
                message = generate_answer(question, cypher_query, processed_results, note, org_id, job_id)
                processed_results = get_config(processed_results, message)
                send_to_websocket(connection_id, q_id, processed_results, message_type=7)
                send_to_websocket(connection_id, q_id, message, message_type=4)
                print(f"processed result with config:- {processed_results}")
                logger.info(f"Query NLU response:- {message}")
            return {'statusCode': 200}
        elif cypher_query and not generated_query:
            status_code, result = execute_query(cypher_query)
            if status_code != 200:
                logger.info("Query returned empty or error")
                message = respond_to_empty_or_error_result(question, result, job_id, org_id)
                send_to_websocket(connection_id, q_id, add_org_id(replace_job_id(cypher_query, job_id), org_id))
                send_to_websocket(connection_id, q_id, message, message_type=4)
                logger.info(f"Query NLU response:- {message}")
                return {'statusCode': 400}
            else:
                send_to_websocket(connection_id, q_id, add_org_id(replace_job_id(cypher_query, job_id), org_id),
                                  message_type=6)
                processed_results = getNeptuneResponse.process_response(result)
                message = generate_answer(question, cypher_query, processed_results, note, org_id, job_id)
                processed_results = get_config(processed_results, message)
                send_to_websocket(connection_id, q_id, processed_results, message_type=7)
                send_to_websocket(connection_id, q_id, message, message_type=4)
                print(f"processed result with config:- {processed_results}")
                logger.info(f"got query from LIKE response")
                logger.info(f"Query NLU response:- {message}")
                return {'statusCode': 200}
        else:
            error_message = "Unable to generate a valid query for your question"
            send_to_websocket(connection_id, q_id, error_message, message_type=4)
            logger.info(f"Error message:- {error_message}")

        return {'statusCode': 200}

    except Exception as e:
        trace_back = traceback.format_exc()
        error_query = f'question:{question}\nquery:{cypher_query}\nerror:{e}'
        logger.error(f'{trace_back}\n{error_query}')
        sqs_helper.queue_message(context.function_name, error_query, trace_back)

        return {
            'statusCode': 500,
            'body': str(e)
        }
