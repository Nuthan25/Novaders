import boto3
import json
import time as t
import os
import traceback
import logging
import uuid
from error_helper import sqs_helper
import tiktoken

logger = logging.getLogger()
logger.setLevel(logging.INFO)
error_message = 'Could not process the input. Please try to phrase it differently'

def calculate_tokens(text):
    encoder = tiktoken.get_encoding("cl100k_base")
    token_count = len(encoder.encode(text))
    return token_count

def transform_event(event):  # Read event from event-bridge
    try:
        event_params = event['detail']
        return event_params

    except Exception as e:
        logger.error(e, exc_info=True)
        raise e


def get_prompt(prompt):  # Read prompt to generate response from txt file
    with open('prompt.txt', 'r') as prompt_file:
        claude_prompt = prompt_file.read()
    claude_prompt = claude_prompt.replace('$$PROMPT$$', prompt)
    return claude_prompt


def generate_response(prompt):  # Generate response using Bedrock
    claude_prompt = get_prompt(prompt)
    bedrock_region = os.environ.get('BEDROCK_REGION')
    input_token = calculate_tokens(claude_prompt)
    client = boto3.client(service_name="bedrock-runtime", region_name=bedrock_region)
    model_id = "anthropic.claude-3-haiku-20240307-v1:0"
    response = client.invoke_model(
        modelId=model_id, body=json.dumps(
            {"anthropic_version": "bedrock-2023-05-31", "max_tokens": 8196,
             "messages": [
                 {
                     "role": "user",
                     "content": [{"type": "text", "text": claude_prompt}],
                 }
             ],
             }
        ),
    )
    response_body = json.loads(response.get("body").read())
    result = str(response_body['content'][0]['text'])
    output_token = calculate_tokens(result)
    trimmed_result = trim_result(result)
    return trimmed_result, input_token, output_token


def trim_result(result):  # Trim the result generated from bedrock to JSON
    # start_index = result.find("{")
    # end_index = result.find(";", start_index)
    # trimmed_result = result[start_index:end_index].strip()
    # Convert to json
    # trimmed_result = trimmed_result.replace("\n", "")
    trimmed_result = "[" + result + "]"
    trimmed_result = json.loads(trimmed_result)
    if not trimmed_result:  # Check if the result is empty
        return 'empty result'
    else:
        return trimmed_result


def check_json(result, message_to_queue):
    """ Validate JSON structure """
    if not isinstance(result, dict):
        logger.info(f"Invalid result format: Expected a dictionary but got {type(result)}")
        message_to_queue['statuscode'] = 400
        message_to_queue['answer'] = "Invalid response format"
        return

    required_keys = ['relationships', 'terms_definition']
    missing_keys = [key for key in required_keys if key not in result]

    if missing_keys:
        logger.info(f"Missing required keys in JSON response: {missing_keys}")
        message_to_queue['statuscode'] = 400
        message_to_queue['answer'] = f"Invalid response structure. Missing keys: {missing_keys}"
        return

    if not isinstance(result['relationships'], list) or not isinstance(result['terms_definition'], list):
        logger.info("Invalid type for 'relationships' or 'terms_definition'")
        message_to_queue['statuscode'] = 400
        message_to_queue['answer'] = "Invalid response structure"
        return

    if not result['relationships'] or not result['terms_definition']:
        logger.info("Empty 'relationships' or 'terms_definition' list")
        message_to_queue['statuscode'] = 400
        message_to_queue['answer'] = "Empty response data"
        return

    # Validate inner dictionary structure
    if isinstance(result['relationships'][0].get('source'), str) and isinstance(
            result['terms_definition'][0].get('term'), str):
        message_to_queue['statuscode'] = 200
        message_to_queue['answer'] = result
    else:
        logger.info("Invalid data types in JSON response")
        message_to_queue['statuscode'] = 400
        message_to_queue['answer'] = "Invalid response data format"


def send_message_to_queue(connection_id, message_to_queue):  # Send message to Dispatcher Queue
    sqs_client = boto3.client('sqs')
    queue_url = os.environ.get('QUEUE_URL')
    try:
        logger.info(f'Queue Message{message_to_queue}')
        message = {
            "target": [connection_id],
            "data": {
                "message": message_to_queue,
                'type': 5
            }
        }
        sqs_client.send_message(QueueUrl=queue_url,
                                MessageBody=json.dumps(message, default=str),
                                MessageDeduplicationId=str(uuid.uuid4()),
                                MessageGroupId=str(uuid.uuid4())
                                )
        logger.debug("Message pushed")
    except Exception as e:
        logger.error(e, exc_info=True)
        raise e


def error_message_to_queue(log_message, connection_id, message_to_queue):  # Write error message to send to queue
    logger.info(log_message)
    if log_message == 'Incorrect Question':
        message_to_queue['statuscode'] = 400
    start_time = message_to_queue['time_taken']
    message_to_queue['time_taken'] = t.time() - start_time
    send_message_to_queue(connection_id, message_to_queue)


def lambda_handler(event, context):
    print("event", event)
    event_params = transform_event(event)
    prompt = event_params['question']
    connection_id = event_params['ConnectionID']
    start_time = event_params['start_time']
    result = 'None'
    message_to_queue = {'q_id': event_params['q_id'], 'statuscode': 500, 'answer': error_message,
                        'time_taken': start_time}
    try:
        logger.info(prompt)
        if len(prompt.split()) <= 1:  # Check if number of words is 1 or 0
            error_message_to_queue('Incorrect Question', connection_id, message_to_queue)
            return

        result, input_token, output_token = generate_response(prompt)
        logger.info(result)
        logger.info(f"total_input_token:- {input_token}")
        logger.info(f"total_output_token:- {output_token}")

        if result == 'empty result':  # Check if result is empty
            error_message_to_queue(result, connection_id, message_to_queue)
            return

        if not isinstance(result, list) or len(result) == 0:
            error_message_to_queue('Invalid or empty result', connection_id, message_to_queue)
            return

        result = result[0]  # Extract the first item from the list

        if not isinstance(result, dict):
            error_message_to_queue('Unexpected result format', connection_id, message_to_queue)
            return

        if not ('terms_definition' in result and 'relationships' in result):
            # Extract nested dictionary if it exists
            possible_key = list(result.keys())[0]  # Example: "football team"
            if isinstance(result[possible_key], dict):
                result = result[possible_key]

        if 'terms_definition' not in result or 'relationships' not in result:
            error_message_to_queue("Missing required keys in JSON response", connection_id, message_to_queue)
            return

        check_json(result, message_to_queue)

        message_to_queue['time_taken'] = t.time() - start_time
        send_message_to_queue(connection_id, message_to_queue)
        return {
            'statuscode': 200,
            'body': 'Message Sent'
        }

    except Exception as e:
        trace_back = traceback.format_exc()
        logger.error(trace_back)
        error_prompt = f'Prompt:{prompt}\nAnswer:{result}\nerror:{e}'
        message_to_queue['time_taken'] = t.time() - start_time
        send_message_to_queue(connection_id, message_to_queue)
        sqs_helper.queue_message(context.function_name, error_prompt, trace_back)
        return {
            'statuscode': 200,
            'body': 'Message Sent'
        }
