import json
import boto3
import os
import logging
import uuid
import time as t
import re
import traceback
from error_helper import sqs_helper
import tiktoken

logger = logging.getLogger()
logger.setLevel(logging.INFO)
bedrock_region = os.environ.get('BEDROCK_REGION')
MODEL_ID = os.environ.get('MODEL_ID')

# def calculate_tokens(text):
#     encoder = tiktoken.get_encoding("cl100k_base")
#     token_count = len(encoder.encode(text))
#     return token_count

def transform_event(event):  # Read event from event-bridge
    try:
        event_params = event['detail']
        id_key = 'job_id'
        if 'awsAccountId' in event_params.keys():
            id_key = 'awsAccountId'
        return event_params, id_key

    except Exception as e:
        logger.error(e, exc_info=True)
        raise e


def check_query(query):  # Check if query does create update delete operations
    keywords = ["CREATE", "DELETE", "UPDATE", "DETACH"]
    words = query.split()
    found_keyword = False
    for word in words:
        if word.strip('\'') in keywords and '\'' not in word:
            found_keyword = True
            break
    return found_keyword


def connect_to_neptune():  # Connect to NeptuneDB
    session = boto3.Session()
    host = os.environ.get('CLUSTER_ENDPOINT')
    port = os.environ.get('CLUSTER_PORT')
    client_params = {'endpoint_url': f"https://{host}:{port}"}
    neptune_client = session.client("neptunedata", **client_params)
    return neptune_client


def node_object(data):
    node = {'id': data['~id'], 'label': (',').join(data['~labels']),
            '$$Name$$': data['~properties']['$$Name$$'],
            'properties': data['~properties']}

    return node


def edge_object(data):
    edge = {'id': data['~id'], 'label': data['~type'], 'properties': data['~properties'],
            'source': data["~start"], 'target': data["~end"]}

    return edge


def round_results(response):
    # Loop through the list in 'results' and round any numeric values
    for record in response['results']:
        for key, value in record.items():
            if isinstance(value, (int, float)):
                # Format the numeric values to show exactly two decimal places as a string
                record[key] = round(value, 2)

    return response


def execute_query(query, question, context):  # Execute opencypher query
    try:
        query = query.replace('\n', ' ')
        if check_query(query):
            return "error in query"
        else:
            neptune_client = connect_to_neptune()
            result = neptune_client.execute_open_cypher_query(
                openCypherQuery=query)  # Execute the Opencypher Query
            logger.info("executed query")
            return result
    except Exception as e:
        trace_back = traceback.format_exc()
        logger.error(trace_back)
        logger.error(f"Malformed Query\n{str(e)}")
        error_query = f'question:{question}\nquery:{query}'
        sqs_helper.queue_message(context.function_name, error_query, trace_back)
        logger.info("error raised")
        return 'error in query'


def is_table(data):
    for result in data['results']:
        for values in result.values():
            if isinstance(values, dict) or isinstance(values, list):
                return False
            else:
                return True

def extract_and_validate_json(response):
    """Extracts and validates AI-generated JSON responses."""
    try:
        response = response.strip()  # Remove leading/trailing whitespace

        # Attempt to extract JSON if response does not start with { or [
        if not response.startswith("{") and not response.startswith("["):
            match = re.search(r"(\{.*\}|\[.*\])", response, re.DOTALL)
            response = match.group(0) if match else None

        if not response:
            raise ValueError("No valid JSON found in the response.")

        return json.loads(response)  # Parse JSON
    except (json.JSONDecodeError, ValueError) as e:
        print(f"JSON parsing error: {e}")
        return None  # Handle invalid JSON gracefully

def get_list(data, entity_type):
    vertex_r_edge_list = []
    table_data = None
    for result in data['results']:
        for values in result.values():
            if isinstance(values, dict):
                if values.get('~entityType'):
                    if entity_type == 'node':
                        vertex = node_object(values)
                        if vertex not in vertex_r_edge_list:
                            vertex_r_edge_list.append(vertex)
                    elif entity_type == 'relationship':
                        edge = edge_object(values)
                        vertex_r_edge_list.append(edge)
                elif entity_type == 'table':
                    table_data = data
            elif isinstance(values, list):
                for value in values:
                    if isinstance(value, dict):
                        if entity_type == value['~entityType'] == 'node':
                            vertex = node_object(value)
                            if vertex not in vertex_r_edge_list:
                                vertex_r_edge_list.append(vertex)
                        elif entity_type == value['~entityType'] == 'relationship':
                            edge = edge_object(value)
                            vertex_r_edge_list.append(edge)
                        elif entity_type == 'table':
                            table_data = data
            elif entity_type == 'path':
                return data['results'], table_data

    return vertex_r_edge_list, table_data


def create_vertex_for_edge(data):
    vertex_list = []
    for result in data['results']:
        for values in result.values():
            if isinstance(values, dict):
                vertex_list.append({'id': values['~start'], 'label': None, 'properties': None})
                vertex_list.append({'id': values['~end'], 'label': None, 'properties': None})
            else:
                logger.debug(f"Type of values: {type(values)}")

    return vertex_list


def detect_query_data(data):
    result_set = set()
    for result in data['results']:
        for values in result.values():
            if isinstance(values, dict):
                if '~entityType' in values:
                    result_set.add(values['~entityType'])
                else:
                    result_set.add('table')
            elif isinstance(values, list):
                for value in values:
                    if '~entityType' in value:
                        result_set.add(value['~entityType'])
                    else:
                        result_set.add('table')
            else:
                result_set.add('path')

    return result_set


def process_query(query_result):
    logger.info(f"Query: {query_result['results']}")
    node_r_edge = {
        'node': 'nodes',
        'relationship': 'edges',
        'path': 'path',
        'table': 'table'
    }
    if is_table(query_result):
        return {'table': query_result['results'], 'type': 'table'}

    if not query_result['results']:
        return {'table': query_result['results'], 'type': 'table'}

    results = detect_query_data(query_result)

    if len(results) == 1 and results.__contains__('relationship'):
        edge, table_data = get_list(query_result, 'relationship')
        vertex_for_edge = create_vertex_for_edge(query_result)
        return {'graph': {'nodes': vertex_for_edge, 'edges': edge}, 'type': 'graph'}

    elif len(results) == 2 and results.__contains__('path') and results.__contains__('table'):
        return {'table': query_result['results'], 'type': 'table'}

    data = {}
    for result in results:
        edge_r_node_list, table_data = get_list(query_result, result)
        data[node_r_edge.get(result)] = edge_r_node_list

    if data.get('path'):
        return {'path': data['path'], 'graph': {
            'nodes': data.get('nodes', []), 'edges': data.get('edges', [])
        }, 'type': 'path'}

    if table_data:
        return {'table': table_data, 'type': 'table'}

    return {'graph': {'nodes': data['nodes'], 'edges': data.get('edges', [])},
            'type': 'graph'}


def generate_response(prompt):
    client = boto3.client(service_name="bedrock-runtime", region_name=bedrock_region)
    # input_token = calculate_tokens(prompt)
    response = client.invoke_model(
        modelId=MODEL_ID,
        body=json.dumps(
            {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 1024,
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
    # output_token = calculate_tokens(completion)
    return completion


def read_test_schema(bucket, org_id, job_id):
    try:
        # Fetch the file from S3
        # Fetch the file content from S3
        s3 = boto3.client('s3')
        key = f"import/embedding_schema/{org_id}/{job_id}/schema.txt"
        response = s3.get_object(Bucket=bucket, Key=key)

        # Read the file content
        file_content = response['Body'].read().decode('utf-8')
        file_lines = file_content.splitlines()

        # Debug: Check if the content is empty or invalid
        if not file_content:
            raise ValueError(f"The file '{key}' in bucket '{bucket}' is empty or invalid.")
        # Parse the JSON data
        return file_lines
    except s3.exceptions.NoSuchKey as e:
        logger.error(f'Schema not Found, Invalid {key}\n{str(e)}')
    except ValueError as ve:
        logger.error(ve)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to decode JSON: {e}")
    except Exception as e:
        logger.error(f"An error occurred: {e}")


def read_data_after_removal_imp(lines, note_start="Important Note:"):
    # Flag to track if we are inside an "Important Note" section
    inside_note = False

    # List to hold modified lines
    modified_lines = []

    try:
        for line in lines:
            # Detect the start of the "Important Note" section
            if note_start in line:
                inside_note = True  # Enter the "Important Note" section
                continue  # Skip the note's start line

            # If inside the note, skip lines until the note ends
            if inside_note:
                # Assuming a blank line or a specific condition signals the end of the note
                if re.match(r"^\s*$", line):  # Adjust this condition if needed
                    inside_note = False  # Exit the note section
                continue  # Skip the lines inside the note

            # Add the line to the modified lines if it's not part of a note
            modified_lines.append(line)

        # Join and return the modified content as a single string
        return '\n'.join(modified_lines)

    except Exception as e:
        logger.error(f"An error occurred: {e}")


def get_prompt(result, question, schema):
    with open('query_response_prompt.txt', 'r') as file:
        prompt = file.read().replace("$$RESULT$$", result).replace("$$QUESTION$$", question).replace("$$SCHEMA$$",
                                                                                                     schema)
    return prompt


def get_prompt_query(result, value):
    with open('check_job_id.txt', 'r') as file:
        prompt = file.read().replace("$$QUERY$$", result).replace("$$VALUE$$", value)
    return prompt


def get_prompt_var(result):
    with open('check_edge_variable.txt', 'r') as file:
        prompt = file.read().replace("$$QUERY$$", result)
    return prompt


def get_prompt_low(result):
    with open('check_toLower_prompt.txt', 'r') as file:
        prompt = file.read().replace("$$QUERY$$", result)
    return prompt


def get_prompt_condition(result, question):
    with open('remove_condition_prompt.txt', 'r') as file:
        prompt = file.read().replace("$$QUERY$$", result).replace("$$QUESTION$$", question)
    return prompt


def handle_neptune_errors(query_result, bucket, org_id, job_id, question):
    file = read_test_schema(bucket, org_id, job_id)
    modified_content = read_data_after_removal_imp(file)
    prompt = get_prompt(str(query_result), question, modified_content)
    result = generate_response(prompt)  # This is returning an invalid response
    logger.info(f"Raw response from generate_response: {repr(result)}")  # Debug log

    # if not res or res.strip() == "":  # Check if response is empty
    #     raise ValueError("Received empty response from generate_response")
    #
    # try:
    #     result = extract_and_validate_json(res)  # Parse JSON safely
    # except json.JSONDecodeError as e:
    #     print(f"JSON decoding failed: {e}")
    #     raise ValueError(f"Invalid JSON from generate_response: {repr(res)}")
    #
    # print("Parsed JSON result:", result)
    return result


def check_job_id_in_query(query):
    # Convert the query to lowercase to avoid case-sensitivity issues
    query_lower = query.lower()

    # Check if 'job_id' is present in the WHERE clause
    if 'where' in query_lower and 'job_id' not in query_lower:
        return True  # Missing job_id
    else:
        return False


def check_edge_variables(query):
    # Regular expression to find edges in the query
    relationship_pattern = r"-\[(.*?)\]-"
    # Extract all relationships
    relationships = re.findall(relationship_pattern, query)

    for rel in relationships:
        # Check if the relationship contains valid variable and type
        if not re.match(r"^\s*[a-zA-Z_][a-zA-Z0-9_]*\s*:\s*`[^`]+`\s*$", rel) and not re.match(r"^\s*:\s*`[^`]+`\s*$",
                                                                                               rel):
            return True  # Return True for incorrect relationships
    return False  # Return False for correct relationships

# def check_missing_values(query_result):
#     """Check if any key in the query result has a missing or None value."""
#     if isinstance(query_result, list):
#         for record in query_result:
#             if isinstance(record, dict):  # Ensure each record is a dictionary
#                 if any(value is None or value == "" for value in record.values()):
#                     return True
#     return False

def check_query_result(query_result, connection_id, message_to_queue, event_params):
    start_time = message_to_queue['time_taken']
    exec_time = t.time() - start_time
    org_id = event_params['org_id']
    job_id = event_params['job_id']
    question = event_params['question']
    bucket = os.environ.get('BUCKET_NAME')
    message_to_dispatcher_queue = {'q_id': message_to_queue['q_id'],
                                   'statuscode': 500,
                                   'answer': 'The information you requested is not available in the current data set',
                                   'time_taken': exec_time
                                   }
    logger.info(f"in_check_query_result :  {query_result}")

    if isinstance(query_result, str):
        try:
            query_result = json.loads(query_result)
        except json.JSONDecodeError:
            logger.error("query_result is not a valid JSON string")
            query_result = {'results': [], 'error': 'Invalid query result format'}

    # **Step 2: Ensure query_result is a dictionary**
    if isinstance(query_result, list):  # If it's a list, wrap it in a dictionary
        query_result = {"results": query_result}

    if len(query_result.get('results', [])) == 0 or 'error in query' in str(query_result).lower():
        logger.info("Empty Result")
        query_result = "Empty or Error in Result"
        result = handle_neptune_errors(query_result, bucket, org_id, job_id, question)
        message_to_dispatcher_queue['answer'] = result
        send_message_to_dispatcher_queue(connection_id, message_to_dispatcher_queue, 4, org_id)
    else:
        logger.info("Executed")
        query_result = round_results(query_result)
        message_to_queue['answer'] = process_query(query_result)
        send_message_to_dispatcher_queue(connection_id, message_to_queue, 7, org_id)
        message_to_queue['answer'] = str(query_result['results'])
        send_message_to_queue(message_to_queue, org_id)


def send_message_to_queue(message_to_queue, org_id):
    sqs = boto3.client('sqs')
    queue_url = os.environ.get('QUEUE_URL')
    start_time = t.time() - message_to_queue['time_taken']
    message_to_queue['time_taken'] = start_time
    try:
        sqs.send_message(QueueUrl=queue_url,
                         MessageBody=json.dumps(message_to_queue),
                         MessageDeduplicationId=str(uuid.uuid4()),
                         MessageGroupId='POS'
                         )
        logger.info(json.dumps({"log_type": "sqs", "value": org_id}))
    except Exception as e:
        logger.error(e, exc_info=True)
        raise e


def send_message_to_dispatcher_queue(connection_id, message_to_queue, data_type, org_id):  # Send message to dispatcher queue
    sqs_client = boto3.client('sqs')
    queue_url = os.environ.get('DISPATCHER_QUEUE_URL')
    try:
        message = {
            "target": [connection_id],
            "data": {
                "message": {'answer': message_to_queue['answer'], 'q_id': message_to_queue['q_id'], 'statuscode': 200},
                'type': data_type
            }
        }
        sqs_client.send_message(
            QueueUrl=queue_url,
            MessageBody=json.dumps(message, default=str),
            MessageDeduplicationId=str(uuid.uuid4()),
            MessageGroupId=str(uuid.uuid4())
        )
        logger.info(json.dumps({"log_type": "sqs", "value": org_id}))
    except Exception as e:
        logger.error(e, exc_info=True)
        raise e


def lambda_handler(event, context):
    event_params = event['detail']
    query = event_params['query']
    question = event_params['question']
    org_id = event_params['org_id']
    logger.info(json.dumps({"log_type": "org_id", "value": org_id}))
    job_id = event_params['job_id']
    bucket = os.environ.get('BUCKET_NAME')
    connection_id = event_params['ConnectionID']
    message_to_queue = {'q_id': event_params['q_id'],
                        'statuscode': 400,
                        'query': query,
                        'question': question,
                        'answer': 'malformed query',
                        'org_id': org_id,
                        'job_id': job_id,
                        'time_taken': float(event_params['start_time']),
                        'ConnectionID': connection_id,
                        'type': event_params['type']
                        }
    try:
        logger.info(f"final query:- {query}")
        if check_job_id_in_query(query):
            logger.info("Missing job_id")
            query_result = "Empty or Error in Result"
            result = handle_neptune_errors(query_result, bucket, org_id, job_id, question)
            logger.info(result)
            message_to_queue['answer'] = result
            # logger.info(f"total_input_token:- {input_token}")
            # logger.info(f"total_output_token:- {output_token}")
            send_message_to_dispatcher_queue(connection_id, message_to_queue, 4, org_id)
        elif check_edge_variables(query):
            logger.info("Missing variable")
            query_result = "Empty or Error in Result"
            result = handle_neptune_errors(query_result, bucket, org_id, job_id, question)
            logger.info(result)
            message_to_queue['answer'] = result
            # logger.info(f"total_input_token:- {input_token}")
            # logger.info(f"total_output_token:- {output_token}")
            send_message_to_dispatcher_queue(connection_id, message_to_queue, 4, org_id)
        else:
            query_result = execute_query(query, question, context)
            check_query_result(query_result, connection_id, message_to_queue, event_params)
        return {
            'statusCode': 200,
            'body': 'Message Sent'
        }

    except Exception as e:
        trace_back = traceback.format_exc()
        error_query = f'question:{question}\nquery:{query}\nerror:{e}'
        logger.error(f'{trace_back}\n{error_query}')
        sqs_helper.queue_message(context.function_name, error_query, trace_back)
        return {
            'statusCode': 400,
            'body': str(e)
        }
