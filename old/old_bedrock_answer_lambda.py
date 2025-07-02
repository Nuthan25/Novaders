import base64
import json
from io import BytesIO
import re
import boto3
import os
import logging
import uuid
import time as t
import traceback
from error_helper import sqs_helper
import ast
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import networkx as nx
import numpy as np
import io
import pandas as pd
from PIL import Image
import tiktoken
from datetime import datetime

logger = logging.getLogger()
logger.setLevel(logging.INFO)
error_message = 'Could not process the input. Please try to phrase it differently'
bedrock_region = os.environ.get('BEDROCK_REGION')
MODEL_ID = os.environ.get('MODEL_ID')


def calculate_tokens(text):
    encoder = tiktoken.get_encoding("cl100k_base")
    token_count = len(encoder.encode(text))
    return token_count

def transform_event(event):  # Read event from event-bridge
    try:
        event_params = event['Records'][0]["body"]
        event_params = json.loads(event_params)
        return event_params
    except Exception as e:
        logger.error(e, exc_info=True)
        raise e


def get_prompt(file_name):
    with open(file_name, 'r') as file:
        prompt = file.read()
    return prompt

def get_prompt_qry_resp(result, question, schema):
    result = result if result is not None else ""
    question = question if question is not None else ""
    schema = schema if schema is not None else ""

    with open('query_response_prompt.txt', 'r') as file:
        prompt = file.read().replace("$$RESULT$$", result).replace("$$QUESTION$$", question).replace("$$SCHEMA$$", schema)
    return prompt


def generate_nlp_response(prompt):  # Get response from bedrock
    bedrock_client = boto3.client(service_name='bedrock-runtime', region_name=bedrock_region)
    body = json.dumps({
        "prompt": prompt,
        "temperature": 0.5,
        "maxTokens": 8191,
    })
    response = bedrock_client.invoke_model(modelId="ai21.j2-mid-v1", body=body)
    response_body = json.loads(response.get('body').read())
    result = response_body["completions"][0]["data"]["text"]
    return result


# def generate_claude_response(prompt):
#     client = boto3.client(service_name="bedrock-runtime", region_name=bedrock_region)
#     model_id = MODEL_ID
#     input_token = calculate_tokens(prompt)
#     response = client.invoke_model(modelId=model_id, body=json.dumps(
#         {
#             "anthropic_version": "bedrock-2023-05-31", "max_tokens": 8196, "top_k": 250, "temperature": 1,
#             "top_p": 0.999,
#             "messages": [
#                 {
#                     "role": "user", "content": [{"type": "text", "text": prompt}],
#                 }
#             ],
#         }), )
#     result = json.loads(response.get("body").read())
#     answer = str(result['content'][0]['text'])
#     output_token = calculate_tokens(answer)
#     return answer, input_token, output_token

def generate_claude_response(html_prompt):
    client = boto3.client(service_name="bedrock-runtime", region_name=bedrock_region)
    # Define the prompt
    prompt = html_prompt
    # Request payload
    payload = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 4098,
        "temperature": 0.7,
        "messages": [{"role": "user", "content": prompt}],
    }
    # Invoke model with streaming
    response = client.invoke_model_with_response_stream(
        modelId=MODEL_ID,
        contentType="application/json",
        accept="application/json",
        body=json.dumps(payload),  # No "stream": True inside this
    )
    result = ""  # Initialize an empty string to store the response
    input_tokens = 0
    output_tokens = 0

    for event in response["body"]:
        chunk = json.loads(event["chunk"]["bytes"])  # Decode chunk

        if "delta" in chunk and "text" in chunk["delta"]:
            text = chunk["delta"]["text"]
            # print(text, end="", flush=True)  # Print response incrementally
            result += text  # Append to result

        # Capture input and output token counts
        if chunk.get("type") == "message_stop" and "amazon-bedrock-invocationMetrics" in chunk:
            metrics = chunk["amazon-bedrock-invocationMetrics"]
            input_tokens = metrics.get("inputTokenCount", 0)
            output_tokens = metrics.get("outputTokenCount", 0)

    return result, input_tokens, output_tokens


def extract_and_validate_json(response):
    """Extracts and validates AI-generated JSON responses."""
    try:
        # If response is a tuple, extract the first element
        if isinstance(response, tuple):
            response = response[0]  # Extract the string part

        if not isinstance(response, str):
            raise ValueError("Response must be a string or a tuple containing a string.")

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

# def check_for_table(result):
#     try:
#         # Convert string representation to a Python list
#         result_list = ast.literal_eval(result)
#
#         # Ensure the result is a list with at least 2 entries
#         if isinstance(result_list, list) and len(result_list) > 1:
#             for row in result_list:
#                 if isinstance(row, dict):  # Ensure it's a dictionary
#                     has_non_float = any(not isinstance(value, float) for value in row.values())  # Accepts int & str
#                     has_number = any(isinstance(value, (int, float)) for value in row.values())
#
#                     if has_non_float and has_number:
#                         return True  # Use table format
#
#         return False  # Otherwise, don't use table format
#
#     except (ValueError, SyntaxError):
#         return False  # If parsing fails, assume it's not table-compatible

def check_for_table(result):
    try:
        result_list = ast.literal_eval(result)  # Convert string to list

        if isinstance(result_list, list) and len(result_list) > 1:
            for row in result_list:
                if not isinstance(row, dict):  # Ensure all elements are dictionaries
                    return False
            return True  # If all rows are dictionaries, return True

        return False  # If it's not a valid list, return False

    except (ValueError, SyntaxError):
        return False  # If parsing fails, return False


def generate_html_response(event_params):
    result = event_params['answer']
    question = event_params['question']
    # Check if the result can be formatted as a table
    is_table_possible = check_for_table(result)

    # Determine the correct prompt
    if is_table_possible:
        # html_prompt = get_prompt('html_prompts/neptune_table_html_prompt.txt')
        html_prompt = get_prompt('html_prompts/neptune_table_html_prompt_AG.txt')
    else:
        html_prompt = get_prompt('html_prompts/neptune_list_html_prompt.txt')
        # html_prompt = get_prompt('html_prompts/neptune_list_html_prompt-AG.txt')

    # Format the response
    result_str = str(result)
    html_prompt = html_prompt.replace('$$RESULT$$', result_str).replace('$$QUESTION$$', question)
    ans, input_token_html, output_token_html = generate_claude_response(html_prompt)
    logger.info(f"Claude response: {ans}")
    if not ans or ans.strip() == "":
        raise ValueError("Received empty JSON response from Claude")
    # Ensure it's a valid JSON response
    if is_table_possible:
        try:
            data = extract_and_validate_json(ans)  # Prefer json.loads() for structured data
        except json.JSONDecodeError as e:
            logger.error(f"JSON decoding failed: {e}")
            raise ValueError(f"Claude returned invalid JSON: {ans}")
        logger.info(f"json data: {data}")
    else:
        data = ans
    return data, input_token_html, output_token_html

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

def handle_neptune_errors(query_result, bucket, org_id, job_id, question):
    file = read_test_schema(bucket, org_id, job_id)
    modified_content = read_data_after_removal_imp(file)
    prompt = get_prompt_qry_resp(str(query_result), question, modified_content)
    result, input_token, output_token = generate_response(prompt)  # This is returning an invalid response
    logger.info(f"Raw response from generate_response: {repr(result)}")  # Debug log
    return result, input_token, output_token

def validate_response(response):
    try:
        # Check if 'json' type is present in the response
        json_entry = next((item for item in response if item.get("type") == "json"), None)

        if json_entry is None:
            return False

        # Check if 'value' key is present in the json_entry
        # if "value" not in json_entry:
        #     return False

        # Check if 'value' is a list
        if not isinstance(json_entry["value"], list):
            return False

        # Handle case where list is empty or contains empty dictionaries
        if not json_entry["value"]:
            return False

        all_none = True  # Track if all values are None

        for item in json_entry["value"]:
            if not isinstance(item, dict):
                return False
            if not item:
                return False
            if any(k is None or k == "" for k in item.keys()):
                return False

            # Check if at least one value is not None
            if any(v is not None for v in item.values()):
                all_none = False

        return not all_none  # Return False only if all values are None

    except Exception:
        return False

def generate_answer(event_params, message_to_queue):
    job_id = event_params.get('job_id')
    org_id = event_params.get('org_id')
    bucket = os.environ.get('BUCKET_NAME_IN')
    question = event_params.get('question')
    result = event_params.get('answer')
    input_token_html = 0
    output_token_html = 0
    input_token = 0
    output_token = 0

    if event_params['type'] == '1':  # Neptune Opencypher
        answer_prompt = get_prompt('neptune_answer_prompt.txt')
        html_result, input_token_html, output_token_html = generate_html_response(event_params)
        logger.info(f"json response from claude:-{html_result}")
        if check_for_table(result):
            if validate_response(html_result):
                message_to_queue['answer'] = html_result
            else:
                query_result = "error"
                result, input_token, output_token = handle_neptune_errors(query_result, bucket, org_id, job_id, question)
                message_to_queue['answer'] = result
        else:
            message_to_queue['answer'] = html_result

    elif event_params['type'] == '2':  # SQL RDS
        answer_prompt = get_prompt('sql_answer_prompt.txt')
    input_token_html = input_token_html + input_token
    output_token_html = output_token_html + output_token
    # answer_prompt = answer_prompt.replace('$$RESULT$$', query_result).replace('$$QUESTION$$', question)
    # result = generate_nlp_response(answer_prompt)

    message_to_queue['statuscode'] = 200
    return message_to_queue, input_token_html, output_token_html


def send_message_to_queue(connection_id, message_to_queue, org_id):  # Send message to dispatcher queue
    sqs_client = boto3.client('sqs')
    queue_url = os.environ.get('QUEUE_URL')
    try:
        message = {
            "target": [connection_id],
            "data": {
                "message": message_to_queue,
                'type': 4
            }
        }
        sqs_client.send_message(QueueUrl=queue_url,
                                MessageBody=json.dumps(message, default=str),
                                MessageDeduplicationId=str(uuid.uuid4()),
                                MessageGroupId=str(uuid.uuid4())
                                )
        logger.debug("Message pushed")
        logger.info(json.dumps({"log_type": "sqs", "value": org_id}))
    except Exception as e:
        logger.error(e, exc_info=True)
        raise e


# def error_message_to_queue(log_message, connection_id, message_to_queue):  # Send error message to dispatcher queue
#     start_time = message_to_queue['time_taken']
#     message_to_queue['time_taken'] = t.time() - start_time
#     send_message_to_queue(connection_id, message_to_queue)


def get_graph_type(answer):
    with open('graph_type_prompt.txt', 'r') as file:
        graph_type_prompt = file.read()
    graph_type_prompt = graph_type_prompt.replace('$$QUERY_RESULT', str(answer))
    graph_type, input_token, output_token = generate_claude_response(graph_type_prompt)
    return graph_type, input_token, output_token


def generate_response(prompt):
    client = boto3.client(service_name="bedrock-runtime", region_name=bedrock_region)
    model_id = MODEL_ID
    input_token = calculate_tokens(prompt)
    response = client.invoke_model(
        modelId=model_id,
        body=json.dumps(
            {
                "anthropic_version": "bedrock-2023-05-31", "max_tokens": 1024, "top_k": 250, "temperature": 1,
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
    output_token = calculate_tokens(completion)
    return completion, input_token, output_token


def get_graph_title(question):
    title_creation_prompt = get_prompt('graph_prompts/graph_title_creation_prompt.txt')
    title_creation_prompt = title_creation_prompt.replace("$$QUESTION$$", question)
    graph_title, input_token, output_token = generate_response(title_creation_prompt)
    return graph_title, input_token, output_token


def generate_labels(file_name, question, graph_json):
    label_prompt = get_prompt(file_name)
    label_prompt = label_prompt.replace("$$QUESTION$$", question).replace("$$GRAPH_JSON$$", str(graph_json))
    label, input_token, output_token = generate_response(label_prompt)
    return label, input_token, output_token


def generate_graph_descript(xLabel, yLabel, gTitle, gJson, gType):
    graph_prompt = get_prompt('graph_prompts/graph_description_prompt.txt')
    graph_prompt = graph_prompt.replace('$$XLABEL$$', xLabel).replace('$$YLABEL$$', yLabel).replace('$$GTITLE$$',
                                                                                                    gTitle).replace(
        '$$GJSON$$', str(gJson)).replace('$$GTYPE$$', gType)
    description, input_token, output_token = generate_response(graph_prompt)
    return str(description), input_token, output_token


def check_graph_json(graph_json):
    for key, value in graph_json.items():
        if not isinstance(value, (int, float)):
            return True


def abbreviate_x_value(x_values):
    abbreviated_labels = []
    for x_value in x_values:
        if len(x_value) > 3:
            if x_value[:3] in abbreviated_labels and x_value[:4] in abbreviated_labels:
                abbreviated_labels.append(x_value[:5])
            elif x_value[:3] in abbreviated_labels:
                abbreviated_labels.append(x_value[:4])
            else:
                abbreviated_labels.append(x_value[:3])
        else:
            abbreviated_labels.append(x_value)
    return abbreviated_labels


def plot_pareto_by(df, x, y, hlines=[80]):
    df['Cumulative Percentage'] = df[y].cumsum() / df[y].sum() * 100

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(df[x], df[y], color='C0')
    ax2 = ax.twinx()
    ax2.plot(df[x], df['Cumulative Percentage'], color='C1', marker="D", ms=7)
    ax2.yaxis.set_major_formatter(PercentFormatter())
    ax.tick_params(axis='y', colors='C0')
    ax2.tick_params(axis='y', colors='C1')

    for tick in ax.get_xticklabels():
        tick.set_rotation(45)

    plt.title(f'Pareto Chart for {x} by {y}')
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax2.set_ylabel('Cumulative Percentage')

    for hline_at in hlines:
        ax2.axhline(y=hline_at, color='red', linestyle='dotted')


def create_graph_image(graph_json, graph_type, question, org_id, job_id):
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
              '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
              '#c49c94', '#f7b6d2', '#c7c7c7', '#dbdb8d', '#9edae5']
    graph_title, input_token_title, output_token_title  = get_graph_title(question)
    x_label, input_token_xlabel, output_token_xlabel  = generate_labels('graph_prompts/x_label_prompt.txt', question, graph_json)
    y_label, input_token_ylabel, output_token_ylabel  = generate_labels('graph_prompts/y_label_prompt.txt', question, graph_json)
    try:
        if 'network graph' in graph_type.lower():
            g = nx.DiGraph()
            for node in graph_json["nodes"]:
                g.add_node(node["id"], group=node["group"])
            for link in graph_json["links"]:
                g.add_edge(link["source"], link["target"], weight=link["value"])
            pos = nx.spring_layout(g)
            nx.draw_networkx_nodes(g, pos, node_size=700)
            nx.draw_networkx_edges(g, pos, width=2)
            nx.draw_networkx_labels(g, pos, font_size=10, font_family="sans-serif")
        elif 'stacked' in graph_type.lower():
            # Preprocessing for stacked bar chart
            categories = list(graph_json.keys())  # Outer keys (e.g., genres)
            items = set(
                item for category_data in graph_json.values() for item in category_data)  # Inner keys (e.g., singers)

            # Prepare data for plotting
            data_for_plot = {item: [] for item in items}
            for category in categories:
                for item in items:
                    data_for_plot[item].append(graph_json[category].get(item, 0))

            # Plotting
            x = range(len(categories))  # X positions for each category
            bottom = [0] * len(categories)  # Initialize bottom values for stacking
            colors = plt.cm.tab20.colors  # Use a colormap to handle many colors dynamically

            for i, (item, counts) in enumerate(data_for_plot.items()):
                plt.bar(x, counts, bottom=bottom, label=item, color=colors[i % len(colors)])
                bottom = [b + c for b, c in zip(bottom, counts)]  # Update bottom for stacking

            # Add labels and title
            plt.xticks(x, categories, rotation=45, ha='right')
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        else:
            if check_graph_json(graph_json):
                return 'error'
            x_values = graph_json.keys()
            y_values = graph_json.values()

            if 'bar' in graph_type.lower():
                abbreviated_labels = abbreviate_x_value(x_values)
                bars = plt.bar(range(len(graph_json)), y_values, tick_label=abbreviated_labels)

                for i, bar in enumerate(bars):
                    bar.set_color(colors[i % len(colors)])

                # for bar, y_value in zip(bars, y_values):
                #     plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2 - 0.2, y_value, ha='center',
                #              va='center', color='black', fontsize=8, rotation=90)

                plt.ylim(0, max(y_values) * 1.1)
                plt.xticks(rotation=45)
                legend_labels = [f"{full_label} ({abbreviated_label})" for full_label, abbreviated_label in
                                 zip(x_values, abbreviated_labels)]
                plt.legend(handles=bars, labels=legend_labels, loc='center left', bbox_to_anchor=(1, 0.5))

            elif 'pareto' in graph_type.lower():
                df = pd.DataFrame(list(graph_json.items()), columns=[x_label, y_label])
                df = df.sort_values(by=y_label, ascending=False).reset_index(drop=True)
                plot_pareto_by(df, x=x_label, y=y_label)

            elif 'scatter' in graph_type.lower():
                # Convert dict_keys to a list for indexing
                x_values = list(graph_json.keys())
                y_values = list(graph_json.values())
                abbreviated_labels = abbreviate_x_value(x_values)  # Abbreviate x-axis labels
                # Scatter plot with colors
                scatter = plt.scatter(range(len(x_values)), y_values, color=colors[:len(x_values)], s=100, alpha=0.7,
                                      edgecolors='black')
                # Manually create legend with category names, abbreviated forms, and colors
                legend_elements = [
                    plt.Line2D([0], [0], marker='o', color='w',
                               label=f"{x_values[i]} ({abbreviated_labels[i]})",  # Full name and abbreviation
                               markerfacecolor=colors[i], markersize=10)
                    for i in range(len(x_values))
                ]
                plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))
                # Set x-axis labels to abbreviated labels
                plt.xticks(ticks=range(len(x_values)), labels=abbreviated_labels, rotation=45, ha='right')
                plt.style.use("seaborn-v0_8-whitegrid")
                plt.grid(visible=True, linestyle="--", linewidth=0.5)

            elif 'line' in graph_type.lower():
                # Line Graph with custom colors and legend
                plt.plot(x_values, y_values, marker='o', color=colors[0], linewidth=2)
                plt.fill_between(x_values, y_values, color=colors[1], alpha=0.3)  # Optional shading below line
                # Creating the full legend with category name and its color
                plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                plt.style.use("seaborn-v0_8-whitegrid")
                plt.xticks(rotation=45, ha='right')

            elif 'pie' in graph_type.lower():  # New condition for pie chart
                labels = list(graph_json.keys())  # Category labels
                sizes = list(graph_json.values())  # Data values
                # Calculate percentages
                total = sum(sizes)
                percentages = [(size / total) * 100 for size in sizes]
                myexplode = [0.01] * len(labels)
                # Create pie chart (no percentages inside slices)
                wedges, texts = plt.pie(
                    sizes,
                    labels=None,  # Disable default labels on slices
                    colors=colors[:len(labels)],
                    explode=myexplode,
                    startangle=140
                )
                # Create labels for the legend with percentages
                legend_labels = [f"{label} ({percentage:.1f}%)" for label, percentage in zip(labels, percentages)]
                # Add legend to the right-hand side
                plt.legend(
                    wedges, legend_labels,
                    loc="center left",
                    bbox_to_anchor=(1, 0.5),  # Position legend on the right
                    frameon=False
                )
                plt.axis('equal')  # Equal aspect ratio for a perfect circle


            elif 'donut' in graph_type.lower():  # New condition for donut chart
                labels = list(graph_json.keys())  # Category labels
                sizes = list(graph_json.values())  # Data values
                # Calculate percentages
                total = sum(sizes)
                percentages = [(size / total) * 100 for size in sizes]
                myexplode = [0.01] * len(labels)

                # Create donut chart
                wedges, texts = plt.pie(
                    sizes,
                    labels=None,  # Disable default labels on slices
                    colors=colors[:len(labels)],
                    startangle=140,
                    explode=myexplode,
                    wedgeprops={'width': 0.3}  # Adjust width for donut hole
                )

                # Create labels for the legend with percentages
                legend_labels = [f"{label} ({percentage:.1f}%)" for label, percentage in zip(labels, percentages)]

                # Add legend to the right-hand side
                plt.legend(
                    wedges, legend_labels,
                    loc="center left",
                    bbox_to_anchor=(1, 0.5),  # Position legend on the right
                    frameon=False
                )

                plt.axis('equal')  # Equal aspect ratio for a perfect circle
            else:
                return 'error'

        logger.info(f"x_label:-- {x_label}")
        logger.info(f"y_label:-- {y_label}")
        logger.info(f"graph_title:-- {graph_title}")
        logger.info(f"graph_json:-- {graph_json}")
        logger.info(f"graph_type:-- {graph_type}")
        graph_description, input_token_desc, output_token_desc = generate_graph_descript(x_label, y_label, graph_title, graph_json, graph_type)
        total_graph_image_in_token = input_token_desc + input_token_xlabel + input_token_ylabel + input_token_title
        total_graph_image_out_token = output_token_desc + output_token_xlabel + output_token_ylabel + output_token_title
        logger.info(f"graph_description:--{graph_description}")
        if 'donut' or 'pie' in graph_type.lower():
            plt.xlabel(x_label)
        else:
            plt.xlabel(x_label)
            plt.ylabel(y_label)
        plt.title(graph_title)
        graph_type = graph_type.replace(" ", "_")

        temp_filename = f"/tmp/{graph_type}_{uuid.uuid4()}.png"
        plt.savefig(temp_filename, format='png', bbox_inches="tight", dpi=100)
        plt.clf()

        # Upload the image to S3
        bucket_name = os.environ.get('BUCKET_NAME')
        image_link = os.environ.get('IMAGE_LINK')
        s3_client = boto3.client('s3')
        s3_key = f"graph/{org_id}/{job_id}/{graph_type}_{uuid.uuid4()}.png"
        s3_client.upload_file(
            Filename=temp_filename,
            Bucket=bucket_name,
            Key=s3_key,
            ExtraArgs={
                'ContentType': 'image/png',  # Correct MIME type
            }
        )

        # Generate the public URL for the image
        image_url = f"https://{image_link}/{s3_key}"

        # Clean up the temporary file
        os.remove(temp_filename)

        return image_url, graph_description, total_graph_image_in_token, total_graph_image_out_token
    except Exception as e:
        logger.info(e)
        return 'error', 'error', total_graph_image_in_token, total_graph_image_out_token


def check_graph(question):
    prompt = get_prompt('graph_prompts/graph_check_prompt.txt')
    prompt = prompt.replace("$$QUESTION$$", question)
    graph_type, input_token, output_token = generate_claude_response(prompt)
    if 'table' in graph_type.lower():  # Table is also a type of graph
        graph_type = 'None'
    return graph_type, input_token, output_token


def check_table(question):
    prompt = get_prompt('graph_prompts/table_check_prompt.txt')
    prompt = prompt.replace("$$QUESTION$$", question)
    table, input_token, output_token = generate_claude_response(prompt)
    return table


def generate_graph(event_params, graph_type, context):
    question = event_params['question']
    try:
        answer = event_params['answer']
        org_id = event_params['org_id']
        job_id = event_params['job_id']
        answer = ast.literal_eval(answer)
        logger.info(f"event_params:- {event_params}")
        logger.info(f"graph_type:- {graph_type}")
        logger.info(f"answer:- {answer}")
        logger.info(f"context:- {context}")
        if len(answer[0]) == 1:
            return None
        if 'stacked' in graph_type.lower():
            stacked_prompt = get_prompt('graph_prompts/stacked_graph_creation_prompt.txt')
            stacked_prompt = stacked_prompt.replace('$$QUERY_RESULT$$', str(answer)).replace('$$GRAPH_TYPE$$',
                                                                                             graph_type).replace(
                '$$QUESTION$$', question)
            graph_json, input_token, output_token = generate_claude_response(stacked_prompt)
            logger.info(f"graph_json:- {graph_json}")
        else:
            graph_prompt = get_prompt('graph_prompts/graph_creation_prompt.txt')
            graph_prompt = graph_prompt.replace('$$QUERY_RESULT$$', str(answer)).replace('$$GRAPH_TYPE$$', graph_type)
            graph_json, input_token, output_token = generate_claude_response(graph_prompt)
            logger.info(f"graph_json:- {graph_json}")
        graph_json = json.loads(graph_json)
        graph_raw, graph_description, input_token_graph_img, output_token_graph_img = create_graph_image(graph_json, graph_type, question, org_id, job_id)
        logger.info(f"graph_raw:- {graph_raw}")
        total_input_token_graph = input_token + input_token_graph_img
        total_output_token_graph = output_token + output_token_graph_img
        if graph_raw == 'error':
            logger.info('Graph Not Supported')
            return 'error', 'error',  total_input_token_graph, total_output_token_graph
        else:
            logger.info(graph_raw)
            return graph_raw, graph_description, total_input_token_graph, total_output_token_graph


    except Exception as e:
        trace_back = traceback.format_exc()
        logger.error(trace_back)
        sqs_helper.queue_message(context.function_name, question, trace_back)
        return 'error'


def resize_graph(graph):
    try:
        # Create a buffer to hold the resized image
        buffer = io.BytesIO()
        # Decode the base64 string to get the image data
        imgdata = base64.b64decode(graph)
        # Open the image using PIL
        img = Image.open(io.BytesIO(imgdata))
        # Define the coordinates of the chart area (left, top, right, bottom)
        chart_coords = (120, 50, 480, 410)  # Adjust based on the location of the pie chart
        # Extract the chart area
        left, top, right, bottom = chart_coords
        chart_area = img.crop((left, top, right, bottom))
        # Resize the chart area to make it square
        square_size = min(chart_area.size)  # Smallest dimension to maintain circularity
        resized_chart = chart_area.resize((square_size, square_size), Image.Resampling.LANCZOS)
        # Paste the resized chart back onto the original image
        img.paste(resized_chart, (left, top))
        # Save the modified image to the buffer in PNG format
        img.save(buffer, format="PNG")
        # Encode the modified image back to base64
        img_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        # Return the base64 string without any extra characters
        return img_b64
    except Exception as e:
        logger.info(e)
        return 'error'

# def clean_izzy_response(res):
#     match_queries = re.findall(r'```json(.*?)```', res, re.DOTALL)
#     cleaned_queries = [re.sub(r'\s+', ' ', query).strip() for query in match_queries]
#     cleaned_string = cleaned_queries[0]  # Extract the string from the list
#     data_list = json.loads(cleaned_string)
#     return data_list

def lambda_handler(event, context):
    start_time = t.time()
    logger.info(event)
    event_params = transform_event(event)
    org_id = event_params.get('org_id')
    logger.info(json.dumps({"log_type": "org_id", "value": org_id}))
    question = event_params['question']
    connection_id = event_params['ConnectionID']
    query = None
    input_token_html_graph = 0
    output_token_html_graph = 0
    total_input_token_graph = 0
    total_output_token_graph = 0
    input_token_html = 0
    output_token_html = 0

    if 'query' in event_params.keys():
        query = event_params['query']

    prev_lambda_time = float(event_params['time_taken'])
    message_to_queue = {'q_id': event_params['q_id'],
                        'statuscode': event_params['statuscode'],
                        'answer': error_message,
                        'query': query
                        }
    try:
        graph_question, input_token_graph, output_token_graph = check_graph(question)
        if 'none' not in graph_question.lower():
            logger.info('Creating Graph')
            graph, description, total_input_token_graph, total_output_token_graph  = generate_graph(event_params, graph_question, context)
            html_prompt = get_prompt('graph_prompts/graph_table_html_prompt.txt')
            logger.info('got html prompt')
            result = event_params['answer']
            html_prompt = html_prompt.replace('$$RESULT$$', result).replace('$$QUESTION$$', question)
            ans, input_token_html_graph, output_token_html_graph = generate_claude_response(html_prompt)
            answer = extract_and_validate_json(ans)
            # answer = clean_izzy_response(answer)
            logger.info(f"answer:- {answer}")
            if validate_response(answer):
                graph_output = "<p>The requested chart type cannot be generated due to incompatible data or missing information.</p>"
            else:
                graph_output = [
                    {
                        "type": "json",
                        "value": answer
                    },
                    {
                        "type": "html",
                        "value": f"""<br>
                                    <div class="image_izzy">
                                        <a href="{graph}" target="_blank">
                                            <img alt="graph" src="{graph}" />
                                        </a>
                                    </div>
                                    {description}
                                 """
                    }
                ]
            message_to_queue['answer'] = graph_output
            message_to_queue['statuscode'] = 200
            print(f"graph_output:- {graph_output}")
        else:
            message_to_queue, input_token_html, output_token_html = generate_answer(event_params, message_to_queue)
        end_time = t.time() - start_time
        total_input_token = input_token_html_graph + input_token_graph + total_input_token_graph + input_token_html
        total_output_token = output_token_html_graph + output_token_graph + total_output_token_graph + output_token_html
        logger.info(f"total_input_token:- {total_input_token}")
        logger.info(f"total_output_token:- {total_output_token}")
        timestamp = datetime.utcnow().isoformat()
        logger.info(
            json.dumps(
                {
                    "timestamp": timestamp,
                    "function_name": context.function_name,
                    "org-id": org_id,
                    "total_input_token": total_input_token,
                    "total_output_token": total_output_token,
                }
            )
        )

        message_to_queue['time_taken'] = end_time + prev_lambda_time
        send_message_to_queue(connection_id, message_to_queue, org_id)
        return {
            'statuscode': 200,
            'body': 'Message Sent'
        }
    except Exception as e:
        trace_back = traceback.format_exc()
        logger.error(trace_back)
        sqs_helper.queue_message(context.function_name, question, trace_back)
        return {
            'statuscode': 500,
            'message': str(e)
        }
