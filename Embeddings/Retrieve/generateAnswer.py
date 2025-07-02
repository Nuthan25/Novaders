import os
import re
import ast
import uuid
import json
import boto3
import logging
# import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import time

# file imports
import claude
import constants

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Constants
COLORS = constants.colors
HTML_PROMPTS_PATH = constants.html_prompts_path
GRAPH_PROMPTS_PATH = constants.graph_prompts_path


class GenerateAnswer:
    def __init__(self, question, query, query_result, note, org_id, job_id):
        self.question = question
        self.query = query
        self.result = query_result
        self.org_id = org_id
        self.job_id = job_id
        self.note = note

    def check_graph_applicability(self):
        """Check if a graph is applicable for the given question."""
        prompt = claude.get_update_prompt(f"{GRAPH_PROMPTS_PATH}graph_check_prompt.txt", {
            '$$QUESTION$$': self.question
        })
        startTime = time.time()
        graph_type, input_tokens, output_tokens = claude.generate_response(prompt, max_tokens=100, temperature=0.7)
        print("Time taken to check graph applicability: ", time.time() - startTime)
        
        # Standardize response
        if 'table' in graph_type.lower() or 'none' in graph_type.lower():
            return None, input_tokens, output_tokens
        return graph_type, input_tokens, output_tokens

    def check_for_table(self):
        """Determine if result should be displayed as a table."""
        try:
            result_list = self.result.get("table")

            if isinstance(result_list, list) and len(result_list) > 1:
                return all(isinstance(row, dict) for row in result_list)
            return False
        except (ValueError, SyntaxError):
            return False

    def extract_and_validate_json(self, response):
        """Extract and validate JSON from AI response."""
        try:
            if isinstance(response, tuple):
                response = response[0]

            if not isinstance(response, str):
                raise ValueError("Response must be a string or a tuple containing a string.")

            response = response.strip()
            json_content = None

            # Method 1: Handle Markdown code blocks with ```json format
            if "```json" in response:
                code_block_pattern = r"```json\s*([\s\S]*?)\s*```"
                match = re.search(code_block_pattern, response)
                if match:
                    json_content = match.group(1).strip()
                else:
                    # If no closing marker, get everything after the opening marker
                    json_start = response.find("```json") + 7
                    json_content = response[json_start:].strip()

            # Method 2: Look for JSON that starts with [ or { (with or without preceding text)
            if not json_content:
                # Find JSON array or object pattern
                json_pattern = r'(\[[\s\S]*\]|\{[\s\S]*\})'
                match = re.search(json_pattern, response)
                if match:
                    json_content = match.group(1).strip()

            # Method 3: Try to extract content between $$ markers (from original code)
            if not json_content:
                match = re.search(r'\$\$(.*?)\$\$', response, re.DOTALL)
                if match:
                    json_content = match.group(1).strip()

            # Method 4: If response starts directly with JSON
            if not json_content and (response.startswith("{") or response.startswith("[")):
                json_content = response

            if not json_content:
                raise ValueError("No valid JSON found in the response.")

            # Parse and return the JSON
            return json.loads(json_content)

        except (json.JSONDecodeError, ValueError) as e:
            print(f"JSON parsing error: {e}")
            return None

    def generate_html_response(self):
        """Generate HTML response based on event parameters."""
        # Determine prompt based on data format
        is_table = self.check_for_table()
        prompt_file = f"{HTML_PROMPTS_PATH}neptune_table_html_prompt_AG.txt" if is_table else f"{HTML_PROMPTS_PATH}neptune_list_html_prompt.txt"

        # Get additional information and format prompt
        info_note = self.note
        html_prompt = claude.get_update_prompt(prompt_file, {
            '$$RESULT$$': str(self.result),
            '$$QUESTION$$': str(self.question),
            '$$NOTE$$': str(info_note),
            '$$COUNT$$': self.count_dict_values()
        })

        # Generate response
        startTime = time.time()
        ans, input_tokens, output_tokens = claude.generate_stream_response(html_prompt, max_tokens=500, temperature=0.7)
        print("Time taken to generate html response(without graph): ", time.time() - startTime)

        if not ans or ans.strip() == "":
            raise ValueError("Received empty response from Claude")
        # print(f"response from claude for table:- {ans}")
        # Process response based on format
        if is_table:
            data = self.extract_and_validate_json(ans)
            if isinstance(data,list):
                for item in data:
                    if item.get('type') == 'json':
                        item['value'] = self.result.get("table")
                if data is None:
                    raise ValueError(f"Claude returned invalid JSON: {ans}")
            else:
                data = ans
        else:
            data = ans

        return data, input_tokens, output_tokens

    def validate_graph_response(self, response):
        """Validate that the graph response is properly formatted."""
        try:
            # Check if 'json' type is present
            json_entry = next((item for item in response if item.get("type") == "json"), None)
            if json_entry is None:
                return False

            # Validate JSON structure
            if "value" not in json_entry or not isinstance(json_entry["value"], list):
                return False

            if not json_entry["value"]:
                return False

            # Check if all items are valid dictionaries
            for item in json_entry["value"]:
                if not isinstance(item, dict) or not item:
                    return False
                if any(k is None or k == "" for k in item.keys()):
                    return False
                if all(v is None for v in item.values()):
                    return False

            return True
        except Exception:
            return False

    def generate_final_answer(self):
        """Generate the final answer response."""
        input_tokens = 0
        output_tokens = 0

        # For Neptune Opencypher queries
        data, input_tokens_html, output_tokens_html = self.generate_html_response()
        input_tokens += input_tokens_html
        output_tokens += output_tokens_html

        if self.check_for_table():
            if not self.validate_graph_response(data):
                error_prompt = claude.get_update_prompt(f'{GRAPH_PROMPTS_PATH}query_response_prompt.txt', {
                    '$$RESULT$$': "error",
                    '$$QUESTION$$': self.question,
                })
                error_result, error_input_tokens, error_output_tokens = claude.generate_response(error_prompt,
                                                                                                 max_tokens=1000,
                                                                                                 temperature=0.7)
                input_tokens += error_input_tokens
                output_tokens += error_output_tokens
                return error_result, input_tokens, output_tokens

        return data, input_tokens, output_tokens

    def process_graph_data(self, graph_type):
        """Process data for graph creation."""
        # answer = ast.literal_eval(self.result)
        answer = self.result
        # Handle single-value results
        # if len(answer) == 1 and len(answer[0]) == 1:
        #     return None, None, 0, 0

        # Generate graph JSON based on type
        if 'stacked' in graph_type.lower():
            prompt_file = f"{GRAPH_PROMPTS_PATH}stacked_graph_creation_prompt.txt"
            replacements = {
                '$$QUERY_RESULT$$': str(answer),
                '$$GRAPH_TYPE$$': graph_type,
                '$$QUESTION$$': self.question
            }
        else:
            prompt_file = f"{GRAPH_PROMPTS_PATH}graph_creation_prompt.txt"
            replacements = {
                '$$QUERY_RESULT$$': str(answer),
                '$$GRAPH_TYPE$$': graph_type
            }

        prompt = claude.get_update_prompt(prompt_file, replacements)
        startTime = time.time()
        graph_json_str, input_tokens, output_tokens = claude.generate_response(prompt, max_tokens=4098, temperature=0.7)
        print("Time taken to generate graph json: ", time.time() - startTime)

        try:
            graph_json = json.loads(graph_json_str)
            return graph_json, input_tokens, output_tokens
        except json.JSONDecodeError:
            logger.error(f"Failed to decode graph JSON: {graph_json_str}")
            return None, input_tokens, output_tokens

    def get_graph_metadata(self, graph_json, graph_type):
        """Get metadata for graph (title, labels, description)."""

        # Prepare prompt for graph description generation
        desc_prompt = claude.get_update_prompt(f"{GRAPH_PROMPTS_PATH}graph_metadata_prompt.txt", {
            '$$QUESTION$$': self.question,
            '$$GRAPH_JSON$$': str(graph_json),
            '$$GTYPE$$': graph_type
        })

        # Generate response from Claude
        start_time = time.time()
        description_str, input_tokens, output_tokens = claude.generate_response(
            desc_prompt, max_tokens=1000, temperature=0.7
        )
        print("Time taken to get graph metadata: ", time.time() - start_time)

        # Convert string to JSON
        try:
            metadata_json = self.extract_and_validate_json(description_str)
            # Append token information as additional fields
            metadata_json['input_tokens'] = input_tokens
            metadata_json['output_tokens'] = output_tokens
            print(f"metadata_json:- {metadata_json}")
            return metadata_json

        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON from response: {e}")


    def abbreviate_labels(self, labels, max_length=3):
        """Create abbreviated labels for x-axis."""
        seen_abbreviations = set()
        result = []

        for label in labels:
            if len(label) <= max_length:
                result.append(label)
                seen_abbreviations.add(label)
                continue

            # Try progressively longer abbreviations until unique
            for length in range(3, 6):
                abbrev = label[:length]
                if abbrev not in seen_abbreviations:
                    seen_abbreviations.add(abbrev)
                    result.append(abbrev)
                    break
            else:
                # If no unique abbreviation found, use the first 5 chars
                result.append(label[:5])

        return result

    def create_network_graph(self, graph_json):
        """Create a network graph."""
        g = nx.DiGraph()

        for node in graph_json["nodes"]:
            g.add_node(node["id"], group=node["group"])

        for link in graph_json["links"]:
            g.add_edge(link["source"], link["target"], weight=link["value"])

        pos = nx.spring_layout(g)
        nx.draw_networkx_nodes(g, pos, node_size=700)
        nx.draw_networkx_edges(g, pos, width=2)
        nx.draw_networkx_labels(g, pos, font_size=10, font_family="sans-serif")

    def create_stacked_bar_chart(self, graph_json, metadata):
        """Create a stacked bar chart."""
        categories = list(graph_json.keys())
        items = set(item for category_data in graph_json.values() for item in category_data)

        data_for_plot = {item: [] for item in items}
        for category in categories:
            for item in items:
                data_for_plot[item].append(graph_json[category].get(item, 0))

        x = range(len(categories))
        bottom = [0] * len(categories)

        for i, (item, counts) in enumerate(data_for_plot.items()):
            plt.bar(x, counts, bottom=bottom, label=item, color=COLORS[i % len(COLORS)])
            bottom = [b + c for b, c in zip(bottom, counts)]

        plt.xticks(x, categories, rotation=45, ha='right')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    def create_bar_chart(self, graph_json, metadata):
        """Create a bar chart."""
        x_values = list(graph_json.keys())
        y_values = list(graph_json.values())

        abbreviated_labels = self.abbreviate_labels(x_values)
        bars = plt.bar(range(len(graph_json)), y_values, tick_label=abbreviated_labels)

        for i, bar in enumerate(bars):
            bar.set_color(COLORS[i % len(COLORS)])

        plt.ylim(0, max(y_values) * 1.1)
        plt.xticks(rotation=45)

        legend_labels = [f"{full_label} ({abbreviated_label})" for full_label, abbreviated_label in
                         zip(x_values, abbreviated_labels)]
        plt.legend(handles=bars, labels=legend_labels, loc='center left', bbox_to_anchor=(1, 0.5))

    def create_pareto_chart(self, graph_json, metadata):
        """Create a Pareto chart."""
        # Convert dictionary to list of tuples and sort by value (descending)
        data_items = list(graph_json.items())
        data_items.sort(key=lambda x: x[1], reverse=True)

        # Extract sorted labels and values
        labels = [item[0] for item in data_items]
        values = [item[1] for item in data_items]

        # Calculate cumulative percentages
        total_sum = sum(values)
        cumulative_sum = 0
        cumulative_percentages = []

        for value in values:
            cumulative_sum += value
            cumulative_percentage = (cumulative_sum / total_sum) * 100
            cumulative_percentages.append(cumulative_percentage)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(labels, values, color='C0')
        ax2 = ax.twinx()
        ax2.plot(labels, cumulative_percentages, color='C1', marker="D", ms=7)
        ax2.yaxis.set_major_formatter(PercentFormatter())

        ax.tick_params(axis='y', colors='C0')
        ax2.tick_params(axis='y', colors='C1')

        for tick in ax.get_xticklabels():
            tick.set_rotation(45)

        ax.set_xlabel(metadata['x_label'])
        ax.set_ylabel(metadata['y_label'])
        ax2.set_ylabel('Cumulative Percentage')

        ax2.axhline(y=80, color='red', linestyle='dotted')

    def create_scatter_plot(self, graph_json, metadata):
        """Create a scatter plot."""
        x_values = list(graph_json.keys())
        y_values = list(graph_json.values())
        abbreviated_labels = self.abbreviate_labels(x_values)

        scatter = plt.scatter(
            range(len(x_values)),
            y_values,
            color=[COLORS[i % len(COLORS)] for i in range(len(x_values))],
            s=100,
            alpha=0.7,
            edgecolors='black'
        )

        # Create legend
        legend_elements = [
            plt.Line2D(
                [0], [0],
                marker='o',
                color='w',
                label=f"{x_values[i]} ({abbreviated_labels[i]})",
                markerfacecolor=COLORS[i % len(COLORS)],
                markersize=10
            ) for i in range(len(x_values))
        ]

        plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))
        plt.xticks(ticks=range(len(x_values)), labels=abbreviated_labels, rotation=45, ha='right')
        plt.style.use("seaborn-v0_8-whitegrid")
        plt.grid(visible=True, linestyle="--", linewidth=0.5)

    def create_line_graph(self, graph_json, metadata):
        """Create a line graph."""
        x_values = list(graph_json.keys())
        y_values = list(graph_json.values())

        plt.plot(x_values, y_values, marker='o', color=COLORS[0], linewidth=2)
        plt.fill_between(x_values, y_values, color=COLORS[1], alpha=0.3)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.style.use("seaborn-v0_8-whitegrid")
        plt.xticks(rotation=45, ha='right')

    def create_pie(self, graph_json, metadata):
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
            colors=COLORS[:len(labels)],
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

    def donut_chart(self, graph_json, is_donut=False):
        """Create a pie or donut chart."""
        labels = list(graph_json.keys())
        sizes = list(graph_json.values())

        # Calculate percentages
        total = sum(sizes)
        percentages = [(size / total) * 100 for size in sizes]
        myexplode = [0.01] * len(labels)

        # Create chart
        wedgeprops = {'width': 0.3} if is_donut else None
        wedges, _ = plt.pie(
            sizes,
            labels=None,
            colors=COLORS[:len(labels)],
            explode=myexplode,
            startangle=140,
            wedgeprops=wedgeprops
        )

        # Create legend with percentages
        legend_labels = [f"{label} ({percentage:.1f}%)" for label, percentage in zip(labels, percentages)]
        plt.legend(
            wedges,
            legend_labels,
            loc="center left",
            bbox_to_anchor=(1, 0.5),
            frameon=False
        )

        plt.axis('equal')

    def create_graph_image(self, graph_json, graph_type, metadata):
        """Create and save a graph image based on the specified type."""
        try:
            plt.figure(figsize=(10, 6))

            # Set graph title and labels
            plt.title(metadata['title'])

            graph_type_mapping = {
                'network': self.create_network_graph,
                'stacked': self.create_stacked_bar_chart,
                'bar': self.create_bar_chart,
                'pareto': self.create_pareto_chart,
                'scatter': self.create_scatter_plot,
                'line': self.create_line_graph,
                'pie': self.create_pie,
                'donut': self.donut_chart,
            }

            matching_type = next((key for key in graph_type_mapping.keys() if key in graph_type.lower()), None)

            if not matching_type:
                return 'error', metadata['description'], metadata['input_tokens'], metadata['output_tokens']

            print(f"graph_json:-{graph_json}")
            print(f"metadata;- {metadata}")
            graph_type_mapping[matching_type](graph_json, metadata)

            # Apply common settings (labels depend on graph type)
            if 'pie' not in graph_type.lower() and 'donut' not in graph_type.lower():
                plt.xlabel(metadata['x_label'])
                plt.ylabel(metadata['y_label'])

            # Save image to temp file
            sanitized_type = graph_type.replace(" ", "_")
            temp_filename = f"/tmp/{sanitized_type}_{uuid.uuid4()}.png"
            plt.savefig(temp_filename, format='png', bbox_inches="tight", dpi=100)
            plt.close()

            # Upload to S3
            s3_client = boto3.client('s3')
            s3_key = f"graph/{self.org_id}/{self.job_id}/{sanitized_type}_{uuid.uuid4()}.png"

            s3_client.upload_file(
                Filename=temp_filename,
                Bucket=constants.bucket_out,
                Key=s3_key,
                ExtraArgs={'ContentType': 'image/png'}
            )

            # Generate image URL and clean up
            image_url = f"https://{constants.image_link}/{s3_key}"
            os.remove(temp_filename)

            return image_url, metadata['description'], metadata['input_tokens'], metadata['output_tokens']

        except Exception as e:
            logger.error(f"Error creating graph image: {e}", exc_info=True)
            return 'error', metadata['description'], metadata['input_tokens'], metadata['output_tokens']

    def count_dict_values(self):
        """
        Count non-empty values for each key across a list of dictionaries.

        Args:
            data_list: List of dictionaries

        Returns:
            Dictionary with counts for each key
        """
        try:
            data_list = self.result.get('table', [])
            if not data_list:
                return {}

            # Get all unique keys from all dictionaries
            all_keys = set()
            for item in data_list:
                all_keys.update(item.keys())

            # Count non-empty values for each key
            counts = {}
            for key in all_keys:
                count = 0
                for item in data_list:
                    if key in item and item[key] and str(item[key]).strip():
                        count += 1
                counts[f"{key} Count"] = count
            count = str(counts).replace('{', ' " ').replace('}', ' " ')

            return f"{count} \n Total records processed: {len(data_list)}"
        except Exception as e:
            logger.error(f"Error counting dict values: {e}", exc_info=True)
            return {}

def handler(question, query, query_result, note, org_id, job_id):
    """Main function to handle the generation of answers."""
    try:
        generate_answer = GenerateAnswer(question, query, query_result, note, org_id, job_id)
        total_input_tokens = 0
        total_output_tokens = 0

        graph_type, input_tokens, output_tokens = generate_answer.check_graph_applicability()
        total_input_tokens += input_tokens
        total_output_tokens += output_tokens

        if graph_type:
            graph_json, input_tokens, output_tokens = generate_answer.process_graph_data(graph_type)
            total_input_tokens += input_tokens
            total_output_tokens += output_tokens

            if graph_json:
                metadata = generate_answer.get_graph_metadata(graph_json, graph_type)
                total_input_tokens += metadata['input_tokens']
                total_output_tokens += metadata['output_tokens']

                graph_url, description, input_tokens, output_tokens = generate_answer.create_graph_image(graph_json,
                                                                                                         graph_type,
                                                                                                         metadata)

                total_input_tokens += input_tokens
                total_output_tokens += output_tokens

                
                html_prompt = claude.get_update_prompt(f"{GRAPH_PROMPTS_PATH}graph_table_html_prompt.txt", {
                    '$$RESULT$$': query_result.get("table")[0],
                    '$$QUESTION$$': question,
                    '$$NOTE$$': str(note)
                })
                start_time = time.time()
                html_result, input_tokens, output_tokens = claude.generate_response(html_prompt, max_tokens=500,
                                                                                           temperature=0.7)
                print("Time taken to generate graph html(preprocess): ", time.time() - start_time)
                total_input_tokens += input_tokens
                total_output_tokens += output_tokens

                answer_json = generate_answer.extract_and_validate_json(html_result)
                if answer_json.get('type') == 'json':
                    answer_json['value'] = query_result.get("table")

                if generate_answer.validate_graph_response(answer_json):
                    graph_output = answer_json
                else:
                    graph_output = [
                        {
                            "type": "html",
                            "value": "<div> </div>"
                        },
                        answer_json,
                        {
                            "type": "html",
                            "value": f"""<br>
                                        <div class="image_izzy">
                                            <a href="{graph_url}" target="_blank">
                                                <img alt="graph" src="{graph_url}" />
                                            </a>
                                        </div>
                                        {description}
                                     """
                        }
                    ]

                return graph_output
            else:
                message_result, input_tokens, output_tokens = generate_answer.generate_final_answer()
                total_input_tokens += input_tokens
                total_output_tokens += output_tokens
                return message_result
        else:
            message_result, input_tokens, output_tokens = generate_answer.generate_final_answer()
            total_input_tokens += input_tokens
            total_output_tokens += output_tokens
            return message_result
    except Exception as e:
        raise e
