import re
import json
import boto3
import claude
import logging
import time

# file imports
import constants
import opensearch

logger = logging.getLogger()
logger.setLevel(logging.INFO)

PROMPT_CACHE = {}


def get_embeddings_data(client, document_type, data):
    """Retrieve embeddings data from OpenSearch"""
    try:
        if client.check_query_index(document_type):
            embedding_data = client.query_shared_index(document_type, data)
            if embedding_data:
                logger.info(f"{document_type} data present")
                logger.info(f"Embedding data for '{document_type}':- {embedding_data}")
                return embedding_data

        logger.info(f"{document_type} data not present")
        return ""
    except Exception as e:
        raise Exception(f"get_embeddings_data: {e}")


def get_graph_schema(job_id, org_id):
    """Retrieve and format graph schema from S3"""
    try:
        types = {
            1: 'STRING', 2: 'INTEGER', 3: 'FLOAT', 4: 'DATE',
            5: 'BOOLEAN', 6: 'DATE', 7: 'DATE', 8: 'DATE', 9: 'DATE',
        }
        key = f"import/multifile_schema/{org_id}/{job_id}/schema.json"
        s3_client = boto3.client('s3')
        res = s3_client.get_object(Bucket=constants.bucket, Key=key)
        graph_schema = json.loads(res['Body'].read())

        schema = graph_schema['schema']

        prompt = 'Graph Schema:\n'
        for edge in schema['edge']:
            prompt += f"(:{edge['fromLabel']})-[:{edge['label']}]->(:{edge['toLabel']})\n"

        node_properties = ''
        for node in schema['vertex']:
            node_properties += f"Properties of the Node ({node['label']}):\n"
            node_properties += f"$$Name$$: STRING\n"
            for prop in node['properties']:
                name = prop['replaceHeader'] if prop.get('replaceHeader') else prop['key']
                node_properties += f"{name.lower().replace(' ', '_')}: {types.get(prop['type'], 'STRING')}\n"
            node_properties += '\n'

        return f"{prompt}\n{node_properties}"
    except Exception as e:
        raise Exception(f"get_graph_schema: {e}")


def generate_query(job, files_data, question, notes, schema, catalogue, error=None):
    """Generate OpenCypher query using LLM"""
    try:
        # Prepare context
        important_note = read_file_cached("prompt_file/default_notes.txt").replace("$$ID$$", job)
        limitations = read_file_cached("prompt_file/AWS_Neptune_OpenCypher_Restrictions.txt")
        # context = f"{schema}\n{important_note}\nAdditional Info:\n{notes}"
        context = schema
        if important_note:
            context += "\n" + important_note
        if notes and catalogue:
            context += "\nAdditional Info:\n" + notes + "\n" + catalogue
        elif notes:
            context += "\nAdditional Info:\n" + notes
        elif catalogue:
            context += "\nAdditional Info:\n" + catalogue

        # Prepare prompt
        replacements = {
            "$$ID$$": job,
            "$$Limitation$$": limitations,
            "{context}": context,
            "{question}": question,
            "$$file_imp_data$$": str(files_data) if files_data else ""
        }

        if error:
            error_prompt = read_file_cached("prompt_file/error_prompt.txt").replace("$$error_response$$", error)
            replacements["$$error$$"] = error_prompt

        prompt = format_prompt("prompt_file/prompt.txt", replacements)

        return claude.generate_response(prompt, max_tokens=1000)
    except Exception as e:
        raise Exception(f"generate_query: {e}")


def read_file_cached(file_path):
    """Read file content with caching"""
    if file_path not in PROMPT_CACHE:
        try:
            with open(file_path, 'r') as file:
                PROMPT_CACHE[file_path] = file.read()
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            return ""
    return PROMPT_CACHE[file_path]


def format_prompt(template_path, replacements):
    """Format prompt template with replacements"""
    template = read_file_cached(template_path)
    for key, value in replacements.items():
        template = template.replace(key, str(value))
    return template


def extract_cypher_query(text):
    """Extract Cypher query from text response"""
    try:
        if isinstance(text, str):
            # Try to extract from code blocks
            matches = re.findall(r'```(?:cypher)?\s*(.*?)\s*```', text, re.DOTALL)
            if matches:
                return matches[0].strip()

            # If no code blocks, look for MATCH or RETURN statements
            if "MATCH" in text or "RETURN" in text:
                lines = text.split('\n')
                query_lines = []
                in_query = False

                for line in lines:
                    if "MATCH" in line or "RETURN" in line:
                        in_query = True

                    if in_query:
                        query_lines.append(line)

                    if in_query and ";" in line:
                        break

                if query_lines:
                    return ' '.join(query_lines).strip()

        return text
    except Exception as e:
        raise Exception(f"extract_cypher_query: {e}")


def handler(question: str, org: str, job: str):
    try:
        model_id = f"model_{org}_{job}"
        opensearch_client = opensearch.OpenSearchClient(constants.opensearch_host, constants.opensearch_port,
                                                        constants.index, constants.region, model_id)
        opensearch_client.connect()
        embedded_question = opensearch_client.create_embedding(question)

        embeddings = {
            'like': get_embeddings_data(opensearch_client, 'like', question),
            'note': get_embeddings_data(opensearch_client, 'note', embedded_question),
            'file': get_embeddings_data(opensearch_client, 'file', embedded_question),
            'catalogue': get_embeddings_data(opensearch_client, 'catalogue', embedded_question)
        }
        graph_schema = get_graph_schema(job, org)

        cypher_query = None
        generated = False

        if embeddings.get('like', {}):
            liked_embedded = embeddings['like']
            if question == liked_embedded['question']:
                query = liked_embedded['query']
                cypher_query = query.replace('get_job_id()', f'"{job}"')

        if not cypher_query:
            start_time = time.time()
            generated_query, input_token, output_token = generate_query(job, embeddings['file'], question,
                                                                        embeddings['note'], graph_schema,
                                                                        embeddings['catalogue'])
            print(f"Time taken to generate query: {time.time() - start_time}")
            cypher_query = extract_cypher_query(generated_query)
            generated = True

        return cypher_query, generated, embeddings['note']
    except Exception as e:
        raise Exception(f"generateQuery: {e}")
