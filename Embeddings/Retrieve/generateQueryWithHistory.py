import json
import boto3
import logging
import constants
import opensearch
import re
from datetime import datetime
from langchain_aws import ChatBedrock
from botocore.exceptions import ClientError

logger = logging.getLogger()
logger.setLevel(logging.INFO)


class SimpleChatMemory:
    """Simple in-memory chat storage using DynamoDB backend"""

    def __init__(self, table_name: str, session_id: str, region: str, k: int = 5):
        self.table_name = table_name
        self.session_id = session_id
        self.region = region
        self.k = k  # Number of message pairs to keep
        self.dynamodb = boto3.resource('dynamodb', region_name=region)
        self.table = self.dynamodb.Table(table_name)

    def add_message(self, human_message: str, ai_message: str):
        """Add a human-AI message pair to memory"""
        try:
            timestamp = datetime.utcnow().isoformat()

            # Get existing messages
            messages = self.get_messages()

            # Add new message pair
            messages.append({
                'timestamp': timestamp,
                'human': human_message,
                'ai': ai_message
            })

            # Keep only the last k message pairs
            if len(messages) > self.k:
                messages = messages[-self.k:]

            # Store back to DynamoDB
            self.table.put_item(
                Item={
                    'SessionId': self.session_id,
                    'Messages': messages,
                    'UpdatedAt': timestamp
                }
            )
        except Exception as e:
            logger.error(f"Error adding message to memory: {e}")

    def get_messages(self) -> list:
        """Get stored messages from DynamoDB"""
        try:
            response = self.table.get_item(Key={'SessionId': self.session_id})
            if 'Item' in response:
                return response['Item'].get('Messages', [])
            return []
        except Exception as e:
            logger.error(f"Error getting messages from memory: {e}")
            return []

    def format_chat_history(self, system_prompt: str) -> str:
        """Format chat history as a string for the model"""
        messages = self.get_messages()

        formatted_history = f"System: {system_prompt}\n\n"

        if messages:
            formatted_history += "Previous conversation:\n"
            for msg in messages:
                formatted_history += f"Human: {msg['human']}\n"
                formatted_history += f"Assistant: {msg['ai']}\n\n"

        return formatted_history


class GenerateQueryWithHistory:
    def __init__(self, org_id, job_id, chat_id):
        self.org_id = org_id
        self.job_id = job_id
        self.chat_id = chat_id

    def get_graph_schema(self):
        try:
            types = {
                1: 'STRING', 2: 'INTEGER', 3: 'FLOAT', 4: 'DATE',
                5: 'BOOLEAN', 6: 'DATE', 7: 'DATE', 8: 'DATE', 9: 'DATE',
            }
            s3_client = boto3.client('s3')
            key = f"import/multifile_schema/{self.org_id}/{self.job_id}/schema.json"
            res = s3_client.get_object(Bucket=constants.bucket, Key=key)
            graph_schema = json.loads(res['Body'].read())

            schema = graph_schema['schema']

            prompt = '\n'
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

    def get_system_prompt(self):
        try:
            with open('prompt_file/system_prompt.txt', 'r') as file:
                system_prompt = file.read()

            system_prompt = system_prompt.replace('{{graph_schema}}', self.get_graph_schema()).replace('{{job_id}}',
                                                                                                       self.job_id)
            return system_prompt
        except Exception as e:
            raise Exception(f"get_system_prompt: {e}")

    def init_llm(self):
        try:
            llm = ChatBedrock(
                model_id=constants.model_id,
                region_name=constants.region,
                model_kwargs={"temperature": 0.7}
            )
            return llm
        except Exception as e:
            raise Exception(f"init_llm: {e}")

    def get_memory(self, k: int = 5):
        try:
            memory = SimpleChatMemory(
                table_name=constants.table_name,
                session_id=f'{self.job_id}_{self.chat_id}',
                region=constants.region,
                k=k
            )
            return memory
        except Exception as e:
            raise Exception(f"get_memory: {e}")

    def generate_query(self, question: str, context: str = None, error: str = None, k: int = None):
        try:
            llm = self.init_llm()
            memory = self.get_memory(k or 5)
            system_prompt = self.get_system_prompt()

            # Format the input content
            if context and error:
                content = f'Got Error:\n{error}\nContext:\n{context}\nQuestion:\n{question}'
            elif context:
                content = f'Context:\n{context}\nQuestion:\n{question}'
            elif error:
                content = f'Got Error:\n{error}\nQuestion:\n{question}'
            else:
                content = f'Question:\n{question}'

            # Get chat history and format the complete prompt
            chat_history = memory.format_chat_history(system_prompt)
            full_prompt = f"{chat_history}Human: {content}\nAssistant:"

            # Generate response using ChatBedrock directly
            response = llm.invoke(full_prompt)

            # Extract the response content
            if hasattr(response, 'content'):
                ai_response = response.content
            else:
                ai_response = str(response)

            # Store the conversation in memory
            memory.add_message(content, ai_response)

            return ai_response
        except Exception as e:
            raise Exception(f"generate_query: {e}")


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


def get_embeddings_data(client, document_type, data):
    """Retrieve embeddings data from OpenSearch"""
    try:
        if client.check_query_index(document_type):
            embedding_data = client.query_shared_index(document_type, data)
            if embedding_data:
                logger.info(f"{document_type} data present")
                return embedding_data

        logger.info(f"{document_type} data not present")
        return ""
    except Exception as e:
        raise Exception(f"get_embeddings_data: {e}")


def create_dynamodb_table_if_not_exists(table_name, region):
    """Create DynamoDB table if it doesn't exist"""
    try:
        dynamodb = boto3.client('dynamodb', region_name=region)

        try:
            # Check if table exists
            dynamodb.describe_table(TableName=table_name)
            logger.info(f"Table {table_name} already exists")
            return True
        except ClientError as e:
            if e.response['Error']['Code'] == 'ResourceNotFoundException':
                logger.info(f"Creating table {table_name}")

                # Create table
                table = dynamodb.create_table(
                    TableName=table_name,
                    KeySchema=[
                        {
                            'AttributeName': 'SessionId',
                            'KeyType': 'HASH'
                        }
                    ],
                    AttributeDefinitions=[
                        {
                            'AttributeName': 'SessionId',
                            'AttributeType': 'S'
                        }
                    ],
                    BillingMode='PAY_PER_REQUEST'
                )

                # Wait for table to be created
                waiter = dynamodb.get_waiter('table_exists')
                waiter.wait(TableName=table_name)
                logger.info(f"Table {table_name} created successfully")
                return True
            else:
                logger.error(f"Error checking/creating table: {e}")
                return False
    except Exception as e:
        logger.error(f"Error in create_dynamodb_table_if_not_exists: {e}")
        return False


def handler(question: str, org: str, job: str, chat: str, k: int = None):
    try:
        # Ensure DynamoDB table exists
        create_dynamodb_table_if_not_exists(constants.table_name, constants.region)

        model = f"model_{org}_{job}"

        ops_client = opensearch.OpenSearchClient(constants.opensearch_host, constants.opensearch_port,
                                                 constants.index, constants.region, model)
        ops_client.connect()
        embedded_question = ops_client.create_embedding(question)

        embeddings = {
            'like': get_embeddings_data(ops_client, 'like', question),
            'note': get_embeddings_data(ops_client, 'note', embedded_question),
            'file': get_embeddings_data(ops_client, 'file', embedded_question),
            'catalogue': get_embeddings_data(ops_client, 'catalogue', embedded_question)
        }

        # Initialize context as empty string instead of None
        context = ""

        for embedding in ['note', 'file', 'catalogue']:
            if embeddings.get(embedding):
                context += f'\n{embeddings[embedding]}\n'

        if embeddings.get('like'):
            context += f'\nExample Queries:\n{embeddings["like"]["query"]}\n'

        generate_query_with_history = GenerateQueryWithHistory(org, job, chat)
        generated_query = generate_query_with_history.generate_query(k=k, question=question, context=context)
        query = extract_cypher_query(generated_query)
        return query, True, embeddings['note']
    except Exception as e:
        raise Exception(f"generateQueryWithHistory: {e}")