import json
import boto3
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)


class NeptuneClient:
    def __init__(self, host: str, port: str):
        self.host = host
        self.port = port
        self.client = None

    def connect(self):
        try:
            session = boto3.Session()
            client_params = {
                "endpoint_url": f"https://{self.host}:{self.port}",
            }
            self.client = session.client("neptunedata", **client_params)
            logger.info("Connected to Neptune")
        except Exception as e:
            raise Exception(f"Error connecting to Neptune: {str(e)}")

    def execute_query(self, query: str):
        try:
            response = self.client.execute_open_cypher_query(
                openCypherQuery=query
            )
            logger.info("Executed query successfully")
            return 200, response
        except Exception as e:
            logger.error(f"Error executing query: {str(e)}")
            return 400, str(e)
