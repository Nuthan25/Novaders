import os

env = os.getenv("ENV")
region = os.getenv("REGION")
bucket = os.getenv("BUCKET_NAME")
bucket_out = os.getenv("BUCKET_NAME_IMAGE")
dispatcher_queue_url = os.getenv("DISPATCHER_QUEUE_URL")

# OpenSearch variables
index = os.getenv("INDEX_NAME")
opensearch_host = os.getenv("OPENSEARCH_HOST")
opensearch_port = os.getenv("OPENSEARCH_PORT")

# Neptune variables
neptune_endpoint = os.getenv("NEPTUNE_ENDPOINT")
neptune_port = os.getenv("NEPTUNE_PORT")

# Claude variables
model_id = os.getenv("MODEL_ID")
model_id_hq = os.getenv("MODEL_ID_HQ")
model_id_c4 = os.getenv("MODEL_ID_C4")
image_link = os.getenv("IMAGE_LINK")

# DynamoDB variables
table_name = os.getenv("TABLE_NAME")

error_message = "Could not process the input. Please try to phrase it differently"
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
          '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
          '#c49c94', '#f7b6d2', '#c7c7c7', '#dbdb8d', '#9edae5']

graph_prompts_path = 'graph_prompts/'
html_prompts_path = 'html_prompts/'
