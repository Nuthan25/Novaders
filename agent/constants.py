import os
from dotenv import load_dotenv

load_dotenv()

model = os.getenv('MODEL')
region = os.getenv('REGION')