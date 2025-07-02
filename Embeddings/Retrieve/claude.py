import os
import json
import boto3
import traceback
import constants

region = os.getenv('REGION')
model = constants.model_id
model_hq = constants.model_id_hq
model_c4 = constants.model_id_c4

client = boto3.client(service_name="bedrock-runtime", region_name=region)


def process_stream_response(response):
    try:
        result = ""
        input_tokens, output_tokens = 0, 0
        for event in response["body"]:
            chunk = json.loads(event["chunk"]["bytes"])

            if "delta" in chunk and "text" in chunk["delta"]:
                result += chunk["delta"]["text"]

            if chunk.get("type") == "message_stop" and "amazon-bedrock-invocationMetrics" in chunk:
                metrics = chunk["amazon-bedrock-invocationMetrics"]
                input_tokens = metrics.get("inputTokenCount", 0)
                output_tokens = metrics.get("outputTokenCount", 0)

        return result, input_tokens, output_tokens
    except Exception as e:
        raise RuntimeError(f"Error processing Claude stream response: {str(e)}")


def generate_stream_response(prompt: str, max_tokens: int = 2000, temperature: float = 0.7):
    try:
        payload = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }

        response = client.invoke_model_with_response_stream(
            body=json.dumps(payload),
            modelId=model,
            accept="application/json",
            contentType="application/json"
        )
        return process_stream_response(response)
    except Exception as e:
        raise RuntimeError(f"Error generating Claude stream response: {str(e)}")


def generate_response(prompt: str, max_tokens: int = 1500, temperature: float = 1, top_k: int = 250,
                      top_p: float = 0.999):
    try:
        payload = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "top_k": top_k,
            "temperature": temperature,
            "top_p": top_p,
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": prompt}],
                }
            ],
        }

        response = client.invoke_model(
            body=json.dumps(payload),
            modelId=model,
            accept="application/json",
            contentType="application/json"
        )

        body = json.loads(response["body"].read())
        text = body["content"][0]["text"]
        return text, 0, 0
    except Exception as e:
        print(traceback.format_exc())
        raise RuntimeError(f"Error generating Claude response: {str(e)}")

def read_file(file):
    try:
        with open(file, "r") as f:
            return f.read()
    except Exception as e:
        raise f"Error reading file: {str(e)}"


def get_update_prompt(file, update_prompt):
    try:
        prompt = read_file(file)
        for key, value in update_prompt.items():
            prompt = prompt.replace(key, str(value))
        return prompt
    except Exception as e:
        raise f"Error updating prompt: {str(e)}"
