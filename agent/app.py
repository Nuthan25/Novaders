import constants
from tools import tools as _tools
from langchain_aws import ChatBedrock
# from langgraph.prebuilt import chat_agent_executor
# from langgraph.graph import END, StateGraph
# from langchain_community.llms.bedrock import Bedrock
from langchain.agents import initialize_agent
from langchain.schema import SystemMessage


system_prompt = """
When user ask for create schema for the given file. 
return the graph-relation in the following structure with columns(header)
```{
  "schema": {
    "edge": [
      {
        "fromLabel": "",
        "toLabel": "",
        "label": "",
        "fromHeader": "",
        "toHeader": "",
        "properties": [
          {
            "header": ""
          }
        ]
      }
    ],
    "vertex": [
      {
        "header": "",
        "label": "",
        "properties": [
          {
            "header": ""
          }
        ]
      }
    ]
  }
}```
"""


def init_llm():
    try:
        return ChatBedrock(
            model_id=constants.model,
            region_name=constants.region,
            model_kwargs={"temperature": 0.7, 'system':system_prompt},
            max_tokens=3072
        )
    except Exception as e:
        raise e


agent = initialize_agent(
    tools=_tools,
    llm=init_llm(),
    verbose=True,
)

while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        break

    # Run agent
    result = agent.invoke(user_input)

    response = result.get("output")
    print("Claude:", response)
