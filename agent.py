import asyncio
import os
import random
from mcp import ClientSession
from mcp.client.sse import sse_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv

from traceloop.sdk import Traceloop
from traceloop.sdk.decorators import workflow

from ibm_watsonx_ai.metanames import GenTextParamsMetaNames
from ibm_watsonx_ai.foundation_models import ModelInference
from langchain_ibm import ChatWatsonx

load_dotenv()
Traceloop.init(app_name="Weather-Agent", api_endpoint=os.getenv("TRACELOOP_BASE_URL"))

def watsonx_llm_init():
    watsonx_llm_parameters = {
        GenTextParamsMetaNames.DECODING_METHOD: "sample",
        GenTextParamsMetaNames.MAX_NEW_TOKENS: 100,
        GenTextParamsMetaNames.MIN_NEW_TOKENS: 1,
        GenTextParamsMetaNames.TEMPERATURE: 0.5,
        GenTextParamsMetaNames.TOP_K: 50,
        GenTextParamsMetaNames.TOP_P: 1,
    }
    models = ['ibm/granite-3-3-8b-instruct']
    model_id = random.choice(models)
    watsonx_chat = ChatWatsonx(
        model_id=model_id,
        url=os.getenv("WATSONX_URL"),
        apikey=os.getenv("WATSONX_API_KEY"),
        project_id=os.getenv("WATSONX_PROJECT_ID"),
        params=watsonx_llm_parameters,
    )
    return watsonx_chat

# Initialize WatsonX model
model = watsonx_llm_init()

prompt = "Get weather data using tools. Return one-line summary only."
mcp_server_url = "http://0.0.0.0:8000/sse"

class MCPClient:
    """
    Connect to an SSE server.
    Parameters:
    url (str): The URL of the SSE server to connect to.

    Returns:
    ClientSession: The initialized ClientSession object.
    """
    async def connect_to_sse_server(self, url):
        self.stream_context = sse_client(url = url)
        streams = await self.stream_context.__aenter__()
        self._session_context = ClientSession(*streams)
        self.session: ClientSession = await self._session_context.__aenter__()
        await self.session.initialize()
    
    async def cleanup(self):
        if self._session_context:
            await self._session_context.__aexit__(None, None, None)
        if self.stream_context:
            await self.stream_context.__aexit__(None, None, None)

@workflow(name="instana_agent_workflow")
async def run_agent():
    mcp_server = MCPClient()
    await mcp_server.connect_to_sse_server(mcp_server_url)
    tools = await load_mcp_tools(mcp_server.session)
    print(f"Tool list : {tools}")
    
    agent = create_react_agent(model, tools, prompt = prompt)
    
    user_input = input("Query : ")
    if user_input == "":
        user_input = "Get me weather alerts and forecast for NYC"
    inputs = {"messages" : user_input}
    
    response = await agent.ainvoke(inputs)
    
    print_stream(response)
    await mcp_server.cleanup()

def print_stream(stream):
    for message in stream['messages']:
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()

asyncio.run(run_agent())
