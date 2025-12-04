import asyncio
import os
from mcp import ClientSession
from mcp.client.sse import sse_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from langchain_groq import ChatGroq
from dotenv import load_dotenv

from traceloop.sdk import Traceloop
from traceloop.sdk.decorators import workflow

load_dotenv()
Traceloop.init(app_name="Weather-Agent", api_endpoint=os.getenv("TRACELOOP_BASE_URL"))

# Read Groq configuration from environment so models/keys can be changed without code edits.
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
model = ChatGroq(model=GROQ_MODEL, groq_api_key=os.getenv("GROQ_API_KEY"))
prompt = "You are a helpful assistant capable of getting weather forecast and weather alerts. You are provided with state or co-ordinates. Call relavant tools to complete the input task and return summarised response"
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
