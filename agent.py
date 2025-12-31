import asyncio
import os
import random
from mcp import ClientSession
from mcp.client.sse import sse_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt.chat_agent_executor import create_react_agent
from dotenv import load_dotenv

from traceloop.sdk import Traceloop
from traceloop.sdk.decorators import workflow, tool

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
    
    # # Test various error scenarios
    # print("\n" + "="*50)
    # print("Testing Error Tools:")
    # print("="*50)
    
    # # Test 1: Division by zero
    # print("\n1. Testing Division by Zero Error:")
    # try:
    #     error_tool_division()
    # except Exception as e:
    #     print(f"   ✗ Caught error: {type(e).__name__}: {e}")
    
    # # Test 2: KeyError
    # print("\n2. Testing KeyError:")
    # try:
    #     error_tool_key()
    # except Exception as e:
    #     print(f"   ✗ Caught error: {type(e).__name__}: {e}")
    
    # # Test 3: TypeError
    # print("\n3. Testing TypeError:")
    # try:
    #     error_tool_type()
    # except Exception as e:
    #     print(f"   ✗ Caught error: {type(e).__name__}: {e}")
    
    # # Test 4: ValueError
    # print("\n4. Testing ValueError:")
    # try:
    #     error_tool_value()
    # except Exception as e:
    #     print(f"   ✗ Caught error: {type(e).__name__}: {e}")
    
    # # Test 5: Custom Exception
    # print("\n5. Testing Custom Exception:")
    # try:
    #     error_tool_custom()
    # except Exception as e:
    #     print(f"   ✗ Caught error: {type(e).__name__}: {e}")
    
    # # Test 6: Successful tool call (multiple times for observability)
    # print("\n6. Testing Successful Tool Call (Multiple Times):")

    # try:
    #     result = hello_tool()
    #     print(f"   ✓ Call: {result}")
    # except Exception as e:
    #     print(f"   ✗ Call - Caught error: {type(e).__name__}: {e}")

    # print("\n" + "="*50)
    # print("Error Testing Complete")
    # print("="*50 + "\n")
    
    await mcp_server.cleanup()

# @tool(name="hello_tool")
# def hello_tool():
#     """A simple tool that returns a greeting"""
#     return "Hello from hello_tool"

# @tool(name="error_tool_division_by_zero")
# def error_tool_division():
#     """Tool that intentionally causes a division by zero error"""
#     result = 10 / 0  # This will raise ZeroDivisionError
#     return result

# @tool(name="error_tool_key_error")
# def error_tool_key():
#     """Tool that intentionally causes a KeyError"""
#     data = {"name": "test"}
#     return data["nonexistent_key"]  # This will raise KeyError

# @tool(name="error_tool_type_error")
# def error_tool_type():
#     """Tool that intentionally causes a TypeError"""
#     result = "string" + 123  # This will raise TypeError
#     return result

# @tool(name="error_tool_value_error")
# def error_tool_value():
#     """Tool that intentionally causes a ValueError"""
#     number = int("not_a_number")  # This will raise ValueError
#     return number

# @tool(name="error_tool_custom_exception")
# def error_tool_custom():
#     """Tool that raises a custom exception"""
#     raise Exception("This is a custom error for testing purposes")

@tool(name="print_stream_agent_tool")
def print_stream(stream):
    """Print the agent's response stream"""
    for message in stream['messages']:
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()

asyncio.run(run_agent())
