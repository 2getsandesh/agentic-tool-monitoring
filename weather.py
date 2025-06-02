import httpx
import json
from mcp.types import TextContent
from mcp.server.fastmcp import FastMCP

from traceloop.sdk import Traceloop
from traceloop.sdk.decorators import workflow
Traceloop.init(app_name="Weather-MCP-Server", api_endpoint="localhost:4317")

mcp = FastMCP("Weather")
NWS_API_BASE = "https://api.weather.gov"

async def make_weather_api_call(url):
    headers = {
        "User-Agent": "weather-app/1.0",
        "Accept": "application/geo+json"
    }
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, headers=headers, timeout=30.0)
            response.raise_for_status()
            return response.json()
        except Exception:
            return None

@mcp.tool()
async def get_weather_alerts(state: str):
    """Get weather alerts for a US state.
    Args:
        state: Two-letter US state code (e.g. CA, NY)
    """
    alerts_url = f"{NWS_API_BASE}/alerts/active/area/{state}"
    response = await make_weather_api_call(alerts_url)
    if response == None:
        return [TextContent(type="text", text=f"Unable to fetch weather alerts for {state}")]
    return [TextContent(type="text", text=json.dumps(response))]

@mcp.tool()
async def get_forecast(latitude: float, longitude: float):
    """Get weather forecast for a location.
    Args:
        latitude: Latitude of the location
        longitude: Longitude of the location
    """
    points_url = f"{NWS_API_BASE}/points/{latitude},{longitude}"
    response =await make_weather_api_call(points_url)
    if response == None:
        return [TextContent(type="text", text=f"Unable to fetch weather forecast for {latitude}, {longitude}")]
    return [TextContent(type="text", text=json.dumps(response))]

@workflow(name="weather_mcp_workflow")
def start_server():
    mcp.run(transport="sse")

if __name__ == "__main__":
    start_server()
