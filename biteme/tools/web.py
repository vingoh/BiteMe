import os

from langchain_tavily import TavilySearch

_PLACEHOLDER = "tvly-not-configured"

tavily_search = TavilySearch(
    max_results=3,
    tavily_api_key=os.environ.get("TAVILY_API_KEY", _PLACEHOLDER),
)
