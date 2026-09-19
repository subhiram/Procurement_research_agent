"""Search and extract backends."""

from .arxiv import ArxivProvider
from .base import Provider
from .brave import BraveProvider
from .crossref import CrossrefProvider
from .duckduckgo import DuckDuckGoProvider
from .exa import ExaProvider
from .firecrawl import FirecrawlProvider
from .google_cse import GoogleCSEProvider
from .hackernews import HackerNewsProvider
from .http_extract import HTTPExtractProvider
from .jina import JinaProvider
from .pubmed import PubMedProvider
from .registry import available, get, register
from .searchapi import SearchApiProvider
from .searxng import SearXNGProvider
from .serpapi import SerpAPIProvider
from .serper import SerperProvider
from .tavily import TavilyProvider
from .wikipedia import WikipediaProvider

__all__ = [
    "Provider",
    "ArxivProvider",
    "BraveProvider",
    "CrossrefProvider",
    "DuckDuckGoProvider",
    "ExaProvider",
    "FirecrawlProvider",
    "GoogleCSEProvider",
    "HackerNewsProvider",
    "HTTPExtractProvider",
    "JinaProvider",
    "PubMedProvider",
    "SearchApiProvider",
    "SearXNGProvider",
    "SerpAPIProvider",
    "SerperProvider",
    "TavilyProvider",
    "WikipediaProvider",
    "available",
    "get",
    "register",
]
