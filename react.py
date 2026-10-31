from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
import os
from langchain_ollama import ChatOllama

load_dotenv()

@tool
def triple(num: float) -> float:
    """
    param num: a number to triple
    returns: the triple of the given number
    """
    return 3 * float(num)

tools = [TavilySearch(max_results=1), triple]

llm = ChatOllama(model="gemma4:e4b", temperature=0).bind_tools(tools)

