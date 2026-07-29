from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain.tools import tool
from langchain_tavily import TavilySearch
from typing import List
from pydantic import BaseModel, Field

load_dotenv()

class Source(BaseModel):
    """
    Schema for a source used by the agent
    """

    url: str = Field(description="The URL of the source")

class AgentResponse(BaseModel):
    """
    Schema for a response from the agent with answer and sources
    """

    answer: str = Field(description="Thr agent's answer to the query")
    sources: List[Source] = Field(default_factory=list, description="List of sources used to generate the answer")

llm = ChatOllama(model="qwen3:1.7b", temperature=0)
tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)

def main():
    print("Hello from langchain-course!")
    result = agent.invoke({"messages": HumanMessage(content="What is the weather in Tokyo?")})
    print(result)

if __name__ == "__main__":
    main()
