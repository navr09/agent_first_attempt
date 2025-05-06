from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain.tools import tool
from langchain.agents import AgentExecutor, create_tool_calling_agent
import requests
from bs4 import BeautifulSoup
import readline
import os
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# source .venv/bin/activate
# which python

# Load the API keys from .env file
load_dotenv()
tavily_api_key = os.getenv("TAVILY_API_KEY")

# 1. Define tools

# Use the Langchain tools decorater on the method to make it recognisable as one.
# Providing the expected input and output type is essential for the agent to validate the input and parse output correctly. It also must contain a docstring 
# that gives a descrition for the tool
@tool
def tavily_search_api(query: str) -> dict:
    """
    This method takes the user query and makes an api call to tavily and provides the response
    """
    tavliy_search = TavilySearchResults(api_key=tavily_api_key, max_results=2)
    return tavliy_search.invoke({"query": query})

tools = [tavily_search_api]

# To test the tool working 
# print(tavily_search_api('Weather in SF?'))
# print(tools[0].invoke({"query": "Weather in SF?"}))

#2. Configure LLM for agent
llm = ChatGroq(model="llama3-8b-8192", temperature=0, api_key=os.getenv("GROQ_API_KEY"))  # Critical for error recovery
# Define prompt (REQUIRED). This must match the key you use in agent_executor.invoke({"input": ...}).
prompt = ChatPromptTemplate.from_messages([
    ("system", """
        1. Be polite and concise. (Max 3 sentences)
        2. Only you the tool when asked about weather. For everything else don't use the tools
        3. Use the LLM for generic questions.
        Tool available:
        - tavily_search_api: for current information (weather, news, etc.)"""),
    ("human", "{input}"), # Fallback for direct input
    MessagesPlaceholder("agent_scratchpad") # Required for tools
])
# Create agent
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True) # remove verbose=True if you want agent thoughts

# 3. Execute: 
query = "Hi, How's the weather in Bangalore?"
response = agent_executor.invoke({"input": query})

print(response["output"])
print(response)

# Test case to test the agent bluntly
# def test_agent(query: str):
#     try:
#         response = agent_executor.invoke({"input": query})
#         print("Success:", response["output"])
#         print(response)
#     except Exception as e:
#         print(f"Error: {str(e)}")
#         # Fallback to direct LLM response
#         print("Fallback response:", llm.invoke(query).content)

# # Test cases
# test_agent("What's the weather in San Francisco?")  # Should use tool
# test_agent("Tell me a joke")  # Should respond directly

# NOTE: 
# About tool diff
# test groq
# what's temp
# Human message and why?
# System prompt?
# threads and DB links
