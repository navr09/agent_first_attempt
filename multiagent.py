import os
from dotenv import load_dotenv
from langgraph.prebuilt import create_react_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain.tools import tool
import requests
from bs4 import BeautifulSoup
import readline
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# source .venv/bin/activate
# 1. Initialize Memory Store
chat_history = InMemoryChatMessageHistory()

load_dotenv()
tavily_api_key = os.getenv("TAVILY_API_KEY")

@tool
def get_indian_stock_analysis(symbol: str):
    """Get fundamental analysis and summary for Indian stocks (NSE/BSE symbols)"""
    try:
        # Use Alpha Vantage (free tier)
        url = f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={symbol}.BSE&apikey={os.getenv('ALPHA_VANTAGE_KEY')}"
        response = requests.get(url).json()
        
        if "Error Message" in response:
            return "Invalid stock symbol or API error"
            
        return {
            "Symbol": symbol,
            "Name": response.get("Name"),
            "Sector": response.get("Sector"),
            "PE Ratio": response.get("PERatio"),
            "Description": response.get("Description"),
            "Analysis": f"{response.get('Name')} ({symbol}) has P/E of {response.get('PERatio')} in {response.get('Sector')} sector"
        }
    except Exception as e:
        return f"Error fetching data: {str(e)}"

@tool
def get_screener_in_analysis(symbol: str):
    """Scrape fundamental data from Screener.in"""
    try:
        url = f"https://www.screener.in/company/{symbol}/"
        page = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(page.content, 'html.parser')
        
        pe = soup.find("li", {"data-testid": "PE-value"}).text
        roe = soup.find("li", {"data-testid": "ROE-value"}).text
        
        return {
            "PE Ratio": pe,
            "ROE": roe,
            "Analysis": f"{symbol} has PE {pe} and ROE {roe}"
        }
    except Exception as e:
        return f"Scraping failed: {str(e)}"
    
# Add to your tools list
tools = [TavilySearchResults(api_key=tavily_api_key, max_results=2)
         , get_indian_stock_analysis
         ,get_screener_in_analysis]

SYSTEM_PROMPT = """You are a helpful AI assistant named BobBot. Follow these rules:
1. For discussion not about stocks, just reply asking to talk about stocks only
2. Always be friendly and maintain a casual tone. Only use tools for stock information. 
3. Use get_indian_stock_analysis tool for NSE/BSE stocks. If no good information, then use get_screener_in_analysis
4. Summarise the pattern of last 6 months of the stock and provide buy and sell patterns
5. For news, use tavily_search3 to get more about the stock. Never make up information - use tools when unsure
6. Keep responses under 3 sentences"""

model = init_chat_model("llama3-8b-8192", model_provider="groq")
agent_executor = create_react_agent(model, tools)

# Add memory wrapper
agent_with_history = RunnableWithMessageHistory(
    agent_executor,
    lambda session_id: chat_history,  # Could use DB-backed storage here
    input_messages_key="input",
    history_messages_key="chat_history",
)

# 4. Continuous chat with memory
print("Chat with memory (type 'quit' to exit)\n")
config = {
    "configurable": {
        "thread_id": "terminal_session_1",  # Could make this user-specific
        "session_id": "user_123"  # Needed for RunnableWithMessageHistory
    }
}
# 4. Continuous chat with memory
print("Chat with memory (type 'quit' to exit)\n")
config = {
    "configurable": {
        "thread_id": "terminal_session_1",  # Could make this user-specific
        "session_id": "user_123"  # Needed for RunnableWithMessageHistory
    }
}
while True:
    # Get user input from terminal
    try:
        user_input = input("\nYou: ")
        if user_input.lower() in ('quit', 'exit'):
            break
            
        # Process with agent
        for step in agent_executor.stream(
            {
                "messages": [
                    SystemMessage(content=SYSTEM_PROMPT),
                    HumanMessage(content=user_input)
                ]
            },
            config,
            stream_mode="values",
        ):
            # Print AI response
            if isinstance(step["messages"][-1], AIMessage):
                print(f"\nAI: {step['messages'][-1].content}")
                
    except KeyboardInterrupt:
        print("\nSession ended by user")
        break
    except Exception as e:
        print(f"\nError: {str(e)}")
        continue

