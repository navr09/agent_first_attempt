# agent_first_attempt
This is my first effort at developing agents using Langchain. Here are some notes that I documented.

LangChain has 3 core libraries
1. Lagchain core - Base abstractions like messages, Documents, Runnables, History
2. Langchain - Main framework like Chains, Agents, Memory, tool (for binding)
3. Langchain-community - For 3rd party integrations like LLMs,tools, vectorstores.

Langchain Model Providers
1. langchain-openai	- GPT-4, ChatGPT, Embeddings. (Best for OpenAI models)
2. langchain-groq - Llama 3, Mixtral via Groq. (Ultra-fast inference)
3. langchain-anthropic - Claude 3	(Long-context reasoning, coding)
4. langchain-google-genai - Gemini, VertexAI.

Retrieval and RAG
1. Langchain-text-splitters - Smart document chunking
2. Langchain-vectorstores - FAISS, Chroma, Pinecone
3. Langchain-retrievers - Hybrid search, Multi-query

Advanced Tools
1. Langgraph - Multi-agent workflows (Example usecase - Agent supervision)
2. Langserve - Deploy chains as APIs (Example use - Rest API endpoints)
3. Langsmit - Debug/trace LLM calls (Production monitoring)

Storage integrations
1. Langchain-mongodb - MongoDB Atlas vector search
2. Langchain-sql - SQL database agents
3. Langchain-aws - Bedrock, S3, DynamoDB

Utility libraries
1. Langchain-cli - Project scaffolfding
2. Langchain-evaluation - Benchmark LLM outputs
3. Langchain-experimental - Cutting-edge features

Additional:
Prompt precision makes a huge difference. If the prompt isn't right, you will get errors like Error: Failed to call a function. Please adjust your prompt. See 'failed_generation' for more details.

Langchain agent tool usage
1. (old)create_react_agent
    - Characteristics:
        - Based on the ReAct framework (Reasoning + Acting)
        - Uses text-based reasoning to decide when/which tools to use
        - Generates intermediate thoughts before tool calls
        - More verbose outputs (explains reasoning steps)
    - When to Use:
        - When you need explicit reasoning traces (good for debugging)
        - With older LLMs that don't natively support tool calls
        - For educational purposes (to understand agent workflows)
2. (new)create_tool_calling_agent
    - Characteristics:
        - Designed for LLMs with native tool-calling ability (e.g., GPT-4, Llama 3, Claude 3)
        - Uses structured JSON/function calls instead of text-based reasoning
        - More efficient (fewer tokens, faster execution)
        - Cleaner output (no intermediate thoughts)
    - When to Use:
        - With modern LLMs (Groq/Llama 3, OpenAI GPT-4-turbo, etc.)
        - For production applications (better performance)
        - When you want direct tool invocation without reasoning text

System Prompt vs System message
Feature	    | System Prompt	                            | System Message
Format	    | Raw text (string)	                        | Structured SystemMessage object
Usage	    | Used in ChatPromptTemplate	            | Passed directly in messages list
Flexibility	| Static (usually hardcoded)	            | Dynamic (can be modified at runtime)
Example	    | "You are a helpful assistant named Bob."	| SystemMessage(content="You are Bob..
Which Should You Use?
- For agents with create_tool_calling_agent(): Use System Messages (part of the messages list)
- For custom chains with ChatPromptTemplate: Use System Prompts (embedded in the template)