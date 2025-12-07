# step 1: Install Langchain & dependencies
# pip install langchain langchain_community langchain-ollama langchain-google-genai duckduckgo-search

# step 2: Import Required Classes
import os
from langchain.agents import initialize_agent, AgentType, Tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.llms import ollama
from langchain.tools import DuckDuckGoSearchRun

# step 3: Define Tools (extra abilities like search, calculator, python execution)
search = DuckDuckGoSearchRun()

tools = [
    Tool(
        name="Search",
        func=search.run,
        description="Useful for searching the internet to find information."
    )
]

# step 4: Load LLM (Google Gemini)
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

# step 5: Initialize the Agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,  # tells agent how to use tools
    verbose=True
)

# step 6: Run the Agent
response = agent.invoke("Who is the Prime Minister of India?")
print(response)
