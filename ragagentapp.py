# step 1: Install dependencies
# pip install streamlit langgraph langchain_community langchain-google-genai chromadb duckduckgo-search python-dotenv

import streamlit as st
import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.agents import Tool
from langchain_core.messages import HumanMessage, AIMessage

# ---------------------------
# Step 1: Load environment variables
# ---------------------------
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("🚨 GOOGLE_API_KEY not found. Please set it in your .env file.")
    st.stop()

# ---------------------------
# Step 2: Streamlit UI
# ---------------------------
st.set_page_config(page_title="LangGraph RAG Agent", page_icon="🤖", layout="wide")
st.title("📚 LangGraph RAG Agent with Gemini")

# Helper: convert stored messages into LangChain messages
def convert_messages(msgs):
    converted = []
    for role, content in msgs:
        if role == "user":
            converted.append(HumanMessage(content=content))
        else:
            converted.append(AIMessage(content=content))
    return converted


# ---------------------------
# Step 3: File uploader
# ---------------------------
uploaded_file = st.file_uploader("Upload a text file", type=["txt"])

if uploaded_file:
    # Save file temporarily
    with open("uploaded_doc.txt", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # ---------------------------
    # Step 4: Prepare Vector Store
    # ---------------------------
    loader = TextLoader("uploaded_doc.txt")
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    documents = text_splitter.split_documents(docs)

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001", google_api_key=GOOGLE_API_KEY
    )
    vectorstore = Chroma.from_documents(documents, embedding=embeddings)
    retriever = vectorstore.as_retriever()

    # ---------------------------
    # Step 5: Define RAG Tool
    # ---------------------------
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash", temperature=0, google_api_key=GOOGLE_API_KEY
    )

    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever
    )

    tools = [
        Tool(
            name="KnowledgeBase",
            func=rag_chain.run,
            description="Use this tool to answer questions from the uploaded documents."
        )
    ]

    # ---------------------------
    # Step 6: Create LangGraph Agent
    # ---------------------------
    agent_node = create_react_agent(llm, tools)

    workflow = StateGraph(dict)
    workflow.add_node("agent", agent_node)
    workflow.set_entry_point("agent")
    workflow.add_edge("agent", END)

    app = workflow.compile()

    # ---------------------------
    # Step 7: Chat UI
    # ---------------------------
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for role, content in st.session_state.messages:
        with st.chat_message(role):
            st.write(content)

    # User input
    if prompt := st.chat_input("Ask me something about your document..."):
        st.session_state.messages.append(("user", prompt))

        # Run agent with proper message objects
        try:
            response = app.invoke({"messages": convert_messages(st.session_state.messages)})
            st.write("🔍 DEBUG:", response)  # Debugging output

            if "messages" in response and len(response["messages"]) > 0:
                answer = response["messages"][-1].content
            else:
                answer = "⚠️ No response from agent."

        except Exception as e:
            answer = f"❌ Error: {e}"

        st.session_state.messages.append(("assistant", answer))

        with st.chat_message("assistant"):
            st.write(answer)
