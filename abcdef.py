import os
import sys
import asyncio
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain.schema import Document

# ----------------------------
# Fix Async Loop for Windows/Streamlit
# ----------------------------
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# ----------------------------
# Load environment variables
# ----------------------------
load_dotenv()

# ----------------------------
# Streamlit Page Config
# ----------------------------
st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="wide")

# ----------------------------
# Sidebar UI
# ----------------------------
with st.sidebar:
    st.header("⚙️ Configuration")
    st.write("Customize your chatbot settings here.")

    k_val = st.slider("Retriever top-k", 1, 5, 2)
    temperature = st.slider("LLM Temperature", 0.0, 1.0, 0.3)

    st.markdown("---")
    st.caption("Powered by LangChain + Gemini")

# ----------------------------
# Title
# ----------------------------
st.title("🤖 RAG Application built on Gemini Model")
st.markdown("Ask me anything about **LangChain** 📚")

# ----------------------------
# Initialize Chat History
# ----------------------------
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# ----------------------------
# Load LLM
# ----------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=temperature
)

# ----------------------------
# Load Documents (Example)
# ----------------------------
docs = [Document(page_content="LangChain helps build chatbots using LLMs and retrieval.")]
text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
split = text_splitter.split_documents(docs)

# ----------------------------
# Embedding + VectorStore
# ----------------------------
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=os.environ["GOOGLE_API_KEY"]
)
vectordb = Chroma.from_documents(split, embedding=embeddings)

# ----------------------------
# Retriever
# ----------------------------
retriever = vectordb.as_retriever(search_kwargs={"k": k_val})

# ----------------------------
# Prompt Template
# ----------------------------
prompt_template = """
You are a helpful AI assistant.
Answer the question based only on the given content.

Context:
{context}

Question:
{question}

Answer:
"""
QA_PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# ----------------------------
# QA Chain
# ----------------------------
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": QA_PROMPT}
)

# ----------------------------
# Display Chat History
# ----------------------------
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ----------------------------
# User Input (Chat Box)
# ----------------------------
query = st.chat_input("💬 Ask me anything:")

if query:
    # Save user message
    st.session_state["messages"].append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Get response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = qa_chain.invoke({"query": query})
            answer = result["result"]
            st.markdown(answer)

    # Save assistant message
    st.session_state["messages"].append({"role": "assistant", "content": answer})
