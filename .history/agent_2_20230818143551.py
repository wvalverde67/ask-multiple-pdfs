import streamlit as st
import os

from langchain.document_loaders import RecursiveUrlLoader
from langchain.document_transformers import Html2TextTransformer
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.callbacks import StreamlitCallbackHandler
from langchain.agents import OpenAIFunctionsAgent, AgentExecutor
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain.agents.openai_functions_agent.agent_token_buffer_memory import (
    AgentTokenBufferMemory,
)
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, AIMessage, HumanMessage
from langchain.prompts import MessagesPlaceholder
from langsmith import Client


client = Client()

st.set_page_config(
    page_title="Un Chatbot diferente 🦜🔗",
    page_icon="🦜",
    layout="wide",
    initial_sidebar_state="collapsed",
)

"# Chat🦜🔗"

@st.cache_resource(ttl="1h")
def configure_retriever():
    loader = RecursiveUrlLoader("https://docs.smith.langchain.com/")
    raw_documents = loader.load()
    docs = Html2TextTransformer().transform_documents(raw_documents)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    documents = text_splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(documents, embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 4})

tool = create_retriever_tool(
    configure_retriever(),
    "search_langsmith_docs",
    "Searches and returns documents regarding LangSmith. LangSmith is a platform for debugging, testing, evaluating, and monitoring LLM applications. You do not know anything about LangSmith, so if you are ever asked about LangSmith you should use this tool.",
)

tools = [tool]

llm = ChatOpenAI(temperature=0, streaming=True, model="gpt-4")
llm = OpenAI(temperature=0.5, streaming=True, openai_api_key=os.getenv("OPENAI_API_KEY"))
message = SystemMessage(
    content=(
        "You are a helpful chatbot who is tasked with answering questions about LangSmith. "
        "Unless otherwise explicitly stated, it is probably fair to assume that questions are about LangSmith. "
        "If there is any ambiguity, you probably assume they are about that."
    )
)