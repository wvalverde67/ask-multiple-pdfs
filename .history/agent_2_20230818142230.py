import streamlit as st

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
from langsmith import Clien