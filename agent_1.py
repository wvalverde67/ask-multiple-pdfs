import os
from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.callbacks import StreamlitCallbackHandler



import streamlit as st

load_dotenv()
os.environ["GOOGLE_CSE_ID"] = "928d54af02f30487d"
os.environ["GOOGLE_API_KEY"] = "AIzaSyC_tTrpUCHaRTunCxKTqNIqxdp7vgCCLcE"

st.title("LangChain Agent")

llm = OpenAI(temperature=0.5, streaming=True, openai_api_key=os.getenv("OPENAI_API_KEY"))
tools = load_tools(["ddg-search"])

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    # verbose=True
)

# try: "what are the names of the kids of the 44th president of america"
# try: "top 3 largest shareholders of nvidia"
if prompt := st.chat_input():
    st.chat_message("user").write(prompt)
    with st.chat_message("assistant"):
        st.write("🧠 thinking...")
        st_callback = StreamlitCallbackHandler(st.container())
        response = agent.run(prompt, callbacks=[st_callback])
        st.write(response)