from langchain.agents import AgentExecutor
from langchain.memory import ConversationSummaryBufferMemory
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI, OpenAI
from utils.agent_tools import agenda_tool, retriever_tool_fn
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from utils.agent_tools import agenda_tool, retriever_tool_fn
from langchain_community.document_loaders import PyPDFLoader
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder,SystemMessagePromptTemplate, HumanMessagePromptTemplate, PromptTemplate
import os
import streamlit as st
import dotenv

dotenv.load_dotenv()

st.set_page_config(page_title="LangChain: agent chatbot", page_icon="🦜")
st.title("🦜 LangChain: Agent Chatbot")

with st.sidebar:
    # Input for OpenAI API Key
    openai_api_key = st.text_input("OpenAI API Key", type="password")

    # Check if OpenAI API Key is provided
    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()

        # Set OPENAI_API_KEY as an environment variable
    os.environ["OPENAI_API_KEY"] = openai_api_key


# Sidebar section for uploading files and providing a YouTube URL
with st.sidebar:
    uploaded_files = st.file_uploader("Please upload your files", accept_multiple_files=True, type=None)
    st.info("Please refresh the browser if you decide to upload more files to reset the session", icon="🚨")

# Check if files are uploaded or YouTube URL is provided
if uploaded_files:
    # Print the number of files uploaded or YouTube URL provided to the console
    st.write(f"Number of files uploaded: {len(uploaded_files)}")

    # Load the data and perform preprocessing only if it hasn't been loaded before
    if "processed_data" not in st.session_state:
        # Load the data from uploaded files
        documents = []
        if uploaded_files:
            for uploaded_file in uploaded_files:
                # Get the full file path of the uploaded file
                file_path = os.path.join(os.getcwd(), uploaded_file.name)

                # Save the uploaded file to disk
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getvalue())
                
                raw_document = PyPDFLoader(file_path).load()
                documents.extend(raw_document)
                print(f"Number of files loaded: {len(raw_document)}")
                
        # Chunk the data, create embeddings, and save in vectorstore
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=50,
            length_function=len,
            is_separator_regex=False,
        )
        document_chunks = text_splitter.split_documents(documents)
        embeddings = OpenAIEmbeddings()
        vectorstore = Chroma.from_documents(document_chunks, embeddings)
        
        # Store the processed data in session state for reuse
        st.session_state.processed_data = {
            "document_chunks": document_chunks,
            "vectorstore": vectorstore,
        }

    else:
        # If the processed data is already available, retrieve it from session state
        document_chunks = st.session_state.processed_data["document_chunks"]
        vectorstore = st.session_state.processed_data["vectorstore"]

# Initialize the chat history and memory
msgs = StreamlitChatMessageHistory()
memory = ConversationSummaryBufferMemory(
    llm=OpenAI(), chat_memory=msgs, return_messages=True, max_token_limit=1500, memory_key="chat_history", output_key="output"
)
if len(msgs.messages) == 0 or st.sidebar.button("Reset chat history"):
    msgs.clear()
    msgs.add_ai_message("Hola!! me llamo LangBot, ¿en qué puedo ayudarte hoy?")
    st.session_state.steps = {}

# Creating avatars and chat interface
avatars = {"human": "user", "ai": "assistant"}
for idx, msg in enumerate(msgs.messages):
    with st.chat_message(avatars[msg.type]):
        # Render intermediate steps if any were saved
        for step in st.session_state.steps.get(str(idx), []):
            if step[0].tool == "_Exception":
                continue
            with st.status(f"**{step[0].tool}**: {step[0].tool_input}", state="complete"):
                st.write(step[0].log)
                st.write(step[1])
        st.write(msg.content)

if prompt := st.chat_input(placeholder="Enter your question"):
    st.chat_message("user").write(prompt)
    
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=openai_api_key)
    retriever = vectorstore.as_retriever()
    retriever_tool = retriever_tool_fn(retriever=retriever)
    tools = [retriever_tool, agenda_tool]
    
    # Creating a prompt template to guide the agent
    prompt_template = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate(prompt=PromptTemplate(input_variables=[], template='You are a helpful assistant')),
        MessagesPlaceholder(variable_name='chat_history', optional=True),
        HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['input'], template='{input}')),
        MessagesPlaceholder(variable_name='agent_scratchpad')
        ]
    )
     # Genaerating the agent and its executor
    agent = create_openai_tools_agent(llm, tools, prompt_template)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,
        return_intermediate_steps=True,
        handle_parsing_errors=True,
    )
    
    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        cfg = RunnableConfig()
        cfg["callbacks"] = [st_cb]
        response = agent_executor.invoke({"input":prompt, "chat_history": []}, cfg)
        st.write(response["output"])
        st.session_state.steps[str(len(msgs.messages) - 1)] = response["intermediate_steps"]