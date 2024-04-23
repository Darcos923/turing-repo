import os
import time
import gradio as gr
import dotenv

from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.memory import ConversationSummaryBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import (
    ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate,
    HumanMessagePromptTemplate, PromptTemplate
)
from utils.agent_tools import agenda_tool, retriever_tool_fn

# Inport enviroment variables
dotenv.load_dotenv()

# Initialize 'vector_store' global object
vector_store = Chroma(embedding_function=OpenAIEmbeddings())

def print_like_dislike(x: gr.LikeData):
    """
    Prints the index, value, and like status of an event.

    Args:
    x (gr.LikeData): Data containing index, value, and liked status of the event.
    """
    print(x.index, x.value, x.liked)

def agent_pipeline(vectorstore: Chroma) -> AgentExecutor:
    """
    Initializes and configures an agent with necessary tools and memory components.

    Args:
    vectorstore (VectorStore): The vectorstore used for retrieving documents.

    Returns:
    AgentExecutor: Configured agent ready for executing inferences.
    """
    retriever = vectorstore.as_retriever(search_kwargs={'k': 3})
    tools = [retriever_tool_fn(retriever), agenda_tool]
    
    memory = ConversationSummaryBufferMemory(
        llm=OpenAI(), return_messages=True, max_token_limit=1500, memory_key="chat_history", output_key="output"
    )
    
    system_template = """Eres un asistente personal con herramientas que puedes utilizar."""
    
    prompt_template = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate(prompt=PromptTemplate(input_variables=[], template=system_template)),
        MessagesPlaceholder(variable_name='chat_history', optional=True),
        HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['input'], template='{input}')),
        MessagesPlaceholder(variable_name='agent_scratchpad')
    ])
    
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", streaming=True)
    
    agent = create_openai_tools_agent(llm, tools, prompt_template)
    agent_executor = AgentExecutor(
        agent=agent, tools=tools, memory=memory, verbose=True, handle_parsing_errors=True,
    )
    return agent_executor

def pdf_files(pdfs: list) -> Chroma:
    """
    Processes and stores contents of PDF files in a vectorstore.

    Args:
    pdfs (list of str): File paths to the PDF documents.

    Returns:
    VectorStore: Updated vectorstore containing embedded chunks of PDF documents.
    """
    documents = []
    for file in pdfs:
        raw_document = PyPDFLoader(file).load()
        documents.extend(raw_document)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=50, length_function=len, is_separator_regex=False,
    )
    document_chunks = text_splitter.split_documents(documents)
    vectorstore = Chroma.from_documents(document_chunks, embedding=OpenAIEmbeddings())
    return vectorstore

def add_message(history: list, message: dict):
    """
    Adds a new message or file to the conversation history and updates the vectorstore if necessary.

    Args:
    history (list of tuples): The conversation history.
    message (dict): Contains the message text and/or files.

    Returns:
    tuple: Updated history and a Gradio MultimodalTextbox component for the next input.
    """
    global vs
    if message["files"]:
        vs = pdf_files(message['files'])
    if message["text"]:
        history.append((message["text"], None))
    else:
        history.append((None, "Archivo subido"))
    return history, gr.MultimodalTextbox(value=None, interactive=True)

def bot(history: list):
    """
    Generates responses for the latest message in history using the agent.

    Args:
    history (list of tuples): The conversation history containing text and response pairs.

    Yields:
    list: Updated history with the agent's response appended.
    """
    if history[-1][0] is None:
        response = "He recibido tu/s archivo/s! 📚😉"
    else:
        response = agent.invoke({"input": history[-1][0], "chat_history": []})['output']
    history[-1][1] = ""
    for character in response:
        history[-1][1] += character
        time.sleep(0.005)
        yield history

# Initialize the agent
agent = agent_pipeline(vector_store)

# Setup Gradio interface
with gr.Blocks() as demo:
    chatbot = gr.Chatbot([], elem_id="chatbot", bubble_full_width=True, height=600)
    chat_input = gr.MultimodalTextbox(interactive=True, file_types=[".pdf"], placeholder="Indica una pregunta o sube archivo/s...", show_label=False)
    chat_msg = chat_input.submit(add_message, [chatbot, chat_input], [chatbot, chat_input])
    bot_msg = chat_msg.then(bot, chatbot, chatbot, api_name="bot_response")
    bot_msg.then(lambda: gr.MultimodalTextbox(interactive=True, file_types=[".pdf"]), None, [chat_input])
    chatbot.like(print_like_dislike, None, None)

demo.queue()
demo.launch()
