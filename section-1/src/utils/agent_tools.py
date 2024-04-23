
import os
from langchain.tools import tool, StructuredTool, BaseTool, Tool
from langchain.tools.retriever import create_retriever_tool
import logging
from langchain_core.vectorstores import VectorStoreRetriever

# Logger
logger = logging.getLogger(__name__)

@tool
def agenda_tool(note:str):
    """Utiliza esta herramienta para añadir informacion a la agenda. Únicamente añade esa información si uso la palabra 'agenda'."""
    note_file = "./agenda.txt"
    if not os.path.exists(note_file):
        open(note_file, 'w')
        
    with open(note_file, 'a') as f:
        f.writelines([note + "\n"])
    return "agenda actualizada"

def retriever_tool_fn(retriever: VectorStoreRetriever) -> Tool:
    try:
        retriever_tool = create_retriever_tool(
            retriever,
            name="retriever_tool",
            description="Utiliza esta herramienta para responder a preguntas relacionadas con la busqueda en los documentos adjuntos.",
        )
        return retriever_tool
    except Exception as e:
        raise f"Error creating the retriever tool"