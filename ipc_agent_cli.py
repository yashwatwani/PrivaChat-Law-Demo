# ipc_agent_cli.py

import sys
import os

# Ensure the src directory is in the Python path (if running from project root)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Imports from your existing RAG system
from src.rag_query_engine import get_query_engine as get_ipc_query_engine # Renaming to avoid conflict if you had a generic one
from src.document_processor import load_documents, chunk_documents
from src.vector_store_manager import get_vector_store, create_index_from_nodes

# LlamaIndex Agent imports
from llama_index.core.tools import FunctionTool
from llama_index.core.agent import ReActAgent
from llama_index.llms.ollama import Ollama # Or your preferred LLM interface
# from llama_index.llms.openai import OpenAI # Example if using OpenAI

# Configuration for your RAG system (similar to app.py)
DOCS_DIR = "sample_documents/"
CHROMA_PERSIST_DIR = "./chroma_db_agent" # Use a separate DB or ensure it's the same as app
CHROMA_COLLECTION_NAME = "ipc_agent_collection"

# ipc_agent_cli.py (continued)

def initialize_ipc_rag_engine():
    """Initializes or loads the RAG query engine for IPC documents."""
    print("Initializing IPC RAG Engine...")
    # Check if index needs to be built or can be loaded
    # (Simplified: always rebuilds for this CLI tool, or add load logic like in app.py)
    if not os.path.exists(CHROMA_PERSIST_DIR) or not os.listdir(CHROMA_PERSIST_DIR):
        print(f"Building index for IPC documents in {CHROMA_PERSIST_DIR}...")
        docs = load_documents(folder_path=DOCS_DIR)
        if not docs:
            print(f"ERROR: No documents found in {DOCS_DIR}. Please add IPC PDF files.")
            return None
        nodes = chunk_documents(docs)
        vector_store = get_vector_store(collection_name=CHROMA_COLLECTION_NAME, persist_dir=CHROMA_PERSIST_DIR)
        create_index_from_nodes(nodes, vector_store) # This populates the store
        print("IPC Index built successfully!")
    
    query_engine = get_ipc_query_engine(collection_name=CHROMA_COLLECTION_NAME, persist_dir=CHROMA_PERSIST_DIR)
    if query_engine:
        print("IPC RAG Engine loaded successfully.")
    else:
        print("ERROR: IPC RAG Engine could not be initialized.")
    return query_engine

# Get the RAG query engine instance
ipc_rag_engine = initialize_ipc_rag_engine()

if ipc_rag_engine is None:
    print("Exiting due to RAG engine initialization failure.")
    exit()

# ipc_agent_cli.py (continued)

def query_ipc_documents(user_query: str) -> str:
    """
    Use this tool to find information and answer questions specifically from the
    provided Indian Penal Code (IPC) documents. This tool is best for definitions,
    explanations of sections, punishments, and understanding the structure of the IPC.
    Input should be a specific question about the IPC.
    """
    print(f"\n>> DocumentQueryTool called with query: '{user_query}'")
    if ipc_rag_engine:
        response = ipc_rag_engine.query(user_query)
        print(f"<< DocumentQueryTool response: '{str(response)}'")
        return str(response.response) # Return only the text response
    return "IPC RAG engine not available."

# Create the FunctionTool
ipc_document_query_tool = FunctionTool.from_defaults(
    fn=query_ipc_documents,
    name="IPC_Document_Query_Tool",
    description=(
        "Provides answers to questions based on the indexed Indian Penal Code (IPC) documents. "
        "Use this for specific queries about IPC sections, definitions, offences, punishments, "
        "or structure of the code."
    )
)

# ipc_agent_cli.py (continued)

# Initialize the LLM for the Agent's reasoning
# Option 1: Ollama (ensure Ollama server is running with the model)
agent_llm = Ollama(model="mistral:7b-instruct", request_timeout=120.0) # Or llama3:8b-instruct
# For better agent performance, a more powerful model might be needed.
# If you have llama3 70b running in ollama, try that, but it will be slower.

# Option 2: OpenAI (if you have an API key set as an environment variable OPENAI_API_KEY)
# from llama_index.llms.openai import OpenAI
# agent_llm = OpenAI(model="gpt-3.5-turbo")

# List of tools the agent can use
tools = [ipc_document_query_tool]
# You could add more tools here later, e.g., a general web search tool.

# Create the ReActAgent
print("\nInitializing ReAct Agent...")
ipc_agent = ReActAgent.from_tools(
    tools=tools,
    llm=agent_llm,
    verbose=True # Set to True to see the agent's thought process, very useful for debugging!
)
print("ReAct Agent initialized.")

# ipc_agent_cli.py (continued)

def main_interaction_loop():
    print("\n--- IPC Agent ---")
    print("Ask complex questions or give tasks related to the Indian Penal Code.")
    print("Type 'exit' or 'quit' to end.")

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Exiting IPC Agent. Goodbye!")
            break

        if not user_input.strip():
            continue
        
        print("Agent thinking...")
        try:
            agent_response = ipc_agent.chat(user_input) # For conversational interaction
            # Or for single tasks: agent_response = ipc_agent.query(user_input)
            print(f"\nAgent: {agent_response}")
        except Exception as e:
            print(f"An error occurred: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main_interaction_loop()