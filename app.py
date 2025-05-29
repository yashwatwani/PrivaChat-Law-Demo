# In app.py (at the very top)
import sys
import os

# Add the project root to the Python path to ensure 'src' can be found
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Original imports
import streamlit as st
from src.rag_query_engine import get_query_engine # Assuming your module structure
from src.document_processor import load_documents, chunk_documents
from src.vector_store_manager import get_vector_store, create_index_from_nodes

# --- Configuration ---
DOCS_DIR = "sample_documents/"
CHROMA_PERSIST_DIR = "./chroma_db_streamlit" # Use a separate DB for the app if needed
CHROMA_COLLECTION_NAME = "privachat_streamlit_demo"

# --- Helper Functions ---
@st.cache_resource # Cache the query engine for performance
def load_query_engine():
    # Check if index needs to be built or can be loaded
    if not os.path.exists(CHROMA_PERSIST_DIR) or not os.listdir(CHROMA_PERSIST_DIR):
        st.info("Building index for the first time. This may take a moment...")
        docs = load_documents(folder_path=DOCS_DIR)
        if not docs:
            st.error(f"No documents found in {DOCS_DIR}. Please add some.")
            return None
        nodes = chunk_documents(docs)
        vector_store = get_vector_store(collection_name=CHROMA_COLLECTION_NAME, persist_dir=CHROMA_PERSIST_DIR)
        create_index_from_nodes(nodes, vector_store) # This populates the store
        st.success("Index built successfully!")
    
    query_engine = get_query_engine(collection_name=CHROMA_COLLECTION_NAME, persist_dir=CHROMA_PERSIST_DIR)
    return query_engine

# --- Streamlit App ---
st.title("📄 PrivaChat Law Demo")
st.caption("A local RAG system for querying your legal documents.")

# Initialize query engine
query_engine = load_query_engine()

if query_engine:
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("Ask something about your documents..."):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.spinner("Thinking..."):
            response = query_engine.query(prompt)
            response_text = response.response

            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                st.markdown(response_text)
                
                # Display source nodes
                if response.source_nodes:
                    st.subheader("Sources:")
                    for i, node in enumerate(response.source_nodes):
                        with st.expander(f"Source {i+1}: {node.metadata.get('file_name', 'N/A')} (Score: {node.score:.2f})"):
                            st.markdown(node.get_content())
            
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response_text}) # Store only text
else:
    st.warning("Query engine could not be initialized. Please check document loading and indexing.")


# Sidebar for potential re-indexing (optional advanced feature)
# with st.sidebar:
#     st.header("Admin")
#     if st.button("Re-build Index"):
#         if os.path.exists(CHROMA_PERSIST_DIR):
#             import shutil
#             shutil.rmtree(CHROMA_PERSIST_DIR) # Clear old DB
#             st.info("Cleared old index. Re-building...")
#         else:
#             st.info("Re-building index...")
        
#         # Re-run the caching function by clearing its cache and rerunning the app
#         st.cache_resource.clear()
#         st.rerun()