from llama_index.core import Settings
# from llm_interface_ollama import get_llm # Assuming this is your primary
from llama_index.llms.ollama import Ollama # Direct use for simplicity here
from .vector_store_manager import load_index_from_store, get_vector_store
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Configure LLM and Embedding model globally or pass to query engine
# Settings.llm = get_llm()
# Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
# OR direct instantiation:
llm = Ollama(model="mistral:7b-instruct", request_timeout=120.0)
embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")


def get_query_engine(collection_name="privachat_law_demo", persist_dir="./chroma_db"):
    vector_store = get_vector_store(collection_name, persist_dir)
    index = load_index_from_store(vector_store)
    query_engine = index.as_query_engine(
        llm=llm, # Pass LLM explicitly
        # embed_model is usually handled by the index, but good to be aware
        similarity_top_k=3 # Retrieve top 3 similar chunks
    )
    return query_engine

if __name__ == "__main__":
    query_engine = get_query_engine()
    
    # Make sure you've run vector_store_manager.py once to populate the DB
    test_query = "What are the obligations mentioned in the sample document?"
    print(f"Querying: {test_query}")
    response = query_engine.query(test_query)
    
    print("\nResponse:")
    print(response.response) # .response to get the text
    
    print("\nSource Nodes:")
    for node in response.source_nodes:
        print(f"  Score: {node.score:.4f}, File: {node.metadata.get('file_name', 'N/A')}")
        # print(f"  Content: {node.get_content()[:100]}...") # Optional: show source content