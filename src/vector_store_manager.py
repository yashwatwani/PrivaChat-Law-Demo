import chromadb
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding # Or OllamaEmbedding
# from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import Settings

# Global settings for embedding model
# Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
# Or for Ollama embeddings if you have an embedding model running in Ollama:
# Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text") # Make sure nomic-embed-text is pulled via ollama

# Using HuggingFaceEmbedding as it's more common for direct use
embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
# embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")


def get_vector_store(collection_name="privachat_law_demo", persist_dir="./chroma_db"):
    db = chromadb.PersistentClient(path=persist_dir)
    chroma_collection = db.get_or_create_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    return vector_store

def create_index_from_nodes(nodes, vector_store):
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex(
        nodes,
        storage_context=storage_context,
        embed_model=embed_model # Pass explicitly if not set globally
    )
    return index

def load_index_from_store(vector_store):
    index = VectorStoreIndex.from_vector_store(
        vector_store,
        embed_model=embed_model # Pass explicitly
    )
    return index

if __name__ == "__main__":
    from document_processor import load_documents, chunk_documents

    # 1. Load and chunk documents
    docs = load_documents()
    nodes = chunk_documents(docs)

    # 2. Get vector store
    vector_store = get_vector_store()

    # 3. Create index (if first time or re-indexing)
    print("Creating index...")
    index = create_index_from_nodes(nodes, vector_store)
    print("Index created.")

    # (Optional) Test loading index
    # print("Loading index from store...")
    # index_loaded = load_index_from_store(vector_store)
    # print("Index loaded from store.")

    # Test a query (will be part of RAG chain later)
    query_engine = index.as_query_engine()
    response = query_engine.query("What are the main points of this document?")
    print("\nQuery Response:")
    print(response)