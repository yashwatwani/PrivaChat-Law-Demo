from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter

def load_documents(folder_path="sample_documents/"):
    reader = SimpleDirectoryReader(folder_path)
    documents = reader.load_data()
    return documents

def chunk_documents(documents, chunk_size=512, chunk_overlap=20):
    parser = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    nodes = parser.get_nodes_from_documents(documents)
    return nodes

if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} documents.")
    for doc in docs:
        print(f"--- Content of {doc.metadata.get('file_name', 'N/A')} ---")
        print(doc.text[:200] + "...") # Print first 200 chars
        print("-" * 20)

    nodes = chunk_documents(docs)
    print(f"\nChunked into {len(nodes)} nodes.")
    if nodes:
        print("First node content:")
        print(nodes[0].get_content()[:300] + "...")