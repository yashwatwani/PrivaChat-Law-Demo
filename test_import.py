# test_import.py
try:
    from llama_index.vector_stores.chroma import ChromaVectorStore
    print("SUCCESS: Imported ChromaVectorStore from llama_index.vector_stores.chroma")
except ImportError as e:
    print(f"ERROR importing from llama_index.vector_stores.chroma: {e}")

print("-" * 20)

try:
    import llama_index_vector_stores_chroma
    print("SUCCESS: Imported llama_index_vector_stores_chroma package directly")
except ImportError as e:
    print(f"ERROR importing llama_index_vector_stores_chroma package directly: {e}")

print("-" * 20)
import sys
print("Python sys.path:")
for p in sys.path:
    print(p)
print("-" * 20)

try:
    import llama_index.core
    print(f"LlamaIndex Core version: {llama_index.core.__version__}")
except (ImportError, AttributeError):
    try:
        import llama_index
        print(f"LlamaIndex (main) version: {llama_index.__version__}")
    except (ImportError, AttributeError) as e:
        print(f"Could not get LlamaIndex version: {e}")