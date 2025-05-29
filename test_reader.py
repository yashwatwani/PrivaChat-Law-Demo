# test_reader.py (place in your project root PrivaChat-Law-Demo/)
import os
from llama_index.core import SimpleDirectoryReader

# --- Configuration ---
# This path is relative to where test_reader.py is run (the project root)
DOCS_DIR = "sample_documents/"
ABS_DOCS_DIR = os.path.abspath(DOCS_DIR)

print(f"Attempting to read from directory: {DOCS_DIR}")
print(f"Absolute path being checked: {ABS_DOCS_DIR}")

if not os.path.exists(ABS_DOCS_DIR):
    print(f"ERROR: The directory '{ABS_DOCS_DIR}' does not exist.")
elif not os.path.isdir(ABS_DOCS_DIR):
    print(f"ERROR: The path '{ABS_DOCS_DIR}' is not a directory.")
else:
    print(f"Directory '{ABS_DOCS_DIR}' exists and is a directory.")
    print("Listing files in the directory (from Python's perspective):")
    try:
        files_in_dir = os.listdir(ABS_DOCS_DIR)
        if not files_in_dir:
            print("  Directory is EMPTY according to os.listdir().")
        else:
            for f_name in files_in_dir:
                print(f"  - {f_name}")
    except Exception as e:
        print(f"  Error listing directory contents: {e}")

print("-" * 30)

try:
    print("Initializing SimpleDirectoryReader...")
    # Explicitly tell it to look for PDFs if necessary, though it should by default
    # reader = SimpleDirectoryReader(DOCS_DIR, required_exts=[".pdf"]) 
    reader = SimpleDirectoryReader(DOCS_DIR)
    print("SimpleDirectoryReader initialized.")
    print("Loading data...")
    documents = reader.load_data()
    if documents:
        print(f"SUCCESS: Loaded {len(documents)} document(s).")
        for doc in documents:
            print(f"  - File: {doc.metadata.get('file_name', 'N/A')}, Text length: {len(doc.text)}")
    else:
        print("WARNING: load_data() returned an empty list, but no error was raised by SimpleDirectoryReader init.")
        print("This might happen if files are found but cannot be parsed (e.g. corrupted PDFs, password protected).")

except ValueError as ve:
    print(f"VALUE ERROR from SimpleDirectoryReader: {ve}")
except Exception as e:
    print(f"GENERAL EXCEPTION from SimpleDirectoryReader: {e}")
    import traceback
    traceback.print_exc()