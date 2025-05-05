from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    Settings
)
from llama_index.core.node_parser import SentenceSplitter, CodeSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
import os
import argparse

from html_doc_loader import load_html_documents

# === CONFIGURATION ===
# Set the path to the directory containing the documents
DATA_DIR_DOCS = "../data/scanpy_docs"
# Set the path to the directory containing the code
DATA_DIR_CODE = "../data/scanpy_src/core"
DATA_DIR_TUTORIALs = "../data/scanpy_src/tutorials"
CHROMA_PATH = "../chroma_db"
COLLECTION_NAME = "scoracle_index"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def parse_args():
    parse = argparse.ArgumentParser(description="Build index for scOracle")
    parse.add_argument("--docs", type=str, default=DATA_DIR_DOCS, help="Path to the directory containing the documents")
    parse.add_argument("--code", type=str, default=DATA_DIR_CODE, help="Path to the directory containing the code")
    parse.add_argument("--tutorials", type=str, default=DATA_DIR_TUTORIALs, help="Path to the directory containing the tutorials")
    parse.add_argument("--vector_db_path", type=str, default=CHROMA_PATH, help="Path to the Chroma database")
    parse.add_argument("--collection_name", type=str, default=COLLECTION_NAME, help="Name of the Chroma collection")
    parse.add_argument("--embed_model", type=str, default=EMBED_MODEL, help="Embedding model name")
    return parse.parse_args()

# === Load documentation files (.html) ===
html_docs = load_html_documents(DATA_DIR_DOCS)

# === Load code files (.py) ===
code_loader = SimpleDirectoryReader(
    input_dir=DATA_DIR_CODE,
    recursive=True,
    required_exts=[".py"],
)
code_docs = code_loader.load_data()
for doc in code_docs:
    doc.metadata["type"] = "code"
    
## === Load tutorial files (.ipynb) ===
notebook_loader = SimpleDirectoryReader(
    input_dir=DATA_DIR_TUTORIALs,
    recursive=True,
    required_exts=[".ipynb"],
)
notebook_docs = notebook_loader.load_data()
for doc in notebook_docs:
    doc.metadata["type"] = "tutorial"
    
# === Combine all nodes ===
text_docs = html_docs + notebook_docs

# === Set up embedding model ===
embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL)

# === Set up Chroma vector store ===
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
chroma_collection = chroma_client.get_or_create_collection(COLLECTION_NAME)
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# === Set up index ===
Settings.embed_model = embed_model

# === Index docs + notebooks with SentenceSplitter ===
Settings.node_parser = SentenceSplitter(chunk_size=500, chunk_overlap=100)
index_docs = VectorStoreIndex.from_documents(
    documents=text_docs,
    storage_context=storage_context,
    show_progress=True
)

# === Index code with CodeSplitter ===
Settings.node_parser = CodeSplitter(chunk_lines=40, chunk_lines_overlap=15, language="python")
index_code = VectorStoreIndex.from_documents(
    documents=code_docs,
    storage_context=storage_context,
    show_progress=True
)
# === Persist index ===
storage_context.persist()
print(f"Index built and stored at {CHROMA_PATH}")