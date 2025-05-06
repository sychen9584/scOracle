from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    Settings
)
from llama_index.core.node_parser import SentenceSplitter, CodeSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
import argparse

import doc_loaders

# === CONFIGURATION ===
# Set the path to the directory containing the documents
#DATA_DIR_DOCS = "../data/scanpy_docs"
# Set the path to the directory containing the code
#DATA_DIR_CODE = "../data/scanpy_src/core"
#DATA_DIR_TUTORIALs = "../data/scanpy_src/tutorials"
VECTOR_DB_PATH = "../chroma_db"
COLLECTION_NAME = "scoracle_index"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def parse_args():
    parse = argparse.ArgumentParser(description="Build index for scOracle")
    parse.add_argument("--html", type=str, default=None, help="Path to the directory containing the html documents")
    parse.add_argument("--python-code", type=str, default=None, help="Path to the directory containing python code")
    parse.add_argument("--r-code", type=str, default=None, help="Path to the directory containing R code")
    parse.add_argument("--ipynb", type=str, default=None, help="Path to the directory containing the ipynb notebooks")
    parse.add_argument("--vector_db_path", type=str, default=VECTOR_DB_PATH, help="Path to the Chroma database")
    parse.add_argument("--collection_name", type=str, default=COLLECTION_NAME, help="Name of the Chroma collection")
    parse.add_argument("--embed_model", type=str, default=EMBED_MODEL, help="Embedding model name")
    
    args = parse.parse_args()
    
    if not (args.html or args.python_code or args.r_code or args.ipynb):
        parse.error("At least one of --html, --python-code, --r-code, or --ipynb must be specified.")
        
    return args

def index_with_splitter(docs, storage_context, splitter):
    """
    Index documents using a specified splitter.
    """
    # Set the node parser to the specified splitter
    Settings.node_parser = splitter
    
    # Create the index
    VectorStoreIndex.from_documents(
        documents=docs,
        storage_context=storage_context,
        show_progress=True
    )
    

def index_documents(args):
    all_docs = []
    
    if args.html:
        html_docs = doc_loaders.load_html_documents(args.html)
        all_docs.extend(html_docs)
        unique_sources = set(doc.metadata.get("source") for doc in html_docs)
        print(f"Loaded {len(html_docs)} html chunks from {len(unique_sources)} files")
        
    if args.ipynb:
        ipynb_docs = doc_loaders.load_ipynb_docs(args.ipynb)
        all_docs.extend(ipynb_docs)
        unique_sources = set(doc.metadata.get("source") for doc in ipynb_docs)
        print(f"Loaded {len(ipynb_docs)} notebook chunks from {len(unique_sources)} files")
        
    if args.python_code:
        python_docs = doc_loaders.load_python_code_docs(args.python_code)
        print(f"Loaded {len(python_docs)} Python code documents")
    else:
        python_docs = []
        
    if args.r_code:
        r_docs = doc_loaders.load_r_code_docs(args.r_code)
        print(f"Loaded {len(r_docs)} R code documents")
    else:
        r_docs = []
        
    # set embedding model
    embed_model = HuggingFaceEmbedding(model_name=args.embed_model)
    Settings.embed_model = embed_model
    
    # set up Chroma vector store
    chroma_client = chromadb.PersistentClient(path=args.vector_db_path)
    chroma_collection = chroma_client.get_or_create_collection(args.collection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # index documents with SentenceSplitter
    if all_docs:
        index_with_splitter(all_docs, storage_context, SentenceSplitter(chunk_size=500, chunk_overlap=100))
        
    if python_docs:
        index_with_splitter(python_docs, storage_context, CodeSplitter(chunk_lines=40, chunk_lines_overlap=15, language="python"))
        
    if r_docs:
        index_with_splitter(r_docs, storage_context, CodeSplitter(chunk_lines=40, chunk_lines_overlap=15, language="r"))
        
    # persist index
    storage_context.persist()
    print(f"Index built and stored at {args.vector_db_path}")
        
if __name__ == "__main__":
    args = parse_args()
    index_documents(args)