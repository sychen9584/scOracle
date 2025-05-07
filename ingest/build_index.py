from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    Settings
)
from llama_index.core.node_parser import SentenceSplitter, CodeSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
import os
import qdrant_client
import argparse

import doc_loaders

# === CONFIGURATION ===
COLLECTION_NAME = "scoracle_index"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def parse_args():
    parse = argparse.ArgumentParser(description="Build index for scOracle")
    parse.add_argument("--html", type=str, default=None, help="Path to the directory containing the html documents")
    parse.add_argument("--python-code", type=str, default=None, help="Path to the directory containing python code")
    parse.add_argument("--r-code", type=str, default=None, help="Path to the directory containing R code")
    parse.add_argument("--cpp-code", type=str, default=None, help="Path to the directory containing C++ code")
    parse.add_argument("--notebook", type=str, default=None, help="Path to the directory containing the markdown notebooks")
    parse.add_argument("--collection_name", type=str, default=COLLECTION_NAME, help="Name of the Chroma collection")
    parse.add_argument("--embed_model", type=str, default=EMBED_MODEL, help="Embedding model name")
    
    args = parse.parse_args()
    
    if not (args.html or args.python_code or args.r_code or args.notebook or args.cpp_code):
        parse.error("At least one of --html, --python-code, --r-code, --cpp-code, or --notebook must be specified.")
        
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
        
    if args.notebook:
        notebook_docs = doc_loaders.load_notebook_docs(args.notebook)
        all_docs.extend(notebook_docs)
        unique_sources = set(doc.metadata.get("source") for doc in notebook_docs)
        print(f"Loaded {len(notebook_docs)} notebook chunks from {len(unique_sources)} files")
    
    python_docs = []
    if args.python_code:
        python_docs = doc_loaders.load_python_code_docs(args.python_code)
        print(f"Loaded {len(python_docs)} Python code documents")
      
    r_docs = []
    if args.r_code:
        r_docs = doc_loaders.load_r_code_docs(args.r_code)
        print(f"Loaded {len(r_docs)} R code documents")
        
    cpp_docs = []
    if args.cpp_code:
        cpp_docs = doc_loaders.load_cpp_code_docs(args.cpp_code)
        print(f"Loaded {len(cpp_docs)} C++ code documents")
        
    # set embedding model
    embed_model = HuggingFaceEmbedding(model_name=args.embed_model)
    Settings.embed_model = embed_model
    
    # set up qdrant vector store
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    if not QDRANT_API_KEY:
        raise EnvironmentError("❌ QDRANT_API_KEY not set in environment variables.")
    
    client = qdrant_client.QdrantClient(
        "https://d4a4a1c8-43a2-4017-ac58-27ee2d6630d7.us-east-1-0.aws.cloud.qdrant.io",
        api_key=QDRANT_API_KEY
    )
    vector_store = QdrantVectorStore(client=client, collection_name=args.collection_name)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # index documents with SentenceSplitter
    if all_docs:
        index_with_splitter(all_docs, storage_context, SentenceSplitter(chunk_size=500, chunk_overlap=100))
        
    if python_docs:
        index_with_splitter(python_docs, storage_context, CodeSplitter(chunk_lines=40, chunk_lines_overlap=15, language="python"))
        
    if r_docs:
        index_with_splitter(r_docs, storage_context, SentenceSplitter(chunk_size=500, chunk_overlap=100))
        
    if cpp_docs:
        index_with_splitter(cpp_docs, storage_context, CodeSplitter(chunk_lines=40, chunk_lines_overlap=15, language="cpp"))
        
    count = client.count(collection_name=args.collection_name).count
    print(f"Total vectors stored in Qdrant: {count}")
    print(f"Index built and pushed to Qdrant collection: {args.collection_name}")
        
if __name__ == "__main__":
    args = parse_args()
    index_documents(args)