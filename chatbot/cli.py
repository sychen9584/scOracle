from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.openai import OpenAI
import chromadb
import openai
import os
import asyncio
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings

# === CONFIGURATION ===
CHROMA_PATH = "../chroma_db"
COLLECTION_NAME = "scoracle_index"
TOP_K = 8
Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

# === OPENAI API KEY ===
openai.api_key = os.getenv("OPENAI_API_KEY")
if openai.api_key is None:
    raise ValueError("Please set the OPENAI_API_KEY environment variable.")

# === Retrieve the Chroma vector store ===
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_or_create_collection(COLLECTION_NAME)
vector_store = ChromaVectorStore(chroma_collection=collection)
index = VectorStoreIndex.from_vector_store(vector_store)

# === Set up LLM and chat engine ===
llm = OpenAI(
    model="gpt-4o-mini", 
    temperature=0.0,
    system_prompt="""
    You are scOracle, a scientific assistant for single-cell RNA-seq analysis.
    You can answer technical questions using your knowledge of documentation, code, and tutorials from bioinformatics tools such as Scanpy.
    """
)
chat_engine = index.as_chat_engine(similarity_top_k=TOP_K, chat_mode="best", llm=llm)
print("Docs in index:", len(index.docstore.docs))

# === Chatbot loop ===
async def main():

    print("🔬 Welcome to scOracle — Ask about single-cell analysis!\n")
    while True:
        user_input = input("❓ Ask a question (or 'exit'): ")
        if user_input.lower() == "exit":
            break
        response = await chat_engine.achat(user_input)
        for source_node in response.source_nodes:
            print(f"\n📄 Retrieved Chunk:\n{source_node.node.get_content()[:300]}...")  # truncate for readability
        print("\n📘 Answer:")
        print(response)
        print("\n" + "-" * 60 + "\n")
        
if __name__ == "__main__":
    asyncio.run(main())