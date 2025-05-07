import streamlit as st
import asyncio
import chromadb
import os
import openai
import qdrant_client
from llama_index.core import Settings, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.llms.openai import OpenAI

# === CONFIGURATION ===
COLLECTION_NAME = "scoracle_index"
TOP_K = 8
Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")


# === Streamlit UI Setup ===
st.set_page_config(page_title="scOracle", page_icon="🔮", layout="wide")
st.title("🔮 scOracle: Your Single-Cell Analysis Assistant")

st.caption("A Retrieval-Augmented Generation (RAG) chatbot assistant that helps users explore, understand, and troubleshoot single-cell data analysis by leveraging real package documentation, tutorials, and source code.")

# === Initialize session state for chat history ===
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = [
        {"role": "scOracle", "content": "🔬 Welcome to scOracle — Ask about single-cell analysis!"}
    ]
    
# === Load default API key from secrets if available ===
DEFAULT_API_KEY = st.secrets.get("OPENAI_API_KEY", "")

# === Sidebar for OpenAI API Key and LLM parameters ===
with st.sidebar:
    openai_api_key = st.text_input(
        "OpenAI API Key", key="openai_api_key", type="password", value=DEFAULT_API_KEY,
    )
    "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"
    
    openai_model = st.selectbox(
        "Select OpenAI LLM",
        options=["gpt-4o-mini", "gpt-4o", 'gpt-4.1-nano'],
        index=0,
    )
    
    temperature = st.slider(
        "Temperature", min_value=0.0, max_value=1.0, value=0.1, step=0.1,
    )
    
    if "last_model" in st.session_state and (
        openai_model != st.session_state.last_model or temperature != st.session_state.last_temperature
    ):
        st.warning("⚠️ Changing model or temperature will reset scOracle's chat history.")

    st.session_state.last_model = openai_model
    st.session_state.last_temperature = temperature
    
    st.markdown(
        """
        This chatbot is powered by [OpenAI](https://openai.com/), [LlamaIndex](https://www.llamaindex.ai/), and [Qdrant](https://qdrant.tech/).
        It uses a vector database to retrieve relevant information from the documentation and tutorials.
        
        ### Current Knowledge Base
        - Scanpy
        - Seurat
        - Awesome Single Cell
        
        ### Disclaimer
        This is a research prototype. Please use it responsibly and verify any information before acting on it.
        """
    )

# === Require API Key ===
if not openai_api_key:
    st.warning("Please enter your OpenAI API key in the sidebar to begin.")
    st.stop()

openai.api_key = openai_api_key

# === Retrieve the Chroma vector store, set up LLM and chat engine ===
@st.cache_resource(show_spinner=False)
def load_index_and_engine(model, temperature, api_key):
    """Load the QDrant vector store"""
    
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    if not QDRANT_API_KEY:
        raise EnvironmentError("❌ QDRANT_API_KEY not set in environment variables.")
    
    client = qdrant_client.QdrantClient(
        "https://d4a4a1c8-43a2-4017-ac58-27ee2d6630d7.us-east-1-0.aws.cloud.qdrant.io",
        api_key=QDRANT_API_KEY
    )
    vector_store = QdrantVectorStore(client=client, collection_name=COLLECTION_NAME)
    index = VectorStoreIndex.from_vector_store(vector_store)
    
    llm = OpenAI(
            model=model, 
            temperature=temperature,
            system_prompt="""
            You are scOracle, a scientific assistant for single-cell RNA-seq analysis.
            You can answer technical questions using your knowledge of documentation, code, and tutorials from bioinformatics tools such as Scanpy.
            """
        )
    print(f"Using OpenAI model: {model} with temperature: {temperature}")
    chat_engine = index.as_chat_engine(similarity_top_k=TOP_K, chat_mode="best", llm=llm, streaming=True)    
    return chat_engine

chat_engine = load_index_and_engine(openai_model, temperature, openai_api_key)

# === Chatbot UI ===

# User input
if user_input:= st.chat_input("❓ Ask a question:"):
    # save user input to chat history
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    
# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])
    
# If last message is not from assistant, generate a new response
if st.session_state.chat_history[-1]["role"] != "scOracle":
    with st.chat_message("scOracle"):
        with st.spinner("Thinking..."):
            # Generate response
            stream_response = chat_engine.stream_chat(user_input)
            full_response = ""
            stream_placeholder = st.empty()
            
            for chunk in stream_response.response_gen:
                full_response += chunk
                stream_placeholder.markdown(full_response + "▌")

            stream_placeholder.markdown(full_response)

            # Save complete response to chat history
            st.session_state.chat_history.append({
                "role": "scOracle",
                "content": full_response
            })