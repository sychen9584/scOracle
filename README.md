# 🔮 scOracle: An AI Assistant for Single-Cell Analysis

**scOracle** is a Retrieval-Augmented Generation (RAG) chatbot designed to answer questions about single-cell data analysis using LLM-powered search over curated documentation. It integrates tools like LlamaIndex, Qdrant, and Streamlit to deliver fast, context-aware assistance grounded in real-world biological tooling and workflows.

![screenshot](./images/scoracle_st2.png)

## Features
- Target audience: Scientists or students navigating single-cell workflows
- Ingests real docs, notebooks, and repo source code
- Responds to natural language questions (e.g., "How do I run Leiden clustering?")
- Offers code snippets, parameter guidance, and troubleshooting tips based on ingested knowledge
- Modular backend: ability to swap models and tune parameters

Try it here: [https://scoracle-syc.streamlit.app](https://scoracle-syc.streamlit.app)  
>  Requires your own OpenAI API key for full functionality.

## App structure
![screenshot](./images/scoracle_mermaid.png)

## Tools to include in the knowledge base
- [X] Umbrella analysis framework: Scanpy & Seurat ==> MVP for first iteration
  - [] Can be extended to whole [scVerse ecosystem](https://scverse.org/)
- [] Upstream processing: Cell Ranger, Alevin-Fry, and NF-Core
- [] scATAC-seq analysis: Signac & ArchR
- [] Gene regulatory network inference: SCENIC & scPRINT
- [] Spatial Transcriptomics: squidpy
- [X] Awesome Single Cell [repo](https://github.com/seandavi/awesome-single-cell)

## Tech Stack

|---
| Component | Choice | Function 
|:-:|:-:|:-:
| LLM API | GPT-4o mini | Generates natural language responses conditioned on retrieved context
| Retrieval | LlamaIndex | Manages document ingestion, chunking, indexing, and context retrieval for RAG
| Embedding | all-MiniLM-L6-v2 (SBERT) | Converts text into dense vector representations for semantic search
| Vector DB | Qdrant | Stores and retrieves embeddings efficiently to support fast similarity search
| UI | Streamlit | Provides a simple, interactive web interface for user input and chatbot output

## Future improvements for scOracle
1. Scaling up the knowledge base
2. Improve document chunking strategy
3. Decouple backend and frontend services
4. Adapter-based fine-tuning or embedding optimization

## Implementation details
- [Building a RAG-Powered AI Chatbot for Single-Cell Analysis - Part 1: Foundations](https://sychen9584.github.io/posts/2025/04/rag-part1/)
- [Building a RAG-Powered AI Chatbot for Single-Cell Analysis - Part 2: Data Ingestion & Indexing](https://sychen9584.github.io/posts/2025/05/rag-part2/)
- [Building a RAG-Powered AI Chatbot for Single-Cell Analysis - Part 3: CLI Query Engine](https://sychen9584.github.io/posts/2025/05/rag-part3/)
- [Building a RAG-Powered AI Chatbot for Single-Cell Analysis - Part 4: Streamlit UI](https://sychen9584.github.io/posts/2025/05/rag-part4/)
- [Building a RAG-Powered AI Chatbot for Single-Cell Analysis - Part 5: Scaling Up & Cloud hosting](https://sychen9584.github.io/posts/2025/05/rag-part5/)