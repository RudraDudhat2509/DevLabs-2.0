# TechStore Agentic AI

An autonomous customer support agent built with LangGraph and a local LLM (`qwen3:8b`). It handles customer refunds, inventory checks, and policy enforcement without human intervention.

## Core Architecture
*   **LangGraph Routing:** Uses a cyclical state graph to dynamically route user queries to Python tools.
*   **Stateful Memory:** Maintains multi-turn conversational context using `MemorySaver`.
*   **RAG Engine:** Enforces business rules (like the 30-day return window) by performing semantic searches on `policy.txt` via FAISS.
*   **Temporal Logic:** Injects the live system clock into the prompt so the LLM can calculate date differences for refunds.
*   **Prompt Security:** Uses a hidden Master Override Protocol (`ADMIN_CODE`) and strict negative constraints to defend against prompt injection.

## Setup
1. Install [Ollama](https://ollama.com/) and run: `ollama run qwen3:8b`
2. Install requirements: `pip install langchain-core langgraph langchain-ollama langchain-community langchain-text-splitters langchain-huggingface faiss-cpu pandas pydantic cryptography`
3. Start the agent: `python main_agent.py`
