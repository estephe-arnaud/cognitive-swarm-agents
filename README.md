# MAKERS: Multi Agent Knowledge Exploration & Retrieval System

<!-- 
Optional Badges:
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
-->

## 🚀 Overview

"**MAKERS**" is a multi-agent system designed to collaboratively search, analyze, and synthesize information from complex document corpora (e.g., scientific research papers). It leverages Large Language Models (LLMs) through frameworks like LangGraph for orchestration and LlamaIndex for advanced Retrieval Augmented Generation (RAG), with MongoDB serving as the vector database and for persisting agent states.

The primary goal of this project is to build an intelligent engine capable of:
*   Ingesting and processing large sets of documents (e.g., ArXiv papers).
*   Performing semantic searches and retrieving relevant information chunks.
*   Employing a team of specialized AI agents (Planner, ArXiv Searcher, Document Analyzer, Synthesizer) to:
    *   **Plan** research tasks based on user queries.
    *   **Search** external sources (ArXiv) and internal knowledge bases.
    *   **Analyze** search results, with the capability to perform deep dives into full PDF documents via their URLs.
    *   **Synthesize** all findings into a coherent, final report.
*   Allowing for robust evaluation of its RAG and synthesis capabilities.
*   Tracking experiments and metrics using Weights & Biases.

## ✨ Features

* **Modular Data Pipeline**: Scripts for downloading ArXiv papers, parsing PDFs, cleaning text, chunking, and generating embeddings.
* **MongoDB Integration**:
    * Vector store using Atlas Vector Search for semantic retrieval.
    * Storage for document metadata and text chunks.
    * LangGraph checkpointer (`MongoDBSaver`) for persistent and resumable workflows.
* **Advanced RAG with LlamaIndex**: A `RetrievalEngine` for efficient information retrieval, supporting vector search and metadata filtering.
* **Multi-Agent System with LangGraph**:
    * Orchestration of specialized agents with clearly defined roles.
    * Dynamic, state-based routing between agents.
    * Persistent state management for complex, long-running tasks.
* **Robust Tooling for Agents**: Includes tools for live ArXiv searching, internal knowledge base retrieval, and a powerful **PDF deep-dive analysis tool** that works directly from a URL.
* **Comprehensive Evaluation Suite**: Evaluators for RAG metrics (Hit Rate, MRR) and synthesis quality (LLM-as-a-Judge).
* **CLI Interface**: Scripts (`run_ingestion.py`, `run_makers.py`, `run_evaluation.py`) to manage the system.
* **Jupyter Notebooks**: For setup, component demonstration, and end-to-end testing.
* **Reproducible Environment**: Defined via `environment.yml` (Conda) and `requirements.txt`.
* **Containerization & API**: A `Dockerfile` and a basic FastAPI layer are provided.

### 🛠️ Tech Stack & Architecture

* **Core Language**: Python 3.11+
* **LLM Orchestration**: LangGraph
* **Specialized Agent Teams**: CrewAI (for the deep document analysis sub-task)
* **RAG & Data Indexing**: LlamaIndex
* **LLM Interactions**: LangChain
* **Centralized LLM Management**: The `src/llm_services/llm_factory.py` module centralizes the instantiation of LLMs (OpenAI, Hugging Face, Ollama) for all components.
* **Vector Database & Checkpointing**: MongoDB Atlas
* **Experiment Tracking**: Weights & Biases
* **API**: FastAPI, Uvicorn

### High-Level Architecture

1.  **Data Ingestion**: ArXiv papers are downloaded, processed, chunked, embedded, and stored in MongoDB. Vector search indexes are created automatically.
2.  **User Query**: A user submits a research query via the CLI or API.
3.  **LangGraph Workflow (`MAKERS`)**:
    *   A `ResearchPlannerAgent` deconstructs the query into a structured plan.
    *   Based on the plan, an `ArxivSearchAgent` finds relevant new papers using the `arxiv_search_tool`.
    *   The `DocumentAnalysisAgent` receives the list of papers. It first reviews the summaries. For the most promising papers, it is **explicitly instructed to use the `document_deep_dive_analysis_tool`**, providing the PDF URL to get a full analysis of the paper's content.
    *   A `SynthesisAgent` consolidates all the collected information (summaries and deep-dive analyses) into a final, polished report.
    *   The entire workflow state is checkpointed in MongoDB at each step, ensuring persistence and resumability.
4.  **Evaluation**: Separate scripts allow for the evaluation of the RAG and Synthesis components, with results logged to W&B.

## 📁 Directory Structure
```
makers/
├── config/              # Configuration files
├── data/                # Local data (corpus, evaluation datasets)
├── notebooks/           # Jupyter notebooks for demonstration and testing
├── scripts/             # CLI scripts (run_ingestion.py, run_makers.py)
├── src/                 # Source code
│   ├── agents/          # Agent architectures and tool definitions
│   │   └── crewai_teams/ # CrewAI sub-task definitions
│   ├── api/             # FastAPI application
│   ├── data_processing/ # Data ingestion and preprocessing modules
│   ├── evaluation/      # Evaluation and W&B logging modules
│   ├── graph/           # LangGraph workflow definition
│   ├── llm_services/    # Centralized LLM factory
│   ├── rag/             # RAG engine (LlamaIndex)
│   └── vector_store/    # MongoDB management
├── .env.example         # Example template for .env
├── environment.yml      # Conda environment definition
├── requirements.txt     # Pip requirements file
├── Dockerfile           # Docker image definition
└── README.md            # This file
```

## ⚙️ Setup Instructions

1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/estephe-arnaud/makers
    cd makers
    ```

2.  **Create and Activate Conda Environment**:
    ```bash
    conda env create -f environment.yml
    conda activate makers
    ```

3.  **Set Up Environment Variables (`.env` file)**:
    *   Create a `.env` file by copying `.env.example`.
    *   **Ollama is the default provider.** Ensure your Ollama server is running.
    *   Configure `MONGODB_URI` with your MongoDB connection string.
    *   To use other providers like OpenAI, set `DEFAULT_LLM_MODEL_PROVIDER="openai"` and provide the `OPENAI_API_KEY`.

4.  **(Optional) W&B Login**:
    ```bash
    wandb login
    ```

## 🚀 Running the Project

All commands should be run from the root directory (`makers/`) with the `makers` Conda environment activated.

### 1. Data Ingestion

Populate your MongoDB database with ArXiv papers:
```bash
python -m scripts.run_ingestion --query "Your Research Topic" --max_results 10
```
*   This script downloads, processes, and embeds papers into your database.

### 2. Running the MAKERS Workflow

Submit a query to the multi-agent system:
```bash
python -m scripts.run_makers --query "What are the latest advancements in using large language models for robot task planning?"
```
*   A `thread_id` will be generated for the session.
*   Agent and tool outputs will stream to the console, followed by the final report.
*   Use `--log_level DEBUG` for more detailed output.

### 3. Running Evaluations

Evaluate the system's performance:
```bash
python -m scripts.run_evaluation --eval_type all
```
*   This script uses the example evaluation datasets in `data/evaluation/`.

## 📓 Notebooks for Demonstration

The `notebooks/` directory contains Jupyter Notebooks (`.ipynb`) for a step-by-step exploration of the system's components.

*   `00_setup_environment.ipynb`: Verify your environment configuration.
*   `01_data_ingestion_and_embedding.ipynb`: Demonstrate the data ingestion pipeline.
*   `02_rag_strategies_exploration.ipynb`: Explore RAG strategies.
*   `03_agent_development_and_tooling.ipynb`: Test individual agents and tools.
*   `04_langgraph_workflow_design.ipynb`: Run and observe the main LangGraph workflow interactively.
*   `05_crewai_team_integration.ipynb`: Demonstrate the CrewAI deep-dive integration.
*   `06_end_to_end_pipeline_test.ipynb`: Run an in-depth test of the full pipeline.
*   `07_evaluation_and_logging.ipynb`: Detail the use of evaluation modules.

## 🔮 Future Work

*   Implement more sophisticated, LLM-based routing logic.
*   Integrate more advanced RAG strategies (e.g., Parent Document Retriever).
*   Add a `QualityCheckAgent` to review the final synthesis.
*   Expand the toolset for agents (e.g., web search).
*   Enhance the FastAPI with more features.
*   Implement a comprehensive test suite.
*   Add LLM-based keyword extraction to optimize the data ingestion search.

## 📄 License
```
MIT License

Copyright (c) 2025 [Estèphe ARNAUD / MAKERS: Multi Agent Knowledge Exploration & Retrieval System]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.