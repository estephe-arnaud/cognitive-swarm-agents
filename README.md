# Cognitive Swarm: Multi-Agent Knowledge Discovery Engine

## 🚀 Overview

"Cognitive Swarm" is a multi-agent system designed and implemented to collaboratively search, analyze, and synthesize information from complex document corpora (e.g., scientific research papers, technical documentation). It leverages Large Language Models (LLMs) through frameworks like LangGraph (for orchestration) and LlamaIndex (for advanced Retrieval Augmented Generation - RAG), with MongoDB Atlas serving as the vector database and for persisting agent states.

The primary goal of this project is to build an intelligent engine capable of:
* Ingesting and processing large sets of documents (specifically ArXiv papers on "Reinforcement Learning for Robotics" in the current implementation).
* Performing semantic searches and retrieving relevant information chunks.
* Employing a team of specialized AI agents (Planner, ArXiv Searcher, Document Analyzer, Synthesizer, and a conceptual CrewAI team for deep document analysis) to:
    * Plan research tasks based on user queries.
    * Search external sources (ArXiv) and internal knowledge bases (MongoDB).
    * Analyze and extract key information from documents, potentially using specialized agent teams for deep dives.
    * Synthesize findings into coherent reports or answers to complex questions.
* Allowing for robust evaluation of its RAG and synthesis capabilities.
* Tracking experiments and metrics using Weights & Biases.

This project serves as a portfolio piece demonstrating expertise in Generative AI, LLM-powered agent systems, RAG pipelines, LangGraph, CrewAI integration concepts, and MLOps practices.

## ✨ Features

* **Modular Data Pipeline**: Python scripts for downloading ArXiv papers, parsing PDFs, cleaning text, chunking by tokens, and generating OpenAI embeddings.
* **MongoDB Atlas Integration**:
    * Vector store using Atlas Vector Search for semantic retrieval.
    * Storage for document metadata and processed text chunks.
    * LangGraph checkpointer backend (`MongoDBSaver`) for persistent and resumable agent workflows.
* **Advanced RAG with LlamaIndex**: `RetrievalEngine` utilizing LlamaIndex with `MongoDBAtlasVectorSearch` for efficient information retrieval, supporting vector search and metadata filtering.
* **Multi-Agent System with LangGraph**:
    * Orchestration of specialized agents with defined roles.
    * Dynamic routing capabilities initiated by a planner and rule-based router.
    * Persistent state management for complex, potentially long-running tasks.
* **Hybrid Agent Architecture Concept**: Includes a `DocumentAnalysisCrew` (built with CrewAI) integrated as a tool for optional in-depth analysis of specific documents by a dedicated team of sub-agents.
* **Customizable Tools for Agents**: Includes tools for live ArXiv searching, knowledge base retrieval, and the CrewAI-powered deep document analysis.
* **Comprehensive Evaluation Suite**:
    * `RagEvaluator` for retrieval metrics (Hit Rate, MRR, Precision@K).
    * `SynthesisEvaluator` using LLM-as-a-Judge for assessing synthesis quality (relevance, faithfulness).
    * `WandBMetricsLogger` for seamless integration with Weights & Biases experiment tracking.
* **CLI Interface**: Python scripts (`run_ingestion.py`, `run_cognitive_swarm.py`, `run_evaluation.py`) to manage the system.
* **Jupyter Notebooks**: For environment setup, component demonstration, experimentation, and end-to-end testing.
* **Reproducible Environment**: Defined via `environment.yml` (Conda) and `requirements.txt` (pip).
* **Containerization**: `Dockerfile` provided for building a portable application image.
* **API Layer (Basic)**: A FastAPI application (`src/api/main.py`) providing an endpoint to interact with the Cognitive Swarm.

### 🛠️ Tech Stack & Architecture

* **Core Language**: Python 3.11+
* **LLM Orchestration**: LangGraph
* **Specialized Agent Teams**: CrewAI (for specific sub-tasks like deep document analysis)
* **RAG & Data Indexing**: LlamaIndex (interfacing with MongoDB Atlas Vector Search)
* **LLM Interactions**: LangChain (agents, prompts, LLM wrappers).
* **Centralized LLM Management**: The `src/llm_services/llm_factory.py` module plays a crucial role by centralizing the instantiation of Language Models (LLMs). It allows for consistent selection and configuration of the LLM (OpenAI, Hugging Face API, Ollama) for all agents (LangGraph, CrewAI) and other components (like `SynthesisEvaluator`) based on parameters defined in `.env` and `config/settings.py`. This approach facilitates maintenance and flexibility in choosing model providers. It also integrates specific mechanisms like `StreamFallbackChatHuggingFace` to improve streaming compatibility for certain providers.
* **Generative LLMs (Agents, Synthesis)**:
    * **Default Provider:** Hugging Face API (using `mistralai/Mixtral-8x7B-Instruct-v0.1` by default).
    * **Configurable:** Supports OpenAI (e.g., `gpt-4o`), Hugging Face API (e.g., `Mixtral-8x7B`), and Ollama (e.g., `mistral`, `llama3`) via the `DEFAULT_LLM_MODEL_PROVIDER` variable in the `.env` file. *The instantiation and configuration of these models are managed by `src/llm_services/llm_factory.py`.*
* **Embedding Models (RAG)**:
    * **Default Provider:** Hugging Face (using `sentence-transformers/all-MiniLM-L6-v2` by default, 384 dimension).
    * **Configurable:** Supports OpenAI (e.g., `text-embedding-3-small`, 1536 dimension), Hugging Face (Sentence Transformers models), and Ollama (e.g., `nomic-embed-text`, 768 dimension) via the `DEFAULT_EMBEDDING_PROVIDER` variable in the `.env` file.
* **Required API Keys:** `OPENAI_API_KEY` (if OpenAI is used for generative LLMs or embeddings), `HUGGINGFACE_API_KEY` (if the Hugging Face API is used for generative LLMs). No API key for local Hugging Face embeddings (Sentence Transformers) or Ollama (requires a running Ollama instance).
* **Vector Database & Checkpointing**: MongoDB Atlas
* **Data Processing**: PyMuPDF (PDF parsing), TikToken (tokenization), Pandas
* **External APIs**: ArXiv Python library
* **Experiment Tracking**: Weights & Biases (W&B)
* **Environment Management**: Conda, Pip
* **Evaluation**: Custom evaluators, LLM-as-a-Judge
* **API**: FastAPI, Uvicorn

**High-Level Architecture:**

1.  **Data Ingestion**: ArXiv papers are downloaded, parsed, chunked, embedded (configurable provider: OpenAI, HuggingFace, or Ollama; defaults to HuggingFace with `sentence-transformers/all-MiniLM-L6-v2`), and stored in a MongoDB collection. Atlas Vector Search and text indexes are created (the vector index dimension adapts to the chosen embedding model).
2.  **User Query**: Submitted via CLI or API.
3.  **LangGraph Workflow (`CognitiveSwarm`)**:
    * *LLM instances for the agents in this workflow (Planner, ArXiv Searcher, Document Analyzer, Synthesizer) are provided by the centralized module `src/llm_services/llm_factory.py`, ensuring consistent configuration and provider selection (OpenAI, Hugging Face API, Ollama) across the system.*
    * A `ResearchPlannerAgent` creates a research plan.
    * A `router_after_planner` directs flow:
        * `ArxivSearchAgent` may search ArXiv for new papers (using `arxiv_search_tool`).
        * `DocumentAnalysisAgent` retrieves relevant chunks from MongoDB (using `knowledge_base_retrieval_tool`). If a deep dive on a specific document is needed per the plan, it can use the `document_deep_dive_analysis_tool` (which runs the CrewAI team).
    * A `SynthesisAgent` consolidates all information into a final report/answer.
    * Workflow state is persisted in MongoDB using `MongoDBSaver`.
4.  **Evaluation**: Scripts and notebooks use `RagEvaluator` and `SynthesisEvaluator`, logging results to W&B via `WandBMetricsLogger`.

## 📁 Directory Structure
```
cognitive-swarm-agents/
├── config/              # Configuration files (settings.py, logging_config.py)
├── data/                # Local data (corpus, evaluation dataset examples)
│   ├── corpus/          # Potentially downloaded ArXiv papers (PDFs, metadata)
│   └── evaluation/      # Example JSON evaluation dataset files
├── notebooks/           # Jupyter notebooks for setup, demos, experiments
├── scripts/             # CLI scripts (run_ingestion.py, run_cognitive_swarm.py, run_evaluation.py)
├── src/                 # Source code for the project
│   ├── agents/          # Agent architectures (LangGraph) and tool definitions
│   │   └── crewai_teams/ # CrewAI team definitions (e.g., document_analysis_crew.py)
│   ├── api/             # FastAPI application (main.py, schemas.py)
│   ├── data_processing/ # Modules for data ingestion and preprocessing
│   ├── evaluation/      # Modules for RAG/synthesis evaluation and W&B logging
│   ├── graph/           # LangGraph workflow definition and checkpointer
│   ├── llm_services/    # Modules for LLM management and instantiation (e.g., llm_factory.py)
│   ├── rag/             # RAG engine using LlamaIndex
│   └── vector_store/    # MongoDB manager for collections and indexes
├── .env                 # Local environment variables (API Keys, MONGODB_URI - GIT IGNORED)
├── .env.example         # Example template for .env
├── environment.yml      # Conda environment definition
├── requirements.txt     # Pip requirements file
├── Dockerfile           # Instructions to build the Docker image
├── .dockerignore        # Specifies files to ignore when building the Docker image
└── README.md            # This file
```

## ⚙️ Setup Instructions

1.  **Clone the Repository**:
    ```bash
    git clone <repository_url> # Remplacez par l'URL de votre dépôt GitHub
    cd cognitive-swarm-agents
    ```

2.  **Create and Activate Conda Environment**:
    * Ensure you have Conda installed.
    * Create the environment from the `environment.yml` file:
        ```bash
        conda env create -f environment.yml
        ```
    * Activate the environment:
        ```bash
        conda activate cognitive-swarm
        ```

3.  **Set Up Environment Variables (`.env` file)**:
    * Create a file named `.env` in the root directory of the project by copying `.env.example`.
    * Check and adjust the following variables in your `.env` file to select your language model and embedding providers (default values are in `config/settings.py` and commented out in `.env.example`):
        * `DEFAULT_LLM_MODEL_PROVIDER` (ex: "huggingface_api", "openai", "ollama") et les variables associées (`HUGGINGFACE_API_KEY`, `HUGGINGFACE_REPO_ID`, `OLLAMA_GENERATIVE_MODEL_NAME`, etc.).
        * `DEFAULT_EMBEDDING_PROVIDER` (ex: "huggingface", "openai", "ollama") et les variables associées (`OPENAI_EMBEDDING_MODEL_NAME`, `HUGGINGFACE_EMBEDDING_MODEL_NAME`, `OLLAMA_EMBEDDING_MODEL_NAME`, etc.).
        * `OLLAMA_BASE_URL` si vous utilisez Ollama.

4.  **(Optional) W&B Login**: If `WANDB_API_KEY` is not set in `.env`, you might need to log in via the CLI for W&B logging to work:
    ```bash
    wandb login
    ```

5.  **Verify Setup**: Run the first Jupyter notebook to ensure your environment is correctly configured:
    ```bash
    # Assurez-vous que votre environnement Conda 'cognitive-swarm' est activé
    jupyter notebook notebooks/00_setup_environment.ipynb
    ```
    Follow the instructions within the notebook.

## 🚀 Running the Project

All commands below should be run from the root directory of the project (`cognitive-swarm-agents/`) with the `cognitive-swarm` Conda environment activated.

### 1. Data Ingestion

To populate your MongoDB database with ArXiv papers (default query is on "Reinforcement Learning for Robotics"):
```bash
python -m scripts.run_ingestion --max_results 5 --log_level INFO
```

* Use `--query "your custom query"` to change the search topic.
* Use `--skip_download` to re-process local files in `data/corpus/rl_robotics_arxiv/` without re-downloading.
* This script handles downloading, parsing, chunking, embedding, storage, and index creation.

### 2. Running the Cognitive Swarm

To submit a query to the multi-agent system:
```bash
python -m scripts.run_cognitive_swarm --query "What are the latest advancements in using large language models for robot task planning?" --log_level INFO
```

* A `thread_id` will be generated (or you can provide one with `--thread_id`) for conversation history and checkpointing.
* Output from agents and tools will stream to the console. The final synthesized output will be displayed at the end.
* Use `--log_level DEBUG` for highly detailed verbose output.

### 3. Running Evaluations

To evaluate the RAG performance and synthesis quality (ensure evaluation datasets are prepared as per `notebooks/07_evaluation_and_logging.ipynb`):
```bash
python -m scripts.run_evaluation --eval_type all \
    --rag_dataset data/evaluation/rag_eval_dataset.json \
    --synthesis_dataset data/evaluation/synthesis_eval_dataset.json \
    --wandb_project "CognitiveSwarm-MyEvals" \
    --wandb_run_name "Eval_Run_$(date +%Y%m%d_%H%M)" \
    --log_level INFO
```

* Use `--wandb_disabled` to skip W&B logging.

### 4. Running the API (Optional)

If you want to expose the Cognitive Swarm via a FastAPI:
```bash
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

You can then send POST requests to `http://localhost:8000/invoke_swarm`.

## 📓 Notebooks Overview

The `notebooks/` directory provides detailed examples and demonstrations:

* `00_setup_environment.ipynb`: Environment configuration and verification.
* `01_data_ingestion_and_embedding.ipynb`: Step-by-step data ingestion pipeline.
* `02_rag_strategies_exploration.ipynb`: Exploring RAG with `RetrievalEngine`.
* `03_agent_development_and_tooling.ipynb`: Testing individual agents and their tools.
* `04_langgraph_workflow_design.ipynb`: Executing and observing the LangGraph workflow.
* `05_crewai_team_integration.ipynb`: Demonstrating the CrewAI `DocumentAnalysisCrew`.
* `06_end_to_end_pipeline_test.ipynb`: In-depth test of the full pipeline on a complex query.
* `07_evaluation_and_logging.ipynb`: Using evaluation modules and logging to W&B.

## 🔮 Future Work / To-Do (Conceptual)

* Implement more sophisticated, LLM-based routing logic in `main_workflow.py`.
* Integrate more advanced RAG strategies into `RetrievalEngine` (e.g., Parent Document Retriever, HyDE).
* Develop and integrate a `QualityCheckAgent` for reviewing synthesized outputs.
* Expand the toolset for agents (e.g., broader web search, code execution for data analysis).
* Enhance the FastAPI with more features and robust error handling.
* Implement comprehensive unit and integration tests in the `tests/` directory.

## 📄 License
```
MIT License

Copyright (c) 2025 [Estèphe ARNAUD / Cognitive Swarm: Multi-Agent Knowledge Discovery Engine]

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
```