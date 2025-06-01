# cognitive-swarm-agents/config/settings.py
import os
from typing import List, Optional
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

# Charger les variables d'environnement depuis un fichier .env s'il existe
load_dotenv()

class Settings(BaseSettings):
    """
    Centralized application settings.
    Settings are loaded from environment variables and/or a .env file.
    """
    # General Project Settings
    PROJECT_NAME: str = "Cognitive Swarm: Multi-Agent Knowledge Discovery Engine"
    DEBUG: bool = False
    PYTHON_ENV: str = "development" # e.g., development, staging, production

    # --- LLM API Keys & Configuration (pour les modèles génératifs) ---
    OPENAI_API_KEY: Optional[str] = None
    ANTHROPIC_API_KEY: Optional[str] = None
    GROQ_API_KEY: Optional[str] = None

    # Hugging Face API Configuration (pour l'API d'inférence des modèles génératifs)
    HUGGINGFACE_API_KEY: Optional[str] = None
    HUGGINGFACE_REPO_ID: Optional[str] = "mistralai/Mixtral-8x7B-Instruct-v0.1" # Pour les modèles génératifs

    # Ollama Configuration (pour modèles génératifs ET embeddings servis localement)
    OLLAMA_BASE_URL: Optional[str] = "http://localhost:11434" # Utilisé pour les LLMs génératifs et les embeddings Ollama
    OLLAMA_GENERATIVE_MODEL_NAME: Optional[str] = "mistral" # Modèle génératif par défaut via Ollama

    # Tool/Service API Keys
    TAVILY_API_KEY: Optional[str] = None
    WANDB_API_KEY: Optional[str] = None

    # MongoDB Configuration
    MONGO_URI: str = "mongodb://localhost:27017/"
    MONGO_DATABASE_NAME: str = "cognitive_swarm_db"
    MONGO_MAX_POOL_SIZE: int = 50
    MONGO_TIMEOUT_MS: int = 5000 # 5 seconds

    # LangGraph Checkpointer Collection
    LANGGRAPH_CHECKPOINTS_COLLECTION: str = "langgraph_checkpoints"

    # --- Configuration des Modèles Génératifs par Défaut ---
    # Options pour DEFAULT_LLM_MODEL_PROVIDER: "openai", "huggingface_api", "ollama"
    DEFAULT_LLM_MODEL_PROVIDER: str = "huggingface_api"
    DEFAULT_OPENAI_GENERATIVE_MODEL: str = "gpt-4o" # Utilisé si DEFAULT_LLM_MODEL_PROVIDER="openai"
    # HUGGINGFACE_REPO_ID est utilisé pour le provider "huggingface_api"
    # OLLAMA_GENERATIVE_MODEL_NAME est utilisé pour le provider "ollama"

    # --- Configuration des Modèles d'Embedding (MODIFIÉE POUR INCLURE OLLAMA) ---
    # Options pour DEFAULT_EMBEDDING_PROVIDER: "openai", "huggingface", "ollama"
    DEFAULT_EMBEDDING_PROVIDER: str = "huggingface"  # Par défaut sur open source (HuggingFace)

    # Configuration pour les embeddings OpenAI
    OPENAI_EMBEDDING_MODEL_NAME: str = "text-embedding-3-small"
    OPENAI_EMBEDDING_DIMENSION: int = 1536 # Dimension native de text-embedding-3-small

    # Configuration pour les embeddings Hugging Face (via SentenceTransformers)
    HUGGINGFACE_EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
    HUGGINGFACE_EMBEDDING_MODEL_DIMENSION: int = 384 # Dimension de all-MiniLM-L6-v2

    # Configuration pour les embeddings Ollama
    # `nomic-embed-text` est un exemple populaire, assurez-vous qu'il est servi par votre instance Ollama.
    # D'autres modèles comme `mxbai-embed-large` ou même des modèles généralistes peuvent être utilisés.
    OLLAMA_EMBEDDING_MODEL_NAME: str = "nomic-embed-text"
    OLLAMA_EMBEDDING_MODEL_DIMENSION: int = 768 # Dimension de nomic-embed-text par défaut

    # --- Data Processing Configuration ---
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200

    # ArXiv Specific Configuration & Data Directory
    DATA_DIR: Path = Path(__file__).resolve().parent.parent / "data"
    ARXIV_DEFAULT_QUERY: str = "Reinforcement Learning for Robotics"
    ARXIV_MAX_RESULTS: int = 10
    ARXIV_SORT_BY: str = "submittedDate"
    ARXIV_SORT_ORDER: str = "descending"
    ARXIV_DOWNLOAD_DELAY_SECONDS: int = 3

    # Evaluation Configuration
    EVALUATION_DATASET_PATH: Optional[str] = str(DATA_DIR / "evaluation/rag_eval_dataset.json")

    # API Configuration (if using FastAPI)
    API_V1_STR: str = "/api/v1"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()

if __name__ == "__main__":
    print(f"Project Name: {settings.PROJECT_NAME}")
    print(f"Debug Mode: {settings.DEBUG}")

    print(f"\n--- Generative LLM Configuration ---")
    print(f"Default Generative LLM Provider: {settings.DEFAULT_LLM_MODEL_PROVIDER}")
    print(f"  OpenAI Model (if provider is openai): {settings.DEFAULT_OPENAI_GENERATIVE_MODEL}")
    print(f"  HuggingFace Repo ID (if provider is huggingface_api): {settings.HUGGINGFACE_REPO_ID}")
    print(f"  Ollama Generative Model Name (if provider is ollama): {settings.OLLAMA_GENERATIVE_MODEL_NAME}")
    print(f"  Ollama Base URL (for generative and embeddings): {settings.OLLAMA_BASE_URL}")


    print(f"\n--- Embedding Configuration ---")
    print(f"Default Embedding Provider: {settings.DEFAULT_EMBEDDING_PROVIDER}")
    print(f"  OpenAI Embedding Model: {settings.OPENAI_EMBEDDING_MODEL_NAME}")
    print(f"  OpenAI Embedding Dimension: {settings.OPENAI_EMBEDDING_DIMENSION}")
    print(f"  HuggingFace Embedding Model: {settings.HUGGINGFACE_EMBEDDING_MODEL_NAME}")
    print(f"  HuggingFace Embedding Dimension: {settings.HUGGINGFACE_EMBEDDING_MODEL_DIMENSION}")
    print(f"  Ollama Embedding Model Name: {settings.OLLAMA_EMBEDDING_MODEL_NAME}")
    print(f"  Ollama Embedding Model Dimension: {settings.OLLAMA_EMBEDDING_MODEL_DIMENSION}")


    print(f"\n--- API Keys (Presence) ---")
    print(f"OpenAI API Key Loaded: {bool(settings.OPENAI_API_KEY)}")
    print(f"HuggingFace API Key Loaded: {bool(settings.HUGGINGFACE_API_KEY)}")

    print(f"\n--- MongoDB Configuration ---")
    print(f"Mongo URI: {settings.MONGO_URI}")
    print(f"Mongo Database: {settings.MONGO_DATABASE_NAME}")

    print(f"\n--- Data & Paths ---")
    print(f"Data Directory: {settings.DATA_DIR}")
    print(f"RAG Evaluation Dataset Path: {settings.EVALUATION_DATASET_PATH}")