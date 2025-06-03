"""
MAKERS Configuration Module

This module defines the central configuration for the MAKERS system using Pydantic settings.
It handles:
- Environment variable loading
- Default configuration values
- Configuration validation
- Provider-specific settings (LLM, Embeddings, etc.)

Configuration is loaded from:
1. Environment variables
2. .env file
3. Default values (as fallback)
"""

import os
from typing import List, Optional
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

class Settings(BaseSettings):
    """
    Centralized application settings.
    
    Settings are loaded from environment variables and/or a .env file.
    All settings can be overridden by environment variables using the same name.
    """
    
    # --- General Project Settings ---
    PROJECT_NAME: str = "MAKERS: Multi Agent Knowledge Exploration & Retrieval System"
    DEBUG: bool = False
    PYTHON_ENV: str = "development"  # development, staging, production

    # --- API Keys & Authentication ---
    OPENAI_API_KEY: Optional[str] = None
    ANTHROPIC_API_KEY: Optional[str] = None
    GROQ_API_KEY: Optional[str] = None
    HUGGINGFACE_API_KEY: Optional[str] = None
    TAVILY_API_KEY: Optional[str] = None
    WANDB_API_KEY: Optional[str] = None

    # --- MongoDB Configuration ---
    MONGODB_URI: str = "mongodb://localhost:27017/"
    MONGO_DATABASE_NAME: str = "makers_db"
    MONGODB_COLLECTION_NAME: str = "makers_collection"  # Collection principale pour les données
    MONGO_MAX_POOL_SIZE: int = 50
    MONGO_TIMEOUT_MS: int = 5000  # 5 seconds
    LANGGRAPH_CHECKPOINTS_COLLECTION: str = "langgraph_checkpoints"

    # --- LLM Provider Configuration ---
    DEFAULT_LLM_MODEL_PROVIDER: str = "ollama"
    DEFAULT_OPENAI_GENERATIVE_MODEL: str = "gpt-4"
    HUGGINGFACE_REPO_ID: Optional[str] = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    OLLAMA_BASE_URL: Optional[str] = "http://localhost:11434"
    OLLAMA_GENERATIVE_MODEL_NAME: Optional[str] = "mistral"

    # --- Embedding Configuration ---
    DEFAULT_EMBEDDING_PROVIDER: str = "ollama"
    
    # OpenAI Embeddings
    OPENAI_EMBEDDING_MODEL_NAME: str = "text-embedding-3-small"
    OPENAI_EMBEDDING_DIMENSION: int = 1536
    
    # HuggingFace Embeddings
    HUGGINGFACE_EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
    HUGGINGFACE_EMBEDDING_MODEL_DIMENSION: int = 384
    
    # Ollama Embeddings
    OLLAMA_EMBEDDING_MODEL_NAME: str = "nomic-embed-text"
    OLLAMA_EMBEDDING_MODEL_DIMENSION: int = 768

    # --- Data Processing Configuration ---
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200

    # --- ArXiv Configuration ---
    DATA_DIR: Path = Path(__file__).resolve().parent.parent / "data"
    ARXIV_DEFAULT_QUERY: str = "Reinforcement Learning for Robotics"
    ARXIV_MAX_RESULTS: int = 10
    ARXIV_SORT_BY: str = "SubmittedDate"
    ARXIV_SORT_ORDER: str = "Descending"
    ARXIV_DOWNLOAD_DELAY_SECONDS: int = 3

    # --- Evaluation Configuration ---
    EVALUATION_DATASET_PATH: Optional[str] = str(DATA_DIR / "evaluation/rag_eval_dataset.json")

    # --- API Configuration ---
    API_V1_STR: str = "/api/v1"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

# Create global settings instance
settings = Settings()

if __name__ == "__main__":
    """Print current configuration when run directly."""
    print(f"Project Name: {settings.PROJECT_NAME}")
    print(f"Debug Mode: {settings.DEBUG}")

    print("\n--- Generative LLM Configuration ---")
    print(f"Default Generative LLM Provider: {settings.DEFAULT_LLM_MODEL_PROVIDER}")
    print(f"  OpenAI Model: {settings.DEFAULT_OPENAI_GENERATIVE_MODEL}")
    print(f"  HuggingFace Repo ID: {settings.HUGGINGFACE_REPO_ID}")
    print(f"  Ollama Model: {settings.OLLAMA_GENERATIVE_MODEL_NAME}")
    print(f"  Ollama Base URL: {settings.OLLAMA_BASE_URL}")

    print("\n--- Embedding Configuration ---")
    print(f"Default Embedding Provider: {settings.DEFAULT_EMBEDDING_PROVIDER}")
    print(f"  OpenAI Model: {settings.OPENAI_EMBEDDING_MODEL_NAME}")
    print(f"  OpenAI Dimension: {settings.OPENAI_EMBEDDING_DIMENSION}")
    print(f"  HuggingFace Model: {settings.HUGGINGFACE_EMBEDDING_MODEL_NAME}")
    print(f"  HuggingFace Dimension: {settings.HUGGINGFACE_EMBEDDING_MODEL_DIMENSION}")
    print(f"  Ollama Model: {settings.OLLAMA_EMBEDDING_MODEL_NAME}")
    print(f"  Ollama Dimension: {settings.OLLAMA_EMBEDDING_MODEL_DIMENSION}")

    print("\n--- API Keys (Presence) ---")
    print(f"OpenAI API Key: {'✓' if settings.OPENAI_API_KEY else '✗'}")
    print(f"HuggingFace API Key: {'✓' if settings.HUGGINGFACE_API_KEY else '✗'}")
    print(f"Anthropic API Key: {'✓' if settings.ANTHROPIC_API_KEY else '✗'}")
    print(f"Groq API Key: {'✓' if settings.GROQ_API_KEY else '✗'}")
    print(f"Tavily API Key: {'✓' if settings.TAVILY_API_KEY else '✗'}")
    print(f"Weights & Biases API Key: {'✓' if settings.WANDB_API_KEY else '✗'}")

    print("\n--- MongoDB Configuration ---")
    print(f"MongoDB URI: {settings.MONGODB_URI}")
    print(f"Database Name: {settings.MONGO_DATABASE_NAME}")
    print(f"Collection Name: {settings.MONGODB_COLLECTION_NAME}")
    print(f"Max Pool Size: {settings.MONGO_MAX_POOL_SIZE}")
    print(f"Timeout (ms): {settings.MONGO_TIMEOUT_MS}")

    print("\n--- Data & Paths ---")
    print(f"Data Directory: {settings.DATA_DIR}")
    print(f"Evaluation Dataset: {settings.EVALUATION_DATASET_PATH}")