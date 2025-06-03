"""
Embedder Module

This module provides functionality for generating embeddings from text chunks using various
embedding providers (OpenAI, HuggingFace, Ollama). It handles the embedding generation process,
metadata management, and batch processing of text chunks.

Key features:
- Multiple embedding provider support
- Batch processing with configurable batch size
- Comprehensive metadata management
- Robust error handling and logging
"""

import logging
import time
from typing import List, Dict, Optional, TypedDict, Any

from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings, OllamaEmbeddings

from config.settings import settings
# Importer ProcessedChunk depuis preprocessor.py pour utiliser la nouvelle structure
from src.data_processing.preprocessor import ProcessedChunk 

logger = logging.getLogger(__name__)

# La typé ProcessedChunkWithEmbedding n'est plus utilisée directement pour la sortie de cette fonction,
# car la structure change pour inclure un champ 'metadata' imbriqué.
# La fonction retournera List[Dict[str, Any]]

def get_embedding_client() -> Any:
    """
    Initialize and return an embedding client based on the configured provider.
    
    Returns:
        An embedding client instance (OpenAIEmbeddings, HuggingFaceEmbeddings, or OllamaEmbeddings)
        
    Raises:
        ValueError: If the provider is not supported or required configuration is missing
    """
    provider = settings.DEFAULT_EMBEDDING_PROVIDER.lower()
    logger.info(f"Initializing embedding client for provider: {provider}")

    if provider == "openai":
        if not settings.OPENAI_API_KEY:
            logger.error("OpenAI API key is not configured in settings for embeddings.")
            raise ValueError("OpenAI API key is missing for OpenAI embeddings.")
        
        model_kwargs = {}
        if settings.OPENAI_EMBEDDING_MODEL_NAME in ["text-embedding-3-small", "text-embedding-3-large"]:
            native_max_dim = 1536 if settings.OPENAI_EMBEDDING_MODEL_NAME == "text-embedding-3-small" else 3072
            if settings.OPENAI_EMBEDDING_DIMENSION < native_max_dim:
                model_kwargs["dimensions"] = settings.OPENAI_EMBEDDING_DIMENSION
        
        logger.info(
            f"Using OpenAIEmbeddings with model: {settings.OPENAI_EMBEDDING_MODEL_NAME}, "
            f"dimension: {settings.OPENAI_EMBEDDING_DIMENSION}"
        )
        return OpenAIEmbeddings(
            openai_api_key=settings.OPENAI_API_KEY,
            model=settings.OPENAI_EMBEDDING_MODEL_NAME,
            **model_kwargs
        )
    elif provider == "huggingface":
        logger.info(f"Using HuggingFaceEmbeddings with model: {settings.HUGGINGFACE_EMBEDDING_MODEL_NAME}")
        return HuggingFaceEmbeddings(
            model_name=settings.HUGGINGFACE_EMBEDDING_MODEL_NAME,
        )
    elif provider == "ollama":
        if not settings.OLLAMA_BASE_URL:
            logger.error("Ollama base URL is not configured for embeddings.")
            raise ValueError("Ollama base URL is missing for Ollama embeddings.")
        logger.info(f"Using OllamaEmbeddings with model: {settings.OLLAMA_EMBEDDING_MODEL_NAME} via {settings.OLLAMA_BASE_URL}")
        return OllamaEmbeddings(
            base_url=settings.OLLAMA_BASE_URL,
            model=settings.OLLAMA_EMBEDDING_MODEL_NAME
        )
    else:
        logger.error(f"Unsupported embedding provider: {provider}")
        raise ValueError(f"Unsupported embedding provider: {provider}")

def generate_embeddings_for_chunks(
    processed_chunks: List[ProcessedChunk], # Utilise ProcessedChunk mis à jour
    batch_size: int = 32 
) -> List[Dict[str, Any]]: # Le type de retour est maintenant List[Dict[str, Any]]
    """
    Generate embeddings for a list of processed text chunks.
    
    Args:
        processed_chunks: List of processed text chunks to embed
        batch_size: Number of chunks to process in each batch
        
    Returns:
        List of dictionaries containing the embedded chunks with their metadata
    """
    if not processed_chunks:
        logger.warning("No processed chunks provided for embedding.")
        return []

    logger.info(f"Starting embedding generation for {len(processed_chunks)} chunks.")
    
    embedding_client = get_embedding_client()
    texts_to_embed: List[str] = [chunk["text_chunk"] for chunk in processed_chunks]
    # MODIFICATION: La liste retournée sera de dictionnaires génériques
    final_chunks_for_db: List[Dict[str, Any]] = [] 
    
    current_embedding_provider = settings.DEFAULT_EMBEDDING_PROVIDER.lower()
    actual_model_name = ""
    # actual_dimension n'est plus nécessaire ici car on stocke len(embedding_vector)

    if current_embedding_provider == "openai":
        actual_model_name = settings.OPENAI_EMBEDDING_MODEL_NAME
    elif current_embedding_provider == "huggingface":
        actual_model_name = settings.HUGGINGFACE_EMBEDDING_MODEL_NAME
    elif current_embedding_provider == "ollama":
        actual_model_name = settings.OLLAMA_EMBEDDING_MODEL_NAME
    else: 
        logger.error(f"Unknown embedding provider '{current_embedding_provider}' in generate_embeddings_for_chunks.")
        return []

    for i in range(0, len(texts_to_embed), batch_size):
        batch_texts = texts_to_embed[i:i + batch_size]
        batch_original_chunks = processed_chunks[i:i + batch_size]
        
        logger.info(f"Embedding batch {i//batch_size + 1}/{(len(texts_to_embed) -1)//batch_size + 1} (size: {len(batch_texts)}) using {current_embedding_provider} provider.")
        
        try:
            embeddings = embedding_client.embed_documents(batch_texts)
            
            if len(embeddings) != len(batch_texts):
                logger.error(f"Mismatch in number of embeddings ({len(embeddings)}) and texts ({len(batch_texts)}) in batch {i//batch_size + 1}.")
                continue

            for original_chunk, embedding_vector in zip(batch_original_chunks, embeddings):
                # MODIFICATION: Construire la structure du document avec un champ 'metadata' imbriqué
                metadata_sub_document: Dict[str, Any] = {
                    "chunk_id": original_chunk['chunk_id'], # Peut être utile de le garder dans les métadonnées aussi
                    "arxiv_id": original_chunk['arxiv_id'],
                    "original_document_title": original_chunk.get('original_document_title'),
                    "embedding_model": actual_model_name,
                    "embedding_provider": current_embedding_provider,
                    "embedding_dimension": len(embedding_vector) if embedding_vector else 0
                }
                
                # Ajouter les métadonnées de la source originale (si elles existent)
                # au dictionnaire de métadonnées
                source_meta = original_chunk.get("source_document_metadata")
                if source_meta and isinstance(source_meta, dict):
                    # On peut choisir d'ajouter toutes les clés ou seulement certaines
                    # Ici, on ajoute celles qui ne sont pas déjà explicitement gérées
                    # pour éviter les écrasements, ou on fusionne intelligemment.
                    # Pour la simplicité, on fusionne, en donnant la priorité aux clés déjà définies.
                    for key, value in source_meta.items():
                        if key not in metadata_sub_document: # Évite d'écraser 'title' si 'original_document_title' est déjà là
                            metadata_sub_document[key] = value
                        elif key == "title" and "original_document_title" not in metadata_sub_document: # Cas spécifique pour le titre
                             metadata_sub_document["original_document_title"] = value


                chunk_for_db = {
                    "chunk_id": original_chunk['chunk_id'], # Utilisé pour _id dans MongoDB
                    "text_chunk": original_chunk['text_chunk'],
                    "embedding": embedding_vector,
                    "metadata": metadata_sub_document # Le dictionnaire de métadonnées imbriqué
                }
                final_chunks_for_db.append(chunk_for_db)

        except Exception as e:
            logger.error(f"Error embedding batch {i//batch_size + 1} with {current_embedding_provider}: {e}", exc_info=True)
        
    logger.info(f"Finished embedding generation. Successfully structured {len(final_chunks_for_db)} chunks for DB out of {len(processed_chunks)}.")
    return final_chunks_for_db

def _create_test_chunks() -> List[ProcessedChunk]:
    """
    Create sample processed chunks for testing.
    
    Returns:
        List of sample processed chunks
    """
    return [
        {
            "chunk_id": "test001_chunk_001",
            "arxiv_id": "test001",
            "text_chunk": "This is the first chunk of text from document one. It discusses reinforcement learning.",
            "original_document_title": "A Study of Interesting Things",
            "source_document_metadata": {
                "title": "A Study of Interesting Things",
                "primary_category": "cs.AI",
                "authors": ["A. B."]
            }
        },
        {
            "chunk_id": "test001_chunk_002",
            "arxiv_id": "test001",
            "text_chunk": "The second chunk continues exploring concepts related to robotics and AI.",
            "original_document_title": "A Study of Interesting Things",
            "source_document_metadata": {
                "title": "A Study of Interesting Things",
                "primary_category": "cs.AI",
                "authors": ["A. B."]
            }
        }
    ]

def _check_provider_configuration(provider: str) -> bool:
    """
    Check if the embedding provider is properly configured.
    
    Args:
        provider: Name of the embedding provider to check
        
    Returns:
        True if the provider is properly configured, False otherwise
    """
    if provider == "openai":
        if not settings.OPENAI_API_KEY:
            logger.error("OPENAI_API_KEY not found. Skipping OpenAI embedding test.")
            return False
        return True
        
    elif provider == "huggingface":
        return True
        
    elif provider == "ollama":
        if not (settings.OLLAMA_BASE_URL and settings.OLLAMA_EMBEDDING_MODEL_NAME):
            logger.error(
                "OLLAMA_BASE_URL or OLLAMA_EMBEDDING_MODEL_NAME not set. "
                "Skipping Ollama embedding test."
            )
            return False
            
        logger.info(
            f"Attempting Ollama test. Ensure Ollama is running at {settings.OLLAMA_BASE_URL} "
            f"and model '{settings.OLLAMA_EMBEDDING_MODEL_NAME}' is pulled."
        )
        return True
        
    return False

if __name__ == "__main__":
    from config.logging_config import setup_logging
    setup_logging(level="INFO")

    logger.info("--- Starting embedder.py test run ---")

    # Create test data
    sample_processed_chunks = _create_test_chunks()
    provider_to_test = settings.DEFAULT_EMBEDDING_PROVIDER
    logger.info(f"Testing with embedding provider: {provider_to_test}")

    # Check provider configuration
    if _check_provider_configuration(provider_to_test):
        try:
            structured_chunks_for_db = generate_embeddings_for_chunks(
                sample_processed_chunks,
                batch_size=2
            )

            if structured_chunks_for_db:
                logger.info(
                    f"Successfully generated structured data for {len(structured_chunks_for_db)} "
                    f"chunks using '{provider_to_test}'."
                )
                
                for i, chunk_data in enumerate(structured_chunks_for_db):
                    logger.info(f"--- Chunk {i+1} for DB ({chunk_data['chunk_id']}) ---")
                    logger.info(f"  Text: {chunk_data['text_chunk'][:50]}...")
                    logger.info(f"  Embedding Vector (first 3 dims): {chunk_data['embedding'][:3]}...")
                    logger.info(f"  Metadata field: {chunk_data['metadata']}")
                    
                    # Verify required metadata fields
                    assert "arxiv_id" in chunk_data["metadata"], "arxiv_id missing in metadata"
                    assert "embedding_model" in chunk_data["metadata"], "embedding_model missing in metadata"
                    assert chunk_data["metadata"].get("primary_category") == "cs.AI"
            else:
                logger.warning(
                    f"No structured chunks were generated in the test run for provider "
                    f"'{provider_to_test}'."
                )
        except Exception as e:
            logger.error(
                f"Error during embedding generation test for provider '{provider_to_test}': {e}",
                exc_info=True
            )
    else:
        logger.info(
            f"Skipping test for provider '{provider_to_test}' due to missing configuration "
            "or setup."
        )

    logger.info("--- embedder.py test run finished ---")