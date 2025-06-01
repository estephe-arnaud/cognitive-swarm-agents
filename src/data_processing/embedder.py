# cognitive-swarm-agents/src/data_processing/embedder.py
import logging
from typing import List, Dict, Optional, TypedDict, Any
import time

from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings, OllamaEmbeddings

from config.settings import settings

logger = logging.getLogger(__name__)

# Define ProcessedChunk and ProcessedChunkWithEmbedding structures for clarity
class ProcessedChunk(TypedDict): # Duplicating from preprocessor for standalone clarity if needed
    chunk_id: str
    arxiv_id: str
    text_chunk: str
    original_document_title: Optional[str]
    # Ajout possible : metadata: Dict[str, Any] pour hériter des métadonnées du document d'origine

class ProcessedChunkWithEmbedding(ProcessedChunk):
    embedding: List[float]
    embedding_model: str # Store which model was used (e.g., "text-embedding-3-small" or "sentence-transformers/all-MiniLM-L6-v2")
    embedding_provider: str # Store which provider was used (e.g., "openai", "huggingface", "ollama")
    embedding_dimension: int # Store the actual dimension of the embedding

def get_embedding_client() -> Any: # LangChain's Embeddings classes are diverse, using Any for simplicity here
    """
    Initializes and returns an embedding client based on global settings.
    Supports OpenAI, HuggingFace (SentenceTransformers), and Ollama.
    """
    provider = settings.DEFAULT_EMBEDDING_PROVIDER.lower()
    logger.info(f"Initializing embedding client for provider: {provider}")

    if provider == "openai":
        if not settings.OPENAI_API_KEY:
            logger.error("OpenAI API key is not configured in settings for embeddings.")
            raise ValueError("OpenAI API key is missing for OpenAI embeddings.")
        
        model_kwargs = {}
        # Pour les modèles OpenAI supportant la paramétrisation de la dimension
        if settings.OPENAI_EMBEDDING_MODEL_NAME in ["text-embedding-3-small", "text-embedding-3-large"]:
            # text-embedding-3-small default = 1536, text-embedding-3-large default = 3072
            # On ne passe "dimensions" que si on veut une dimension plus petite que le max natif du modèle
            native_max_dim = 1536 if settings.OPENAI_EMBEDDING_MODEL_NAME == "text-embedding-3-small" else 3072
            if settings.OPENAI_EMBEDDING_DIMENSION < native_max_dim:
                model_kwargs["dimensions"] = settings.OPENAI_EMBEDDING_DIMENSION
        
        logger.info(f"Using OpenAIEmbeddings with model: {settings.OPENAI_EMBEDDING_MODEL_NAME}, dimension: {settings.OPENAI_EMBEDDING_DIMENSION} (model_kwargs: {model_kwargs})")
        return OpenAIEmbeddings(
            openai_api_key=settings.OPENAI_API_KEY,
            model=settings.OPENAI_EMBEDDING_MODEL_NAME,
            **model_kwargs
        )
    elif provider == "huggingface":
        # Utilise typiquement sentence-transformers.
        # HUGGINGFACE_API_KEY n'est généralement pas nécessaire pour les embeddings HuggingFace locaux/SentenceTransformers.
        # Il serait nécessaire si on utilisait une API d'embedding HuggingFace spécifique.
        logger.info(f"Using HuggingFaceEmbeddings with model: {settings.HUGGINGFACE_EMBEDDING_MODEL_NAME}")
        return HuggingFaceEmbeddings(
            model_name=settings.HUGGINGFACE_EMBEDDING_MODEL_NAME,
            # model_kwargs={'device': 'cuda'} # Optionnel: pour forcer l'utilisation du GPU si disponible
            # cache_folder: str = "./embedding_models_cache" # Optionnel: pour spécifier un dossier de cache
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
    processed_chunks: List[ProcessedChunk],
    batch_size: int = 32 
) -> List[ProcessedChunkWithEmbedding]:
    """
    Generates embeddings for a list of processed text chunks.
    """
    if not processed_chunks:
        logger.warning("No processed chunks provided for embedding.")
        return []

    logger.info(f"Starting embedding generation for {len(processed_chunks)} chunks.")
    
    embedding_client = get_embedding_client()
    texts_to_embed: List[str] = [chunk["text_chunk"] for chunk in processed_chunks]
    embedded_chunks: List[ProcessedChunkWithEmbedding] = []
    
    # Déterminer le nom du modèle et la dimension en fonction du fournisseur
    current_embedding_provider = settings.DEFAULT_EMBEDDING_PROVIDER.lower()
    actual_model_name = ""
    actual_dimension = 0

    if current_embedding_provider == "openai":
        actual_model_name = settings.OPENAI_EMBEDDING_MODEL_NAME
        actual_dimension = settings.OPENAI_EMBEDDING_DIMENSION
    elif current_embedding_provider == "huggingface":
        actual_model_name = settings.HUGGINGFACE_EMBEDDING_MODEL_NAME
        actual_dimension = settings.HUGGINGFACE_EMBEDDING_MODEL_DIMENSION
    elif current_embedding_provider == "ollama":
        actual_model_name = settings.OLLAMA_EMBEDDING_MODEL_NAME
        actual_dimension = settings.OLLAMA_EMBEDDING_MODEL_DIMENSION
    else: # Devrait être attrapé par get_embedding_client mais par sécurité
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
                # Vérifier la dimension réelle de l'embedding retourné si possible/nécessaire, 
                # surtout si le `actual_dimension` est juste une config et pas une garantie
                if embedding_vector and len(embedding_vector) != actual_dimension:
                    logger.warning(f"Embedding for chunk {original_chunk['chunk_id']} has dimension {len(embedding_vector)}, "
                                   f"but settings indicate {actual_dimension} for provider {current_embedding_provider} "
                                   f"with model {actual_model_name}. Using actual returned dimension.")
                    # On pourrait décider de stocker la dimension réelle retournée si elle diffère.
                    # Pour l'instant, on logue un avertissement et on stocke la dimension de la config.
                    # Il serait plus robuste de stocker len(embedding_vector).
                    
                chunk_with_embedding: ProcessedChunkWithEmbedding = {
                    **original_chunk, # type: ignore
                    "embedding": embedding_vector,
                    "embedding_model": actual_model_name,
                    "embedding_provider": current_embedding_provider,
                    "embedding_dimension": len(embedding_vector) if embedding_vector else 0 # Stocker la dimension réelle
                }
                embedded_chunks.append(chunk_with_embedding)

        except Exception as e:
            logger.error(f"Error embedding batch {i//batch_size + 1} with {current_embedding_provider}: {e}", exc_info=True)
        
    logger.info(f"Finished embedding generation. Successfully embedded {len(embedded_chunks)} chunks out of {len(processed_chunks)}.")
    return embedded_chunks


if __name__ == "__main__":
    from config.logging_config import setup_logging
    setup_logging(level="INFO")

    logger.info("--- Starting embedder.py test run ---")

    sample_chunks: List[ProcessedChunk] = [
        {
            "chunk_id": "test001_chunk_001", "arxiv_id": "test001",
            "text_chunk": "This is the first chunk of text from document one. It discusses reinforcement learning.",
            "original_document_title": "A Study of Interesting Things"
        },
        {
            "chunk_id": "test001_chunk_002", "arxiv_id": "test001",
            "text_chunk": "The second chunk continues exploring concepts related to robotics and AI.",
            "original_document_title": "A Study of Interesting Things"
        }
    ]

    # Pour tester un fournisseur spécifique, vous pouvez surcharger temporairement settings
    # ou configurer votre .env et settings.py en conséquence.
    # Exemple: settings.DEFAULT_EMBEDDING_PROVIDER = "huggingface"
    # Assurez-vous que les dépendances sont là (ex: sentence-transformers pour HuggingFaceEmbeddings)
    # et que Ollama est configuré et actif si vous testez "ollama".

    provider_to_test = settings.DEFAULT_EMBEDDING_PROVIDER
    logger.info(f"Testing with embedding provider: {provider_to_test}")

    # Vérifications de base pour les clés/configs nécessaires au provider testé
    can_run_test = False
    if provider_to_test == "openai":
        if settings.OPENAI_API_KEY:
            can_run_test = True
        else:
            logger.error("OPENAI_API_KEY not found. Skipping OpenAI embedding test.")
    elif provider_to_test == "huggingface":
        # HuggingFaceEmbeddings (local sentence-transformers) ne nécessite pas de clé API.
        can_run_test = True
    elif provider_to_test == "ollama":
        if settings.OLLAMA_BASE_URL and settings.OLLAMA_EMBEDDING_MODEL_NAME:
            # Idéalement, on pinguerait Ollama ici ou on vérifierait que le modèle est disponible.
            logger.info(f"Attempting Ollama test. Ensure Ollama is running at {settings.OLLAMA_BASE_URL} and model '{settings.OLLAMA_EMBEDDING_MODEL_NAME}' is pulled.")
            can_run_test = True
        else:
            logger.error("OLLAMA_BASE_URL or OLLAMA_EMBEDDING_MODEL_NAME not set. Skipping Ollama embedding test.")
    
    if can_run_test:
        try:
            chunks_with_embeddings = generate_embeddings_for_chunks(sample_chunks, batch_size=2)

            if chunks_with_embeddings:
                logger.info(f"Successfully generated embeddings for {len(chunks_with_embeddings)} chunks using '{provider_to_test}'.")
                for i, chunk_data in enumerate(chunks_with_embeddings):
                    logger.info(f"--- Chunk {i+1} ({chunk_data['chunk_id']}) ---")
                    logger.info(f"  Text: {chunk_data['text_chunk'][:50]}...")
                    logger.info(f"  Embedding Provider: {chunk_data['embedding_provider']}")
                    logger.info(f"  Embedding Model: {chunk_data['embedding_model']}")
                    logger.info(f"  Embedding Dimension (actual): {chunk_data['embedding_dimension']}") # Affiche la dimension réelle
                    logger.info(f"  Embedding Vector (first 3 dims): {chunk_data['embedding'][:3]}...")
                    
                    # Vérification de la dimension attendue basée sur la configuration
                    expected_dim = 0
                    if chunk_data['embedding_provider'] == "openai":
                        expected_dim = settings.OPENAI_EMBEDDING_DIMENSION
                    elif chunk_data['embedding_provider'] == "huggingface":
                        expected_dim = settings.HUGGINGFACE_EMBEDDING_MODEL_DIMENSION
                    elif chunk_data['embedding_provider'] == "ollama":
                        expected_dim = settings.OLLAMA_EMBEDDING_MODEL_DIMENSION
                    
                    if chunk_data['embedding_dimension'] != expected_dim:
                        logger.warning(f"    Dimension mismatch for {chunk_data['chunk_id']}: "
                                       f"Actual {chunk_data['embedding_dimension']} vs Configured {expected_dim}. "
                                       f"This might be due to model capabilities or configuration.")
                    else:
                        logger.info(f"    Dimension matches configured: {expected_dim}")
            else:
                logger.warning(f"No embeddings were generated in the test run for provider '{provider_to_test}'.")
        except Exception as e:
            logger.error(f"Error during embedding generation test for provider '{provider_to_test}': {e}", exc_info=True)
    else:
        logger.info(f"Skipping test for provider '{provider_to_test}' due to missing configuration or setup.")

    logger.info("--- embedder.py test run finished ---")