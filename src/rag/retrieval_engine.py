# src/rag/retrieval_engine.py
import logging
from typing import List, Optional, Dict, Any

from llama_index.core import VectorStoreIndex, StorageContext, QueryBundle
from llama_index.core.vector_stores import VectorStoreQuery, MetadataFilters, ExactMatchFilter
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.settings import Settings as LlamaSettings
from llama_index.vector_stores.mongodb import MongoDBAtlasVectorSearch

from llama_index.embeddings.openai import OpenAIEmbedding as LlamaOpenAIEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding as LlamaHuggingFaceEmbedding
from llama_index.embeddings.ollama import OllamaEmbedding as LlamaOllamaEmbedding

from config.settings import settings

logger = logging.getLogger(__name__)

class RetrievedNode:
    def __init__(self, text: str, score: Optional[float], metadata: Dict[str, Any]):
        self.text = text
        self.score = score
        self.metadata = metadata

    def __repr__(self) -> str:
        return f"RetrievedNode(score={self.score:.4f}, text='{self.text[:100]}...', metadata={self.metadata})"


class RetrievalEngine:
    DEFAULT_CHUNK_COLLECTION_NAME = "arxiv_chunks" 
    DEFAULT_VECTOR_INDEX_NAME = "default_vector_index" 

    def __init__(
        self,
        mongo_uri: str = settings.MONGODB_URI,
        db_name: str = settings.MONGO_DATABASE_NAME,
        collection_name: str = DEFAULT_CHUNK_COLLECTION_NAME,
        vector_index_name: str = DEFAULT_VECTOR_INDEX_NAME, # Nom du paramètre dans notre constructeur
        embedding_field: str = "embedding", 
        text_key: str = "text_chunk", 
        metadata_keys: Optional[List[str]] = None 
    ):
        self.mongo_uri = mongo_uri
        self.db_name = db_name
        self.collection_name = collection_name
        self.vector_index_name = vector_index_name # Stocke le nom de l'index
        self.embedding_field = embedding_field
        self.text_key = text_key
        
        if metadata_keys is None:
            self.metadata_keys = ["chunk_id", "arxiv_id", "original_document_title", 
                                  "embedding_model", "embedding_provider", "embedding_dimension"]
        else:
            self.metadata_keys = metadata_keys

        self._vector_store: Optional[MongoDBAtlasVectorSearch] = None
        self._index: Optional[VectorStoreIndex] = None
        self._retriever: Optional[BaseRetriever] = None 

        self._setup_llamaindex_components()
        logger.info("RetrievalEngine initialized with LlamaIndex components.")

    def _configure_llama_settings(self) -> None:
        provider = settings.DEFAULT_EMBEDDING_PROVIDER.lower()
        logger.info(f"Configuring LlamaIndex global embed_model for provider: {provider}")

        if provider == "openai":
            if not settings.OPENAI_API_KEY:
                logger.error("OpenAI API key is not configured for LlamaIndex OpenAIEmbedding.")
                raise ValueError("OpenAI API key is missing for LlamaIndex OpenAIEmbedding.")

            model_kwargs = {}
            if settings.OPENAI_EMBEDDING_MODEL_NAME in ["text-embedding-3-small", "text-embedding-3-large"]:
                native_max_dim = 1536 if settings.OPENAI_EMBEDDING_MODEL_NAME == "text-embedding-3-small" else 3072
                if settings.OPENAI_EMBEDDING_DIMENSION < native_max_dim:
                    model_kwargs["dimensions"] = settings.OPENAI_EMBEDDING_DIMENSION
            
            LlamaSettings.embed_model = LlamaOpenAIEmbedding(
                api_key=settings.OPENAI_API_KEY,
                model=settings.OPENAI_EMBEDDING_MODEL_NAME,
                **model_kwargs 
            )
            logger.info(f"LlamaIndex embed_model configured for OpenAI: {settings.OPENAI_EMBEDDING_MODEL_NAME} with dimension {settings.OPENAI_EMBEDDING_DIMENSION} (model_kwargs: {model_kwargs})")

        elif provider == "huggingface":
            logger.info(f"LlamaIndex embed_model configured for HuggingFace: {settings.HUGGINGFACE_EMBEDDING_MODEL_NAME}")
            LlamaSettings.embed_model = LlamaHuggingFaceEmbedding(
                model_name=settings.HUGGINGFACE_EMBEDDING_MODEL_NAME
            )
        elif provider == "ollama":
            if not settings.OLLAMA_BASE_URL:
                logger.error("Ollama base URL is not configured for LlamaIndex OllamaEmbedding.")
                raise ValueError("Ollama base URL is missing for LlamaIndex OllamaEmbedding.")
            logger.info(f"LlamaIndex embed_model configured for Ollama: model='{settings.OLLAMA_EMBEDDING_MODEL_NAME}', base_url='{settings.OLLAMA_BASE_URL}'")
            LlamaSettings.embed_model = LlamaOllamaEmbedding(
                model_name=settings.OLLAMA_EMBEDDING_MODEL_NAME,
                base_url=settings.OLLAMA_BASE_URL,
            )
        else:
            logger.error(f"Unsupported embedding provider for LlamaIndex: {provider}")
            raise NotImplementedError(f"Embedding provider {provider} not yet supported for LlamaIndex in RetrievalEngine.")

    def _setup_llamaindex_components(self) -> None:
        try:
            self._configure_llama_settings()

            self._vector_store = MongoDBAtlasVectorSearch(
                mongodb_client=None, 
                db_name=self.db_name,
                collection_name=self.collection_name,
                # MODIFICATION : Utiliser vector_index_name au lieu de index_name
                vector_index_name=self.vector_index_name, 
                uri=self.mongo_uri, 
                embedding_key=self.embedding_field,
                text_key=self.text_key,
            )
            logger.info(f"MongoDBAtlasVectorSearch store configured for collection '{self.collection_name}'.")

            if LlamaSettings.embed_model is None:
                    raise ValueError("LlamaSettings.embed_model was not configured prior to VectorStoreIndex creation.")

            self._index = VectorStoreIndex.from_vector_store(self._vector_store)
            logger.info(f"VectorStoreIndex loaded from MongoDB store.")

            self._retriever = self._index.as_retriever(similarity_top_k=5) 
            logger.info("Default retriever configured.")

        except Exception as e:
            logger.error(f"Error initializing LlamaIndex components: {e}", exc_info=True)
            self._vector_store = None
            self._index = None
            self._retriever = None
            raise 

    def retrieve_simple_vector_search(
        self,
        query_text: str,
        top_k: int = 5,
        metadata_filters: Optional[List[Dict[str, Any]]] = None 
    ) -> List[RetrievedNode]:
        # ... (corps de la fonction inchangé) ...
        if not self._index or not self._retriever:
            logger.error("RetrievalEngine not properly initialized. Call setup first or check for errors.")
            if not self._index:
                try:
                    self._setup_llamaindex_components()
                except Exception:
                    return [] 
            if not self._retriever: 
                logger.error("Retriever still not available after re-initialization attempt.")
                return []

        current_retriever = self._index.as_retriever(similarity_top_k=top_k)

        llama_filters = None
        if metadata_filters:
            filters_list = []
            for f_dict in metadata_filters:
                if "key" in f_dict and "value" in f_dict:
                    filters_list.append(ExactMatchFilter(key=f_dict["key"], value=f_dict["value"]))
                else:
                    logger.warning(f"Malformed metadata filter skipped (missing 'key' or 'value'): {f_dict}")
            if filters_list:
                llama_filters = MetadataFilters(filters=filters_list) 

        try:
            if llama_filters:
                filtered_retriever = self._index.as_retriever(
                    similarity_top_k=top_k,
                    filters=llama_filters 
                )
                retrieved_nodes_with_scores = filtered_retriever.retrieve(query_text)
            else:
                retrieved_nodes_with_scores = current_retriever.retrieve(query_text)
            
            results = []
            for node_with_score in retrieved_nodes_with_scores:
                results.append(
                    RetrievedNode(
                        text=node_with_score.get_content(), 
                        score=node_with_score.get_score(), 
                        metadata=node_with_score.metadata 
                    )
                )
            logger.info(f"Retrieved {len(results)} nodes for query: '{query_text[:50]}...' with top_k={top_k} and filters={metadata_filters}")
            return results
        except Exception as e:
            logger.error(f"Error during LlamaIndex retrieval: {e}", exc_info=True)
            return []

if __name__ == "__main__":
    # ... (bloc de test inchangé) ...
    from config.logging_config import setup_logging
    setup_logging(level="INFO")

    logger.info("Starting RetrievalEngine test run...")

    current_embedding_provider_for_test = settings.DEFAULT_EMBEDDING_PROVIDER.lower()
    logger.info(f"RetrievalEngine test will run using LlamaIndex configured for provider: '{current_embedding_provider_for_test}'")

    can_run_retrieval_test = False
    if current_embedding_provider_for_test == "openai":
        if settings.OPENAI_API_KEY:
            can_run_retrieval_test = True
        else:
            logger.error("OPENAI_API_KEY not set. Retrieval test with OpenAI embeddings will fail.")
    elif current_embedding_provider_for_test == "huggingface":
        can_run_retrieval_test = True
    elif current_embedding_provider_for_test == "ollama":
        if settings.OLLAMA_BASE_URL and settings.OLLAMA_EMBEDDING_MODEL_NAME:
            logger.info(f"Ensure Ollama is running at {settings.OLLAMA_BASE_URL} and model '{settings.OLLAMA_EMBEDDING_MODEL_NAME}' is available.")
            can_run_retrieval_test = True
        else:
            logger.error("OLLAMA_BASE_URL or OLLAMA_EMBEDDING_MODEL_NAME not set. Retrieval test with Ollama embeddings will fail.")
    
    if not settings.MONGODB_URI or ("<user>" in settings.MONGODB_URI and "localhost" not in settings.MONGODB_URI) : # Ajustement pour permettre localhost sans user/pass
        logger.warning(f"MONGODB_URI ('{settings.MONGODB_URI}') may not be correctly configured for Atlas or a remote instance. Localhost without credentials might be intended.")
        if "localhost" not in settings.MONGODB_URI :
             can_run_retrieval_test = False # Ne pas exécuter si ce n'est pas localhost et que l'URI semble incomplet


    if can_run_retrieval_test:
        try:
            retrieval_engine = RetrievalEngine() 

            test_query = "reinforcement learning for robotic arm manipulation"
            logger.info(f"\n--- Testing Simple Vector Search for query: '{test_query}' ---")
            
            results = retrieval_engine.retrieve_simple_vector_search(test_query, top_k=3)
            if results:
                logger.info(f"Found {len(results)} results:")
                for i, node in enumerate(results):
                    logger.info(f"  Result {i+1}: Score={node.score}, ArxivID={node.metadata.get('arxiv_id', 'N/A')}, Title={node.metadata.get('original_document_title', 'N/A')}")
                    logger.info(f"  Text: {node.text[:150]}...")
            else:
                logger.warning("No results returned from simple vector search. Check MongoDB data, index, API keys, and embedding model configuration.")
        except Exception as e:
            logger.error(f"An error occurred during RetrievalEngine test: {e}", exc_info=True)
    else:
        logger.info("Skipping RetrievalEngine test due to missing configuration for the selected provider or incomplete MONGODB_URI.")
    
    logger.info("RetrievalEngine test run finished.")