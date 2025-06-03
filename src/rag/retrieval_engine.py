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
    """Represents a retrieved document node with its metadata and relevance score."""
    
    def __init__(self, text: str, score: Optional[float], metadata: Dict[str, Any]):
        self.text = text
        self.score = score
        self.metadata = metadata

    def __repr__(self) -> str:
        return f"RetrievedNode(score={self.score:.4f}, text='{self.text[:100]}...', metadata={self.metadata})"


class RetrievalEngine:
    """Engine for retrieving relevant documents using vector similarity search."""
    
    DEFAULT_CHUNK_COLLECTION_NAME = "arxiv_chunks"
    DEFAULT_VECTOR_INDEX_NAME = "default_vector_index"
    DEFAULT_METADATA_KEYS = [
        "chunk_id", "arxiv_id", "original_document_title",
        "embedding_model", "embedding_provider", "embedding_dimension"
    ]

    def __init__(
        self,
        mongo_uri: str = settings.MONGODB_URI,
        db_name: str = settings.MONGO_DATABASE_NAME,
        collection_name: str = DEFAULT_CHUNK_COLLECTION_NAME,
        vector_index_name: str = DEFAULT_VECTOR_INDEX_NAME,
        embedding_field: str = "embedding",
        text_key: str = "text_chunk",
        metadata_keys: Optional[List[str]] = None
    ):
        self.mongo_uri = mongo_uri
        self.db_name = db_name
        self.collection_name = collection_name
        self.vector_index_name = vector_index_name
        self.embedding_field = embedding_field
        self.text_key = text_key
        self.metadata_keys = metadata_keys or self.DEFAULT_METADATA_KEYS

        self._vector_store: Optional[MongoDBAtlasVectorSearch] = None
        self._index: Optional[VectorStoreIndex] = None
        self._retriever: Optional[BaseRetriever] = None

        self._setup_llamaindex_components()
        logger.info("RetrievalEngine initialized with LlamaIndex components")

    def _configure_llama_settings(self) -> None:
        """Configure the global embedding model based on the selected provider."""
        provider = settings.DEFAULT_EMBEDDING_PROVIDER.lower()
        logger.info(f"Configuring LlamaIndex embed_model for provider: {provider}")

        if provider == "openai":
            if not settings.OPENAI_API_KEY:
                raise ValueError("OpenAI API key is missing")

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
            logger.info(f"Configured OpenAI embedding model: {settings.OPENAI_EMBEDDING_MODEL_NAME}")

        elif provider == "huggingface":
            LlamaSettings.embed_model = LlamaHuggingFaceEmbedding(
                model_name=settings.HUGGINGFACE_EMBEDDING_MODEL_NAME
            )
            logger.info(f"Configured HuggingFace embedding model: {settings.HUGGINGFACE_EMBEDDING_MODEL_NAME}")

        elif provider == "ollama":
            if not settings.OLLAMA_BASE_URL:
                raise ValueError("Ollama base URL is missing")

            LlamaSettings.embed_model = LlamaOllamaEmbedding(
                model_name=settings.OLLAMA_EMBEDDING_MODEL_NAME,
                base_url=settings.OLLAMA_BASE_URL
            )
            logger.info(f"Configured Ollama embedding model: {settings.OLLAMA_EMBEDDING_MODEL_NAME}")

        else:
            raise NotImplementedError(f"Unsupported embedding provider: {provider}")

    def _setup_llamaindex_components(self) -> None:
        """Initialize LlamaIndex components for vector search."""
        try:
            self._configure_llama_settings()

            self._vector_store = MongoDBAtlasVectorSearch(
                mongodb_client=None,
                db_name=self.db_name,
                collection_name=self.collection_name,
                vector_index_name=self.vector_index_name,
                uri=self.mongo_uri,
                embedding_key=self.embedding_field,
                text_key=self.text_key
            )
            logger.info(f"Configured MongoDB vector store for collection: {self.collection_name}")

            if LlamaSettings.embed_model is None:
                raise ValueError("Embedding model not configured")

            self._index = VectorStoreIndex.from_vector_store(self._vector_store)
            self._retriever = self._index.as_retriever(similarity_top_k=5)
            logger.info("Initialized vector store index and retriever")

        except Exception as e:
            logger.error(f"Failed to initialize LlamaIndex components: {e}", exc_info=True)
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
        """Retrieve relevant documents using vector similarity search."""
        if not self._index or not self._retriever:
            logger.error("RetrievalEngine not properly initialized")
            if not self._index:
                try:
                    self._setup_llamaindex_components()
                except Exception:
                    return []
            if not self._retriever:
                logger.error("Retriever not available after re-initialization")
                return []

        current_retriever = self._index.as_retriever(similarity_top_k=top_k)
        llama_filters = None

        if metadata_filters:
            filters_list = []
            for f_dict in metadata_filters:
                if "key" in f_dict and "value" in f_dict:
                    filters_list.append(ExactMatchFilter(key=f_dict["key"], value=f_dict["value"]))
                else:
                    logger.warning(f"Skipped malformed metadata filter: {f_dict}")
            if filters_list:
                llama_filters = MetadataFilters(filters=filters_list)

        try:
            if llama_filters:
                filtered_retriever = self._index.as_retriever(
                    similarity_top_k=top_k,
                    filters=llama_filters
                )
                retrieved_nodes = filtered_retriever.retrieve(query_text)
            else:
                retrieved_nodes = current_retriever.retrieve(query_text)

            results = [
                RetrievedNode(
                    text=node.get_content(),
                    score=node.get_score(),
                    metadata=node.metadata
                )
                for node in retrieved_nodes
            ]

            logger.info(f"Retrieved {len(results)} nodes for query: '{query_text[:50]}...'")
            return results

        except Exception as e:
            logger.error(f"Error during retrieval: {e}", exc_info=True)
            return []


if __name__ == "__main__":
    from config.logging_config import setup_logging
    setup_logging(level="INFO")

    logger.info("Starting RetrievalEngine test")

    provider = settings.DEFAULT_EMBEDDING_PROVIDER.lower()
    logger.info(f"Testing with embedding provider: {provider}")

    can_run_test = False
    if provider == "openai":
        can_run_test = bool(settings.OPENAI_API_KEY)
    elif provider == "huggingface":
        can_run_test = True
    elif provider == "ollama":
        can_run_test = bool(settings.OLLAMA_BASE_URL and settings.OLLAMA_EMBEDDING_MODEL_NAME)
        if can_run_test:
            logger.info(f"Ensure Ollama is running at {settings.OLLAMA_BASE_URL}")

    if not settings.MONGODB_URI or ("<user>" in settings.MONGODB_URI and "localhost" not in settings.MONGODB_URI):
        logger.warning("MongoDB URI may not be correctly configured")
        if "localhost" not in settings.MONGODB_URI:
            can_run_test = False

    if can_run_test:
        try:
            engine = RetrievalEngine()
            query = "reinforcement learning for robotic arm manipulation"
            logger.info(f"\nTesting vector search for query: '{query}'")

            results = engine.retrieve_simple_vector_search(query, top_k=3)
            if results:
                logger.info(f"Found {len(results)} results:")
                for i, node in enumerate(results):
                    logger.info(f"Result {i+1}:")
                    logger.info(f"  Score: {node.score}")
                    logger.info(f"  ArxivID: {node.metadata.get('arxiv_id', 'N/A')}")
                    logger.info(f"  Title: {node.metadata.get('original_document_title', 'N/A')}")
                    logger.info(f"  Text: {node.text[:150]}...")
            else:
                logger.warning("No results found. Check MongoDB data and configuration")

        except Exception as e:
            logger.error(f"Test failed: {e}", exc_info=True)
    else:
        logger.info("Skipping test due to missing configuration")

    logger.info("Test complete")