# src/vector_store/mongodb_manager.py
import logging
from typing import List, Dict, Any, Optional, Union
import time

from pymongo import MongoClient, TEXT
from pymongo.collection import Collection
from pymongo.database import Database
from pymongo.errors import ConnectionFailure, OperationFailure, BulkWriteError
from pymongo.operations import SearchIndexModel

from config.settings import settings

logger = logging.getLogger(__name__)

class MongoDBManager:
    """Manager for MongoDB operations including vector search and text search."""
    
    DEFAULT_CHUNK_COLLECTION_NAME = "arxiv_chunks"
    DEFAULT_VECTOR_INDEX_NAME = "default_vector_search_index"
    DEFAULT_TEXT_INDEX_NAME = "default_text_index"

    def __init__(self, mongo_uri: str = settings.MONGODB_URI, db_name: str = settings.MONGO_DATABASE_NAME):
        self.mongo_uri = mongo_uri
        self.db_name = db_name
        self.client: Optional[MongoClient] = None
        self.db: Optional[Database] = None
        logger.info(f"Initialized MongoDB manager for database: {self.db_name}")

    def connect(self) -> None:
        """Establish connection to MongoDB if not already connected."""
        if self.client is not None and self.db is not None:
            try:
                self.client.admin.command('ping')
                logger.debug("Already connected to MongoDB")
                return
            except ConnectionFailure:
                logger.warning("MongoDB connection lost, reconnecting...")
                self.client = None
                self.db = None

        try:
            self.client = MongoClient(
                self.mongo_uri,
                maxPoolSize=settings.MONGO_MAX_POOL_SIZE,
                serverSelectionTimeoutMS=settings.MONGO_TIMEOUT_MS,
                appName="MAKERSClient"
            )
            self.client.admin.command('ping')
            self.db = self.client[self.db_name]
            logger.info(f"Connected to MongoDB database: {self.db_name}")
        except ConnectionFailure as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            self.client = None
            self.db = None
            raise
        except Exception as e:
            logger.error(f"Unexpected error during MongoDB connection: {e}")
            self.client = None
            self.db = None
            raise

    def close(self) -> None:
        """Close the MongoDB connection."""
        if self.client:
            self.client.close()
            self.client = None
            self.db = None
            logger.info("MongoDB connection closed")

    def collection_exists(self, collection_name: str) -> bool:
        """Check if a collection exists in the current database."""
        if self.db is None:
            logger.warning("Database not connected, attempting to connect to check collection existence.")
            try:
                self.connect()
            except Exception: # Broad exception to catch connection issues
                logger.error("Failed to connect to database. Cannot check if collection exists.")
                return False # Or raise an error, but returning False is safer for a check

        if self.db is None: # If connection still failed
             logger.error("Database connection is not available. Cannot check collection existence.")
             return False

        try:
            return collection_name in self.db.list_collection_names()
        except OperationFailure as e:
            logger.error(f"Failed to list collection names (permissions issue or other DB error): {e}")
            return False # Assuming if we can't list, we can't be sure it exists or want to proceed.
        except Exception as e:
            logger.error(f"An unexpected error occurred while checking if collection '{collection_name}' exists: {e}")
            return False

    def get_collection(self, collection_name: str) -> Optional[Collection]:
        """Get a MongoDB collection, connecting if necessary."""
        if self.client is None or self.db is None:
            logger.warning("MongoDB client not connected, attempting to connect")
            try:
                self.connect()
            except Exception:
                logger.error("Failed to establish connection")
                return None

        return self.db[collection_name] if self.db is not None else None

    def get_effective_embedding_dimension(self) -> int:
        """Get the embedding dimension based on the configured provider."""
        provider = settings.DEFAULT_EMBEDDING_PROVIDER.lower()
        if provider == "openai":
            return settings.OPENAI_EMBEDDING_DIMENSION
        elif provider == "huggingface":
            return settings.HUGGINGFACE_EMBEDDING_MODEL_DIMENSION
        elif provider == "ollama":
            return settings.OLLAMA_EMBEDDING_MODEL_DIMENSION
        else:
            raise ValueError(f"Unsupported embedding provider: {provider}")

    def create_vector_search_index(
        self,
        collection_name: str = DEFAULT_CHUNK_COLLECTION_NAME,
        index_name: str = DEFAULT_VECTOR_INDEX_NAME,
        embedding_field: str = "embedding",
        filter_fields: Optional[List[str]] = None
    ) -> bool:
        """Create a vector search index on the specified collection."""
        collection = self.get_collection(collection_name)
        if collection is None:
            logger.error(f"Collection '{collection_name}' not accessible")
            return False

        try:
            existing_indexes = list(collection.list_search_indexes(name=index_name))
            if existing_indexes:
                logger.info(f"Vector search index '{index_name}' already exists")
                return True
        except OperationFailure as e:
            if "command listSearchIndexes is not supported" in str(e):
                logger.warning("Search index listing not supported, attempting creation")
            else:
                logger.warning(f"Could not check for existing index: {e}")
        except Exception as e:
            logger.warning(f"Error checking for existing index: {e}")

        try:
            dimension = self.get_effective_embedding_dimension()
            logger.info(f"Using embedding dimension {dimension} for index '{index_name}'")
        except ValueError as e:
            logger.error(f"Cannot create vector index: {e}")
            return False

        fields = [{
            "type": "vector",
            "path": embedding_field,
            "numDimensions": dimension,
            "similarity": "cosine"
        }]

        if filter_fields:
            fields.extend({
                "type": "filter",
                "path": field_path
            } for field_path in filter_fields)

        index_definition = {"fields": fields}

        try:
            search_index = SearchIndexModel(
                definition=index_definition,
                name=index_name,
                type="vectorSearch"
            )
            collection.create_search_index(model=search_index)
            logger.info(f"Vector search index '{index_name}' creation initiated")
            return True
        except OperationFailure as e:
            error_details = e.details if hasattr(e, 'details') else str(e)
            if "Index already exists" in str(e):
                logger.info(f"Index '{index_name}' already exists")
                return True
            if "command createSearchIndexes is not supported" in str(e):
                logger.warning("Search index creation not supported")
                return False
            logger.error(f"Failed to create vector search index: {error_details}")
            return False
        except Exception as e:
            logger.error(f"Error creating vector search index: {e}")
            return False

    def create_text_search_index(
        self,
        collection_name: str = DEFAULT_CHUNK_COLLECTION_NAME,
        index_name: str = DEFAULT_TEXT_INDEX_NAME,
        text_field: str = "text_chunk",
        additional_text_fields: Optional[Dict[str, str]] = None
    ) -> bool:
        """Create a text search index on the specified collection."""
        collection = self.get_collection(collection_name)
        if collection is None:
            logger.error(f"Collection '{collection_name}' not accessible")
            return False

        try:
            existing_indexes = list(collection.list_search_indexes(name=index_name))
            if existing_indexes:
                logger.info(f"Text search index '{index_name}' already exists")
                return True
        except OperationFailure as e:
            if "command listSearchIndexes is not supported" in str(e):
                logger.warning("Search index listing not supported")
            else:
                logger.warning(f"Could not check for existing index: {e}")

        fields = {
            text_field: {"type": "string", "analyzer": "lucene.standard"}
        }

        if additional_text_fields:
            for field_path, field_type in additional_text_fields.items():
                fields[field_path] = {
                    "type": field_type,
                    "analyzer": "lucene.standard" if field_type == "string" else None
                }

        index_definition = {
            "mappings": {
                "dynamic": False,
                "fields": fields
            }
        }

        try:
            search_index = SearchIndexModel(
                definition=index_definition,
                name=index_name
            )
            collection.create_search_index(model=search_index)
            logger.info(f"Text search index '{index_name}' creation initiated")
            return True
        except OperationFailure as e:
            error_details = e.details if hasattr(e, 'details') else str(e)
            if "Index already exists" in str(e):
                logger.info(f"Text search index '{index_name}' already exists")
                return True
            if "command createSearchIndexes is not supported" in str(e):
                logger.warning("Text search index creation not supported")
                return False
            logger.error(f"Failed to create text search index: {error_details}")
            return False
        except Exception as e:
            logger.error(f"Error creating text search index: {e}")
            return False

    def insert_chunks_with_embeddings(
        self,
        chunks: List[Dict[str, Any]],
        collection_name: str = DEFAULT_CHUNK_COLLECTION_NAME,
        batch_size: int = 1000
    ) -> Dict[str, Union[int, List[Any]]]:
        """Insert document chunks with their embeddings into MongoDB."""
        collection = self.get_collection(collection_name)
        if collection is None:
            logger.error(f"Collection '{collection_name}' not accessible")
            return {"inserted_count": 0, "duplicate_count": 0, "errors": ["Collection not accessible"]}

        results = {
            "inserted_count": 0,
            "duplicate_count": 0,
            "errors": []
        }

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            try:
                result = collection.insert_many(batch, ordered=False)
                results["inserted_count"] += len(result.inserted_ids)
            except BulkWriteError as e:
                results["inserted_count"] += len(e.details["nInserted"])
                results["duplicate_count"] += len(e.details["writeErrors"])
                for error in e.details["writeErrors"]:
                    if "duplicate key error" not in str(error).lower():
                        results["errors"].append(str(error))
            except Exception as e:
                results["errors"].append(str(e))
                logger.error(f"Error inserting batch: {e}")

        return results

    def perform_vector_search(
        self,
        query_embedding: List[float],
        collection_name: str = DEFAULT_CHUNK_COLLECTION_NAME,
        index_name: str = DEFAULT_VECTOR_INDEX_NAME,
        num_candidates: int = 150,
        limit: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Perform vector similarity search using the specified index."""
        collection = self.get_collection(collection_name)
        if collection is None:
            logger.error(f"Collection '{collection_name}' not accessible")
            return []

        try:
            pipeline = [
                {
                    "$search": {
                        "index": index_name,
                        "knnBeta": {
                            "vector": query_embedding,
                            "path": "embedding",
                            "k": num_candidates,
                            "filter": filter_dict
                        }
                    }
                },
                {"$limit": limit},
                {"$project": {"_id": 0, "score": {"$meta": "searchScore"}}}
            ]

            results = list(collection.aggregate(pipeline))
            logger.info(f"Found {len(results)} results for vector search")
            return results

        except Exception as e:
            logger.error(f"Error performing vector search: {e}")
            return []

if __name__ == "__main__":
    from config.logging_config import setup_logging
    setup_logging(level="INFO")

    logger.info("Testing MongoDB Manager")
    try:
        manager = MongoDBManager()
        manager.connect()

        # Test vector search index creation
        logger.info("Testing vector search index creation")
        success = manager.create_vector_search_index()
        logger.info(f"Vector search index creation {'succeeded' if success else 'failed'}")

        # Test text search index creation
        logger.info("Testing text search index creation")
        success = manager.create_text_search_index()
        logger.info(f"Text search index creation {'succeeded' if success else 'failed'}")

        manager.close()
        logger.info("Test complete")

    except Exception as e:
        logger.error(f"Test failed: {e}")