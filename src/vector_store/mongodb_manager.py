# cognitive-swarm-agents/src/vector_store/mongodb_manager.py
import logging
from typing import List, Dict, Any, Optional, Union
import time

from pymongo import MongoClient, TEXT
from pymongo.collection import Collection
from pymongo.database import Database
from pymongo.errors import ConnectionFailure, OperationFailure, BulkWriteError
from pymongo.operations import SearchIndexModel

from config.settings import settings
# ProcessedChunkWithEmbedding n'est pas directement utilisé ici, mais gardé pour contexte si besoin futur
# class ProcessedChunkWithEmbedding(dict): 
#     pass


logger = logging.getLogger(__name__)

class MongoDBManager:
    DEFAULT_CHUNK_COLLECTION_NAME = "arxiv_chunks"
    DEFAULT_VECTOR_INDEX_NAME = "default_vector_index"
    DEFAULT_TEXT_INDEX_NAME = "default_text_index"

    def __init__(self, mongo_uri: str = settings.MONGO_URI, db_name: str = settings.MONGO_DATABASE_NAME):
        self.mongo_uri = mongo_uri
        self.db_name = db_name
        self.client: Optional[MongoClient] = None
        self.db: Optional[Database] = None
        logger.info(f"MongoDBManager initialized for database: {self.db_name}")

    def connect(self) -> None:
        if self.client and self.db:
            try: 
                self.client.admin.command('ping')
                logger.debug("Already connected to MongoDB.") # Changed to debug for less noise
                return
            except ConnectionFailure:
                logger.warning("MongoDB connection lost. Reconnecting...")
                self.client = None
                self.db = None

        try:
            self.client = MongoClient(
                self.mongo_uri,
                maxPoolSize=settings.MONGO_MAX_POOL_SIZE,
                serverSelectionTimeoutMS=settings.MONGO_TIMEOUT_MS,
                appName="CognitiveSwarmClient"
            )
            self.client.admin.command('ping')
            self.db = self.client[self.db_name]
            logger.info(f"Successfully connected to MongoDB database: {self.db_name}")
        except ConnectionFailure as e:
            logger.error(f"Failed to connect to MongoDB at {self.mongo_uri}: {e}", exc_info=True)
            self.client = None
            self.db = None
            raise
        except Exception as e: 
            logger.error(f"An unexpected error occurred during MongoDB connection: {e}", exc_info=True)
            self.client = None
            self.db = None
            raise

    def close(self) -> None:
        if self.client:
            self.client.close()
            self.client = None
            self.db = None
            logger.info("MongoDB connection closed.")

    def get_collection(self, collection_name: str) -> Optional[Collection]:
        if not self.client or not self.db:
            logger.warning("MongoDB client not connected. Attempting to connect.")
            try:
                self.connect()
            except Exception: 
                return None
        
        if self.db: 
            return self.db[collection_name]
        return None

    def get_effective_embedding_dimension(self) -> int:
        """
        Determines the effective embedding dimension based on the configured provider.
        """
        provider = settings.DEFAULT_EMBEDDING_PROVIDER.lower()
        if provider == "openai":
            return settings.OPENAI_EMBEDDING_DIMENSION
        elif provider == "huggingface":
            return settings.HUGGINGFACE_EMBEDDING_MODEL_DIMENSION
        elif provider == "ollama":
            return settings.OLLAMA_EMBEDDING_MODEL_DIMENSION
        else:
            logger.error(f"Unsupported embedding provider '{provider}' in settings. Cannot determine embedding dimension.")
            # Fallback to a common dimension or raise error. Raising error is safer.
            raise ValueError(f"Unsupported embedding provider: {provider}")

    def create_vector_search_index(
        self,
        collection_name: str = DEFAULT_CHUNK_COLLECTION_NAME,
        index_name: str = DEFAULT_VECTOR_INDEX_NAME,
        embedding_field: str = "embedding",
        filter_fields: Optional[List[str]] = None,
    ) -> bool:
        collection = self.get_collection(collection_name)
        if not collection:
            logger.error(f"Cannot create vector index: Collection '{collection_name}' not accessible.")
            return False

        try:
            existing_indexes = list(collection.list_search_indexes(name=index_name))
            if existing_indexes:
                logger.info(f"Vector search index '{index_name}' already exists on collection '{collection_name}'. Skipping creation.")
                return True
        except OperationFailure as e:
            if "command listSearchIndexes is not supported" in str(e) or \
               "Unrecognized command: listSearchIndexes" in str(e):
                logger.warning(f"Command listSearchIndexes not supported. Cannot check for existing index '{index_name}'.")
            else:
                logger.warning(f"Could not check for existing index '{index_name}' due to: {e}")
        
        try:
            effective_dimension = self.get_effective_embedding_dimension()
            logger.info(f"Using effective embedding dimension {effective_dimension} for index '{index_name}' based on provider '{settings.DEFAULT_EMBEDDING_PROVIDER}'.")
        except ValueError as e:
            logger.error(f"Cannot create vector index due to configuration error: {e}")
            return False

        index_fields_definition = [
            {
                "type": "vector",
                "path": embedding_field,
                "numDimensions": effective_dimension, # <<< MODIFIED: Using dynamic dimension
                "similarity": "cosine",
            }
        ]
        if filter_fields:
            for field_path in filter_fields:
                index_fields_definition.append({"type": "filter", "path": field_path})

        index_definition_payload = {
            "name": index_name,
            "definition": {
                "mappings": {
                    "dynamic": True, 
                    "fields": index_fields_definition
                }
            }
        }
        
        logger.info(f"Attempting to create vector search index '{index_name}' on '{collection_name}' with definition: {index_definition_payload['definition']}")
        try:
            search_index_model = SearchIndexModel(
                definition=index_definition_payload['definition'],
                name=index_name
            )
            collection.create_search_index(model=search_index_model)
            logger.info(f"Vector search index '{index_name}' creation initiated on '{collection_name}'. It may take a few minutes to become active.")
            return True
        except OperationFailure as e:
            if "Index already exists" in str(e):
                logger.info(f"Vector search index '{index_name}' already exists (confirmed by creation attempt).")
                return True
            elif "command createSearchIndexes is not supported" in str(e) or \
                 "Unrecognized command: createSearchIndexes" in str(e):
                logger.warning(f"Vector search index creation is not supported on this MongoDB environment. Index '{index_name}' not created. Error: {e}")
                return False
            else:
                logger.error(f"Failed to create vector search index '{index_name}' on '{collection_name}': {e}", exc_info=True)
                return False
        except Exception as e:
            logger.error(f"An unexpected error occurred during vector search index creation for '{index_name}': {e}", exc_info=True)
            return False

    def create_text_search_index(
        self,
        collection_name: str = DEFAULT_CHUNK_COLLECTION_NAME,
        index_name: str = DEFAULT_TEXT_INDEX_NAME,
        text_field: str = "text_chunk",
        additional_text_fields: Optional[Dict[str, str]] = None
    ) -> bool:
        # ... (corps de la fonction inchangé, car non dépendant de la dimension des embeddings) ...
        collection = self.get_collection(collection_name)
        if not collection:
            logger.error(f"Cannot create text index: Collection '{collection_name}' not accessible.")
            return False

        try:
            existing_indexes = list(collection.list_search_indexes(name=index_name))
            if existing_indexes:
                logger.info(f"Text search index '{index_name}' already exists on collection '{collection_name}'. Skipping creation.")
                return True
        except OperationFailure as e:
            if "command listSearchIndexes is not supported" in str(e):
                logger.warning(f"Command listSearchIndexes not supported. Cannot check for existing text index '{index_name}'.")
            else:
                logger.warning(f"Could not check for existing text index '{index_name}' due to: {e}")
        
        fields_to_index = {
            text_field: {"type": "string", "analyzer": "lucene.standard"}
        }
        if additional_text_fields:
            for field, field_type in additional_text_fields.items():
                fields_to_index[field] = {"type": field_type} 

        index_definition = {
            "name": index_name,
            "definition":{
                "mappings": {
                    "dynamic": False, 
                    "fields": fields_to_index
                }
            }
        }
        logger.info(f"Attempting to create text search index '{index_name}' on '{collection_name}' with definition: {index_definition['definition']}")
        try:
            search_index_model = SearchIndexModel(definition=index_definition['definition'], name=index_name)
            collection.create_search_index(model=search_index_model)
            logger.info(f"Text search index '{index_name}' creation initiated on '{collection_name}'.")
            return True
        except OperationFailure as e:
            if "Index already exists" in str(e):
                logger.info(f"Text search index '{index_name}' already exists (confirmed by creation attempt).")
                return True
            if "command createSearchIndexes is not supported" in str(e):
                logger.warning(f"Text search index creation is not supported on this MongoDB environment. Index '{index_name}' not created. Error: {e}")
                return False
            logger.error(f"Failed to create text search index '{index_name}': {e}", exc_info=True)
            return False
        except Exception as e:
            logger.error(f"An unexpected error occurred during text search index creation for '{index_name}': {e}", exc_info=True)
            return False
        
    def insert_chunks_with_embeddings(
        self,
        chunks: List[Dict[str, Any]], # Changed from ProcessedChunkWithEmbedding to generic Dict
        collection_name: str = DEFAULT_CHUNK_COLLECTION_NAME,
        batch_size: int = 1000
    ) -> Dict[str, Union[int, List[Any]]]:
        # ... (corps de la fonction inchangé, mais le type hint pour `chunks` est plus générique) ...
        # Il est important que les chunks contiennent "chunk_id" et "embedding"
        collection = self.get_collection(collection_name)
        if not collection:
            logger.error(f"Cannot insert chunks: Collection '{collection_name}' not accessible.")
            return {"inserted_count": 0, "duplicate_count": 0, "errors": ["Collection not accessible."]}

        if not chunks:
            logger.info("No chunks provided for insertion.")
            return {"inserted_count": 0, "duplicate_count": 0, "errors": []}

        logger.info(f"Attempting to insert {len(chunks)} chunks into collection '{collection_name}'.")
        
        documents_to_insert = []
        for chunk_data in chunks:
            doc = dict(chunk_data) 
            if "chunk_id" not in doc:
                logger.error(f"Chunk data missing 'chunk_id', cannot use as _id. Skipping: {str(doc)[:100]}")
                continue # Skip this problematic chunk
            doc["_id"] = doc["chunk_id"] 
            documents_to_insert.append(doc)

        total_inserted_count = 0
        total_duplicate_count = 0 
        write_errors_details = []

        for i in range(0, len(documents_to_insert), batch_size):
            batch = documents_to_insert[i : i + batch_size]
            logger.debug(f"Inserting batch {i//batch_size + 1} of {len(batch)} documents.")
            try:
                result = collection.insert_many(batch, ordered=False) 
                total_inserted_count += len(result.inserted_ids)
            except BulkWriteError as bwe:
                total_inserted_count += bwe.details.get("nInserted", 0)
                batch_duplicates = 0
                for error in bwe.details.get("writeErrors", []):
                    if error.get("code") == 11000: 
                        batch_duplicates += 1
                    else:
                        write_errors_details.append(error) 
                total_duplicate_count += batch_duplicates
                if batch_duplicates > 0:
                    logger.warning(f"Encountered {batch_duplicates} duplicate chunk_ids (already exist) in batch {i//batch_size + 1}.")
                if write_errors_details and (len(bwe.details.get("writeErrors", [])) - batch_duplicates > 0) : 
                    logger.error(f"Other write errors in batch {i//batch_size + 1}: {len(bwe.details.get('writeErrors', [])) - batch_duplicates} errors.")
            except Exception as e:
                logger.error(f"An unexpected error occurred during batch insert: {e}", exc_info=True)
                write_errors_details.append(str(e))
        
        logger.info(
            f"Insertion completed for collection '{collection_name}'. "
            f"Total documents attempted: {len(documents_to_insert)}. " # Changed from len(chunks)
            f"Successfully inserted: {total_inserted_count}. "
            f"Duplicates skipped (based on _id=chunk_id): {total_duplicate_count}."
        )
        if write_errors_details:
            logger.error(f"Details of non-duplicate write errors: {write_errors_details}")
            
        return {
            "inserted_count": total_inserted_count,
            "duplicate_count": total_duplicate_count,
            "errors": write_errors_details
        }

    def perform_vector_search(
        self,
        query_embedding: List[float],
        collection_name: str = DEFAULT_CHUNK_COLLECTION_NAME,
        index_name: str = DEFAULT_VECTOR_INDEX_NAME,
        num_candidates: int = 150,
        limit: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        # ... (corps de la fonction inchangé, car il prend un query_embedding déjà généré) ...
        collection = self.get_collection(collection_name)
        if not collection:
            logger.error(f"Cannot perform vector search: Collection '{collection_name}' not accessible.")
            return []

        if not query_embedding:
            logger.warning("Query embedding is empty. Cannot perform vector search.")
            return []

        vector_search_stage: Dict[str, Any] = {
            "$vectorSearch": {
                "index": index_name,
                "queryVector": query_embedding,
                "path": "embedding", 
                "numCandidates": num_candidates,
                "limit": limit,
            }
        }
        
        if filter_dict:
            filter_clauses = []
            for key, value in filter_dict.items():
                filter_clauses.append({"term": {"path": key, "query": value}})  
            
            if filter_clauses:
                if len(filter_clauses) == 1:
                    vector_search_stage["$vectorSearch"]["filter"] = filter_clauses[0]
                else: 
                    vector_search_stage["$vectorSearch"]["filter"] = {
                        "compound": { "must": filter_clauses }
                    }

        pipeline = [
            vector_search_stage,
            { 
                "$project": {
                    "_id": 1, 
                    "arxiv_id": 1,
                    "text_chunk": 1,
                    "original_document_title": 1,
                    "metadata": 1, 
                    "embedding_provider": 1, # Ajouté pour voir le provider dans les résultats
                    "embedding_model": 1,    # Ajouté pour voir le modèle dans les résultats
                    "score": {"$meta": "vectorSearchScore"}
                }
            }
        ]
        
        logger.debug(f"Executing vector search pipeline on '{collection_name}': {pipeline}") # Changed to debug
        try:
            results = list(collection.aggregate(pipeline))
            logger.info(f"Vector search returned {len(results)} results.")
            return results
        except OperationFailure as e:
            if "index not found" in str(e).lower() or "no such index" in str(e).lower():
                logger.error(f"Vector search index '{index_name}' not found or not ready on collection '{collection_name}'. Please create it first. Error: {e}")
            else:
                logger.error(f"Error during vector search on '{collection_name}': {e}", exc_info=True)
            return []
        except Exception as e:
            logger.error(f"An unexpected error occurred during vector search: {e}", exc_info=True)
            return []

if __name__ == "__main__":
    from config.logging_config import setup_logging
    setup_logging(level="INFO")

    logger.info("Starting MongoDBManager test run...")

    if not settings.MONGO_URI or "localhost" in settings.MONGO_URI: 
        logger.warning(f"MONGO_URI is '{settings.MONGO_URI}'. Ensure MongoDB is running locally or configured for Atlas for full testing.")
    
    mongo_mgr = MongoDBManager()
    try:
        mongo_mgr.connect()

        logger.info("\n--- Testing Index Creation (with dynamic dimension) ---")
        # La dimension sera maintenant récupérée dynamiquement par get_effective_embedding_dimension()
        # lors de l'appel à create_vector_search_index.
        vector_filter_fields = [
            "metadata.arxiv_id", 
            "metadata.primary_category",
            "original_document_title" 
        ]
        mongo_mgr.create_vector_search_index(
            collection_name="test_arxiv_chunks_dynamic_dim", 
            filter_fields=vector_filter_fields,
            index_name="test_vector_index_dynamic"
        )
        mongo_mgr.create_text_search_index(
            collection_name="test_arxiv_chunks_dynamic_dim",
            index_name="test_text_index_dynamic",
            text_field="text_chunk",
            additional_text_fields={"original_document_title": "string"}
        )

        logger.info("\n--- Testing Data Insertion ---")
        # Utiliser la dimension effective pour créer des embeddings factices
        try:
            effective_dim_for_test = mongo_mgr.get_effective_embedding_dimension()
        except ValueError:
            logger.error("Cannot run insertion test: effective embedding dimension could not be determined from settings.")
            raise # Re-raise to stop the test if dimension is unknown

        sample_chunks_for_db: List[Dict[str, Any]] = [ # Type hint to Dict
            {
                "chunk_id": "db_dyn001_chunk_001", "arxiv_id": "db_dyn001", 
                "text_chunk": "Dynamic dim chunk 1: RL in robotics.", 
                "original_document_title": "RL Robotics Dynamic", 
                "embedding": [0.1] * effective_dim_for_test, 
                "embedding_model": "test_model", # Sera déterminé par l'embedder réel
                "embedding_provider": settings.DEFAULT_EMBEDDING_PROVIDER,
                "embedding_dimension": effective_dim_for_test,
                "metadata": {"arxiv_id": "db_dyn001", "primary_category": "cs.RO"}
            },
        ]
        test_collection_dynamic = mongo_mgr.get_collection("test_arxiv_chunks_dynamic_dim")
        if test_collection_dynamic:
            logger.info("Deleting existing documents in 'test_arxiv_chunks_dynamic_dim' for fresh test.")
            test_collection_dynamic.delete_many({})
        
        insert_summary = mongo_mgr.insert_chunks_with_embeddings(sample_chunks_for_db, collection_name="test_arxiv_chunks_dynamic_dim")
        logger.info(f"Insertion summary (dynamic dim): {insert_summary}")

        logger.info("\n--- Testing Vector Search (with dynamic dimension embedding) ---")
        # Le test de la recherche vectorielle dépend de la disponibilité de l'index.
        # Une pause est nécessaire pour laisser Atlas construire l'index.
        # Si vous exécutez sur un MongoDB local sans support Atlas Search, $vectorSearch échouera.
        if "atlas" in settings.MONGO_URI.lower() or "mongodb.net" in settings.MONGO_URI.lower() : # Heuristique pour Atlas
            logger.info("Waiting 30 seconds for potential Atlas index creation...")
            time.sleep(30) 
        else:
            logger.warning("MONGO_URI does not seem to be Atlas. Vector search test might fail if not supported.")


        dummy_query_embedding_dynamic = [0.15] * effective_dim_for_test
        
        search_results_all_dynamic = mongo_mgr.perform_vector_search(
            query_embedding=dummy_query_embedding_dynamic,
            collection_name="test_arxiv_chunks_dynamic_dim",
            index_name="test_vector_index_dynamic", # Assurez-vous que cet index a été créé avec la bonne dimension
            limit=2
        )
        logger.info(f"Vector search results (dynamic, all, limit 2): {len(search_results_all_dynamic)}")
        for res in search_results_all_dynamic:
            logger.info(f"  ID: {res['_id']}, Score: {res['score']:.4f}, Title: {res.get('original_document_title')}")

    except ConnectionFailure:
        logger.error("Could not connect to MongoDB. Skipping further tests.")
    except Exception as e:
        logger.error(f"An error occurred during MongoDBManager test: {e}", exc_info=True)
    finally:
        if mongo_mgr:
            # Clean up test collection if it was created
            if mongo_mgr.db: # Check if db was successfully initialized
                try:
                    logger.info("Attempting to drop test collection: test_arxiv_chunks_dynamic_dim")
                    mongo_mgr.db.drop_collection("test_arxiv_chunks_dynamic_dim")
                except Exception as e_drop:
                    logger.error(f"Error dropping test collection: {e_drop}")
            mongo_mgr.close()

    logger.info("MongoDBManager test run finished.")