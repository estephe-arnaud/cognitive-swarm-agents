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
    DEFAULT_CHUNK_COLLECTION_NAME = "arxiv_chunks"
    DEFAULT_VECTOR_INDEX_NAME = "default_vector_search_index" # Nom suggéré pour clarté
    DEFAULT_TEXT_INDEX_NAME = "default_text_index"

    def __init__(self, mongo_uri: str = settings.MONGODB_URI, db_name: str = settings.MONGO_DATABASE_NAME):
        self.mongo_uri = mongo_uri
        self.db_name = db_name
        self.client: Optional[MongoClient] = None
        self.db: Optional[Database] = None
        logger.info(f"MongoDBManager initialized for database: {self.db_name}")

    def connect(self) -> None:
        if self.client is not None and self.db is not None:
            try:
                self.client.admin.command('ping')
                logger.debug("Already connected to MongoDB.")
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
                appName="MAKERSClient"
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
        if self.client is None or self.db is None:
            logger.warning("MongoDB client not connected. Attempting to connect.")
            try:
                self.connect()
            except Exception:
                logger.error("Failed to establish connection in get_collection.")
                return None

        if self.db is not None:
            return self.db[collection_name]
        return None

    def get_effective_embedding_dimension(self) -> int:
        provider = settings.DEFAULT_EMBEDDING_PROVIDER.lower()
        if provider == "openai":
            return settings.OPENAI_EMBEDDING_DIMENSION
        elif provider == "huggingface":
            return settings.HUGGINGFACE_EMBEDDING_MODEL_DIMENSION
        elif provider == "ollama":
            return settings.OLLAMA_EMBEDDING_MODEL_DIMENSION
        else:
            logger.error(f"Unsupported embedding provider '{provider}' in settings. Cannot determine embedding dimension.")
            raise ValueError(f"Unsupported embedding provider: {provider}")

    def create_vector_search_index(
        self,
        collection_name: str = DEFAULT_CHUNK_COLLECTION_NAME,
        index_name: str = DEFAULT_VECTOR_INDEX_NAME,
        embedding_field: str = "embedding", # Champ contenant le vecteur (au premier niveau)
        filter_fields: Optional[List[str]] = None, # Liste des CHEMINS COMPLETS vers les champs filtrables, ex: "metadata.arxiv_id"
    ) -> bool:
        collection = self.get_collection(collection_name)
        if collection is None:
            logger.error(f"Cannot create vector index: Collection '{collection_name}' not accessible.")
            return False

        try:
            existing_indexes = list(collection.list_search_indexes(name=index_name))
            if existing_indexes:
                logger.info(f"Search index '{index_name}' (type vectorSearch or other) already exists on collection '{collection_name}'. Skipping creation.")
                return True
        except OperationFailure as e:
            if "command listSearchIndexes is not supported" in str(e) or \
               "Unrecognized command: listSearchIndexes" in str(e):
                logger.warning(f"Command listSearchIndexes not supported. Cannot check for existing index '{index_name}'. Will attempt creation.")
            else:
                logger.warning(f"Could not check for existing index '{index_name}' due to: {e}. Will attempt creation.")
        except Exception as e_list:
             logger.warning(f"An unexpected error occurred while checking for existing index '{index_name}': {e_list}. Will attempt creation.")

        try:
            effective_dimension = self.get_effective_embedding_dimension()
            logger.info(f"Using effective embedding dimension {effective_dimension} for index '{index_name}' based on provider '{settings.DEFAULT_EMBEDDING_PROVIDER}'.")
        except ValueError as e:
            logger.error(f"Cannot create vector index due to configuration error: {e}")
            return False

        vector_field_definition = {
            "type": "vector",
            "path": embedding_field, # Le champ "embedding" est au premier niveau du document
            "numDimensions": effective_dimension,
            "similarity": "cosine"
        }

        fields_array_for_definition = [vector_field_definition]

        if filter_fields:
            for field_path in filter_fields:
                # Pour un index de type "vectorSearch", les champs de filtre sont de type "filter"
                # field_path doit être le chemin complet, ex: "metadata.arxiv_id"
                fields_array_for_definition.append({
                    "type": "filter",
                    "path": field_path
                })
        
        index_definition_payload = {
            "fields": fields_array_for_definition
        }

        logger.info(f"Attempting to create 'vectorSearch' type index '{index_name}' on '{collection_name}' with definition: {index_definition_payload}")
        try:
            search_index_model = SearchIndexModel(
                definition=index_definition_payload,
                name=index_name,
                type="vectorSearch" 
            )
            collection.create_search_index(model=search_index_model)
            logger.info(f"'vectorSearch' type index '{index_name}' creation initiated on '{collection_name}'. It may take a few minutes to become active.")
            return True
        except OperationFailure as e:
            error_details = e.details if hasattr(e, 'details') else str(e)
            if "Index already exists" in str(e) or (isinstance(error_details, dict) and "Index already exists" in error_details.get('errmsg','')):
                logger.info(f"Index '{index_name}' already exists (confirmed by creation attempt).")
                return True
            elif isinstance(error_details, dict) and "Invalid index definition" in error_details.get('errmsg',''):
                 logger.error(f"Failed to create 'vectorSearch' index '{index_name}'. Error: Invalid index definition. Details: {error_details}", exc_info=True)
                 return False
            elif "command createSearchIndexes is not supported" in str(e) or \
                 (isinstance(error_details, dict) and "Unrecognized command: createSearchIndexes" in error_details.get('errmsg','')):
                logger.warning(f"Search index creation (type 'vectorSearch') is not supported on this MongoDB environment. Index '{index_name}' not created. Error: {e}")
                return False
            else:
                logger.error(f"Failed to create 'vectorSearch' index '{index_name}' on '{collection_name}'. Full error: {error_details}", exc_info=True)
                return False
        except Exception as e:
            logger.error(f"An unexpected error occurred during 'vectorSearch' index creation for '{index_name}': {e}", exc_info=True)
            return False

    def create_text_search_index(
        self,
        collection_name: str = DEFAULT_CHUNK_COLLECTION_NAME,
        index_name: str = DEFAULT_TEXT_INDEX_NAME,
        text_field: str = "text_chunk", # Champ textuel principal (au premier niveau)
        additional_text_fields: Optional[Dict[str, str]] = None # Chemins complets, ex: {"metadata.original_document_title": "string"}
    ) -> bool:
        collection = self.get_collection(collection_name)
        if collection is None:
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

        # Pour un index "search" standard, la définition utilise "mappings"
        fields_for_mappings: Dict[str, Any] = {
            # Le champ text_field (ex: "text_chunk") est au premier niveau du document
            text_field: {"type": "string", "analyzer": "lucene.standard"}
        }
        if additional_text_fields:
            for field_path, field_type in additional_text_fields.items():
                 # field_path ici est le chemin complet, ex: "metadata.original_document_title"
                if field_type == "string":
                     fields_for_mappings[field_path] = {"type": "string", "analyzer": "lucene.standard"}
                else: # Pour types comme "stringFacet", "autocomplete", etc.
                     fields_for_mappings[field_path] = {"type": field_type}


        index_definition_payload = {
            "mappings": {
                "dynamic": False, # Ou True si vous voulez une indexation dynamique des autres champs
                "fields": fields_for_mappings
            }
        }
        logger.info(f"Attempting to create standard text search index '{index_name}' on '{collection_name}' with definition: {index_definition_payload}")
        try:
            # Pour un index "search" standard, le type est implicite ou peut être "search"
            search_index_model = SearchIndexModel(definition=index_definition_payload, name=index_name) # type="search" est le défaut
            collection.create_search_index(model=search_index_model)
            logger.info(f"Text search index '{index_name}' creation initiated on '{collection_name}'.")
            return True
        except OperationFailure as e:
            error_details = e.details if hasattr(e, 'details') else str(e)
            if "Index already exists" in str(e) or (isinstance(error_details, dict) and "Index already exists" in error_details.get('errmsg','')):
                logger.info(f"Text search index '{index_name}' already exists (confirmed by creation attempt).")
                return True
            if "command createSearchIndexes is not supported" in str(e) or \
               (isinstance(error_details, dict) and "Unrecognized command: createSearchIndexes" in error_details.get('errmsg','')):
                logger.warning(f"Text search index creation is not supported on this MongoDB environment. Index '{index_name}' not created. Error: {e}")
                return False
            logger.error(f"Failed to create text search index '{index_name}': {error_details}", exc_info=True)
            return False
        except Exception as e:
            logger.error(f"An unexpected error occurred during text search index creation for '{index_name}': {e}", exc_info=True)
            return False

    def insert_chunks_with_embeddings(
        self,
        chunks: List[Dict[str, Any]],
        collection_name: str = DEFAULT_CHUNK_COLLECTION_NAME,
        batch_size: int = 1000
    ) -> Dict[str, Union[int, List[Any]]]:
        collection = self.get_collection(collection_name)
        if collection is None:
            logger.error(f"Cannot insert chunks: Collection '{collection_name}' not accessible.")
            return {"inserted_count": 0, "duplicate_count": 0, "errors": ["Collection not accessible."]}

        if not chunks:
            logger.info("No chunks provided for insertion.")
            return {"inserted_count": 0, "duplicate_count": 0, "errors": []}

        logger.info(f"Attempting to insert {len(chunks)} chunks into collection '{collection_name}'.")

        documents_to_insert = []
        for chunk_data in chunks: # chunk_data a maintenant la structure {"chunk_id", "text_chunk", "embedding", "metadata":{...}}
            doc = dict(chunk_data)
            if "chunk_id" not in doc: # Le chunk_id est toujours au premier niveau pour l'_id
                logger.error(f"Chunk data missing 'chunk_id', cannot use as _id. Skipping: {str(doc)[:100]}")
                continue
            doc["_id"] = doc["chunk_id"]
            documents_to_insert.append(doc)

        total_inserted_count = 0
        total_duplicate_count = 0
        write_errors_details: List[Any] = []

        for i in range(0, len(documents_to_insert), batch_size):
            batch = documents_to_insert[i : i + batch_size]
            logger.debug(f"Inserting batch {i//batch_size + 1} of {len(batch)} documents.")
            try:
                result = collection.insert_many(batch, ordered=False)
                total_inserted_count += len(result.inserted_ids)
            except BulkWriteError as bwe:
                total_inserted_count += bwe.details.get("nInserted", 0)
                batch_duplicates = 0
                current_batch_errors = []
                for error in bwe.details.get("writeErrors", []):
                    if error.get("code") == 11000:
                        batch_duplicates += 1
                    else:
                        current_batch_errors.append(error)
                total_duplicate_count += batch_duplicates
                if batch_duplicates > 0:
                    logger.warning(f"Encountered {batch_duplicates} duplicate chunk_ids (already exist) in batch {i//batch_size + 1}.")
                if current_batch_errors:
                    logger.error(f"Other write errors in batch {i//batch_size + 1}: {len(current_batch_errors)} errors. Details: {current_batch_errors}")
                    write_errors_details.extend(current_batch_errors)
            except Exception as e:
                logger.error(f"An unexpected error occurred during batch insert: {e}", exc_info=True)
                write_errors_details.append(str(e))

        logger.info(
            f"Insertion completed for collection '{collection_name}'. "
            f"Total documents attempted: {len(documents_to_insert)}. "
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
        filter_dict: Optional[Dict[str, Any]] = None # Les clés ici doivent être les chemins complets, ex: "metadata.arxiv_id"
    ) -> List[Dict[str, Any]]:
        collection = self.get_collection(collection_name)
        if collection is None:
            logger.error(f"Cannot perform vector search: Collection '{collection_name}' not accessible.")
            return []

        if not query_embedding:
            logger.warning("Query embedding is empty. Cannot perform vector search.")
            return []

        vector_search_stage: Dict[str, Any] = {
            "$vectorSearch": {
                "index": index_name, 
                "queryVector": query_embedding,
                "path": "embedding", # Le champ embedding est au premier niveau
                "numCandidates": num_candidates,
                "limit": limit,
            }
        }

        if filter_dict:
            filter_clauses = []
            for key, value in filter_dict.items(): # key est le chemin complet, ex: "metadata.arxiv_id"
                # Pour les champs de type "filter" dans un index "vectorSearch",
                # la syntaxe pour le filtrage exact peut être "equals" ou "term" selon le type inféré.
                # "term" est souvent utilisé pour les chaînes.
                filter_clauses.append({"term": {"path": key, "query": value}})
            
            if filter_clauses:
                if len(filter_clauses) == 1:
                    vector_search_stage["$vectorSearch"]["filter"] = filter_clauses[0]
                else:
                    vector_search_stage["$vectorSearch"]["filter"] = {
                        "compound": { "must": filter_clauses } 
                    }
        # La projection doit refléter la nouvelle structure où la plupart des métadonnées sont dans "metadata"
        pipeline = [
            vector_search_stage,
            {
                "$project": {
                    "_id": 1, 
                    "text_chunk": 1, 
                    "metadata": 1, # Projeter le champ metadata complet
                    # Les champs individuels comme arxiv_id, original_document_title sont maintenant DANS metadata
                    "score": {"$meta": "vectorSearchScore"}
                }
            }
        ]

        logger.debug(f"Executing vector search pipeline on '{collection_name}' with index '{index_name}': {pipeline}")
        try:
            results = list(collection.aggregate(pipeline))
            logger.info(f"Vector search returned {len(results)} results.")
            return results
        except OperationFailure as e:
            error_details = e.details if hasattr(e, 'details') else str(e)
            if "index not found" in str(e).lower() or "no such index" in str(e).lower() or \
               (isinstance(error_details, dict) and "index not found" in error_details.get('errmsg','').lower()):
                logger.error(f"Vector search index '{index_name}' not found or not ready on collection '{collection_name}'. Please create it first. Error: {error_details}")
            else:
                logger.error(f"Error during vector search on '{collection_name}': {error_details}", exc_info=True)
            return []
        except Exception as e:
            logger.error(f"An unexpected error occurred during vector search: {e}", exc_info=True)
            return []

if __name__ == "__main__":
    from config.logging_config import setup_logging
    setup_logging(level="INFO")

    logger.info("Starting MongoDBManager test run (attempting 'vectorSearch' type index with filters)...")

    if not settings.MONGODB_URI or "localhost" in settings.MONGODB_URI:
        logger.warning(f"MONGODB_URI is '{settings.MONGODB_URI}'. Ensure MongoDB is running locally or configured for Atlas for full testing.")

    mongo_mgr = MongoDBManager()
    test_coll_name = "test_vs_type_with_filters" 
    test_vector_idx_name = "test_idx_vs_type_with_filters"
    test_text_idx_name = "test_txt_idx_vs_type_with_filters"

    try:
        mongo_mgr.connect()
        
        temp_coll = mongo_mgr.get_collection(test_coll_name)
        if temp_coll is not None:
            logger.info(f"Dropping existing test collection: {test_coll_name}")
            temp_coll.drop() 
            try:
                logger.info(f"Attempting to drop pre-existing search index '{test_vector_idx_name}' from collection '{test_coll_name}' if any.")
                temp_coll.drop_search_index(test_vector_idx_name)
            except OperationFailure: pass 
            try:
                logger.info(f"Attempting to drop pre-existing search index '{test_text_idx_name}' from collection '{test_coll_name}' if any.")
                temp_coll.drop_search_index(test_text_idx_name)
            except OperationFailure: pass

        logger.info(f"\n--- Testing 'vectorSearch' type Index Creation ({test_vector_idx_name}) with filters ---")
        
        # IMPORTANT: Ces chemins DOIVENT correspondre à la structure des documents après modification de l'embedder
        # c'est-à-dire, préfixés par "metadata."
        filter_fields_for_test = [
            "metadata.arxiv_id", 
            "metadata.original_document_title",
            "metadata.primary_category",  # Assurez-vous que primary_category est bien dans metadata
            "metadata.embedding_provider",
            "metadata.embedding_model"
        ]
       
        mongo_mgr.create_vector_search_index(
            collection_name=test_coll_name,
            index_name=test_vector_idx_name,
            filter_fields=filter_fields_for_test 
        )
        
        # Pour l'index textuel, les chemins doivent aussi refléter la structure
        additional_text_fields_for_test = {
            "metadata.original_document_title": "string", 
            "metadata.title": "string" # Si vous avez un champ 'title' distinct dans metadata
        }
        mongo_mgr.create_text_search_index(
            collection_name=test_coll_name,
            index_name=test_text_idx_name,
            text_field="text_chunk", # text_chunk est au premier niveau
            additional_text_fields=additional_text_fields_for_test
        )

        logger.info("\n--- Testing Data Insertion (with nested metadata structure) ---")
        try:
            effective_dim_for_test = mongo_mgr.get_effective_embedding_dimension()
        except ValueError:
            logger.error("Cannot run insertion test: effective embedding dimension could not be determined from settings.")
            raise

        # Cet exemple de chunk doit correspondre à la sortie de votre embedder.py modifié
        sample_chunks_for_db: List[Dict[str, Any]] = [
            {
                "chunk_id": "vsfilter001_chunk_001",
                "text_chunk": "vectorSearch type with filters test chunk 1: RL and robotics.",
                "embedding": [0.1] * effective_dim_for_test,
                "metadata": {
                    "chunk_id": "vsfilter001_chunk_001",
                    "arxiv_id": "vsfilter001",
                    "original_document_title": "RL Robotics vectorSearch Filters Test",
                    "embedding_model": "test_model_ollama_vs_filters",
                    "embedding_provider": "ollama",
                    "embedding_dimension": effective_dim_for_test,
                    "primary_category": "cs.RO", 
                    "authors": ["Author X", "Author Y"], # Exemple de métadonnées supplémentaires
                    "year": 2025
                }
            },
        ]
        
        insert_summary = mongo_mgr.insert_chunks_with_embeddings(sample_chunks_for_db, collection_name=test_coll_name)
        logger.info(f"Insertion summary (vectorSearch with filters test): {insert_summary}")

        logger.info(f"\n--- Testing Vector Search ({test_vector_idx_name}) with filters ---")
        if "atlas" in settings.MONGODB_URI.lower() or "mongodb.net" in settings.MONGODB_URI.lower() :
            logger.info("Waiting 30 seconds for potential Atlas index creation...")
            time.sleep(30) 
        else:
            logger.warning("MONGODB_URI does not seem to be Atlas. Vector search test might fail if not supported.")

        dummy_query_embedding_dynamic = [0.15] * effective_dim_for_test

        # Test sans filtre
        search_results_no_filter = mongo_mgr.perform_vector_search(
            query_embedding=dummy_query_embedding_dynamic,
            collection_name=test_coll_name,
            index_name=test_vector_idx_name, 
            limit=2
        )
        logger.info(f"Vector search results (no filter): {len(search_results_no_filter)}")
        if search_results_no_filter:
             logger.info(f"  Result 1 metadata: {search_results_no_filter[0].get('metadata')}")


        # Test avec filtre sur arxiv_id (qui est maintenant dans metadata.arxiv_id)
        filter1 = {"metadata.arxiv_id": "vsfilter001"}
        search_results_filter1 = mongo_mgr.perform_vector_search(
            query_embedding=dummy_query_embedding_dynamic,
            collection_name=test_coll_name,
            index_name=test_vector_idx_name, 
            limit=2,
            filter_dict=filter1
        )
        logger.info(f"Vector search results (filter: {filter1}): {len(search_results_filter1)}")
        if search_results_filter1:
             logger.info(f"  Result 1 metadata (filter1): {search_results_filter1[0].get('metadata')}")

        # Test avec filtre sur primary_category (qui est maintenant dans metadata.primary_category)
        filter2 = {"metadata.primary_category": "cs.RO"}
        search_results_filter2 = mongo_mgr.perform_vector_search(
            query_embedding=dummy_query_embedding_dynamic,
            collection_name=test_coll_name,
            index_name=test_vector_idx_name, 
            limit=2,
            filter_dict=filter2
        )
        logger.info(f"Vector search results (filter: {filter2}): {len(search_results_filter2)}")
        if search_results_filter2:
             logger.info(f"  Result 1 metadata (filter2): {search_results_filter2[0].get('metadata')}")
            
    except ConnectionFailure:
        logger.error("Could not connect to MongoDB. Skipping further tests.")
    except Exception as e:
        logger.error(f"An error occurred during MongoDBManager test: {e}", exc_info=True)
    finally:
        if mongo_mgr:
            if mongo_mgr.db is not None:
                try:
                    logger.info(f"Attempting to drop test collection: {test_coll_name}")
                    mongo_mgr.db.drop_collection(test_coll_name)
                except Exception as e_drop:
                    logger.error(f"Error during final cleanup: {e_drop}")
            mongo_mgr.close()

    logger.info("MongoDBManager test run finished.")