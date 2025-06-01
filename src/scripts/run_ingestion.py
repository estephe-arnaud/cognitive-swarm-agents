# cognitive-swarm-agents/scripts/run_ingestion.py
import argparse
import logging
import asyncio # Not strictly needed here if all called functions are sync, but good for consistency if some become async

from config.settings import settings
from config.logging_config import setup_logging

# Importer les modules de traitement et de stockage
from src.data_processing.arxiv_downloader import download_pipeline as download_arxiv_papers
from src.data_processing.document_parser import parse_document_collection
from src.data_processing.preprocessor import preprocess_parsed_documents
from src.data_processing.embedder import generate_embeddings_for_chunks
from src.vector_store.mongodb_manager import MongoDBManager

# <<< CORRECTION : Importation de ConnectionFailure >>>
from pymongo.errors import ConnectionFailure


# Configurer le logger pour ce script
# setup_logging() # Appelé dans main() pour permettre le contrôle du niveau via args
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Cognitive Swarm: Data Ingestion Pipeline for ArXiv papers.")
    parser.add_argument(
        "--query",
        type=str,
        default=settings.ARXIV_DEFAULT_QUERY,
        help=f"ArXiv search query (default: '{settings.ARXIV_DEFAULT_QUERY}')."
    )
    parser.add_argument(
        "--max_results",
        type=int,
        default=settings.ARXIV_MAX_RESULTS,
        help=f"Maximum number of papers to download from ArXiv (default: {settings.ARXIV_MAX_RESULTS})."
    )
    parser.add_argument(
        "--sort_by",
        type=str,
        default=settings.ARXIV_SORT_BY,
        choices=["relevance", "lastUpdatedDate", "submittedDate"],
        help=f"ArXiv sort criterion (default: '{settings.ARXIV_SORT_BY}')."
    )
    parser.add_argument(
        "--sort_order",
        type=str,
        default=settings.ARXIV_SORT_ORDER,
        choices=["ascending", "descending"],
        help=f"ArXiv sort order (default: '{settings.ARXIV_SORT_ORDER}')."
    )
    parser.add_argument(
        "--collection_name",
        type=str,
        default=MongoDBManager.DEFAULT_CHUNK_COLLECTION_NAME,
        help=f"MongoDB collection name for chunks (default: '{MongoDBManager.DEFAULT_CHUNK_COLLECTION_NAME}')."
    )
    parser.add_argument(
        "--vector_index_name",
        type=str,
        default=MongoDBManager.DEFAULT_VECTOR_INDEX_NAME,
        help=f"MongoDB vector index name (default: '{MongoDBManager.DEFAULT_VECTOR_INDEX_NAME}')."
    )
    parser.add_argument(
        "--text_index_name",
        type=str,
        default=MongoDBManager.DEFAULT_TEXT_INDEX_NAME,
        help=f"MongoDB text index name (default: '{MongoDBManager.DEFAULT_TEXT_INDEX_NAME}')."
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level (default: INFO)."
    )
    parser.add_argument(
        "--skip_download",
        action="store_true",
        help="Skip ArXiv download and use existing PDFs in data/corpus/rl_robotics_arxiv/pdfs/."
    )
    parser.add_argument(
        "--force_create_indexes", # Note: This flag is informational; methods check existence
        action="store_true",
        help="Attempt to create MongoDB indexes (methods already check for existence)."
    )


    args = parser.parse_args()

    setup_logging(level=args.log_level.upper())

    logger.info("Starting Data Ingestion Pipeline...")
    logger.info(f"Parameters: query='{args.query}', max_results={args.max_results}, collection='{args.collection_name}'")

    if not args.skip_download:
        logger.info("Step 1: Downloading ArXiv papers...")
        try:
            download_results = download_arxiv_papers(
                query=args.query,
                max_results=args.max_results,
                sort_by=args.sort_by,
                sort_order=args.sort_order
            )
            logger.info(f"ArXiv download complete. PDFs: {len(download_results['pdfs'])}, Metadata: {len(download_results['metadata'])}")
            if not download_results['pdfs']:
                logger.warning("No PDFs were downloaded. Subsequent steps might not process any data if corpus is empty.")
        except Exception as e:
            logger.error(f"Error during ArXiv download: {e}", exc_info=True)
            return 
    else:
        logger.info("Step 1: Skipped ArXiv paper download as per --skip_download flag.")

    logger.info("\nStep 2: Parsing downloaded documents...")
    try:
        parsed_documents = parse_document_collection() 
        if not parsed_documents:
            logger.warning("No documents were parsed. Pipeline might not have data to process further.")
        logger.info(f"Successfully parsed {len(parsed_documents)} documents.")
    except Exception as e:
        logger.error(f"Error during document parsing: {e}", exc_info=True)
        return

    logger.info("\nStep 3: Preprocessing documents (cleaning and chunking)...")
    try:
        processed_chunks = preprocess_parsed_documents(parsed_documents)
        if not processed_chunks:
            logger.warning("No chunks were generated after preprocessing.")
        logger.info(f"Successfully preprocessed documents into {len(processed_chunks)} chunks.")
    except Exception as e:
        logger.error(f"Error during preprocessing: {e}", exc_info=True)
        return

    if processed_chunks:
        logger.info("\nStep 4: Generating embeddings for chunks...")
        try:
            chunks_with_embeddings = generate_embeddings_for_chunks(processed_chunks)
            if not chunks_with_embeddings:
                logger.warning("No embeddings were generated.")
            logger.info(f"Successfully generated embeddings for {len(chunks_with_embeddings)} chunks.")
        except Exception as e:
            logger.error(f"Error during embedding generation: {e}", exc_info=True)
            return
    else:
        logger.info("\nStep 4: Skipped embedding generation as there are no processed chunks.")
        chunks_with_embeddings = []

    logger.info("\nStep 5: Managing MongoDB (inserting data and creating indexes)...")
    mongo_mgr = None 
    try:
        mongo_mgr = MongoDBManager(mongo_uri=settings.MONGO_URI, db_name=settings.MONGO_DATABASE_NAME)
        mongo_mgr.connect() 

        if chunks_with_embeddings:
            logger.info(f"Inserting {len(chunks_with_embeddings)} chunks into collection '{args.collection_name}'...")
            insert_summary = mongo_mgr.insert_chunks_with_embeddings(
                chunks_with_embeddings,
                collection_name=args.collection_name
            )
            logger.info(f"MongoDB insertion summary: {insert_summary}")
        else:
            logger.info("No chunks with embeddings to insert into MongoDB.")

        logger.info(f"Ensuring vector search index '{args.vector_index_name}' exists/is created...")
        vector_filter_fields = [
            "arxiv_id", 
            "original_document_title", 
            "metadata.primary_category" 
        ]
        mongo_mgr.create_vector_search_index(
            collection_name=args.collection_name,
            index_name=args.vector_index_name,
            embedding_field="embedding", 
            filter_fields=vector_filter_fields
        )

        logger.info(f"Ensuring text search index '{args.text_index_name}' exists/is created...")
        mongo_mgr.create_text_search_index(
            collection_name=args.collection_name,
            index_name=args.text_index_name,
            text_field="text_chunk", 
            additional_text_fields={ 
                "original_document_title": "string",
                "metadata.title": "string", 
                "metadata.summary": "string" 
            }
        )
        logger.info("MongoDB index management complete.")

    except ConnectionFailure: # <<< ConnectionFailure est maintenant défini via l'import
        logger.error("Failed to connect to MongoDB. Aborting MongoDB operations.", exc_info=True)
    except Exception as e:
        logger.error(f"Error during MongoDB operations: {e}", exc_info=True)
    finally:
        if mongo_mgr:
            mongo_mgr.close()

    logger.info("\nData Ingestion Pipeline Finished Successfully!")

if __name__ == "__main__":
    main()