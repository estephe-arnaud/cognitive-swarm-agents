# makers/scripts/run_ingestion.py
import argparse
import logging
import re
from pathlib import Path

from config.settings import settings
from config.logging_config import setup_logging
from src.data_processing.arxiv_downloader import (
    search_arxiv_papers, 
    download_papers as execute_arxiv_downloads
)
from src.data_processing.document_parser import parse_document_collection
from src.data_processing.preprocessor import preprocess_parsed_documents
from src.data_processing.embedder import generate_embeddings_for_chunks
from src.vector_store.mongodb_manager import MongoDBManager
from pymongo.errors import ConnectionFailure

logger = logging.getLogger(__name__)

def sanitize_query_for_directory_name(query: str) -> str:
    """Convert query to a valid directory name."""
    if not query:
        return "default_corpus"
    name = query.lower()
    name = re.sub(r'[\s\W-]+', '_', name)
    name = name.strip('_')
    return name[:50]

def setup_corpus_directories(corpus_name: str) -> tuple[Path, Path]:
    """Create and return paths for PDF and metadata directories."""
    corpus_path = Path(settings.DATA_DIR) / "corpus" / corpus_name
    pdf_path = corpus_path / "pdfs"
    metadata_path = corpus_path / "metadata"

    pdf_path.mkdir(parents=True, exist_ok=True)
    metadata_path.mkdir(parents=True, exist_ok=True)

    return pdf_path, metadata_path

def download_papers(args, pdf_path: Path, metadata_path: Path) -> dict:
    """Download papers from ArXiv."""
    if args.skip_download:
        logger.info("Skipping ArXiv paper download")
        return {"downloaded_count": 0, "failed_count": 0, "skipped_count": 0}

    query_to_use = args.arxiv_keywords or args.query
    logger.info(f"Searching ArXiv with query: '{query_to_use}'")

    try:
        # Step 1: Search for papers
        search_results = search_arxiv_papers(
            query=query_to_use,
            max_results=args.max_results,
            sort_by=args.sort_by,
            sort_order=args.sort_order
        )

        if not search_results:
            logger.info("No papers found on ArXiv for the given query.")
            return {"downloaded_count": 0, "failed_count": 0, "skipped_count": 0}

        logger.info(f"Found {len(search_results)} paper(s) on ArXiv. Starting download process...")

        # Step 2: Download the found papers
        # The execute_arxiv_downloads (originally download_papers from arxiv_downloader.py)
        # takes the list of results and the output directories.
        # It also has an internal delay parameter it uses from settings.
        download_stats = execute_arxiv_downloads(
            results=search_results,
            pdf_dir=pdf_path,
            metadata_dir=metadata_path
            # delay is handled internally by execute_arxiv_downloads using settings.ARXIV_DOWNLOAD_DELAY_SECONDS
        )
        
        logger.info(f"ArXiv download process completed. Stats: Successfully downloaded: {download_stats['successful']}, Failed: {download_stats['failed']}, Skipped: {download_stats['skipped']}")
        
        # The original function expected a dict with "pdfs" and "metadata" keys listing files.
        # For now, we'll return the stats. Subsequent functions (parse_document_collection)
        # operate on the directories, so they don't strictly need the lists of files if downloads were successful.
        # If a more detailed list of downloaded files is needed later, we can adjust.
        return {
            "downloaded_count": download_stats['successful'], 
            "failed_count": download_stats['failed'],
            "skipped_count": download_stats['skipped']
        }

    except Exception as e:
        logger.error(f"Paper download and processing failed: {e}", exc_info=True)
        raise # Re-throw the exception to stop the script if downloading fails

def process_documents(pdf_path: Path, metadata_path: Path) -> list:
    """Parse and preprocess documents."""
    logger.info("Parsing documents...")
    try:
        documents = parse_document_collection(pdf_path, metadata_path)
        if not documents:
            logger.warning("No documents parsed")
            return []

        logger.info(f"Parsed {len(documents)} documents")
        logger.info("Preprocessing documents...")
        
        chunks = preprocess_parsed_documents(documents)
        if not chunks:
            logger.warning("No chunks generated")
            return []

        logger.info(f"Generated {len(chunks)} chunks")
        return chunks
    except Exception as e:
        logger.error(f"Processing failed: {e}", exc_info=True)
        raise

def generate_embeddings(chunks: list) -> list:
    """Generate embeddings for document chunks."""
    if not chunks:
        logger.info("No chunks to embed")
        return []

    logger.info("Generating embeddings...")
    try:
        embedded_chunks = generate_embeddings_for_chunks(chunks)
        logger.info(f"Generated embeddings for {len(embedded_chunks)} chunks")
        return embedded_chunks
    except Exception as e:
        logger.error(f"Embedding generation failed: {e}", exc_info=True)
        raise

def setup_mongodb(args, chunks: list) -> None:
    """Set up MongoDB with chunks and indexes."""
    if not chunks:
        logger.info("No chunks to store in MongoDB")
        return

    logger.info("Setting up MongoDB...")
    mongo_mgr = None
    try:
        mongo_mgr = MongoDBManager(
            mongo_uri=settings.MONGODB_URI,
            db_name=settings.MONGO_DATABASE_NAME
        )
        mongo_mgr.connect()

        # Insert chunks
        logger.info(f"Inserting {len(chunks)} chunks into {args.collection_name}")
        insert_summary = mongo_mgr.insert_chunks_with_embeddings(
            chunks,
            collection_name=args.collection_name
        )
        logger.info(f"Insertion summary: {insert_summary}")

        # Create vector index
        vector_fields = [
            "metadata.arxiv_id",
            "metadata.original_document_title",
            "metadata.primary_category"
        ]
        mongo_mgr.create_vector_search_index(
            collection_name=args.collection_name,
            index_name=args.vector_index_name,
            embedding_field="embedding",
            filter_fields=vector_fields
        )

        # Create text index
        text_fields = {
            "metadata.original_document_title": "string",
            "metadata.summary": "string"
        }
        mongo_mgr.create_text_search_index(
            collection_name=args.collection_name,
            index_name=args.text_index_name,
            text_field="text_chunk",
            additional_text_fields=text_fields
        )
        logger.info("MongoDB setup complete")

    except ConnectionFailure:
        logger.error("MongoDB connection failed", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"MongoDB setup failed: {e}", exc_info=True)
        raise
    finally:
        if mongo_mgr:
            mongo_mgr.close()

def main():
    parser = argparse.ArgumentParser(description="MAKERS: Data Ingestion Pipeline")
    parser.add_argument(
        "--query",
        type=str,
        default=settings.ARXIV_DEFAULT_QUERY,
        help="Natural language query for corpus naming and ArXiv search"
    )
    parser.add_argument(
        "--arxiv_keywords",
        type=str,
        help="Optimized keywords for ArXiv search (e.g., English, using AND/OR)"
    )
    parser.add_argument(
        "--max_results",
        type=int,
        default=settings.ARXIV_MAX_RESULTS,
        help=f"Maximum papers to download (default: {settings.ARXIV_MAX_RESULTS})"
    )
    parser.add_argument(
        "--sort_by",
        type=str,
        default=settings.ARXIV_SORT_BY,
        choices=["Relevance", "LastUpdatedDate", "SubmittedDate"],
        help=f"Sort criterion (default: {settings.ARXIV_SORT_BY})"
    )
    parser.add_argument(
        "--sort_order",
        type=str,
        default=settings.ARXIV_SORT_ORDER,
        choices=["Ascending", "Descending"],
        help=f"Sort order (default: {settings.ARXIV_SORT_ORDER})"
    )
    parser.add_argument(
        "--corpus_name",
        type=str,
        help="Specific corpus name (default: derived from query)"
    )
    parser.add_argument(
        "--collection_name",
        type=str,
        default=MongoDBManager.DEFAULT_CHUNK_COLLECTION_NAME,
        help=f"MongoDB collection name (default: {MongoDBManager.DEFAULT_CHUNK_COLLECTION_NAME})"
    )
    parser.add_argument(
        "--vector_index_name",
        type=str,
        default=MongoDBManager.DEFAULT_VECTOR_INDEX_NAME,
        help=f"Vector index name (default: {MongoDBManager.DEFAULT_VECTOR_INDEX_NAME})"
    )
    parser.add_argument(
        "--text_index_name",
        type=str,
        default=MongoDBManager.DEFAULT_TEXT_INDEX_NAME,
        help=f"Text index name (default: {MongoDBManager.DEFAULT_TEXT_INDEX_NAME})"
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default: INFO)"
    )
    parser.add_argument(
        "--skip_download",
        action="store_true",
        help="Skip ArXiv download and use existing PDFs"
    )

    args = parser.parse_args()
    setup_logging(level=args.log_level.upper())
    logger.info("Starting ingestion pipeline")

    try:
        # Setup directories
        corpus_name = sanitize_query_for_directory_name(args.corpus_name or args.query)
        pdf_path, metadata_path = setup_corpus_directories(corpus_name)
        logger.info(f"Using corpus: {corpus_name}")

        # Download papers
        download_results = download_papers(args, pdf_path, metadata_path)

        # Process documents
        chunks = process_documents(pdf_path, metadata_path)

        # Generate embeddings
        embedded_chunks = generate_embeddings(chunks)

        # Setup MongoDB
        setup_mongodb(args, embedded_chunks)

        logger.info("Pipeline completed successfully")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()