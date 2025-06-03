"""
ArXiv Downloader Module

This module provides functionality for searching and downloading papers from ArXiv.
It handles the search process, metadata extraction, and PDF downloads with rate limiting
to respect ArXiv's API constraints.

Key features:
- ArXiv paper search with customizable parameters
- Metadata extraction and storage
- PDF download with rate limiting
- Robust error handling and logging
"""

import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import json

import arxiv

from config.settings import settings

# Configure logger for this module
logger = logging.getLogger(__name__)

# Les constantes globales PDF_OUTPUT_DIR et METADATA_OUTPUT_DIR sont supprimées ici.
# Les chemins complets seront maintenant déterminés par l'appelant (run_ingestion.py)
# ou par les fonctions si elles sont appelées directement sans surcharge (moins idéal pour la flexibilité).

def search_arxiv_papers(
    query: str = settings.ARXIV_DEFAULT_QUERY,
    max_results: int = settings.ARXIV_MAX_RESULTS,
    sort_by: str = settings.ARXIV_SORT_BY,
    sort_order: str = settings.ARXIV_SORT_ORDER
) -> List[arxiv.Result]:
    """
    Search for papers on ArXiv using the provided query and parameters.
    
    Args:
        query: Search query string
        max_results: Maximum number of results to return
        sort_by: Field to sort results by (e.g., 'relevance', 'lastUpdatedDate')
        sort_order: Sort order ('ascending' or 'descending')
        
    Returns:
        List of ArXiv search results
        
    Raises:
        arxiv.ArxivError: If there's an error with the ArXiv API
    """
    logger.info(f"Searching ArXiv with query: {query}")
    logger.debug(f"Search parameters: max_results={max_results}, sort_by={sort_by}, sort_order={sort_order}")
    
    try:
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=getattr(arxiv.SortCriterion, sort_by),
            sort_order=getattr(arxiv.SortOrder, sort_order)
        )
        
        results = list(search.results())
        logger.info(f"Found {len(results)} results for query: {query}")
        return results
        
    except Exception as e:
        logger.error(f"Error searching ArXiv: {e}", exc_info=True)
        raise

def download_paper(
    result: arxiv.Result,
    pdf_dir: Path,
    metadata_dir: Path
) -> bool:
    """
    Download a single paper's PDF and metadata.
    
    Args:
        result: ArXiv search result
        pdf_dir: Directory to save PDF files
        metadata_dir: Directory to save metadata files
        
    Returns:
        True if download was successful, False otherwise
    """
    arxiv_id = result.entry_id.split('/')[-1]
    pdf_filename = f"{arxiv_id}.pdf"
    pdf_path = pdf_dir / pdf_filename
    metadata_path = metadata_dir / f"{arxiv_id}_metadata.json"
    
    # Skip if both files already exist
    if pdf_path.exists() and metadata_path.exists():
        logger.info(f"Paper {arxiv_id} already downloaded, skipping")
        return True
        
    try:
        # Download PDF
        if not pdf_path.exists():
            logger.info(f"Downloading PDF for {arxiv_id}")
            result.download_pdf(dirpath=str(pdf_dir), filename=pdf_filename)
            logger.debug(f"PDF downloaded to {pdf_path}")

            # Check if the downloaded PDF is valid (exists and is not empty)
            if not pdf_path.exists() or pdf_path.stat().st_size == 0:
                # Log error and raise an exception to be caught by the generic handler below
                # This ensures cleanup of metadata and consistent error reporting
                logger.error(f"PDF for {arxiv_id} is missing or empty after download attempt.")
                raise IOError(f"PDF for {arxiv_id} is missing or empty after download attempt.")
            
        # Save metadata
        if not metadata_path.exists():
            logger.info(f"Saving metadata for {arxiv_id}")
            metadata = {
                "entry_id": result.entry_id,
                "title": result.title,
                "authors": [author.name for author in result.authors],
                "published": result.published.isoformat(),
                "summary": result.summary,
                "pdf_url": result.pdf_url,
                "primary_category": result.primary_category,
                "categories": result.categories,
                "comment": result.comment,
                "journal_ref": result.journal_ref,
                "doi": result.doi,
                "links": [link.href for link in result.links]
            }
            
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            logger.debug(f"Metadata saved to {metadata_path}")
            
        return True
        
    except Exception as e:
        logger.error(f"Error downloading paper {arxiv_id}: {e}", exc_info=True)
        # Clean up partial downloads
        if pdf_path.exists():
            pdf_path.unlink()
        if metadata_path.exists():
            metadata_path.unlink()
        return False

def download_papers(
    results: List[arxiv.Result],
    pdf_dir: Path,
    metadata_dir: Path,
    delay: float = float(settings.ARXIV_DOWNLOAD_DELAY_SECONDS)
) -> Dict[str, int]:
    """
    Download multiple papers with rate limiting.
    
    Args:
        results: List of ArXiv search results
        pdf_dir: Directory to save PDF files
        metadata_dir: Directory to save metadata files
        delay: Delay between downloads in seconds
        
    Returns:
        Dictionary with download statistics
    """
    stats = {
        "total": len(results),
        "successful": 0,
        "failed": 0,
        "skipped": 0
    }
    
    # Create output directories
    pdf_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    
    for i, result in enumerate(results, 1):
        arxiv_id = result.entry_id.split('/')[-1]
        logger.info(f"Processing paper {i}/{len(results)}: {arxiv_id}")
        
        # Check if already downloaded
        if (pdf_dir / f"{arxiv_id}.pdf").exists() and \
           (metadata_dir / f"{arxiv_id}_metadata.json").exists():
            logger.info(f"Paper {arxiv_id} already downloaded, skipping")
            stats["skipped"] += 1
            continue
            
        # Download with rate limiting
        if download_paper(result, pdf_dir, metadata_dir):
            stats["successful"] += 1
        else:
            stats["failed"] += 1
            
        if i < len(results):
            logger.debug(f"Waiting {delay:.1f} seconds before next download")
            time.sleep(delay)
            
    logger.info("Download statistics:")
    logger.info(f"  Total papers: {stats['total']}")
    logger.info(f"  Successfully downloaded: {stats['successful']}")
    logger.info(f"  Failed downloads: {stats['failed']}")
    logger.info(f"  Skipped (already downloaded): {stats['skipped']}")
    
    return stats

def _create_test_directories() -> tuple[Path, Path]:
    """
    Create test directories for paper downloads.
    
    Returns:
        Tuple of (PDF directory path, metadata directory path)
    """
    test_base_data_dir = Path(settings.DATA_DIR) / "corpus" / "test_downloader_corpus"
    test_pdf_output_dir = test_base_data_dir / "pdfs"
    test_metadata_output_dir = test_base_data_dir / "metadata"
    
    test_pdf_output_dir.mkdir(parents=True, exist_ok=True)
    test_metadata_output_dir.mkdir(parents=True, exist_ok=True)
    
    return test_pdf_output_dir, test_metadata_output_dir

if __name__ == "__main__":
    from config.logging_config import setup_logging
    setup_logging(level="INFO")
    
    logger.info("Starting ArXiv downloader test run")
    
    # Create test directories
    test_pdf_output_dir, test_metadata_output_dir = _create_test_directories()
    
    # Search for papers
    try:
        results = search_arxiv_papers(
            query=settings.ARXIV_DEFAULT_QUERY,
            max_results=settings.ARXIV_MAX_RESULTS
        )
        
        # Download papers
        stats = download_papers(
            results=results,
            pdf_dir=test_pdf_output_dir,
            metadata_dir=test_metadata_output_dir
        )
        
    except Exception as e:
        logger.error(f"Test run failed: {e}", exc_info=True)
    
    logger.info("ArXiv downloader test run finished")