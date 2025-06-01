# cognitive-swarm-agents/src/data_processing/arxiv_downloader.py
import arxiv
import logging
import os
from pathlib import Path
import time
from typing import List, Dict, Any, Optional

from config.settings import settings

# Configure logger for this module
logger = logging.getLogger(__name__)

# Ensure the output directory for PDFs exists
PDF_OUTPUT_DIR = Path(settings.DATA_DIR) / "corpus/rl_robotics_arxiv/pdfs/"
METADATA_OUTPUT_DIR = Path(settings.DATA_DIR) / "corpus/rl_robotics_arxiv/metadata/"
PDF_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
METADATA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def search_arxiv_papers(
    query: str = settings.ARXIV_DEFAULT_QUERY,
    max_results: int = settings.ARXIV_MAX_RESULTS,
    sort_by: str = settings.ARXIV_SORT_BY,
    sort_order: str = settings.ARXIV_SORT_ORDER,
) -> List[arxiv.Result]:
    """
    Searches for papers on ArXiv based on a query and other criteria.

    Args:
        query (str): The search query.
        max_results (int): Maximum number of results to fetch.
        sort_by (str): Sorting criterion ('relevance', 'lastUpdatedDate', 'submittedDate').
        sort_order (str): Sorting order ('ascending', 'descending').

    Returns:
        List[arxiv.Result]: A list of search results from ArXiv.
    """
    logger.info(
        f"Searching ArXiv with query='{query}', max_results={max_results}, "
        f"sort_by='{sort_by}', sort_order='{sort_order}'"
    )

    # Determine sort_by criterion
    if sort_by.lower() == "relevance":
        sort_criterion = arxiv.SortCriterion.Relevance
    elif sort_by.lower() == "lastupdateddate":
        sort_criterion = arxiv.SortCriterion.LastUpdatedDate
    elif sort_by.lower() == "submitteddate":
        sort_criterion = arxiv.SortCriterion.SubmittedDate
    else:
        logger.warning(
            f"Invalid sort_by value: {sort_by}. Defaulting to Relevance."
        )
        sort_criterion = arxiv.SortCriterion.Relevance

    # Determine sort_order
    if sort_order.lower() == "ascending":
        order_criterion = arxiv.SortOrder.Ascending
    elif sort_order.lower() == "descending":
        order_criterion = arxiv.SortOrder.Descending
    else:
        logger.warning(
            f"Invalid sort_order value: {sort_order}. Defaulting to Descending."
        )
        order_criterion = arxiv.SortOrder.Descending
    
    search_client = arxiv.Client(
        page_size = 100, # Number of results per page
        delay_seconds = 5, # Delay between requests to ArXiv API
        num_retries = 3 # Number of retries for requests
    )

    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=sort_criterion,
        sort_order=order_criterion,
    )

    try:
        results = list(search_client.results(search))
        logger.info(f"Found {len(results)} papers on ArXiv.")
        return results
    except Exception as e:
        logger.error(f"Error during ArXiv search: {e}", exc_info=True)
        return []


def download_paper_pdf(
    paper: arxiv.Result, output_dir: Path = PDF_OUTPUT_DIR
) -> Optional[Path]:
    """
    Downloads the PDF of a single ArXiv paper.

    Args:
        paper (arxiv.Result): The ArXiv paper object.
        output_dir (Path): The directory to save the PDF to.

    Returns:
        Optional[Path]: Path to the downloaded PDF, or None if download failed.
    """
    # Extract a unique ID for the filename, typically the entry_id without version
    paper_id = paper.entry_id.split("/")[-1].split("v")[0]
    filename = f"{paper_id}.pdf"
    filepath = output_dir / filename

    if filepath.exists():
        logger.info(f"PDF already exists for paper {paper_id}: {filepath}")
        return filepath

    logger.info(f"Downloading PDF for paper {paper_id} ('{paper.title[:50]}...') to {filepath}")
    try:
        # The arxiv library handles the download to a specified directory and filename
        paper.download_pdf(dirpath=str(output_dir), filename=filename)
        logger.info(f"Successfully downloaded {filepath}")
        # Add a small delay to be polite to the ArXiv servers
        time.sleep(settings.ARXIV_DOWNLOAD_DELAY_SECONDS)
        return filepath
    except Exception as e:
        logger.error(
            f"Failed to download PDF for paper {paper_id} ('{paper.title[:50]}...'): {e}",
            exc_info=True,
        )
        return None

def save_paper_metadata(
    paper: arxiv.Result, output_dir: Path = METADATA_OUTPUT_DIR
) -> Optional[Path]:
    """
    Saves the metadata of a single ArXiv paper to a JSON file.

    Args:
        paper (arxiv.Result): The ArXiv paper object.
        output_dir (Path): The directory to save the metadata JSON to.

    Returns:
        Optional[Path]: Path to the saved metadata file, or None if saving failed.
    """
    paper_id = paper.entry_id.split("/")[-1].split("v")[0]
    metadata_filename = f"{paper_id}_metadata.json"
    filepath = output_dir / metadata_filename

    if filepath.exists():
        logger.info(f"Metadata file already exists for paper {paper_id}: {filepath}")
        return filepath

    logger.info(f"Saving metadata for paper {paper_id} ('{paper.title[:50]}...') to {filepath}")
    
    metadata = {
        "entry_id": paper.entry_id,
        "title": paper.title,
        "authors": [str(author) for author in paper.authors],
        "summary": paper.summary,
        "comment": paper.comment,
        "journal_ref": paper.journal_ref,
        "doi": paper.doi,
        "primary_category": paper.primary_category,
        "categories": paper.categories,
        "links": [link.href for link in paper.links],
        "pdf_url": paper.pdf_url,
        "published": paper.published.isoformat() if paper.published else None,
        "updated": paper.updated.isoformat() if paper.updated else None,
    }
    
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            import json
            json.dump(metadata, f, ensure_ascii=False, indent=4)
        logger.info(f"Successfully saved metadata {filepath}")
        return filepath
    except Exception as e:
        logger.error(
            f"Failed to save metadata for paper {paper_id} ('{paper.title[:50]}...'): {e}",
            exc_info=True,
        )
        return None


def download_pipeline(
    query: str = settings.ARXIV_DEFAULT_QUERY,
    max_results: int = settings.ARXIV_MAX_RESULTS,
    sort_by: str = settings.ARXIV_SORT_BY,
    sort_order: str = settings.ARXIV_SORT_ORDER,
    pdf_output_dir: Path = PDF_OUTPUT_DIR,
    metadata_output_dir: Path = METADATA_OUTPUT_DIR,
) -> Dict[str, List[Path]]:
    """
    Full pipeline to search for papers on ArXiv, download their PDFs, and save their metadata.

    Args:
        query (str): The search query.
        max_results (int): Maximum number of results to fetch.
        sort_by (str): Sorting criterion.
        sort_order (str): Sorting order.
        pdf_output_dir (Path): Directory to save PDFs.
        metadata_output_dir (Path): Directory to save metadata JSON files.


    Returns:
        Dict[str, List[Path]]: A dictionary containing lists of paths to downloaded PDFs
                                and saved metadata files.
                                {'pdfs': [...], 'metadata': [...]}
    """
    papers_to_process = search_arxiv_papers(query, max_results, sort_by, sort_order)
    
    downloaded_pdf_paths: List[Path] = []
    saved_metadata_paths: List[Path] = []

    if not papers_to_process:
        logger.warning("No papers found or an error occurred during search. Skipping download.")
        return {"pdfs": [], "metadata": []}

    logger.info(f"Starting download and metadata saving for {len(papers_to_process)} papers.")
    for i, paper_result in enumerate(papers_to_process):
        logger.info(f"Processing paper {i+1}/{len(papers_to_process)}: {paper_result.entry_id}")
        
        pdf_path = download_paper_pdf(paper_result, pdf_output_dir)
        if pdf_path:
            downloaded_pdf_paths.append(pdf_path)
        
        metadata_path = save_paper_metadata(paper_result, metadata_output_dir)
        if metadata_path:
            saved_metadata_paths.append(metadata_path)
        
        # Optional: add a small delay between processing each paper if experiencing issues
        # time.sleep(1) 

    logger.info(
        f"Finished ArXiv download pipeline. Downloaded {len(downloaded_pdf_paths)} PDFs "
        f"and saved {len(saved_metadata_paths)} metadata files."
    )
    return {"pdfs": downloaded_pdf_paths, "metadata": saved_metadata_paths}


if __name__ == "__main__":
    # This block is for testing the module directly.
    # It should be called from a script in the `scripts/` directory for actual use.
    from config.logging_config import setup_logging
    setup_logging(level="INFO") # Setup logging when run as a script

    logger.info("Starting ArXiv downloader test run...")
    
    # Update settings for the test run if needed, or rely on .env / defaults
    # For example, to fetch fewer results for a quick test:
    # test_max_results = 2
    # results_paths = download_pipeline(max_results=test_max_results)
    
    results_paths = download_pipeline(
        query="reinforcement learning robotics", # More specific query for robotics
        max_results=settings.ARXIV_MAX_RESULTS, # Use configured max results
    )

    logger.info(f"Test run completed. PDFs downloaded: {len(results_paths['pdfs'])}")
    for path in results_paths['pdfs']:
        logger.info(f" - {path}")
    logger.info(f"Metadata files saved: {len(results_paths['metadata'])}")
    for path in results_paths['metadata']:
        logger.info(f" - {path}")

    # Add a new setting to settings.py for DATA_DIR
    # DATA_DIR: Path = Path(__file__).resolve().parent.parent / "data"
    # And also:
    # ARXIV_DOWNLOAD_DELAY_SECONDS: int = 2 # Delay in seconds between PDF downloads

    # Add these to config/settings.py:
    # DATA_DIR: Path = Path(__file__).resolve().parent.parent / "data"
    # ARXIV_DOWNLOAD_DELAY_SECONDS: int = 3 # Delay in seconds between PDF downloads, to be polite