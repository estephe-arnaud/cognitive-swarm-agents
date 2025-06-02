# src/data_processing/arxiv_downloader.py
import arxiv
import logging
import os # Cet import n'est plus utilisé directement, peut être enlevé si non nécessaire ailleurs
from pathlib import Path
import time
from typing import List, Dict, Any, Optional

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
    sort_order: str = settings.ARXIV_SORT_ORDER,
) -> List[arxiv.Result]:
    """
    Searches for papers on ArXiv based on a query and other criteria.
    (Contenu de la fonction inchangé)
    """
    logger.info(
        f"Searching ArXiv with query='{query}', max_results={max_results}, "
        f"sort_by='{sort_by}', sort_order='{sort_order}'"
    )

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
        page_size = 100, 
        delay_seconds = 5, 
        num_retries = 3 
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
    paper: arxiv.Result, 
    # MODIFICATION: output_dir est maintenant obligatoire ou doit être géré différemment si None
    output_dir: Path 
) -> Optional[Path]:
    """
    Downloads the PDF of a single ArXiv paper.
    Args:
        paper (arxiv.Result): The ArXiv paper object.
        output_dir (Path): The directory to save the PDF to. (RENDU OBLIGATOIRE OU GÉRÉ)
    """
    if not output_dir:
        logger.error("Output directory not provided for PDF download.")
        return None
    output_dir.mkdir(parents=True, exist_ok=True) # S'assurer que le répertoire existe

    paper_id = paper.entry_id.split("/")[-1].split("v")[0]
    filename = f"{paper_id}.pdf"
    filepath = output_dir / filename

    if filepath.exists():
        logger.info(f"PDF already exists for paper {paper_id}: {filepath}")
        return filepath

    logger.info(f"Downloading PDF for paper {paper_id} ('{paper.title[:50]}...') to {filepath}")
    try:
        paper.download_pdf(dirpath=str(output_dir), filename=filename)
        logger.info(f"Successfully downloaded {filepath}")
        time.sleep(settings.ARXIV_DOWNLOAD_DELAY_SECONDS)
        return filepath
    except Exception as e:
        logger.error(
            f"Failed to download PDF for paper {paper_id} ('{paper.title[:50]}...'): {e}",
            exc_info=True,
        )
        return None

def save_paper_metadata(
    paper: arxiv.Result, 
    # MODIFICATION: output_dir est maintenant obligatoire ou doit être géré différemment si None
    output_dir: Path 
) -> Optional[Path]:
    """
    Saves the metadata of a single ArXiv paper to a JSON file.
    Args:
        paper (arxiv.Result): The ArXiv paper object.
        output_dir (Path): The directory to save the metadata JSON to. (RENDU OBLIGATOIRE OU GÉRÉ)
    """
    if not output_dir:
        logger.error("Output directory not provided for metadata saving.")
        return None
    output_dir.mkdir(parents=True, exist_ok=True) # S'assurer que le répertoire existe

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
            import json # Importation locale au cas où
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
    query: str, # Rendu non optionnel, car nécessaire pour search_arxiv_papers
    max_results: int, # Rendu non optionnel
    pdf_output_dir: Path, # CHEMIN COMPLET OBLIGATOIRE
    metadata_output_dir: Path, # CHEMIN COMPLET OBLIGATOIRE
    sort_by: str = settings.ARXIV_SORT_BY, # Garde les valeurs par défaut de settings pour sort
    sort_order: str = settings.ARXIV_SORT_ORDER
) -> Dict[str, List[Path]]:
    """
    Full pipeline to search for papers on ArXiv, download their PDFs, and save their metadata.
    The pdf_output_dir and metadata_output_dir are now expected to be full, specific paths.
    """
    # S'assurer que les répertoires de sortie existent (au cas où ils seraient passés directement)
    pdf_output_dir.mkdir(parents=True, exist_ok=True)
    metadata_output_dir.mkdir(parents=True, exist_ok=True)

    papers_to_process = search_arxiv_papers(query, max_results, sort_by, sort_order)
    
    downloaded_pdf_paths: List[Path] = []
    saved_metadata_paths: List[Path] = []

    if not papers_to_process:
        logger.warning("No papers found or an error occurred during search. Skipping download.")
        return {"pdfs": [], "metadata": []}

    logger.info(f"Starting download and metadata saving for {len(papers_to_process)} papers.")
    for i, paper_result in enumerate(papers_to_process):
        logger.info(f"Processing paper {i+1}/{len(papers_to_process)}: {paper_result.entry_id}")
        
        # Passe les chemins complets aux fonctions internes
        pdf_path = download_paper_pdf(paper_result, pdf_output_dir)
        if pdf_path:
            downloaded_pdf_paths.append(pdf_path)
        
        metadata_path = save_paper_metadata(paper_result, metadata_output_dir)
        if metadata_path:
            saved_metadata_paths.append(metadata_path)
            
    logger.info(
        f"Finished ArXiv download pipeline. Downloaded {len(downloaded_pdf_paths)} PDFs "
        f"and saved {len(saved_metadata_paths)} metadata files."
    )
    return {"pdfs": downloaded_pdf_paths, "metadata": saved_metadata_paths}


if __name__ == "__main__":
    from config.logging_config import setup_logging
    setup_logging(level="INFO") 

    logger.info("Starting ArXiv downloader test run (with modified path handling)...")
    
    # Pour tester, on doit définir un répertoire de base pour ce test
    test_base_data_dir = Path(settings.DATA_DIR) / "corpus" / "test_downloader_corpus"
    test_pdf_dir = test_base_data_dir / "pdfs"
    test_metadata_dir = test_base_data_dir / "metadata"

    # S'assurer que ces répertoires de test sont créés pour cet exemple
    test_pdf_dir.mkdir(parents=True, exist_ok=True)
    test_metadata_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Test PDF output directory: {test_pdf_dir}")
    logger.info(f"Test Metadata output directory: {test_metadata_dir}")

    results_paths = download_pipeline(
        query="explainable artificial intelligence", 
        max_results=1, # Juste 1 pour un test rapide
        pdf_output_dir=test_pdf_dir,
        metadata_output_dir=test_metadata_dir
    )

    logger.info(f"Test run completed. PDFs downloaded: {len(results_paths['pdfs'])}")
    for path in results_paths['pdfs']:
        logger.info(f" - {path}")
    logger.info(f"Metadata files saved: {len(results_paths['metadata'])}")
    for path in results_paths['metadata']:
        logger.info(f" - {path}")