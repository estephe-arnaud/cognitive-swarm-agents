# cognitive-swarm-agents/src/data_processing/document_parser.py
import fitz # PyMuPDF
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, TypedDict, Any

from config.settings import settings

# Configure logger for this module
logger = logging.getLogger(__name__)

# Define input directories based on settings (consistent with arxiv_downloader.py)
PDF_INPUT_DIR = Path(settings.DATA_DIR) / "corpus/rl_robotics_arxiv/pdfs/"
METADATA_INPUT_DIR = Path(settings.DATA_DIR) / "corpus/rl_robotics_arxiv/metadata/"

class ParsedDocument(TypedDict):
    """
    Represents a document after parsing, containing its ID, text content, and metadata.
    """
    arxiv_id: str
    text_content: str
    metadata: Dict
    pdf_path: str
    metadata_path: Optional[str]

def load_metadata(metadata_filepath: Path) -> Optional[Dict]:
    """
    Loads metadata from a JSON file.

    Args:
        metadata_filepath (Path): Path to the metadata JSON file.

    Returns:
        Optional[Dict]: Loaded metadata as a dictionary, or None if loading fails.
    """
    if not metadata_filepath.exists():
        logger.warning(f"Metadata file not found: {metadata_filepath}")
        return None
    try:
        with open(metadata_filepath, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        logger.debug(f"Successfully loaded metadata from {metadata_filepath}")
        return metadata
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from metadata file: {metadata_filepath}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"Failed to load metadata from {metadata_filepath}: {e}", exc_info=True)
        return None

def parse_single_pdf(pdf_filepath: Path) -> Optional[str]:
    """
    Parses a single PDF file and extracts its text content using PyMuPDF.

    Args:
        pdf_filepath (Path): Path to the PDF file.

    Returns:
        Optional[str]: Extracted text content, or None if parsing fails.
    """
    if not pdf_filepath.exists():
        logger.error(f"PDF file not found: {pdf_filepath}")
        return None

    logger.info(f"Parsing PDF: {pdf_filepath.name}")
    try:
        doc = fitz.open(pdf_filepath)
        text_content = ""
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text_content += page.get_text("text") # Extract plain text
            text_content += "\n" # Add a newline between pages
        doc.close()
        logger.debug(f"Successfully parsed PDF: {pdf_filepath.name}, extracted {len(text_content)} characters.")
        return text_content.strip()
    except Exception as e:
        logger.error(f"Error parsing PDF {pdf_filepath.name}: {e}", exc_info=True)
        return None

def parse_document_collection(
    pdf_dir: Path = PDF_INPUT_DIR,
    metadata_dir: Path = METADATA_INPUT_DIR
) -> List[ParsedDocument]:
    """
    Parses all PDF documents in the specified directory and combines
    their text content with corresponding metadata.

    Args:
        pdf_dir (Path): Directory containing PDF files.
        metadata_dir (Path): Directory containing metadata JSON files.

    Returns:
        List[ParsedDocument]: A list of ParsedDocument objects.
    """
    parsed_documents: List[ParsedDocument] = []
    if not pdf_dir.is_dir():
        logger.error(f"PDF input directory not found or is not a directory: {pdf_dir}")
        return parsed_documents
    if not metadata_dir.is_dir():
        logger.warning(f"Metadata input directory not found or is not a directory: {metadata_dir}. Proceeding without metadata.")

    pdf_files = list(pdf_dir.glob("*.pdf"))
    logger.info(f"Found {len(pdf_files)} PDF files in {pdf_dir} to parse.")

    for pdf_filepath in pdf_files:
        arxiv_id_from_filename = pdf_filepath.stem # e.g., "2301.12345"
        
        text_content = parse_single_pdf(pdf_filepath)
        if text_content is None:
            logger.warning(f"Skipping document {arxiv_id_from_filename} due to PDF parsing error.")
            continue

        # Attempt to load corresponding metadata
        metadata_filepath = metadata_dir / f"{arxiv_id_from_filename}_metadata.json"
        metadata = load_metadata(metadata_filepath)
        
        if metadata is None:
            logger.warning(f"Proceeding with {arxiv_id_from_filename} without its metadata file.")
            # Create a minimal metadata if file not found or unreadable
            # The arxiv_id from filename is the most crucial part here
            document_metadata: Dict[str, Any] = {"arxiv_id_inferred": arxiv_id_from_filename}
            metadata_path_str = None
        else:
            document_metadata = metadata
            metadata_path_str = str(metadata_filepath)


        parsed_doc: ParsedDocument = {
            "arxiv_id": document_metadata.get("entry_id", arxiv_id_from_filename).split("/")[-1].split("v")[0], # Normalize ID
            "text_content": text_content,
            "metadata": document_metadata,
            "pdf_path": str(pdf_filepath),
            "metadata_path": metadata_path_str
        }
        parsed_documents.append(parsed_doc)
        logger.info(f"Successfully processed and added document: {parsed_doc['arxiv_id']}")

    logger.info(f"Finished parsing collection. Successfully processed {len(parsed_documents)} documents.")
    return parsed_documents


if __name__ == "__main__":
    from config.logging_config import setup_logging
    setup_logging(level="INFO") # Setup logging when run as a script

    logger.info("Starting document parser test run...")

    # Ensure some PDFs and metadata files exist from the arxiv_downloader step for testing.
    # For example, run arxiv_downloader.py first if its __main__ block is configured for a test download.
    if not any(PDF_INPUT_DIR.iterdir()):
        logger.warning(f"No PDF files found in {PDF_INPUT_DIR}. Please run arxiv_downloader first or place some PDFs for testing.")
    else:
        documents = parse_document_collection()
        if documents:
            logger.info(f"Successfully parsed {len(documents)} documents.")
            for i, doc in enumerate(documents[:2]): # Log details of first 2 docs
                logger.info(f"--- Document {i+1} ---")
                logger.info(f"  ArXiv ID: {doc['arxiv_id']}")
                logger.info(f"  Title (from metadata): {doc['metadata'].get('title', 'N/A')}")
                logger.info(f"  Text Snippet: {doc['text_content'][:200].replace(chr(10), ' ')}...")
                logger.info(f"  PDF Path: {doc['pdf_path']}")
                logger.info(f"  Metadata Path: {doc['metadata_path']}")
        else:
            logger.warning("No documents were successfully parsed in the test run.")