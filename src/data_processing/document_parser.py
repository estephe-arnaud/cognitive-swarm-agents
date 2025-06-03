"""
Document Parser Module

This module provides functionality for parsing PDF documents and their associated metadata.
It handles the extraction of text content from PDF files and combines it with metadata
from JSON files to create a comprehensive document representation.

Key features:
- PDF text extraction using PyMuPDF
- Metadata loading from JSON files
- Document collection processing
- Robust error handling and logging
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, TypedDict, Any

import fitz  # PyMuPDF

from config.settings import settings

logger = logging.getLogger(__name__)

class ParsedDocument(TypedDict):
    """
    Represents a parsed document with its content and metadata.
    
    Attributes:
        arxiv_id: Unique identifier for the document
        text_content: Extracted text content from the PDF
        metadata: Document metadata from JSON file
        pdf_path: Path to the original PDF file
        metadata_path: Path to the original metadata JSON file
    """
    arxiv_id: str
    text_content: str
    metadata: Dict[str, Any]
    pdf_path: str
    metadata_path: Optional[str]

def load_metadata(metadata_filepath: Path) -> Optional[Dict[str, Any]]:
    """
    Load metadata from a JSON file.
    
    Args:
        metadata_filepath: Path to the metadata JSON file
        
    Returns:
        Dictionary containing metadata if successful, None otherwise
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
    Extract text content from a single PDF file.
    
    Args:
        pdf_filepath: Path to the PDF file
        
    Returns:
        Extracted text content if successful, None otherwise
    """
    if not pdf_filepath.exists():
        logger.error(f"PDF file not found: {pdf_filepath}")
        return None

    logger.info(f"Parsing PDF: {pdf_filepath.name}")
    
    try:
        doc = fitz.open(pdf_filepath)
        text_content = []
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text_content.append(page.get_text("text"))
            
        doc.close()
        
        content = "\n".join(text_content).strip()
        logger.debug(f"Successfully parsed PDF: {pdf_filepath.name}, extracted {len(content)} characters")
        return content
        
    except Exception as e:
        logger.error(f"Error parsing PDF {pdf_filepath.name}: {e}", exc_info=True)
        return None

def parse_document_collection(
    pdf_dir: Path,
    metadata_dir: Path
) -> List[ParsedDocument]:
    """
    Parse all PDF documents in a directory and combine with their metadata.
    
    Args:
        pdf_dir: Directory containing PDF files
        metadata_dir: Directory containing metadata JSON files
        
    Returns:
        List of parsed documents with their content and metadata
    """
    parsed_documents: List[ParsedDocument] = []
    
    # Validate input directories
    if not pdf_dir.is_dir():
        logger.error(f"PDF input directory not found or is not a directory: {pdf_dir}")
        return parsed_documents
        
    if not metadata_dir.is_dir():
        logger.warning(
            f"Metadata input directory not found or is not a directory: {metadata_dir}. "
            "Attempting to proceed; metadata might be missing for some PDFs."
        )

    # Process PDF files
    pdf_files = list(pdf_dir.glob("*.pdf"))
    logger.info(f"Found {len(pdf_files)} PDF files in {pdf_dir} to parse")

    for pdf_filepath in pdf_files:
        arxiv_id_from_filename = pdf_filepath.stem
        
        # Extract text content
        text_content = parse_single_pdf(pdf_filepath)
        if text_content is None:
            logger.warning(f"Skipping document {arxiv_id_from_filename} due to PDF parsing error")
            continue

        # Load metadata
        metadata_filepath = metadata_dir / f"{arxiv_id_from_filename}_metadata.json"
        metadata = load_metadata(metadata_filepath)
        
        # Prepare document metadata
        if metadata is None:
            logger.warning(
                f"Metadata file for {arxiv_id_from_filename} not found or failed to load from {metadata_filepath}. "
                "Proceeding with minimal metadata."
            )
            document_metadata = {"arxiv_id_inferred_from_filename": arxiv_id_from_filename}
            metadata_path_str = None
        else:
            document_metadata = metadata
            metadata_path_str = str(metadata_filepath)

        # Normalize ArXiv ID
        final_arxiv_id = document_metadata.get("entry_id", arxiv_id_from_filename)
        if isinstance(final_arxiv_id, str):
            final_arxiv_id = final_arxiv_id.split("/")[-1].split("v")[0]
        else:
            final_arxiv_id = arxiv_id_from_filename

        # Create parsed document
        parsed_doc: ParsedDocument = {
            "arxiv_id": final_arxiv_id,
            "text_content": text_content,
            "metadata": document_metadata,
            "pdf_path": str(pdf_filepath),
            "metadata_path": metadata_path_str
        }
        
        parsed_documents.append(parsed_doc)
        logger.info(f"Successfully processed and added document: {parsed_doc['arxiv_id']}")

    logger.info(f"Finished parsing collection. Successfully processed {len(parsed_documents)} documents")
    return parsed_documents

def _create_test_directories() -> tuple[Path, Path]:
    """
    Create test directories for document parsing.
    
    Returns:
        Tuple of (PDF directory path, metadata directory path)
    """
    test_base_data_dir = Path(settings.DATA_DIR) / "corpus" / "test_parser_corpus"
    test_pdf_input_dir = test_base_data_dir / "pdfs"
    test_metadata_input_dir = test_base_data_dir / "metadata"
    
    test_pdf_input_dir.mkdir(parents=True, exist_ok=True)
    test_metadata_input_dir.mkdir(parents=True, exist_ok=True)
    
    return test_pdf_input_dir, test_metadata_input_dir

def _display_test_results(documents: List[ParsedDocument]) -> None:
    """
    Display test results for parsed documents.
    
    Args:
        documents: List of parsed documents to display
    """
    if not documents:
        logger.warning("No documents were successfully parsed in the test run")
        return
        
    logger.info(f"Successfully parsed {len(documents)} documents in test run")
    
    for i, doc in enumerate(documents[:2]):
        logger.info(f"--- Document {i+1} ---")
        logger.info(f"  ArXiv ID: {doc['arxiv_id']}")
        logger.info(f"  Title (from metadata): {doc['metadata'].get('title', 'N/A')}")
        logger.info(f"  Text Snippet: {doc['text_content'][:200].replace(chr(10), ' ')}...")
        logger.info(f"  PDF Path: {doc['pdf_path']}")
        logger.info(f"  Metadata Path: {doc['metadata_path']}")

if __name__ == "__main__":
    from config.logging_config import setup_logging
    setup_logging(level="INFO")
    
    logger.info("Starting document parser test run")
    
    # Create test directories
    test_pdf_input_dir, test_metadata_input_dir = _create_test_directories()
    
    # Check for test files
    if not any(test_pdf_input_dir.iterdir()):
        logger.warning(
            f"No PDF files found in the test directory: {test_pdf_input_dir}.\n"
            "Please ensure `arxiv_downloader.py` has run and populated a similar test directory,\n"
            "or manually add test PDF and metadata JSON files to the paths above for a meaningful test.\n"
            "For example, create 'testdoc01.pdf' and 'testdoc01_metadata.json'."
        )
    else:
        # Parse documents
        logger.info(
            f"Attempting to parse documents from PDF dir: {test_pdf_input_dir} "
            f"and Metadata dir: {test_metadata_input_dir}"
        )
        
        documents = parse_document_collection(
            pdf_dir=test_pdf_input_dir,
            metadata_dir=test_metadata_input_dir
        )
        
        _display_test_results(documents)
    
    logger.info("Document parser test run finished")