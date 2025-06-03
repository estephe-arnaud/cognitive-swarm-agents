"""
Text Preprocessor Module

This module provides functionality for preprocessing text documents, including cleaning,
chunking, and metadata management. It handles the transformation of raw text into
processed chunks suitable for embedding and storage.

Key features:
- Text cleaning and normalization
- Token-based text chunking
- Metadata preservation and management
- Robust error handling and logging
"""

import logging
import re
from typing import List, Dict, TypedDict, Any, Optional
import tiktoken

from config.settings import settings
# Assumant que ParsedDocument est défini dans document_parser.py ou de manière similaire
from src.data_processing.document_parser import ParsedDocument 

logger = logging.getLogger(__name__)

# class ParsedDocument(TypedDict): # Au cas où l'import ne fonctionnerait pas
#     arxiv_id: str
#     text_content: str
#     metadata: Dict
#     pdf_path: str
#     metadata_path: Optional[str]

class ProcessedChunk(TypedDict):
    """
    Represents a processed text chunk with its associated metadata.
    
    Attributes:
        chunk_id: Unique identifier for the chunk
        arxiv_id: ArXiv ID of the source document
        text_chunk: The processed text content
        original_document_title: Title of the source document
        source_document_metadata: Complete metadata from the source document
    """
    chunk_id: str
    arxiv_id: str
    text_chunk: str
    original_document_title: Optional[str]
    source_document_metadata: Optional[Dict[str, Any]]

def clean_text(text: str) -> str:
    """
    Clean and normalize text content.
    
    Args:
        text: Raw text to clean
        
    Returns:
        Cleaned and normalized text
    """
    if not text:
        return ""
        
    # Remove multiple newlines and spaces
    text = re.sub(r"\n\s*\n", "\n\n", text)
    text = re.sub(r" +", " ", text)
    
    # Clean lines and remove empty ones
    text = "\n".join([line.strip() for line in text.splitlines() if line.strip()])
    
    # Remove hyphenation
    text = text.replace("-\n", "")
    
    return text.strip()

def chunk_text_by_tokens(
    text: str,
    chunk_size: int = settings.CHUNK_SIZE,
    chunk_overlap: int = settings.CHUNK_OVERLAP,
    encoding_name: str = "cl100k_base"
) -> List[str]:
    """
    Split text into chunks based on token count.
    
    Args:
        text: Text to split into chunks
        chunk_size: Maximum number of tokens per chunk
        chunk_overlap: Number of tokens to overlap between chunks
        encoding_name: Name of the tokenizer encoding to use
        
    Returns:
        List of text chunks
    """
    if not text:
        return []
        
    try:
        encoding = tiktoken.get_encoding(encoding_name)
    except Exception as e:
        logger.error(
            f"Failed to get tiktoken encoding '{encoding_name}': {e}. "
            "Defaulting to 'p50k_base'."
        )
        encoding = tiktoken.get_encoding("p50k_base")

    tokens = encoding.encode(text)
    num_tokens = len(tokens)
    
    chunks: List[str] = []
    current_pos = 0
    
    while current_pos < num_tokens:
        end_pos = min(current_pos + chunk_size, num_tokens)
        chunk_tokens = tokens[current_pos:end_pos]
        chunk_text_content = encoding.decode(chunk_tokens)
        chunks.append(chunk_text_content)
        
        if end_pos == num_tokens:
            break
            
        current_pos += (chunk_size - chunk_overlap)
        
        if current_pos >= num_tokens:
            break
            
        if current_pos >= end_pos and end_pos < num_tokens:
            logger.warning(
                "Chunking logic resulted in current_pos >= end_pos. "
                "Breaking to avoid infinite loop."
            )
            break
            
    logger.debug(
        f"Chunked text into {len(chunks)} chunks. "
        f"Original tokens: {num_tokens}, "
        f"Chunk size: {chunk_size}, "
        f"Overlap: {chunk_overlap}"
    )
    return chunks

def preprocess_parsed_documents(
    parsed_documents: List[ParsedDocument]
) -> List[ProcessedChunk]:
    """
    Preprocess a list of parsed documents into chunks.
    
    Args:
        parsed_documents: List of parsed documents to process
        
    Returns:
        List of processed chunks with metadata
    """
    all_processed_chunks: List[ProcessedChunk] = []
    
    if not parsed_documents:
        logger.warning("No parsed documents provided for preprocessing.")
        return all_processed_chunks

    logger.info(f"Starting preprocessing for {len(parsed_documents)} parsed documents.")

    for doc_idx, doc in enumerate(parsed_documents):
        logger.info(
            f"Preprocessing document {doc_idx + 1}/{len(parsed_documents)}: "
            f"{doc['arxiv_id']}"
        )
        
        # Clean text
        cleaned_text = clean_text(doc["text_content"])
        if not cleaned_text:
            logger.warning(
                f"Document {doc['arxiv_id']} has no content after cleaning. Skipping."
            )
            continue

        # Chunk text
        text_chunks = chunk_text_by_tokens(
            cleaned_text,
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP
        )

        # Get document metadata
        doc_title = doc["metadata"].get("title", "N/A")
        source_metadata = doc.get("metadata")

        # Create processed chunks
        for chunk_idx, chunk_content in enumerate(text_chunks):
            chunk_id = f"{doc['arxiv_id']}_chunk_{str(chunk_idx + 1).zfill(3)}"
            processed_chunk: ProcessedChunk = {
                "chunk_id": chunk_id,
                "arxiv_id": doc["arxiv_id"],
                "text_chunk": chunk_content,
                "original_document_title": doc_title,
                "source_document_metadata": source_metadata
            }
            all_processed_chunks.append(processed_chunk)
        
        logger.debug(
            f"Document {doc['arxiv_id']} processed into {len(text_chunks)} chunks."
        )

    logger.info(
        f"Finished preprocessing. Generated {len(all_processed_chunks)} chunks in total."
    )
    return all_processed_chunks

def _create_test_documents() -> List[ParsedDocument]:
    """
    Create sample documents for testing.
    
    Returns:
        List of sample parsed documents
    """
    sample_doc_1_text = (
        "This is the first sentence of document one. It contains several interesting points.\n"
        "Here is another paragraph. It discusses various aspects of a complex topic.\n"
        "The conclusion summarizes the main findings and suggests future work.\n"
    ) * 20

    sample_doc_2_text = (
        "Document two starts here. It's a bit shorter but no less important.\n"
        "It focuses on a specific methodology.\n"
    ) * 15

    return [
        {
            "arxiv_id": "test001",
            "text_content": sample_doc_1_text,
            "metadata": {
                "title": "A Study of Interesting Things",
                "authors": ["Author A", "Author B"],
                "primary_category": "cs.AI"
            },
            "pdf_path": "/fake/path/test001.pdf",
            "metadata_path": "/fake/path/test001_metadata.json"
        },
        {
            "arxiv_id": "test002",
            "text_content": sample_doc_2_text,
            "metadata": {
                "title": "Methodologies Explored",
                "primary_category": "cs.RO"
            },
            "pdf_path": "/fake/path/test002.pdf",
            "metadata_path": "/fake/path/test002_metadata.json"
        },
        {
            "arxiv_id": "test003",
            "text_content": "",
            "metadata": {"title": "Empty Document Test"},
            "pdf_path": "/fake/path/test003.pdf",
            "metadata_path": "/fake/path/test003_metadata.json"
        }
    ]

if __name__ == "__main__":
    from config.logging_config import setup_logging
    setup_logging(level="INFO")

    logger.info("Starting preprocessor test run...")

    # Create and process test documents
    test_parsed_documents = _create_test_documents()
    processed_chunks = preprocess_parsed_documents(test_parsed_documents)

    if processed_chunks:
        logger.info(
            f"Successfully preprocessed documents into {len(processed_chunks)} chunks."
        )
        
        # Display sample chunks
        for i, chunk_data in enumerate(processed_chunks[:2]):
            logger.info(f"--- Chunk {i+1} ({chunk_data['chunk_id']}) ---")
            logger.info(f"  ArXiv ID: {chunk_data['arxiv_id']}")
            logger.info(f"  Original Title: {chunk_data['original_document_title']}")
            logger.info(
                f"  Chunk Text Snippet: "
                f"{chunk_data['text_chunk'][:100].replace(chr(10), ' ')}..."
            )
            logger.info(
                f"  Source Metadata (extrait): "
                f"{ {k: v for k, v in chunk_data.get('source_document_metadata', {}).items() if k in ['primary_category']} }"
            )
    else:
        logger.warning("No chunks were generated in the test run.")