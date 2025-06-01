# cognitive-swarm-agents/src/data_processing/preprocessor.py
import logging
import re
from typing import List, Dict, TypedDict, Any, Optional
import tiktoken

from config.settings import settings
# Assuming ParsedDocument structure is available or defined similarly
# from src.data_processing.document_parser import ParsedDocument # Or define here if preferred

logger = logging.getLogger(__name__)

# Define ParsedDocument structure if not imported, for clarity
class ParsedDocument(TypedDict):
    arxiv_id: str
    text_content: str
    metadata: Dict
    pdf_path: str
    metadata_path: Optional[str]

class ProcessedChunk(TypedDict):
    chunk_id: str # e.g., arxiv_id_chunk_001
    arxiv_id: str
    text_chunk: str
    # Potentially add some original metadata for context, e.g., title
    original_document_title: Optional[str]
    # We can add more structured metadata per chunk if needed later

def clean_text(text: str) -> str:
    """
    Performs basic cleaning of the extracted text.

    Args:
        text (str): The raw text content.

    Returns:
        str: Cleaned text.
    """
    if not text:
        return ""

    # Remove excessive newlines, leaving at most two consecutive newlines
    text = re.sub(r"\n\s*\n", "\n\n", text)
    # Replace multiple spaces with a single space
    text = re.sub(r" +", " ", text)
    # Remove leading/trailing whitespace from each line
    text = "\n".join([line.strip() for line in text.splitlines() if line.strip()])
    
    # Specific cleaning for scientific papers (can be expanded):
    # - Remove hyphenation at the end of lines if it's part of a word split.
    #   This is a more complex NLP task; for now, we'll keep it simple.
    #   A basic approach might be:
    text = text.replace("-\n", "") # Remove hyphen followed immediately by newline

    # TODO: Consider more advanced cleaning:
    # - Handling ligatures (e.g., "ﬁ" -> "fi") if not handled by PDF parser.
    # - Removing headers/footers if they are consistently present and noisy (complex).

    logger.debug(f"Cleaned text. Original length: {len(text)}, New length: {len(text)}") # This log is incorrect, text is modified in place for len()
    # Corrected logging for length comparison:
    # original_length = len(text) # This would be before any modifications in this function
    # cleaned_text = ... # result of cleaning
    # logger.debug(f"Cleaned text. Original length: {original_length}, New length: {len(cleaned_text)}")
    # For now, this basic logging is fine as the function is simple.

    return text.strip()

def chunk_text_by_tokens(
    text: str,
    chunk_size: int = settings.CHUNK_SIZE,
    chunk_overlap: int = settings.CHUNK_OVERLAP,
    encoding_name: str = "cl100k_base", # Used by OpenAI's text-embedding-ada-002 and newer
) -> List[str]:
    """
    Splits text into overlapping chunks based on token count using tiktoken.
    Inspired by the function in agentic_rag_factory_safety_assistant notebook.

    Args:
        text (str): The text to chunk.
        chunk_size (int): Maximum number of tokens per chunk.
        chunk_overlap (int): Number of tokens to overlap between chunks.
        encoding_name (str): The name of the tiktoken encoding to use.

    Returns:
        List[str]: A list of text chunks.
    """
    if not text:
        return []
    
    try:
        encoding = tiktoken.get_encoding(encoding_name)
    except Exception as e:
        logger.error(f"Failed to get tiktoken encoding '{encoding_name}': {e}. Defaulting to 'p50k_base'.")
        # Fallback or raise error. For RAG, using the correct tokenizer is important.
        # For now, let's assume cl100k_base is generally available with tiktoken.
        # If critical, this should raise an error or have a more robust fallback.
        encoding = tiktoken.get_encoding("p50k_base") # A common older encoding

    tokens = encoding.encode(text)
    num_tokens = len(tokens)
    
    chunks: List[str] = []
    current_pos = 0
    while current_pos < num_tokens:
        end_pos = min(current_pos + chunk_size, num_tokens)
        chunk_tokens = tokens[current_pos:end_pos]
        chunk_text_content = encoding.decode(chunk_tokens)
        chunks.append(chunk_text_content)
        
        if end_pos == num_tokens: # Reached the end
            break
        
        current_pos += (chunk_size - chunk_overlap)
        if current_pos >= num_tokens: # Ensure we don't go past the end with overlap logic
            break
        # Handle cases where overlap makes current_pos > end_pos if chunk_size is small
        if current_pos >= end_pos and end_pos < num_tokens : # Safety break for unusual overlap scenarios
            logger.warning("Chunking logic resulted in current_pos >= end_pos. Breaking to avoid infinite loop.")
            break


    logger.debug(f"Chunked text into {len(chunks)} chunks. Original tokens: {num_tokens}, Chunk size: {chunk_size}, Overlap: {chunk_overlap}")
    return chunks

def preprocess_parsed_documents(
    parsed_documents: List[ParsedDocument]
) -> List[ProcessedChunk]:
    """
    Preprocesses a list of parsed documents: cleans text and splits into chunks.

    Args:
        parsed_documents (List[ParsedDocument]): List of documents from the parsing stage.

    Returns:
        List[ProcessedChunk]: A list of processed text chunks, ready for embedding.
    """
    all_processed_chunks: List[ProcessedChunk] = []
    if not parsed_documents:
        logger.warning("No parsed documents provided for preprocessing.")
        return all_processed_chunks

    logger.info(f"Starting preprocessing for {len(parsed_documents)} parsed documents.")

    for doc_idx, doc in enumerate(parsed_documents):
        logger.info(f"Preprocessing document {doc_idx + 1}/{len(parsed_documents)}: {doc['arxiv_id']}")
        
        cleaned_text = clean_text(doc["text_content"])
        if not cleaned_text:
            logger.warning(f"Document {doc['arxiv_id']} has no content after cleaning. Skipping.")
            continue

        text_chunks = chunk_text_by_tokens(
            cleaned_text,
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP
        )

        doc_title = doc["metadata"].get("title", "N/A")

        for chunk_idx, chunk_content in enumerate(text_chunks):
            chunk_id = f"{doc['arxiv_id']}_chunk_{str(chunk_idx + 1).zfill(3)}"
            processed_chunk: ProcessedChunk = {
                "chunk_id": chunk_id,
                "arxiv_id": doc["arxiv_id"],
                "text_chunk": chunk_content,
                "original_document_title": doc_title,
                # Store other relevant metadata from doc['metadata'] if needed
                # e.g., "authors": doc["metadata"].get("authors", []),
                # "publication_date": doc["metadata"].get("published", None)
            }
            all_processed_chunks.append(processed_chunk)
        
        logger.debug(f"Document {doc['arxiv_id']} processed into {len(text_chunks)} chunks.")

    logger.info(f"Finished preprocessing. Generated {len(all_processed_chunks)} chunks in total.")
    return all_processed_chunks


if __name__ == "__main__":
    from config.logging_config import setup_logging
    setup_logging(level="INFO")

    logger.info("Starting preprocessor test run...")

    # Create some dummy ParsedDocument data for testing
    sample_doc_1_text = """This is the first sentence of document one. It contains several interesting points.
    Here is another paragraph. It discusses various aspects of a complex topic.
    The conclusion summarizes the main findings and suggests future work.
    """ * 20 # Make it long enough for multiple chunks

    sample_doc_2_text = """Document two starts here. It's a bit shorter but no less important.
    It focuses on a specific methodology.
    """ * 15


    test_parsed_documents: List[ParsedDocument] = [
        {
            "arxiv_id": "test001",
            "text_content": sample_doc_1_text,
            "metadata": {"title": "A Study of Interesting Things", "authors": ["Author A", "Author B"]},
            "pdf_path": "/fake/path/test001.pdf",
            "metadata_path": "/fake/path/test001_metadata.json"
        },
        {
            "arxiv_id": "test002",
            "text_content": sample_doc_2_text,
            "metadata": {"title": "Methodologies Explored"},
            "pdf_path": "/fake/path/test002.pdf",
            "metadata_path": "/fake/path/test002_metadata.json"
        },
        { # Test with empty content
            "arxiv_id": "test003",
            "text_content": "",
            "metadata": {"title": "Empty Document Test"},
            "pdf_path": "/fake/path/test003.pdf",
            "metadata_path": "/fake/path/test003_metadata.json"
        }
    ]

    # Override chunk settings for testing if needed
    # settings.CHUNK_SIZE = 50
    # settings.CHUNK_OVERLAP = 10
    
    processed_chunks = preprocess_parsed_documents(test_parsed_documents)

    if processed_chunks:
        logger.info(f"Successfully preprocessed documents into {len(processed_chunks)} chunks.")
        for i, chunk_data in enumerate(processed_chunks[:5]): # Log details of first 5 chunks
            logger.info(f"--- Chunk {i+1} ({chunk_data['chunk_id']}) ---")
            logger.info(f"  ArXiv ID: {chunk_data['arxiv_id']}")
            logger.info(f"  Original Title: {chunk_data['original_document_title']}")
            logger.info(f"  Chunk Text Snippet: {chunk_data['text_chunk'][:100].replace(chr(10), ' ')}...")
    else:
        logger.warning("No chunks were generated in the test run.")

    # Test with tiktoken encoding not found (to see fallback, requires manual tiktoken manipulation or mock)
    # logger.info("\nTesting with non-existent encoding (expect warning and fallback):")
    # try:
    #     chunk_text_by_tokens("This is a test.", encoding_name="non_existent_encoding")
    # except Exception as e:
    #    logger.error(f"Test with non_existent_encoding failed as expected in main: {e}")