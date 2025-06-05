"""
Tool Definitions Module

This module defines the core tools used by the research assistant agents:
- ArXiv Search Tool: Searches scientific papers on ArXiv
- Knowledge Base Retrieval Tool: Retrieves information from the ingested knowledge base
- Document Deep Dive Analysis Tool: Performs in-depth analysis of scientific documents

Each tool is implemented as a LangChain tool with proper error handling and logging.
"""

import logging
from typing import List, Dict, Any, Optional
import json
import io

from langchain_core.tools import tool
import arxiv
import requests
from pypdf import PdfReader

from config.settings import settings
from src.agents.crewai_teams.document_analysis_crew import run_document_deep_dive_crew

logger = logging.getLogger(__name__)

# Constants for ArXiv search
ARXIV_SORT_CRITERIA = {
    "relevance": arxiv.SortCriterion.Relevance,
    "lastupdateddate": arxiv.SortCriterion.LastUpdatedDate,
    "submitteddate": arxiv.SortCriterion.SubmittedDate,
}

ARXIV_SORT_ORDERS = {
    "ascending": arxiv.SortOrder.Ascending,
    "descending": arxiv.SortOrder.Descending,
}

@tool
def arxiv_search_tool(
    query: str,
    max_results: int = 5,
    sort_by: str = "relevance",
    sort_order: str = "descending",
) -> List[Dict[str, Any]]:
    """
    Use this tool to search the ArXiv repository for scientific papers.

    It is your primary method for finding new research papers on a specific topic.
    The query should be a concise string, similar to what you would use in a search engine.

    Args:
        query: A specific and targeted search query.
               Examples: "explainable AI in robotics", "author:Geoffrey Hinton", "cat:cs.CV"
        max_results: The maximum number of papers to return. Keep this low (e.g., 3 to 5)
                     to avoid overwhelming the analysis stage.
        sort_by: The criterion for sorting results. Options: 'relevance', 'lastUpdatedDate', 'submittedDate'.
        sort_order: The order of results. Options: 'ascending', 'descending'.

    Returns:
        A list of dictionaries, where each dictionary represents a found paper
        with its title, authors, summary, PDF link, and other metadata.
        Returns an empty list if no papers are found.
    """
    logger.info(
        f"Executing arxiv_search_tool: query='{query}', max_results={max_results}"
    )

    try:
        # Validate and map sort parameters
        sort_criterion = ARXIV_SORT_CRITERIA.get(
            sort_by.lower(), arxiv.SortCriterion.Relevance
        )
        if sort_by.lower() not in ARXIV_SORT_CRITERIA:
            logger.warning(f"Invalid sort_by value '{sort_by}'. Using 'relevance'.")

        order_criterion = ARXIV_SORT_ORDERS.get(
            sort_order.lower(), arxiv.SortOrder.Descending
        )
        if sort_order.lower() not in ARXIV_SORT_ORDERS:
            logger.warning(f"Invalid sort_order value '{sort_order}'. Using 'descending'.")

        # Execute search
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=sort_criterion,
            sort_order=order_criterion,
        )

        # Process results
        results = []
        for result in search.results():
            results.append(
                {
                    "entry_id": result.entry_id,
                    "title": result.title,
                    "authors": [str(author) for author in result.authors],
                    "summary": result.summary.replace("\n", " "),
                    "published_date": result.published.isoformat()
                    if result.published
                    else None,
                    "pdf_url": result.pdf_url,
                    "primary_category": result.primary_category,
                }
            )

        logger.info(f"Found {len(results)} papers for query: '{query}'")
        return results

    except Exception as e:
        logger.error(f"ArXiv search failed for query '{query}': {e}", exc_info=True)
        return [{"error": f"ArXiv search failed: {str(e)}"}]

@tool
def knowledge_base_retrieval_tool(
    query_text: str,
    top_k: int = 5,
    metadata_filters: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """
    Use this tool to retrieve relevant text chunks from our internal, curated
    knowledge base of already processed documents.

    This is useful for finding specific, targeted information or facts within
    documents that are already part of our system. It is faster than a full
    deep dive.

    Args:
        query_text: The specific question or topic to search for in the knowledge base.
        top_k: The number of most relevant text chunks to return.
        metadata_filters: Optional list of filters to apply to the search,
                          e.g., `[{"key": "arxiv_id", "value": "2301.12345"}]`.

    Returns:
        A list of retrieved document chunks, each with its content, source, and
        relevance score. Returns an empty list if no relevant chunks are found.
    """
    # Import here for lazy initialization
    from src.rag.retrieval_engine import RetrievalEngine

    logger.info(
        f"Executing knowledge_base_retrieval_tool: query='{query_text[:50]}...'"
    )

    try:
        retrieval_engine = RetrievalEngine()
        logger.info("RetrievalEngine initialized successfully.")

        # Process metadata filters
        llama_filters = []
        if metadata_filters:
            from llama_index.core.vector_stores import ExactMatchFilter

            for filter_dict in metadata_filters:
                if "key" in filter_dict and "value" in filter_dict:
                    llama_filters.append(
                        ExactMatchFilter(
                            key=filter_dict["key"], value=filter_dict["value"]
                        )
                    )
                else:
                    logger.warning(f"Invalid metadata filter format: {filter_dict}")

        # Execute retrieval
        retrieved_nodes = retrieval_engine.retrieve_simple_vector_search(
            query_text=query_text,
            top_k=top_k,
            metadata_filters=llama_filters if llama_filters else None,
        )

        # Format results
        results = [
            {
                "chunk_id": node.metadata.get("chunk_id", "N/A"),
                "arxiv_id": node.metadata.get("arxiv_id", "N/A"),
                "original_document_title": node.metadata.get(
                    "original_document_title", "N/A"
                ),
                "text_chunk": node.text,
                "retrieval_score": node.score,
            }
            for node in retrieved_nodes
        ]

        logger.info(f"Retrieved {len(results)} chunks from knowledge base.")
        return results

    except Exception as e:
        logger.error(f"Knowledge base retrieval failed: {e}", exc_info=True)
        return [{"error": f"Knowledge base retrieval failed: {str(e)}"}]

def _fetch_pdf_content(pdf_url: str) -> str:
    """Downloads a PDF from a URL and extracts its text content."""
    logger.info(f"Fetching PDF content from: {pdf_url}")
    try:
        response = requests.get(pdf_url, timeout=30)
        response.raise_for_status()  # Raise an exception for bad status codes

        with io.BytesIO(response.content) as pdf_file:
            reader = PdfReader(pdf_file)
            text_content = " ".join(
                page.extract_text() for page in reader.pages if page.extract_text()
            )
        logger.info(f"Successfully extracted text from {pdf_url}")
        return text_content
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to download PDF from {pdf_url}: {e}")
        return f"Error: Failed to download PDF. {e}"
    except Exception as e:
        logger.error(f"Failed to parse PDF from {pdf_url}: {e}", exc_info=True)
        return f"Error: Failed to parse PDF content. {e}"

@tool
def document_deep_dive_analysis_tool(
    pdf_url: str, research_focus: str
) -> str:
    """
    Use this powerful tool to perform a comprehensive, deep analysis of a *single*
    scientific paper by providing its PDF URL.

    This tool is "expensive" as it reads and analyzes the entire document, so
    use it judiciously on the most promising papers identified by an ArXiv search.
    It is your best method for understanding the core contributions, methodology,
    and conclusions of a paper.

    Args:
        pdf_url: The direct URL to the PDF of the paper to be analyzed.
        research_focus: A clear, specific question or a set of themes to guide the
                      analysis. For example: "What is the main contribution of this
                      paper?" or "Analyze the methodology and experimental results."

    Returns:
        A structured, in-depth analytical report of the document, or a detailed
        error message if the analysis fails.
    """
    logger.info(
        f"Executing document_deep_dive_analysis_tool: url='{pdf_url}', focus='{research_focus}'"
    )

    document_content = _fetch_pdf_content(pdf_url)
    if document_content.startswith("Error:"):
        return document_content  # Propagate the error message

    if not document_content or not document_content.strip():
        logger.error(f"Could not extract any text content from PDF at {pdf_url}")
        return "Error: Document content is empty or could not be extracted."

    if not research_focus or not research_focus.strip():
        logger.warning("Research focus is empty; analysis may be too generic.")

    try:
        # Extract ArXiv ID from URL for logging and identification
        arxiv_id = pdf_url.split("/")[-1]
        report = run_document_deep_dive_crew(
            document_id=arxiv_id,
            document_content=document_content,
            research_focus=research_focus,
        )

        logger.info(f"Analysis completed successfully for doc_id='{arxiv_id}'")
        return report

    except ImportError as e:
        logger.error(f"CrewAI import error: {e}", exc_info=True)
        return f"Error: CrewAI components are not available. {e}"
    except Exception as e:
        logger.error(f"Analysis failed for url='{pdf_url}': {e}", exc_info=True)
        return f"Error during deep dive analysis: {e}"

def _test_tools():
    """Run tests for all tools with proper configuration checks."""
    logger.info("Testing Tool Definitions")
    
    # Test ArXiv search
    logger.info("\nTesting ArXiv Search Tool")
    arxiv_results = arxiv_search_tool.invoke({
        "query": "explainable AI in robotics",
        "max_results": 1,
        "sort_by": "submittedDate"
    })
    print(f"ArXiv Search Results:\n{json.dumps(arxiv_results, indent=2, ensure_ascii=False)}\n")
    
    # Test knowledge base retrieval
    logger.info("\nTesting Knowledge Base Retrieval Tool")
    if _can_test_knowledge_base():
        kb_results = knowledge_base_retrieval_tool.invoke({
            "query_text": "explainable AI in robotics",
            "top_k": 1
        })
        print(f"Knowledge Base Results:\n{json.dumps(kb_results, indent=2, ensure_ascii=False)}\n")
    else:
        print("Skipping knowledge base test: Configuration incomplete\n")
    
    # Test deep dive analysis
    logger.info("\nTesting Deep Dive Analysis Tool")
    if _can_test_deep_dive():
        test_doc = {
            "pdf_url": "https://arxiv.org/pdf/2301.12345.pdf",
            "research_focus": "Extract methodology, results, and limitations"
        }
        report = document_deep_dive_analysis_tool.invoke(test_doc)
        print("\nDeep Dive Analysis Report:")
        print(report)
        print("------------------------\n")
    else:
        print("Skipping deep dive test: Configuration incomplete\n")
    
    logger.info("Tool testing completed")

def _can_test_knowledge_base() -> bool:
    """Check if knowledge base testing is possible with current configuration."""
    if not (settings.MONGODB_URI and settings.DEFAULT_EMBEDDING_PROVIDER):
        return False
        
    if settings.DEFAULT_EMBEDDING_PROVIDER == "openai":
        return bool(settings.OPENAI_API_KEY)
    elif settings.DEFAULT_EMBEDDING_PROVIDER == "huggingface":
        return True
    elif settings.DEFAULT_EMBEDDING_PROVIDER == "ollama":
        return bool(settings.OLLAMA_BASE_URL and settings.OLLAMA_EMBEDDING_MODEL_NAME)
    
    return False

def _can_test_deep_dive() -> bool:
    """Check if deep dive testing is possible with current configuration."""
    if settings.DEFAULT_LLM_MODEL_PROVIDER == "openai":
        return bool(settings.OPENAI_API_KEY)
    elif settings.DEFAULT_LLM_MODEL_PROVIDER == "huggingface_api":
        return bool(settings.HUGGINGFACE_API_KEY and settings.HUGGINGFACE_REPO_ID)
    elif settings.DEFAULT_LLM_MODEL_PROVIDER == "ollama":
        return bool(settings.OLLAMA_BASE_URL and settings.OLLAMA_GENERATIVE_MODEL_NAME)
    
    return False

if __name__ == "__main__":
    from config.logging_config import setup_logging
    setup_logging(level="DEBUG")
    _test_tools()