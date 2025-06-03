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

from langchain_core.tools import tool
import arxiv

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
    max_results: int = 3, 
    sort_by: str = "relevance", 
    sort_order: str = "descending"
) -> List[Dict[str, Any]]:
    """
    Search the ArXiv repository for scientific papers.
    
    Args:
        query: The search query (e.g., "quantum machine learning", "author:Geoffrey Hinton")
        max_results: Maximum number of results to return (default: 3)
        sort_by: Sort criterion ('relevance', 'lastUpdatedDate', 'submittedDate')
        sort_order: Sort order ('ascending', 'descending')
        
    Returns:
        List of paper summaries with metadata including:
        - entry_id: ArXiv ID
        - title: Paper title
        - authors: List of author names
        - summary: Paper abstract
        - published_date: Publication date
        - pdf_url: Link to PDF
        - primary_category: Main category
    """
    logger.info(f"Executing arxiv_search_tool: query='{query}', max_results={max_results}")
    
    try:
        # Validate and map sort parameters
        sort_criterion = ARXIV_SORT_CRITERIA.get(sort_by.lower(), arxiv.SortCriterion.Relevance)
        if sort_by.lower() not in ARXIV_SORT_CRITERIA:
            logger.warning(f"Invalid sort_by value '{sort_by}'. Using 'relevance'")
            
        order_criterion = ARXIV_SORT_ORDERS.get(sort_order.lower(), arxiv.SortOrder.Descending)
        if sort_order.lower() not in ARXIV_SORT_ORDERS:
            logger.warning(f"Invalid sort_order value '{sort_order}'. Using 'descending'")

        # Execute search
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=sort_criterion,
            sort_order=order_criterion
        )
        
        # Process results
        results = []
        for result in search.results():
            results.append({
                "entry_id": result.entry_id,
                "title": result.title,
                "authors": [str(author) for author in result.authors],
                "summary": result.summary.replace('\n', ' '),
                "published_date": result.published.isoformat() if result.published else None,
                "pdf_url": result.pdf_url,
                "primary_category": result.primary_category,
            })
            
        logger.info(f"Found {len(results)} papers")
        return results
        
    except Exception as e:
        logger.error(f"ArXiv search failed: {e}", exc_info=True)
        return [{"error": f"ArXiv search failed: {str(e)}"}]

@tool
def knowledge_base_retrieval_tool(
    query_text: str,
    top_k: int = 3,
    metadata_filters: Optional[List[Dict[str, Any]]] = None
) -> List[Dict[str, Any]]:
    """
    Retrieve relevant information from the knowledge base.
    
    Args:
        query_text: The search query or topic
        top_k: Number of most relevant chunks to return
        metadata_filters: Optional list of metadata filters
        
    Returns:
        List of retrieved document chunks with:
        - chunk_id: Unique chunk identifier
        - arxiv_id: Source ArXiv paper ID
        - original_document_title: Title of source document
        - text_chunk: Retrieved text content
        - retrieval_score: Relevance score
    """
    # Import here for lazy initialization
    from src.rag.retrieval_engine import RetrievalEngine, RetrievedNode
    
    logger.info(f"Executing knowledge_base_retrieval_tool: query='{query_text[:50]}...'")
    
    try:
        # Initialize retrieval engine
        retrieval_engine = RetrievalEngine()
        logger.info("RetrievalEngine initialized successfully")
        
        # Process metadata filters
        llama_filters = []
        if metadata_filters:
            from llama_index.core.vector_stores import ExactMatchFilter
            for filter_dict in metadata_filters:
                if "key" in filter_dict and "value" in filter_dict:
                    llama_filters.append(
                        ExactMatchFilter(
                            key=filter_dict["key"],
                            value=filter_dict["value"]
                        )
                    )
                else:
                    logger.warning(f"Invalid metadata filter: {filter_dict}")
        
        # Execute retrieval
        retrieved_nodes = retrieval_engine.retrieve_simple_vector_search(
            query_text=query_text,
            top_k=top_k,
            metadata_filters=llama_filters if llama_filters else None
        )
        
        # Format results
        results = []
        for node in retrieved_nodes:
            results.append({
                "chunk_id": node.metadata.get("chunk_id", "N/A"),
                "arxiv_id": node.metadata.get("arxiv_id", "N/A"),
                "original_document_title": node.metadata.get("original_document_title", "N/A"),
                "text_chunk": node.text,
                "retrieval_score": node.score,
            })
            
        logger.info(f"Retrieved {len(results)} chunks")
        return results
        
    except Exception as e:
        logger.error(f"Knowledge base retrieval failed: {e}", exc_info=True)
        return [{"error": f"Knowledge base retrieval failed: {str(e)}"}]

@tool
def document_deep_dive_analysis_tool(
    document_id: str, 
    document_content: str, 
    research_focus: str
) -> str:
    """
    Perform in-depth analysis of a scientific document using CrewAI.
    
    Args:
        document_id: Unique identifier of the document
        document_content: Full text content to analyze
        research_focus: Specific questions or themes for analysis
        
    Returns:
        Structured analytical report or error message
    """
    logger.info(f"Executing document_deep_dive_analysis_tool: doc_id='{document_id}'")
    
    # Validate inputs
    if not document_content or not document_content.strip():
        logger.error("Empty document content")
        return "Error: Document content is empty"
        
    if not research_focus or not research_focus.strip():
        logger.warning("Empty research focus - analysis may be too generic")
    
    try:
        # Execute analysis
        report = run_document_deep_dive_crew(
            document_id=document_id,
            document_content=document_content,
            research_focus=research_focus
        )
        
        logger.info(f"Analysis completed for doc_id='{document_id}'")
        return report
        
    except ImportError as e:
        logger.error(f"CrewAI import error: {e}", exc_info=True)
        return f"Error: CrewAI components not available - {str(e)}"
    except Exception as e:
        logger.error(f"Analysis failed for doc_id='{document_id}': {e}", exc_info=True)
        return f"Error during analysis: {str(e)}"

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
            "document_id": "test_001",
            "document_content": "Sample document about robotic learning. New algorithm. Simulation methodology. Improved results. Computational cost limitation.",
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