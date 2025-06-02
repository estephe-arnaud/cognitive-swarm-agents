# src/agents/tool_definitions.py
import logging
from typing import List, Dict, Any, Optional

from langchain_core.tools import tool
import arxiv 

# RetrievalEngine sera importé localement dans la fonction knowledge_base_retrieval_tool
# pour permettre une initialisation paresseuse et une meilleure gestion des erreurs d'init.
from config.settings import settings
from src.agents.crewai_teams.document_analysis_crew import run_document_deep_dive_crew

logger = logging.getLogger(__name__)

@tool
def arxiv_search_tool(
    query: str,
    max_results: int = 3, 
    sort_by: str = "relevance", 
    sort_order: str = "descending"
) -> List[Dict[str, Any]]:
    """
    Searches the ArXiv repository for scientific papers based on a query.
    Useful for finding recent papers, or papers on a specific topic.
    Returns a list of paper summaries including title, authors, summary, PDF link, and publication date.
    Args:
        query (str): The search query (e.g., "quantum machine learning", "author:Geoffrey Hinton").
        max_results (int): Maximum number of results to return (default is 3).
        sort_by (str): Criterion to sort results by. Options: 'relevance', 'lastUpdatedDate', 'submittedDate'. Default 'relevance'.
        sort_order (str): Order of results. Options: 'ascending', 'descending'. Default 'descending'.
    """
    logger.info(f"Executing arxiv_search_tool with query='{query}', max_results={max_results}, sort_by='{sort_by}', sort_order='{sort_order}'")
    try:
        sort_criterion_map = {
            "relevance": arxiv.SortCriterion.Relevance,
            "lastupdateddate": arxiv.SortCriterion.LastUpdatedDate,
            "submitteddate": arxiv.SortCriterion.SubmittedDate,
        }
        sort_order_map = {
            "ascending": arxiv.SortOrder.Ascending,
            "descending": arxiv.SortOrder.Descending,
        }
        
        sort_criterion = sort_criterion_map.get(sort_by.lower(), arxiv.SortCriterion.Relevance)
        if sort_by.lower() not in sort_criterion_map:
            logger.warning(f"Invalid sort_by value '{sort_by}' for arxiv_search_tool. Defaulting to Relevance.")
            
        order_criterion = sort_order_map.get(sort_order.lower(), arxiv.SortOrder.Descending)
        if sort_order.lower() not in sort_order_map:
            logger.warning(f"Invalid sort_order value '{sort_order}' for arxiv_search_tool. Defaulting to Descending.")

        search = arxiv.Search(query=query, max_results=max_results, sort_by=sort_criterion, sort_order=order_criterion)
        
        # Utilisation du client par défaut de la bibliothèque arxiv.
        # Pour des configurations de client plus complexes (proxies, etc.), il faudrait instancier arxiv.Client() ici.
        results = []
        for r in search.results(): # search.results() utilise un client par défaut
            results.append({
                "entry_id": r.entry_id, "title": r.title, 
                "authors": [str(author) for author in r.authors], 
                "summary": r.summary.replace('\n', ' '),
                "published_date": r.published.isoformat() if r.published else None,
                "pdf_url": r.pdf_url, "primary_category": r.primary_category,
            })
        logger.info(f"arxiv_search_tool found {len(results)} papers.")
        return results
    except Exception as e:
        logger.error(f"Error in arxiv_search_tool: {e}", exc_info=True)
        return [{"error": f"arxiv_search_tool failed: {str(e)}"}]


@tool
def knowledge_base_retrieval_tool(
    query_text: str,
    top_k: int = 3,
    metadata_filters: Optional[List[Dict[str, Any]]] = None
) -> List[Dict[str, Any]]:
    """
    Retrieves relevant information chunks from the ingested knowledge base.
    Args:
        query_text (str): The query or topic to search for.
        top_k (int): The number of most relevant chunks to return.
        metadata_filters (Optional[List[Dict[str, Any]]]): Optional metadata filters.
    Returns:
        List[Dict[str, Any]]: A list of retrieved document chunks.
    """
    # Importation locale pour initialisation paresseuse et meilleure gestion des erreurs d'import/init.
    from src.rag.retrieval_engine import RetrievalEngine, RetrievedNode 
    
    logger.info(f"Executing knowledge_base_retrieval_tool with query='{query_text[:50]}...', top_k={top_k}, filters={metadata_filters}")
    
    retrieval_engine_instance: Optional[RetrievalEngine] = None
    try:
        # Initialisation de RetrievalEngine ici. 
        # RetrievalEngine utilise les valeurs par défaut pour collection_name et vector_index_name
        # (ou celles de settings.py si RetrievalEngine est configuré pour cela).
        # Assurez-vous que la configuration d'embedding (settings.DEFAULT_EMBEDDING_PROVIDER) 
        # et MONGODB_URI sont corrects pour que cette initialisation réussisse.
        retrieval_engine_instance = RetrievalEngine() 
        logger.info("RetrievalEngine instance for knowledge_base_retrieval_tool created/accessed successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize RetrievalEngine for knowledge_base_retrieval_tool: {e}", exc_info=True)
        return [{"error": f"RetrievalEngine initialization failed: {str(e)}"}]

    try:
        llama_filters_list = [] 
        if metadata_filters:
            from llama_index.core.vector_stores import ExactMatchFilter 
            for f_dict in metadata_filters:
                if "key" in f_dict and "value" in f_dict:
                    llama_filters_list.append(ExactMatchFilter(key=f_dict["key"], value=f_dict["value"])) 
                else: 
                    logger.warning(f"Malformed metadata filter skipped (missing 'key' or 'value'): {f_dict}")
        
        retrieved_nodes: List[RetrievedNode] = retrieval_engine_instance.retrieve_simple_vector_search(
            query_text=query_text, top_k=top_k,
            metadata_filters=llama_filters_list if llama_filters_list else None 
        )
        results_for_agent = []
        for node in retrieved_nodes:
            results_for_agent.append({
                "chunk_id": node.metadata.get("chunk_id", "N/A"),
                "arxiv_id": node.metadata.get("arxiv_id", "N/A"),
                "original_document_title": node.metadata.get("original_document_title", "N/A"),
                "text_chunk": node.text,
                "retrieval_score": node.score,
            })
        logger.info(f"knowledge_base_retrieval_tool retrieved {len(results_for_agent)} chunks.")
        return results_for_agent
    except Exception as e:
        logger.error(f"Error in knowledge_base_retrieval_tool during search: {e}", exc_info=True)
        return [{"error": f"knowledge_base_retrieval_tool search failed: {str(e)}"}]


@tool
def document_deep_dive_analysis_tool(
    document_id: str, 
    document_content: str, 
    research_focus: str
) -> str:
    """
    Performs an in-depth analysis of a single scientific document's text content using CrewAI.
    Args:
        document_id (str): The unique identifier of the document.
        document_content (str): The full text content of the document.
        research_focus (str): Specific questions or themes for the analysis.
    Returns:
        str: A structured analytical report, or an error message.
    """
    logger.info(f"Executing document_deep_dive_analysis_tool for doc_id='{document_id}', focus='{research_focus}'")
    if not document_content or not document_content.strip():
        logger.error("Document content for deep dive analysis is empty.")
        return "Error: Document content provided for deep dive analysis was empty or whitespace only."
    if not research_focus or not research_focus.strip(): # Le focus est important pour CrewAI
        logger.warning("Research focus for deep dive analysis is empty. Analysis might be very generic.")
        # On pourrait retourner une erreur ou laisser CrewAI gérer, mais un focus est généralement attendu.

    try:
        # La fonction run_document_deep_dive_crew gère l'exécution de la CrewAI.
        # Elle utilise llm_factory.py pour instancier le LLM, donc elle bénéficiera des corrections là-bas.
        report = run_document_deep_dive_crew(
            document_id=document_id,
            document_content=document_content,
            research_focus=research_focus
        )
        logger.info(f"document_deep_dive_analysis_tool completed for doc_id='{document_id}'. Report length: {len(report)}")
        return report
    except ImportError as ie: # Au cas où crewai ou une de ses dépendances ne serait pas là
        logger.error(f"CrewAI related import error for deep_dive_tool: {ie}", exc_info=True)
        return f"Error: CrewAI components not available. {str(ie)}"
    except Exception as e:
        logger.error(f"Error in document_deep_dive_analysis_tool for doc_id='{document_id}': {e}", exc_info=True)
        return f"Error during deep dive analysis for document {document_id}: {str(e)}"


if __name__ == "__main__":
    from config.logging_config import setup_logging
    import json 
    setup_logging(level="DEBUG")

    logger.info("--- Testing Tool Definitions ---")

    # Test arxiv_search_tool
    logger.info("\n--- Test Direct de arxiv_search_tool ---")
    arxiv_results = arxiv_search_tool.invoke({"query": "explainable AI in robotics", "max_results": 1, "sort_by": "submittedDate"})
    print(f"ArXiv Search Tool Direct Result:\n{json.dumps(arxiv_results, indent=2, ensure_ascii=False)}\n")

    # Test knowledge_base_retrieval_tool
    # Ce test nécessite une configuration fonctionnelle pour MONGODB_URI et DEFAULT_EMBEDDING_PROVIDER
    logger.info("\n--- Test Direct de knowledge_base_retrieval_tool (avec lazy init) ---")
    if settings.MONGODB_URI and settings.DEFAULT_EMBEDDING_PROVIDER :
        # Vérifications spécifiques pour les providers d'embedding
        can_test_kb = False
        if settings.DEFAULT_EMBEDDING_PROVIDER == "openai" and settings.OPENAI_API_KEY:
            can_test_kb = True
        elif settings.DEFAULT_EMBEDDING_PROVIDER == "huggingface": # Local, pas de clé API
            can_test_kb = True
        elif settings.DEFAULT_EMBEDDING_PROVIDER == "ollama" and settings.OLLAMA_BASE_URL and settings.OLLAMA_EMBEDDING_MODEL_NAME:
            can_test_kb = True
        
        if can_test_kb:
            kb_results = knowledge_base_retrieval_tool.invoke({"query_text": "explainable AI in robotics", "top_k": 1})
            print(f"Knowledge Base Retrieval Tool Direct Result:\n{json.dumps(kb_results, indent=2, ensure_ascii=False)}\n")
        else:
            print("Skipping knowledge_base_retrieval_tool test: Configuration for the selected embedding provider is incomplete (e.g., missing API key or URL).\n")
    else:
        print("Skipping knowledge_base_retrieval_tool test: MONGODB_URI or DEFAULT_EMBEDDING_PROVIDER not configured.\n")

    # Test document_deep_dive_analysis_tool
    # Ce test nécessite une configuration fonctionnelle pour DEFAULT_LLM_MODEL_PROVIDER (utilisé par CrewAI via get_llm)
    logger.info("\n--- Test Direct de document_deep_dive_analysis_tool ---")
    can_test_deep_dive = False
    if settings.DEFAULT_LLM_MODEL_PROVIDER == "openai" and settings.OPENAI_API_KEY:
        can_test_deep_dive = True
    elif settings.DEFAULT_LLM_MODEL_PROVIDER == "huggingface_api" and settings.HUGGINGFACE_API_KEY and settings.HUGGINGFACE_REPO_ID:
        can_test_deep_dive = True
    elif settings.DEFAULT_LLM_MODEL_PROVIDER == "ollama" and settings.OLLAMA_BASE_URL and settings.OLLAMA_GENERATIVE_MODEL_NAME:
        can_test_deep_dive = True

    if can_test_deep_dive:
        sample_doc_id_crew_test = "crew_tool_test_001"
        sample_doc_content_crew_test = "This is a sample document about advanced robotic learning for direct tool testing. It details a new algorithm. Methodology involves simulation. Key results showed improvement. A limitation is computational cost."
        sample_research_focus_crew_test = "Extract methodology, results, and limitations."
        deep_dive_report = document_deep_dive_analysis_tool.invoke({
            "document_id": sample_doc_id_crew_test,
            "document_content": sample_doc_content_crew_test,
            "research_focus": sample_research_focus_crew_test
        })
        print("\n--- Deep Dive Analysis Report (from Tool Test) ---")
        print(deep_dive_report)
        print("--------------------------------------------------\n")
    else:
        print("Skipping document_deep_dive_analysis_tool test: Configuration for the selected LLM provider is incomplete.\n")
    
    logger.info("Tool definition test run finished.")