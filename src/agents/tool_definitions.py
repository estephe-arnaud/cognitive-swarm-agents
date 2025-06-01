# cognitive-swarm-agents/src/agents/tool_definitions.py
import logging
from typing import List, Dict, Any, Optional

from langchain_core.tools import tool
import arxiv 

from src.rag.retrieval_engine import RetrievalEngine, RetrievedNode
from config.settings import settings
# <<< AJOUT : Importer la fonction d'exécution de notre CrewAI >>>
from src.agents.crewai_teams.document_analysis_crew import run_document_deep_dive_crew

logger = logging.getLogger(__name__)

# Initialisation du RetrievalEngine (existante)
try:
    retrieval_engine_instance = RetrievalEngine(
        collection_name=RetrievalEngine.DEFAULT_CHUNK_COLLECTION_NAME,
        vector_index_name=RetrievalEngine.DEFAULT_VECTOR_INDEX_NAME
    )
    logger.info("RetrievalEngine instance created successfully for tools.")
except Exception as e:
    logger.error(f"Failed to initialize RetrievalEngine for tools: {e}", exc_info=True)
    retrieval_engine_instance = None

@tool
def arxiv_search_tool(
    query: str,
    max_results: int = 3, # Réduit par défaut pour un usage plus ciblé par agent
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
    logger.info(f"Executing arxiv_search_tool with query='{query}', max_results={max_results}")
    # ... (code existant de arxiv_search_tool, je le garde concis ici pour ne pas tout répéter)
    try:
        if sort_by.lower() == "relevance": sort_criterion = arxiv.SortCriterion.Relevance
        elif sort_by.lower() == "lastupdateddate": sort_criterion = arxiv.SortCriterion.LastUpdatedDate
        elif sort_by.lower() == "submitteddate": sort_criterion = arxiv.SortCriterion.SubmittedDate
        else: sort_criterion = arxiv.SortCriterion.Relevance; logger.warning(f"Invalid sort_by for arxiv_search_tool: {sort_by}. Defaulting to Relevance.")
        if sort_order.lower() == "ascending": order_criterion = arxiv.SortOrder.Ascending
        elif sort_order.lower() == "descending": order_criterion = arxiv.SortOrder.Descending
        else: order_criterion = arxiv.SortOrder.Descending; logger.warning(f"Invalid sort_order for arxiv_search_tool: {sort_order}. Defaulting to Descending.")
        
        search = arxiv.Search(query=query, max_results=max_results, sort_by=sort_criterion, sort_order=order_criterion)
        client = arxiv.Client(page_size = min(max_results, 100), delay_seconds = 3, num_retries = 2)
        results = []
        for r in client.results(search):
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
    Retrieves relevant information chunks from the ingested knowledge base (ArXiv papers
    on Reinforcement Learning for Robotics) based on a query.
    Use this to find specific information, concepts, or methods within the already processed documents.
    Args:
        query_text (str): The query or topic to search for within the knowledge base.
        top_k (int): The number of most relevant chunks to return (default is 3).
        metadata_filters (Optional[List[Dict[str, Any]]]): Optional list of metadata filters to apply.
                                                            Each filter is a dict like {"key": "field_name", "value": "field_value"}.
                                                            Example: [{"key": "arxiv_id", "value": "2301.12345"}]
    Returns:
        List[Dict[str, Any]]: A list of retrieved document chunks, each containing
                              the text chunk, its source ArXiv ID, title, and relevance score.
    """
    logger.info(f"Executing knowledge_base_retrieval_tool with query='{query_text[:50]}...', top_k={top_k}, filters={metadata_filters}")
    # ... (code existant de knowledge_base_retrieval_tool, je le garde concis ici)
    if retrieval_engine_instance is None:
        logger.error("RetrievalEngine instance is not available for knowledge_base_retrieval_tool.")
        return [{"error": "RetrievalEngine not initialized."}]
    try:
        llama_filters_list = []
        if metadata_filters:
            for f in metadata_filters:
                if "key" in f and "value" in f: llama_filters_list.append({"key": f["key"], "value": f["value"]})
                else: logger.warning(f"Malformed metadata filter skipped: {f}")
        
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
        logger.error(f"Error in knowledge_base_retrieval_tool: {e}", exc_info=True)
        return [{"error": f"knowledge_base_retrieval_tool failed: {str(e)}"}]

# <<< NOUVEL OUTIL BASÉ SUR CREWAI >>>
@tool
def document_deep_dive_analysis_tool(
    document_id: str, 
    document_content: str, 
    research_focus: str
) -> str:
    """
    Performs an in-depth analysis of a single scientific document's text content using a specialized team of AI agents (CrewAI).
    Use this tool when a detailed, structured report focusing on specific aspects of a document is required.
    The analysis will cover key information extraction, section summaries (if applicable), and a critical analysis 
    (strengths, weaknesses, contributions) based on the provided research focus.

    Args:
        document_id (str): The unique identifier of the document (e.g., ArXiv ID like '2301.12345'). 
                           This is primarily for reference in the output report.
        document_content (str): The full text content of the document to be analyzed. 
                                This should be substantial enough for a meaningful analysis.
        research_focus (str): Specific questions, themes, or aspects the detailed analysis should concentrate on. 
                              Example: "Identify sim-to-real transfer techniques and their effectiveness."

    Returns:
        str: A structured analytical report about the document, generated by the CrewAI team.
             Returns an error message string if the analysis fails.
    """
    logger.info(f"Executing document_deep_dive_analysis_tool for doc_id='{document_id}', focus='{research_focus}'")
    if not document_content or not document_content.strip():
        logger.error("Document content for deep dive analysis is empty.")
        return "Error: Document content provided for deep dive analysis was empty or whitespace only."
    if not research_focus or not research_focus.strip():
        logger.warning("Research focus for deep dive analysis is empty. Analysis might be very generic.")
        # On pourrait décider de retourner une erreur ici ou de laisser la crew gérer un focus vide.
        # return "Error: Research focus must be provided for deep dive analysis."

    try:
        # Appel de la fonction qui exécute la CrewAI
        # Cette fonction est synchrone dans sa définition actuelle dans document_analysis_crew.py
        report = run_document_deep_dive_crew(
            document_id=document_id,
            document_content=document_content,
            research_focus=research_focus
        )
        logger.info(f"document_deep_dive_analysis_tool completed for doc_id='{document_id}'. Report length: {len(report)}")
        return report
    except ImportError as ie: # Au cas où crewai ne serait pas installé, bien que ce soit une dépendance maintenant
        logger.error(f"CrewAI related import error for deep_dive_tool: {ie}", exc_info=True)
        return f"Error: CrewAI components not available. {str(ie)}"
    except Exception as e:
        logger.error(f"Error in document_deep_dive_analysis_tool for doc_id='{document_id}': {e}", exc_info=True)
        return f"Error during deep dive analysis for document {document_id}: {str(e)}"


if __name__ == "__main__":
    from config.logging_config import setup_logging
    import json # Pour l'affichage des résultats d'outils
    setup_logging(level="DEBUG")

    logger.info("--- Testing Tool Definitions (including new CrewAI tool) ---")

    # ... (tests existants pour arxiv_search_tool et knowledge_base_retrieval_tool) ...
    logger.info("\n--- Test Direct de arxiv_search_tool ---")
    arxiv_results = arxiv_search_tool.invoke({"query": "explainable AI in robotics", "max_results": 1})
    print(f"ArXiv Search Tool Direct Result:\n{json.dumps(arxiv_results, indent=2)}\n")

    logger.info("\n--- Test Direct de knowledge_base_retrieval_tool ---")
    # Ce test nécessite que RetrievalEngine soit initialisé et que la DB soit peuplée
    if retrieval_engine_instance:
        kb_results = knowledge_base_retrieval_tool.invoke({"query_text": "robot path planning", "top_k": 1})
        print(f"Knowledge Base Retrieval Tool Direct Result:\n{json.dumps(kb_results, indent=2)}\n")
    else:
        print("Skipping knowledge_base_retrieval_tool direct test as RetrievalEngine instance is not available.\n")

    # --- Test du nouvel outil document_deep_dive_analysis_tool ---
    logger.info("\n--- Test Direct de document_deep_dive_analysis_tool ---")
    if not settings.OPENAI_API_KEY:
        logger.error("OPENAI_API_KEY not set. Cannot run document_deep_dive_analysis_tool test.")
    else:
        sample_doc_id_crew_test = "crew_test_doc_001"
        sample_doc_content_crew_test = """
        This is a sample document about advanced robotic learning. 
        It details a new algorithm called 'RoboLearn'.
        The methodology involves using a simulated environment and then transferring skills to a physical robot.
        Key results showed a 20% improvement in task completion time.
        However, a limitation is the high computational cost during the simulation phase.
        Future work aims to optimize this. This document also discusses ethical implications.
        """
        sample_research_focus_crew_test = "Extract methodology, results, limitations, and discuss ethical implications."

        print(f"Testing deep dive tool for doc: {sample_doc_id_crew_test}, focus: {sample_research_focus_crew_test}")
        
        # L'appel à l'outil est synchrone car run_document_deep_dive_crew l'est
        deep_dive_report = document_deep_dive_analysis_tool.invoke({
            "document_id": sample_doc_id_crew_test,
            "document_content": sample_doc_content_crew_test,
            "research_focus": sample_research_focus_crew_test
        })

        print("\n--- Deep Dive Analysis Report (from Tool Test) ---")
        print(deep_dive_report)
        print("--------------------------------------------------")
        # Vérifier si le rapport n'est pas un message d'erreur simple
        assert "Error:" not in deep_dive_report[:20], "Deep dive tool returned an error string."

    logger.info("\nTool definition test run finished.")