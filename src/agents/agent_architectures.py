"""
Agent Architectures Module

This module defines the core agent architectures used in the research assistant system.
Each agent is specialized for a specific task in the research workflow:
- Research Planner: Breaks down complex queries into structured research plans
- Document Analysis: Analyzes scientific documents using specialized tools
- ArXiv Search: Performs targeted searches on ArXiv
- Synthesis: Combines and structures information from various sources

The module uses LangChain's agent framework and integrates with custom tools for
knowledge base retrieval, document analysis, and ArXiv searching.
"""

import logging
from typing import List, Optional, Sequence

from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import BaseTool
from langchain.agents import create_openai_tools_agent, AgentExecutor

from config.settings import settings
# Importation des outils (inchangée)
from src.agents.tool_definitions import (
    knowledge_base_retrieval_tool,
    arxiv_search_tool,
    document_deep_dive_analysis_tool
)
# MODIFICATION: Importer get_llm et les constantes de température depuis le nouveau factory
from src.llm_services.llm_factory import get_llm, SYNTHESIS_LLM_TEMPERATURE
# DEFAULT_LLM_TEMPERATURE est utilisé par défaut dans get_llm si temperature n'est pas spécifiée,
# donc pas besoin de l'importer ici à moins d'un usage explicite.

logger = logging.getLogger(__name__)

# La définition de get_llm et les constantes DEFAULT_LLM_TEMPERATURE, SYNTHESIS_LLM_TEMPERATURE
# ont été déplacées vers src.llm_services.llm_factory.py

# --- Agent 1: Research Planner Agent ---
RESEARCH_PLANNER_SYSTEM_PROMPT = """You are a Research Planner Agent.
Your role is to take a complex user query or research topic related to scientific literature, 
particularly in the fields of Machine Learning, Artificial Intelligence, and related domains 
(such as robotics, computer vision, etc., depending on the query).

Your goal is to break the user's query down into a structured research plan. 
This plan will be executed by other specialist agents.

The plan should consist of:
1. Key Questions: A list of specific questions that need to be answered to comprehensively 
   address the user's query.
2. Information Sources: Identify potential information sources (e.g., our internal knowledge 
   base of ArXiv papers, new ArXiv searches, specific journals, conference proceedings, or 
   general academic search engines like Google Scholar if appropriate for the query's scope).
3. Search Queries: Suggest specific, effective search queries for the identified sources. 
   Include keywords from the user's query and relevant synonyms or related concepts. 
   Specify if a date range for publications is relevant (e.g., recent trends).
4. Analysis Steps: Outline what kind of analysis should be performed on the retrieved 
   information to answer the key questions.
5. Final Output Structure: Briefly describe what the final report or answer should look like, 
   ensuring it directly addresses all parts of the user's original query.

You do not have tools to search or analyze documents directly. Your output is solely the research plan.
Provide the plan in a clear, actionable, and preferably structured format (e.g., markdown).
Ensure your plan is tailored to the specifics of the user's query, including any specified output language.
Respond ONLY with the research plan based on the user's query.
"""

def create_research_planner_agent(llm: Optional[BaseLanguageModel] = None) -> AgentExecutor:
    """
    Creates a Research Planner Agent that breaks down complex queries into structured research plans.
    
    Args:
        llm: Optional language model to use. If None, uses the default model from llm_factory.
    
    Returns:
        AgentExecutor: Configured agent for research planning tasks.
    """
    if llm is None:
        llm = get_llm()
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", RESEARCH_PLANNER_SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="messages"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    agent = create_openai_tools_agent(llm, tools=[], prompt=prompt)
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=[], 
        verbose=settings.DEBUG, 
        handle_parsing_errors=True
    )
    
    logger.info("Research Planner Agent created successfully")
    return agent_executor

# --- Agent 2: Document Analysis Agent ---
DOCUMENT_ANALYSIS_SYSTEM_PROMPT_V2 = """You are a Document Analysis Agent.
Your primary task is to analyze scientific documents (chunks of ArXiv papers)
retrieved from a knowledge base to answer specific questions or extract key information.

Available Tools:
1. knowledge_base_retrieval_tool:
   - Purpose: Fetch relevant text chunks from the knowledge base
   - Use for: Targeted information retrieval or initial context gathering

2. document_deep_dive_analysis_tool:
   - Purpose: Comprehensive analysis of a single document
   - Required parameters:
     * document_id: ArXiv ID (e.g., '2301.12345')
     * document_content: Full text content of the document
     * research_focus: Specific questions or themes to analyze

Workflow:
1. Understand the question or analysis task
2. Choose appropriate tool based on task requirements:
   - Use knowledge_base_retrieval_tool for:
     * Specific fact retrieval
     * Direct question answering
     * Initial context gathering
   - Use document_deep_dive_analysis_tool for:
     * Deep dive analysis
     * Detailed reports
     * Structured analysis of single documents
3. Analyze retrieved information
4. Synthesize findings
5. Provide clear, factual answers with source citations

Note: Always state if retrieved information is insufficient. Do not invent information.
"""

def create_document_analysis_agent(llm: Optional[BaseLanguageModel] = None) -> AgentExecutor:
    """
    Creates a Document Analysis Agent for analyzing scientific documents.
    
    Args:
        llm: Optional language model to use. If None, uses the default model from llm_factory.
    
    Returns:
        AgentExecutor: Configured agent for document analysis tasks.
    """
    if llm is None:
        llm = get_llm()

    tools: List[BaseTool] = [
        knowledge_base_retrieval_tool,
        document_deep_dive_analysis_tool
    ]
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", DOCUMENT_ANALYSIS_SYSTEM_PROMPT_V2),
        MessagesPlaceholder(variable_name="messages"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    agent = create_openai_tools_agent(llm, tools=tools, prompt=prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=settings.DEBUG,
        handle_parsing_errors=True,
        max_iterations=10
    )
    
    logger.info(f"Document Analysis Agent created with tools: {[tool.name for tool in tools]}")
    return agent_executor

# --- Agent 3: ArXiv Search Agent ---
ARXIV_SEARCH_SYSTEM_PROMPT = """You are an ArXiv Search Agent. Your role is to find relevant scientific papers based on the given search query.

You must ALWAYS use the arxiv_search_tool to perform searches. The tool requires the following parameters:
- query: The search query string
- max_results: Maximum number of results to return (default: 3)
- sort_by: How to sort results (default: "relevance")

When presenting search results, format them clearly with:
1. Title
2. Authors
3. Summary
4. PDF URL

If no results are found, clearly state that. If there's an error, report it.

Example usage:
Input: "What are the latest advancements in machine learning?"
Action: Use arxiv_search_tool with query="machine learning" and max_results=3

Remember to always use the arxiv_search_tool and format the results clearly."""

def create_arxiv_search_agent(llm: Optional[BaseLanguageModel] = None) -> AgentExecutor:
    """
    Creates an ArXiv Search Agent for finding relevant scientific papers.
    
    Args:
        llm: Optional language model to use. If None, uses the default model from llm_factory.
    
    Returns:
        AgentExecutor: Configured agent for ArXiv searching tasks.
    """
    if llm is None:
        llm = get_llm()
    
    tools: List[BaseTool] = [arxiv_search_tool]
    prompt = ChatPromptTemplate.from_messages([
        ("system", ARXIV_SEARCH_SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="messages"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    agent = create_openai_tools_agent(llm, tools=tools, prompt=prompt)
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=settings.DEBUG, 
        handle_parsing_errors=True
    )
    
    logger.info(f"ArXiv Search Agent created with tools: {[tool.name for tool in tools]}")
    return agent_executor

# --- Agent 4: Synthesis Agent ---
SYNTHESIS_AGENT_SYSTEM_PROMPT = """You are a Synthesis Agent.
Your role is to synthesize analyzed information, research findings, and extracted data 
from various sources into coherent, well-structured outputs.

Available Information:
- Initial user query
- Research findings
- Analyzed data
- Previous agent outputs

Instructions:
1. Review all provided information
2. Understand the main goal/question
3. Structure output logically
4. Write clearly and factually
5. Attribute information to sources
6. Match output language to user's query
7. Highlight any contradictions or gaps

Note: You work only with provided information. No search or retrieval tools available.
"""

def create_synthesis_agent(llm: Optional[BaseLanguageModel] = None) -> AgentExecutor:
    """
    Creates a Synthesis Agent for combining and structuring information from various sources.
    
    Args:
        llm: Optional language model to use. If None, uses the synthesis-optimized model 
             with SYNTHESIS_LLM_TEMPERATURE.
    
    Returns:
        AgentExecutor: Configured agent for synthesis tasks.
    """
    if llm is None:
        llm = get_llm(temperature=SYNTHESIS_LLM_TEMPERATURE)
    
    tools_for_synthesis: List[BaseTool] = []
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYNTHESIS_AGENT_SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="messages"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    agent = create_openai_tools_agent(llm, tools=tools_for_synthesis, prompt=prompt)
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools_for_synthesis, 
        verbose=settings.DEBUG, 
        handle_parsing_errors=True
    )
    
    logger.info("Synthesis Agent created successfully")
    return agent_executor


if __name__ == "__main__":
    from config.logging_config import setup_logging # Déplacé ici pour être utilisé seulement si le script est exécuté
    setup_logging(level="DEBUG" if settings.DEBUG else "INFO")

    logger.info("--- Testing Agent Creation with potentially different LLM providers (using llm_factory) ---")

    try:
        logger.info(f"Attempting to get LLM with provider: {settings.DEFAULT_LLM_MODEL_PROVIDER} using llm_factory.get_llm")
        # Test get_llm (maintenant importé)
        llm_instance = get_llm()
        logger.info(f"Successfully instantiated LLM: {type(llm_instance)}")

        # Test de création de chaque agent
        planner = create_research_planner_agent(llm_instance)
        logger.info(f"Planner agent created with LLM: {type(planner.agent.llm_chain.llm)}") # type: ignore

        doc_analyzer = create_document_analysis_agent(llm_instance)
        logger.info(f"Document Analysis agent created with LLM: {type(doc_analyzer.agent.llm_chain.llm)}") # type: ignore
        assert "document_deep_dive_analysis_tool" in [tool.name for tool in doc_analyzer.tools]

        arxiv_searcher = create_arxiv_search_agent(llm_instance)
        logger.info(f"ArXiv Search agent created with LLM: {type(arxiv_searcher.agent.llm_chain.llm)}") # type: ignore

        # La fonction create_synthesis_agent utilise SYNTHESIS_LLM_TEMPERATURE par défaut si llm est None.
        # Si on passe un llm_instance, il l'utilise. Pour tester SYNTHESIS_LLM_TEMPERATURE,
        # on peut soit appeler create_synthesis_agent() sans argument,
        # soit créer un LLM spécifique avec cette température.
        # L'approche actuelle dans la fonction est de toute façon d'appliquer SYNTHESIS_LLM_TEMPERATURE si llm=None.
        # Pour tester explicitement que la constante importée est utilisée par get_llm quand appelé par create_synthesis_agent:
        synthesis_llm_for_test = get_llm(temperature=SYNTHESIS_LLM_TEMPERATURE)
        synthesizer = create_synthesis_agent(synthesis_llm_for_test)
        logger.info(f"Synthesis agent created with LLM: {type(synthesizer.agent.llm_chain.llm)} and explicit temp.") # type: ignore

        logger.info("All agents created successfully with the configured LLM provider via llm_factory.")

    except ValueError as ve:
        logger.error(f"ValueError during agent creation tests: {ve}. Check API keys and model configurations.")
    except Exception as e:
        logger.error(f"An unexpected error occurred during agent creation tests: {e}", exc_info=True)

    logger.info("Agent architectures adaptation test run finished.")