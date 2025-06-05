"""
Agent Architectures Module

This module defines the core agent architectures used in the research assistant system.
Each agent is specialized for a specific task in the research workflow, created
using a factory pattern that combines a specific language model, a structured
prompt, and a set of tools.

- Research Planner: Breaks down complex queries into structured research plans.
- ArXiv Search: Performs targeted searches on ArXiv.
- Document Analysis: Analyzes scientific documents, with the ability to perform
  deep dives into full PDF contents.
- Synthesis: Combines and structures information into a final, polished report.
"""

import logging
from typing import List, Optional

from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import BaseTool
from langchain.agents import create_openai_tools_agent, AgentExecutor

from config.settings import settings
from src.agents.tool_definitions import (
    knowledge_base_retrieval_tool,
    arxiv_search_tool,
    document_deep_dive_analysis_tool,
)
from src.llm_services.llm_factory import get_llm, SYNTHESIS_LLM_TEMPERATURE

logger = logging.getLogger(__name__)


# --- Agent 1: Research Planner Agent ---

RESEARCH_PLANNER_SYSTEM_PROMPT = """
**Role:** You are a meticulous and strategic Research Planner.

**Goal:** To transform a user's research query into a structured, actionable plan
that other specialized agents will execute. You do not perform any searches or
analysis yourself; your sole output is the plan.

**User Query Context:** The user is asking about a scientific topic, likely in the
domain of Machine Learning or Artificial Intelligence.

### Directives:
1.  **Deconstruct the Query:** Break down the user's request into fundamental questions.
2.  **Identify Sources:** Pinpoint the best places to find answers (e.g., new ArXiv
    searches, specific journals, our internal knowledge base).
3.  **Formulate Search Queries:** Create specific, effective search queries for each
    source. If the user's query implies recent trends, suggest a date range.
    **Crucially, if you recommend an ArXiv search, provide a clear, single query
    for it like this: `arxiv: "your query here"`**.
4.  **Define Analysis Steps:** Briefly outline how the retrieved information should be
    analyzed.
5.  **Structure the Output:** Present the plan in a clear, easy-to-read format
    (like Markdown).

**Constraint:** You MUST ONLY output the research plan. Do not write any
introductions or conversational text. Your entire response should be the plan itself.
"""


def create_research_planner_agent(
    llm: Optional[BaseLanguageModel] = None,
) -> AgentExecutor:
    """
    Creates the Research Planner Agent.

    This agent is tool-less. Its only job is to receive a user query and
    generate a structured research plan based on its system prompt.

    Args:
        llm: An optional language model. If None, the default is used.

    Returns:
        An AgentExecutor configured for research planning.
    """
    if llm is None:
        llm = get_llm()

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", RESEARCH_PLANNER_SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="messages"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    # This agent has no tools, as its only job is to plan.
    agent = create_openai_tools_agent(llm, tools=[], prompt=prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=[],
        verbose=settings.DEBUG,
        handle_parsing_errors=True,
        name="ResearchPlannerAgent",
    )

    logger.info("Research Planner Agent created successfully.")
    return agent_executor


# --- Agent 2: ArXiv Search Agent ---

ARXIV_SEARCH_SYSTEM_PROMPT = """
**Role:** You are a focused ArXiv Search Specialist.

**Task:** Your one and only task is to take a user's query and use the provided
`arxiv_search` tool to find relevant scientific papers.

### Instructions:
1.  Receive the search query.
2.  Immediately call the `arxiv_search` tool with the query.
3.  Return the direct, raw output from the tool.

**Constraint:** Do not add any commentary, analysis, or formatting. Your job is
to execute the search and nothing else.
"""


def create_arxiv_search_agent(
    llm: Optional[BaseLanguageModel] = None,
) -> AgentExecutor:
    """
    Creates the ArXiv Search Agent.

    This is a simple, tool-focused agent. It is given a query and is expected
    to use the `arxiv_search_tool` to find papers.

    Args:
        llm: An optional language model. If None, the default is used.

    Returns:
        An AgentExecutor configured for ArXiv searching.
    """
    if llm is None:
        llm = get_llm()

    tools: List[BaseTool] = [arxiv_search_tool]
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", ARXIV_SEARCH_SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="messages"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    agent = create_openai_tools_agent(llm, tools=tools, prompt=prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=settings.DEBUG,
        handle_parsing_errors=True,
        name="ArxivSearchAgent",
    )

    logger.info(f"ArXiv Search Agent created with tool: {tools[0].name}")
    return agent_executor


# --- Agent 3: Document Analysis Agent ---

DOCUMENT_ANALYSIS_SYSTEM_PROMPT = """
**Role:** You are a Deep Research Analyst.

**Goal:** To conduct a comprehensive analysis of scientific topics using the
provided information and tools. You will be given a list of papers (summaries
and links) and are expected to produce a detailed analysis.

### Your Tools:
1.  **`knowledge_base_retrieval_tool`**: Use this for quick, targeted information
    retrieval from our internal document collection.
2.  **`document_deep_dive_analysis_tool`**: This is your primary power tool. Use
    it when a paper's summary seems particularly relevant or when you need more
    detail than the summary provides. It reads the *entire PDF* and gives you a
    thorough analysis.

### Workflow:
1.  **Assess the Material:** Start by reviewing the list of paper titles and
    summaries provided in the prompt.
2.  **Strategize Your Analysis:** Identify the 1-3 most promising papers that are
    key to answering the research question.
3.  **Conduct Deep Dives:** For each key paper you identified, use the
    `document_deep_dive_analysis_tool`. This is critical for a high-quality result.
4.  **Synthesize:** Combine the initial summaries with the rich details from your
    deep dives.
5.  **Formulate the Final Analysis:** Structure all your findings into a single,
    comprehensive answer that addresses the user's original request, covering
    key findings, trends, and future directions.

**Constraint:** Your final output should be the complete analysis, not just the
tool outputs. You must synthesize the information into a coherent report.
"""


def create_document_analysis_agent(
    llm: Optional[BaseLanguageModel] = None,
) -> AgentExecutor:
    """
    Creates the Document Analysis Agent.

    This is a sophisticated agent designed to analyze research papers. It is
    equipped with a powerful `document_deep_dive_analysis_tool` that allows it
    to read the full content of PDFs, enabling a much deeper level of analysis
    than just reading summaries.

    Args:
        llm: An optional language model. If None, the default is used.

    Returns:
        An AgentExecutor configured for in-depth document analysis.
    """
    if llm is None:
        llm = get_llm()

    tools: List[BaseTool] = [
        knowledge_base_retrieval_tool,
        document_deep_dive_analysis_tool,
    ]

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", DOCUMENT_ANALYSIS_SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="messages"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    agent = create_openai_tools_agent(llm, tools=tools, prompt=prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=settings.DEBUG,
        handle_parsing_errors=True,
        max_iterations=10,
        name="DocumentAnalysisAgent",
    )

    logger.info(
        f"Document Analysis Agent created with tools: {[tool.name for tool in tools]}"
    )
    return agent_executor


# --- Agent 4: Synthesis Agent ---

SYNTHESIS_AGENT_SYSTEM_PROMPT = """
**Role:** You are a Senior Research Editor.

**Goal:** Your purpose is to transform a detailed, technical analysis into a
final, polished, and easy-to-understand report for a user. You do not conduct
new research or use tools.

### Instructions:
1.  **Review the Content:** Carefully read the entire analysis provided to you.
2.  **Identify Key Insights:** Extract the most important findings, trends, and conclusions.
3.  **Structure the Report:** Organize the information logically. The specific
    structure (e.g., Executive Summary, Key Findings, etc.) will be requested
    in the prompt. Your job is to populate that structure.
4.  **Clarify and Refine:** Rewrite complex ideas in clear, concise language
    without sacrificing accuracy.
5.  **Ensure Coherence:** Create a smooth narrative that connects all parts of the
    analysis.

**Constraint:** You work ONLY with the information provided in the prompt. Do not
add external information or personal opinions.
"""


def create_synthesis_agent(
    llm: Optional[BaseLanguageModel] = None,
) -> AgentExecutor:
    """
    Creates the Synthesis Agent.

    This agent is tool-less. It is designed to take a large body of structured
    text (the analysis from the previous step) and reformat it into a final,
    polished report according to the structure requested in the prompt. It uses
    a higher-temperature LLM to encourage more creative and fluent writing.

    Args:
        llm: An optional language model. If None, a specific synthesis model is used.

    Returns:
        An AgentExecutor configured for synthesis and reporting.
    """
    if llm is None:
        # Use a model with higher temperature for more creative/fluent synthesis
        llm = get_llm(temperature=SYNTHESIS_LLM_TEMPERATURE)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYNTHESIS_AGENT_SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="messages"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    agent = create_openai_tools_agent(llm, tools=[], prompt=prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=[],
        verbose=settings.DEBUG,
        handle_parsing_errors=True,
        name="SynthesisAgent",
    )

    logger.info("Synthesis Agent created successfully.")
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