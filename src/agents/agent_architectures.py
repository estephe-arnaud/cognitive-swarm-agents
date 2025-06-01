# cognitive-swarm-agents/src/agents/agent_architectures.py
import logging
from typing import List, Optional, Sequence, TypeVar # Ajout de TypeVar

from langchain_core.language_models import BaseLanguageModel # Type de retour générique
from langchain_openai import ChatOpenAI
# Nouveaux imports pour les LLMs open-source
from langchain_community.llms import HuggingFaceHub # Pour l'API d'inférence Hugging Face
from langchain_community.chat_models import ChatOllama # Pour les modèles via Ollama

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import BaseTool
from langchain.agents import create_openai_tools_agent, AgentExecutor # create_openai_tools_agent fonctionne avec BaseLanguageModel

from config.settings import settings
from src.agents.tool_definitions import (
    knowledge_base_retrieval_tool,
    arxiv_search_tool,
    document_deep_dive_analysis_tool
)

logger = logging.getLogger(__name__)

# Définir un TypeVar pour plus de flexibilité si on voulait spécifier LLM vs ChatModel plus tard
# LLMType = TypeVar('LLMType', bound=BaseLanguageModel) # Optionnel pour l'instant

# --- Helper to get LLM (MODIFIÉ) ---
DEFAULT_LLM_TEMPERATURE = 0.0
SYNTHESIS_LLM_TEMPERATURE = 0.5 # Généralement plus élevé pour la créativité/synthèse

def get_llm(
    temperature: Optional[float] = None, # Permettre de ne pas spécifier pour prendre une valeur par défaut
    model_provider_override: Optional[str] = None, # Pour surcharger le provider au besoin
    model_name_override: Optional[str] = None # Pour surcharger le nom du modèle au besoin
) -> BaseLanguageModel: # MODIFIÉ: Type de retour générique
    """
    Initializes and returns a language model client based on global settings
    or overrides. Supports OpenAI, HuggingFace Inference API, and Ollama.
    """
    provider = model_provider_override or settings.DEFAULT_LLM_MODEL_PROVIDER
    provider = provider.lower()

    # Déterminer la température effective
    # Si la température n'est pas passée en argument, on utilise la température par défaut pour le LLM,
    # sauf si le provider est celui de la synthèse où on pourrait vouloir une température spécifique.
    # Pour l'instant, la température de synthèse est gérée au moment de l'appel à get_llm.
    effective_temperature = DEFAULT_LLM_TEMPERATURE if temperature is None else temperature

    logger.info(f"Initializing LLM for provider: '{provider}' with temperature: {effective_temperature}")

    if provider == "openai":
        if not settings.OPENAI_API_KEY:
            logger.error("OpenAI API key is not configured.")
            raise ValueError("OpenAI API key is missing.")
        openai_model_name = model_name_override or settings.DEFAULT_OPENAI_MODEL
        logger.info(f"Using OpenAI model: {openai_model_name}")
        return ChatOpenAI(
            model=openai_model_name,
            temperature=effective_temperature,
            api_key=settings.OPENAI_API_KEY
        )
    elif provider == "huggingface_api":
        if not settings.HUGGINGFACE_API_KEY:
            logger.error("HuggingFace API key is not configured for Inference API.")
            raise ValueError("HuggingFace API key is missing.")
        hf_repo_id = model_name_override or settings.HUGGINGFACE_REPO_ID
        if not hf_repo_id:
            logger.error("HuggingFace Repository ID is not configured.")
            raise ValueError("HuggingFace Repository ID is missing.")
        logger.info(f"Using HuggingFace Inference API model: {hf_repo_id}")
        return HuggingFaceHub(
            repo_id=hf_repo_id,
            huggingfacehub_api_token=settings.HUGGINGFACE_API_KEY,
            model_kwargs={
                "temperature": effective_temperature,
                # "max_length": 2048, # Certains modèles peuvent nécessiter max_new_tokens
                "max_new_tokens": 1024, # Exemple, à ajuster
                # D'autres paramètres spécifiques au modèle peuvent être ajoutés ici
            }
        )
    elif provider == "ollama":
        if not settings.OLLAMA_BASE_URL:
            logger.error("Ollama base URL is not configured.")
            raise ValueError("Ollama base URL is missing.")
        ollama_model_name = model_name_override or settings.OLLAMA_MODEL_NAME
        if not ollama_model_name:
            logger.error("Ollama model name is not configured.")
            raise ValueError("Ollama model name is missing.")
        logger.info(f"Using Ollama model: {ollama_model_name} via {settings.OLLAMA_BASE_URL}")
        return ChatOllama(
            model=ollama_model_name,
            base_url=settings.OLLAMA_BASE_URL,
            temperature=effective_temperature
            # Vous pouvez ajouter d'autres options ici, ex: num_ctx, top_k, top_p
        )
    else:
        logger.error(f"Unsupported LLM provider: {provider}")
        raise ValueError(f"Unsupported LLM provider: {provider}")

# --- Agent 1: Research Planner Agent ---
RESEARCH_PLANNER_SYSTEM_PROMPT = """You are a Research Planner Agent.
Your role is to take a complex user query or research topic related to scientific literature (specifically on Reinforcement Learning for Robotics)
and break it down into a structured research plan. This plan will be executed by other specialist agents.
Your plan should consist of:
1.  **Key Questions:** A list of specific questions that need to be answered to address the user's query.
2.  **Information Sources:** Identify potential information sources (e.g., our internal knowledge base of ArXiv papers, new ArXiv searches).
3.  **Search Queries (if applicable):** Suggest specific search queries for ArXiv or the knowledge base.
4.  **Analysis Steps:** Outline what kind of analysis should be performed on the retrieved information.
5.  **Final Output Structure:** Briefly describe what the final report or answer should look like.
You do not have tools to search or analyze documents directly. Your output is solely the research plan.
Provide the plan in a clear, actionable, and preferably structured format (e.g., markdown).
Respond ONLY with the research plan based on the user's query.
"""
def create_research_planner_agent(llm: Optional[BaseLanguageModel] = None) -> AgentExecutor: # MODIFIÉ: Type hint pour llm
    if llm is None:
        llm = get_llm() # Utilise la température par défaut de get_llm (DEFAULT_LLM_TEMPERATURE)
    prompt = ChatPromptTemplate.from_messages([
        ("system", RESEARCH_PLANNER_SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="messages"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    # create_openai_tools_agent est compatible avec BaseLanguageModel si le modèle supporte les outils/fonctions.
    # Pour les LLMs open-source qui ne supportent pas nativement les outils OpenAI,
    # cette partie pourrait nécessiter des ajustements (ex: utiliser un autre type d'agent comme ReAct,
    # ou un modèle LLM qui a été fine-tuné pour le tool/function calling et dont l'intégration LangChain le supporte).
    # Pour l'instant, on garde la structure, en supposant que le LLM choisi (ou son intégration)
    # gérera les appels d'outils ou que les outils ne sont pas critiques pour cet agent spécifique (Planner n'a pas d'outils).
    agent = create_openai_tools_agent(llm, tools=[], prompt=prompt) # Planner n'a pas d'outils, donc moins de risque ici.
    agent_executor = AgentExecutor(agent=agent, tools=[], verbose=settings.DEBUG, handle_parsing_errors=True)
    logger.info("Research Planner Agent created.")
    return agent_executor

# --- Agent 2: Document Analysis Agent (MODIFIÉ pour le type hint de llm) ---
DOCUMENT_ANALYSIS_SYSTEM_PROMPT_V2 = """You are a Document Analysis Agent.
Your primary task is to analyze scientific documents (chunks of ArXiv papers on Reinforcement Learning for Robotics)
retrieved from a knowledge base to answer specific questions or extract key information.

You have access to the following tools:
- `knowledge_base_retrieval_tool`: Use this tool to fetch relevant chunks of text from the ingested knowledge base based on a query or topic. This is good for targeted information retrieval or getting initial context.
- `document_deep_dive_analysis_tool`: Use this tool when a comprehensive, structured, and in-depth analysis of a *single, specific document's content* is required, focusing on particular aspects outlined in a 'research_focus'. This tool internally uses a specialized team of AI agents (CrewAI) to produce a detailed report.
    - To use `document_deep_dive_analysis_tool`, you **must** provide:
        1. `document_id` (str): The ArXiv ID of the document (e.g., '2301.12345').
        2. `document_content` (str): The full text content (or a substantial concatenation of relevant chunks) of the *single* document to be analyzed. You might need to use `knowledge_base_retrieval_tool` first to gather all chunks for a specific document if you only have its ID.
        3. `research_focus` (str): Specific questions or themes the deep dive analysis should concentrate on (e.g., "Identify methodology, key results, and limitations regarding sim-to-real transfer techniques.").

Instructions:
1.  Carefully understand the question, research plan, or analysis task given to you.
2.  **Decision Point**:
    * If the task is to retrieve specific facts, answer a direct question using a few relevant chunks, or get initial context from multiple documents, use `knowledge_base_retrieval_tool`.
    * If the task explicitly asks for a "deep dive", "detailed report", "structured analysis" of a *single known document* for which you can gather sufficient content, and a `research_focus` is clear or can be derived, then use `document_deep_dive_analysis_tool`. Ensure you have gathered enough `document_content` for the specified `document_id` before calling this tool.
3.  After using a tool (or if no tool is needed for a simple summarization of provided text), analyze the retrieved text chunks or the report from the deep dive tool to extract the required information.
4.  Synthesize information if needed to provide a comprehensive answer based on your findings.
5.  If retrieved information (from either tool) is insufficient, state that clearly. Do not invent information.
6.  Provide concise and factual answers. Cite source ArXiv IDs if possible (available in metadata from tools).

Example task requiring deep dive: "The research plan indicates paper 'arxiv:2301.12345' is highly relevant. Perform a deep dive analysis on it focusing on its novelty and limitations."
    - Step 1 (thought): I need to get the content for '2301.12345'.
    - Step 2 (action): Use `knowledge_base_retrieval_tool` to get all chunks for '2301.12345'.
    - Step 3 (thought): Concatenate chunks to form `document_content`. The `research_focus` is "novelty and limitations".
    - Step 4 (action): Call `document_deep_dive_analysis_tool` with `document_id='2301.12345'`, the concatenated `document_content`, and `research_focus="novelty and limitations"`.
    - Step 5 (thought): The tool will return a detailed report. I will present this report as my output.
"""

def create_document_analysis_agent(llm: Optional[BaseLanguageModel] = None) -> AgentExecutor: # MODIFIÉ: Type hint pour llm
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

    # IMPORTANT: create_openai_tools_agent est optimisé pour les modèles OpenAI.
    # Son efficacité avec d'autres modèles dépendra de leur support pour un format d'appel d'outil similaire
    # ou de la capacité de l'intégration LangChain (ex: ChatHuggingFace, ChatOllama) à traduire
    # les demandes d'outils en quelque chose que le modèle open-source comprend.
    # Si le LLM open-source choisi ne supporte pas bien cela, il faudra envisager:
    # 1. Un LLM open-source qui supporte bien les "tool calls" via son intégration LangChain (ex: avec .bind_tools()).
    # 2. Changer le type d'agent (ex: ReAct). C'est une modification plus profonde.
    # 3. Simplifier/adapter les outils ou le prompt pour que le LLM puisse les appeler via du texte structuré.
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

# --- Agent 3: ArXiv Search Agent (MODIFIÉ pour le type hint de llm) ---
ARXIV_SEARCH_SYSTEM_PROMPT = """You are an ArXiv Search Agent.
Your sole responsibility is to find relevant scientific papers on ArXiv based on given search queries.
You have access to the `arxiv_search_tool`.
Instructions:
1.  Receive a specific search query and parameters (like max results, sort order).
2.  Use the `arxiv_search_tool` to perform the search on arxiv.org.
3.  Return the search results (list of paper summaries) as provided by the tool. Do not modify or analyze them.
4.  If the tool fails or returns an error, report that error.
"""
def create_arxiv_search_agent(llm: Optional[BaseLanguageModel] = None) -> AgentExecutor: # MODIFIÉ: Type hint pour llm
    if llm is None:
        llm = get_llm()
    tools: List[BaseTool] = [arxiv_search_tool]
    prompt = ChatPromptTemplate.from_messages([
        ("system", ARXIV_SEARCH_SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="messages"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    agent = create_openai_tools_agent(llm, tools=tools, prompt=prompt) # Même remarque que pour DocumentAnalysisAgent sur la compatibilité des outils
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=settings.DEBUG, handle_parsing_errors=True)
    logger.info(f"ArXiv Search Agent created with tools: {[tool.name for tool in tools]}")
    return agent_executor

# --- Agent 4: Synthesis Agent (MODIFIÉ pour le type hint de llm) ---
SYNTHESIS_AGENT_SYSTEM_PROMPT = """You are a Synthesis Agent.
Your role is to take analyzed information, research findings, and extracted data from various sources
(provided as context in the conversation or from previous agent steps) and synthesize it into a
coherent and well-structured final output (e.g., a report, an answer to a complex question).
You do not have tools to search or retrieve new information. You work solely with the information provided to you.
Instructions:
1.  Carefully review all the provided information.
2.  Understand the overall goal or the main question that needs to be answered.
3.  Structure your output logically.
4.  Write clearly, concisely, and factually. Attribute information to sources if available.
5.  If provided information is contradictory or insufficient, highlight these limitations.
"""
def create_synthesis_agent(llm: Optional[BaseLanguageModel] = None) -> AgentExecutor: # MODIFIÉ: Type hint pour llm
    if llm is None:
        llm = get_llm(temperature=SYNTHESIS_LLM_TEMPERATURE) # Température spécifique pour la synthèse
    tools_for_synthesis: List[BaseTool] = [] # Le synthétiseur n'a généralement pas d'outils de recherche
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYNTHESIS_AGENT_SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="messages"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    agent = create_openai_tools_agent(llm, tools=tools_for_synthesis, prompt=prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools_for_synthesis, verbose=settings.DEBUG, handle_parsing_errors=True)
    logger.info("Synthesis Agent created.")
    return agent_executor


if __name__ == "__main__":
    from config.logging_config import setup_logging
    setup_logging(level="DEBUG" if settings.DEBUG else "INFO")

    logger.info("--- Testing Agent Creation with potentially different LLM providers ---")

    # Pour tester, vous pouvez surcharger les settings temporairement ou via .env
    # Exemple: settings.DEFAULT_LLM_MODEL_PROVIDER = "ollama"
    # settings.OLLAMA_MODEL_NAME = "mistral" # Assurez-vous qu'Ollama sert ce modèle

    try:
        logger.info(f"Attempting to get LLM with provider: {settings.DEFAULT_LLM_MODEL_PROVIDER}")
        # Test get_llm
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

        synthesizer = create_synthesis_agent(get_llm(temperature=SYNTHESIS_LLM_TEMPERATURE)) # Test avec une température différente pour la synthèse
        logger.info(f"Synthesis agent created with LLM: {type(synthesizer.agent.llm_chain.llm)}") # type: ignore

        logger.info("All agents created successfully with the configured LLM provider.")

    except ValueError as ve:
        logger.error(f"ValueError during agent creation tests: {ve}. Check API keys and model configurations.")
    except Exception as e:
        logger.error(f"An unexpected error occurred during agent creation tests: {e}", exc_info=True)

    logger.info("Agent architectures adaptation test run finished.")