# cognitive-swarm-agents/src/agents/crewai_teams/document_analysis_crew.py
import os
import logging
from typing import List, Dict, Any, Optional

from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI # Utilisé par CrewAI par défaut si aucun LLM n'est passé aux agents

# Importer notre helper pour obtenir un LLM configuré si nous voulons être consistants
# ou si nous voulons utiliser un LLM non-OpenAI supporté par LangChain pour CrewAI.
# Pour l'instant, CrewAI s'intègre bien avec les LLM LangChain, y compris ChatOpenAI.
from src.agents.agent_architectures import get_llm # Réutilise notre configuration LLM
from config.settings import settings

logger = logging.getLogger(__name__)

class DocumentAnalysisCrew:
    """
    Manages a CrewAI team dedicated to performing a deep dive analysis of a single document.
    """
    def __init__(self, document_id: str, document_content: str, research_focus: str):
        """
        Args:
            document_id (str): Identifier for the document being analyzed (e.g., ArXiv ID).
            document_content (str): The full text content of the document.
            research_focus (str): Specific aspects or questions the analysis should focus on,
                                  derived from the research plan or user query.
        """
        self.document_id = document_id
        self.document_content = document_content
        self.research_focus = research_focus
        
        # Utiliser notre fonction get_llm pour la consistance, ou laisser CrewAI utiliser son défaut (souvent gpt-4 via API key)
        # Pour ce projet, utilisons notre get_llm pour s'assurer que les settings sont respectés.
        # CrewAI agents acceptent un `llm` passé à leur constructeur.
        try:
            self.llm = get_llm(temperature=0.2) # Température légèrement augmentée pour analyse/résumé
        except ValueError as e:
            logger.error(f"Failed to get LLM for CrewAI: {e}. Ensure LLM provider (e.g., OpenAI) is configured.")
            # Fallback ou lever une exception plus spécifique si le LLM est critique.
            # Pour CrewAI, si self.llm est None, il essaiera d'utiliser son propre LLM par défaut.
            # On peut aussi le configurer globalement : from langchain_openai import ChatOpenAI; os.environ["OPENAI_MODEL_NAME"] = 'gpt-4'
            # Mais passer explicitement aux agents est mieux.
            # Si get_llm échoue (ex: OPENAI_API_KEY manquante), on ne pourra pas créer les agents avec ce LLM.
            # On peut laisser CrewAI tenter d'utiliser son LLM par défaut si le nôtre n'est pas dispo.
            logger.warning("LLM from get_llm() failed. CrewAI agents will attempt to use their default LLM if OPENAI_API_KEY is set globally.")
            self.llm = None # Permettra à CrewAI d'utiliser son LLM par défaut si OPENAI_API_KEY est dans l'env.


    def _create_agents(self) -> List[Agent]:
        """Defines the agents for the document analysis crew."""

        info_extractor = Agent(
            role='Expert Information Extractor for Scientific Papers',
            goal=f"Meticulously extract key structured information (such as main methodology, dataset used, key quantitative results, and explicitly stated limitations) from the provided scientific paper text related to '{self.research_focus}'. Focus solely on information present in the text.",
            backstory="You are an AI assistant with deep expertise in parsing scientific literature and identifying core components of research papers. You are extremely precise and only extract information that is explicitly stated.",
            verbose=True,
            allow_delegation=False, # Cet agent ne délègue pas
            llm=self.llm # Utiliser notre LLM configuré
        )

        section_summarizer = Agent(
            role='Scientific Section Summarizer',
            goal=f"Provide concise summaries for the main sections (e.g., Abstract, Introduction, Methodology, Results, Conclusion) of the provided scientific paper text, keeping the research focus '{self.research_focus}' in mind. If distinct sections are not obvious, provide a general summary.",
            backstory="You are an AI skilled in academic writing and can distill the essence of complex scientific sections into clear and brief summaries.",
            verbose=True,
            allow_delegation=False,
            llm=self.llm
        )

        critical_analyst = Agent(
            role='Critical Analyst of Scientific Research',
            goal=f"Critically analyze the provided scientific paper based on extracted information and summaries, focusing on '{self.research_focus}'. Identify key strengths, potential weaknesses or limitations, and novel contributions or insights. Do not invent information not supported by the provided text context.",
            backstory="You are an experienced peer reviewer AI with a keen eye for scientific rigor, innovation, and potential areas of improvement in research papers.",
            verbose=True,
            allow_delegation=False, # Pourrait déléguer à l'extracteur si besoin de plus d'infos spécifiques
            llm=self.llm
        )

        report_compiler = Agent(
            role='Lead Research Report Compiler',
            goal=f"Compile a comprehensive and structured analytical report for the scientific paper (ID: {self.document_id}) focusing on '{self.research_focus}'. Integrate the extracted key information, section summaries, and critical analysis from other team members into a coherent final document.",
            backstory="You are a senior AI research lead responsible for synthesizing detailed analyses from your team into a final, publishable-quality report. Your work is structured, clear, and directly addresses the research focus.",
            verbose=True,
            allow_delegation=True, # Peut déléguer pour affiner des sections si besoin
            llm=self.llm
        )
        return [info_extractor, section_summarizer, critical_analyst, report_compiler]

    def _create_tasks(self, agents: List[Agent]) -> List[Task]:
        """Defines the tasks for the document analysis crew."""
        info_extractor, section_summarizer, critical_analyst, report_compiler = agents

        # Le contexte (contenu du document) sera passé à chaque tâche.
        # CrewAI gère le passage du contexte entre les tâches si elles sont dépendantes.
        # Pour les tâches initiales, on injecte le contenu du document.

        task_extract_info = Task(
            description=f"Analyze the following document (ID: {self.document_id}) with a research focus on '{self.research_focus}'. Extract key information: primary methodology, datasets used (if any), main quantitative results, and author-stated limitations. Present this as a structured list or key-value pairs. Document Content:\n\n---\n{self.document_content}\n---",
            expected_output="A structured list or dictionary containing the extracted methodology, datasets, key results, and limitations.",
            agent=info_extractor,
            # human_input=False # True si on veut une validation humaine
        )

        task_summarize_sections = Task(
            description=f"Based on the document (ID: {self.document_id}) and focusing on '{self.research_focus}', provide concise summaries for its main logical sections (e.g., Abstract, Introduction, Methods, Results, Conclusion). If sections are not clearly delineated, provide a general summary. Document Content:\n\n---\n{self.document_content}\n---",
            expected_output="A set of concise summaries for each main section of the document, or an overall summary if sections are not distinct.",
            agent=section_summarizer,
            context=[task_extract_info] # Optionnel: pourrait utiliser les infos extraites pour mieux cibler les sections
        )

        task_critical_analysis = Task(
            description=f"Perform a critical analysis of the document (ID: {self.document_id}) focusing on '{self.research_focus}'. Use the extracted information and section summaries (if available from previous tasks) to identify strengths, weaknesses, and novel contributions. Document Content (for reference, primary input should be outputs of previous tasks if available):\n\n---\n{self.document_content}\n---",
            expected_output="A brief critical analysis highlighting strengths, weaknesses, and novel contributions of the paper relevant to the research focus.",
            agent=critical_analyst,
            context=[task_extract_info, task_summarize_sections] # Dépend des tâches précédentes
        )

        task_compile_report = Task(
            description=f"Compile a final, structured analytical report for document ID: {self.document_id} with a focus on '{self.research_focus}'. Integrate the outputs from the Information Extractor, Section Summarizer, and Critical Analyst. The report should be well-organized and directly address the research focus.",
            expected_output=f"A comprehensive analytical report for document {self.document_id} covering: 1. Extracted Key Information (Methodology, Results, Limitations). 2. Section Summaries. 3. Critical Analysis (Strengths, Weaknesses, Contributions). Ensure the report is tailored to the research focus: '{self.research_focus}'.",
            agent=report_compiler,
            context=[task_extract_info, task_summarize_sections, task_critical_analysis] # Dépend de toutes les tâches précédentes
        )
        return [task_extract_info, task_summarize_sections, task_critical_analysis, task_compile_report]

    def run(self) -> str:
        """
        Assembles and runs the CrewAI crew to perform document analysis.
        Returns the final compiled report as a string.
        """
        if self.llm is None and not os.environ.get("OPENAI_API_KEY"):
            logger.error("Cannot run CrewAI DocumentAnalysisCrew: Neither a local LLM nor a global OPENAI_API_KEY is configured.")
            return "Error: LLM for CrewAI agents is not configured. Please set OPENAI_API_KEY or ensure local LLM setup for agents."

        logger.info(f"Starting Document Analysis Crew for doc ID: {self.document_id}, focus: '{self.research_focus}'")
        agents = self._create_agents()
        tasks = self._create_tasks(agents)

        crew = Crew(
            agents=agents,
            tasks=tasks,
            process=Process.sequential, # Les tâches sont exécutées séquentiellement
            verbose=2 # Niveau de verbosité (0, 1, ou 2)
            # memory=True # Si on veut que la crew ait une mémoire à court terme entre les tâches
        )

        try:
            # Le `kickoff()` retourne le résultat de la dernière tâche.
            result = crew.kickoff()
            logger.info(f"Document Analysis Crew finished for doc ID: {self.document_id}. Result length: {len(result)}")
            return result
        except Exception as e:
            logger.error(f"Error during CrewAI kickoff for document {self.document_id}: {e}", exc_info=True)
            return f"Error during document analysis by CrewAI: {str(e)}"

# --- Fonction utilitaire pour appeler cette crew ---
def run_document_deep_dive_crew(
    document_id: str, 
    document_content: str, 
    research_focus: str
) -> str:
    """
    Initializes and runs the DocumentAnalysisCrew for a given document.

    Args:
        document_id (str): Identifier of the document.
        document_content (str): Full text content of the document.
        research_focus (str): Specific focus for the analysis.

    Returns:
        str: The compiled analytical report from the crew.
    """
    if not document_content or not document_content.strip():
        logger.warning(f"Document content for {document_id} is empty. Skipping deep dive analysis.")
        return "Error: Document content provided was empty."
        
    crew_runner = DocumentAnalysisCrew(
        document_id=document_id,
        document_content=document_content,
        research_focus=research_focus
    )
    return crew_runner.run()


if __name__ == "__main__":
    # Pour tester ce module directement (nécessite OPENAI_API_KEY dans .env ou globalement)
    from config.logging_config import setup_logging
    setup_logging(level="DEBUG") # DEBUG pour voir les logs de CrewAI

    logger.info("--- Testing DocumentAnalysisCrew ---")

    if not settings.OPENAI_API_KEY:
        logger.error("OPENAI_API_KEY not configured in .env. Cannot run CrewAI test.")
    else:
        sample_doc_id = "test_arxiv_123"
        sample_doc_content = """
        Abstract: This paper introduces a novel method for robotic grasping using deep reinforcement learning. 
        Our approach, "GraspRL", leverages synthetic data and domain randomization to achieve robust sim-to-real transfer.

        Introduction: Robotic grasping is a fundamental challenge. Traditional methods rely on precise modeling... 
        RL offers a data-driven alternative... This paper focuses on improving sim-to-real transfer.

        Methodology: We use a PPO algorithm with an LSTM policy. The agent is trained in a PyBullet simulation. 
        We apply domain randomization to visual textures, lighting, and object positions. Key parameters include...

        Results: GraspRL achieved a 75% success rate on a physical robot, outperforming baseline X by 15%. 
        The sim-to-real drop was only 10%. Ablation studies show domain randomization is crucial.

        Limitations: The current system only supports single-object grasping and requires extensive simulation time. Future work includes multi-object grasping.

        Conclusion: GraspRL demonstrates effective sim-to-real transfer for robotic grasping using deep RL.
        """
        sample_research_focus = "sim-to-real transfer techniques and their effectiveness"

        logger.info(f"Test - Document ID: {sample_doc_id}")
        logger.info(f"Test - Research Focus: {sample_research_focus}")
        # logger.info(f"Test - Document Content (snippet): {sample_doc_content[:200]}...")

        # Exécuter la fonction utilitaire
        # C'est une opération bloquante/synchrone ici, bien que CrewAI puisse avoir des aspects asynchrones en interne.
        # Si run_document_deep_dive_crew devient `async`, il faudra `asyncio.run()`
        final_report = run_document_deep_dive_crew(
            document_id=sample_doc_id,
            document_content=sample_doc_content,
            research_focus=sample_research_focus
        )

        print("\n--- Final Compiled Report from CrewAI ---")
        print(final_report)
        print("--------------------------------------")

    logger.info("--- DocumentAnalysisCrew Test Finished ---")