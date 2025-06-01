# src/agents/crewai_teams/document_analysis_crew.py
import os
import logging
from typing import List, Dict, Any, Optional # 'Any' est utilisé pour le type hint de crew_output_obj

from crewai import Agent, Task, Crew, Process
# MODIFICATION: Suppression de l'import de CrewOutput car le chemin semble incorrect ou non nécessaire
# from crewai.outputs import CrewOutput 

from src.llm_services.llm_factory import get_llm 
from config.settings import settings

logger = logging.getLogger(__name__)

class DocumentAnalysisCrew:
    def __init__(self, document_id: str, document_content: str, research_focus: str):
        self.document_id = document_id
        self.document_content = document_content
        self.research_focus = research_focus
        
        try:
            self.llm = get_llm(temperature=0.2) 
        except ValueError as e:
            logger.error(f"Failed to get LLM for CrewAI from llm_factory: {e}. Ensure LLM provider is configured.")
            logger.warning("LLM from get_llm() failed. CrewAI agents will attempt to use their default LLM if OPENAI_API_KEY is set globally.")
            self.llm = None

    def _create_agents(self) -> List[Agent]:
        # ... (définitions des agents inchangées par rapport à la version précédente complète)
        info_extractor = Agent(
            role='Expert Information Extractor for Scientific Papers',
            goal=f"Meticulously extract key structured information (such as main methodology, dataset used, key quantitative results, and explicitly stated limitations) from the provided scientific paper text related to '{self.research_focus}'. Focus solely on information present in the text.",
            backstory="You are an AI assistant with deep expertise in parsing scientific literature and identifying core components of research papers. You are extremely precise and only extract information that is explicitly stated.",
            verbose=True, 
            allow_delegation=False,
            llm=self.llm 
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
            allow_delegation=False,
            llm=self.llm
        )
        report_compiler = Agent(
            role='Lead Research Report Compiler',
            goal=f"Compile a comprehensive and structured analytical report for the scientific paper (ID: {self.document_id}) focusing on '{self.research_focus}'. Integrate the extracted key information, section summaries, and critical analysis from other team members into a coherent final document.",
            backstory="You are a senior AI research lead responsible for synthesizing detailed analyses from your team into a final, publishable-quality report. Your work is structured, clear, and directly addresses the research focus.",
            verbose=True,
            allow_delegation=True,
            llm=self.llm
        )
        return [info_extractor, section_summarizer, critical_analyst, report_compiler]

    def _create_tasks(self, agents: List[Agent]) -> List[Task]:
        info_extractor, section_summarizer, critical_analyst, report_compiler = agents
        # ... (définitions des tâches inchangées) ...
        task_extract_info = Task(
            description=f"Analyze the following document (ID: {self.document_id}) with a research focus on '{self.research_focus}'. Extract key information: primary methodology, datasets used (if any), main quantitative results, and author-stated limitations. Present this as a structured list or key-value pairs. Document Content:\n\n---\n{self.document_content}\n---",
            expected_output="A structured list or dictionary containing the extracted methodology, datasets, key results, and limitations.",
            agent=info_extractor,
        )
        task_summarize_sections = Task(
            description=f"Based on the document (ID: {self.document_id}) and focusing on '{self.research_focus}', provide concise summaries for its main logical sections (e.g., Abstract, Introduction, Methods, Results, Conclusion). If sections are not clearly delineated, provide a general summary. Document Content:\n\n---\n{self.document_content}\n---",
            expected_output="A set of concise summaries for each main section of the document, or an overall summary if sections are not distinct.",
            agent=section_summarizer,
            context=[task_extract_info]
        )
        task_critical_analysis = Task(
            description=f"Perform a critical analysis of the document (ID: {self.document_id}) focusing on '{self.research_focus}'. Use the extracted information and section summaries (if available from previous tasks) to identify strengths, weaknesses, and novel contributions. Document Content (for reference, primary input should be outputs of previous tasks if available):\n\n---\n{self.document_content}\n---",
            expected_output="A brief critical analysis highlighting strengths, weaknesses, and novel contributions of the paper relevant to the research focus.",
            agent=critical_analyst,
            context=[task_extract_info, task_summarize_sections]
        )
        task_compile_report = Task(
            description=f"Compile a final, structured analytical report for document ID: {self.document_id} with a focus on '{self.research_focus}'. Integrate the outputs from the Information Extractor, Section Summarizer, and Critical Analyst. The report should be well-organized and directly address the research focus.",
            expected_output=f"A comprehensive analytical report for document {self.document_id} covering: 1. Extracted Key Information (Methodology, Results, Limitations). 2. Section Summaries. 3. Critical Analysis (Strengths, Weaknesses, Contributions). Ensure the report is tailored to the research focus: '{self.research_focus}'.",
            agent=report_compiler,
            context=[task_extract_info, task_summarize_sections, task_critical_analysis]
        )
        return [task_extract_info, task_summarize_sections, task_critical_analysis, task_compile_report]

    def run(self) -> str:
        if self.llm is None and not os.environ.get("OPENAI_API_KEY"): 
            logger.error("Cannot run CrewAI DocumentAnalysisCrew: Neither a local LLM nor a global OPENAI_API_KEY is configured.")
            return "Error: LLM for CrewAI agents is not configured. Please set API keys via .env or ensure OPENAI_API_KEY is set for CrewAI's default."

        logger.info(f"Starting Document Analysis Crew for doc ID: {self.document_id}, focus: '{self.research_focus}'")
        agents = self._create_agents()
        tasks = self._create_tasks(agents)

        crew = Crew(
            agents=agents,
            tasks=tasks,
            process=Process.sequential,
            verbose=False 
        )

        try:
            # MODIFICATION: Changer le type hint de crew_output_obj en Optional[Any]
            crew_output_obj: Optional[Any] = crew.kickoff() 
            
            final_text_report = ""
            if crew_output_obj:
                if hasattr(crew_output_obj, 'raw') and crew_output_obj.raw:
                    final_text_report = str(crew_output_obj.raw)
                elif hasattr(crew_output_obj, 'tasks_output') and crew_output_obj.tasks_output:
                    last_task_output = crew_output_obj.tasks_output[-1]
                    if hasattr(last_task_output, 'exported_output') and last_task_output.exported_output:
                        final_text_report = str(last_task_output.exported_output)
                    elif hasattr(last_task_output, 'raw_output') and last_task_output.raw_output:
                         final_text_report = str(last_task_output.raw_output)
                    else: 
                        final_text_report = str(last_task_output) 
                        logger.warning(f"Used str(last_task_output) as exported_output/raw_output not found. Output: {final_text_report[:100]}")
                elif hasattr(crew_output_obj, 'data') and hasattr(crew_output_obj.data, 'output') and crew_output_obj.data.output:
                    final_text_report = str(crew_output_obj.data.output)
                else: 
                    final_text_report = str(crew_output_obj) 
                    logger.warning(f"crew_output_obj.raw and specific task outputs were None or not present. Using str(crew_output_obj). Output: {final_text_report[:200]}...")
            else:
                logger.warning("crew.kickoff() returned None.")

            report_length = len(final_text_report) if final_text_report else 0
            logger.info(f"Document Analysis Crew finished for doc ID: {self.document_id}. Result length: {report_length}")
            return final_text_report 
            
        except Exception as e:
            logger.error(f"Error during CrewAI kickoff for document {self.document_id}: {e}", exc_info=True)
            return f"Error during document analysis by CrewAI (kickoff exception): {str(e)}"

def run_document_deep_dive_crew(
    document_id: str, 
    document_content: str, 
    research_focus: str
) -> str:
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
    from config.logging_config import setup_logging 
    setup_logging(level="DEBUG") 

    logger.info("--- Testing DocumentAnalysisCrew (with CrewOutput type hint fix) ---")
    can_run_crew_test = False
    try:
        provider_check = settings.DEFAULT_LLM_MODEL_PROVIDER.lower()
        if provider_check == "openai" and settings.OPENAI_API_KEY:
            can_run_crew_test = True
        elif provider_check == "huggingface_api" and settings.HUGGINGFACE_API_KEY and settings.HUGGINGFACE_REPO_ID:
            can_run_crew_test = True
        elif provider_check == "ollama" and settings.OLLAMA_BASE_URL and settings.OLLAMA_GENERATIVE_MODEL_NAME:
            can_run_crew_test = True
        elif os.environ.get("OPENAI_API_KEY"): 
            logger.warning("Default LLM provider in settings might not be configured, but OPENAI_API_KEY is in env. CrewAI might work with its default.")
            can_run_crew_test = True
    except Exception as e_llm_check:
        logger.warning(f"Pre-check for LLM configuration failed: {e_llm_check}. CrewAI test might fail.")
        if os.environ.get("OPENAI_API_KEY"):
             can_run_crew_test = True

    if not can_run_crew_test:
        logger.error("Necessary LLM configuration not found. Cannot run CrewAI test effectively.")
    else:
        sample_doc_id = "test_arxiv_crewoutput_typehint_fix"
        sample_doc_content = "Sample content for testing CrewOutput. This document discusses methodology, results, and limitations of a new technique."
        sample_research_focus = "Extract methodology, results, and limitations."

        logger.info(f"Test - Document ID: {sample_doc_id}")
        logger.info(f"Test - Research Focus: {sample_research_focus}")
        
        final_report = run_document_deep_dive_crew(
            document_id=sample_doc_id,
            document_content=sample_doc_content,
            research_focus=sample_research_focus
        )

        print("\n--- Final Compiled Report from CrewAI (with CrewOutput type hint fix) ---")
        print(final_report)
        print("-------------------------------------------------------------")

    logger.info("--- DocumentAnalysisCrew Test Finished ---")