"""
Document Analysis Crew Module

This module implements a CrewAI-based document analysis system that uses a team of specialized
agents to perform in-depth analysis of scientific documents. The crew consists of:
- Information Extractor: Extracts key structured information
- Section Summarizer: Provides concise summaries of document sections
- Critical Analyst: Performs critical analysis of the research
- Report Compiler: Compiles the final analytical report

The module handles document analysis through a sequential process where each agent
contributes to the final comprehensive report.
"""

import os
import logging
from typing import List, Dict, Any, Optional

from crewai import Agent, Task, Crew, Process
# MODIFICATION: Suppression de l'import de CrewOutput car le chemin semble incorrect ou non nécessaire
# from crewai.outputs import CrewOutput 

from src.llm_services.llm_factory import get_llm 
from config.settings import settings

logger = logging.getLogger(__name__)

class DocumentAnalysisCrew:
    """
    A CrewAI-based document analysis system that coordinates multiple specialized agents
    to perform comprehensive analysis of scientific documents.
    """
    
    def __init__(self, document_id: str, document_content: str, research_focus: str):
        """
        Initialize the document analysis crew.
        
        Args:
            document_id: Unique identifier for the document
            document_content: Full text content to analyze
            research_focus: Specific focus or questions for the analysis
        """
        self.document_id = document_id
        self.document_content = document_content
        self.research_focus = research_focus
        
        try:
            self.llm = get_llm(temperature=0.2)
        except ValueError as e:
            logger.error(f"Failed to initialize LLM: {e}")
            logger.warning("Falling back to CrewAI's default LLM if OPENAI_API_KEY is set")
            self.llm = None

    def _create_agents(self) -> List[Agent]:
        """
        Create the team of specialized agents for document analysis.
        
        Returns:
            List of configured CrewAI agents
        """
        info_extractor = Agent(
            role='Expert Information Extractor for Scientific Papers',
            goal=f"Extract key structured information from the paper focusing on '{self.research_focus}'",
            backstory="Expert in parsing scientific literature and identifying core research components",
            verbose=True,
            allow_delegation=False,
            llm=self.llm
        )
        
        section_summarizer = Agent(
            role='Scientific Section Summarizer',
            goal=f"Provide concise summaries of document sections focusing on '{self.research_focus}'",
            backstory="Skilled in distilling complex scientific sections into clear summaries",
            verbose=True,
            allow_delegation=False,
            llm=self.llm
        )
        
        critical_analyst = Agent(
            role='Critical Analyst of Scientific Research',
            goal=f"Analyze the paper's strengths, weaknesses, and contributions regarding '{self.research_focus}'",
            backstory="Experienced peer reviewer with expertise in scientific rigor and innovation",
            verbose=True,
            allow_delegation=False,
            llm=self.llm
        )
        
        report_compiler = Agent(
            role='Lead Research Report Compiler',
            goal=f"Compile a comprehensive report for document {self.document_id} focusing on '{self.research_focus}'",
            backstory="Senior research lead responsible for synthesizing detailed analyses into final reports",
            verbose=True,
            allow_delegation=True,
            llm=self.llm
        )
        
        return [info_extractor, section_summarizer, critical_analyst, report_compiler]

    def _create_tasks(self, agents: List[Agent]) -> List[Task]:
        """
        Create the sequence of tasks for document analysis.
        
        Args:
            agents: List of agents to assign tasks to
            
        Returns:
            List of configured CrewAI tasks
        """
        info_extractor, section_summarizer, critical_analyst, report_compiler = agents
        
        task_extract_info = Task(
            description=(
                f"Analyze document {self.document_id} focusing on '{self.research_focus}'.\n"
                "Extract: methodology, datasets, results, limitations.\n"
                f"Content:\n---\n{self.document_content}\n---"
            ),
            expected_output="Structured list of extracted information",
            agent=info_extractor
        )
        
        task_summarize_sections = Task(
            description=(
                f"Summarize main sections of document {self.document_id} focusing on '{self.research_focus}'.\n"
                f"Content:\n---\n{self.document_content}\n---"
            ),
            expected_output="Concise summaries of document sections",
            agent=section_summarizer,
            context=[task_extract_info]
        )
        
        task_critical_analysis = Task(
            description=(
                f"Analyze document {self.document_id} focusing on '{self.research_focus}'.\n"
                "Use previous task outputs to identify strengths, weaknesses, and contributions."
            ),
            expected_output="Critical analysis of the paper's key aspects",
            agent=critical_analyst,
            context=[task_extract_info, task_summarize_sections]
        )
        
        task_compile_report = Task(
            description=(
                f"Compile final report for document {self.document_id} focusing on '{self.research_focus}'.\n"
                "Integrate all previous analyses into a comprehensive report."
            ),
            expected_output=(
                f"Comprehensive report covering:\n"
                "1. Key Information (Methodology, Results, Limitations)\n"
                "2. Section Summaries\n"
                "3. Critical Analysis"
            ),
            agent=report_compiler,
            context=[task_extract_info, task_summarize_sections, task_critical_analysis]
        )
        
        return [task_extract_info, task_summarize_sections, task_critical_analysis, task_compile_report]

    def run(self) -> str:
        """
        Execute the document analysis process.
        
        Returns:
            Final analytical report or error message
        """
        if self.llm is None and not os.environ.get("OPENAI_API_KEY"):
            error_msg = "LLM not configured. Set API keys via .env or OPENAI_API_KEY"
            logger.error(error_msg)
            return f"Error: {error_msg}"

        logger.info(f"Starting analysis for document {self.document_id}")
        
        try:
            # Initialize and run the crew
            crew = Crew(
                agents=self._create_agents(),
                tasks=self._create_tasks(self._create_agents()),
                process=Process.sequential,
                verbose=False
            )
            
            # Execute analysis and process results
            crew_output = crew.kickoff()
            final_report = self._process_crew_output(crew_output)
            
            logger.info(f"Analysis completed for document {self.document_id}")
            return final_report
            
        except Exception as e:
            error_msg = f"Analysis failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return f"Error: {error_msg}"

    def _process_crew_output(self, output: Any) -> str:
        """
        Process the crew's output into a final report.
        
        Args:
            output: Raw output from the crew
            
        Returns:
            Processed report text
        """
        if not output:
            logger.warning("Crew returned no output")
            return "Error: No analysis results available"
            
        # Try different output formats
        if hasattr(output, 'raw') and output.raw:
            return str(output.raw)
            
        if hasattr(output, 'tasks_output') and output.tasks_output:
            last_task = output.tasks_output[-1]
            if hasattr(last_task, 'exported_output') and last_task.exported_output:
                return str(last_task.exported_output)
            if hasattr(last_task, 'raw_output') and last_task.raw_output:
                return str(last_task.raw_output)
            return str(last_task)
            
        if hasattr(output, 'data') and hasattr(output.data, 'output'):
            return str(output.data.output)
            
        return str(output)

def run_document_deep_dive_crew(
    document_id: str,
    document_content: str,
    research_focus: str
) -> str:
    """
    Run a deep dive analysis on a document using the DocumentAnalysisCrew.
    
    Args:
        document_id: Unique identifier for the document
        document_content: Full text content to analyze
        research_focus: Specific focus or questions for the analysis
        
    Returns:
        Analysis report or error message
    """
    if not document_content or not document_content.strip():
        logger.warning(f"Empty document content for {document_id}")
        return "Error: Document content is empty"
        
    crew = DocumentAnalysisCrew(
        document_id=document_id,
        document_content=document_content,
        research_focus=research_focus
    )
    return crew.run()

def _can_run_crew_test() -> bool:
    """
    Check if the crew test can be run with current configuration.
    
    Returns:
        True if configuration is valid, False otherwise
    """
    try:
        provider = settings.DEFAULT_LLM_MODEL_PROVIDER.lower()
        
        if provider == "openai" and settings.OPENAI_API_KEY:
            return True
        if provider == "huggingface_api" and settings.HUGGINGFACE_API_KEY and settings.HUGGINGFACE_REPO_ID:
            return True
        if provider == "ollama" and settings.OLLAMA_BASE_URL and settings.OLLAMA_GENERATIVE_MODEL_NAME:
            return True
        if os.environ.get("OPENAI_API_KEY"):
            logger.warning("Using OPENAI_API_KEY from environment")
            return True
            
    except Exception as e:
        logger.warning(f"Configuration check failed: {e}")
        return bool(os.environ.get("OPENAI_API_KEY"))
        
    return False

if __name__ == "__main__":
    from config.logging_config import setup_logging
    setup_logging(level="DEBUG")
    
    logger.info("Testing Document Analysis Crew")
    
    if not _can_run_crew_test():
        logger.error("Cannot run test: LLM configuration missing")
    else:
        test_doc = {
            "id": "test_001",
            "content": "Sample document about methodology, results, and limitations.",
            "focus": "Extract methodology, results, and limitations"
        }
        
        report = run_document_deep_dive_crew(
            document_id=test_doc["id"],
            document_content=test_doc["content"],
            research_focus=test_doc["focus"]
        )
        
        print("\nAnalysis Report:")
        print(report)
        print("-" * 50)
    
    logger.info("Test completed")