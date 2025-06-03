"""
Synthesis Evaluator Module

This module provides tools for evaluating the quality of synthesized research content.
It implements a LLM-as-a-Judge approach to assess different aspects of synthesis quality:
- Relevance: How well the synthesis addresses the original query
- Faithfulness: How accurately the synthesis reflects the source context
- Coherence: (Planned future feature) How well-structured and logical the synthesis is

The evaluator uses LangChain's framework and supports multiple LLM providers.
"""

import logging
from typing import List, Dict, Any, Optional, TypedDict

from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from config.settings import settings
# MODIFICATION: Mettre à jour l'import pour get_llm et DEFAULT_LLM_TEMPERATURE
from src.llm_services.llm_factory import get_llm, DEFAULT_LLM_TEMPERATURE

logger = logging.getLogger(__name__)

# --- Structures de Données pour l'Évaluation (inchangées) ---
class EvaluationAspectScore(TypedDict):
    """Represents the score and reasoning for a specific evaluation aspect."""
    score: float
    reasoning: str

class SynthesisEvaluationResult(TypedDict):
    """Contains evaluation scores for different aspects of a synthesis."""
    relevance: Optional[EvaluationAspectScore]
    faithfulness: Optional[EvaluationAspectScore]
    coherence: Optional[EvaluationAspectScore]  # Future feature

# --- Prompts pour LLM-as-a-Judge (inchangés) ---
RELEVANCE_EVAL_PROMPT_TEMPLATE = """
You are an expert evaluator tasked with assessing the relevance of a generated synthesis to an original user query.
Relevance measures how well the synthesis directly and appropriately addresses the query.

**Original User Query:**
{query}

**Generated Synthesis:**
{synthesis}

**Evaluation Instructions for Relevance:**
1. Carefully read the user query and generated synthesis.
2. Evaluate how well the synthesis addresses the key points of the query.
3. Ignore factual accuracy or writing quality for now, focus only on relevance to the query.
4. Provide a relevance score between 0.0 (completely irrelevant) and 1.0 (perfectly relevant).
5. Provide a brief justification (1-2 sentences) for your score.

**Expected Output Format (JSON):**
{{
    "score": <float, ex: 0.8>,
    "reasoning": "<string, your justification>"
}}
"""

FAITHFULNESS_EVAL_PROMPT_TEMPLATE = """
You are an expert evaluator tasked with assessing the faithfulness (factual accuracy) of a generated synthesis against provided source context.
Faithfulness measures whether claims in the synthesis are properly supported by the source context and don't contain information contradicting or hallucinated beyond this context.

**Original User Query (for reference, but evaluation is based on context):**
{query}

**Source Context Provided to Synthesis Agent:**
{context}

**Generated Synthesis (to evaluate for faithfulness TO SOURCE CONTEXT):**
{synthesis}

**Evaluation Instructions for Faithfulness:**
1. Carefully read the source context and generated synthesis.
2. Verify each major claim in the synthesis. Is it directly supported by information in the source context?
3. Identify any information in the synthesis that contradicts the source context or appears to be additional information not present in the context.
4. Provide a faithfulness score between 0.0 (completely unfaithful, many hallucinations or contradictions) and 1.0 (perfectly faithful, all claims supported by context).
5. Provide a brief justification (1-2 sentences) for your score, citing examples if possible.

**Expected Output Format (JSON):**
{{
    "score": <float, ex: 0.9>,
    "reasoning": "<string, your justification with examples if possible>"
}}
"""

class SynthesisEvaluator:
    """
    Evaluates the quality of research content synthesis using LLM-as-a-Judge approach.
    
    This class provides methods to evaluate different aspects of synthesis quality:
    - Relevance to original query
    - Faithfulness to source context
    - (Future) Coherence of the synthesis
    """
    
    def __init__(
        self,
        judge_llm_provider: Optional[str] = None,
        judge_llm_model_name: Optional[str] = None,
        # Utilise DEFAULT_LLM_TEMPERATURE importé depuis llm_factory
        judge_llm_temperature: float = DEFAULT_LLM_TEMPERATURE
    ):
        """
        Initialize the synthesis evaluator.
        
        Args:
            judge_llm_provider: Optional override for the LLM provider
            judge_llm_model_name: Optional override for the LLM model name
            judge_llm_temperature: Temperature setting for the judge LLM
        """
        self.judge_llm_provider_init = judge_llm_provider
        self.judge_llm_model_name_init = judge_llm_model_name
        self.judge_llm_temperature = judge_llm_temperature
        self.judge_llm: BaseLanguageModel = self._get_judge_llm()
        logger.info(f"SynthesisEvaluator initialized with judge LLM type: {type(self.judge_llm)}")

    def _get_judge_llm(self) -> BaseLanguageModel:
        """
        Get the judge LLM instance using the centralized get_llm function.
        
        Returns:
            BaseLanguageModel: The initialized LLM instance for evaluation
            
        Raises:
            ValueError: If LLM initialization fails
        """
        try:
            return get_llm(
                temperature=self.judge_llm_temperature,
                model_provider_override=self.judge_llm_provider_init,
                model_name_override=self.judge_llm_model_name_init
            )
        except ValueError as e:
            logger.error(f"Failed to initialize Judge LLM for SynthesisEvaluator: {e}")
            raise

    async def _evaluate_aspect(
        self,
        prompt_template_str: str,
        query: str,
        synthesis: str,
        context: Optional[str] = None
    ) -> Optional[EvaluationAspectScore]:
        """
        Evaluate a specific aspect of the synthesis using the provided prompt template.
        
        Args:
            prompt_template_str: The prompt template for the evaluation
            query: The original user query
            synthesis: The generated synthesis to evaluate
            context: Optional source context for faithfulness evaluation
            
        Returns:
            Optional[EvaluationAspectScore]: The evaluation score and reasoning, or None if evaluation fails
        """
        if not self.judge_llm:
            logger.error("Judge LLM not initialized for aspect evaluation.")
            return None

        prompt_inputs = {"query": query, "synthesis": synthesis}
        if "{context}" in prompt_template_str:
            if context is None:
                logger.warning("Context required for evaluation aspect but not provided. Skipping this aspect.")
                return None
            prompt_inputs["context"] = context

        eval_prompt = ChatPromptTemplate.from_template(prompt_template_str)
        
        current_provider = self.judge_llm_provider_init or settings.DEFAULT_LLM_MODEL_PROVIDER
        supports_json_mode = current_provider.lower() in ["openai", "ollama"]

        if hasattr(self.judge_llm, 'bind') and supports_json_mode:
            try:
                judge_llm_with_json_mode = self.judge_llm.bind(
                    response_format={"type": "json_object"}
                )
                chain_with_json_mode = eval_prompt | judge_llm_with_json_mode | JsonOutputParser()
                response_dict = await chain_with_json_mode.ainvoke(prompt_inputs)
                logger.debug(f"JSON mode response for aspect: {response_dict}")
            except Exception as e_json_bind:
                logger.warning(f"Failed to use JSON mode with LLM {type(self.judge_llm)} (provider: {current_provider}), possibly not supported or model error: {e_json_bind}. Falling back to standard parsing.")
                chain = eval_prompt | self.judge_llm | JsonOutputParser()
                response_dict = await chain.ainvoke(prompt_inputs)
        else:
            logger.info(f"LLM {type(self.judge_llm)} (provider: {current_provider}) may not support native JSON mode binding or not attempted. Relying on prompt for JSON output.")
            chain = eval_prompt | self.judge_llm | JsonOutputParser()
            response_dict = await chain.ainvoke(prompt_inputs)

        try:
            if isinstance(response_dict, dict) and "score" in response_dict and "reasoning" in response_dict:
                return EvaluationAspectScore(score=float(response_dict["score"]), reasoning=str(response_dict["reasoning"]))
            else:
                logger.error(f"Failed to parse valid score/reasoning from Judge LLM response: {response_dict}")
                return None
        except Exception as e:
            logger.error(f"Error processing LLM-as-a-Judge response for an aspect: {e}", exc_info=True)
            return None

    async def evaluate_relevance(self, query: str, synthesis: str) -> Optional[EvaluationAspectScore]:
        """
        Evaluate the relevance of the synthesis to the original query.
        
        Args:
            query: The original user query
            synthesis: The generated synthesis to evaluate
            
        Returns:
            Optional[EvaluationAspectScore]: The relevance score and reasoning
        """
        logger.info(f"Evaluating relevance for query: '{query[:50]}...'")
        return await self._evaluate_aspect(RELEVANCE_EVAL_PROMPT_TEMPLATE, query, synthesis)

    async def evaluate_faithfulness(self, query: str, synthesis: str, context: str) -> Optional[EvaluationAspectScore]:
        """
        Evaluate the faithfulness of the synthesis to the source context.
        
        Args:
            query: The original user query (for reference)
            synthesis: The generated synthesis to evaluate
            context: The source context to evaluate against
            
        Returns:
            Optional[EvaluationAspectScore]: The faithfulness score and reasoning
        """
        logger.info(f"Evaluating faithfulness for query: '{query[:50]}...' against context (len: {len(context)}).")
        if not context or not context.strip():
            logger.warning("Context is empty or whitespace-only. Faithfulness evaluation will be skipped or unreliable.")
            return EvaluationAspectScore(score=0.0, reasoning="Context was not provided or was empty.")
        return await self._evaluate_aspect(FAITHFULNESS_EVAL_PROMPT_TEMPLATE, query, synthesis, context=context)

    async def evaluate_synthesis(
        self,
        query: str,
        synthesis: str,
        context: str
    ) -> SynthesisEvaluationResult:
        """
        Perform a complete evaluation of the synthesis.
        
        Args:
            query: The original user query
            synthesis: The generated synthesis to evaluate
            context: The source context for faithfulness evaluation
            
        Returns:
            SynthesisEvaluationResult: Complete evaluation results including relevance and faithfulness scores
        """
        logger.info(f"Starting synthesis evaluation for query: '{query[:50]}...'")

        relevance_score = await self.evaluate_relevance(query, synthesis)
        faithfulness_score = await self.evaluate_faithfulness(query, synthesis, context)

        result: SynthesisEvaluationResult = {
            "relevance": relevance_score,
            "faithfulness": faithfulness_score,
            "coherence": None,  # Future feature
        }
        logger.info(f"Synthesis evaluation completed. Relevance: {relevance_score}, Faithfulness: {faithfulness_score}")
        return result

    def print_results(self, results: SynthesisEvaluationResult, query: Optional[str] = None):
        """
        Print the evaluation results in a human-readable format.
        
        Args:
            results: The evaluation results to print
            query: Optional original query for context
        """
        print("\n--- Synthesis Evaluation Results ---")
        if query:
            print(f"For Query: {query}")

        if results["relevance"]:
            print(f"  Relevance Score: {results['relevance']['score']:.2f}")
            print(f"    Reasoning: {results['relevance']['reasoning']}")
        else:
            print("  Relevance: Not Evaluated / Error")

        if results["faithfulness"]:
            print(f"  Faithfulness Score: {results['faithfulness']['score']:.2f}")
            print(f"    Reasoning: {results['faithfulness']['reasoning']}")
        else:
            print("  Faithfulness: Not Evaluated / Error")
        print("------------------------------------")


if __name__ == "__main__":
    import asyncio
    from config.logging_config import setup_logging

    setup_logging(level="DEBUG")
    logger.info("--- Starting Synthesis Evaluator Test Run ---")
    
    # Test initialization
    can_run_eval_test = False
    try:
        temp_evaluator = SynthesisEvaluator()
        logger.info(f"Test evaluator initialized with judge LLM: {type(temp_evaluator.judge_llm)}")
        can_run_eval_test = True
    except ValueError as ve:
        logger.error(f"Cannot run SynthesisEvaluator test: Failed to initialize judge LLM. Error: {ve}")
    except Exception as e_init:
        logger.error(f"Unexpected error initializing SynthesisEvaluator for test: {e_init}", exc_info=True)

    if can_run_eval_test:
        evaluator = SynthesisEvaluator()
        logger.info(f"Evaluator for test run using judge LLM: {type(evaluator.judge_llm)}")

        # Test data
        test_query = "What are the key benefits of using reinforcement learning in robotics, and what are some notable challenges?"
        test_context = """
        Reinforcement learning (RL) in robotics offers several advantages. One key benefit is the ability for robots to learn complex behaviors
        in dynamic environments without explicit programming for every scenario. For instance, RL allows robots to adapt to unforeseen
        obstacles or changes in task requirements. Another benefit is the potential for discovering novel solutions that human programmers
        might not conceive. Robots can learn manipulation skills, navigation strategies, and even human-robot interaction through trial and error.

        However, applying RL to real-world robotics faces significant challenges. Sample inefficiency is a major hurdle; robots often require
        a vast number of interactions with the environment to learn a task, which can be time-consuming and costly, especially with physical hardware.
        Safety during learning is another critical concern, as a robot learning by trial and error might perform actions that could damage itself,
        its surroundings, or humans. The sim-to-real gap also poses a problem: models trained in simulation often do not transfer well to the real
        world due to discrepancies in physics modeling and sensory inputs. Finally, reward shaping can be difficult; designing effective reward
        functions that guide the robot towards the desired behavior without leading to unintended consequences is a complex art. Some papers (e.g. ArXiv:123.456) mention new methods for safe exploration.
        """
        test_synthesis = """
        Reinforcement learning in robotics offers significant benefits but also presents notable challenges. On the positive side, RL enables robots to learn complex behaviors in dynamic environments without needing explicit programming for every possible scenario. This adaptability is particularly valuable for handling unforeseen obstacles or changing task requirements. Additionally, RL can lead to the discovery of novel solutions that might not be conceived by human programmers, spanning areas like manipulation skills, navigation strategies, and human-robot interaction.

        However, several challenges must be addressed. A major issue is sample inefficiency, as robots typically require numerous environment interactions to learn tasks, which can be time-consuming and expensive, especially with physical hardware. Safety during the learning process is another critical concern, since trial-and-error learning could potentially lead to actions that damage the robot, its environment, or humans. The sim-to-real gap presents another significant challenge, as models trained in simulation often fail to transfer effectively to real-world applications due to differences in physics modeling and sensory inputs. Finally, reward shaping remains a complex task, requiring careful design of reward functions that guide robots toward desired behaviors without unintended consequences.
        """

        async def run_evaluation_test():
            """Run a complete evaluation test with the test data."""
            results = await evaluator.evaluate_synthesis(test_query, test_synthesis, test_context)
            evaluator.print_results(results, test_query)

        asyncio.run(run_evaluation_test())

    logger.info("--- Synthesis Evaluator Test Run Finished ---")