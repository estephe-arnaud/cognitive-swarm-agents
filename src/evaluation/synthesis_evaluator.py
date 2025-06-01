# src/evaluation/synthesis_evaluator.py
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
    score: float
    reasoning: str

class SynthesisEvaluationResult(TypedDict):
    relevance: Optional[EvaluationAspectScore]
    faithfulness: Optional[EvaluationAspectScore]
    coherence: Optional[EvaluationAspectScore] # Pour plus tard

# --- Prompts pour LLM-as-a-Judge (inchangés) ---
RELEVANCE_EVAL_PROMPT_TEMPLATE = """
Vous êtes un évaluateur expert chargé d'estimer la pertinence d'une synthèse générée par rapport à une requête utilisateur originale.
La pertinence mesure si la synthèse répond directement et de manière appropriée à la requête.

**Requête Utilisateur Originale:**
{query}

**Synthèse Générée:**
{synthesis}

**Instructions d'Évaluation pour la Pertinence:**
1.  Lisez attentivement la requête utilisateur et la synthèse générée.
2.  Évaluez dans quelle mesure la synthèse adresse les points clés de la requête.
3.  Ignorez pour l'instant la factualité ou la qualité de l'écriture, concentrez-vous uniquement sur la pertinence par rapport à la requête.
4.  Fournissez un score de pertinence entre 0.0 (totalement non pertinent) et 1.0 (parfaitement pertinent).
5.  Fournissez une brève justification (1-2 phrases) pour votre score.

**Format de Sortie Attendu (JSON):**
{{
    "score": <float, ex: 0.8>,
    "reasoning": "<string, votre justification>"
}}
"""

FAITHFULNESS_EVAL_PROMPT_TEMPLATE = """
Vous êtes un évaluateur expert chargé d'estimer la fidélité (factualité) d'une synthèse générée par rapport à un contexte source fourni.
La fidélité mesure si les affirmations faites dans la synthèse sont correctement étayées par le contexte source et ne contiennent pas d'informations contredisant ce contexte ou inventées (hallucinations).

**Requête Utilisateur Originale (pour information, mais l'évaluation se base sur le contexte):**
{query}

**Contexte Source Fourni à l'Agent de Synthèse:**
{context}

**Synthèse Générée (à évaluer pour sa fidélité AU CONTEXTE SOURCE):**
{synthesis}

**Instructions d'Évaluation pour la Fidélité:**
1.  Lisez attentivement le contexte source et la synthèse générée.
2.  Vérifiez chaque affirmation majeure dans la synthèse. Est-elle directement soutenue par une information présente dans le contexte source ?
3.  Identifiez toute information dans la synthèse qui contredit le contexte source ou qui semble être une information supplémentaire non présente dans le contexte.
4.  Fournissez un score de fidélité entre 0.0 (totalement infidèle, beaucoup d'hallucinations ou de contradictions) et 1.0 (parfaitement fidèle, toutes les affirmations sont étayées par le contexte).
5.  Fournissez une brève justification (1-2 phrases) pour votre score, en citant des exemples si possible.

**Format de Sortie Attendu (JSON):**
{{
    "score": <float, ex: 0.9>,
    "reasoning": "<string, votre justification avec exemples si possible>"
}}
"""

class SynthesisEvaluator:
    def __init__(
        self,
        judge_llm_provider: Optional[str] = None,
        judge_llm_model_name: Optional[str] = None,
        # Utilise DEFAULT_LLM_TEMPERATURE importé depuis llm_factory
        judge_llm_temperature: float = DEFAULT_LLM_TEMPERATURE
    ):
        self.judge_llm_provider_init = judge_llm_provider
        self.judge_llm_model_name_init = judge_llm_model_name
        self.judge_llm_temperature = judge_llm_temperature
        self.judge_llm: BaseLanguageModel = self._get_judge_llm()
        logger.info(f"SynthesisEvaluator initialized with judge LLM type: {type(self.judge_llm)}")

    def _get_judge_llm(self) -> BaseLanguageModel:
        """
        Récupère l'instance du LLM juge en utilisant la fonction get_llm centralisée.
        """
        try:
            # Utilise get_llm importé depuis llm_factory
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
        supports_json_mode = False
        if current_provider.lower() == "openai":
            supports_json_mode = True
        elif current_provider.lower() == "ollama":
            supports_json_mode = True 

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
        logger.info(f"Evaluating relevance for query: '{query[:50]}...'")
        return await self._evaluate_aspect(RELEVANCE_EVAL_PROMPT_TEMPLATE, query, synthesis)

    async def evaluate_faithfulness(self, query: str, synthesis: str, context: str) -> Optional[EvaluationAspectScore]:
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
        logger.info(f"Starting synthesis evaluation for query: '{query[:50]}...'")

        relevance_score = await self.evaluate_relevance(query, synthesis)
        faithfulness_score = await self.evaluate_faithfulness(query, synthesis, context)

        result: SynthesisEvaluationResult = {
            "relevance": relevance_score,
            "faithfulness": faithfulness_score,
            "coherence": None, # Placeholder
        }
        logger.info(f"Synthesis evaluation completed. Relevance: {relevance_score}, Faithfulness: {faithfulness_score}")
        return result

    def print_results(self, results: SynthesisEvaluationResult, query: Optional[str] = None):
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
    from config.logging_config import setup_logging # Uniquement pour le test direct

    setup_logging(level="DEBUG")
    logger.info("--- Starting Synthesis Evaluator Test Run (with llm_factory) ---")
    
    # Ce test dépend de la configuration correcte du LLM juge via settings et llm_factory
    can_run_eval_test = False
    try:
        # Tentative d'instanciation pour vérifier si get_llm fonctionne avec la config actuelle
        # SynthesisEvaluator() fait cela dans son __init__
        temp_evaluator = SynthesisEvaluator()
        logger.info(f"Test evaluator initialized with judge LLM: {type(temp_evaluator.judge_llm)}")
        can_run_eval_test = True
    except ValueError as ve:
        logger.error(f"Cannot run SynthesisEvaluator test: Failed to initialize judge LLM via llm_factory. Error: {ve}")
    except Exception as e_init:
        logger.error(f"Unexpected error initializing SynthesisEvaluator for test: {e_init}", exc_info=True)


    if can_run_eval_test:
        evaluator = SynthesisEvaluator(
            # judge_llm_model_name="gpt-4o" # Peut être surchargé ici pour tester un juge spécifique
            # judge_llm_provider="openai" # Pour forcer un provider
        )
        logger.info(f"Evaluator for test run using judge LLM: {type(evaluator.judge_llm)}")

        test_query = "What are the key benefits of using reinforcement learning in robotics, and what are some notable challenges?"
        test_context_for_synthesis = """
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
        good_synthesis = """
        Reinforcement learning (RL) provides significant benefits for robotics, including the capacity for robots to acquire complex behaviors
        in dynamic settings without exhaustive manual programming, enabling adaptation to new situations. RL can also lead to innovative solutions.
        Key applications include robot manipulation, navigation, and interaction.
        Despite these advantages, RL in robotics encounters challenges such as sample inefficiency (requiring many trials), ensuring safety
        during the learning process, bridging the sim-to-real gap, and the complexities of designing effective reward functions.
        Recent research, like that in ArXiv:123.456, addresses safe exploration.
        """

        async def run_evaluations_test_main(): # Renommé pour éviter conflit avec la fonction du notebook
            logger.info("\n--- Evaluating Good Synthesis (Judge LLM may take time) ---")
            results_good = await evaluator.evaluate_synthesis(test_query, good_synthesis, test_context_for_synthesis)
            evaluator.print_results(results_good, test_query)

        asyncio.run(run_evaluations_test_main())

    logger.info("--- Synthesis Evaluator Test Run Finished ---")