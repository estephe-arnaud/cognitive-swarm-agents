# cognitive-swarm-agents/scripts/run_evaluation.py
import argparse
import asyncio
import logging
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, TypedDict
import pandas as pd 
import os

from config.settings import settings
from config.logging_config import setup_logging
from src.rag.retrieval_engine import RetrievalEngine
from src.evaluation.rag_evaluator import RagEvaluator, RagEvaluationMetrics # EvalQuery renommé en RagEvalQuery dans le fichier source
from src.evaluation.synthesis_evaluator import SynthesisEvaluator, SynthesisEvaluationResult
from src.evaluation.metrics_logger import WandBMetricsLogger
from src.vector_store.mongodb_manager import MongoDBManager


logger = logging.getLogger(__name__)

class SynthesisEvalItem(TypedDict):
    query_id: str
    query_text: str
    context: str 
    synthesis_to_evaluate: str 

async def run_rag_evaluation(
    wb_logger: WandBMetricsLogger,
    rag_eval_dataset_path_str: Optional[str],
    collection_name: str,
    vector_index_name: str,
    top_k: int
) -> Optional[RagEvaluationMetrics]:
    logger.info(f"\n--- Starting RAG Evaluation (k={top_k}) ---")
    
    try:
        retrieval_engine_instance = RetrievalEngine(
            collection_name=collection_name,
            vector_index_name=vector_index_name
        )
    except Exception as e:
        logger.error(f"Failed to initialize RetrievalEngine for RAG evaluation: {e}", exc_info=True)
        return None

    eval_dataset_path = Path(rag_eval_dataset_path_str) if rag_eval_dataset_path_str else None
    if eval_dataset_path and not eval_dataset_path.exists():
        logger.error(f"RAG evaluation dataset specified at '{eval_dataset_path}' not found.")
    
    evaluator = RagEvaluator(
        retrieval_engine=retrieval_engine_instance,
        eval_dataset_path=eval_dataset_path
    )

    if not evaluator.eval_dataset:
        logger.error("RAG evaluation dataset is effectively empty. Aborting RAG evaluation.")
        return None

    rag_metrics = evaluator.evaluate(k=top_k)

    if rag_metrics:
        evaluator.print_results(rag_metrics)
        if wb_logger and wb_logger.wandb_run: 
            wb_logger.log_rag_evaluation_results(rag_metrics, eval_name=f"RAG_Eval_k{top_k}")
        return rag_metrics
    else:
        logger.warning("RAG evaluation did not produce metrics.")
        return None


async def run_synthesis_evaluation(
    wb_logger: WandBMetricsLogger,
    synth_eval_dataset_path_str: str, 
    judge_llm_model: Optional[str] # Note: provider for judge is derived from settings like other agents
) -> List[Dict[str, Any]]: 
    # Déterminer le provider du LLM Juge à partir des settings globaux
    judge_llm_provider = settings.DEFAULT_LLM_MODEL_PROVIDER 
    # Si judge_llm_model n'est pas surchargé, utiliser le modèle par défaut du provider
    effective_judge_model = judge_llm_model
    if not effective_judge_model:
        if judge_llm_provider == "openai":
            effective_judge_model = settings.DEFAULT_OPENAI_GENERATIVE_MODEL
        elif judge_llm_provider == "huggingface_api":
            effective_judge_model = settings.HUGGINGFACE_REPO_ID
        elif judge_llm_provider == "ollama":
            effective_judge_model = settings.OLLAMA_GENERATIVE_MODEL_NAME
        else:
            effective_judge_model = "default_model_unknown_provider"

    logger.info(f"\n--- Starting Synthesis Evaluation (Judge LLM Provider: {judge_llm_provider}, Model: {effective_judge_model}) ---")
    
    eval_dataset_path = Path(synth_eval_dataset_path_str)
    if not eval_dataset_path.exists():
        logger.error(f"Synthesis evaluation dataset not found at: {eval_dataset_path}. Aborting.")
        return []

    try:
        with open(eval_dataset_path, "r", encoding="utf-8") as f:
            synthesis_eval_items: List[SynthesisEvalItem] = json.load(f)
        logger.info(f"Loaded {len(synthesis_eval_items)} items for synthesis evaluation.")
    except Exception as e:
        logger.error(f"Failed to load or parse synthesis evaluation dataset: {e}", exc_info=True)
        return []

    if not synthesis_eval_items:
        logger.warning("Synthesis evaluation dataset is empty.")
        return []

    try:
        # SynthesisEvaluator utilisera DEFAULT_LLM_MODEL_PROVIDER par défaut pour son juge,
        # sauf si judge_llm_model (nom du modèle) est spécifié, auquel cas il essaiera de l'utiliser
        # avec ce provider.
        evaluator = SynthesisEvaluator(
            judge_llm_provider=None, # Sera pris de settings.DEFAULT_LLM_MODEL_PROVIDER par get_llm
            judge_llm_model_name=judge_llm_model # Peut surcharger le nom du modèle pour le provider par défaut
        )
    except Exception as e:
        logger.error(f"Failed to initialize SynthesisEvaluator: {e}", exc_info=True)
        return []
        
    all_item_eval_results = []
    aggregated_scores: Dict[str, List[float]] = {"relevance": [], "faithfulness": []}

    for i, item in enumerate(synthesis_eval_items):
        logger.info(f"Evaluating synthesis for query_id: {item.get('query_id', 'Unknown')} ({i+1}/{len(synthesis_eval_items)})")
        
        if not all(k in item for k in ["query_text", "context", "synthesis_to_evaluate"]):
            logger.warning(f"Skipping item {item.get('query_id', 'UnknownID')} due to missing fields.")
            continue

        synth_result = await evaluator.evaluate_synthesis(
            query=item["query_text"],
            synthesis=item["synthesis_to_evaluate"],
            context=item["context"]
        )
        evaluator.print_results(synth_result, query=item["query_text"])
        
        item_log = {
            "query_id": item.get("query_id", f"item_{i+1}"),
            "query_text": item["query_text"]
        }
        if synth_result["relevance"]:
            item_log["relevance_score"] = synth_result["relevance"]["score"]
            item_log["relevance_reasoning"] = synth_result["relevance"]["reasoning"]
            aggregated_scores["relevance"].append(synth_result["relevance"]["score"])
        if synth_result["faithfulness"]:
            item_log["faithfulness_score"] = synth_result["faithfulness"]["score"]
            item_log["faithfulness_reasoning"] = synth_result["faithfulness"]["reasoning"]
            aggregated_scores["faithfulness"].append(synth_result["faithfulness"]["score"])
        
        all_item_eval_results.append(item_log)

    avg_metrics_to_log = {}
    if aggregated_scores["relevance"]:
        avg_metrics_to_log["Avg_Synthesis_Relevance"] = sum(aggregated_scores["relevance"]) / len(aggregated_scores["relevance"])
    if aggregated_scores["faithfulness"]:
        avg_metrics_to_log["Avg_Synthesis_Faithfulness"] = sum(aggregated_scores["faithfulness"]) / len(aggregated_scores["faithfulness"])
    
    if wb_logger and wb_logger.wandb_run and avg_metrics_to_log:
        wb_logger.log_summary_metrics(avg_metrics_to_log)
    
    if wb_logger and wb_logger.wandb_run and all_item_eval_results:
        try:
            synth_details_df = pd.DataFrame(all_item_eval_results)
            wb_logger.log_dataframe_as_table(synth_details_df, "Synthesis_Evaluation_Details")
        except ImportError:
            logger.warning("Pandas not installed, cannot log synthesis details as W&B Table.")
        except Exception as e_df:
            logger.error(f"Error logging synthesis details DataFrame to W&B: {e_df}")
            
    return all_item_eval_results


async def async_main(args: argparse.Namespace):
    """Fonction asynchrone principale pour l'orchestration des évaluations."""
    
    config_for_wandb = vars(args).copy() 
    config_for_wandb.pop('wandb_disabled', None) 

    # --- AJOUT/MODIFICATION : Enrichir config_for_wandb avec les détails d'embedding actifs ---
    active_embedding_provider = settings.DEFAULT_EMBEDDING_PROVIDER.lower()
    config_for_wandb["active_embedding_provider"] = active_embedding_provider
    if active_embedding_provider == "openai":
        config_for_wandb["active_embedding_model"] = settings.OPENAI_EMBEDDING_MODEL_NAME
        config_for_wandb["active_embedding_dimension"] = settings.OPENAI_EMBEDDING_DIMENSION
    elif active_embedding_provider == "huggingface":
        config_for_wandb["active_embedding_model"] = settings.HUGGINGFACE_EMBEDDING_MODEL_NAME
        config_for_wandb["active_embedding_dimension"] = settings.HUGGINGFACE_EMBEDDING_MODEL_DIMENSION
    elif active_embedding_provider == "ollama":
        config_for_wandb["active_embedding_model"] = settings.OLLAMA_EMBEDDING_MODEL_NAME
        config_for_wandb["active_embedding_dimension"] = settings.OLLAMA_EMBEDDING_MODEL_DIMENSION
    # --- FIN DE L'AJOUT/MODIFICATION ---
    
    wb_logger_instance = WandBMetricsLogger(
        project_name=args.wandb_project,
        run_name=args.wandb_run_name,
        tags=args.wandb_tags.split(',') if args.wandb_tags else None,
        config_to_log=config_for_wandb, # config_for_wandb inclut maintenant les détails d'embedding
        disabled=args.wandb_disabled
    )
    if not wb_logger_instance.is_disabled:
        wb_logger_instance.start_run()

    if args.eval_type == "rag" or args.eval_type == "all":
        await run_rag_evaluation(
            wb_logger=wb_logger_instance,
            rag_eval_dataset_path_str=args.rag_dataset,
            collection_name=args.collection_name,
            vector_index_name=args.vector_index_name,
            top_k=args.rag_top_k
        )

    if args.eval_type == "synthesis" or args.eval_type == "all":
        if not args.synthesis_dataset:
            logger.error("Synthesis evaluation requested but --synthesis_dataset path not provided. Skipping.")
        else:
            await run_synthesis_evaluation(
                wb_logger=wb_logger_instance,
                synth_eval_dataset_path_str=args.synthesis_dataset,
                judge_llm_model=args.judge_llm_model # Le nom du modèle juge
            )
    
    if not wb_logger_instance.is_disabled and wb_logger_instance.wandb_run:
        wb_logger_instance.end_run()


def main():
    parser = argparse.ArgumentParser(description="Cognitive Swarm: Evaluation Pipeline.")
    parser.add_argument(
        "--eval_type", type=str, default="all", choices=["rag", "synthesis", "all"],
        help="Type of evaluation to run (default: all)."
    )
    # Args RAG
    parser.add_argument("--rag_dataset", type=str, default=settings.EVALUATION_DATASET_PATH, help="Path to RAG evaluation JSON dataset.")
    parser.add_argument("--rag_top_k", type=int, default=5, help="Top K retrievals for RAG evaluation.")
    parser.add_argument("--collection_name", type=str, default=MongoDBManager.DEFAULT_CHUNK_COLLECTION_NAME, help="MongoDB collection for RAG.")
    parser.add_argument("--vector_index_name", type=str, default=MongoDBManager.DEFAULT_VECTOR_INDEX_NAME, help="MongoDB vector index for RAG.")
    
    # Args Synthèse
    parser.add_argument("--synthesis_dataset", type=str, help="Path to Synthesis evaluation JSON dataset (required if eval_type includes 'synthesis').")
    # Modification: clarifier que judge_llm_model est le *nom* du modèle, le provider sera celui par défaut.
    parser.add_argument("--judge_llm_model", type=Optional[str], default=None, 
                        help="Name of the LLM model to use as judge for synthesis evaluation. "
                             "If None, the default generative model for the configured provider (settings.DEFAULT_LLM_MODEL_PROVIDER) will be used.")

    # Args W&B
    parser.add_argument("--wandb_project", type=str, default="CognitiveSwarm-Evaluations", help="Weights & Biases project name.")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="Weights & Biases run name (autogenerated if None).")
    parser.add_argument("--wandb_tags", type=str, default="evaluation", help="Comma-separated tags for W&B run.")
    parser.add_argument("--wandb_disabled", action="store_true", help="Disable W&B logging for this run.")
    
    parser.add_argument(
        "--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level (default: INFO)."
    )
    
    args = parser.parse_args()
    setup_logging(level=args.log_level.upper())

    if not args.wandb_disabled and not settings.WANDB_API_KEY and not os.environ.get("WANDB_API_KEY"):
        logger.warning("WANDB_API_KEY not found. W&B logging might fail if not logged in via CLI.")
    
    # Vérification pour le LLM juge - il utilisera le provider configuré dans settings.py
    # et le modèle spécifié par --judge_llm_model, ou le modèle par défaut du provider.
    # La clé API correspondante au provider doit être disponible.
    llm_judge_provider = settings.DEFAULT_LLM_MODEL_PROVIDER.lower()
    if (args.eval_type == "synthesis" or args.eval_type == "all"):
        if llm_judge_provider == "openai" and not settings.OPENAI_API_KEY:
            logger.error("OpenAI API key not set. Synthesis evaluation (LLM-as-judge with OpenAI) will fail if OpenAI is the selected provider.")
        elif llm_judge_provider == "huggingface_api" and not settings.HUGGINGFACE_API_KEY:
            logger.error("HuggingFace API key not set. Synthesis evaluation (LLM-as-judge with HuggingFace API) will fail if HuggingFace API is the selected provider.")
        # Pour Ollama, on suppose qu'il est accessible via OLLAMA_BASE_URL s'il est le provider.
        
    try:
        asyncio.run(async_main(args))
    except KeyboardInterrupt:
        logger.info("Evaluation process interrupted by user.")
        print("\nEvaluation interrupted.")
    except Exception as e: 
        logger.critical(f"A critical error occurred in the evaluation script: {e}", exc_info=True)
        print(f"\nA critical error occurred: {e}")

if __name__ == "__main__":
    main()