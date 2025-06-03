# makers/scripts/run_evaluation.py
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
from src.evaluation.rag_evaluator import RagEvaluator, RagEvaluationMetrics
from src.evaluation.synthesis_evaluator import SynthesisEvaluator
from src.evaluation.metrics_logger import WandBMetricsLogger
from src.vector_store.mongodb_manager import MongoDBManager

logger = logging.getLogger(__name__)

class SynthesisEvalItem(TypedDict):
    query_id: str
    query_text: str
    context: str
    synthesis_to_evaluate: str

def get_judge_model(provider: str, model_name: Optional[str] = None) -> str:
    """Get the appropriate judge model name based on provider and optional override."""
    if model_name:
        return model_name

    if provider == "openai":
        return settings.DEFAULT_OPENAI_GENERATIVE_MODEL
    elif provider == "huggingface_api":
        return settings.HUGGINGFACE_REPO_ID
    elif provider == "ollama":
        return settings.OLLAMA_GENERATIVE_MODEL_NAME
    return "default_model_unknown_provider"

async def run_rag_evaluation(
    wb_logger: WandBMetricsLogger,
    rag_eval_dataset_path_str: Optional[str],
    collection_name: str,
    vector_index_name: str,
    top_k: int
) -> Optional[RagEvaluationMetrics]:
    """Run RAG evaluation pipeline."""
    logger.info(f"\n--- Starting RAG Evaluation (k={top_k}) ---")
    
    try:
        retrieval_engine = RetrievalEngine(
            collection_name=collection_name,
            vector_index_name=vector_index_name
        )
    except Exception as e:
        logger.error(f"Failed to initialize RetrievalEngine: {e}", exc_info=True)
        return None

    eval_dataset_path = Path(rag_eval_dataset_path_str) if rag_eval_dataset_path_str else None
    if eval_dataset_path and not eval_dataset_path.exists():
        logger.error(f"RAG evaluation dataset not found: {eval_dataset_path}")
        return None
    
    evaluator = RagEvaluator(
        retrieval_engine=retrieval_engine,
        eval_dataset_path=eval_dataset_path
    )

    if not evaluator.eval_dataset:
        logger.error("Empty RAG evaluation dataset")
        return None

    metrics = evaluator.evaluate(k=top_k)
    if not metrics:
        logger.warning("No RAG metrics produced")
        return None

    evaluator.print_results(metrics)
    if wb_logger and wb_logger.wandb_run:
        wb_logger.log_rag_evaluation_results(metrics, eval_name=f"RAG_Eval_k{top_k}")
    
    return metrics

async def run_synthesis_evaluation(
    wb_logger: WandBMetricsLogger,
    synth_eval_dataset_path_str: str,
    judge_llm_model: Optional[str]
) -> List[Dict[str, Any]]:
    """Run synthesis evaluation pipeline."""
    provider = settings.DEFAULT_LLM_MODEL_PROVIDER
    model = get_judge_model(provider, judge_llm_model)
    logger.info(f"\n--- Starting Synthesis Evaluation (Provider: {provider}, Model: {model}) ---")
    
    eval_dataset_path = Path(synth_eval_dataset_path_str)
    if not eval_dataset_path.exists():
        logger.error(f"Synthesis dataset not found: {eval_dataset_path}")
        return []

    try:
        with open(eval_dataset_path, "r", encoding="utf-8") as f:
            eval_items: List[SynthesisEvalItem] = json.load(f)
        logger.info(f"Loaded {len(eval_items)} evaluation items")
    except Exception as e:
        logger.error(f"Failed to load synthesis dataset: {e}", exc_info=True)
        return []

    if not eval_items:
        logger.warning("Empty synthesis evaluation dataset")
        return []

    try:
        evaluator = SynthesisEvaluator(
            judge_llm_provider=None,
            judge_llm_model_name=judge_llm_model
        )
    except Exception as e:
        logger.error(f"Failed to initialize SynthesisEvaluator: {e}", exc_info=True)
        return []
        
    results = []
    scores = {"relevance": [], "faithfulness": []}

    for i, item in enumerate(eval_items):
        logger.info(f"Evaluating item {i+1}/{len(eval_items)}: {item.get('query_id', 'Unknown')}")
        
        if not all(k in item for k in ["query_text", "context", "synthesis_to_evaluate"]):
            logger.warning(f"Skipping item {item.get('query_id', 'UnknownID')} - missing fields")
            continue

        result = await evaluator.evaluate_synthesis(
            query=item["query_text"],
            synthesis=item["synthesis_to_evaluate"],
            context=item["context"]
        )
        evaluator.print_results(result, query=item["query_text"])
        
        item_result = {
            "query_id": item.get("query_id", f"item_{i+1}"),
            "query_text": item["query_text"]
        }

        if result["relevance"]:
            item_result["relevance_score"] = result["relevance"]["score"]
            item_result["relevance_reasoning"] = result["relevance"]["reasoning"]
            scores["relevance"].append(result["relevance"]["score"])

        if result["faithfulness"]:
            item_result["faithfulness_score"] = result["faithfulness"]["score"]
            item_result["faithfulness_reasoning"] = result["faithfulness"]["reasoning"]
            scores["faithfulness"].append(result["faithfulness"]["score"])
        
        results.append(item_result)

    # Calculate and log average metrics
    avg_metrics = {}
    if scores["relevance"]:
        avg_metrics["Avg_Synthesis_Relevance"] = sum(scores["relevance"]) / len(scores["relevance"])
    if scores["faithfulness"]:
        avg_metrics["Avg_Synthesis_Faithfulness"] = sum(scores["faithfulness"]) / len(scores["faithfulness"])
    
    if wb_logger and wb_logger.wandb_run:
        if avg_metrics:
            wb_logger.log_summary_metrics(avg_metrics)
        if results:
            try:
                results_df = pd.DataFrame(results)
                wb_logger.log_dataframe_as_table(results_df, "Synthesis_Evaluation_Details")
            except ImportError:
                logger.warning("Pandas not installed - cannot log details as W&B Table")
            except Exception as e:
                logger.error(f"Failed to log synthesis details to W&B: {e}")
            
    return results

async def async_main(args: argparse.Namespace):
    """Main async function for evaluation orchestration."""
    config = vars(args).copy()
    config.pop('wandb_disabled', None)

    # Add embedding details to config
    provider = settings.DEFAULT_EMBEDDING_PROVIDER.lower()
    config["active_embedding_provider"] = provider
    
    if provider == "openai":
        config["active_embedding_model"] = settings.OPENAI_EMBEDDING_MODEL_NAME
        config["active_embedding_dimension"] = settings.OPENAI_EMBEDDING_DIMENSION
    elif provider == "huggingface":
        config["active_embedding_model"] = settings.HUGGINGFACE_EMBEDDING_MODEL_NAME
        config["active_embedding_dimension"] = settings.HUGGINGFACE_EMBEDDING_MODEL_DIMENSION
    elif provider == "ollama":
        config["active_embedding_model"] = settings.OLLAMA_EMBEDDING_MODEL_NAME
        config["active_embedding_dimension"] = settings.OLLAMA_EMBEDDING_MODEL_DIMENSION
    
    wb_logger = WandBMetricsLogger(
        project_name=args.wandb_project,
        run_name=args.wandb_run_name,
        tags=args.wandb_tags.split(',') if args.wandb_tags else None,
        config_to_log=config,
        disabled=args.wandb_disabled
    )

    if not wb_logger.is_disabled:
        wb_logger.start_run()

    try:
        if args.eval_type in ["rag", "all"]:
            await run_rag_evaluation(
                wb_logger=wb_logger,
                rag_eval_dataset_path_str=args.rag_dataset,
                collection_name=args.collection_name,
                vector_index_name=args.vector_index_name,
                top_k=args.rag_top_k
            )

        if args.eval_type in ["synthesis", "all"]:
            if not args.synthesis_dataset:
                logger.error("Synthesis evaluation requested but no dataset provided")
            else:
                await run_synthesis_evaluation(
                    wb_logger=wb_logger,
                    synth_eval_dataset_path_str=args.synthesis_dataset,
                    judge_llm_model=args.judge_llm_model
                )
    finally:
        if not wb_logger.is_disabled and wb_logger.wandb_run:
            wb_logger.end_run()

def main():
    parser = argparse.ArgumentParser(description="MAKERS: Evaluation Pipeline")
    
    # Evaluation type
    parser.add_argument(
        "--eval_type",
        type=str,
        default="all",
        choices=["rag", "synthesis", "all"],
        help="Type of evaluation to run"
    )
    
    # RAG evaluation args
    parser.add_argument(
        "--rag_dataset",
        type=str,
        default=settings.EVALUATION_DATASET_PATH,
        help="Path to RAG evaluation dataset"
    )
    parser.add_argument(
        "--rag_top_k",
        type=int,
        default=5,
        help="Number of top retrievals for RAG evaluation"
    )
    parser.add_argument(
        "--collection_name",
        type=str,
        default=MongoDBManager.DEFAULT_CHUNK_COLLECTION_NAME,
        help="MongoDB collection name"
    )
    parser.add_argument(
        "--vector_index_name",
        type=str,
        default=MongoDBManager.DEFAULT_VECTOR_INDEX_NAME,
        help="MongoDB vector index name"
    )
    
    # Synthesis evaluation args
    parser.add_argument(
        "--synthesis_dataset",
        type=str,
        help="Path to synthesis evaluation dataset"
    )
    parser.add_argument(
        "--judge_llm_model",
        type=str,
        help="LLM model to use as judge (uses default model if not specified)"
    )

    # Weights & Biases args
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="MAKERS-Evaluations",
        help="W&B project name"
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        help="W&B run name (auto-generated if not specified)"
    )
    parser.add_argument(
        "--wandb_tags",
        type=str,
        default="evaluation",
        help="Comma-separated W&B tags"
    )
    parser.add_argument(
        "--wandb_disabled",
        action="store_true",
        help="Disable W&B logging"
    )

    args = parser.parse_args()
    setup_logging()

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
        logger.info("Evaluation interrupted by user")
    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()