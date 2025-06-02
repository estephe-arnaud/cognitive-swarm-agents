# src/evaluation/rag_evaluator.py
import logging
import json
from pathlib import Path
from typing import List, Dict, Any, TypedDict, Optional

# Supposons que RetrievalEngine est correctement initialisé et fonctionnel
from src.rag.retrieval_engine import RetrievalEngine, RetrievedNode
from config.settings import settings

logger = logging.getLogger(__name__)

class EvalQuery(TypedDict):
    query_id: str
    query_text: str
    expected_relevant_chunk_ids: List[str]

class RagEvaluationMetrics(TypedDict):
    hit_rate_at_k: float
    mrr_at_k: float
    average_precision_at_k: float
    num_queries: int
    k: int

class RagEvaluator:
    def __init__(self, retrieval_engine: RetrievalEngine, eval_dataset_path: Optional[Path] = None):
        self.retrieval_engine = retrieval_engine
        self.eval_dataset: List[EvalQuery] = []
        # MODIFICATION: Store the dataset path
        self.dataset_source_path: Optional[Path] = None

        if eval_dataset_path:
            self.dataset_source_path = eval_dataset_path # Store the provided path
            self.load_dataset(eval_dataset_path)
        else:
            logger.info("No evaluation dataset path provided, using a small default demo dataset.")
            self.eval_dataset = self._get_default_demo_dataset()
            # self.dataset_source_path remains None, or could be set to a placeholder string
            # For clarity with W&B logging, keeping it None if default is used is fine,
            # as the logging logic will handle it.

        if not self.eval_dataset:
            logger.warning("RAG evaluation dataset is empty. Evaluator may not produce meaningful results.")

    def _get_default_demo_dataset(self) -> List[EvalQuery]:
        """Provides a very small, generic demo dataset if none is loaded."""
        return [
            {
                "query_id": "demo_q1",
                "query_text": "What are common methods for robot arm path planning?",
                "expected_relevant_chunk_ids": ["db_test001_chunk_001", "some_other_relevant_chunk_id"]
            },
            {
                "query_id": "demo_q2",
                "query_text": "How is reinforcement learning applied to drone navigation?",
                "expected_relevant_chunk_ids": ["db_test002_chunk_001"]
            },
            {
                "query_id": "demo_q3",
                "query_text": "Applications of quantum computing in ancient history.",
                "expected_relevant_chunk_ids": []
            }
        ]

    def load_dataset(self, dataset_path: Path) -> None:
        """Loads the evaluation dataset from a JSON file."""
        logger.info(f"Loading RAG evaluation dataset from: {dataset_path}")
        # MODIFICATION: Store the path if not already stored via __init__ (though it should be)
        if self.dataset_source_path is None:
             self.dataset_source_path = dataset_path

        try:
            with open(dataset_path, "r", encoding="utf-8") as f:
                self.eval_dataset = json.load(f)
            logger.info(f"Successfully loaded {len(self.eval_dataset)} queries from dataset.")
        except FileNotFoundError:
            logger.error(f"Evaluation dataset file not found: {dataset_path}")
            self.eval_dataset = self._get_default_demo_dataset()
            self.dataset_source_path = None # Reset path if fallback to default
            logger.warning(f"Using default demo dataset due to FileNotFoundError.")
        except json.JSONDecodeError:
            logger.error(f"Error decoding JSON from evaluation dataset: {dataset_path}", exc_info=True)
            self.eval_dataset = self._get_default_demo_dataset()
            self.dataset_source_path = None # Reset path if fallback to default
            logger.warning(f"Using default demo dataset due to JSONDecodeError.")
        except Exception as e:
            logger.error(f"An unexpected error occurred loading dataset: {e}", exc_info=True)
            self.eval_dataset = self._get_default_demo_dataset()
            self.dataset_source_path = None # Reset path if fallback to default
            logger.warning(f"Using default demo dataset due to an unexpected error.")

    def evaluate(self, k: int = 5) -> Optional[RagEvaluationMetrics]:
        if not self.eval_dataset:
            logger.error("Evaluation dataset is empty. Cannot run evaluation.")
            return None
        if self.retrieval_engine is None:
            logger.error("RetrievalEngine is not initialized. Cannot run evaluation.")
            return None

        logger.info(f"Starting RAG retrieval evaluation with k={k} for {len(self.eval_dataset)} queries.")

        hits = 0
        sum_reciprocal_ranks = 0.0
        sum_precision_at_k = 0.0
        
        num_queries_with_relevant_docs = 0

        for i, eval_query in enumerate(self.eval_dataset):
            query_text = eval_query["query_text"]
            expected_ids = set(eval_query["expected_relevant_chunk_ids"])
            
            logger.debug(f"Evaluating query {i+1}/{len(self.eval_dataset)}: '{query_text}' (Expected: {expected_ids})")

            try:
                retrieved_nodes: List[RetrievedNode] = self.retrieval_engine.retrieve_simple_vector_search(
                    query_text=query_text,
                    top_k=k
                )
            except Exception as e:
                logger.error(f"Error retrieving documents for query '{query_text}': {e}", exc_info=True)
                continue

            retrieved_chunk_ids = [node.metadata.get("chunk_id") for node in retrieved_nodes if node.metadata.get("chunk_id")]
            logger.debug(f"Retrieved {len(retrieved_chunk_ids)} chunk IDs: {retrieved_chunk_ids}")

            if not expected_ids:
                logger.debug(f"Query '{query_text}' has no expected relevant documents. Skipping for HitRate/MRR calculation.")
                if not retrieved_chunk_ids:
                    sum_precision_at_k += 1.0
                else:
                    sum_precision_at_k += 0.0
                continue

            num_queries_with_relevant_docs += 1
            found_relevant_in_top_k = False
            first_relevant_rank = 0

            for rank, chunk_id in enumerate(retrieved_chunk_ids):
                if chunk_id in expected_ids:
                    if not found_relevant_in_top_k:
                        hits += 1
                        found_relevant_in_top_k = True
                    if first_relevant_rank == 0:
                        first_relevant_rank = rank + 1
            
            if first_relevant_rank > 0:
                sum_reciprocal_ranks += (1.0 / first_relevant_rank)
            
            relevant_retrieved_count = len(set(retrieved_chunk_ids) & expected_ids)
            precision_at_k_for_query = relevant_retrieved_count / len(retrieved_chunk_ids) if retrieved_chunk_ids else (1.0 if not expected_ids else 0.0)
            sum_precision_at_k += precision_at_k_for_query

        final_hit_rate = (hits / num_queries_with_relevant_docs) if num_queries_with_relevant_docs > 0 else 0.0
        final_mrr = (sum_reciprocal_ranks / num_queries_with_relevant_docs) if num_queries_with_relevant_docs > 0 else 0.0
        final_avg_precision_at_k = (sum_precision_at_k / len(self.eval_dataset)) if self.eval_dataset else 0.0

        metrics: RagEvaluationMetrics = {
            "hit_rate_at_k": final_hit_rate,
            "mrr_at_k": final_mrr,
            "average_precision_at_k": final_avg_precision_at_k,
            "num_queries": len(self.eval_dataset),
            "k": k
        }
        logger.info(f"RAG Evaluation Metrics (k={k}): {metrics}")
        return metrics

    def print_results(self, metrics: RagEvaluationMetrics):
        print("\n--- RAG Retrieval Evaluation Results ---")
        print(f"Number of Test Queries: {metrics['num_queries']}")
        print(f"K (Top N Results Considered): {metrics['k']}")
        print(f"Hit Rate@{metrics['k']}: {metrics['hit_rate_at_k']:.4f}")
        print(f"MRR@{metrics['k']}: {metrics['mrr_at_k']:.4f}")
        print(f"Average Precision@{metrics['k']}: {metrics['average_precision_at_k']:.4f}")
        print("--------------------------------------")

if __name__ == "__main__":
    from config.logging_config import setup_logging
    setup_logging(level="INFO")
    logger.info("--- Starting RAG Evaluator Test Run ---")

    if not settings.OPENAI_API_KEY and not (settings.DEFAULT_EMBEDDING_PROVIDER == "huggingface" or (settings.DEFAULT_EMBEDDING_PROVIDER == "ollama" and settings.OLLAMA_BASE_URL)):
        logger.error("Required configuration for the default embedding provider is missing. RetrievalEngine (and thus RAG Evaluator) may fail.")
    
    try:
        retrieval_engine_instance = RetrievalEngine()
        
        # Test with a specified dataset path
        # Create a dummy dataset for testing if it doesn't exist
        dummy_dataset_path = Path(settings.DATA_DIR) / "evaluation" / "dummy_rag_eval_dataset.json"
        dummy_dataset_path.parent.mkdir(parents=True, exist_ok=True)
        if not dummy_dataset_path.exists():
            with open(dummy_dataset_path, "w") as f:
                json.dump([
                    {
                        "query_id": "dummy_q1",
                        "query_text": "Test query for dummy dataset.",
                        "expected_relevant_chunk_ids": ["id_that_might_exist_123"]
                    }
                ], f)
            logger.info(f"Created dummy dataset at {dummy_dataset_path}")

        evaluator_with_file = RagEvaluator(
            retrieval_engine=retrieval_engine_instance,
            eval_dataset_path=dummy_dataset_path # Use the dummy path
        )
        assert evaluator_with_file.dataset_source_path == dummy_dataset_path
        logger.info(f"Evaluator initialized with file, dataset source path: {evaluator_with_file.dataset_source_path}")
        
        if evaluator_with_file.retrieval_engine and evaluator_with_file.eval_dataset:
            evaluation_metrics_file = evaluator_with_file.evaluate(k=3)
            if evaluation_metrics_file:
                evaluator_with_file.print_results(evaluation_metrics_file)
        
        # Test with default internal dataset
        evaluator_default = RagEvaluator(retrieval_engine=retrieval_engine_instance)
        assert evaluator_default.dataset_source_path is None # Default dataset has no source path stored
        logger.info(f"Evaluator initialized with default dataset, source path: {evaluator_default.dataset_source_path}")
        if evaluator_default.retrieval_engine and evaluator_default.eval_dataset:
            evaluation_metrics_default = evaluator_default.evaluate(k=3)
            if evaluation_metrics_default:
                evaluator_default.print_results(evaluation_metrics_default)

    except ValueError as ve:
        logger.error(f"Could not initialize RetrievalEngine for RAG Evaluator: {ve}", exc_info=True)
    except Exception as e:
        logger.error(f"An unexpected error occurred during RAG Evaluator test run: {e}", exc_info=True)

    logger.info("--- RAG Evaluator Test Run Finished ---")