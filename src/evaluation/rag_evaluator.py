# cognitive-swarm-agents/src/evaluation/rag_evaluator.py
import logging
import json
from pathlib import Path
from typing import List, Dict, Any, TypedDict, Optional, Tuple

# Supposons que RetrievalEngine est correctement initialisé et fonctionnel
from src.rag.retrieval_engine import RetrievalEngine, RetrievedNode 
from config.settings import settings

logger = logging.getLogger(__name__)

class EvalQuery(TypedDict):
    query_id: str
    query_text: str
    expected_relevant_chunk_ids: List[str] # IDs des chunks attendus comme pertinents
    # On pourrait ajouter expected_relevant_doc_ids (arxiv_ids) si on évalue au niveau du document

class RagEvaluationMetrics(TypedDict):
    hit_rate_at_k: float
    mrr_at_k: float
    average_precision_at_k: float # Moyenne des Precision@K pour toutes les requêtes
    num_queries: int
    k: int # La valeur de K utilisée pour l'évaluation

class RagEvaluator:
    def __init__(self, retrieval_engine: RetrievalEngine, eval_dataset_path: Optional[Path] = None):
        self.retrieval_engine = retrieval_engine
        self.eval_dataset: List[EvalQuery] = []

        if eval_dataset_path:
            self.load_dataset(eval_dataset_path)
        else:
            # Utiliser un petit jeu de données par défaut intégré pour la démonstration
            logger.info("No evaluation dataset path provided, using a small default demo dataset.")
            self.eval_dataset = self._get_default_demo_dataset()

        if not self.eval_dataset:
            logger.warning("RAG evaluation dataset is empty. Evaluator may not produce meaningful results.")

    def _get_default_demo_dataset(self) -> List[EvalQuery]:
        """Provides a very small, generic demo dataset if none is loaded."""
        # IMPORTANT: Ces IDs de chunks et requêtes sont factices.
        # Ils doivent être remplacés par des données réelles basées sur votre corpus.
        # Par exemple, après avoir ingéré des données, identifiez des chunks pertinents pour des requêtes types.
        return [
            {
                "query_id": "demo_q1",
                "query_text": "What are common methods for robot arm path planning?",
                # Supposez que ce sont des chunk_ids que vous avez identifiés comme pertinents dans votre DB
                "expected_relevant_chunk_ids": ["db_test001_chunk_001", "some_other_relevant_chunk_id"] 
            },
            {
                "query_id": "demo_q2",
                "query_text": "How is reinforcement learning applied to drone navigation?",
                "expected_relevant_chunk_ids": ["db_test002_chunk_001"]
            },
            {
                "query_id": "demo_q3", # Requête qui pourrait ne pas avoir de résultat pertinent
                "query_text": "Applications of quantum computing in ancient history.",
                "expected_relevant_chunk_ids": [] 
            }
        ]

    def load_dataset(self, dataset_path: Path) -> None:
        """Loads the evaluation dataset from a JSON file."""
        logger.info(f"Loading RAG evaluation dataset from: {dataset_path}")
        try:
            with open(dataset_path, "r", encoding="utf-8") as f:
                self.eval_dataset = json.load(f)
            logger.info(f"Successfully loaded {len(self.eval_dataset)} queries from dataset.")
        except FileNotFoundError:
            logger.error(f"Evaluation dataset file not found: {dataset_path}")
            self.eval_dataset = self._get_default_demo_dataset() # Fallback
            logger.warning(f"Using default demo dataset due to FileNotFoundError.")
        except json.JSONDecodeError:
            logger.error(f"Error decoding JSON from evaluation dataset: {dataset_path}", exc_info=True)
            self.eval_dataset = self._get_default_demo_dataset() # Fallback
            logger.warning(f"Using default demo dataset due to JSONDecodeError.")
        except Exception as e:
            logger.error(f"An unexpected error occurred loading dataset: {e}", exc_info=True)
            self.eval_dataset = self._get_default_demo_dataset() # Fallback
            logger.warning(f"Using default demo dataset due to an unexpected error.")


    def evaluate(self, k: int = 5) -> Optional[RagEvaluationMetrics]:
        """
        Runs the RAG retrieval evaluation.

        Args:
            k (int): The number of top results to consider for metrics (e.g., HitRate@k, MRR@k).

        Returns:
            Optional[RagEvaluationMetrics]: Calculated metrics, or None if evaluation cannot be run.
        """
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
        
        num_queries_with_relevant_docs = 0 # Queries that have at least one expected relevant doc

        for i, eval_query in enumerate(self.eval_dataset):
            query_text = eval_query["query_text"]
            expected_ids = set(eval_query["expected_relevant_chunk_ids"])
            
            logger.debug(f"Evaluating query {i+1}/{len(self.eval_dataset)}: '{query_text}' (Expected: {expected_ids})")

            # Retrieve top_k chunks using the RetrievalEngine
            # Assuming retrieve_simple_vector_search for now.
            # Filters could be added if eval_query contains filter specifications.
            try:
                retrieved_nodes: List[RetrievedNode] = self.retrieval_engine.retrieve_simple_vector_search(
                    query_text=query_text,
                    top_k=k 
                )
            except Exception as e:
                logger.error(f"Error retrieving documents for query '{query_text}': {e}", exc_info=True)
                continue # Skip this query if retrieval fails

            retrieved_chunk_ids = [node.metadata.get("chunk_id") for node in retrieved_nodes if node.metadata.get("chunk_id")]
            logger.debug(f"Retrieved {len(retrieved_chunk_ids)} chunk IDs: {retrieved_chunk_ids}")

            if not expected_ids: # If no relevant docs are expected for this query
                # Hit if nothing is retrieved, or if retrieved items are not in a predefined "positive" set (complex)
                # For simplicity, if nothing is expected, we don't count it for hit rate or MRR in this basic setup.
                # Precision@k would be undefined or 1.0 if nothing retrieved. Recall too.
                # We could skip these queries for these specific metrics or handle them differently.
                logger.debug(f"Query '{query_text}' has no expected relevant documents. Skipping for HitRate/MRR calculation.")
                # For Precision@k, if nothing is retrieved and nothing expected, it could be seen as perfect precision.
                # If something is retrieved but nothing expected, precision is 0.
                if not retrieved_chunk_ids: # Nothing retrieved, nothing expected
                    sum_precision_at_k += 1.0 
                else: # Something retrieved, nothing expected
                    sum_precision_at_k += 0.0
                continue # Skip to next query for HitRate/MRR calculations

            num_queries_with_relevant_docs += 1
            found_relevant_in_top_k = False
            first_relevant_rank = 0

            for rank, chunk_id in enumerate(retrieved_chunk_ids):
                if chunk_id in expected_ids:
                    if not found_relevant_in_top_k: # First time finding a relevant doc for this query in top_k
                        hits += 1
                        found_relevant_in_top_k = True
                    if first_relevant_rank == 0: # Record rank of the *very first* relevant doc found
                        first_relevant_rank = rank + 1
            
            if first_relevant_rank > 0:
                sum_reciprocal_ranks += (1.0 / first_relevant_rank)
            
            # Calculate Precision@k for this query
            relevant_retrieved_count = len(set(retrieved_chunk_ids) & expected_ids)
            precision_at_k_for_query = relevant_retrieved_count / len(retrieved_chunk_ids) if retrieved_chunk_ids else (1.0 if not expected_ids else 0.0)
            sum_precision_at_k += precision_at_k_for_query


        # Calculate final metrics
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
    # Assuming RetrievalEngine can be initialized; it needs OpenAI API key for its embed_model
    # and MongoDB URI + populated data for meaningful evaluation.

    setup_logging(level="INFO")
    logger.info("--- Starting RAG Evaluator Test Run ---")

    if not settings.OPENAI_API_KEY:
        logger.error("OPENAI_API_KEY not set. RetrievalEngine (and thus RAG Evaluator) may fail or produce poor results.")
        # Depending on RetrievalEngine's strictness, it might raise an error on init.
    
    # This test run uses the default demo dataset within RagEvaluator.
    # For a real run, provide a path to your JSON dataset.
    # Example: eval_dataset_file = Path(settings.EVALUATION_DATASET_PATH) if settings.EVALUATION_DATASET_PATH else None
    
    # Initialize RetrievalEngine (ensure MongoDB is accessible and populated for this to be meaningful)
    try:
        retrieval_engine_instance = RetrievalEngine() # Uses defaults from settings
        
        evaluator = RagEvaluator(retrieval_engine=retrieval_engine_instance) # Uses default demo dataset
        
        if not evaluator.eval_dataset:
            logger.error("Evaluation dataset is empty, cannot run test.")
        else:
            # Ensure retrieval_engine_instance is not None if it could fail softy
            if evaluator.retrieval_engine:
                evaluation_metrics = evaluator.evaluate(k=3) # Evaluate with k=3
                if evaluation_metrics:
                    evaluator.print_results(evaluation_metrics)
            else:
                logger.error("Retrieval Engine instance could not be initialized for evaluator.")

    except ValueError as ve: # Catch init errors from RetrievalEngine (e.g., missing API key)
        logger.error(f"Could not initialize RetrievalEngine for RAG Evaluator: {ve}", exc_info=True)
    except Exception as e:
        logger.error(f"An unexpected error occurred during RAG Evaluator test run: {e}", exc_info=True)

    logger.info("--- RAG Evaluator Test Run Finished ---")