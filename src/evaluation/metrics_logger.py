"""
Weights & Biases Metrics Logger Module

This module provides integration with Weights & Biases (W&B) for tracking experiment
configurations and evaluation metrics. It supports logging of:
- Experiment configurations
- Summary metrics
- Step-by-step metrics
- RAG evaluation results
- Synthesis evaluation results
- DataFrames as tables

The logger handles cases where W&B is not installed or disabled, and provides
proper error handling and logging throughout.
"""

import logging
import os
from typing import Dict, Any, Optional, List, TYPE_CHECKING
import pandas as pd

# Conditional import for type hinting only
if TYPE_CHECKING:
    import wandb
    from wandb.sdk.wandb_run import Run as WandbRunType

# Actual import for runtime use
try:
    import wandb
except ImportError:
    wandb = None
    print("Warning: 'wandb' library not found. WandBMetricsLogger will not function. Pip install wandb.")

from config.settings import settings
from src.evaluation.rag_evaluator import RagEvaluationMetrics
from src.evaluation.synthesis_evaluator import SynthesisEvaluationResult

logger = logging.getLogger(__name__)

class WandBMetricsLogger:
    """
    A logger class to integrate with Weights & Biases for tracking experiment
    configurations and evaluation metrics.
    
    This class provides methods to:
    - Initialize and manage W&B runs
    - Log experiment configurations
    - Log various types of metrics (summary, step-by-step)
    - Log evaluation results (RAG, synthesis)
    - Log DataFrames as tables
    """
    
    def __init__(
        self,
        project_name: str = "MAKERS-Experiments",
        entity: Optional[str] = None,
        run_name: Optional[str] = None,
        config_to_log: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        disabled: bool = False
    ):
        """
        Initialize the W&B metrics logger.
        
        Args:
            project_name: Name of the W&B project
            entity: W&B entity (user or team)
            run_name: Name for this specific run
            config_to_log: Initial configuration to log
            tags: List of tags for the run
            disabled: Whether to disable W&B logging
        """
        # Initialize wandb_run attribute type based on TYPE_CHECKING
        if TYPE_CHECKING:
            self.wandb_run: Optional[WandbRunType] = None
        else:
            self.wandb_run: Optional[Any] = None

        if wandb is None and not disabled:
            logger.warning("W&B library not installed, but logger is not disabled. No metrics will be logged to W&B.")
            self.is_disabled = True
            return
        
        self.is_disabled = disabled
        if self.is_disabled:
            logger.info("W&B logging is disabled for this instance.")
            return

        self.project_name = project_name
        self.entity = entity
        self.run_name = run_name
        self.config_to_log = config_to_log or {}
        self.tags = tags or []

        if not settings.WANDB_API_KEY and not os.environ.get("WANDB_API_KEY"):
            logger.warning("WANDB_API_KEY not found in settings or environment. W&B logging might fail if not already logged in via CLI.")

    def start_run(self) -> Optional[Any]:
        """
        Initialize and start a new W&B run.
        
        Returns:
            Optional[Any]: The W&B run instance if successful, None otherwise
        """
        if self.is_disabled or wandb is None:
            logger.info("W&B logging disabled or library not available. Skipping start_run.")
            return None

        # Check if a run is already active
        current_run = wandb.run
        if current_run and current_run.id and (self.wandb_run and self.wandb_run.id == current_run.id):
            logger.warning(f"W&B run '{current_run.name}' is already active and matches this logger instance. Not starting a new one.")
            return self.wandb_run
        elif current_run and current_run.id:
            logger.warning(f"An existing W&B run '{current_run.name}' (ID: {current_run.id}) is active globally. Reinitializing for this logger instance.")

        try:
            initialized_run = wandb.init(
                project=self.project_name,
                entity=self.entity,
                name=self.run_name,
                config=self.config_to_log,
                tags=self.tags,
                reinit=True,
            )
            
            if initialized_run is None:
                logger.warning("wandb.init() returned None. W&B run might be disabled globally or an issue occurred.")
                self.is_disabled = True
                self.wandb_run = None
                return None
            
            self.wandb_run = initialized_run
            logger.info(f"W&B run started/reinitialized. Name: {self.wandb_run.name}, ID: {self.wandb_run.id}")
            return self.wandb_run
        except Exception as e:
            logger.error(f"Failed to initialize W&B run: {e}", exc_info=True)
            self.is_disabled = True
            self.wandb_run = None
            return None

    def log_configuration(self, config_dict: Dict[str, Any]) -> None:
        """
        Log configuration to the current W&B run.
        
        Args:
            config_dict: Dictionary of configuration values to log
        """
        if not self.wandb_run or self.is_disabled or wandb is None:
            logger.debug("W&B run not active or logger disabled. Skipping log_configuration.")
            return
        try:
            wandb.config.update(config_dict, allow_val_change=True)
            logger.info(f"Logged configuration to W&B: {config_dict}")
        except Exception as e:
            logger.error(f"Failed to log configuration to W&B: {e}", exc_info=True)

    def log_summary_metrics(self, metrics_dict: Dict[str, Any]) -> None:
        """
        Log summary metrics to the current W&B run.
        
        Args:
            metrics_dict: Dictionary of metrics to log as summary
        """
        if not self.wandb_run or self.is_disabled or wandb is None:
            logger.debug("W&B run not active or logger disabled. Skipping log_summary_metrics.")
            return
        try:
            for key, value in metrics_dict.items():
                wandb.summary[key] = value
            logger.info(f"Logged summary metrics to W&B: {metrics_dict}")
        except Exception as e:
            logger.error(f"Failed to log summary metrics to W&B: {e}", exc_info=True)

    def log_metrics_step(self, metrics_dict: Dict[str, Any], step: Optional[int] = None) -> None:
        """
        Log metrics for a specific step to the current W&B run.
        
        Args:
            metrics_dict: Dictionary of metrics to log
            step: Optional step number for the metrics
        """
        if not self.wandb_run or self.is_disabled or wandb is None:
            logger.debug("W&B run not active or logger disabled. Skipping log_metrics_step.")
            return
        try:
            if step is not None:
                wandb.log(metrics_dict, step=step)
            else:
                wandb.log(metrics_dict)
            logger.info(f"Logged metrics at step {step if step is not None else 'current'} to W&B: {metrics_dict}")
        except Exception as e:
            logger.error(f"Failed to log metrics step to W&B: {e}", exc_info=True)

    def log_rag_evaluation_results(self, rag_metrics: RagEvaluationMetrics, eval_name: str = "RAG_Evaluation") -> None:
        """
        Log RAG evaluation results to the current W&B run.
        
        Args:
            rag_metrics: RAG evaluation metrics to log
            eval_name: Name prefix for the metrics
        """
        if not self.wandb_run or self.is_disabled or wandb is None:
            return
        metrics_to_log = {
            f"{eval_name}/hit_rate_at_{rag_metrics['k']}": rag_metrics['hit_rate_at_k'],
            f"{eval_name}/mrr_at_{rag_metrics['k']}": rag_metrics['mrr_at_k'],
            f"{eval_name}/avg_precision_at_{rag_metrics['k']}": rag_metrics['average_precision_at_k'],
            f"{eval_name}/num_queries": rag_metrics['num_queries'],
        }
        self.log_summary_metrics(metrics_to_log)

    def log_synthesis_evaluation_results(self, synth_eval: SynthesisEvaluationResult, eval_name: str = "Synthesis_Evaluation") -> None:
        """
        Log synthesis evaluation results to the current W&B run.
        
        Args:
            synth_eval: Synthesis evaluation results to log
            eval_name: Name prefix for the metrics
        """
        if not self.wandb_run or self.is_disabled or wandb is None:
            return
        metrics_to_log = {}
        if synth_eval.get("relevance") and synth_eval["relevance"] is not None:
            metrics_to_log[f"{eval_name}/relevance_score"] = synth_eval["relevance"]["score"]
        if synth_eval.get("faithfulness") and synth_eval["faithfulness"] is not None:
            metrics_to_log[f"{eval_name}/faithfulness_score"] = synth_eval["faithfulness"]["score"]
        if metrics_to_log:
            self.log_summary_metrics(metrics_to_log)

    def log_dataframe_as_table(self, df: pd.DataFrame, table_name: str) -> None:
        """
        Log a DataFrame as a table to the current W&B run.
        
        Args:
            df: DataFrame to log
            table_name: Name for the table in W&B
        """
        if not self.wandb_run or self.is_disabled or wandb is None:
            logger.debug(f"W&B run not active or logger disabled. Skipping log_dataframe_as_table for '{table_name}'.")
            return
        try:
            wandb_table = wandb.Table(dataframe=df)
            wandb.log({table_name: wandb_table})
            logger.info(f"Logged DataFrame as W&B Table: '{table_name}'")
        except Exception as e:
            logger.error(f"Failed to log DataFrame as W&B Table '{table_name}': {e}", exc_info=True)

    def end_run(self, exit_code: Optional[int] = None) -> None:
        """
        End the current W&B run.
        
        Args:
            exit_code: Optional exit code to log
        """
        if not self.wandb_run or self.is_disabled or wandb is None:
            logger.info("W&B run not active or logger disabled/unavailable. Skipping end_run.")
            return
        
        active_run = self.wandb_run or wandb.run
        if not active_run:
            logger.info("No active W&B run to finish.")
            return

        try:
            run_name_for_log = active_run.name if hasattr(active_run, 'name') else 'Unknown'
            if exit_code is not None and exit_code != 0:
                wandb.finish(exit_code=exit_code, quiet=True)
            else:
                wandb.finish(quiet=True)
            logger.info(f"W&B run '{run_name_for_log}' finished.")
            if self.wandb_run and active_run.id == self.wandb_run.id:
                self.wandb_run = None
        except Exception as e:
            logger.error(f"Error finishing W&B run: {e}", exc_info=True)
            if self.wandb_run and active_run and active_run.id == self.wandb_run.id:
                self.wandb_run = None


if __name__ == "__main__":
    from config.logging_config import setup_logging
    setup_logging(level="DEBUG")

    logger.info("--- Testing WandBMetricsLogger ---")

    if wandb is None:
        logger.error("wandb library is not installed. Cannot run W&B logger tests.")
    elif not settings.WANDB_API_KEY and not os.environ.get("WANDB_API_KEY") and not (hasattr(wandb, 'api') and wandb.api.api_key):
        logger.warning("WANDB_API_KEY not set and not logged in via CLI. W&B operations might fail or prompt for login.")
        logger.warning("Skipping WandBMetricsLogger actual remote logging test to avoid interactive prompts.")
        disabled_logger = WandBMetricsLogger(disabled=True)
        assert disabled_logger.is_disabled
        disabled_logger.start_run()
        logger.info("Tested disabled logger: OK")
    else:
        # Test configuration
        test_config = {
            "llm_model": settings.DEFAULT_OPENAI_MODEL,
            "embedding_model": settings.DEFAULT_EMBEDDING_MODEL,
        }
        
        # Initialize logger
        wb_logger = WandBMetricsLogger(
            project_name="MAKERS-Tests",
            run_name="Test Run - Metrics Logger",
            config_to_log=test_config,
            tags=["test"]
        )

        # Start run and log metrics
        run_instance = wb_logger.start_run()
        if run_instance:
            run_url = run_instance.get_url() if hasattr(run_instance, 'get_url') else 'N/A'
            logger.info(f"W&B Run URL: {run_url}")

            # Log summary metrics
            summary_data = {"final_accuracy": 0.98, "total_cost": 1.23}
            wb_logger.log_summary_metrics(summary_data)

            # Log step metrics
            for i_step in range(3):
                wb_logger.log_metrics_step({"step_loss": 1.0 / (i_step + 1)}, step=i_step)

            # Log RAG evaluation results
            sample_rag: RagEvaluationMetrics = {
                "hit_rate_at_k": 0.85,
                "mrr_at_k": 0.70,
                "average_precision_at_k": 0.75,
                "num_queries": 20,
                "k": 3
            }
            wb_logger.log_rag_evaluation_results(sample_rag)

            # Log synthesis evaluation results
            sample_synth: SynthesisEvaluationResult = {
                "relevance": {"score": 0.92, "reasoning": "Relevant."},
                "faithfulness": {"score": 0.80, "reasoning": "Mostly faithful."},
                "coherence": None
            }
            wb_logger.log_synthesis_evaluation_results(sample_synth)

            # End run
            wb_logger.end_run()
            logger.info("W&B test run completed and finished successfully.")
        else:
            logger.error("Failed to start W&B run for testing. Check W&B setup and API key.")