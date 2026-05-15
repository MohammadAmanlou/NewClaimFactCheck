"""
Isolated Pipeline for Bias Evaluation.
Evaluates the entire dataset directly without temporal splitting, 
generating metrics for the full datset and comparing the performance 
of multiple prompt methods (e.g. strong_baseline_short vs cognitive_bias_aware).
"""

import json
import logging
from pathlib import Path
from datetime import datetime

from .config import Config
from .logging_config import setup_logger
from .data_loader import load_dataset
from .processing import process_claims
from .evaluation import calculate_metrics
from .reporting import NumpyEncoder


class BiasEvaluationPipeline:
    """Orchestrates an isolated pipeline that evaluates all claims without splitting."""

    def __init__(self, config: Config):
        self.config = config
        self.base_output = self._make_base_output()
        self.base_output.mkdir(parents=True, exist_ok=True)
        self.logger = setup_logger(self.base_output / "main_run_log.txt")
        self._print_header()

    def _make_base_output(self) -> Path:
        model_dir = self.config.model.replace(":", "_").replace(".", "_")
        return self.config.output_root / model_dir / self.config.dataset

    def _print_header(self) -> None:
        self.logger.info("=" * 80)
        self.logger.info("STARTING BIAS EVALUATION PIPELINE (ISOLATED)")
        self.logger.info("Dataset: %s, Model: %s", self.config.dataset, self.config.model)
        self.logger.info("Prompt methods: %s", self.config.prompt_methods)
        print(f"\n{'=' * 80}")
        print(f"Starting pipeline for {self.config.dataset}")
        print(f"Prompt methods: {self.config.prompt_methods}")
        print(f"{'=' * 80}\n")

    def run(self) -> None:
        claims, is_test = load_dataset(self.config.dataset, self.config)
        self.logger.info(f"Loaded {len(claims)} claims (is_test={is_test})")

        all_methods_results = {}
        for method in self.config.prompt_methods:
            print(f"\n--- Running method: {method} ---")
            out_dir = self.base_output / method
            out_dir.mkdir(exist_ok=True)
            method_logger = setup_logger(out_dir / "method_run_log.txt")
            method_logger.info("Prompt method: %s", method)

            results = process_claims(
                claims,
                is_test=is_test,
                config=self.config,
                logger=method_logger,
                prompt_method=method,
            )
            
            metrics = calculate_metrics(results, self.config.labels, method_logger)
            
            # Save predictions and metrics directly
            pred_file = out_dir / "predictions.json"
            with open(pred_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=4, ensure_ascii=False, cls=NumpyEncoder)
            
            metrics_file = out_dir / "metrics.json"
            with open(metrics_file, 'w', encoding='utf-8') as f:
                json.dump(metrics, f, indent=4, cls=NumpyEncoder)

            all_methods_results[method] = {
                "metrics": metrics,
            }

        self._save_combined_summary(all_methods_results)
        self.logger.info("=" * 80)
        self.logger.info("PIPELINE COMPLETE")
        print(f"\n✓ All outputs saved to: {self.base_output}")

    def _save_combined_summary(self, all_methods: dict) -> None:
        combined = {
            "experiment_info": {
                "dataset": self.config.dataset,
                "model": self.config.model,
                "prompt_methods": self.config.prompt_methods,
                "timestamp": datetime.now().isoformat(),
            },
            "methods": all_methods,
        }

        combined_file = self.base_output / "combined_summary.json"
        with open(combined_file, "w", encoding="utf-8") as f:
            json.dump(combined, f, indent=4, ensure_ascii=False, cls=NumpyEncoder)

        # Terminal table
        print("\n" + "=" * 80)
        print("COMBINED RESULTS")
        print("=" * 80)
        header = f"{'Method':<25} {'Accuracy':<10} {'Macro F1':<12}"
        print(header)
        print("-" * len(header))
        for method, data in all_methods.items():
            acc = data["metrics"].get("accuracy", "N/A")
            f1 = data["metrics"].get("macro_f1", "N/A")
            print(f"{method:<25} {str(acc):<10} {str(f1):<12}")
        print("=" * 80)

def run_bias_evaluation_pipeline(config: Config) -> None:
    BiasEvaluationPipeline(config).run()
