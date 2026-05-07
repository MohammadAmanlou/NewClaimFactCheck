"""
Pipeline orchestrator for the fact-checking inference system.
Handles multi‑prompt evaluation, temporal splitting, and statistical comparison.
"""

import random
import json
import logging
from pathlib import Path
from datetime import datetime

from .config import Config
from .logging_config import setup_logger
from .data_loader import load_dataset
from .processing import split_by_date, process_claims
from .evaluation import calculate_metrics, compare_seen_unseen
from .reporting import (
    save_test_predictions,
    save_results_with_metrics,
    generate_summary_report,
)


class Pipeline:
    """Orchestrates the complete fact-checking workflow, possibly across multiple prompt methods."""

    def __init__(self, config: Config):
        self.config = config
        self.base_output = self._make_base_output()
        self.base_output.mkdir(parents=True, exist_ok=True)
        self.logger = setup_logger(self.base_output / "main_run_log.txt")
        self._print_header()

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------
    def _make_base_output(self) -> Path:
        model_dir = self.config.model.replace(":", "_").replace(".", "_")
        return self.config.output_root / model_dir / self.config.dataset

    def _print_header(self) -> None:
        self.logger.info("=" * 80)
        self.logger.info("STARTING FACT-CHECK INFERENCE PIPELINE")
        self.logger.info("Dataset: %s, Model: %s", self.config.dataset, self.config.model)
        self.logger.info("Prompt methods: %s", self.config.prompt_methods)
        print(f"\n{'=' * 80}")
        print(f"Starting pipeline for {self.config.dataset}")
        print(f"Prompt methods: {self.config.prompt_methods}")
        print(f"{'=' * 80}\n")

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def run(self) -> None:
        claims, is_test = load_dataset(self.config.dataset, self.config)
        self.logger.info("Loaded %d claims (is_test=%s)", len(claims), is_test)

        if is_test:
            for method in self.config.prompt_methods:
                self._run_test_method(claims, method)
        else:
            self._run_labeled_multi_method(claims)

        self.logger.info("=" * 80)
        self.logger.info("PIPELINE COMPLETE")
        print(f"\n✓ All outputs saved to: {self.base_output}")

    # ------------------------------------------------------------------
    # Test data branch (no labels, multiple prompts possible)
    # ------------------------------------------------------------------
    def _run_test_method(self, claims: list, method: str) -> None:
        out_dir = self.base_output / method
        out_dir.mkdir(exist_ok=True)
        logger = setup_logger(out_dir / "test_run_log.txt")
        logger.info("Test data, prompt: %s", method)

        results = process_claims(
            claims,
            is_test=True,
            config=self.config,
            logger=logger,
            prompt_method=method,
        )
        save_test_predictions(results, out_dir / "predictions_with_labels.json", logger)

        successful = sum(1 for r in results if r.get("prediction"))
        summary = {
            "dataset": self.config.dataset,
            "model": self.config.model,
            "prompt_method": method,
            "total_claims": len(results),
            "successful_predictions": successful,
            "failed_predictions": len(results) - successful,
            "success_rate": round(successful / len(results), 4) if results else 0.0,
            "timestamp": datetime.now().isoformat(),
        }
        with open(out_dir / "test_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=4)

        print(f"✓ {method}: test predictions saved")

    # ------------------------------------------------------------------
    # Labeled data branch – single split & sample, iterate over methods
    # ------------------------------------------------------------------
    def _run_labeled_multi_method(self, claims: list) -> None:
        self.logger.info("Processing LABELED data with temporal split analysis")

        # 1. Split once
        seen_claims, unseen_claims = split_by_date(
            claims, self.config.temporal_split, self.config.date_format
        )

        # 2. Apply balanced sampling once (if enabled)
        sampling_metadata, seen_claims = self._apply_sampling(seen_claims, unseen_claims)

        all_methods_results = {}
        for method in self.config.prompt_methods:
            print(f"\n--- Running method: {method} ---")
            results = self._run_single_method_labeled(
                seen_claims, unseen_claims, method, sampling_metadata
            )
            all_methods_results[method] = results

        # 3. Generate combined report
        self._save_combined_summary(all_methods_results)

    def _run_single_method_labeled(
        self,
        seen_claims: list,
        unseen_claims: list,
        method: str,
        sampling_metadata: dict,
    ) -> dict:
        """Run inference and evaluation for one prompt method on the given splits."""
        out_dir = self.base_output / method
        out_dir.mkdir(exist_ok=True)
        logger = setup_logger(out_dir / "method_run_log.txt")
        logger.info("Prompt method: %s", method)

        # Process seen
        seen_results = self._process_split(seen_claims, out_dir, "seen", method, logger)
        seen_metrics = calculate_metrics(seen_results, self.config.labels, logger)

        # Process unseen
        unseen_results = self._process_split(unseen_claims, out_dir, "unseen", method, logger)
        unseen_metrics = calculate_metrics(unseen_results, self.config.labels, logger)

        # Compare
        comparison = compare_seen_unseen(
            seen_results, unseen_results, self.config.labels, logger
        )
        comparison["sampling_info"] = sampling_metadata
        comparison["prompt_method"] = method

        # Save per‑method report
        generate_summary_report(
            seen_metrics, unseen_metrics, comparison, self.config, out_dir, logger
        )

        return {
            "seen_metrics": seen_metrics,
            "unseen_metrics": unseen_metrics,
            "comparison": comparison,
        }

    def _process_split(
        self, claims: list, base_dir: Path, split_name: str, method: str, parent_logger: logging.Logger
    ) -> list:
        """Run inference on one split (seen or unseen) and save results."""
        split_dir = base_dir / split_name
        split_dir.mkdir(exist_ok=True)
        split_logger = setup_logger(split_dir / f"{split_name}_run_log.txt")

        results = process_claims(
            claims,
            is_test=False,
            config=self.config,
            logger=split_logger,
            prompt_method=method,
        )
        metrics = calculate_metrics(results, self.config.labels, split_logger)
        save_results_with_metrics(results, metrics, split_dir, split_name, split_logger)
        return results

    def _apply_sampling(self, seen_claims: list, unseen_claims: list) -> tuple[dict, list]:
        """Downsample seen claims to match unseen size if balanced sampling is enabled."""
        if not self.config.balanced_sampling or len(seen_claims) <= len(unseen_claims):
            if self.config.balanced_sampling:
                print(
                    f"\n⚠ Balanced sampling requested but seen "
                    f"({len(seen_claims)}) <= unseen ({len(unseen_claims)}). "
                    "Processing all.\n"
                )
            return {"enabled": False}, seen_claims

        random.seed(self.config.sampling_seed)
        target_size = len(unseen_claims)
        sampled = random.sample(seen_claims, target_size)

        metadata = {
            "enabled": True,
            "seed": self.config.sampling_seed,
            "original_seen_count": len(seen_claims),
            "sampled_seen_count": target_size,
            "unseen_count": len(unseen_claims),
            "sampled_claim_ids": [c.get("claim_id") for c in sampled],
            "timestamp": datetime.now().isoformat(),
        }

        # Persist sampling info for reproducibility
        sampling_file = self.base_output / "sampling_info.json"
        with open(sampling_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)

        print(
            f"\n📊 BALANCED SAMPLING ENABLED:\n"
            f"   Original seen: {len(seen_claims)} → Sampled: {target_size}\n"
            f"   Seed: {self.config.sampling_seed}\n"
        )
        self.logger.info("Balanced sampling: %d → %d", len(seen_claims), target_size)
        return metadata, sampled

    def _save_combined_summary(self, all_methods: dict) -> None:
        """Generate a combined JSON summary and a console comparison table."""
        combined = {
            "experiment_info": {
                "dataset": self.config.dataset,
                "model": self.config.model,
                "temporal_split": self.config.temporal_split,
                "balanced_sampling": self.config.balanced_sampling,
                "sampling_seed": self.config.sampling_seed if self.config.balanced_sampling else None,
                "prompt_methods": self.config.prompt_methods,
                "timestamp": datetime.now().isoformat(),
            },
            "methods": all_methods,
        }

        combined_file = self.base_output / "combined_summary.json"
        with open(combined_file, "w", encoding="utf-8") as f:
            json.dump(combined, f, indent=4, ensure_ascii=False)

        # Terminal table
        print("\n" + "=" * 80)
        print("COMBINED RESULTS")
        print("=" * 80)
        header = (
            f"{'Method':<25} {'Seen Acc':<10} {'Unseen Acc':<10} "
            f"{'Macro F1 (U)':<12} {'NEI Pred Rate (U)':<16}"
        )
        print(header)
        print("-" * len(header))
        for method, data in all_methods.items():
            seen_acc = data["seen_metrics"].get("accuracy", "N/A")
            unseen_acc = data["unseen_metrics"].get("accuracy", "N/A")
            f1 = data["unseen_metrics"].get("macro_f1", "N/A")
            nei_pred = data["unseen_metrics"].get("nei_prediction_rate", "N/A")
            print(
                f"{method:<25} {str(seen_acc):<10} {str(unseen_acc):<10} "
                f"{str(f1):<12} {str(nei_pred):<16}"
            )
        print("=" * 80)


def run_pipeline(config: Config) -> None:
    Pipeline(config).run()