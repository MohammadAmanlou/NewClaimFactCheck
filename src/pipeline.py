"""
Pipeline orchestrator for the fact-checking inference system.

Standard usage:
    from src.config import Config
    from src.pipeline import run_pipeline

    cfg = Config.from_yaml("config.yaml")
    run_pipeline(cfg)
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
    """Orchestrates the complete fact-checking workflow."""

    def __init__(self, config: Config):
        self.config = config
        self.output_dir = self._create_output_directory()
        self.logger = self._initialize_main_logger()
        self._print_header()

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def run(self) -> None:
        """Run the pipeline according to the configuration."""
        claims, is_test = load_dataset(self.config.dataset, self.config)
        self.logger.info("Loaded %d claims (is_test=%s)", len(claims), is_test)

        if is_test:
            self._execute_test_pipeline(claims)
        else:
            self._execute_labeled_pipeline(claims)

        self.logger.info("PIPELINE COMPLETE")
        print(f"\n✓ All outputs saved to: {self.output_dir}")

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------
    def _create_output_directory(self) -> Path:
        model_name = self.config.model.replace(":", "_").replace(".", "_")
        path = self.config.output_root / model_name / self.config.dataset
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _initialize_main_logger(self) -> logging.Logger:
        return setup_logger(self.output_dir / "main_run_log.txt")

    def _print_header(self) -> None:
        self.logger.info("=" * 80)
        self.logger.info("STARTING FACT-CHECK INFERENCE PIPELINE")
        self.logger.info("Dataset: %s, Model: %s", self.config.dataset, self.config.model)
        self.logger.info("Output: %s", self.output_dir)

        print(f"\n{'=' * 80}")
        print(f"Starting inference pipeline for {self.config.dataset}")
        print(f"{'=' * 80}\n")

    # ------------------------------------------------------------------
    # Test data branch (no labels)
    # ------------------------------------------------------------------
    def _execute_test_pipeline(self, claims: list) -> None:
        self.logger.info("Processing TEST data (no labels available)")
        results = process_claims(claims, is_test=True, config=self.config, logger=self.logger)

        output_file = self.output_dir / "predictions_with_labels.json"
        save_test_predictions(results, output_file, self.logger)

        self._save_test_summary(results)

    def _save_test_summary(self, results: list) -> None:
        successful = sum(1 for r in results if r.get("prediction"))
        summary = {
            "dataset": self.config.dataset,
            "model": self.config.model,
            "total_claims": len(results),
            "successful_predictions": successful,
            "failed_predictions": len(results) - successful,
            "success_rate": round(successful / len(results), 4) if results else 0.0,
            "timestamp": datetime.now().isoformat(),
        }

        summary_path = self.output_dir / "test_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=4)

        print("\n" + "=" * 80)
        print("TEST DATA SUMMARY")
        print("=" * 80)
        for key, value in summary.items():
            print(f"{key}: {value}")
        print("=" * 80)

        self.logger.info("Test summary saved to %s", summary_path)

    # ------------------------------------------------------------------
    # Labeled data branch (with temporal split & evaluation)
    # ------------------------------------------------------------------
    def _execute_labeled_pipeline(self, claims: list) -> None:
        self.logger.info("Processing LABELED data with temporal split analysis")

        # 1. Split by date
        seen_claims, unseen_claims = split_by_date(
            claims, self.config.temporal_split, self.config.date_format
        )

        # 2. Optional balanced sampling of the seen set
        sampling_metadata, seen_claims = self._apply_sampling(
            seen_claims, unseen_claims
        )

        # 3. Process both splits
        seen_results, seen_metrics = self._process_split(seen_claims, "seen")
        unseen_results, unseen_metrics = self._process_split(unseen_claims, "unseen")

        # 4. Compare splits statistically
        comparison = compare_seen_unseen(
            seen_results, unseen_results, self.config.labels, self.logger
        )
        comparison["sampling_info"] = sampling_metadata

        # 5. Generate final report
        generate_summary_report(
            seen_metrics,
            unseen_metrics,
            comparison,
            self.config,
            self.output_dir,
            self.logger,
        )

    def _apply_sampling(
        self, seen_claims: list, unseen_claims: list
    ) -> tuple[dict, list]:
        """Return (sampling_metadata, final_seen_claims).
        If balanced sampling is enabled and seen > unseen, we downsample seen.
        """
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

        # Save the sampling information for reproducibility
        sampling_file = self.output_dir / "sampling_info.json"
        with open(sampling_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)

        print(
            f"\n📊 BALANCED SAMPLING ENABLED:\n"
            f"   Original seen: {len(seen_claims)} → Sampled: {target_size}\n"
            f"   Seed: {self.config.sampling_seed}\n"
        )
        self.logger.info("Balanced sampling: %d → %d", len(seen_claims), target_size)

        return metadata, sampled

    def _process_split(
        self, claims: list, split_name: str
    ) -> tuple[list, dict]:
        """Run inference on a single data split and compute metrics."""
        split_dir = self.output_dir / split_name
        split_dir.mkdir(exist_ok=True)

        split_logger = setup_logger(split_dir / f"{split_name}_run_log.txt")

        self.logger.info("Processing %s data…", split_name.upper())
        results = process_claims(
            claims, is_test=False, config=self.config, logger=split_logger
        )
        metrics = calculate_metrics(results, self.config.labels, split_logger)
        save_results_with_metrics(results, metrics, split_dir, split_name, split_logger)

        return results, metrics


# ------------------------------------------------------------------
# Convenience function (keeps old import style intact)
# ------------------------------------------------------------------
def run_pipeline(config: Config) -> None:
    """Entry point that mirrors the original monolithic function."""
    Pipeline(config).run()