"""
Pipeline orchestrator for the fact-checking inference system.
Handles multi-prompt evaluation, temporal splitting, sampling, and statistical comparison.

Sampling behavior:
- balanced_sampling=True:
    Apply label-stratified balanced sampling BEFORE inference.
    No post-hoc balanced evaluation is created.
- balanced_sampling=False:
    Run inference on the full seen/unseen splits.
    Then compute an additional post-hoc label-stratified balanced evaluation
    from the already predicted results, without extra API calls.
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
    NumpyEncoder,
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

        # Do not log the actual API key.
        self.logger.info(
            "Loaded %d claims (is_test=%s). API key configured: %s",
            len(claims),
            is_test,
            bool(getattr(self.config, "api_key", None)),
        )

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
            json.dump(summary, f, indent=4, cls=NumpyEncoder)

        print(f"✓ {method}: test predictions saved")

    # ------------------------------------------------------------------
    # Labeled data branch
    # ------------------------------------------------------------------
    def _run_labeled_multi_method(self, claims: list) -> None:
        self.logger.info("Processing LABELED data with temporal split analysis")

        # 1. Split once
        seen_claims, unseen_claims = split_by_date(
            claims, self.config.temporal_split, self.config.date_format
        )

        # 2. If balanced sampling is enabled, sample BEFORE inference.
        #    If balanced sampling is disabled, keep the full data here and
        #    compute post-hoc balanced metrics later after prediction.
        sampling_metadata, seen_claims, unseen_claims = self._apply_sampling(
            seen_claims, unseen_claims
        )

        if self.config.balanced_sampling:
            sampling_metadata["stage"] = "pre_inference"
            sampling_metadata["posthoc_balanced_metrics"] = False
        else:
            sampling_metadata = {
                "enabled": False,
                "stage": "full_data_before_inference",
                "strategy": "no_pre_inference_sampling",
                "posthoc_balanced_metrics": True,
                "reason": (
                    "balanced_sampling is false, so full seen/unseen data is processed first. "
                    "A post-hoc label-stratified balanced evaluation is computed after prediction."
                ),
                "original_seen_count": len(seen_claims),
                "original_unseen_count": len(unseen_claims),
                "timestamp": datetime.now().isoformat(),
            }

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
        """Run inference and evaluation for one prompt method.

        Behavior:
        - balanced_sampling=True:
            Uses the already pre-inference balanced data and computes only normal metrics.
        - balanced_sampling=False:
            Processes full seen/unseen data, then computes post-hoc label-stratified
            balanced metrics from the already predicted results, without extra API calls.
        """
        out_dir = self.base_output / method
        out_dir.mkdir(exist_ok=True)
        logger = setup_logger(out_dir / "method_run_log.txt")
        logger.info("Prompt method: %s", method)

        # ============================================================
        # 1. Normal evaluation
        # ============================================================
        seen_results = self._process_split(seen_claims, out_dir, "seen", method, logger)
        seen_metrics = calculate_metrics(seen_results, self.config.labels, logger)

        unseen_results = self._process_split(unseen_claims, out_dir, "unseen", method, logger)
        unseen_metrics = calculate_metrics(unseen_results, self.config.labels, logger)

        comparison = compare_seen_unseen(
            seen_results, unseen_results, self.config.labels, logger
        )
        comparison["sampling_info"] = sampling_metadata
        comparison["prompt_method"] = method
        comparison["evaluation_mode"] = (
            "pre_inference_label_stratified_balanced"
            if self.config.balanced_sampling
            else "full_data"
        )

        generate_summary_report(
            seen_metrics, unseen_metrics, comparison, self.config, out_dir, logger
        )

        result_payload = {
            # Kept for backward compatibility with the existing combined summary table
            "seen_metrics": seen_metrics,
            "unseen_metrics": unseen_metrics,
            "comparison": comparison,

            "full_data": {
                "seen_metrics": seen_metrics,
                "unseen_metrics": unseen_metrics,
                "comparison": comparison,
            },
        }

        # ============================================================
        # 2. Post-hoc balanced evaluation ONLY when sampling is OFF
        # ============================================================
        if not self.config.balanced_sampling:
            balanced_seen_results, balanced_unseen_results, posthoc_sampling_info = (
                self._posthoc_stratified_balance_results(
                    seen_results,
                    unseen_results,
                    method=method,
                )
            )

            balanced_out_dir = out_dir / "posthoc_balanced"
            balanced_out_dir.mkdir(exist_ok=True)

            balanced_seen_dir = balanced_out_dir / "seen"
            balanced_unseen_dir = balanced_out_dir / "unseen"
            balanced_seen_dir.mkdir(exist_ok=True)
            balanced_unseen_dir.mkdir(exist_ok=True)

            balanced_seen_metrics = calculate_metrics(
                balanced_seen_results, self.config.labels, logger
            )
            balanced_unseen_metrics = calculate_metrics(
                balanced_unseen_results, self.config.labels, logger
            )

            balanced_comparison = compare_seen_unseen(
                balanced_seen_results,
                balanced_unseen_results,
                self.config.labels,
                logger,
            )
            balanced_comparison["sampling_info"] = posthoc_sampling_info
            balanced_comparison["prompt_method"] = method
            balanced_comparison["evaluation_mode"] = "posthoc_label_stratified_balanced"

            save_results_with_metrics(
                balanced_seen_results,
                balanced_seen_metrics,
                balanced_seen_dir,
                "seen_posthoc_balanced",
                logger,
            )

            save_results_with_metrics(
                balanced_unseen_results,
                balanced_unseen_metrics,
                balanced_unseen_dir,
                "unseen_posthoc_balanced",
                logger,
            )

            generate_summary_report(
                balanced_seen_metrics,
                balanced_unseen_metrics,
                balanced_comparison,
                self.config,
                balanced_out_dir,
                logger,
            )

            posthoc_summary_file = balanced_out_dir / "posthoc_balanced_summary.json"
            with open(posthoc_summary_file, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "dataset": self.config.dataset,
                        "model": self.config.model,
                        "prompt_method": method,
                        "temporal_split": self.config.temporal_split,
                        "evaluation_mode": "posthoc_label_stratified_balanced",
                        "sampling_info": posthoc_sampling_info,
                        "seen_metrics": balanced_seen_metrics,
                        "unseen_metrics": balanced_unseen_metrics,
                        "comparison": balanced_comparison,
                        "timestamp": datetime.now().isoformat(),
                    },
                    f,
                    indent=4,
                    ensure_ascii=False,
                    cls=NumpyEncoder,
                )

            result_payload["posthoc_balanced"] = {
                "seen_metrics": balanced_seen_metrics,
                "unseen_metrics": balanced_unseen_metrics,
                "comparison": balanced_comparison,
            }

            print(f"\n✓ Post-hoc balanced metrics saved to: {balanced_out_dir}")

        else:
            print(
                "\nℹ Balanced sampling is enabled, so post-hoc balanced evaluation is skipped."
            )

        return result_payload

    def _process_split(
        self,
        claims: list,
        base_dir: Path,
        split_name: str,
        method: str,
        parent_logger: logging.Logger,
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

    def _apply_sampling(self, seen_claims: list, unseen_claims: list) -> tuple[dict, list, list]:
        """
        If balanced sampling is enabled, create a label-stratified balanced subset
        from both seen and unseen claims BEFORE inference.

        This guarantees:
        1. seen and unseen have the same total size
        2. seen and unseen have the same label distribution
        3. for each label, both groups contain the same number of examples
        """
        if not self.config.balanced_sampling:
            return {"enabled": False}, seen_claims, unseen_claims

        rng = random.Random(self.config.sampling_seed)

        label_key = "label"

        seen_by_label = {}
        unseen_by_label = {}

        for claim in seen_claims:
            label = claim.get(label_key)
            if label is None:
                continue
            seen_by_label.setdefault(label, []).append(claim)

        for claim in unseen_claims:
            label = claim.get(label_key)
            if label is None:
                continue
            unseen_by_label.setdefault(label, []).append(claim)

        # Keep config label order first, then include any unexpected common labels.
        common_labels = [
            label for label in self.config.labels
            if label in seen_by_label and label in unseen_by_label
        ]

        extra_common_labels = sorted(
            (set(seen_by_label.keys()) & set(unseen_by_label.keys())) - set(common_labels),
            key=lambda x: str(x),
        )
        common_labels.extend(extra_common_labels)

        sampled_seen = []
        sampled_unseen = []
        per_label_info = {}

        for label in common_labels:
            seen_items = seen_by_label.get(label, [])
            unseen_items = unseen_by_label.get(label, [])

            n = min(len(seen_items), len(unseen_items))

            if n == 0:
                continue

            sampled_seen.extend(rng.sample(seen_items, n))
            sampled_unseen.extend(rng.sample(unseen_items, n))

            per_label_info[str(label)] = {
                "original_seen": len(seen_items),
                "original_unseen": len(unseen_items),
                "sampled_seen": n,
                "sampled_unseen": n,
            }

        rng.shuffle(sampled_seen)
        rng.shuffle(sampled_unseen)

        metadata = {
            "enabled": True,
            "stage": "pre_inference",
            "strategy": "label_stratified_seen_unseen_balance_before_prediction",
            "seed": self.config.sampling_seed,
            "original_seen_count": len(seen_claims),
            "original_unseen_count": len(unseen_claims),
            "sampled_seen_count": len(sampled_seen),
            "sampled_unseen_count": len(sampled_unseen),
            "labels": common_labels,
            "per_label_info": per_label_info,
            "sampled_seen_claim_ids": [c.get("claim_id") for c in sampled_seen],
            "sampled_unseen_claim_ids": [c.get("claim_id") for c in sampled_unseen],
            "timestamp": datetime.now().isoformat(),
        }

        sampling_file = self.base_output / "sampling_info.json"
        with open(sampling_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4, ensure_ascii=False, cls=NumpyEncoder)

        print(
            f"\n📊 STRATIFIED BALANCED SAMPLING ENABLED:\n"
            f"   Original seen: {len(seen_claims)} → Sampled: {len(sampled_seen)}\n"
            f"   Original unseen: {len(unseen_claims)} → Sampled: {len(sampled_unseen)}\n"
            f"   Seed: {self.config.sampling_seed}\n"
            f"   Per-label sampling:"
        )

        for label, info in per_label_info.items():
            print(
                f"   - {label}: "
                f"seen {info['original_seen']} → {info['sampled_seen']}, "
                f"unseen {info['original_unseen']} → {info['sampled_unseen']}"
            )

        self.logger.info(
            "Pre-inference stratified balanced sampling: seen %d -> %d, unseen %d -> %d",
            len(seen_claims),
            len(sampled_seen),
            len(unseen_claims),
            len(sampled_unseen),
        )

        return metadata, sampled_seen, sampled_unseen

    def _get_result_label(self, result: dict):
        """Extract the gold label from a processed result."""
        for key in ["label", "gold_label", "true_label", "ground_truth", "actual_label"]:
            if key in result and result.get(key) is not None:
                return result.get(key)

        original = result.get("original") or result.get("claim_data") or result.get("metadata")
        if isinstance(original, dict):
            for key in ["label", "gold_label", "true_label", "ground_truth", "actual_label"]:
                if key in original and original.get(key) is not None:
                    return original.get(key)

        return None

    def _posthoc_stratified_balance_results(
        self,
        seen_results: list,
        unseen_results: list,
        method: str,
    ) -> tuple[list, list, dict]:
        """
        Create a post-hoc label-stratified balanced subset from already predicted results.

        This does NOT call the API again.
        It only samples from prediction results.

        Guarantees:
        1. sampled seen and sampled unseen have the same total size
        2. sampled seen and sampled unseen have the same gold-label distribution
        3. for each label, both groups contain the same number of examples
        """
        rng = random.Random(self.config.sampling_seed)

        seen_by_label = {}
        unseen_by_label = {}

        for result in seen_results:
            label = self._get_result_label(result)
            if label is None:
                continue
            seen_by_label.setdefault(label, []).append(result)

        for result in unseen_results:
            label = self._get_result_label(result)
            if label is None:
                continue
            unseen_by_label.setdefault(label, []).append(result)

        # Keep config label order first, then include unexpected common labels if any.
        common_labels = [
            label for label in self.config.labels
            if label in seen_by_label and label in unseen_by_label
        ]

        extra_common_labels = sorted(
            (set(seen_by_label.keys()) & set(unseen_by_label.keys())) - set(common_labels),
            key=lambda x: str(x),
        )
        common_labels.extend(extra_common_labels)

        sampled_seen = []
        sampled_unseen = []
        per_label_info = {}

        for label in common_labels:
            seen_items = seen_by_label.get(label, [])
            unseen_items = unseen_by_label.get(label, [])

            n = min(len(seen_items), len(unseen_items))

            if n == 0:
                continue

            sampled_seen.extend(rng.sample(seen_items, n))
            sampled_unseen.extend(rng.sample(unseen_items, n))

            per_label_info[str(label)] = {
                "original_seen": len(seen_items),
                "original_unseen": len(unseen_items),
                "sampled_seen": n,
                "sampled_unseen": n,
            }

        rng.shuffle(sampled_seen)
        rng.shuffle(sampled_unseen)

        metadata = {
            "enabled": True,
            "stage": "post_inference",
            "strategy": "label_stratified_seen_unseen_balance_after_prediction",
            "method": method,
            "seed": self.config.sampling_seed,
            "original_seen_count": len(seen_results),
            "original_unseen_count": len(unseen_results),
            "sampled_seen_count": len(sampled_seen),
            "sampled_unseen_count": len(sampled_unseen),
            "labels": common_labels,
            "per_label_info": per_label_info,
            "timestamp": datetime.now().isoformat(),
        }

        sampling_file = self.base_output / method / "posthoc_balanced_sampling_info.json"
        with open(sampling_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4, ensure_ascii=False, cls=NumpyEncoder)

        print(
            f"\n📊 POST-HOC STRATIFIED BALANCED SAMPLING:\n"
            f"   Method: {method}\n"
            f"   Full seen: {len(seen_results)} → Balanced seen: {len(sampled_seen)}\n"
            f"   Full unseen: {len(unseen_results)} → Balanced unseen: {len(sampled_unseen)}\n"
            f"   Seed: {self.config.sampling_seed}\n"
            f"   Per-label sampling:"
        )

        for label, info in per_label_info.items():
            print(
                f"   - {label}: "
                f"seen {info['original_seen']} → {info['sampled_seen']}, "
                f"unseen {info['original_unseen']} → {info['sampled_unseen']}"
            )

        return sampled_seen, sampled_unseen, metadata

    def _save_combined_summary(self, all_methods: dict) -> None:
        """Generate a combined JSON summary and a console comparison table."""
        combined = {
            "experiment_info": {
                "dataset": self.config.dataset,
                "model": self.config.model,
                "temporal_split": self.config.temporal_split,
                "balanced_sampling": self.config.balanced_sampling,
                "sampling_seed": self.config.sampling_seed,
                "prompt_methods": self.config.prompt_methods,
                "timestamp": datetime.now().isoformat(),
            },
            "methods": all_methods,
        }

        combined_file = self.base_output / "combined_summary.json"
        with open(combined_file, "w", encoding="utf-8") as f:
            json.dump(combined, f, indent=4, ensure_ascii=False, cls=NumpyEncoder)

        print("\n" + "=" * 80)
        print("COMBINED RESULTS")
        print("=" * 80)

        header = (
            f"{'Method':<25} "
            f"{'Mode':<18} "
            f"{'Seen Acc':<10} "
            f"{'Unseen Acc':<10} "
            f"{'Macro F1 (U)':<12} "
            f"{'NEI Pred Rate (U)':<16}"
        )
        print(header)
        print("-" * len(header))

        for method, data in all_methods.items():
            rows = [("full", data.get("full_data", data))]

            if "posthoc_balanced" in data:
                rows.append(("posthoc_balanced", data["posthoc_balanced"]))

            for mode, mode_data in rows:
                seen_metrics = mode_data.get("seen_metrics", {})
                unseen_metrics = mode_data.get("unseen_metrics", {})

                seen_acc = seen_metrics.get("accuracy", "N/A")
                unseen_acc = unseen_metrics.get("accuracy", "N/A")
                f1 = unseen_metrics.get("macro_f1", "N/A")
                nei_pred = unseen_metrics.get("nei_prediction_rate", "N/A")

                print(
                    f"{method:<25} "
                    f"{mode:<18} "
                    f"{str(seen_acc):<10} "
                    f"{str(unseen_acc):<10} "
                    f"{str(f1):<12} "
                    f"{str(nei_pred):<16}"
                )

        print("=" * 80)
        print(f"✓ Combined summary saved to: {combined_file}")


def run_pipeline(config: Config) -> None:
    Pipeline(config).run()
