import concurrent.futures
import logging
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
from tqdm import tqdm

from .config import Config
from .logging_config import setup_logger
from .multi_key_manager import MultiKeyManager, DailyQuotaExhaustedException
from .resumable_state import ResumableState
from .distributed_api_client import DistributedApiClient

class ParallelPipeline:
    """
    A highly parallelized, resumable pipeline that processes datasets using multiple API keys 
    safely managed by a ThreadPoolExecutor.
    """

    def __init__(self, config: Config, key_list_path: str, max_workers: int = 20, rpm_limit: int = 15, rpd_limit: int = 1450):
        self.config = config
        self.max_workers = max_workers
        self.out_dir = self._prepare_output_directory()
        self.logger = setup_logger(self.out_dir / "parallel_run_log.txt")
        self.logger.info("Initializing ParallelPipeline with max %d workers.", self.max_workers)
        
        # Initialize the global key manager ensuring RPM/RPD limits
        self.key_manager = MultiKeyManager(
            key_file_path=key_list_path, 
            rpm_limit=rpm_limit, 
            rpd_limit=rpd_limit
        )
        
        # The append-only state file to resume progress seamlessly
        self.state_manager = ResumableState(self.out_dir / "progress.jsonl")
        
        # Instantiate our clean API client which relies on the key manager
        self.api_client = DistributedApiClient(self.config, self.key_manager, self.logger)

    def _prepare_output_directory(self) -> Path:
        """Sets up the distinct output directory to prevent overlapping with base experiments."""
        model_dir = self.config.model.replace(":", "_").replace(".", "_")
        out_dir = self.config.output_root / model_dir / self.config.dataset / "parallel_run"
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir

    def run(self, claims: List[Dict[str, Any]]) -> None:
        """
        Executes the thread pool over all requested prompt methods.
        Skips already completed claims instantly.
        """
        from .label_utils import normalize_label
        
        print(f"\n{'=' * 80}")
        print(f"STARTING PARALLEL EXPERIMENT")
        print(f"Dataset: {self.config.dataset}, Model: {self.config.model}")
        print(f"Prompt methods: {self.config.prompt_methods}")
        print(f"Workers: {self.max_workers}")
        print(f"{'=' * 80}\n")
        
        # Extremely Important optimization: Drop anything we can't map (so we never waste API quotas)
        valid_claims = [c for c in claims if normalize_label(c.get("label")) is not None]
        dropped = len(claims) - len(valid_claims)
        
        self.logger.info("Dataset contains %d raw items. Filtered to %d strictly mapped valid items (dropped %d).", 
                         len(claims), len(valid_claims), dropped)
        print(f"Loaded {len(claims)} raw items. Filtered to {len(valid_claims)} valid items (dropped {dropped}).")
        
        all_methods_results = {}
        
        for method in self.config.prompt_methods:
            print(f"\n--- Running method: {method} ---")
            self.logger.info("Starting processing for prompt method: %s", method)
            
            # Identify which items we STILL need to process
            completed_ids = self.state_manager.load_completed_ids(method)
            pending_claims = [c for c in valid_claims if str(c.get("claim_id")) not in completed_ids]
            
            if completed_ids:
                print(f"Skipping {len(completed_ids)} already evaluated items. {len(pending_claims)} items remaining.")

            if not pending_claims:
                self.logger.info("All claims for method '%s' already completed. Skipping.", method)
                print(f"✓ All items for {method} already completed.")
                continue
                
            self.logger.info("%d claims pending for method '%s' (skipped %d).", 
                             len(pending_claims), method, len(completed_ids))

            try:
                self._execute_pool(pending_claims, method)
            except DailyQuotaExhaustedException:
                self.logger.critical("Shutting down pool securely as all quota is exhausted.")
                print("\n[!] Daily quota exhausted across all provided API Keys.")
                print("[i] Progress saved securely to progress.jsonl. You may resume tomorrow.\n")
                return # Exit the run safely, ready for tomorrow
            
            # 100% complete -> Generate Metrics
            self.logger.info("Method '%s' is 100%% complete. Generating metrics.", method)
            metrics = self._finalize_method_results(claims, method)
            all_methods_results[method] = {"metrics": metrics}

        self._save_combined_summary(all_methods_results)
        self._generate_final_csv_dataset(claims)
        self.logger.info("All prompt methods completed successfully.")
        
        print(f"\n{'=' * 80}")
        print("PIPELINE COMPLETE")
        print(f"✓ All outputs saved to: {self.out_dir}")

    def _finalize_method_results(self, claims: List[Dict[str, Any]], method: str) -> dict:
        """Joins predictions with claims, saves JSONs, and calculates metrics out of the core run loop."""
        import json
        from .reporting import NumpyEncoder
        from .evaluation import calculate_metrics

        results = self._build_final_results(claims, method)
        out_dir_method = self.out_dir / method
        out_dir_method.mkdir(exist_ok=True)
        
        pred_file = out_dir_method / "predictions.json"
        with open(pred_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False, cls=NumpyEncoder)
            
        metrics = calculate_metrics(results, self.config.labels, self.logger)
        
        metrics_file = out_dir_method / "metrics.json"
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=4, cls=NumpyEncoder)
            
        return metrics

    def _build_final_results(self, claims: List[Dict[str, Any]], method: str) -> List[Dict[str, Any]]:
        """Matches original claims with the predictions housed in the progress jsonl."""
        import json
        
        # Load all lines from progress.jsonl into a lookup dictionary
        predictions_map = {}
        if self.state_manager.file_path.exists():
            with open(self.state_manager.file_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip(): continue
                    try:
                        rec = json.loads(line)
                        if rec.get("prompt_method") == method:
                            predictions_map[str(rec.get("claim_id"))] = rec.get("prediction")
                    except json.JSONDecodeError:
                        pass
        
        results = []
        for claim in claims:
            # We copy to avoid mutating the original dict shared across loops
            r = dict(claim)
            c_id = str(r.get("claim_id"))
            r["prediction"] = predictions_map.get(c_id)
            results.append(r)
            
        return results

    def _save_combined_summary(self, all_methods: dict) -> None:
        """Generates the combined JSON summary and console table."""
        from datetime import datetime
        import json
        from .reporting import NumpyEncoder
        
        combined = {
            "experiment_info": {
                "dataset": self.config.dataset,
                "model": self.config.model,
                "prompt_methods": self.config.prompt_methods,
                "timestamp": datetime.now().isoformat(),
            },
            "methods": all_methods,
        }

        combined_file = self.out_dir / "combined_summary.json"
        with open(combined_file, "w", encoding="utf-8") as f:
            json.dump(combined, f, indent=4, ensure_ascii=False, cls=NumpyEncoder)

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

    def _generate_final_csv_dataset(self, claims: List[Dict[str, Any]]) -> None:
        """
        Generates a final combined CSV dataset containing the original claim attributes natively
        merged with the final generated predictions from every executed prompt method.
        """
        import pandas as pd
        import json
        
        # Build a nested lookup for predictions: lookup[claim_id][method] = prediction
        predictions = {str(c.get("claim_id")): {} for c in claims}
        
        if self.state_manager.file_path.exists():
            with open(self.state_manager.file_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip(): continue
                    try:
                        rec = json.loads(line)
                        cid = str(rec.get("claim_id"))
                        method = rec.get("prompt_method")
                        pred = rec.get("prediction")
                        if cid in predictions and method:
                            predictions[cid][method] = pred
                    except json.JSONDecodeError:
                        pass
        
        rows = []
        for claim in claims:
            cid = str(claim.get("claim_id"))
            row = {
                "id": cid,
                "claim": claim.get("claim", ""),
                "source dataset": claim.get("source", ""),
                "biases": claim.get("detected_biases", ""),
                "factcheck label": claim.get("label", ""),
            }
            
            # Add columns natively for each dynamically requested prompt method
            for method in self.config.prompt_methods:
                col_name = f"factcheck predicted label by {method}"
                row[col_name] = predictions[cid].get(method, "")
                
            rows.append(row)
            
        df = pd.DataFrame(rows)
        out_csv = self.out_dir / "final_evaluation_dataset.csv"
        df.to_csv(out_csv, index=False, encoding="utf-8")
        
        self.logger.info("Saved final structured dataset CSV to %s", out_csv)
        print(f"[i] Final structured dataset natively exported to: {out_csv}")

    def _execute_pool(self, claims: List[Dict[str, Any]], method: str) -> None:
        """Submits claims to a thread pool executor and writes results dynamically."""
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_claim = {
                executor.submit(self._worker_task, claim, method): claim for claim in claims
            }
            self._process_futures(executor, future_to_claim, method)

    def _worker_task(self, claim_data: Dict[str, Any], method: str) -> Tuple[str, Optional[str]]:
        """A discrete worker task evaluating a single claim natively."""
        claim_id = str(claim_data.get("claim_id"))
        prediction = self.api_client.evaluate_claim(claim_data, method)
        return claim_id, prediction

    def _process_futures(self, executor: concurrent.futures.ThreadPoolExecutor, future_to_claim: dict, method: str) -> None:
        """Iterates over completed futures and manages state saving or exception handling."""
        
        # Use tqdm to give visual tracking for the parallel operations
        with tqdm(total=len(future_to_claim), desc=f"Evaluating claims ({method})", dynamic_ncols=True) as pbar:
            for future in concurrent.futures.as_completed(future_to_claim):
                claim_data = future_to_claim[future]
                
                if future.exception() is not None:
                    self._handle_future_error(executor, future.exception(), claim_data)
                    pbar.update(1)
                    continue

                claim_id, prediction = future.result()
                self._record_successful_inference(claim_id, method, prediction)
                pbar.update(1)

    def _handle_future_error(self, executor: concurrent.futures.ThreadPoolExecutor, exception: BaseException, claim_data: Dict[str, Any]) -> None:
        """Handles exceptions from futures, triggering safe shutdown if quota is exhausted."""
        if isinstance(exception, DailyQuotaExhaustedException):
            executor.shutdown(wait=False, cancel_futures=True)
            raise exception
        
        self.logger.error("Error evaluating claim %s: %s", claim_data.get("claim_id"), exception)

    def _record_successful_inference(self, claim_id: str, method: str, prediction: Optional[str]) -> None:
        """Saves a successfully processed claim to the resumable state file."""
        record = {
            "claim_id": claim_id,
            "prompt_method": method,
            "prediction": prediction,
            "timestamp": self.key_manager.get_stats()
        }
        self.state_manager.save_result(record)
        
        if prediction:
            self.logger.debug("Successfully inferred ID: %s -> Output: %s", claim_id, prediction)
