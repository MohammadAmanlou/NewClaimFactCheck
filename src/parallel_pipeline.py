import concurrent.futures
import logging
from typing import List, Dict, Any, Tuple
from pathlib import Path

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

    def __init__(self, config: Config, key_list_path: str, max_workers: int = 20):
        self.config = config
        self.max_workers = max_workers
        self.out_dir = self._prepare_output_directory()
        self.logger = setup_logger(self.out_dir / "parallel_run_log.txt")
        self.logger.info("Initializing ParallelPipeline with max %d workers.", self.max_workers)
        
        # Initialize the global key manager ensuring RPM/RPD limits
        self.key_manager = MultiKeyManager(
            key_file_path=key_list_path, 
            rpm_limit=15, 
            rpd_limit=1450
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
        self.logger.info("Dataset contains %d items.", len(claims))
        
        for method in self.config.prompt_methods:
            self.logger.info("Starting processing for prompt method: %s", method)
            
            # Identify which items we STILL need to process
            completed_ids = self.state_manager.load_completed_ids(method)
            pending_claims = [c for c in claims if str(c.get("claim_id")) not in completed_ids]
            
            if not pending_claims:
                self.logger.info("All claims for method '%s' already completed. Skipping.", method)
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

        self.logger.info("All prompt methods completed successfully.")
        print("\n✓ Parallel execution completed entirely.")

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
        for future in concurrent.futures.as_completed(future_to_claim):
            claim_data = future_to_claim[future]
            
            if future.exception() is not None:
                self._handle_future_error(executor, future.exception(), claim_data)
                continue

            claim_id, prediction = future.result()
            self._record_successful_inference(claim_id, method, prediction)

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
