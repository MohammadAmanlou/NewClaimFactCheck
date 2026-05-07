import time
import random
import logging
from typing import List, Dict, Any, Tuple, Optional
from tqdm import tqdm
from .config import Config
from .api_client import ask_model
from .label_utils import normalize_label


def split_by_date(claims: List[Dict], split_date: str, date_format: str) -> Tuple[List[Dict], List[Dict]]:
    """Split claims into seen (before split_date) and unseen."""
    from datetime import datetime
    split_dt = datetime.strptime(split_date, "%Y-%m-%d")
    seen, unseen, no_date = [], [], []

    for claim in claims:
        date_str = claim.get("claim_date", "")
        if not date_str:
            no_date.append(claim)
            continue
        try:
            claim_dt = datetime.strptime(date_str, date_format)
        except ValueError:
            try:
                claim_dt = datetime.fromisoformat(date_str.replace('T00:00:00', ''))
            except ValueError:
                no_date.append(claim)
                continue
        if claim_dt < split_dt:
            seen.append(claim)
        else:
            unseen.append(claim)

    seen.extend(no_date)
    print(f"✓ Split complete: {len(seen)} seen, {len(unseen)} unseen, {len(no_date)} without date")
    return seen, unseen


class ClaimProcessor:
    """Processes claims through the fact‑checking model, recording results and statistics."""

    def __init__(self, config: Config, logger: logging.Logger, is_test: bool):
        self.config = config
        self.logger = logger
        self.is_test = is_test
        # Counters
        self.successful = 0
        self.failed = 0
        self.correct = 0
        self.incorrect = 0

    def run(self, claims: List[Dict]) -> List[Dict]:
        """Main entry point: slice, set up progress, process every claim."""
        claims = self._limit_samples(claims)
        self.logger.info(f"Processing {len(claims)} claims...")

        pbar = self._build_progress_bar(claims)
        results = []
        for idx, claim_dict in enumerate(pbar):
            result = self._process_one(claim_dict, idx)
            results.append(result)
            self._update_progress(pbar)
            time.sleep(self.config.sleep_between_calls)
        pbar.close()
        self._log_summary()
        return results

    def _process_one(self, claim_dict: Dict, idx: int) -> Dict:
        """Handle a single claim: query model, evaluate, build result."""
        claim = claim_dict["claim"]
        date = claim_dict.get("claim_date", "")
        claim_id = claim_dict.get("claim_id", idx)

        self.logger.info("=" * 80)
        self.logger.info(f"Processing claim #{idx} (ID: {claim_id})")
        self.logger.info(f"Claim: {claim[:150]}...")
        self.logger.info(f"Date: {date}")

        prediction = ask_model(claim=claim, date=date, config=self.config, logger=self.logger)

        result = claim_dict.copy()
        if prediction:
            result["prediction"] = prediction
            result["prediction_error"] = None
            self.logger.info(f"Prediction: {prediction}")
            self.successful += 1
            if not self.is_test and claim_dict.get("label"):
                true_label = normalize_label(claim_dict["label"], self.config.labels)
                if true_label:
                    if prediction == true_label:
                        self.correct += 1
                        self.logger.info("✓ Correct prediction")
                    else:
                        self.incorrect += 1
                        self.logger.info(f"✗ Incorrect (True: {true_label}, Pred: {prediction})")
            self._maybe_print_claim(idx, claim, prediction, claim_dict)
        else:
            result["prediction"] = None
            result["prediction_error"] = "Failed to get valid prediction"
            self.logger.warning("Failed to get valid prediction")
            self.failed += 1
        return result

    def _maybe_print_claim(self, idx: int, claim: str, prediction: str, claim_dict: Dict):
        """Periodic console output (every display_every_n claims)."""
        if (idx + 1) % self.config.display_every_n == 0:
            print(f"\n[Claim #{idx + 1}] {claim[:100]}...")
            print(f"  → Prediction: {prediction}")
            if not self.is_test and claim_dict.get("label"):
                true_label = normalize_label(claim_dict["label"], self.config.labels)
                if true_label:
                    print(f"  → Ground Truth: {true_label}")
                    print(f"  → {'✓ Correct' if prediction == true_label else '✗ Incorrect'}")

    def _limit_samples(self, claims: List[Dict]) -> List[Dict]:
        """Apply max_samples trimming if configured."""
        if self.config.max_samples and len(claims) > self.config.max_samples:
            self.logger.info(f"Processing {self.config.max_samples} samples (limited)")
            return claims[:self.config.max_samples]
        return claims

    def _build_progress_bar(self, claims: List[Dict]) -> tqdm:
        """Create a tqdm bar with appropriate format and initial counters."""
        if self.is_test:
            fmt = ('{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] '
                   'Success: {postfix[0]}, Failed: {postfix[1]}')
            return tqdm(claims, desc="Processing claims", bar_format=fmt, postfix=[0, 0])
        else:
            fmt = ('{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] '
                   'Correct: {postfix[0]}, Incorrect: {postfix[1]}, Failed: {postfix[2]}')
            return tqdm(claims, desc="Processing claims", bar_format=fmt, postfix=[0, 0, 0])

    def _update_progress(self, pbar: tqdm):
        """Update postfix values and description from current counters."""
        if self.is_test:
            pbar.postfix = [self.successful, self.failed]
            pbar.set_description(f"Processing claims [✓{self.successful} ✗{self.failed}]")
        else:
            pbar.postfix = [self.correct, self.incorrect, self.failed]
            pbar.set_description(f"Processing claims [✓{self.correct} ✗{self.incorrect} Failed:{self.failed}]")

    def _log_summary(self):
        """Write final counts to the logger."""
        self.logger.info("=" * 80)
        if self.is_test:
            self.logger.info(f"Processing complete: {self.successful} successful, {self.failed} failed")
        else:
            self.logger.info(
                f"Processing complete: {self.successful} predictions "
                f"({self.correct} correct, {self.incorrect} incorrect), {self.failed} failed"
            )


def process_claims(claims: List[Dict], is_test: bool, config: Config, logger: logging.Logger) -> List[Dict]:
    """Public wrapper for backwards compatibility."""
    processor = ClaimProcessor(config, logger, is_test)
    return processor.run(claims)