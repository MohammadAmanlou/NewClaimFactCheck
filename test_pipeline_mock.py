import os
import shutil
import unittest
import json
import pandas as pd
from unittest.mock import patch, MagicMock

from src.parallel_pipeline import ParallelPipeline
from src.config import Config

from pathlib import Path

class TestPipelineMock(unittest.TestCase):
    def setUp(self):
        # Create a mock output dir
        self.test_dir = "test_output_mock"
        os.makedirs(self.test_dir, exist_ok=True)
        
        # Create mock keys file
        self.keys_path = os.path.join(self.test_dir, "keys.txt")
        with open(self.keys_path, "w", encoding="utf-8") as f:
            f.write("KEY1\nKEY2\n")
            
        self.config = Config(
            dataset="mock_dataset",
            model="test-model",
            api_url="http://fake-api",
            labels=["Supported", "Refuted"],
            temporal_split="default",
            date_format="%Y-%m-%d",
            max_retries=1,
            initial_retry_delay=0.1,
            sleep_between_calls=0.0,
            max_samples=100,
            balanced_sampling=False,
            sampling_seed=42,
            display_every_n=1,
            output_root=Path(self.test_dir)
        )
        self.config.prompt_methods = ["naive"]
        
        self.claims = [
            {"claim_id": i, "claim": f"claim {i}", "label": "Supported", "source": "test", "detected_biases": ""} 
            for i in range(1, 10) # 9 claims, capacity is 4.
        ]

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    @patch("src.distributed_api_client.send_api_request")
    def test_pipeline_quota_exhaustion_and_resume(self, mock_send):
        # Setup mock to return a fake Avalai response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": '{"label": "Supported"}'}}]
        }
        mock_send.return_value = mock_response

        print("\n--- RUN 1: Simulating Day 1 (Exhausing quota after 4 claims) ---")
        pipeline1 = ParallelPipeline(self.config, key_list_path=self.keys_path, max_workers=2, rpm_limit=10, rpd_limit=2)
        pipeline1.run(self.claims)
        
        self.out_dir = os.path.join(self.config.output_root, "test-model", "mock_dataset", "parallel_run")
        
        self.assertTrue(os.path.exists(os.path.join(self.out_dir, "progress.jsonl")), "Progress JSONL should be created.")
        
        with open(os.path.join(self.out_dir, "progress.jsonl"), "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        # Since it crashes somewhat non-deterministically based on thread completion before the pool shutdown,
        # we check if it is either 3 or 4
        self.assertTrue(3 <= len(lines) <= 4, f"It should have processed 3 or 4 items before the exception stopped it, got {len(lines)}")
        day1_lines = len(lines)
        self.assertFalse(os.path.exists(os.path.join(self.out_dir, "final_evaluation_dataset.csv")), "CSV should not logically exist on early abort.")
        
        print("\n--- RUN 2: Simulating Day 2 (Resuming fresh limits) ---")
        pipeline2 = ParallelPipeline(self.config, key_list_path=self.keys_path, max_workers=2, rpm_limit=10, rpd_limit=2) 
        pipeline2.run(self.claims)
        
        with open(os.path.join(self.out_dir, "progress.jsonl"), "r", encoding="utf-8") as f:
            lines = f.readlines()
        day2_lines = len(lines)
        self.assertTrue(day1_lines + 2 <= day2_lines <= day1_lines + 4, f"It should add about 2-4 more items in the second run. Total: {day2_lines}")
        
        print("\n--- RUN 3: Simulating Day 3 (Finishing the last claim) ---")
        pipeline3 = ParallelPipeline(self.config, key_list_path=self.keys_path, max_workers=2, rpm_limit=10, rpd_limit=2) 
        pipeline3.run(self.claims)
        
        with open(os.path.join(self.out_dir, "progress.jsonl"), "r", encoding="utf-8") as f:
            lines = f.readlines()
        self.assertEqual(len(lines), 9, "All 9 claims should be finished.")
        
        csv_path = os.path.join(self.out_dir, "final_evaluation_dataset.csv")
        self.assertTrue(os.path.exists(csv_path), "Final CSV should be written after complete execution.")
        
        df = pd.read_csv(csv_path)
        self.assertEqual(len(df), 9, "CSV should contain exactly 9 rows.")
        self.assertEqual(df["factcheck predicted label by naive"].iloc[0], "Supported")
        
        print("\n[✓] All Integration mock tests passed effectively.")

if __name__ == "__main__":
    unittest.main()
