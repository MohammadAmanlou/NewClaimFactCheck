#!/usr/bin/env python3
"""
Entry point for the isolated parallel Bias Evaluation experiment.
Evaluates the 'combined_biased' dataset using multiple API keys 
across 'strong_baseline_short' and 'cognitive_bias_aware' prompts.
Handles strict Daily limits safely using resumable state.
"""

from dotenv import load_dotenv
from src.config import Config
from src.data_loader import load_dataset
from src.parallel_pipeline import ParallelPipeline

def main():
    # Load environment variables
    load_dotenv()
    
    # Load the dedicated configuration file
    cfg = Config("bias_eval_config.yaml")
    
    # Load the comprehensive combined dataset
    print(f"Loading '{cfg.dataset}' dataset...")
    claims, _ = load_dataset(cfg.dataset, cfg)
    
    # Run the isolated parallel pipeline
    # max_workers=50 easily accommodates 165 keys running 15 requests per minute simultaneously
    pipeline = ParallelPipeline(
        config=cfg, 
        key_list_path="google_api_list.txt", 
        max_workers=50
    )
    pipeline.run(claims)

if __name__ == "__main__":
    main()
