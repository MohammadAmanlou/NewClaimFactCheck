#!/usr/bin/env python3
"""
Entry point for the isolated Bias Evaluation experiment.
Evaluates the 'combined_biased' dataset using Gemini across 'strong_baseline_short' 
and 'cognitive_bias_aware' prompts without applying temporal splitting.
"""

from dotenv import load_dotenv
from src.config import Config
from src.bias_evaluation_pipeline import run_bias_evaluation_pipeline

def main():
    # Load environment variables (e.g. GEMINI_API_KEY)
    load_dotenv()
    
    # Load the dedicated configuration file
    cfg = Config("bias_eval_config.yaml")
    
    # Run the isolated pipeline
    run_bias_evaluation_pipeline(cfg)

if __name__ == "__main__":
    main()
