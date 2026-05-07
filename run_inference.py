import argparse
from src.config import Config
from src.pipeline import run_pipeline

def main():
    parser = argparse.ArgumentParser(description="Run fact-check inference pipeline")
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config file")
    args = parser.parse_args()
    cfg = Config.from_yaml(args.config)
    run_pipeline(cfg)

if __name__ == "__main__":
    main()