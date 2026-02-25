#!/usr/bin/env python3
"""
Split FEVER Test Results by Temporal Date

This standalone script splits FEVER test prediction results into two CSV files:
- seen_predictions.csv: All claims with labels only for seen claims (before split date)
- unseen_predictions.csv: All claims with labels only for unseen claims (after split date)

Usage:
    python split_fever_results.py <results_file> <split_date> [--output-dir <dir>] [--date-format <format>]

Examples:
    python split_fever_results.py predictions.json 2024-04-01
    python split_fever_results.py predictions.json 2024-04-01 --output-dir results/split
    python split_fever_results.py predictions.json 2024-04-01 --date-format "%Y-%m-%d"

Author: Generated for FEVER Dataset Processing
Version: 1.0.0
"""

import json
import argparse
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

try:
    import pandas as pd
except ImportError:
    print("Error: pandas is required. Install with: pip install pandas")
    sys.exit(1)


def split_fever_results(
    results_file: Path,
    split_date: str,
    output_dir: Path,
    date_format: str = "%d-%m-%Y"
) -> None:
    """
    Split FEVER test results into seen/unseen CSV files with masked labels.
    
    Args:
        results_file: Path to the predictions JSON file
        split_date: Temporal split date (YYYY-MM-DD format)
        output_dir: Directory to save output CSV files
        date_format: Date format in the input data
    """
    
    # Load predictions
    print(f"\n📂 Loading predictions from: {results_file}")
    try:
        with open(results_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        print(f"✓ Loaded {len(results)} predictions")
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        sys.exit(1)
    
    # Parse split date
    try:
        split_dt = datetime.strptime(split_date, "%Y-%m-%d")
        print(f"✓ Split date: {split_date}")
    except ValueError as e:
        print(f"❌ Error parsing split date (use YYYY-MM-DD format): {e}")
        sys.exit(1)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"✓ Output directory: {output_dir}")
    
    # Process results
    print(f"\n⚙️  Processing claims...")
    seen_rows = []
    unseen_rows = []
    seen_count = 0
    unseen_count = 0
    no_date_count = 0
    
    for result in results:
        claim_id = result.get("claim_id", "")
        claim = result.get("claim", "")
        prediction = result.get("prediction", "")
        date_str = result.get("claim_date", "")
        
        # Extract evidence if available
        evi = result.get("evidence", result.get("evi", ""))
        
        # Determine if claim is seen or unseen
        is_seen = False
        if date_str:
            try:
                # Try parsing with the provided format
                claim_dt = datetime.strptime(date_str, date_format)
                is_seen = claim_dt < split_dt
            except:
                try:
                    # Try ISO format
                    claim_dt = datetime.fromisoformat(date_str.replace('T00:00:00', ''))
                    is_seen = claim_dt < split_dt
                except:
                    # Default to seen if date parsing fails
                    is_seen = True
                    no_date_count += 1
        else:
            # No date = treat as seen
            is_seen = True
            no_date_count += 1
        
        # Track counts
        if is_seen:
            seen_count += 1
        else:
            unseen_count += 1
        
        # For seen.csv: show labels for seen claims only
        seen_label = prediction if is_seen else "None"
        seen_rows.append({
            "id": claim_id,
            "claim": claim,
            "evi": evi,
            "label": seen_label
        })
        
        # For unseen.csv: show labels for unseen claims only
        unseen_label = prediction if not is_seen else "None"
        unseen_rows.append({
            "id": claim_id,
            "claim": claim,
            "evi": evi,
            "label": unseen_label
        })
    
    # Save seen.csv
    print(f"\n💾 Saving CSV files...")
    seen_file = output_dir / "seen_predictions.csv"
    df_seen = pd.DataFrame(seen_rows)
    df_seen.to_csv(seen_file, index=False, encoding='utf-8')
    print(f"✓ Seen predictions: {seen_file}")
    
    # Save unseen.csv
    unseen_file = output_dir / "unseen_predictions.csv"
    df_unseen = pd.DataFrame(unseen_rows)
    df_unseen.to_csv(unseen_file, index=False, encoding='utf-8')
    print(f"✓ Unseen predictions: {unseen_file}")
    
    # Print statistics
    print(f"\n📊 Split Statistics:")
    print(f"  • Total claims: {len(results)}")
    print(f"  • Seen claims (before {split_date}): {seen_count}")
    print(f"  • Unseen claims (after {split_date}): {unseen_count}")
    if no_date_count > 0:
        print(f"  • Claims without date (treated as seen): {no_date_count}")
    
    # Show label counts
    seen_labeled = sum(1 for row in seen_rows if row["label"] is not None)
    unseen_labeled = sum(1 for row in unseen_rows if row["label"] is not None)
    print(f"\n📋 Label Distribution:")
    print(f"  • seen_predictions.csv: {seen_labeled} labeled, {seen_count - seen_labeled} masked")
    print(f"  • unseen_predictions.csv: {unseen_labeled} labeled, {unseen_count - unseen_labeled} masked")
    
    print(f"\n✅ Done! Split complete.")


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    
    parser = argparse.ArgumentParser(
        description="Split FEVER test results into seen/unseen CSV files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with default output directory
  python split_fever_results.py predictions.json 2024-04-01
  
  # Specify output directory
  python split_fever_results.py predictions.json 2024-04-01 --output-dir results/split
  
  # Custom date format (if dates in file are in different format)
  python split_fever_results.py predictions.json 2024-04-01 --date-format "%Y-%m-%d"
  
Output:
  Creates two CSV files:
  - seen_predictions.csv: id,claim,evi,label (unseen claims have label=None)
  - unseen_predictions.csv: id,claim,evi,label (seen claims have label=None)
        """
    )
    
    parser.add_argument(
        "results_file",
        type=str,
        help="Path to FEVER predictions JSON file"
    )
    
    parser.add_argument(
        "split_date",
        type=str,
        help="Temporal split date in YYYY-MM-DD format (e.g., 2024-04-01)"
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=None,
        help="Output directory for CSV files (default: same as input file directory)"
    )
    
    parser.add_argument(
        "--date-format",
        type=str,
        default="%d-%m-%Y",
        help="Date format in input data (default: %%d-%%m-%%Y)"
    )
    
    parser.add_argument(
        "--version", "-v",
        action="version",
        version="%(prog)s 1.0.0"
    )
    
    args = parser.parse_args()
    
    # Validate results file exists
    args.results_file = Path(args.results_file)
    if not args.results_file.exists():
        parser.error(f"Results file not found: {args.results_file}")
    
    # Set output directory
    if args.output_dir:
        args.output_dir = Path(args.output_dir)
    else:
        args.output_dir = args.results_file.parent
    
    # Validate split date format
    try:
        datetime.strptime(args.split_date, "%Y-%m-%d")
    except ValueError:
        parser.error(f"Invalid split date format. Use YYYY-MM-DD (e.g., 2024-04-01)")
    
    return args


def main():
    """Main entry point."""
    
    print("=" * 80)
    print("FEVER Test Results Splitter")
    print("=" * 80)
    
    args = parse_arguments()
    
    split_fever_results(
        results_file=args.results_file,
        split_date=args.split_date,
        output_dir=args.output_dir,
        date_format=args.date_format
    )
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
