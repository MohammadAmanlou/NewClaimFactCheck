# Fact-Checking Model Inference Pipeline

Production-level inference pipeline for running fact-checking across multiple datasets with temporal analysis and statistical testing.

**Available in two formats:**
- 📓 **Jupyter Notebook** (`run_fact_check_inference.ipynb`) - Interactive exploration
- 🐍 **Python CLI Script** (`run_fact_check_inference.py`) - Command-line automation

## Features

✅ **Multi-Dataset Support**
- ClaimReview Plus (HuggingFace)
- FACTors (CSV format)
- AVeriTeC FEVER (Train/Dev/Test JSON)

✅ **Temporal Split Analysis**
- Automatic splitting into seen/unseen data
- Statistical significance testing (t-test, chi-square)
- Performance comparison across time periods

✅ **Balanced Sampling** (NEW)
- Sample seen data to match unseen data size
- Reproducible with configurable random seed
- Saves sampled claim IDs for reproducibility

✅ **Production-Ready**
- Comprehensive error handling
- Detailed logging
- Configurable parameters
- Clean code structure with type hints
- Relative path addressing

✅ **Flexible Configuration**
- Dataset selection
- Model selection
- Custom label sets
- Rate limiting controls
- Sampling for testing

---

## Quick Start

### Option 1: Python CLI Script (Recommended for Automation)

```bash
# Install dependencies
pip install requests tqdm datasets pandas scipy

# Basic usage
python run_fact_check_inference.py --dataset factors

# With balanced sampling
python run_fact_check_inference.py --dataset factors --balanced-sampling --seed 42

# Custom model
python run_fact_check_inference.py --dataset factors --model gpt-4o-mini --api-key YOUR_KEY

# FEVER test data
python run_fact_check_inference.py --dataset fever_test_2025

# Dry run (validate config without processing)
python run_fact_check_inference.py --dataset factors --dry-run
```

### Option 2: Jupyter Notebook (Interactive)

#### 1. Install Dependencies

```bash
pip install requests tqdm datasets pandas scipy
```

#### 2. Configure Parameters

Edit Section 2 in the notebook:

```python
# Dataset Selection
DATASET_NAME = "fever_train"  # or "factors", "claimreview_plus", etc.

# Model Configuration
MODEL_NAME = "meta.llama3-1-70b-instruct-v1:0"
AVALAI_API_KEY = "your-api-key"

# Label Configuration
CANONICAL_LABELS = ["Supported", "Refuted", "Misleading", "Partially true"]

# Temporal Split
TEMPORAL_SPLIT_DATE = "2024-01-01"

# Balanced Sampling (NEW)
BALANCED_SAMPLING = True   # Sample seen data to match unseen size
SAMPLING_SEED = 42         # Random seed for reproducibility
```

#### 3. Run the Notebook

Execute all cells. The pipeline will:
- Load the selected dataset
- Split into seen/unseen (if labeled data)
- Apply balanced sampling (if enabled)
- Process claims through the model
- Calculate metrics and statistics
- Save results and reports

---

## CLI Arguments Reference

### Required Arguments

| Argument | Short | Description | Default |
|----------|-------|-------------|---------|
| `--dataset` | `-d` | Dataset to use for inference | **Required** |

**Available datasets:**
- `claimreview_plus` - ClaimReview Plus from HuggingFace
- `factors` - FACTors dataset (CSV)
- `fever_train` - FEVER training set (labeled)
- `fever_dev` - FEVER development set (labeled)
- `fever_test_2023_2024` - FEVER test set 2023-2024 (unlabeled)
- `fever_test_2025` - FEVER test set 2025 (unlabeled)

### Model & API Configuration

| Argument | Short | Description | Default |
|----------|-------|-------------|---------|
| `--model` | `-m` | Model name/identifier to use for inference | `anthropic.claude-3-5-sonnet-20241022-v2:0` |
| `--api-key` | | API authentication key for the model provider | `AVALAI_API_KEY` environment variable |
| `--api-url` | | Full URL of the API endpoint | `https://api.avalai.ir/v1/chat/completions` |

**Notes:**
- `--model`: Use the exact model identifier as required by your API provider
- `--api-key`: If not provided, falls back to `AVALAI_API_KEY` environment variable. Script will error if neither is set.
- `--api-url`: Change this if using a different API provider (e.g., OpenAI, xAI, etc.)

### Label Configuration

| Argument | Description | Default |
|----------|-------------|---------|
| `--labels` | Comma-separated list of canonical labels for classification | `false,misleading,partially true,true` |

**Example:**
```bash
# For FEVER-style labels
--labels "supported,refuted,misleading,partially true"

# For binary classification
--labels "true,false"

# Custom labels
--labels "verified,debunked,inconclusive"
```

### Temporal Split Configuration

| Argument | Description | Default |
|----------|-------------|---------|
| `--split-date` | Date to split seen/unseen data (format: YYYY-MM-DD) | `2024-04-01` |
| `--date-format` | Format of dates in the dataset (Python strftime format) | `%d-%m-%Y` |

**Notes:**
- Claims **before** `--split-date` are classified as "seen" (in model's training data)
- Claims **on or after** `--split-date` are classified as "unseen" (after model's knowledge cutoff)
- `--date-format`: Must match the date format in your dataset. Common formats:
  - `%d-%m-%Y` → `25-12-2024`
  - `%Y-%m-%d` → `2024-12-25`
  - `%m/%d/%Y` → `12/25/2024`

### Sampling Configuration

| Argument | Description | Default |
|----------|-------------|---------|
| `--max-samples` | Maximum number of samples to process (for testing/debugging) | `None` (all) |
| `--balanced-sampling` | Enable balanced sampling (sample seen data to match unseen size) | `False` |
| `--seed` | Random seed for balanced sampling (for reproducibility) | `42` |

**Notes:**
- `--max-samples`: Useful for quick testing. Set to 50-100 to validate configuration before full run.
- `--balanced-sampling`: When enabled, randomly samples from "seen" claims to match the number of "unseen" claims. This ensures fair performance comparison.
- `--seed`: Change this value (e.g., 42, 123, 456, 789) to run multiple experiments with different random samples.

### API Rate Limiting

| Argument | Description | Default |
|----------|-------------|---------|
| `--max-retries` | Maximum number of retry attempts for failed API calls | `5` |
| `--retry-delay` | Initial delay (seconds) before retrying. Doubles with each retry (exponential backoff). | `0.5` |
| `--sleep` | Sleep time (seconds) between successful API calls | `0.1` |

**Notes:**
- `--max-retries`: Increase for unstable connections
- `--retry-delay`: Initial delay. Actual delays: 0.5s → 1s → 2s → 4s → 8s
- `--sleep`: **Increase this if you get HTTP 429 (rate limit) errors**. Recommended: 1.0-3.0 for rate-limited APIs.

### Output Configuration

| Argument | Short | Description | Default |
|----------|-------|-------------|---------|
| `--output-dir` | `-o` | Directory to save all output files | `results/<model>/<dataset>` |
| `--display-every` | | Show detailed progress every N claims processed | `10` |

**Notes:**
- `--output-dir`: Model name and dataset are sanitized (`:` and `.` replaced with `_`)
- `--display-every`: Set higher (e.g., 50, 100) for cleaner output on large datasets

### HuggingFace Configuration

| Argument | Description | Default |
|----------|-------------|---------|
| `--hf-token` | HuggingFace API token for accessing private/gated datasets | `HF_TOKEN` environment variable |

**Notes:**
- Only required when using `--dataset claimreview_plus`
- If not provided, falls back to `HF_TOKEN` environment variable

### Utility Options

| Argument | Short | Description | Default |
|----------|-------|-------------|---------|
| `--dry-run` | | Load data and validate configuration without processing | `False` |
| `--version` | `-v` | Show program version and exit | - |

**Notes:**
- `--dry-run`: Useful to verify dataset loading, label configuration, and output paths before committing to a full run
- `--version`: Prints version number (currently 1.0.0) and exits

### CLI Examples

```bash
# Basic FACTors with default settings
python run_fact_check_inference.py -d factors

# FACTors with balanced sampling (different seeds for multiple runs)
python run_fact_check_inference.py -d factors --balanced-sampling --seed 42
python run_fact_check_inference.py -d factors --balanced-sampling --seed 123
python run_fact_check_inference.py -d factors --balanced-sampling --seed 456

# Custom labels for FEVER
python run_fact_check_inference.py -d fever_train --labels "supported,refuted,misleading,partially true"

# Test run with limited samples
python run_fact_check_inference.py -d factors --max-samples 50 --dry-run

# Custom output directory
python run_fact_check_inference.py -d factors -o results/experiment_1

# Increase sleep for rate-limited APIs
python run_fact_check_inference.py -d factors --sleep 2.0
```

## Dataset Configurations

### FACTors Dataset
```python
# Notebook
DATASET_NAME = "factors"
CANONICAL_LABELS = ["True", "False", "Misleading", "Mixture"]

# CLI
python run_fact_check_inference.py -d factors --labels "true,false,misleading,mixture"
```

**Location**: `Datasets/factors.csv`

**Format**: CSV with columns: `claim`, `date_published`, `normalised_rating`, etc.

**Note**: Labels `other` and `unverifiable` are automatically filtered out at load time.

### FEVER Train/Dev (Labeled)
```python
# Notebook
DATASET_NAME = "fever_train"  # or "fever_dev"
CANONICAL_LABELS = ["Supported", "Refuted", "Misleading", "Partially true"]

# CLI
python run_fact_check_inference.py -d fever_train --labels "supported,refuted,misleading,partially true"
```

**Location**: 
- Train: `Datasets/AVeriTeC_FEVER/train.json`
- Dev: `Datasets/AVeriTeC_FEVER/data_dev.json`

**Format**: JSON with `claim`, `label`, `claim_date`, etc.

### FEVER Test (No Labels)
```python
# Notebook
DATASET_NAME = "fever_test_2023_2024"  # or "fever_test_2025"

# CLI
python run_fact_check_inference.py -d fever_test_2025
```

**Location**: 
- Test 2023-2024: `Datasets/AVeriTeC_FEVER/test_2023_2024.json`
- Test 2025: `Datasets/AVeriTeC_FEVER/test_2025.json`

**Format**: JSON with `claim`, `claim_id`, `claim_date` (no labels)

**Output**: Original format + `prediction` field

### ClaimReview Plus
```python
# Notebook
DATASET_NAME = "claimreview_plus"
CANONICAL_LABELS = ["True", "False", "Misleading", "Mixture"]

# CLI
python run_fact_check_inference.py -d claimreview_plus --labels "true,false,misleading,mixture"
```

**Source**: HuggingFace dataset `Webis/claimreview-2025`

## Output Structure

### For Labeled Data (Train/Dev/FACTors/ClaimReview)

```
results/
└── {model_name}/
    └── {dataset_name}/
        ├── main_run_log.txt           # Main execution log
        ├── summary.json                # Overall summary with statistics
        ├── sampling_info.json          # (NEW) Balanced sampling details (if enabled)
        ├── seen/
        │   ├── seen_run_log.txt       # Seen data log
        │   ├── seen_predictions.csv   # Predictions (CSV)
        │   ├── seen_predictions.json  # Predictions (JSON)
        │   └── seen_metrics.json      # Performance metrics
        └── unseen/
            ├── unseen_run_log.txt     # Unseen data log
            ├── unseen_predictions.csv # Predictions (CSV)
            ├── unseen_predictions.json# Predictions (JSON)
            └── unseen_metrics.json    # Performance metrics
```

### For Test Data (FEVER Test)

```
results/
└── {model_name}/
    └── {dataset_name}/
        ├── main_run_log.txt              # Execution log
        ├── predictions_with_labels.json  # Original + predictions
        └── test_summary.json             # Success/failure stats
```

## Output Formats

### Labeled Data Metrics (JSON)
```json
{
  "accuracy": 0.8532,
  "macro_precision": 0.8421,
  "macro_recall": 0.8398,
  "macro_f1": 0.8409,
  "total_samples": 1000,
  "correct_predictions": 853,
  "per_label_metrics": {
    "Supported": {
      "precision": 0.8912,
      "recall": 0.8654,
      "f1": 0.8781,
      "support": 450
    }
  }
}
```

### Statistical Comparison (JSON)
```json
{
  "seen_metrics": { ... },
  "unseen_metrics": { ... },
  "statistical_tests": {
    "t_test": {
      "t_statistic": 2.3456,
      "p_value": 0.0234,
      "significant_at_0.05": true,
      "interpretation": "Significant difference"
    },
    "chi_square_test": {
      "chi2_statistic": 5.4321,
      "p_value": 0.0198,
      "significant_at_0.05": true
    }
  },
  "accuracy_difference": -0.0342
}
```

### Test Data Predictions (JSON)
```json
[
  {
    "claim": "Example claim text",
    "claim_id": 0,
    "claim_date": "22-12-2021",
    "speaker": "SPEAKER NAME",
    "original_claim_url": "https://...",
    "reporting_source": "twitter",
    "location_ISO_code": "US",
    "prediction": "Refuted"
  }
]
```

## Configuration Parameters

### Rate Limiting
```python
MAX_RETRIES = 3                    # Max retry attempts for API calls
INITIAL_RETRY_DELAY = 0.5          # Initial retry delay (seconds)
SLEEP_BETWEEN_CALLS = 1.0          # Delay between successful calls
```

**Important**: If you encounter HTTP 429 errors, increase `SLEEP_BETWEEN_CALLS` to 2.0 or 3.0 seconds.

### Sampling
```python
# Notebook
MAX_SAMPLES = None  # Process all data
MAX_SAMPLES = 100   # Process only first 100 samples (testing)

# CLI
python run_fact_check_inference.py -d factors --max-samples 100
```

### Balanced Sampling (NEW)

For datasets with imbalanced seen/unseen splits (e.g., FACTors), enable balanced sampling to sample seen data to match unseen data size:

```python
# Notebook
BALANCED_SAMPLING = True   # Enable balanced sampling
SAMPLING_SEED = 42         # Random seed for reproducibility

# CLI
python run_fact_check_inference.py -d factors --balanced-sampling --seed 42
```

**Output** (`sampling_info.json`):
```json
{
  "enabled": true,
  "seed": 42,
  "original_seen_count": 15000,
  "sampled_seen_count": 3500,
  "unseen_count": 3500,
  "sampled_claim_ids": [123, 456, 789, ...],
  "timestamp": "2025-12-21T10:30:00"
}
```

**Use Cases:**
- Run multiple experiments with different seeds (42, 123, 456) for robust evaluation
- Ensure fair comparison between seen and unseen performance
- Reproducible sampling with saved claim IDs

### Display
```python
# Notebook
DISPLAY_EVERY_N = 10  # Show progress every 10 claims

# CLI
python run_fact_check_inference.py -d factors --display-every 10
```

### Temporal Split
```python
# Notebook
TEMPORAL_SPLIT_DATE = "2024-01-01"  # Split date (YYYY-MM-DD)
DATE_FORMAT = "%d-%m-%Y"            # Date format in datasets

# CLI
python run_fact_check_inference.py -d factors --split-date 2024-01-01 --date-format "%d-%m-%Y"
```

## Label Normalization

The pipeline includes intelligent label normalization:
- Case-insensitive matching
- Partial matching
- Common label mappings
- Filtering of unverifiable/inconclusive labels

### Common Mappings
- "true" → "Supported"
- "false" → "Refuted"
- "mostly true" → "Supported"
- "half true" → "Partially true"
- "mixture" → "Misleading"

You can customize label sets for each dataset in the configuration.

## API Configuration

### AVALAI API
```python
# Notebook
AVALAI_API_KEY = os.environ.get("AVALAI_API_KEY", "your-key-here")
AVALAI_CHAT_URL = "https://api.avalai.ir/v1/chat/completions"

# CLI
python run_fact_check_inference.py -d factors --api-key YOUR_KEY --api-url https://api.avalai.ir/v1/chat/completions
```

**Recommended**: Store API key in environment variable:
```bash
# Linux/macOS
export AVALAI_API_KEY="your-key-here"

# Windows PowerShell
$env:AVALAI_API_KEY = "your-key-here"

# Windows CMD
set AVALAI_API_KEY=your-key-here
```

## Troubleshooting

### HTTP 429 (Rate Limit) Errors
**Solution**: Increase sleep time between API calls
```bash
# CLI
python run_fact_check_inference.py -d factors --sleep 2.0

# Notebook: Set SLEEP_BETWEEN_CALLS = 2.0
```

### Memory Issues with Large Datasets
**Solution**: Use balanced sampling or limit samples
```bash
# CLI
python run_fact_check_inference.py -d factors --balanced-sampling
python run_fact_check_inference.py -d factors --max-samples 1000

# Notebook: Set BALANCED_SAMPLING = True or MAX_SAMPLES = 1000
```

### Date Parsing Errors
**Solution**: Verify date format matches your dataset
```bash
# CLI
python run_fact_check_inference.py -d factors --date-format "%Y-%m-%d"
```

### Missing Labels in Output
**Solution**: Check that canonical labels include all labels in your dataset

### API Key Not Found
**Solution**: Set environment variable or pass via CLI
```bash
export AVALAI_API_KEY="your-key"
# or
python run_fact_check_inference.py -d factors --api-key "your-key"
```

## Best Practices

1. **Testing**: Start with `--max-samples 100` or `--dry-run` to test configuration
2. **Rate Limits**: Monitor logs for 429 errors and adjust `--sleep` time
3. **Label Sets**: Always verify canonical labels match dataset
4. **Balanced Sampling**: Use for large datasets like FACTors to ensure fair comparison
5. **Multiple Seeds**: Run with different `--seed` values (42, 123, 456) for robust evaluation
6. **Backups**: Results are automatically saved with timestamps
7. **Logs**: Check log files for detailed execution information
8. **CLI for Automation**: Use Python script for batch processing and CI/CD pipelines

## Statistical Testing

For labeled data, the pipeline performs:
- **T-test**: Tests if accuracy difference is statistically significant
- **Chi-square test**: Tests independence of performance and time period
- **Significance level**: α = 0.05

Results indicate whether model performance differs significantly between seen and unseen data.

## Code Structure

### Notebook (`run_fact_check_inference.ipynb`)
- **Section 1**: Setup & Installation
- **Section 2**: Configuration (MODIFY HERE)
- **Section 3**: Utility Functions (logging, normalization, API)
- **Section 4**: Dataset Loaders (multi-format support)
- **Section 5**: Temporal Split & Processing
- **Section 6**: Evaluation & Statistics
- **Section 7**: Output & Reporting
- **Section 8**: Main Execution (RUN THIS)
- **Section 9**: Configuration Examples

### CLI Script (`run_fact_check_inference.py`)
- Standalone Python script with argparse
- All notebook functionality in one file
- Production-ready with proper error handling
- No Jupyter dependency required

## Files

| File | Description |
|------|-------------|
| `run_fact_check_inference.ipynb` | Interactive Jupyter notebook |
| `run_fact_check_inference.py` | Command-line Python script |
| `README_INFERENCE.md` | This documentation |

## Contact & Support

For issues or questions about:
- Dataset formats
- Model configuration
- Statistical interpretation
- Custom label sets

Refer to the inline documentation in the notebook/script or check the execution logs.

---

**Last Updated**: December 2025
**Version**: 1.1.0
