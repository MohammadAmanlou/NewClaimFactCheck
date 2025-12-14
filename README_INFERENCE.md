# Fact-Checking Model Inference Pipeline

Production-level notebook for running fact-checking inference across multiple datasets with temporal analysis and statistical testing.

## Features

✅ **Multi-Dataset Support**
- ClaimReview Plus (HuggingFace)
- FACTors (CSV format)
- AVeriTeC FEVER (Train/Dev/Test JSON)

✅ **Temporal Split Analysis**
- Automatic splitting into seen/unseen data
- Statistical significance testing (t-test, chi-square)
- Performance comparison across time periods

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

## Quick Start

### 1. Install Dependencies

```bash
pip install requests tqdm datasets pandas scipy
```

### 2. Configure Parameters

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
```

### 3. Run the Notebook

Execute all cells. The pipeline will:
- Load the selected dataset
- Split into seen/unseen (if labeled data)
- Process claims through the model
- Calculate metrics and statistics
- Save results and reports

## Dataset Configurations

### FACTors Dataset
```python
DATASET_NAME = "factors"
CANONICAL_LABELS = ["True", "False", "Misleading", "Mixture"]
```

**Location**: `Datasets/FACTor/FACTors.csv`

**Format**: CSV with columns: `claim`, `date_published`, `normalised_rating`, etc.

### FEVER Train/Dev (Labeled)
```python
DATASET_NAME = "fever_train"  # or "fever_dev"
CANONICAL_LABELS = ["Supported", "Refuted", "Misleading", "Partially true"]
```

**Location**: 
- Train: `Datasets/AVeriTeC_FEVER/train.json`
- Dev: `Datasets/AVeriTeC_FEVER/data_dev.json`

**Format**: JSON with `claim`, `label`, `claim_date`, etc.

### FEVER Test (No Labels)
```python
DATASET_NAME = "fever_test_2023_2024"  # or "fever_test_2025"
```

**Location**: 
- Test 2023-2024: `Datasets/AVeriTeC_FEVER/test_2023_2024.json`
- Test 2025: `Datasets/AVeriTeC_FEVER/test_2025.json`

**Format**: JSON with `claim`, `claim_id`, `claim_date` (no labels)

**Output**: Original format + `prediction` field

### ClaimReview Plus
```python
DATASET_NAME = "claimreview_plus"
CANONICAL_LABELS = ["True", "False", "Misleading", "Mixture"]
```

**Source**: HuggingFace dataset `shaharpit809/ClaimReview-Plus`

## Output Structure

### For Labeled Data (Train/Dev/FACTors/ClaimReview)

```
results/
└── {model_name}/
    └── {dataset_name}/
        ├── main_run_log.txt           # Main execution log
        ├── summary.json                # Overall summary with statistics
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
MAX_SAMPLES = None  # Process all data
# or
MAX_SAMPLES = 100   # Process only first 100 samples (testing)
```

### Display
```python
DISPLAY_EVERY_N = 10  # Show progress every 10 claims
```

### Temporal Split
```python
TEMPORAL_SPLIT_DATE = "2024-01-01"  # Split date (YYYY-MM-DD)
DATE_FORMAT = "%d-%m-%Y"            # Date format in datasets
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
AVALAI_API_KEY = os.environ.get("AVALAI_API_KEY", "your-key-here")
AVALAI_CHAT_URL = "https://api.avalai.ir/v1/chat/completions"
```

**Recommended**: Store API key in environment variable:
```bash
export AVALAI_API_KEY="your-key-here"
```

## Troubleshooting

### HTTP 429 (Rate Limit) Errors
**Solution**: Increase `SLEEP_BETWEEN_CALLS` to 2.0 or 3.0 seconds

### Memory Issues with Large Datasets
**Solution**: Set `MAX_SAMPLES` to process data in batches

### Date Parsing Errors
**Solution**: Verify `DATE_FORMAT` matches your dataset's date format

### Missing Labels in Output
**Solution**: Check that `CANONICAL_LABELS` includes all labels in your dataset

## Best Practices

1. **Testing**: Start with `MAX_SAMPLES = 100` to test configuration
2. **Rate Limits**: Monitor logs for 429 errors and adjust sleep time
3. **Label Sets**: Always verify canonical labels match dataset
4. **Backups**: Results are automatically saved with timestamps
5. **Logs**: Check log files for detailed execution information

## Statistical Testing

For labeled data, the pipeline performs:
- **T-test**: Tests if accuracy difference is statistically significant
- **Chi-square test**: Tests independence of performance and time period
- **Significance level**: α = 0.05

Results indicate whether model performance differs significantly between seen and unseen data.

## Code Structure

- **Section 1**: Setup & Installation
- **Section 2**: Configuration (MODIFY HERE)
- **Section 3**: Utility Functions (logging, normalization, API)
- **Section 4**: Dataset Loaders (multi-format support)
- **Section 5**: Temporal Split & Processing
- **Section 6**: Evaluation & Statistics
- **Section 7**: Output & Reporting
- **Section 8**: Main Execution (RUN THIS)
- **Section 9**: Configuration Examples

## Contact & Support

For issues or questions about:
- Dataset formats
- Model configuration
- Statistical interpretation
- Custom label sets

Refer to the inline documentation in the notebook or check the execution logs.

---

**Last Updated**: December 2025
**Version**: 1.0.0
