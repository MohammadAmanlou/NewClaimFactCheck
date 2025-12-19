# Model Predictions Aggregation Summary

## Overview
Successfully aggregated predictions from 5 different models across 2 FEVER test datasets with seen/unseen subset classification based on model knowledge cutoff dates.

## Model Knowledge Cutoff Dates
| Model | Knowledge Cutoff Date |
|-------|----------------------|
| Claude Sonnet 3.5 (anthropic.claude-3-5-sonnet-20241022-v2:0) | April 1, 2024 |
| GPT-4o-mini | December 31, 2023 |
| Grok-2-1212 | December 12, 2024 |
| Llama 3.1 70B Instruct | March 31, 2024 |
| Qwen 2.5 72B Instruct | September 30, 2024 |

## Generated Files

### 1. `results/fever_test_2023_2024_aggregated_predictions.json`
- **Total Claims**: 2,215
- **Date Range**: All claims are from 2023-2024 period
- **Subset Distribution**: All claims are "seen" by all models (pre-cutoff dates)

#### Model Performance:
| Model | Successful | Failed | Seen | Unseen |
|-------|-----------|--------|------|--------|
| Claude Sonnet 3.5 | 2,213 | 2 | 2,215 | 0 |
| GPT-4o-mini | 2,214 | 1 | 2,215 | 0 |
| Grok-2-1212 | 2,215 | 0 | 2,215 | 0 |
| Llama 3.1 70B | 2,202 | 13 | 2,215 | 0 |
| Qwen 2.5 72B | 2,215 | 0 | 2,215 | 0 |

### 2. `results/fever_test_2025_aggregated_predictions.json`
- **Total Claims**: 1,000
- **Date Range**: Claims from 2025 (more recent, unseen by most models)
- **Subset Distribution**: Mixed seen/unseen depending on model cutoff

#### Model Performance:
| Model | Successful | Failed | Seen | Unseen | Unknown Date |
|-------|-----------|--------|------|--------|--------------|
| Claude Sonnet 3.5 | 999 | 1 | 383 | 616 | 1 |
| GPT-4o-mini | 1,000 | 0 | 119 | 880 | 1 |
| Grok-2-1212 | 1,000 | 0 | 980 | 19 | 1 |
| Llama 3.1 70B | 910 | 90 | 379 | 620 | 1 |
| Qwen 2.5 72B | 999 | 1 | 773 | 226 | 1 |

## JSON Structure

Each aggregated file contains an array of claims with the following structure:

```json
[
  {
    "claim_id": "...",
    "claim": "...",
    "claim_date": "...",
    "speaker": "...",
    "original_claim_url": "...",
    "reporting_source": "...",
    "location_ISO_code": "...",
    "model_predictions": {
      "anthropic.claude-3-5-sonnet-20241022-v2:0": {
        "prediction": "Supported|Refuted|Conflicting Evidence/Cherrypicking|Not Enough Evidence",
        "prediction_error": null,
        "subset": "seen|unseen|unknown",
        "knowledge_cutoff": "2024-04-01"
      },
      "gpt-4o-mini": {
        "prediction": "...",
        "prediction_error": null,
        "subset": "seen|unseen|unknown",
        "knowledge_cutoff": "2023-12-31"
      },
      "grok-2-1212": {
        "prediction": "...",
        "prediction_error": null,
        "subset": "seen|unseen|unknown",
        "knowledge_cutoff": "2024-12-12"
      },
      "llama-3.1-70b-instruct": {
        "prediction": "...",
        "prediction_error": null,
        "subset": "seen|unseen|unknown",
        "knowledge_cutoff": "2024-03-31"
      },
      "qwen2.5-72b-instruct": {
        "prediction": "...",
        "prediction_error": null,
        "subset": "seen|unseen|unknown",
        "knowledge_cutoff": "2024-09-30"
      }
    }
  }
]
```

## Key Insights

### FEVER Test 2023-2024
- All models had seen all claims in their training data
- High prediction success rates across all models (99%+)
- Grok-2-1212 and Qwen 2.5 72B had perfect prediction rates

### FEVER Test 2025
- **GPT-4o-mini** (earliest cutoff: Dec 2023): 88% unseen claims
- **Llama 3.1 70B** (cutoff: Mar 2024): 62% unseen claims
- **Claude Sonnet 3.5** (cutoff: Apr 2024): 62% unseen claims
- **Qwen 2.5 72B** (cutoff: Sep 2024): 23% unseen claims
- **Grok-2-1212** (latest cutoff: Dec 2024): Only 2% unseen claims

- Llama 3.1 70B had the highest failure rate (90 failed predictions)
- All other models maintained >99% success rates

## Usage

These aggregated files can be used for:
1. **Temporal analysis**: Compare model performance on seen vs. unseen claims
2. **Model comparison**: Evaluate which models generalize better to future data
3. **Ensemble methods**: Combine predictions from multiple models
4. **Error analysis**: Identify claims where models disagree or fail
5. **Bias detection**: Examine if models exhibit temporal biases

## Script

The aggregation was performed using `aggregate_model_results.py`, which:
- Reads predictions from each model's results directory
- Determines seen/unseen subset based on claim date vs. knowledge cutoff
- Combines all predictions into a single JSON file per dataset
- Provides detailed statistics on coverage and performance
