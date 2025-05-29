# Realistic Fraud Detection Guide

This guide explains the realistic fraud detection setup that avoids data leakage.

## Key Differences from Original

### 1. **Fraud Labels**
- **Original**: Uses `auth_msg` to determine fraud (actually predicting declines)
- **Realistic**: Simulates fraud based on transaction patterns:
  - ~1-2% fraud rate (realistic for credit cards)
  - Even approved transactions can be fraud
  - Based on: unusual amounts, time patterns, velocity, foreign transactions

### 2. **No Data Leakage**
- **Feature Engineering**: Only uses past training data
- **Graph Construction**: No edges between test nodes
- **Neighborhood Features**: Computed only from training graph
- **Scaling/Encoding**: Fit only on training data

### 3. **Temporal Integrity**
- **Train/Test Split**: Strictly time-based (last 30% is test)
- **Aggregated Features**: Only computed from past transactions
- **No Future Information**: Features never use data from the future

## Usage

### Step 1: Preprocess with Realistic Labels
```bash
python feature_engineering/preprocess_creditcard_realistic.py \
    --input data/vod_creditcard.csv \
    --output data/ \
    --test-size 0.3
```

This creates:
- `creditcard_realistic_preprocessed.csv` - Features with no leakage
- `creditcard_realistic_adjlists.pickle` - Graph structure
- `creditcard_realistic_neigh_feat.csv` - Neighborhood features
- `creditcard_realistic_splits.pickle` - Train/test split info

### Step 2: Run RGTAN with Realistic Setup
```bash
python main.py --method rgtan --creditcard --realistic
```

## Expected Performance

### Realistic Fraud Detection Metrics:
- **AUC**: 0.70-0.85 (excellent: >0.80)
- **F1**: 0.20-0.40 (due to extreme imbalance)
- **Precision@1%**: 0.40-0.60 (catching 40-60% of frauds in top 1%)
- **AP**: 0.10-0.30 (due to ~1-2% fraud rate)

### Why Lower Than Before?
1. **Harder Problem**: Real fraud vs. predicting bank decisions
2. **No Leakage**: Can't use future information
3. **Class Imbalance**: ~1-2% fraud vs ~40% declines
4. **Realistic Noise**: Approved frauds, declined legitimates

## What Makes This Realistic?

1. **Fraud Patterns**:
   - High amounts relative to history
   - Unusual times (late night)
   - Foreign transactions
   - Velocity (many transactions quickly)

2. **Authorized Fraud**:
   - ~0.5% of approved transactions are fraud
   - Models must detect fraud that passed authorization

3. **Temporal Reality**:
   - Can only use past data for features
   - Simulates real-time scoring scenario

4. **Graph Constraints**:
   - Test nodes can't see other test nodes
   - Simulates scoring new transactions

## Comparing Results

Run both versions to see the difference:

```bash
# Unrealistic (predicting declines)
python main.py --method rgtan --creditcard

# Realistic (detecting fraud)
python main.py --method rgtan --creditcard --realistic
```

The realistic version will show much lower metrics, but these reflect actual fraud detection performance rather than decline prediction.