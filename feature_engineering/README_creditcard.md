# Credit Card Fraud Detection Preprocessing

This document describes the preprocessing pipeline for credit card fraud detection using RGTAN.

## Overview

The preprocessing pipeline (`preprocess_creditcard.py`) transforms raw credit card transaction data into a format optimized for graph-based fraud detection. It creates:

1. **Engineered Features**: Temporal, velocity, and aggregated risk features
2. **Graph Structure**: Adjacency lists based on shared entities
3. **Neighborhood Statistics**: Risk features from connected transactions

## Feature Engineering

### 1. Temporal Features
- **Hour of Day**: Transaction timing patterns
- **Day of Week**: Weekly patterns
- **Time Since First Transaction**: Seconds/hours/days elapsed
- **Time Between Transactions**: Velocity for same card/user

### 2. Aggregated Risk Features
For each key entity (card, member, IP, email, BIN):
- Previous transaction count
- Running fraud rate (excluding current transaction)
- Average and standard deviation of amounts
- Transaction frequency (transactions per day)

### 3. Neighborhood Features
- **1-hop statistics**: Direct neighbor fraud rate, count, amounts
- **2-hop statistics**: Extended neighborhood patterns (optional)

## Graph Construction

Edges are created between transactions that share:
- **Card Number**: Same payment instrument
- **Member ID**: Same account
- **Customer IP**: Same network location
- **Customer Email**: Same contact information
- **BIN**: Same card issuer

### Edge Frequency Threshold
Use `--min-edge-freq` to control edge creation:
- `2` (default): Create edges only if ≥2 transactions share an entity
- Higher values reduce noise but may miss patterns

## Privacy Protection

By default, sensitive data is hashed:
- Card numbers → SHA256 hash (16 chars)
- Emails, IPs → SHA256 hash
- Names, addresses → SHA256 hash

Use `--no-privacy` to keep original values (for debugging only).

## Output Files

1. **creditcard_preprocessed.csv**: Enhanced dataset with all features
2. **creditcard_homo_adjlists.pickle**: Graph adjacency lists
3. **creditcard_neigh_feat.csv**: Neighborhood risk statistics
4. **creditcard_preprocessing_metadata.pickle**: Processing configuration

## Usage Examples

### Basic Preprocessing
```bash
python feature_engineering/preprocess_creditcard.py
```

### Custom Configuration
```bash
python feature_engineering/preprocess_creditcard.py \
    --input data/my_transactions.csv \
    --output preprocessed/ \
    --min-edge-freq 5 \
    --no-privacy
```

### Integration with RGTAN
```bash
# Step 1: Preprocess
python feature_engineering/preprocess_creditcard.py

# Step 2: Train RGTAN
python main.py --method rgtan --creditcard
```

## Performance Considerations

- **Memory**: Large datasets may require significant RAM for graph construction
- **Time**: 2-hop features are expensive; disabled for datasets >50K transactions
- **Storage**: Preprocessed files are larger due to additional features

## Feature Importance

Based on fraud detection literature, the most important features typically are:
1. **Velocity features**: Rapid successive transactions
2. **Amount anomalies**: Deviation from historical patterns
3. **Neighborhood fraud rate**: Risk from connected entities
4. **Time patterns**: Unusual hours or days
5. **Geographic patterns**: Multiple locations rapidly