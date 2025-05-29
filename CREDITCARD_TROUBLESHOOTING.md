# Credit Card Dataset Troubleshooting Guide

## Common Issues and Solutions

### 1. KeyError: 'trans_status_msg_id'

**Problem**: RGTAN expects categorical features that don't exist in the preprocessed data.

**Solution**: Re-run preprocessing to ensure categorical features are properly encoded:
```bash
python feature_engineering/preprocess_creditcard.py
```

The preprocessing script now:
- Encodes categorical features and saves them with `_encoded` suffix
- RGTAN automatically detects these encoded columns
- Maps them back to original feature names for embeddings

### 2. Memory Issues During Preprocessing

**Problem**: Script gets killed during adjacency list creation.

**Solution**: Use memory optimization parameters:
```bash
python feature_engineering/preprocess_creditcard.py \
    --max-edges-per-entity 50 \
    --min-edge-freq 5
```

### 3. Zero Fraud Rate

**Problem**: All transactions show as non-fraud (0.00% fraud rate).

**Solution**: The script now uses `auth_msg` field to determine fraud:
- Transactions with DECLINE, INSUFF FUNDS, etc. → Fraud (1)
- Transactions with APPROVED → Legitimate (0)

### 4. Missing Columns

**Problem**: Expected columns not found in dataset.

**Solution**: Check your CSV has these required columns:
- `auth_msg` - For fraud labeling
- `amount` - Transaction amount
- `card_number`, `member_id`, `customer_ip`, `customer_email`, `BIN` - For graph edges
- `issue_date` - For temporal features

### 5. Mixed Data Types Warning

**Problem**: DtypeWarning about mixed types in columns.

**Solution**: This is handled automatically by the preprocessing script, which:
- Converts all categorical features to strings
- Handles missing values appropriately
- Encodes them as integers for model input

## Debugging Steps

1. **Check preprocessing output**:
   ```bash
   python feature_engineering/creditcard_label_analysis.py
   ```
   This shows auth_msg distribution and label statistics.

2. **Test data loading**:
   ```bash
   python test_creditcard.py
   ```
   This verifies all components load correctly.

3. **Verify files exist**:
   ```bash
   ls -la data/creditcard_*
   ```
   Should show:
   - creditcard_preprocessed.csv
   - creditcard_homo_adjlists.pickle
   - creditcard_neigh_feat.csv

4. **Check feature availability**:
   The script now prints available categorical features during loading.

## Expected Workflow

1. **Preprocess** (creates all necessary files):
   ```bash
   python feature_engineering/preprocess_creditcard.py
   ```

2. **Run RGTAN**:
   ```bash
   python main.py --method rgtan --creditcard
   ```

## Key Changes Made

1. **Label Creation**: Now based on `auth_msg` patterns, not `IS_TARGETED`
2. **Feature Handling**: Automatically detects and uses preprocessed features
3. **Memory Optimization**: Hub-and-spoke graph pattern for large entities
4. **Categorical Features**: Only uses columns that actually exist in data