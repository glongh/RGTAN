# Credit Card Dataset Usage Guide for RGTAN

This guide explains how to use the credit card fraud detection dataset with RGTAN.

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare the Data

#### Option A: With Preprocessing (Recommended)
```bash
# Run the preprocessing script
python feature_engineering/preprocess_creditcard.py

# This will create:
# - data/creditcard_preprocessed.csv
# - data/creditcard_homo_adjlists.pickle  
# - data/creditcard_neigh_feat.csv
# - data/creditcard_preprocessing_metadata.pickle
```

#### Option B: Without Preprocessing
The system will use basic features from the raw data file.

### 3. Run RGTAN Training

```bash
# Run RGTAN with creditcard dataset
python main.py --method rgtan --creditcard
```

## Command Line Options

```bash
# Basic usage
python main.py --method rgtan --creditcard

# Show help
python main.py --help

# Preprocessing options
python feature_engineering/preprocess_creditcard.py --help
```

## Configuration

The creditcard configuration is stored in `config/creditcard_cfg.yaml`:

```yaml
dataset: creditcard
batch_size: 64
n_layers: 2
dropout: [0.2, 0.1]
lr: 0.003
max_epochs: 15
test_size: 0.3  # Time-based split
nei_att_heads:
    creditcard: 8
```

## Data Requirements

The following file must exist:
- `data/vod_creditcard.csv` - Raw credit card transaction data

Optional preprocessed files (created by preprocessing script):
- `data/creditcard_preprocessed.csv` - Enhanced features
- `data/creditcard_homo_adjlists.pickle` - Graph structure
- `data/creditcard_neigh_feat.csv` - Neighborhood features

## Expected Columns in vod_creditcard.csv

Key columns used:
- `IS_TARGETED` - Fraud label (yes/no)
- `amount` - Transaction amount
- `card_number` - Card identifier (will be hashed)
- `member_id` - User account ID
- `customer_ip` - IP address (will be hashed)
- `customer_email` - Email (will be hashed)
- `BIN` - Bank Identification Number
- Date columns: `issue_date`, `capture_date`, etc.

## Output

The model will output:
- Training metrics per epoch
- Final test set performance (AUC, F1, AP)
- Model checkpoints in `models/` directory

## Troubleshooting

### "Preprocessed data not found" message
Run the preprocessing script first:
```bash
python feature_engineering/preprocess_creditcard.py
```

### Memory issues
Reduce batch size in config file or use fewer edge types.

### Missing columns
Ensure your CSV has the required columns, especially `IS_TARGETED`.

## Advanced Usage

### Custom preprocessing
```bash
python feature_engineering/preprocess_creditcard.py \
    --input path/to/your/data.csv \
    --output path/to/output/ \
    --min-edge-freq 5 \
    --no-privacy  # Don't hash sensitive data
```

### Modify configuration
Edit `config/creditcard_cfg.yaml` to adjust:
- Learning rate
- Batch size
- Number of layers
- Attention heads
- Edge types used