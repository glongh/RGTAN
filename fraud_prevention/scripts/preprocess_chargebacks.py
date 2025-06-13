#!/usr/bin/env python3
"""
Preprocessing script for fraud/chargeback detection
Merges dispute_chargeback, ok_transactions, and creates labeled dataset
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import hashlib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def hash_sensitive_data(value):
    """Hash sensitive information for privacy"""
    if pd.isna(value) or value == '':
        return 'MISSING'
    return hashlib.sha256(str(value).encode()).hexdigest()[:16]

def load_and_merge_data(data_path, date_suffix='20250612'):
    """Load chargeback and ok transaction data"""
    print("Loading transaction data...")
    
    # Load chargebacks (fraudulent transactions)
    chargeback_file = os.path.join(data_path, f'dispute_chargeback_{date_suffix}.csv')
    chargebacks = pd.read_csv(chargeback_file)
    chargebacks['is_fraud'] = 1
    print(f"Loaded {len(chargebacks)} chargeback transactions")
    
    # Load ok transactions (legitimate)
    ok_file = os.path.join(data_path, f'ok_transactions_{date_suffix}.csv')
    ok_transactions = pd.read_csv(ok_file)
    ok_transactions['is_fraud'] = 0
    print(f"Loaded {len(ok_transactions)} legitimate transactions")
    
    # Combine datasets
    all_transactions = pd.concat([chargebacks, ok_transactions], ignore_index=True)
    print(f"Total transactions: {len(all_transactions)}")
    print(f"Fraud rate: {all_transactions['is_fraud'].mean():.2%}")
    
    return all_transactions

def preprocess_features(df):
    """Feature engineering and preprocessing"""
    print("\nPreprocessing features...")
    
    # Convert dates to datetime
    date_columns = ['issue_date', 'capture_date', 'created_date', 'updated_date']
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # Sort by issue_date for temporal consistency
    df = df.sort_values('issue_date').reset_index(drop=True)
    
    # Extract temporal features
    df['hour'] = df['issue_date'].dt.hour
    df['day_of_week'] = df['issue_date'].dt.dayofweek
    df['day_of_month'] = df['issue_date'].dt.day
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 6)).astype(int)
    
    # Time to capture (potential indicator of issues)
    df['time_to_capture'] = (df['capture_date'] - df['issue_date']).dt.total_seconds() / 3600
    df['time_to_capture'] = df['time_to_capture'].fillna(0).clip(0, 168)  # Cap at 1 week
    
    # Amount features
    df['amount_log'] = np.log1p(df['amount'])
    df['is_round_amount'] = (df['amount'] % 10 == 0).astype(int)
    df['is_high_amount'] = (df['amount'] > df['amount'].quantile(0.95)).astype(int)
    
    # Hash sensitive fields
    sensitive_fields = ['card_number', 'customer_email', 'customer_ip', 
                       'bill_name1', 'bill_name2', 'ship_name1', 'ship_name2',
                       'bill_street', 'ship_street', 'customer_phone']
    
    for field in sensitive_fields:
        if field in df.columns:
            df[f'{field}_hash'] = df[field].apply(hash_sensitive_data)
            df.drop(columns=[field], inplace=True)
    
    # Extract email domain
    if 'customer_email' in df.columns:
        df['email_domain'] = df['customer_email'].str.split('@').str[-1]
        df['email_domain'] = df['email_domain'].fillna('UNKNOWN')
    
    # Encode categorical variables
    categorical_cols = ['currency_id', 'card_type_id', 'site_tag_id', 'origin_id', 
                       'processor_id', 'BRAND', 'DEBITCREDIT', 'CARDTYPE', 
                       'ISSUERCOUNTRY', 'bill_country', 'email_domain']
    
    label_encoders = {}
    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[f'{col}_encoded'] = le.fit_transform(df[col].fillna('MISSING'))
            label_encoders[col] = le
    
    # Create entity identifiers for graph construction
    df['entity_card'] = df['card_number_hash']
    df['entity_email'] = df['customer_email_hash']
    df['entity_ip'] = df['customer_ip_hash']
    df['entity_bin'] = df['BIN'].fillna('UNKNOWN')
    
    # Transaction ID
    df['trans_id'] = df.index
    
    return df, label_encoders

def create_temporal_splits(df, test_days=30):
    """Create temporal train/test splits"""
    print("\nCreating temporal splits...")
    
    # Calculate cutoff date
    max_date = df['issue_date'].max()
    cutoff_date = max_date - timedelta(days=test_days)
    
    # Split data
    train_mask = df['issue_date'] < cutoff_date
    train_df = df[train_mask].copy()
    test_df = df[~train_mask].copy()
    
    print(f"Train period: {train_df['issue_date'].min()} to {train_df['issue_date'].max()}")
    print(f"Test period: {test_df['issue_date'].min()} to {test_df['issue_date'].max()}")
    print(f"Train size: {len(train_df)} ({train_df['is_fraud'].mean():.2%} fraud)")
    print(f"Test size: {len(test_df)} ({test_df['is_fraud'].mean():.2%} fraud)")
    
    return train_df, test_df

def compute_entity_statistics(train_df):
    """Compute historical statistics per entity"""
    print("\nComputing entity statistics...")
    
    entity_stats = {}
    
    # For each entity type
    for entity in ['entity_card', 'entity_email', 'entity_ip', 'entity_bin']:
        stats = train_df.groupby(entity).agg({
            'is_fraud': ['count', 'mean'],
            'amount': ['mean', 'std', 'min', 'max'],
            'hour': lambda x: x.mode()[0] if len(x) > 0 else 0,
            'day_of_week': lambda x: x.mode()[0] if len(x) > 0 else 0
        }).reset_index()
        
        # Flatten column names
        stats.columns = [f'{entity}' if col[1] == '' else f'{entity}_{col[0]}_{col[1]}' 
                         for col in stats.columns]
        
        entity_stats[entity] = stats
    
    return entity_stats

def save_processed_data(train_df, test_df, entity_stats, label_encoders, output_path):
    """Save processed data for model training"""
    print("\nSaving processed data...")
    
    os.makedirs(output_path, exist_ok=True)
    
    # Save dataframes
    train_df.to_csv(os.path.join(output_path, 'train_transactions.csv'), index=False)
    test_df.to_csv(os.path.join(output_path, 'test_transactions.csv'), index=False)
    
    # Save entity statistics
    for entity, stats in entity_stats.items():
        stats.to_csv(os.path.join(output_path, f'{entity}_stats.csv'), index=False)
    
    # Save label encoders
    import pickle
    with open(os.path.join(output_path, 'label_encoders.pkl'), 'wb') as f:
        pickle.dump(label_encoders, f)
    
    # Save metadata
    metadata = {
        'train_size': len(train_df),
        'test_size': len(test_df),
        'train_fraud_rate': train_df['is_fraud'].mean(),
        'test_fraud_rate': test_df['is_fraud'].mean(),
        'features': list(train_df.columns),
        'processing_date': datetime.now().isoformat()
    }
    
    import json
    with open(os.path.join(output_path, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Data saved to {output_path}")

def main():
    # Paths
    data_path = '../data'
    output_path = '../data/processed'
    
    # Check if data directory exists
    if not os.path.exists(data_path):
        data_path = '../../data'  # Try parent directory
    
    # Load and merge data
    df = load_and_merge_data(data_path)
    
    # Preprocess features
    df, label_encoders = preprocess_features(df)
    
    # Create temporal splits
    train_df, test_df = create_temporal_splits(df, test_days=30)
    
    # Compute entity statistics on training data only
    entity_stats = compute_entity_statistics(train_df)
    
    # Save processed data
    save_processed_data(train_df, test_df, entity_stats, label_encoders, output_path)
    
    print("\nPreprocessing complete!")

if __name__ == "__main__":
    main()