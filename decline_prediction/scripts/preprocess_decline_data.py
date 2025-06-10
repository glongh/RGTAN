#!/usr/bin/env python3
"""
Preprocessing script for decline prediction
Prepares denied and approved transaction data for real-time decline prediction
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import hashlib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import pickle
import json
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def hash_sensitive_data(value):
    """Hash sensitive information for privacy"""
    if pd.isna(value) or value == '':
        return 'MISSING'
    return hashlib.sha256(str(value).encode()).hexdigest()[:16]

def load_decline_data(data_path, date_suffix='20250609'):
    """Load denied and approved transaction data"""
    print("Loading transaction data for decline prediction...")
    
    # Load denied transactions
    denied_file = os.path.join(data_path, f'denied_transactions_{date_suffix}.csv')
    denied_df = pd.read_csv(denied_file)
    denied_df['is_declined'] = 1
    print(f"Loaded {len(denied_df)} denied transactions")
    
    # Load approved transactions (ok_transactions)
    approved_file = os.path.join(data_path, f'ok_transactions_20250606.csv')  # Using available file
    approved_df = pd.read_csv(approved_file)
    approved_df['is_declined'] = 0
    print(f"Loaded {len(approved_df)} approved transactions")
    
    # Sample approved transactions to balance dataset (optional)
    # For real-time prediction, we want to see both approval and decline patterns
    sample_size = min(len(denied_df) * 2, len(approved_df))  # 2:1 ratio
    if len(approved_df) > sample_size:
        approved_df = approved_df.sample(n=sample_size, random_state=42)
        print(f"Sampled {len(approved_df)} approved transactions for balance")
    
    # Combine datasets
    all_transactions = pd.concat([denied_df, approved_df], ignore_index=True)
    print(f"Total transactions: {len(all_transactions)}")
    print(f"Decline rate: {all_transactions['is_declined'].mean():.2%}")
    
    return all_transactions

def analyze_decline_reasons(df):
    """Analyze decline reasons to create targeted models"""
    print("\nAnalyzing decline reasons...")
    
    decline_reasons = df[df['is_declined'] == 1]['auth_msg'].value_counts()
    print("Top decline reasons:")
    for reason, count in decline_reasons.head(10).items():
        print(f"  {reason}: {count:,} ({count/len(df[df['is_declined']==1]):.1%})")
    
    # Categorize decline reasons
    decline_categories = {
        'insufficient_funds': ['INSUFF FUNDS', 'INSUFFICIENT FUNDS', 'NSF'],
        'invalid_account': ['NO ACCOUNT', 'INVALID ACCOUNT', 'ACCT NOT FOUND'],
        'invalid_merchant': ['INVALID MERCHANT', 'MERCHANT NOT FOUND'],
        'security_decline': ['SECURITY VIOLATION', 'FRAUD SUSPECTED'],
        'expired_card': ['EXPIRED CARD', 'CARD EXPIRED'],
        'limit_exceeded': ['LIMIT EXCEEDED', 'OVER LIMIT'],
        'other': []
    }
    
    # Map decline reasons to categories
    def categorize_decline(auth_msg):
        if pd.isna(auth_msg):
            return 'other'
        auth_msg = str(auth_msg).upper()
        for category, keywords in decline_categories.items():
            if any(keyword in auth_msg for keyword in keywords):
                return category
        return 'other'
    
    df['decline_category'] = df['auth_msg'].apply(categorize_decline)
    
    print("\nDecline categories:")
    decline_category_counts = df[df['is_declined'] == 1]['decline_category'].value_counts()
    for category, count in decline_category_counts.items():
        print(f"  {category}: {count:,}")
    
    return df

def create_realtime_features(df):
    """Create features optimized for real-time prediction"""
    print("\nCreating real-time optimized features...")
    
    # Convert dates to datetime
    date_columns = ['issue_date', 'capture_date', 'created_date', 'updated_date']
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # Sort by issue_date for temporal consistency
    df = df.sort_values('issue_date').reset_index(drop=True)
    
    # Basic temporal features (fast to compute)
    df['hour'] = df['issue_date'].dt.hour
    df['day_of_week'] = df['issue_date'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['is_business_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 17)).astype(int)
    df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 6)).astype(int)
    
    # Amount features (instant computation)
    df['amount_log'] = np.log1p(df['amount'])
    df['is_round_amount'] = (df['amount'] % 10 == 0).astype(int)
    df['amount_cents'] = ((df['amount'] * 100) % 100).astype(int)
    df['is_exact_dollar'] = (df['amount_cents'] == 0).astype(int)
    
    # High/low amount flags (pre-computed thresholds)
    amount_p95 = df['amount'].quantile(0.95)
    amount_p05 = df['amount'].quantile(0.05)
    df['is_high_amount'] = (df['amount'] > amount_p95).astype(int)
    df['is_low_amount'] = (df['amount'] < amount_p05).astype(int)
    
    # Card type features
    df['is_debit'] = (df['DEBITCREDIT'] == 'DEBIT').astype(int)
    df['is_credit'] = (df['DEBITCREDIT'] == 'CREDIT').astype(int)
    
    # Geographic features
    df['is_domestic'] = (df['ISSUERCOUNTRY'] == df['bill_country']).astype(int)
    df['has_billing_zip'] = (~df['bill_zip'].isna()).astype(int)
    
    # Hash sensitive fields
    sensitive_fields = ['card_number', 'customer_email', 'customer_ip', 
                       'bill_name1', 'bill_name2', 'customer_phone']
    
    for field in sensitive_fields:
        if field in df.columns:
            df[f'{field}_hash'] = df[field].apply(hash_sensitive_data)
            df.drop(columns=[field], inplace=True)
    
    # Create entity identifiers for real-time graph lookup
    df['entity_card'] = df['card_number_hash']
    df['entity_email'] = df['customer_email_hash']
    df['entity_ip'] = df['customer_ip_hash']
    df['entity_bin'] = df['BIN'].fillna('UNKNOWN')
    df['entity_merchant'] = df['site_tag_id'].fillna('UNKNOWN')
    
    # Encode categorical variables
    categorical_cols = ['currency_id', 'card_type_id', 'site_tag_id', 'origin_id', 
                       'processor_id', 'BRAND', 'CARDTYPE', 'ISSUERCOUNTRY', 
                       'bill_country', 'decline_category']
    
    label_encoders = {}
    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[f'{col}_encoded'] = le.fit_transform(df[col].fillna('MISSING'))
            label_encoders[col] = le
    
    # Transaction ID
    df['trans_id'] = df.index
    
    return df, label_encoders

def create_velocity_features(df):
    """Create velocity features for decline prediction"""
    print("\nCreating velocity features...")
    
    # Sort by issue_date
    df = df.sort_values('issue_date')
    
    # Calculate time-based features per entity
    velocity_features = {}
    
    for entity in ['entity_card', 'entity_email', 'entity_ip', 'entity_merchant']:
        if entity in df.columns:
            print(f"  Processing {entity} velocity...")
            
            # Group by entity and calculate features
            df[f'{entity}_count'] = df.groupby(entity).cumcount()
            
            # Time since last transaction (in hours)
            df['prev_time'] = df.groupby(entity)['issue_date'].shift(1)
            df[f'{entity}_hours_since_last'] = (
                df['issue_date'] - df['prev_time']
            ).dt.total_seconds() / 3600
            df[f'{entity}_hours_since_last'] = df[f'{entity}_hours_since_last'].fillna(24).clip(0, 168)
            
            # Transaction frequency (transactions per day)
            df[f'{entity}_txn_per_day'] = df.groupby(entity)['trans_id'].rolling(
                window='24H', on='issue_date', min_periods=1
            ).count().reset_index(level=0, drop=True)
            
            # Amount patterns
            df[f'{entity}_avg_amount'] = df.groupby(entity)['amount'].expanding().mean().reset_index(level=0, drop=True)
            df[f'{entity}_amount_ratio'] = df['amount'] / df[f'{entity}_avg_amount']
            df[f'{entity}_amount_ratio'] = df[f'{entity}_amount_ratio'].fillna(1).clip(0, 10)
    
    # Clean up temporary columns
    df.drop(columns=['prev_time'], inplace=True)
    
    return df

def create_temporal_splits(df, test_days=15):
    """Create temporal train/test splits optimized for real-time scenario"""
    print("\nCreating temporal splits for real-time model...")
    
    # Calculate cutoff date (more recent test set for real-time simulation)
    max_date = df['issue_date'].max()
    cutoff_date = max_date - timedelta(days=test_days)
    
    # Split data
    train_mask = df['issue_date'] < cutoff_date
    train_df = df[train_mask].copy()
    test_df = df[~train_mask].copy()
    
    print(f"Train period: {train_df['issue_date'].min()} to {train_df['issue_date'].max()}")
    print(f"Test period: {test_df['issue_date'].min()} to {test_df['issue_date'].max()}")
    print(f"Train size: {len(train_df)} ({train_df['is_declined'].mean():.2%} decline)")
    print(f"Test size: {len(test_df)} ({test_df['is_declined'].mean():.2%} decline)")
    
    return train_df, test_df

def compute_entity_profiles(train_df):
    """Compute entity profiles for real-time lookup"""
    print("\nComputing entity profiles for real-time lookup...")
    
    entity_profiles = {}
    
    # For each entity type, compute historical statistics
    for entity in ['entity_card', 'entity_email', 'entity_ip', 'entity_merchant', 'entity_bin']:
        if entity in train_df.columns:
            profiles = train_df.groupby(entity).agg({
                'is_declined': ['count', 'mean', 'sum'],
                'amount': ['mean', 'std', 'min', 'max'],
                'hour': lambda x: x.mode()[0] if len(x) > 0 else 12,
                'day_of_week': lambda x: x.mode()[0] if len(x) > 0 else 1,
                'issue_date': ['min', 'max']
            }).reset_index()
            
            # Flatten column names
            profiles.columns = [f'{entity}' if col[1] == '' else f'{entity}_{col[0]}_{col[1]}' 
                               for col in profiles.columns]
            
            # Calculate additional metrics
            profiles[f'{entity}_decline_rate'] = profiles[f'{entity}_is_declined_sum'] / profiles[f'{entity}_is_declined_count']
            profiles[f'{entity}_total_amount'] = profiles[f'{entity}_amount_mean'] * profiles[f'{entity}_is_declined_count']
            profiles[f'{entity}_days_active'] = (
                profiles[f'{entity}_issue_date_max'] - profiles[f'{entity}_issue_date_min']
            ).dt.days + 1
            
            entity_profiles[entity] = profiles
    
    return entity_profiles

def save_processed_data(train_df, test_df, entity_profiles, label_encoders, output_path):
    """Save processed data for model training and real-time inference"""
    print("\nSaving processed data...")
    
    os.makedirs(output_path, exist_ok=True)
    
    # Save dataframes
    train_df.to_csv(os.path.join(output_path, 'train_decline_data.csv'), index=False)
    test_df.to_csv(os.path.join(output_path, 'test_decline_data.csv'), index=False)
    
    # Save entity profiles for real-time lookup
    for entity, profiles in entity_profiles.items():
        profiles.to_csv(os.path.join(output_path, f'{entity}_profiles.csv'), index=False)
    
    # Save label encoders
    with open(os.path.join(output_path, 'decline_label_encoders.pkl'), 'wb') as f:
        pickle.dump(label_encoders, f)
    
    # Create feature lists for real-time inference
    feature_columns = {
        'numeric_features': [col for col in train_df.columns 
                           if col.startswith(('amount', 'hour', 'day_of_week', 'is_', 
                                            'entity_card_', 'entity_email_', 'entity_ip_', 
                                            'entity_merchant_', 'entity_bin_'))],
        'categorical_features': [col for col in train_df.columns if col.endswith('_encoded')],
        'entity_columns': ['entity_card', 'entity_email', 'entity_ip', 'entity_merchant', 'entity_bin'],
        'target_column': 'is_declined'
    }
    
    # Save feature configuration
    with open(os.path.join(output_path, 'feature_config.json'), 'w') as f:
        json.dump(feature_columns, f, indent=2)
    
    # Save decline reason analysis
    decline_analysis = {
        'decline_categories': train_df[train_df['is_declined'] == 1]['decline_category'].value_counts().to_dict(),
        'decline_rate_by_hour': train_df.groupby('hour')['is_declined'].mean().to_dict(),
        'decline_rate_by_amount_range': train_df.groupby(pd.cut(train_df['amount'], 10))['is_declined'].mean().to_dict(),
        'processing_date': datetime.now().isoformat()
    }
    
    with open(os.path.join(output_path, 'decline_analysis.json'), 'w') as f:
        json.dump(decline_analysis, f, indent=2, default=str)
    
    # Save metadata
    metadata = {
        'train_size': len(train_df),
        'test_size': len(test_df),
        'train_decline_rate': train_df['is_declined'].mean(),
        'test_decline_rate': test_df['is_declined'].mean(),
        'features_count': len(feature_columns['numeric_features']) + len(feature_columns['categorical_features']),
        'entity_types': len(feature_columns['entity_columns']),
        'processing_date': datetime.now().isoformat()
    }
    
    with open(os.path.join(output_path, 'decline_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Data saved to {output_path}")
    print(f"Features: {metadata['features_count']} total")
    print(f"Entity types: {metadata['entity_types']}")

def main():
    # Paths
    data_path = '../data'
    output_path = '../data/processed'
    
    # Check if data directory exists
    if not os.path.exists(data_path):
        data_path = '../../data'  # Try parent directory
    
    # Load and merge data
    df = load_decline_data(data_path)
    
    # Analyze decline reasons
    df = analyze_decline_reasons(df)
    
    # Create features optimized for real-time processing
    df, label_encoders = create_realtime_features(df)
    
    # Add velocity features
    df = create_velocity_features(df)
    
    # Create temporal splits
    train_df, test_df = create_temporal_splits(df, test_days=15)
    
    # Compute entity profiles on training data only
    entity_profiles = compute_entity_profiles(train_df)
    
    # Save processed data
    save_processed_data(train_df, test_df, entity_profiles, label_encoders, output_path)
    
    print("\nDecline prediction preprocessing complete!")
    print("Ready for real-time model training and deployment.")

if __name__ == "__main__":
    main()