#!/usr/bin/env python3
"""
Realistic preprocessing for credit card fraud detection.
This version simulates proper fraud labels and avoids data leakage.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, LabelEncoder
from collections import defaultdict
import pickle
import os
from tqdm import tqdm
import hashlib
import argparse


def hash_sensitive_data(value, salt="rgtan_salt"):
    """Hash sensitive information for privacy protection"""
    if pd.isna(value) or value == '':
        return 'NULL'
    return hashlib.sha256(f"{value}{salt}".encode()).hexdigest()[:16]


def simulate_fraud_labels(df):
    """
    Simulate realistic fraud labels based on transaction characteristics.
    In real world, these would come from chargebacks/investigations.
    """
    print("Creating realistic fraud labels...")
    
    # Initialize all as legitimate
    df['is_fraud'] = 0
    
    # High-risk patterns that might indicate fraud (but not using auth_msg!)
    fraud_probability = np.zeros(len(df))
    
    # 1. Unusual amounts (very high or specific patterns)
    amount_mean = df['amount'].mean()
    amount_std = df['amount'].std()
    unusual_amount = (df['amount'] > amount_mean + 3*amount_std) | \
                    (df['amount'].astype(str).str.endswith('.99'))
    fraud_probability += unusual_amount.astype(float) * 0.1
    
    # 2. Time patterns (late night, early morning)
    if 'issue_date' in df.columns:
        df['hour'] = pd.to_datetime(df['issue_date']).dt.hour
        unusual_time = (df['hour'] < 6) | (df['hour'] > 23)
        fraud_probability += unusual_time.astype(float) * 0.05
    
    # 3. Velocity - multiple transactions in short time (will be computed per card later)
    # This is a placeholder - real velocity needs grouped computation
    
    # 4. Foreign transactions (different country from usual)
    if 'ISSUERCOUNTRY' in df.columns and 'bill_country' in df.columns:
        foreign_transaction = df['ISSUERCOUNTRY'] != df['bill_country']
        fraud_probability += foreign_transaction.astype(float) * 0.1
    
    # 5. Declined transactions have higher fraud probability (but not 1:1 mapping!)
    # Only use this as a weak signal, not deterministic
    if 'auth_msg' in df.columns:
        # Certain decline types are more associated with fraud attempts
        high_risk_declines = df['auth_msg'].str.contains(
            'STOLEN|LOST|FRAUD|SECURITY|PICKUP', case=False, na=False
        )
        medium_risk_declines = df['auth_msg'].str.contains(
            'DECLINE|CALL|INVALID', case=False, na=False
        ) & ~high_risk_declines
        
        fraud_probability += high_risk_declines.astype(float) * 0.3
        fraud_probability += medium_risk_declines.astype(float) * 0.1
        
        # APPROVED transactions can still be fraud! (authorized fraud)
        approved = df['auth_msg'].str.contains('APPROVED', case=False, na=False)
        # About 2% of approved transactions are later found to be fraud
        fraud_probability[approved] += 0.02
    
    # 6. Add some randomness to simulate investigation outcomes
    fraud_probability += np.random.random(len(df)) * 0.05
    
    # 7. Cap probability at 1.0
    fraud_probability = np.minimum(fraud_probability, 1.0)
    
    # 8. Sample fraud labels based on probability
    # Aim for realistic fraud rate of 1-2%
    df['fraud_score'] = fraud_probability
    
    # Adjust threshold to get ~1.5% fraud rate
    threshold = np.percentile(fraud_probability, 98.5)
    df['is_fraud'] = (fraud_probability > threshold).astype(int)
    
    # Add some guaranteed frauds for very high risk
    df.loc[fraud_probability > 0.7, 'is_fraud'] = 1
    
    # Add some random fraud to approved transactions (authorized fraud)
    approved_mask = df['auth_msg'].str.contains('APPROVED', case=False, na=False)
    n_approved = approved_mask.sum()
    n_approved_fraud = int(n_approved * 0.005)  # 0.5% of approved are fraud
    if n_approved_fraud > 0:
        approved_indices = df[approved_mask].index
        fraud_indices = np.random.choice(approved_indices, n_approved_fraud, replace=False)
        df.loc[fraud_indices, 'is_fraud'] = 1
    
    print(f"Fraud distribution:")
    print(f"  Total transactions: {len(df)}")
    print(f"  Fraudulent: {df['is_fraud'].sum()} ({df['is_fraud'].mean()*100:.2f}%)")
    print(f"  Legitimate: {(~df['is_fraud']).sum()} ({(~df['is_fraud']).mean()*100:.2f}%)")
    
    return df


def extract_time_features(df, cutoff_date=None):
    """Extract temporal features ensuring no future leakage"""
    print("Extracting temporal features...")
    
    date_columns = ['issue_date', 'capture_date', 'created_date', 'updated_date']
    
    for col in date_columns:
        if col in df.columns:
            # Convert to datetime
            df[col] = pd.to_datetime(df[col], errors='coerce')
            
            # Don't use data after cutoff
            if cutoff_date and col == 'issue_date':
                df = df[df[col] <= cutoff_date]
            
            if not df[col].isna().all():
                # Extract various time features
                df[col + '_hour'] = df[col].dt.hour
                df[col + '_dayofweek'] = df[col].dt.dayofweek
                df[col + '_day'] = df[col].dt.day
                df[col + '_month'] = df[col].dt.month
                
                # Time since first transaction
                min_date = df[col].min()
                df[col + '_seconds'] = (df[col] - min_date).dt.total_seconds()
                df[col + '_hours'] = df[col + '_seconds'] / 3600
                df[col + '_days'] = df[col + '_seconds'] / 86400
    
    return df


def create_aggregated_features_no_leakage(df, train_mask):
    """Create aggregated features using only training data to avoid leakage"""
    print("Creating aggregated features (no leakage)...")
    
    agg_features = pd.DataFrame(index=df.index)
    
    # Sort by time for proper historical aggregation
    df = df.sort_values('issue_date')
    
    # Key columns for aggregation
    key_columns = ['card_number', 'member_id', 'customer_ip', 'customer_email', 'BIN']
    
    for key_col in key_columns:
        if key_col not in df.columns:
            continue
            
        print(f"  Processing {key_col}...")
        
        # For each transaction, only use PAST training data
        for idx in tqdm(df.index, desc=f"Computing {key_col} features"):
            # Get transaction date
            trans_date = df.loc[idx, 'issue_date']
            
            # Find all past transactions with same key that are in training set
            same_key_mask = (df[key_col] == df.loc[idx, key_col])
            past_mask = (df['issue_date'] < trans_date)
            train_data_mask = train_mask  # Only use training data
            
            valid_mask = same_key_mask & past_mask & train_data_mask
            past_transactions = df[valid_mask]
            
            # Compute features from past transactions only
            if len(past_transactions) > 0:
                agg_features.loc[idx, f'{key_col}_prev_count'] = len(past_transactions)
                agg_features.loc[idx, f'{key_col}_prev_fraud_rate'] = past_transactions['is_fraud'].mean()
                agg_features.loc[idx, f'{key_col}_prev_avg_amount'] = past_transactions['amount'].mean()
                agg_features.loc[idx, f'{key_col}_prev_std_amount'] = past_transactions['amount'].std()
                
                # Days since last transaction
                last_trans_date = past_transactions['issue_date'].max()
                days_since = (trans_date - last_trans_date).total_seconds() / 86400
                agg_features.loc[idx, f'{key_col}_days_since_last'] = days_since
            else:
                # First transaction for this entity
                agg_features.loc[idx, f'{key_col}_prev_count'] = 0
                agg_features.loc[idx, f'{key_col}_prev_fraud_rate'] = 0
                agg_features.loc[idx, f'{key_col}_prev_avg_amount'] = 0
                agg_features.loc[idx, f'{key_col}_prev_std_amount'] = 0
                agg_features.loc[idx, f'{key_col}_days_since_last'] = -1
    
    return agg_features.fillna(0)


def create_graph_adjacency_lists_no_leakage(df, train_mask, min_edge_freq=2, max_edges_per_entity=100):
    """Create adjacency lists using only training data connections"""
    print("Creating graph adjacency lists (no leakage)...")
    
    adjacency_lists = defaultdict(set)
    edge_counts = defaultdict(int)
    
    key_columns = ['card_number', 'member_id', 'customer_ip', 'customer_email', 'BIN']
    
    # Only create edges between training nodes
    train_indices = df[train_mask].index.tolist()
    train_set = set(train_indices)
    
    for col in key_columns:
        if col not in df.columns:
            continue
            
        print(f"  Processing edges for {col}...")
        
        # Group transactions by key
        groups = defaultdict(list)
        for idx in train_indices:  # Only use training data
            val = df.loc[idx, col]
            if pd.notna(val) and val != '':
                groups[val].append(idx)
        
        # Create edges between transactions with same key
        for val, indices in groups.items():
            if len(indices) >= min_edge_freq:
                # Limit edges for memory efficiency
                if len(indices) > max_edges_per_entity:
                    indices = np.random.choice(indices, max_edges_per_entity, replace=False).tolist()
                
                # Create edges (simplified for efficiency)
                for i in range(len(indices)):
                    for j in range(i + 1, min(i + 10, len(indices))):  # Limit connections
                        if indices[i] in train_set and indices[j] in train_set:
                            adjacency_lists[indices[i]].add(indices[j])
                            adjacency_lists[indices[j]].add(indices[i])
                            edge_counts[col] += 2
    
    # Add edges for test nodes (they can connect to train nodes but not each other)
    test_indices = df[~train_mask].index.tolist()
    for col in key_columns:
        if col not in df.columns:
            continue
            
        for test_idx in test_indices:
            val = df.loc[test_idx, col]
            if pd.notna(val) and val != '':
                # Find training nodes with same value
                train_matches = [idx for idx in train_indices 
                               if df.loc[idx, col] == val]
                
                # Connect to up to 5 training nodes
                for train_idx in train_matches[:5]:
                    adjacency_lists[test_idx].add(train_idx)
                    adjacency_lists[train_idx].add(test_idx)
    
    print(f"  Total edges created: {sum(len(adj) for adj in adjacency_lists.values())}")
    
    return dict(adjacency_lists)


def preprocess_creditcard_realistic(input_file='data/vod_creditcard.csv', 
                                  output_dir='data/',
                                  test_size=0.3,
                                  min_edge_freq=2,
                                  max_edges_per_entity=100,
                                  privacy_mode=True):
    """
    Main preprocessing function with realistic fraud labels and no leakage.
    """
    
    print(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} transactions")
    
    # 1. Create realistic fraud labels
    df = simulate_fraud_labels(df)
    df['Labels'] = df['is_fraud']  # For compatibility
    
    # 2. Hash sensitive data if privacy mode is on
    if privacy_mode:
        print("Hashing sensitive data for privacy...")
        sensitive_cols = ['card_number', 'customer_email', 'customer_ip', 
                         'bill_name1', 'bill_name2', 'ship_name1', 'ship_name2',
                         'customer_phone', 'bill_street', 'ship_street']
        
        for col in sensitive_cols:
            if col in df.columns:
                df[col] = df[col].apply(hash_sensitive_data)
    
    # 3. Time-based train/test split
    print("Creating time-based train/test split...")
    df = df.sort_values('issue_date')
    split_idx = int(len(df) * (1 - test_size))
    train_mask = pd.Series([True] * split_idx + [False] * (len(df) - split_idx), index=df.index)
    
    print(f"Train set: {train_mask.sum()} transactions (up to {df.iloc[split_idx-1]['issue_date']})")
    print(f"Test set: {(~train_mask).sum()} transactions (after {df.iloc[split_idx]['issue_date']})")
    print(f"Train fraud rate: {df[train_mask]['Labels'].mean():.2%}")
    print(f"Test fraud rate: {df[~train_mask]['Labels'].mean():.2%}")
    
    # 4. Extract temporal features
    df = extract_time_features(df)
    
    # 5. Create aggregated features (using only past training data)
    agg_features = create_aggregated_features_no_leakage(df, train_mask)
    
    # 6. Create adjacency lists (no test-test connections)
    adjacency_lists = create_graph_adjacency_lists_no_leakage(df, train_mask, min_edge_freq, max_edges_per_entity)
    
    # 7. Create neighborhood features using only training graph
    print("Creating neighborhood features...")
    neigh_features = pd.DataFrame(index=df.index)
    
    # Simple neighborhood features
    neigh_features['degree'] = [len(adjacency_lists.get(i, [])) for i in df.index]
    
    # Fraud rate in neighborhood (only from training data)
    for idx in df.index:
        neighbors = list(adjacency_lists.get(idx, []))
        train_neighbors = [n for n in neighbors if train_mask[n]]
        
        if train_neighbors:
            neigh_features.loc[idx, 'neigh_fraud_rate'] = df.loc[train_neighbors, 'Labels'].mean()
            neigh_features.loc[idx, 'neigh_avg_amount'] = df.loc[train_neighbors, 'amount'].mean()
        else:
            neigh_features.loc[idx, 'neigh_fraud_rate'] = 0
            neigh_features.loc[idx, 'neigh_avg_amount'] = 0
    
    # 8. Select and scale numerical features
    print("Scaling numerical features...")
    numerical_cols = ['amount', 'issue_date_seconds', 'issue_date_hours', 
                     'issue_date_hour', 'issue_date_dayofweek'] + list(agg_features.columns)
    
    # Add aggregated features to main dataframe
    for col in agg_features.columns:
        df[col] = agg_features[col]
    
    # Scale using only training data statistics
    scaler = StandardScaler()
    for col in numerical_cols:
        if col in df.columns:
            # Fit on training data only
            scaler.fit(df.loc[train_mask, [col]].fillna(0))
            # Transform all data
            df[col + '_scaled'] = scaler.transform(df[[col]].fillna(0))
    
    # 9. Encode categorical features
    print("Encoding categorical features...")
    categorical_cols = ['trans_status_msg_id', 'site_tag_id', 'origin_id', 
                       'currency_id', 'card_type_id', 'processor_id', 
                       'trans_status_code', 'BRAND', 'DEBITCREDIT', 'CARDTYPE']
    
    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            # Handle missing values
            df[col] = df[col].fillna('missing')
            # Fit on training data only
            train_values = df.loc[train_mask, col].astype(str)
            le.fit(train_values)
            
            # Transform all data (unknown categories become 0)
            df[col + '_encoded'] = df[col].apply(
                lambda x: le.transform([str(x)])[0] if str(x) in le.classes_ else 0
            )
    
    # 10. Save preprocessed data
    print("Saving preprocessed data...")
    
    # Save main dataframe with features
    output_file = os.path.join(output_dir, 'creditcard_realistic_preprocessed.csv')
    df.to_csv(output_file, index=False)
    print(f"Saved preprocessed data to {output_file}")
    
    # Save adjacency lists
    adj_file = os.path.join(output_dir, 'creditcard_realistic_adjlists.pickle')
    with open(adj_file, 'wb') as f:
        pickle.dump(adjacency_lists, f)
    print(f"Saved adjacency lists to {adj_file}")
    
    # Save neighborhood features
    neigh_file = os.path.join(output_dir, 'creditcard_realistic_neigh_feat.csv')
    neigh_features.to_csv(neigh_file, index=False)
    print(f"Saved neighborhood features to {neigh_file}")
    
    # Save train/test masks
    split_file = os.path.join(output_dir, 'creditcard_realistic_splits.pickle')
    with open(split_file, 'wb') as f:
        pickle.dump({'train_mask': train_mask, 'split_idx': split_idx}, f)
    print(f"Saved train/test splits to {split_file}")
    
    # Save preprocessing metadata
    metadata = {
        'n_transactions': len(df),
        'n_edges': sum(len(adj) for adj in adjacency_lists.values()),
        'fraud_rate': df['Labels'].mean(),
        'train_fraud_rate': df[train_mask]['Labels'].mean(),
        'test_fraud_rate': df[~train_mask]['Labels'].mean(),
        'numerical_features': numerical_cols,
        'preprocessing_date': datetime.now().isoformat(),
        'privacy_mode': privacy_mode,
        'min_edge_freq': min_edge_freq,
        'realistic_labels': True
    }
    
    metadata_file = os.path.join(output_dir, 'creditcard_realistic_metadata.pickle')
    with open(metadata_file, 'wb') as f:
        pickle.dump(metadata, f)
    print(f"Saved metadata to {metadata_file}")
    
    print("\nRealistic preprocessing complete!")
    print(f"Total transactions: {len(df)}")
    print(f"Total edges: {metadata['n_edges']}")
    print(f"Overall fraud rate: {metadata['fraud_rate']:.2%}")
    print(f"Train fraud rate: {metadata['train_fraud_rate']:.2%}")
    print(f"Test fraud rate: {metadata['test_fraud_rate']:.2%}")
    
    return df, adjacency_lists, neigh_features


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Realistic preprocessing for credit card fraud dataset')
    parser.add_argument('--input', default='data/vod_creditcard.csv', help='Input CSV file')
    parser.add_argument('--output', default='data/', help='Output directory')
    parser.add_argument('--test-size', type=float, default=0.3, help='Test set size')
    parser.add_argument('--min-edge-freq', type=int, default=2, help='Minimum edge frequency')
    parser.add_argument('--max-edges-per-entity', type=int, default=100, 
                       help='Maximum edges per entity to prevent memory issues')
    parser.add_argument('--no-privacy', action='store_true', help='Disable privacy mode (no hashing)')
    
    args = parser.parse_args()
    
    preprocess_creditcard_realistic(
        input_file=args.input,
        output_dir=args.output,
        test_size=args.test_size,
        min_edge_freq=args.min_edge_freq,
        max_edges_per_entity=args.max_edges_per_entity,
        privacy_mode=not args.no_privacy
    )