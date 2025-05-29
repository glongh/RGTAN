#!/usr/bin/env python3
"""
Preprocessing script for credit card fraud dataset.
This script handles all preprocessing steps needed before feeding data to RGTAN.
"""

import pandas as pd
import numpy as np
from datetime import datetime
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


def extract_time_features(df):
    """Extract temporal features from datetime columns"""
    print("Extracting temporal features...")
    
    date_columns = ['issue_date', 'capture_date', 'created_date', 'updated_date']
    
    for col in date_columns:
        if col in df.columns:
            # Convert to datetime
            df[col] = pd.to_datetime(df[col], errors='coerce')
            
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
    
    # Transaction velocity features
    if 'issue_date' in df.columns:
        df = df.sort_values('issue_date')
        
        # Time between transactions for same card
        for key_col in ['card_number', 'member_id', 'customer_email']:
            if key_col in df.columns:
                df[f'time_since_last_{key_col}'] = df.groupby(key_col)['issue_date'].diff().dt.total_seconds()
                df[f'time_since_last_{key_col}'] = df[f'time_since_last_{key_col}'].fillna(0)
    
    return df


def create_aggregated_features(df):
    """Create aggregated risk features based on historical data"""
    print("Creating aggregated features...")
    
    agg_features = pd.DataFrame(index=df.index)
    
    # Sort by time for proper historical aggregation
    df = df.sort_values('issue_date')
    
    # Key columns for aggregation
    key_columns = ['card_number', 'member_id', 'customer_ip', 'customer_email', 'BIN']
    
    for key_col in key_columns:
        if key_col not in df.columns:
            continue
            
        print(f"  Processing {key_col}...")
        
        # Count of previous transactions
        agg_features[f'{key_col}_prev_count'] = df.groupby(key_col).cumcount()
        
        # Running fraud rate (excluding current transaction)
        def running_fraud_rate(group):
            fraud_cumsum = group.shift(1).cumsum().fillna(0)
            count_cumsum = pd.Series(range(len(group)), index=group.index)
            return (fraud_cumsum / count_cumsum.replace(0, 1)).fillna(0)
        
        agg_features[f'{key_col}_prev_fraud_rate'] = df.groupby(key_col)['Labels'].transform(running_fraud_rate)
        
        # Amount statistics
        agg_features[f'{key_col}_prev_avg_amount'] = df.groupby(key_col)['amount'].transform(
            lambda x: x.expanding().mean().shift(1).fillna(0)
        )
        agg_features[f'{key_col}_prev_std_amount'] = df.groupby(key_col)['amount'].transform(
            lambda x: x.expanding().std().shift(1).fillna(0)
        )
        
        # Transaction frequency (transactions per day)
        if 'issue_date_days' in df.columns:
            agg_features[f'{key_col}_tx_frequency'] = (
                agg_features[f'{key_col}_prev_count'] / 
                (df['issue_date_days'] + 1)  # +1 to avoid division by zero
            )
    
    return agg_features


def create_graph_adjacency_lists(df, min_edge_freq=2, max_edges_per_entity=100):
    """Create adjacency lists for graph construction with memory optimization"""
    print("Creating graph adjacency lists...")
    
    adjacency_lists = defaultdict(set)
    edge_counts = defaultdict(int)
    entity_counts = defaultdict(int)
    
    key_columns = ['card_number', 'member_id', 'customer_ip', 'customer_email', 'BIN']
    
    for col in key_columns:
        if col not in df.columns:
            continue
            
        print(f"  Processing edges for {col}...")
        
        # Group transactions by key
        groups = defaultdict(list)
        for idx, val in enumerate(df[col]):
            if pd.notna(val) and val != '':
                groups[val].append(idx)
        
        # Sort groups by size to identify potential issues
        large_groups = [(val, len(indices)) for val, indices in groups.items() if len(indices) > 50]
        if large_groups:
            print(f"    Warning: Found {len(large_groups)} entities with >50 transactions")
            largest = max(large_groups, key=lambda x: x[1])
            print(f"    Largest group has {largest[1]} transactions")
        
        # Create edges between transactions with same key
        for val, indices in groups.items():
            if len(indices) >= min_edge_freq:  # Only create edges if frequency meets threshold
                entity_counts[col] += 1
                
                # For very large groups, sample edges to avoid memory explosion
                if len(indices) > max_edges_per_entity:
                    print(f"    Limiting edges for {col} value with {len(indices)} transactions")
                    # Sample a subset of indices
                    sampled_indices = np.random.choice(indices, max_edges_per_entity, replace=False)
                    indices = sampled_indices.tolist()
                
                # Create edges more efficiently
                # Instead of all permutations, create a connected component
                if len(indices) <= 10:
                    # For small groups, create all edges
                    for i in range(len(indices)):
                        for j in range(i + 1, len(indices)):
                            adjacency_lists[indices[i]].add(indices[j])
                            adjacency_lists[indices[j]].add(indices[i])
                            edge_counts[col] += 2
                else:
                    # For larger groups, create a hub-and-spoke pattern to reduce edges
                    # This maintains connectivity while reducing memory usage
                    hub = indices[0]
                    for i in range(1, len(indices)):
                        adjacency_lists[hub].add(indices[i])
                        adjacency_lists[indices[i]].add(hub)
                        edge_counts[col] += 2
                    
                    # Add some additional random edges for better connectivity
                    num_extra_edges = min(len(indices) - 1, 20)
                    for _ in range(num_extra_edges):
                        i, j = np.random.choice(len(indices), 2, replace=False)
                        if i != j:
                            adjacency_lists[indices[i]].add(indices[j])
                            adjacency_lists[indices[j]].add(indices[i])
                            edge_counts[col] += 2
    
    print(f"  Total edges created: {sum(len(adj) for adj in adjacency_lists.values())}")
    print("  Edge statistics by column:")
    for col, count in edge_counts.items():
        print(f"    {col}: {count} edges from {entity_counts[col]} unique entities")
    
    return dict(adjacency_lists)


def create_neighborhood_features(df, adjacency_lists):
    """Create neighborhood risk statistics"""
    print("Creating neighborhood features...")
    
    neigh_features = pd.DataFrame(index=df.index)
    
    # 1-hop neighborhood features
    neigh_features['1hop_count'] = [len(adjacency_lists.get(i, [])) for i in df.index]
    neigh_features['1hop_fraud_count'] = 0
    neigh_features['1hop_fraud_rate'] = 0.0
    neigh_features['1hop_avg_amount'] = 0.0
    neigh_features['1hop_std_amount'] = 0.0
    
    for idx in tqdm(df.index, desc="Computing 1-hop features"):
        neighbors = list(adjacency_lists.get(idx, []))
        if neighbors:
            neighbor_labels = df.loc[neighbors, 'Labels']
            neighbor_amounts = df.loc[neighbors, 'amount']
            
            neigh_features.loc[idx, '1hop_fraud_count'] = neighbor_labels.sum()
            neigh_features.loc[idx, '1hop_fraud_rate'] = neighbor_labels.mean()
            neigh_features.loc[idx, '1hop_avg_amount'] = neighbor_amounts.mean()
            neigh_features.loc[idx, '1hop_std_amount'] = neighbor_amounts.std() if len(neighbors) > 1 else 0
    
    # 2-hop neighborhood features (optional, can be expensive)
    compute_2hop = len(df) < 50000  # Only for smaller datasets
    
    if compute_2hop:
        print("Computing 2-hop features...")
        neigh_features['2hop_count'] = 0
        neigh_features['2hop_fraud_rate'] = 0.0
        
        for idx in tqdm(df.index, desc="Computing 2-hop features"):
            two_hop_neighbors = set()
            for neighbor in adjacency_lists.get(idx, []):
                two_hop_neighbors.update(adjacency_lists.get(neighbor, []))
            two_hop_neighbors.discard(idx)  # Remove self
            
            if two_hop_neighbors:
                neighbor_labels = df.loc[list(two_hop_neighbors), 'Labels']
                neigh_features.loc[idx, '2hop_count'] = len(two_hop_neighbors)
                neigh_features.loc[idx, '2hop_fraud_rate'] = neighbor_labels.mean()
    
    return neigh_features


def preprocess_creditcard_data(input_file='data/vod_creditcard.csv', 
                              output_dir='data/',
                              min_edge_freq=2,
                              max_edges_per_entity=100,
                              privacy_mode=True):
    """
    Main preprocessing function for credit card dataset.
    
    Args:
        input_file: Path to input CSV file
        output_dir: Directory to save preprocessed files
        min_edge_freq: Minimum frequency for edge creation
        max_edges_per_entity: Maximum edges per entity to prevent memory issues
        privacy_mode: Whether to hash sensitive information
    """
    
    print(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} transactions")
    
    # 1. Convert labels
    print("Converting labels...")
    if 'IS_TARGETED' in df.columns:
        # Check unique values
        unique_values = df['IS_TARGETED'].unique()
        print(f"Unique values in IS_TARGETED: {unique_values}")
        
        # Map string values to binary
        df['Labels'] = df['IS_TARGETED'].map({'yes': 1, 'no': 0, 'YES': 1, 'NO': 0})
        
        # Check for unmapped values
        if df['Labels'].isna().any():
            print(f"Warning: {df['Labels'].isna().sum()} unmapped label values found")
            print(f"Unmapped values: {df[df['Labels'].isna()]['IS_TARGETED'].unique()}")
            # Set unmapped to 0 (non-fraud)
            df['Labels'] = df['Labels'].fillna(0).astype(int)
    else:
        print("Error: IS_TARGETED column not found!")
        raise ValueError("IS_TARGETED column is required for fraud labels")
    
    print(f"Fraud rate: {df['Labels'].mean():.2%}")
    print(f"Fraud count: {df['Labels'].sum()} out of {len(df)} transactions")
    
    # 2. Hash sensitive data if privacy mode is on
    if privacy_mode:
        print("Hashing sensitive data for privacy...")
        sensitive_cols = ['card_number', 'customer_email', 'customer_ip', 
                         'bill_name1', 'bill_name2', 'ship_name1', 'ship_name2',
                         'customer_phone', 'bill_street', 'ship_street']
        
        for col in sensitive_cols:
            if col in df.columns:
                df[col] = df[col].apply(hash_sensitive_data)
    
    # 3. Extract temporal features
    df = extract_time_features(df)
    
    # 4. Create aggregated features
    agg_features = create_aggregated_features(df)
    
    # 5. Create adjacency lists
    adjacency_lists = create_graph_adjacency_lists(df, min_edge_freq, max_edges_per_entity)
    
    # 6. Create neighborhood features
    neigh_features = create_neighborhood_features(df, adjacency_lists)
    
    # 7. Select and scale numerical features
    print("Scaling numerical features...")
    numerical_cols = ['amount', 'issue_date_seconds', 'issue_date_hours', 'issue_date_days',
                     'issue_date_hour', 'issue_date_dayofweek'] + list(agg_features.columns)
    
    # Add aggregated features to main dataframe
    for col in agg_features.columns:
        df[col] = agg_features[col]
    
    # Scale numerical features
    scaler = StandardScaler()
    for col in numerical_cols:
        if col in df.columns:
            df[col + '_scaled'] = scaler.fit_transform(df[[col]].fillna(0))
    
    # 8. Save preprocessed data
    print("Saving preprocessed data...")
    
    # Save main dataframe with features
    output_file = os.path.join(output_dir, 'creditcard_preprocessed.csv')
    df.to_csv(output_file, index=False)
    print(f"Saved preprocessed data to {output_file}")
    
    # Save adjacency lists
    adj_file = os.path.join(output_dir, 'creditcard_homo_adjlists.pickle')
    with open(adj_file, 'wb') as f:
        pickle.dump(adjacency_lists, f)
    print(f"Saved adjacency lists to {adj_file}")
    
    # Save neighborhood features
    neigh_file = os.path.join(output_dir, 'creditcard_neigh_feat.csv')
    neigh_features.to_csv(neigh_file, index=False)
    print(f"Saved neighborhood features to {neigh_file}")
    
    # Save preprocessing metadata
    metadata = {
        'n_transactions': len(df),
        'n_edges': sum(len(adj) for adj in adjacency_lists.values()),
        'fraud_rate': df['Labels'].mean(),
        'numerical_features': numerical_cols,
        'preprocessing_date': datetime.now().isoformat(),
        'privacy_mode': privacy_mode,
        'min_edge_freq': min_edge_freq
    }
    
    metadata_file = os.path.join(output_dir, 'creditcard_preprocessing_metadata.pickle')
    with open(metadata_file, 'wb') as f:
        pickle.dump(metadata, f)
    print(f"Saved metadata to {metadata_file}")
    
    print("\nPreprocessing complete!")
    print(f"Total transactions: {len(df)}")
    print(f"Total edges: {metadata['n_edges']}")
    print(f"Fraud rate: {metadata['fraud_rate']:.2%}")
    
    return df, adjacency_lists, neigh_features


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess credit card fraud dataset for RGTAN')
    parser.add_argument('--input', default='data/vod_creditcard.csv', help='Input CSV file')
    parser.add_argument('--output', default='data/', help='Output directory')
    parser.add_argument('--min-edge-freq', type=int, default=2, help='Minimum edge frequency')
    parser.add_argument('--max-edges-per-entity', type=int, default=100, 
                       help='Maximum edges per entity to prevent memory issues')
    parser.add_argument('--no-privacy', action='store_true', help='Disable privacy mode (no hashing)')
    
    args = parser.parse_args()
    
    preprocess_creditcard_data(
        input_file=args.input,
        output_dir=args.output,
        min_edge_freq=args.min_edge_freq,
        max_edges_per_entity=args.max_edges_per_entity,
        privacy_mode=not args.no_privacy
    )