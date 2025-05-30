#!/usr/bin/env python3
"""
Convert credit card dataset to Amazon.mat format for RGTAN.

This creates a graph where:
- Nodes are transactions
- Edges connect transactions that share attributes (card, IP, email, etc.)
- Features are transaction attributes
- Labels indicate fraud (1) or normal (0)
"""

import argparse
import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.sparse as sp
from sklearn.preprocessing import StandardScaler, LabelEncoder
from collections import defaultdict
import hashlib
from datetime import datetime

def create_fraud_labels_realistic(df):
    """Create realistic fraud labels based on transaction patterns."""
    fraud_mask = np.zeros(len(df), dtype=bool)
    
    # High-risk patterns (without using auth_msg)
    fraud_mask |= (df['amount'] > df['amount'].quantile(0.99))  # Very high amounts
    fraud_mask |= (df['bill_zip'].isna() | (df['bill_zip'] == '')) & (df['amount'] > df['amount'].quantile(0.9))  # Missing zip + high amount
    
    # Add temporal anomalies if capture_date exists
    if 'capture_date' in df.columns and 'issue_date' in df.columns:
        df['issue_datetime'] = pd.to_datetime(df['issue_date'], errors='coerce')
        df['capture_datetime'] = pd.to_datetime(df['capture_date'], errors='coerce')
        time_diff = (df['capture_datetime'] - df['issue_datetime']).dt.total_seconds() / 3600  # hours
        fraud_mask |= (time_diff > 24) & (~time_diff.isna())  # Very delayed capture
    
    # Make fraud ~2% of transactions
    np.random.seed(42)
    fraud_indices = np.where(fraud_mask)[0]
    if len(fraud_indices) > int(0.02 * len(df)):
        fraud_indices = np.random.choice(fraud_indices, int(0.02 * len(df)), replace=False)
        fraud_mask = np.zeros(len(df), dtype=bool)
        fraud_mask[fraud_indices] = True
    
    return fraud_mask.astype(int)

def create_node_features(df):
    """Create feature matrix for nodes (transactions)."""
    # Select numerical features available in the dataset
    numerical_features = ['amount']
    
    # Add time-based features
    df['issue_datetime'] = pd.to_datetime(df['issue_date'], errors='coerce')
    if 'capture_date' in df.columns:
        df['capture_datetime'] = pd.to_datetime(df['capture_date'], errors='coerce')
        df['hours_to_capture'] = (df['capture_datetime'] - df['issue_datetime']).dt.total_seconds() / 3600
        df['hours_to_capture'] = df['hours_to_capture'].fillna(0).clip(0, 168)  # Cap at 1 week
        numerical_features.append('hours_to_capture')
    
    # Encode categorical features
    categorical_features = ['trans_status_msg_id', 'site_tag_id', 'origin_id', 
                          'currency_id', 'card_type_id', 'processor_id',
                          'BRAND', 'DEBITCREDIT', 'CARDTYPE', 'ISSUERCOUNTRY']
    
    features_list = []
    
    # Add numerical features
    scaler = StandardScaler()
    num_data = df[numerical_features].fillna(0)
    num_feats = scaler.fit_transform(num_data)
    features_list.append(num_feats)
    
    # Add categorical features (label encoding with normalization)
    for cat_feat in categorical_features:
        if cat_feat in df.columns:
            # Use label encoding to reduce dimensionality
            le = LabelEncoder()
            # Handle missing values
            cat_data = df[cat_feat].fillna('missing').astype(str)
            encoded = le.fit_transform(cat_data)
            # Normalize to [0, 1]
            if len(le.classes_) > 1:
                encoded = encoded / (len(le.classes_) - 1)
            features_list.append(encoded.reshape(-1, 1))
    
    # Add temporal features
    df['hour'] = df['issue_datetime'].dt.hour
    df['weekday'] = df['issue_datetime'].dt.weekday
    temporal_feats = np.column_stack([
        df['hour'].fillna(0) / 23,  # Normalize hour
        df['weekday'].fillna(0) / 6  # Normalize weekday
    ])
    features_list.append(temporal_feats)
    
    # Combine all features
    features = np.hstack(features_list)
    
    return sp.csr_matrix(features)

def create_adjacency_matrices(df, max_edges_per_entity=100):
    """Create different types of adjacency matrices based on shared attributes."""
    n = len(df)
    
    # Hash entities for memory efficiency
    def hash_entity(value):
        if pd.isna(value) or value == '' or value == 'missing':
            return None
        return hashlib.md5(str(value).encode()).hexdigest()[:8]
    
    # Build entity to node mappings
    entity_mappings = {
        'card': defaultdict(list),
        'ip': defaultdict(list),
        'email': defaultdict(list),
        'member': defaultdict(list)
    }
    
    print("Building entity mappings...")
    for idx, row in df.iterrows():
        node_id = idx
        
        # Card-based connections
        card_hash = hash_entity(row.get('card_number', None))
        if card_hash:
            entity_mappings['card'][card_hash].append(node_id)
        
        # IP-based connections
        ip_hash = hash_entity(row.get('customer_ip', None))
        if ip_hash:
            entity_mappings['ip'][ip_hash].append(node_id)
        
        # Email-based connections
        email_hash = hash_entity(row.get('customer_email', None))
        if email_hash:
            entity_mappings['email'][email_hash].append(node_id)
        
        # Member-based connections (instead of merchant)
        member_hash = hash_entity(row.get('member_id', None))
        if member_hash:
            entity_mappings['member'][member_hash].append(node_id)
    
    # Create adjacency matrices
    adjacency_matrices = {}
    
    for entity_type, entity_dict in entity_mappings.items():
        print(f"Creating {entity_type} adjacency matrix...")
        rows, cols = [], []
        
        for entity_hash, node_list in entity_dict.items():
            if len(node_list) < 2:
                continue
            
            # Limit edges per entity
            if len(node_list) > max_edges_per_entity:
                # Sample nodes to connect
                sampled_nodes = np.random.choice(node_list, max_edges_per_entity, replace=False)
                node_list = sampled_nodes.tolist()
            
            # Create edges between all nodes sharing this entity
            for i in range(len(node_list)):
                for j in range(i + 1, len(node_list)):
                    rows.extend([node_list[i], node_list[j]])
                    cols.extend([node_list[j], node_list[i]])
        
        # Create sparse matrix
        if rows:
            data = np.ones(len(rows))
            adj_matrix = sp.csr_matrix((data, (rows, cols)), shape=(n, n))
        else:
            adj_matrix = sp.csr_matrix((n, n))
        
        adjacency_matrices[entity_type] = adj_matrix
    
    # Create homogeneous graph (union of all relationships)
    print("Creating homogeneous graph...")
    homo = adjacency_matrices['card'] + adjacency_matrices['ip'] + \
           adjacency_matrices['email'] + adjacency_matrices['member']
    homo = (homo > 0).astype(float)  # Binary adjacency
    
    # Add self-loops
    homo += sp.eye(n)
    
    return adjacency_matrices, homo

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='data/vod_creditcard.csv', help='Input CSV file')
    parser.add_argument('--output', default='data/CreditCard.mat', help='Output MAT file')
    parser.add_argument('--max-rows', type=int, default=None, help='Limit number of rows to process')
    parser.add_argument('--max-edges-per-entity', type=int, default=100, 
                        help='Maximum edges per entity')
    args = parser.parse_args()
    
    print(f"Loading data from {args.input}...")
    df = pd.read_csv(args.input)
    
    if args.max_rows:
        df = df.head(args.max_rows)
        print(f"Limited to {len(df)} rows")
    
    # Create fraud labels
    print("Creating fraud labels...")
    labels = create_fraud_labels_realistic(df)
    print(f"Fraud rate: {labels.mean():.2%} ({labels.sum()} fraud / {len(labels)} total)")
    
    # Create node features
    print("Creating node features...")
    features = create_node_features(df)
    print(f"Feature shape: {features.shape}")
    
    # Create adjacency matrices
    adjacency_matrices, homo = create_adjacency_matrices(df, args.max_edges_per_entity)
    
    # Print statistics
    print("\nGraph statistics:")
    print(f"- Nodes: {homo.shape[0]}")
    print(f"- Edges (homo): {(homo.nnz - homo.shape[0]) // 2}")  # Subtract self-loops, divide by 2 for undirected
    for entity_type, adj in adjacency_matrices.items():
        print(f"- Edges ({entity_type}): {adj.nnz // 2}")
    
    # Save to MAT file in Amazon format
    print(f"\nSaving to {args.output}...")
    mat_data = {
        'label': labels,
        'features': features,
        'net_upu': adjacency_matrices['card'],    # Card-based connections
        'net_usu': adjacency_matrices['ip'],       # IP-based connections  
        'net_uvu': adjacency_matrices['email'],    # Email-based connections
        'homo': homo
    }
    
    sio.savemat(args.output, mat_data)
    print("Done!")

if __name__ == '__main__':
    main()