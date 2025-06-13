#!/usr/bin/env python3
"""
Fast graph feature generation for large datasets
Uses sampling and optimization to handle millions of transactions
"""

import os
import sys
import pandas as pd
import numpy as np
import dgl
import torch
from datetime import datetime, timedelta
import pickle
from collections import defaultdict
import random
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def sample_large_dataset(df, max_size=100000):
    """Sample dataset if too large"""
    if len(df) <= max_size:
        return df
    
    print(f"Dataset too large ({len(df):,} transactions), sampling {max_size:,} transactions...")
    
    # Ensure we keep all fraud transactions
    fraud_df = df[df['is_fraud'] == 1]
    normal_df = df[df['is_fraud'] == 0]
    
    # Sample normal transactions
    sample_size = max_size - len(fraud_df)
    if sample_size > 0 and len(normal_df) > sample_size:
        normal_df = normal_df.sample(n=sample_size, random_state=42)
    
    # Combine and shuffle
    sampled_df = pd.concat([fraud_df, normal_df], ignore_index=True)
    sampled_df = sampled_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Reset transaction IDs
    sampled_df['trans_id'] = range(len(sampled_df))
    
    print(f"Sampled dataset: {len(sampled_df):,} transactions ({sampled_df['is_fraud'].mean():.2%} fraud)")
    
    return sampled_df

def build_lightweight_graph(df, max_edges=5000000):
    """Build a lightweight graph for large datasets"""
    print(f"\nBuilding lightweight graph for {len(df):,} transactions...")
    
    # Initialize edge lists
    edges = []
    edge_count = 0
    
    # Group by entities
    entity_groups = {
        'card': df.groupby('entity_card')['trans_id'].apply(list).to_dict(),
        'email': df.groupby('entity_email')['trans_id'].apply(list).to_dict(),
        'ip': df.groupby('entity_ip')['trans_id'].apply(list).to_dict()
    }
    
    # Create edges with strict limits
    for entity_type, groups in entity_groups.items():
        print(f"\nProcessing {entity_type} entities...")
        entity_count = 0
        
        for entity, trans_ids in groups.items():
            if len(trans_ids) < 2 or edge_count >= max_edges:
                continue
            
            entity_count += 1
            if entity_count % 10000 == 0:
                print(f"  Processed {entity_count:,} entities, {edge_count:,} edges")
            
            # Skip very large entities
            if len(trans_ids) > 100:
                trans_ids = random.sample(trans_ids, 100)
            
            # Simple edge creation - connect sequential transactions
            for i in range(1, min(len(trans_ids), 5)):  # Max 4 edges per entity
                if edge_count >= max_edges:
                    break
                edges.append((trans_ids[i-1], trans_ids[i]))
                edge_count += 1
    
    print(f"\nTotal edges created: {edge_count:,}")
    
    # Create DGL graph
    if edges:
        src, dst = zip(*edges)
        g = dgl.graph((src, dst), num_nodes=len(df))
        # Make bidirectional
        g = dgl.to_bidirected(g)
    else:
        g = dgl.graph(([], []), num_nodes=len(df))
    
    print(f"Graph created with {g.num_nodes():,} nodes and {g.num_edges():,} edges")
    
    return g

def compute_simple_features(g, df):
    """Compute simple neighborhood features efficiently"""
    print("\nComputing simple neighborhood features...")
    
    num_nodes = g.num_nodes()
    
    # Initialize features
    features = {
        'neighbor_count': np.zeros(num_nodes),
        'neighbor_fraud_rate': np.zeros(num_nodes),
        'neighbor_avg_amount': np.zeros(num_nodes)
    }
    
    # Node attributes
    fraud_labels = df['is_fraud'].values
    amounts = df['amount'].values
    
    # Compute features in batches
    batch_size = 50000
    for start in range(0, num_nodes, batch_size):
        end = min(start + batch_size, num_nodes)
        if start % 100000 == 0:
            print(f"  Processing nodes {start:,} to {end:,}...")
        
        for node in range(start, end):
            # Get neighbors
            neighbors = g.in_edges(node)[0].numpy()
            neighbors = neighbors[neighbors != node]  # Exclude self
            
            if len(neighbors) > 0:
                features['neighbor_count'][node] = len(neighbors)
                features['neighbor_fraud_rate'][node] = fraud_labels[neighbors].mean()
                features['neighbor_avg_amount'][node] = amounts[neighbors].mean()
    
    # Create DataFrame
    feature_df = pd.DataFrame(features)
    feature_df['trans_id'] = df['trans_id'].values
    
    print("Feature computation complete!")
    
    return feature_df

def main():
    # Find the fraud_prevention directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    fraud_dir = os.path.dirname(script_dir)
    
    # Paths
    data_path = os.path.join(fraud_dir, 'data', 'processed')
    output_path = os.path.join(fraud_dir, 'data', 'graph')
    
    # Load processed data
    print("Loading processed data...")
    train_df = pd.read_csv(os.path.join(data_path, 'train_transactions.csv'))
    test_df = pd.read_csv(os.path.join(data_path, 'test_transactions.csv'))
    
    print(f"Train size: {len(train_df):,}, Test size: {len(test_df):,}")
    
    # Sample if datasets are too large
    max_train_size = 200000  # Limit training data
    max_test_size = 50000    # Limit test data
    
    train_df = sample_large_dataset(train_df, max_train_size)
    test_df = sample_large_dataset(test_df, max_test_size)
    
    # Combine for graph building
    combined_df = pd.concat([train_df, test_df], ignore_index=True)
    combined_df['trans_id'] = range(len(combined_df))
    combined_df['is_train'] = [1] * len(train_df) + [0] * len(test_df)
    
    # Build lightweight graph
    g = build_lightweight_graph(combined_df)
    
    # Compute simple features
    features_df = compute_simple_features(g, combined_df)
    
    # Split back into train/test
    train_features = features_df.iloc[:len(train_df)]
    test_features = features_df.iloc[len(train_df):]
    
    # Save everything
    os.makedirs(output_path, exist_ok=True)
    
    print("\nSaving graph data...")
    dgl.save_graphs(os.path.join(output_path, 'transaction_graph.dgl'), [g])
    train_features.to_csv(os.path.join(output_path, 'train_neigh_features.csv'), index=False)
    test_features.to_csv(os.path.join(output_path, 'test_neigh_features.csv'), index=False)
    combined_df.to_csv(os.path.join(output_path, 'combined_transactions.csv'), index=False)
    
    print(f"Graph data saved to {output_path}")
    print("\nGraph generation complete!")

if __name__ == "__main__":
    main()