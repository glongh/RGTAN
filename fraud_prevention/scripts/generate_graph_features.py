#!/usr/bin/env python3
"""
Generate graph and neighborhood features for fraud detection
Builds transaction graph and computes risk statistics
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
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def build_transaction_graph(df, max_edges_per_entity=100, time_window_hours=24):
    """Build graph connecting transactions with shared entities"""
    print(f"\nBuilding transaction graph for {len(df)} transactions...")
    print(f"Max edges per entity: {max_edges_per_entity}")
    print(f"Time window: {time_window_hours} hours")
    
    # Initialize edge lists
    edge_src = []
    edge_dst = []
    edge_types = []
    
    # Entity to transaction mapping
    entity_maps = {
        'card': defaultdict(list),
        'email': defaultdict(list),
        'ip': defaultdict(list),
        'bin': defaultdict(list)
    }
    
    # Build entity mappings with progress
    print("Building entity mappings...")
    for idx, row in df.iterrows():
        if idx % 100000 == 0:
            print(f"  Processed {idx:,}/{len(df):,} transactions...")
        trans_id = row['trans_id']
        entity_maps['card'][row['entity_card']].append(trans_id)
        entity_maps['email'][row['entity_email']].append(trans_id)
        entity_maps['ip'][row['entity_ip']].append(trans_id)
        entity_maps['bin'][row['entity_bin']].append(trans_id)
    
    # Create edges for each entity type with optimization
    total_edges = 0
    max_total_edges = 10000000  # Limit to 10M edges for memory
    
    for entity_type, entity_map in entity_maps.items():
        print(f"Creating {entity_type} edges...")
        edge_count = 0
        entities_processed = 0
        entities_skipped = 0
        
        for entity, trans_ids in entity_map.items():
            if len(trans_ids) < 2:
                continue
                
            # Skip entities with too many transactions (likely noise)
            if len(trans_ids) > 1000:
                entities_skipped += 1
                continue
            
            # Check total edge limit
            if total_edges >= max_total_edges:
                print(f"  Reached max total edges limit ({max_total_edges:,})")
                break
            
            entities_processed += 1
            if entities_processed % 10000 == 0:
                print(f"  Processed {entities_processed:,} entities, {edge_count:,} edges created...")
            
            # Sample if too many transactions
            if len(trans_ids) > 50:
                import random
                trans_ids = random.sample(trans_ids, 50)
            
            # For efficiency, just use the transaction order (already sorted by time in preprocessing)
            trans_ids_sorted = sorted(trans_ids)
            
            # Create edges based on entity type strategy
            if entity_type == 'card':
                # Connect each transaction to previous 5 (reduced from 10)
                for i in range(len(trans_ids_sorted)):
                    for j in range(max(0, i-5), i):
                        if edge_count >= max_edges_per_entity:
                            break
                        edge_src.append(trans_ids_sorted[j])
                        edge_dst.append(trans_ids_sorted[i])
                        edge_types.append(0)  # Card edge type
                        edge_count += 1
                        total_edges += 1
            
            elif entity_type == 'email':
                # Simplified edge creation for email
                for i in range(1, min(len(trans_ids_sorted), 5)):
                    if edge_count >= max_edges_per_entity or total_edges >= max_total_edges:
                        break
                    edge_src.append(trans_ids_sorted[i-1])
                    edge_dst.append(trans_ids_sorted[i])
                    edge_types.append(1)  # Email edge type
                    edge_count += 1
                    total_edges += 1
            
            elif entity_type == 'ip':
                # Hub-and-spoke for large groups
                if len(trans_ids_sorted) > 10:
                    hub = trans_ids_sorted[len(trans_ids_sorted)//2]
                    for trans_id in trans_ids_sorted:
                        if trans_id != hub:
                            edge_src.append(hub)
                            edge_dst.append(trans_id)
                            edge_types.append(2)  # IP edge type
                            edge_count += 1
                else:
                    # Full connectivity for small groups
                    for i in range(len(trans_ids_sorted)):
                        for j in range(i+1, len(trans_ids_sorted)):
                            edge_src.append(trans_ids_sorted[i])
                            edge_dst.append(trans_ids_sorted[j])
                            edge_types.append(2)
                            edge_count += 1
            
            elif entity_type == 'bin':
                # Connect recent transactions only
                for i in range(len(trans_ids_sorted)):
                    for j in range(max(0, i-5), i):
                        edge_src.append(trans_ids_sorted[j])
                        edge_dst.append(trans_ids_sorted[i])
                        edge_types.append(3)  # BIN edge type
                        edge_count += 1
        
        print(f"  Created {edge_count} {entity_type} edges")
    
    # Create DGL graph
    print(f"\nTotal edges: {len(edge_src)}")
    
    # Create bidirectional edges
    edge_src_bi = edge_src + edge_dst
    edge_dst_bi = edge_dst + edge_src
    edge_types_bi = edge_types + edge_types
    
    # Create graph
    g = dgl.graph((edge_src_bi, edge_dst_bi), num_nodes=len(df))
    g.edata['edge_type'] = torch.tensor(edge_types_bi)
    
    print(f"Graph created with {g.num_nodes()} nodes and {g.num_edges()} edges")
    
    return g

def compute_neighborhood_features(g, df, hop_sizes=[1]):  # Only 1-hop for efficiency
    """Compute risk statistics from graph neighborhoods"""
    print(f"\nComputing neighborhood features for {g.num_nodes()} nodes...")
    print("This may take a while for large graphs...")
    
    # Initialize feature arrays
    num_nodes = g.num_nodes()
    neigh_features = {}
    
    # Node features
    node_fraud = torch.tensor(df['is_fraud'].values, dtype=torch.float32)
    node_amount = torch.tensor(df['amount'].values, dtype=torch.float32)
    node_hour = torch.tensor(df['hour'].values, dtype=torch.float32)
    
    for hop in hop_sizes:
        print(f"Computing {hop}-hop features...")
        
        # Initialize arrays
        neigh_features[f'{hop}hop_count'] = np.zeros(num_nodes)
        neigh_features[f'{hop}hop_fraud_rate'] = np.zeros(num_nodes)
        neigh_features[f'{hop}hop_fraud_count'] = np.zeros(num_nodes)
        neigh_features[f'{hop}hop_avg_amount'] = np.zeros(num_nodes)
        neigh_features[f'{hop}hop_std_amount'] = np.zeros(num_nodes)
        neigh_features[f'{hop}hop_avg_hour'] = np.ones(num_nodes) * 12  # Default to noon
        
        # Process in batches for efficiency
        batch_size = 10000
        for start_idx in range(0, num_nodes, batch_size):
            end_idx = min(start_idx + batch_size, num_nodes)
            if start_idx % 100000 == 0:
                print(f"  Processing nodes {start_idx:,} to {end_idx:,}...")
            
            for node in range(start_idx, end_idx):
                # Get neighbors more efficiently
                neighbors = g.in_edges(node)[0].numpy()
                
                # Exclude self
                neighbors = neighbors[neighbors != node]
                
                if len(neighbors) == 0:
                    # No neighbors - defaults already set
                    continue
                else:
                    # Compute statistics
                    neighbor_fraud = node_fraud[neighbors]
                    neighbor_amount = node_amount[neighbors]
                    neighbor_hour = node_hour[neighbors]
                    
                    neigh_features[f'{hop}hop_count'][node] = len(neighbors)
                    neigh_features[f'{hop}hop_fraud_rate'][node] = neighbor_fraud.mean().item()
                    neigh_features[f'{hop}hop_fraud_count'][node] = neighbor_fraud.sum().item()
                    neigh_features[f'{hop}hop_avg_amount'][node] = neighbor_amount.mean().item()
                    neigh_features[f'{hop}hop_std_amount'][node] = neighbor_amount.std().item() if len(neighbors) > 1 else 0
                    neigh_features[f'{hop}hop_avg_hour'][node] = neighbor_hour.mean().item()
    
    # Convert to DataFrame
    neigh_df = pd.DataFrame(neigh_features)
    neigh_df['trans_id'] = df['trans_id'].values
    
    print(f"Computed {len(neigh_df.columns)-1} neighborhood features")
    
    return neigh_df

def create_train_test_graphs(train_df, test_df):
    """Create separate graphs for train and test with controlled connections"""
    print("\nCreating train/test graphs...")
    
    # Combine dataframes for graph building
    combined_df = pd.concat([train_df, test_df], ignore_index=True)
    combined_df['trans_id'] = range(len(combined_df))
    combined_df['is_train'] = [1] * len(train_df) + [0] * len(test_df)
    
    # Build initial graph
    g_full = build_transaction_graph(combined_df)
    
    # Filter edges to prevent test-test connections
    edge_src, edge_dst = g_full.edges()
    edge_types = g_full.edata['edge_type']
    
    # Identify train and test nodes
    train_mask = torch.tensor(combined_df['is_train'].values, dtype=torch.bool)
    
    # Keep only edges where at least one node is from training
    valid_edges = train_mask[edge_src] | train_mask[edge_dst]
    
    # Create filtered graph
    filtered_src = edge_src[valid_edges]
    filtered_dst = edge_dst[valid_edges]
    filtered_types = edge_types[valid_edges]
    
    g_filtered = dgl.graph((filtered_src, filtered_dst), num_nodes=len(combined_df))
    g_filtered.edata['edge_type'] = filtered_types
    
    print(f"Filtered graph: {g_full.num_edges()} -> {g_filtered.num_edges()} edges")
    
    # Compute neighborhood features on filtered graph
    neigh_features_df = compute_neighborhood_features(g_filtered, combined_df)
    
    # Split back into train/test
    train_neigh = neigh_features_df.iloc[:len(train_df)]
    test_neigh = neigh_features_df.iloc[len(train_df):]
    
    return g_filtered, train_neigh, test_neigh, combined_df

def save_graph_data(g, train_neigh, test_neigh, combined_df, output_path):
    """Save graph and features"""
    print("\nSaving graph data...")
    
    os.makedirs(output_path, exist_ok=True)
    
    # Save graph
    dgl.save_graphs(os.path.join(output_path, 'transaction_graph.dgl'), [g])
    
    # Save neighborhood features
    train_neigh.to_csv(os.path.join(output_path, 'train_neigh_features.csv'), index=False)
    test_neigh.to_csv(os.path.join(output_path, 'test_neigh_features.csv'), index=False)
    
    # Save combined dataframe for reference
    combined_df.to_csv(os.path.join(output_path, 'combined_transactions.csv'), index=False)
    
    print(f"Graph data saved to {output_path}")

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
    
    # Convert dates back to datetime
    for df in [train_df, test_df]:
        df['issue_date'] = pd.to_datetime(df['issue_date'])
    
    # Create graphs and compute features
    g, train_neigh, test_neigh, combined_df = create_train_test_graphs(train_df, test_df)
    
    # Save graph data
    save_graph_data(g, train_neigh, test_neigh, combined_df, output_path)
    
    print("\nGraph generation complete!")

if __name__ == "__main__":
    main()