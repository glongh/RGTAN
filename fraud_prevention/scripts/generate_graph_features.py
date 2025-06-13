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
    print("\nBuilding transaction graph...")
    
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
    
    # Build entity mappings
    for idx, row in df.iterrows():
        trans_id = row['trans_id']
        entity_maps['card'][row['entity_card']].append(trans_id)
        entity_maps['email'][row['entity_email']].append(trans_id)
        entity_maps['ip'][row['entity_ip']].append(trans_id)
        entity_maps['bin'][row['entity_bin']].append(trans_id)
    
    # Create edges for each entity type
    for entity_type, entity_map in entity_maps.items():
        print(f"Creating {entity_type} edges...")
        edge_count = 0
        
        for entity, trans_ids in entity_map.items():
            if len(trans_ids) < 2:
                continue
            
            # Sort by transaction time
            trans_ids_sorted = sorted(trans_ids, 
                                    key=lambda x: df[df['trans_id'] == x]['issue_date'].iloc[0])
            
            # Create edges based on entity type strategy
            if entity_type == 'card':
                # Connect each transaction to previous 10
                for i in range(len(trans_ids_sorted)):
                    for j in range(max(0, i-10), i):
                        edge_src.append(trans_ids_sorted[j])
                        edge_dst.append(trans_ids_sorted[i])
                        edge_types.append(0)  # Card edge type
                        edge_count += 1
            
            elif entity_type == 'email':
                # Connect transactions within time window
                for i in range(len(trans_ids_sorted)):
                    trans_time = df[df['trans_id'] == trans_ids_sorted[i]]['issue_date'].iloc[0]
                    for j in range(i+1, min(len(trans_ids_sorted), i+20)):
                        other_time = df[df['trans_id'] == trans_ids_sorted[j]]['issue_date'].iloc[0]
                        if (other_time - trans_time).total_seconds() / 3600 <= time_window_hours:
                            edge_src.append(trans_ids_sorted[i])
                            edge_dst.append(trans_ids_sorted[j])
                            edge_types.append(1)  # Email edge type
                            edge_count += 1
            
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

def compute_neighborhood_features(g, df, hop_sizes=[1, 2]):
    """Compute risk statistics from graph neighborhoods"""
    print("\nComputing neighborhood features...")
    
    # Initialize feature dictionary
    neigh_features = {}
    
    # Node features
    node_fraud = torch.tensor(df['is_fraud'].values, dtype=torch.float32)
    node_amount = torch.tensor(df['amount'].values, dtype=torch.float32)
    node_hour = torch.tensor(df['hour'].values, dtype=torch.float32)
    
    for hop in hop_sizes:
        print(f"Computing {hop}-hop features...")
        
        # Get k-hop neighbors for each node
        neigh_features[f'{hop}hop_count'] = []
        neigh_features[f'{hop}hop_fraud_rate'] = []
        neigh_features[f'{hop}hop_fraud_count'] = []
        neigh_features[f'{hop}hop_avg_amount'] = []
        neigh_features[f'{hop}hop_std_amount'] = []
        neigh_features[f'{hop}hop_avg_hour'] = []
        
        for node in range(g.num_nodes()):
            # Get k-hop subgraph
            sg, _ = dgl.khop_in_subgraph(g, node, k=hop)
            neighbors = sg.ndata[dgl.NID].numpy()
            
            # Exclude self
            neighbors = neighbors[neighbors != node]
            
            if len(neighbors) == 0:
                # No neighbors
                neigh_features[f'{hop}hop_count'].append(0)
                neigh_features[f'{hop}hop_fraud_rate'].append(0)
                neigh_features[f'{hop}hop_fraud_count'].append(0)
                neigh_features[f'{hop}hop_avg_amount'].append(0)
                neigh_features[f'{hop}hop_std_amount'].append(0)
                neigh_features[f'{hop}hop_avg_hour'].append(12)
            else:
                # Compute statistics
                neighbor_fraud = node_fraud[neighbors]
                neighbor_amount = node_amount[neighbors]
                neighbor_hour = node_hour[neighbors]
                
                neigh_features[f'{hop}hop_count'].append(len(neighbors))
                neigh_features[f'{hop}hop_fraud_rate'].append(neighbor_fraud.mean().item())
                neigh_features[f'{hop}hop_fraud_count'].append(neighbor_fraud.sum().item())
                neigh_features[f'{hop}hop_avg_amount'].append(neighbor_amount.mean().item())
                neigh_features[f'{hop}hop_std_amount'].append(neighbor_amount.std().item() if len(neighbors) > 1 else 0)
                neigh_features[f'{hop}hop_avg_hour'].append(neighbor_hour.mean().item())
    
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