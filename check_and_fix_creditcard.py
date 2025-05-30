#!/usr/bin/env python3
"""
Check and fix CreditCard dataset issues
"""

import os
import numpy as np
from scipy.io import loadmat
import scipy.sparse as sp

# Check if running on your system
data_path = "/home/development/affdf/data/CreditCard.mat"
if not os.path.exists(data_path):
    data_path = "data/CreditCard.mat"

print(f"Checking {data_path}...")

if os.path.exists(data_path):
    # Load and check the MAT file
    mat_data = loadmat(data_path)
    
    print("\n=== Dataset Info ===")
    print(f"Keys: {[k for k in mat_data.keys() if not k.startswith('__')]}")
    
    # Check each component
    labels = mat_data['label']
    features = mat_data['features']
    homo = mat_data['homo']
    
    print(f"\nLabel shape: {labels.shape}")
    print(f"Features shape: {features.shape}")
    print(f"Homo adjacency shape: {homo.shape}")
    
    # Check if shapes match
    n_labels = labels.shape[1] if labels.shape[0] == 1 else labels.shape[0]
    n_features = features.shape[0]
    n_nodes_graph = homo.shape[0]
    
    print(f"\nNumber of labels: {n_labels}")
    print(f"Number of feature rows: {n_features}")
    print(f"Number of nodes in graph: {n_nodes_graph}")
    
    if n_labels != n_features or n_labels != n_nodes_graph:
        print("\n⚠️  MISMATCH DETECTED!")
        print("The dataset components have different sizes.")
        
        # Check which nodes are missing from the graph
        if sp.issparse(homo):
            rows, cols = homo.nonzero()
            unique_nodes = np.unique(np.concatenate([rows, cols]))
            print(f"\nUnique nodes in adjacency matrix: {len(unique_nodes)}")
            print(f"Min node ID: {unique_nodes.min()}, Max node ID: {unique_nodes.max()}")
            
            # Find missing nodes
            all_nodes = set(range(n_labels))
            graph_nodes = set(unique_nodes)
            missing_nodes = all_nodes - graph_nodes
            print(f"Missing nodes: {len(missing_nodes)}")
            if len(missing_nodes) < 10:
                print(f"Missing node IDs: {sorted(missing_nodes)}")
    else:
        print("\n✓ All components have matching sizes!")
else:
    print(f"File not found: {data_path}")
    
# Also check the CSV file if available
csv_path = "/home/development/affdf/data/vod_creditcard.csv"
if not os.path.exists(csv_path):
    csv_path = "data/vod_creditcard.csv"

if os.path.exists(csv_path):
    import pandas as pd
    df = pd.read_csv(csv_path)
    print(f"\n=== CSV Dataset Info ===")
    print(f"CSV file rows: {len(df)}")
    print(f"CSV columns: {df.shape[1]}")
    print("You should regenerate CreditCard.mat with:")
    print(f"python feature_engineering/convert_creditcard_to_mat.py --max-rows {len(df)}")