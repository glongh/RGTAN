#!/usr/bin/env python3
"""
Data validation and coordination script
Ensures all pipeline components use consistent data sizes
"""

import os
import sys
import pandas as pd
import numpy as np
import dgl
import yaml
from datetime import datetime

def load_config():
    """Load configuration"""
    config_paths = [
        '../config/fraud_config.yaml',
        'config/fraud_config.yaml',
        '/home/development/affdf/fraud_prevention/config/fraud_config.yaml'
    ]
    
    for config_path in config_paths:
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
    
    # Default config if no file found
    print("Warning: No config file found, using defaults")
    return {
        'data': {
            'data_path': '/home/development/affdf/fraud_prevention/data'
        }
    }

def validate_data_consistency():
    """Validate that all data components are consistent"""
    print("=== Pipeline Data Validation ===")
    
    config = load_config()
    configured_path = config['data']['data_path']
    
    # Try multiple possible data paths
    possible_paths = [
        configured_path,
        '../data',
        '/home/development/affdf/fraud_prevention/data',
        'data',
        '/home/development/affdf/data'
    ]
    
    data_path = None
    for path in possible_paths:
        if os.path.exists(path):
            print(f"Checking data path: {path}")
            # Check if it has the required processed files
            if os.path.exists(os.path.join(path, 'processed')):
                data_path = path
                break
    
    if data_path is None:
        print("Error: No valid data path found")
        return False
    
    print(f"Using data path: {data_path}")
    
    issues = []
    
    # Check 1: CSV files exist
    print("1. Checking CSV files...")
    required_files = [
        'processed/train_transactions.csv',
        'processed/test_transactions.csv',
        'graph/train_neigh_features.csv',
        'graph/test_neigh_features.csv',
        'graph/combined_transactions.csv'
    ]
    
    file_sizes = {}
    for file_path in required_files:
        full_path = os.path.join(data_path, file_path)
        if os.path.exists(full_path):
            df = pd.read_csv(full_path)
            file_sizes[file_path] = len(df)
            print(f"   ✓ {file_path}: {len(df):,} rows")
        else:
            print(f"   ✗ Missing: {file_path}")
            issues.append(f"Missing file: {file_path}")
    
    # Check 2: Graph file exists and size
    print("\n2. Checking graph file...")
    graph_path = os.path.join(data_path, 'graph/transaction_graph.dgl')
    if os.path.exists(graph_path):
        try:
            graphs, _ = dgl.load_graphs(graph_path)
            g = graphs[0]
            graph_nodes = g.num_nodes()
            graph_edges = g.num_edges()
            print(f"   ✓ Graph: {graph_nodes:,} nodes, {graph_edges:,} edges")
        except Exception as e:
            print(f"   ✗ Error loading graph: {e}")
            issues.append(f"Graph loading error: {e}")
            graph_nodes = 0
    else:
        print(f"   ✗ Missing graph file: {graph_path}")
        issues.append("Missing graph file")
        graph_nodes = 0
    
    # Check 3: Size consistency
    print("\n3. Checking size consistency...")
    if 'graph/combined_transactions.csv' in file_sizes and graph_nodes > 0:
        combined_size = file_sizes['graph/combined_transactions.csv']
        if combined_size == graph_nodes:
            print(f"   ✓ Combined transactions ({combined_size:,}) matches graph nodes ({graph_nodes:,})")
        else:
            print(f"   ✗ Size mismatch: Combined transactions ({combined_size:,}) vs Graph nodes ({graph_nodes:,})")
            issues.append(f"Size mismatch: Combined {combined_size} vs Graph {graph_nodes}")
    
    # Check 4: Neighborhood features consistency
    if 'graph/train_neigh_features.csv' in file_sizes and 'graph/test_neigh_features.csv' in file_sizes:
        train_neigh = file_sizes['graph/train_neigh_features.csv']
        test_neigh = file_sizes['graph/test_neigh_features.csv']
        total_neigh = train_neigh + test_neigh
        
        if total_neigh == graph_nodes:
            print(f"   ✓ Neighborhood features ({total_neigh:,}) matches graph nodes ({graph_nodes:,})")
        else:
            print(f"   ✗ Neighborhood features mismatch: {total_neigh:,} vs {graph_nodes:,}")
            issues.append(f"Neighborhood features mismatch: {total_neigh} vs {graph_nodes}")
    
    # Check 5: Original data vs processed data
    if 'processed/train_transactions.csv' in file_sizes and 'processed/test_transactions.csv' in file_sizes:
        original_train = file_sizes['processed/train_transactions.csv']
        original_test = file_sizes['processed/test_transactions.csv']
        original_total = original_train + original_test
        
        print(f"\n4. Data size comparison:")
        print(f"   Original data: {original_total:,} transactions")
        print(f"   Graph data: {graph_nodes:,} transactions")
        
        if original_total == graph_nodes:
            print("   ✓ Using full dataset")
        else:
            sampling_ratio = graph_nodes / original_total if original_total > 0 else 0
            print(f"   ⚠ Using sampled dataset ({sampling_ratio:.1%} of original)")
    
    # Summary
    print(f"\n=== Validation Summary ===")
    if issues:
        print("❌ Issues found:")
        for issue in issues:
            print(f"   - {issue}")
        return False
    else:
        print("✅ All validations passed!")
        return True

def recommend_action():
    """Recommend which pipeline to run based on data state"""
    print("\n=== Recommendation ===")
    
    config = load_config()
    data_path = config['data']['data_path']
    if not os.path.exists(data_path):
        data_path = '../data'
    
    # Check if we have consistent data
    graph_path = os.path.join(data_path, 'graph/transaction_graph.dgl')
    combined_path = os.path.join(data_path, 'graph/combined_transactions.csv')
    
    if os.path.exists(graph_path) and os.path.exists(combined_path):
        print("✅ Ready for training with existing sampled data")
        print("Command: python train_fraud_rgtan.py")
    else:
        print("⚠ Need to regenerate graph and features")
        print("Command: python generate_graph_features_fast.py")
        print("Then: python train_fraud_rgtan.py")

if __name__ == "__main__":
    is_valid = validate_data_consistency()
    recommend_action()
    
    if not is_valid:
        print("\n⚠ Fix data issues before proceeding with training")
        sys.exit(1)