#!/usr/bin/env python3
"""
Test script to isolate the core dump issue
"""

import os
import sys
import pandas as pd
import numpy as np
import torch
import dgl

def test_data_loading():
    """Test data loading step by step"""
    print("=== Testing Data Loading ===")
    
    data_path = '/home/development/affdf/fraud_prevention/data'
    
    # Test 1: Load CSV files
    try:
        print("1. Loading CSV files...")
        train_df = pd.read_csv(os.path.join(data_path, 'processed/train_transactions.csv'))
        print(f"   Train data: {len(train_df):,} rows, {len(train_df.columns)} columns")
        
        test_df = pd.read_csv(os.path.join(data_path, 'processed/test_transactions.csv'))
        print(f"   Test data: {len(test_df):,} rows, {len(test_df.columns)} columns")
        
        train_neigh = pd.read_csv(os.path.join(data_path, 'graph/train_neigh_features.csv'))
        print(f"   Train neigh: {train_neigh.shape}")
        
        test_neigh = pd.read_csv(os.path.join(data_path, 'graph/test_neigh_features.csv'))
        print(f"   Test neigh: {test_neigh.shape}")
        
        print("   ✓ CSV loading successful")
    except Exception as e:
        print(f"   ✗ CSV loading failed: {e}")
        return False
    
    # Test 2: Load graph
    try:
        print("2. Loading graph...")
        graphs, _ = dgl.load_graphs(os.path.join(data_path, 'graph/transaction_graph.dgl'))
        g = graphs[0]
        print(f"   Graph: {g.num_nodes():,} nodes, {g.num_edges():,} edges")
        print("   ✓ Graph loading successful")
    except Exception as e:
        print(f"   ✗ Graph loading failed: {e}")
        return False
    
    # Test 3: Simple feature extraction
    try:
        print("3. Testing feature extraction...")
        
        # Simple feature columns
        numeric_cols = ['amount_log', 'hour']  # Minimal columns
        if all(col in train_df.columns for col in numeric_cols):
            train_features = train_df[numeric_cols].values
            print(f"   Features shape: {train_features.shape}")
            print("   ✓ Feature extraction successful")
        else:
            print(f"   Available columns: {list(train_df.columns)}")
            print("   ✗ Required columns not found")
            return False
            
    except Exception as e:
        print(f"   ✗ Feature extraction failed: {e}")
        return False
    
    # Test 4: Tensor creation
    try:
        print("4. Testing tensor creation...")
        small_features = train_features[:1000]  # Small subset
        tensor = torch.FloatTensor(small_features)
        print(f"   Tensor shape: {tensor.shape}")
        print("   ✓ Tensor creation successful")
    except Exception as e:
        print(f"   ✗ Tensor creation failed: {e}")
        return False
    
    # Test 5: GPU availability
    try:
        print("5. Testing GPU...")
        if torch.cuda.is_available():
            print(f"   GPU available: {torch.cuda.get_device_name()}")
            print(f"   GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
            
            # Test small tensor on GPU
            small_tensor = torch.ones(100, 100).cuda()
            print("   ✓ GPU test successful")
        else:
            print("   No GPU available, using CPU")
    except Exception as e:
        print(f"   ✗ GPU test failed: {e}")
        return False
    
    print("=== All tests passed! ===")
    return True

if __name__ == "__main__":
    success = test_data_loading()
    if not success:
        sys.exit(1)