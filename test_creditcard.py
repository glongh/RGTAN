#!/usr/bin/env python3
"""
Test script for creditcard dataset integration with RGTAN
"""

import sys
import os
import pandas as pd
import numpy as np
from methods.rgtan.rgtan_main import load_rgtan_data

def test_creditcard_loading():
    """Test the creditcard dataset loading and graph construction"""
    
    print("Testing creditcard dataset loading...")
    
    # Load the dataset
    try:
        feat_data, labels, train_idx, test_idx, g, cat_features, neigh_features = load_rgtan_data('creditcard', 0.3)
        
        print("\n=== Dataset Statistics ===")
        print(f"Total transactions: {len(feat_data)}")
        print(f"Feature dimensions: {feat_data.shape[1]}")
        print(f"Training samples: {len(train_idx)}")
        print(f"Test samples: {len(test_idx)}")
        
        print(f"\n=== Graph Statistics ===")
        print(f"Number of nodes: {g.number_of_nodes()}")
        print(f"Number of edges: {g.number_of_edges()}")
        print(f"Average degree: {g.number_of_edges() / g.number_of_nodes():.2f}")
        
        print(f"\n=== Label Distribution ===")
        print(f"Total fraud rate: {labels.mean():.2%}")
        print(f"Train fraud rate: {labels.iloc[train_idx].mean():.2%}")
        print(f"Test fraud rate: {labels.iloc[test_idx].mean():.2%}")
        
        print(f"\n=== Categorical Features ===")
        print(f"Number of categorical features: {len(cat_features)}")
        print(f"Categorical features: {cat_features}")
        
        print(f"\n=== Feature Values Sample ===")
        print(feat_data.head())
        
        # Check for data quality issues
        print(f"\n=== Data Quality Checks ===")
        print(f"Missing values in features: {feat_data.isna().sum().sum()}")
        print(f"Infinite values in features: {np.isinf(feat_data.values).sum()}")
        
        # Check graph connectivity
        import networkx as nx
        nx_g = g.to_networkx()
        num_components = nx.number_connected_components(nx_g.to_undirected())
        print(f"Number of connected components: {num_components}")
        
        print("\n✅ Creditcard dataset loaded successfully!")
        
    except Exception as e:
        print(f"\n❌ Error loading creditcard dataset: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    test_creditcard_loading()