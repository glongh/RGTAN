#!/usr/bin/env python3
"""
Generate neighborhood features for CreditCard dataset.
This is a lightweight version that doesn't require DGL.
"""

import os
import numpy as np
import pandas as pd
from scipy.io import loadmat
import scipy.sparse as sp
from sklearn.preprocessing import StandardScaler
import pickle
from collections import defaultdict
from tqdm import tqdm

DATADIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data/")


def compute_neighbor_features_from_sparse(homo_adj, labels):
    """Compute neighbor features directly from sparse adjacency matrix"""
    # Convert to CSR format for efficient row access
    if not isinstance(homo_adj, sp.csr_matrix):
        print("Converting adjacency matrix to CSR format...")
        homo_adj = homo_adj.tocsr()
    
    n_nodes = homo_adj.shape[0]
    
    # Initialize feature arrays
    degree = np.zeros(n_nodes)
    riskstat = np.zeros(n_nodes)
    hop1_degree = np.zeros(n_nodes)
    hop2_degree = np.zeros(n_nodes)
    hop1_riskstat = np.zeros(n_nodes)
    hop2_riskstat = np.zeros(n_nodes)
    
    print("Computing node degrees...")
    # Compute degree (number of neighbors) for each node
    degree = np.array(homo_adj.sum(axis=1)).flatten()
    
    print("Computing risk statistics...")
    # For each node, count how many neighbors have label=1 (fraud)
    for i in tqdm(range(n_nodes), desc="Computing 1-hop risk stats"):
        # Get neighbors of node i
        neighbors = homo_adj[i].nonzero()[1]
        if len(neighbors) > 0:
            # Count fraudulent neighbors
            riskstat[i] = labels[neighbors].sum()
            # 1-hop degree is same as degree
            hop1_degree[i] = len(neighbors)
            hop1_riskstat[i] = riskstat[i]
    
    # Skip 2-hop computation for very large graphs (optional)
    compute_2hop = n_nodes < 100000
    
    if compute_2hop:
        print("Computing 2-hop features...")
        # For 2-hop neighbors (more expensive computation)
        # We'll sample for efficiency on large graphs
        sample_size = min(5000, n_nodes)  # Sample subset for 2-hop
        sample_indices = np.random.choice(n_nodes, sample_size, replace=False)
        
        for idx in tqdm(sample_indices, desc="Computing 2-hop features (sampled)"):
            # Get 1-hop neighbors
            neighbors_1hop = set(homo_adj[idx].nonzero()[1])
            
            # Get 2-hop neighbors
            neighbors_2hop = set()
            for n1 in neighbors_1hop:
                n2_candidates = homo_adj[n1].nonzero()[1]
                for n2 in n2_candidates:
                    if n2 != idx and n2 not in neighbors_1hop:
                        neighbors_2hop.add(n2)
            
            if len(neighbors_2hop) > 0:
                neighbors_2hop = list(neighbors_2hop)
                hop2_degree[idx] = len(neighbors_2hop)
                hop2_riskstat[idx] = labels[neighbors_2hop].sum()
    else:
        print("Skipping 2-hop features for large graph (>10k nodes)")
    
    # Create DataFrame
    features = pd.DataFrame({
        'degree': degree,
        'riskstat': riskstat,
        '1hop_degree': hop1_degree,
        '2hop_degree': hop2_degree,
        '1hop_riskstat': hop1_riskstat,
        '2hop_riskstat': hop2_riskstat
    })
    
    return features


def main():
    print("Generating neighborhood features for CreditCard dataset...")
    
    # Load CreditCard.mat
    mat_path = os.path.join(DATADIR, 'CreditCard.mat')
    if not os.path.exists(mat_path):
        print(f"Error: {mat_path} not found. Please run convert_creditcard_to_mat.py first.")
        return
    
    creditcard = loadmat(mat_path)
    homo = creditcard['homo']
    labels = creditcard['label'].flatten()
    
    print(f"Dataset: {homo.shape[0]} nodes, {homo.nnz} edges")
    print(f"Fraud rate: {labels.mean():.2%}")
    
    # Compute neighbor features
    features = compute_neighbor_features_from_sparse(homo, labels)
    
    # Standardize features
    print("Standardizing features...")
    scaler = StandardScaler()
    features_scaled = pd.DataFrame(
        scaler.fit_transform(features),
        columns=features.columns
    )
    
    # Save features
    output_path = os.path.join(DATADIR, 'creditcard_neigh_feat.csv')
    features_scaled.to_csv(output_path, index=False)
    print(f"Saved neighborhood features to {output_path}")
    
    # Print statistics
    print("\nFeature statistics:")
    print(features.describe())


if __name__ == "__main__":
    main()