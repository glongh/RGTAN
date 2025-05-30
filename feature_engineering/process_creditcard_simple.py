#!/usr/bin/env python3
"""
Simple processing of CreditCard.mat to create adjacency lists without DGL.
This creates the pickle file that RGTAN expects.
"""

import os
import pickle
import numpy as np
from scipy.io import loadmat
import scipy.sparse as sp
from collections import defaultdict

DATADIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data/")


def sparse_to_adjlist(sp_matrix, filename):
    """Transfer sparse matrix to adjacency list"""
    # add self loop
    homo_adj = sp_matrix + sp.eye(sp_matrix.shape[0])
    # create adj_list
    adj_lists = defaultdict(set)
    edges = homo_adj.nonzero()
    for index, node in enumerate(edges[0]):
        adj_lists[node].add(edges[1][index])
        adj_lists[edges[1][index]].add(node)
    
    # Convert to regular dict for pickle
    adj_dict = dict(adj_lists)
    
    with open(filename, 'wb') as file:
        pickle.dump(adj_dict, file)
    
    print(f"Created adjacency list with {len(adj_dict)} nodes")
    print(f"Total edges: {sum(len(neighbors) for neighbors in adj_dict.values()) // 2}")


def main():
    print("Processing CreditCard.mat...")
    
    # Load CreditCard.mat
    creditcard = loadmat(os.path.join(DATADIR, 'CreditCard.mat'))
    
    # Extract homogeneous adjacency matrix
    homo = creditcard['homo']
    
    # Create adjacency list file
    print("Creating adjacency list...")
    output_file = os.path.join(DATADIR, "creditcard_homo_adjlists.pickle")
    sparse_to_adjlist(homo, output_file)
    
    print(f"\nSuccessfully created: {output_file}")
    print("\nNow you can run RGTAN with:")
    print("python main.py --creditcard --gpu 0")


if __name__ == "__main__":
    main()