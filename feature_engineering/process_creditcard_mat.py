#!/usr/bin/env python3
"""
Process CreditCard.mat to create adjacency lists and neighbor features for RGTAN.
This is similar to what data_process.py does for Amazon.mat
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from scipy.io import loadmat
import scipy.sparse as sp
from collections import defaultdict
import torch
import dgl
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
    with open(filename, 'wb') as file:
        pickle.dump(adj_lists, file)
    file.close()


def count_risk_neighs(graph: dgl.DGLGraph, risk_label: int = 1) -> torch.Tensor:
    """Count risk neighbors for each node"""
    ret = []
    for center_idx in graph.nodes():
        neigh_idxs = graph.successors(center_idx)
        neigh_labels = graph.ndata['label'][neigh_idxs]
        risk_neigh_num = (neigh_labels == risk_label).sum()
        ret.append(risk_neigh_num)
    return torch.Tensor(ret)


def k_neighs(graph: dgl.DGLGraph, center_idx: int, k: int, where: str) -> torch.Tensor:
    """Get k-hop neighbors"""
    if k == 1:
        if where == "in":
            neigh_idxs = graph.predecessors(center_idx)
        elif where == "out":
            neigh_idxs = graph.successors(center_idx)
    elif k == 2:
        if where == "in":
            subg_in = dgl.khop_in_subgraph(graph, center_idx, 2, store_ids=True)[0]
            neigh_idxs = subg_in.ndata[dgl.NID][subg_in.ndata[dgl.NID] != center_idx]
            neigh1s = graph.predecessors(center_idx)
            neigh_idxs = neigh_idxs[~torch.isin(neigh_idxs, neigh1s)]
        elif where == "out":
            subg_out = dgl.khop_out_subgraph(graph, center_idx, 2, store_ids=True)[0]
            neigh_idxs = subg_out.ndata[dgl.NID][subg_out.ndata[dgl.NID] != center_idx]
            neigh1s = graph.successors(center_idx)
            neigh_idxs = neigh_idxs[~torch.isin(neigh_idxs, neigh1s)]
    return neigh_idxs


def feat_map(graph, edge_feat):
    """Generate neighborhood features"""
    tensor_list = []
    for idx in tqdm(range(graph.num_nodes()), desc="Generating neighbor features"):
        neighs_1_of_center = k_neighs(graph, idx, 1, "in")
        neighs_2_of_center = k_neighs(graph, idx, 2, "in")
        
        # Handle empty neighborhoods
        feat1_sum = edge_feat[neighs_1_of_center, 0].sum().item() if len(neighs_1_of_center) > 0 else 0
        feat2_sum = edge_feat[neighs_2_of_center, 0].sum().item() if len(neighs_2_of_center) > 0 else 0
        risk1_sum = edge_feat[neighs_1_of_center, 1].sum().item() if len(neighs_1_of_center) > 0 else 0
        risk2_sum = edge_feat[neighs_2_of_center, 1].sum().item() if len(neighs_2_of_center) > 0 else 0
        
        tensor = torch.FloatTensor([feat1_sum, feat2_sum, risk1_sum, risk2_sum])
        tensor_list.append(tensor)
    
    feat_names = ["1hop_degree", "2hop_degree", "1hop_riskstat", "2hop_riskstat"]
    tensor_list = torch.stack(tensor_list)
    return tensor_list, feat_names


def main():
    print("Processing CreditCard.mat...")
    
    # Load CreditCard.mat
    creditcard = loadmat(os.path.join(DATADIR, 'CreditCard.mat'))
    
    # Extract adjacency matrices
    net_upu = creditcard['net_upu']  # Card-based connections
    net_usu = creditcard['net_usu']  # IP-based connections
    net_uvu = creditcard['net_uvu']  # Email-based connections
    homo = creditcard['homo']         # Homogeneous graph
    
    # Create adjacency list files
    print("Creating adjacency lists...")
    sparse_to_adjlist(homo, os.path.join(DATADIR, "creditcard_homo_adjlists.pickle"))
    
    # Load labels and features
    labels = pd.DataFrame(creditcard['label'].flatten())[0]
    feat_data = pd.DataFrame(creditcard['features'].todense().A)
    
    # Create DGL graph
    print("Building DGL graph...")
    with open(os.path.join(DATADIR, "creditcard_homo_adjlists.pickle"), 'rb') as file:
        homo_adj = pickle.load(file)
    
    src = []
    tgt = []
    for i in homo_adj:
        for j in homo_adj[i]:
            src.append(i)
            tgt.append(j)
    
    src = np.array(src)
    tgt = np.array(tgt)
    g = dgl.graph((src, tgt))
    g.ndata['label'] = torch.from_numpy(labels.to_numpy()).to(torch.long)
    g.ndata['feat'] = torch.from_numpy(feat_data.to_numpy()).to(torch.float32)
    
    # Save DGL graph
    dgl.data.utils.save_graphs(DATADIR + "graph-creditcard.bin", [g])
    print(f"Saved DGL graph: {g.num_nodes()} nodes, {g.num_edges()} edges")
    
    # Generate neighbor risk-aware features
    print("Generating neighbor risk-aware features...")
    degree_feat = g.in_degrees().unsqueeze_(1).float()
    risk_feat = count_risk_neighs(g).unsqueeze_(1).float()
    
    edge_feat = torch.cat([degree_feat, risk_feat], dim=1)
    origin_feat_name = ['degree', 'riskstat']
    
    features_neigh, feat_names = feat_map(g, edge_feat)
    
    features_neigh = torch.cat((edge_feat, features_neigh), dim=1).numpy()
    feat_names = origin_feat_name + feat_names
    features_neigh[np.isnan(features_neigh)] = 0.
    
    # Save neighbor features
    output_path = DATADIR + "creditcard_neigh_feat.csv"
    features_neigh = pd.DataFrame(features_neigh, columns=feat_names)
    scaler = StandardScaler()
    features_neigh = pd.DataFrame(
        scaler.fit_transform(features_neigh), 
        columns=features_neigh.columns
    )
    features_neigh.to_csv(output_path, index=False)
    print(f"Saved neighbor features to {output_path}")
    
    print("\nProcessing complete! Files created:")
    print(f"- {DATADIR}creditcard_homo_adjlists.pickle")
    print(f"- {DATADIR}graph-creditcard.bin")
    print(f"- {DATADIR}creditcard_neigh_feat.csv")


if __name__ == "__main__":
    main()