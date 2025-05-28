import torch


def load_lpa_subtensor(
    node_feat,  # (|all|, feat_dim)
    work_node_feat,
    neigh_feat: dict,
    neigh_padding_dict: dict,  # {"degree":6, ...}
    labels,  # (|all|,)
    seeds,  # (|batch|,)
    input_nodes,  # (|batch_all|,)
    device,
    blocks,
):
    """
    Efficiently load subgraph data for mini-batch training with label propagation.
    
    This function prepares batch data by extracting relevant node features,
    categorical features, and neighborhood statistics. It also masks certain
    neighborhood features to prevent label leakage during training.
    
    Args:
        node_feat: Node feature tensor of shape (num_nodes, feat_dim)
        work_node_feat: Dictionary of categorical node features
        neigh_feat: Dictionary of neighborhood statistics
        neigh_padding_dict: Padding configuration for neighborhood features
        labels: Node labels tensor
        seeds: Indices of target nodes in the batch
        input_nodes: All nodes involved in the computation (includes neighbors)
        device: Device to place tensors on
        blocks: DGL blocks for multi-layer sampling
        
    Returns:
        tuple: (batch_inputs, batch_work_inputs, batch_neighstat_inputs, 
                batch_labels, propagate_labels)
    """
    # masking to avoid label leakage
    if "1hop_riskstat" in neigh_feat.keys() and len(blocks) >= 2:
        # nei_hop1 = get_k_neighs(graph, seeds, 1)
        nei_hop1 = blocks[-2].dstdata['_ID']
        neigh_feat['1hop_riskstat'][nei_hop1] = 0

    if "2hop_riskstat" in neigh_feat.keys() and len(blocks) >= 3:
        # nei_hop2 = get_k_neighs(graph, seeds, 2)
        nei_hop2 = blocks[-3].dstdata['_ID']
        neigh_feat['2hop_riskstat'][nei_hop2] = 0

    # Efficiently move data to device without unnecessary copies
    batch_inputs = node_feat[input_nodes].to(device)
    batch_work_inputs = {i: work_node_feat[i][input_nodes].to(
        device) for i in work_node_feat if i not in {"labels"}}  # cat feats

    batch_neighstat_inputs = None

    if neigh_feat:
        batch_neighstat_inputs = {col: neigh_feat[col][input_nodes].to(
            device) for col in neigh_feat.keys()}

    batch_labels = labels[seeds].to(device)
    
    # Avoid deep copy - use clone() for memory efficiency
    propagate_labels = labels[input_nodes].clone()
    propagate_labels[:seeds.shape[0]] = 2
    
    return batch_inputs, batch_work_inputs, batch_neighstat_inputs, batch_labels, propagate_labels.to(device)
