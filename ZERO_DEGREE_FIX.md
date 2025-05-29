# Zero In-Degree Nodes Fix

## Problem
`DGLError: There are 0-in-degree nodes in the graph`

This error occurs when some nodes in the graph have no incoming edges. In fraud detection datasets, this is common because:
- Some transactions might be isolated (no shared card, IP, email, etc.)
- First-time users or cards have no connection history
- Privacy/hashing might create unique identifiers

## Solution
Added self-loops to all nodes in the graph using `dgl.add_self_loop(g)`. This ensures:
- Every node has at least one incoming edge (from itself)
- Graph convolution operations can propagate information properly
- No nodes are left with invalid outputs

## Changes Made
Added self-loops in all dataset loading paths:
1. S-FFSD dataset
2. Yelp dataset  
3. Amazon dataset
4. Credit card dataset (both preprocessed and raw data paths)

## Code Changes
```python
# After creating the graph
g = dgl.graph((src, tgt))
# Add self-loops to handle isolated nodes
g = dgl.add_self_loop(g)
```

## Impact
- Isolated transactions can now participate in graph learning
- Model can learn node-specific features even without neighbors
- No performance regression from invalid outputs

## Alternative Solutions
1. Set `allow_zero_in_degree=True` in TransformerConv (less safe)
2. Remove isolated nodes (loses data)
3. Create synthetic edges (changes data distribution)