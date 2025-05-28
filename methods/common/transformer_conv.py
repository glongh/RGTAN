"""
Shared Transformer Convolution module for GTAN and RGTAN models.
This module contains the TransformerConv layer that is used by both models
to avoid code duplication.
"""

import torch
import torch.nn as nn
from dgl.utils import expand_as_pair
from dgl import function as fn
from dgl.base import DGLError
from dgl.nn.functional import edge_softmax


class TransformerConv(nn.Module):
    """
    Transformer-based Graph Convolution layer with multi-head attention.
    
    This layer implements a graph convolution operation using multi-head
    self-attention mechanism, similar to the Transformer architecture.
    It supports gated residual connections and layer normalization.
    """

    def __init__(self,
                 in_feats,
                 out_feats,
                 num_heads,
                 bias=True,
                 allow_zero_in_degree=False,
                 skip_feat=True,
                 gated=True,
                 layer_norm=True,
                 activation=nn.PReLU()):
        """
        Initialize the transformer layer.
        
        Args:
            in_feats: Input feature dimension
            out_feats: Output feature dimension per attention head
            num_heads: Number of attention heads
            bias: Whether to use bias in linear transformations
            allow_zero_in_degree: Whether to allow nodes with zero in-degree
            skip_feat: Whether to use skip connections
            gated: Whether to use gated residual connections
            layer_norm: Whether to apply layer normalization
            activation: Activation function to use
        """
        super(TransformerConv, self).__init__()
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        self._num_heads = num_heads

        # Query, Key, Value transformations
        self.lin_query = nn.Linear(
            self._in_src_feats, self._out_feats * self._num_heads, bias=bias)
        self.lin_key = nn.Linear(
            self._in_src_feats, self._out_feats * self._num_heads, bias=bias)
        self.lin_value = nn.Linear(
            self._in_src_feats, self._out_feats * self._num_heads, bias=bias)

        # Optional skip connection
        if skip_feat:
            self.skip_feat = nn.Linear(
                self._in_src_feats, self._out_feats * self._num_heads, bias=bias)
        else:
            self.skip_feat = None
            
        # Optional gating mechanism
        if gated:
            self.gate = nn.Linear(
                3 * self._out_feats * self._num_heads, 1, bias=bias)
        else:
            self.gate = None
            
        # Optional layer normalization
        if layer_norm:
            self.layer_norm = nn.LayerNorm(self._out_feats * self._num_heads)
        else:
            self.layer_norm = None
            
        self.activation = activation

    def forward(self, graph, feat, get_attention=False):
        """
        Forward pass of the Transformer Graph Convolution.
        
        Args:
            graph: DGL graph object
            feat: Node features (can be a tensor or tuple of tensors)
            get_attention: Whether to return attention weights
            
        Returns:
            Updated node features after graph convolution
            (optionally) Attention weights if get_attention=True
        """
        graph = graph.local_var()

        if not self._allow_zero_in_degree:
            if (graph.in_degrees() == 0).any():
                raise DGLError('There are 0-in-degree nodes in the graph, '
                               'output for those nodes will be invalid. '
                               'This is harmful for some applications, '
                               'causing silent performance regression. '
                               'Adding self-loop on the input graph by '
                               'calling `g = dgl.add_self_loop(g)` will resolve '
                               'the issue. Setting ``allow_zero_in_degree`` '
                               'to be `True` when constructing this module will '
                               'suppress the check and let the code run.')

        # Handle tuple input for heterogeneous graphs
        if isinstance(feat, tuple):
            h_src = feat[0]
            h_dst = feat[1]
        else:
            h_src = feat
            h_dst = h_src[:graph.number_of_dst_nodes()]

        # Step 1: Linear transformations for Q, K, V
        q_src = self.lin_query(h_src).view(-1, self._num_heads, self._out_feats)
        k_dst = self.lin_key(h_dst).view(-1, self._num_heads, self._out_feats)
        v_src = self.lin_value(h_src).view(-1, self._num_heads, self._out_feats)
        
        # Assign features to nodes
        graph.srcdata.update({'ft': q_src, 'ft_v': v_src})
        graph.dstdata.update({'ft': k_dst})
        
        # Step 2: Compute attention scores (Q·K)
        graph.apply_edges(fn.u_dot_v('ft', 'ft', 'a'))

        # Step 3: Apply edge softmax to get attention weights
        graph.edata['sa'] = edge_softmax(
            graph, graph.edata['a'] / self._out_feats**0.5)

        # Step 4: Aggregate messages weighted by attention (attention * V)
        graph.update_all(fn.u_mul_e('ft_v', 'sa', 'attn'),
                         fn.sum('attn', 'agg_u'))

        # Reshape output
        rst = graph.dstdata['agg_u'].reshape(-1, self._out_feats * self._num_heads)

        # Apply skip connection and gating if enabled
        if self.skip_feat is not None:
            skip_feat = self.skip_feat(feat[:graph.number_of_dst_nodes()])
            if self.gate is not None:
                gate = torch.sigmoid(
                    self.gate(
                        torch.concat([skip_feat, rst, skip_feat - rst], dim=-1)))
                rst = gate * skip_feat + (1 - gate) * rst
            else:
                rst = skip_feat + rst

        # Apply layer normalization if enabled
        if self.layer_norm is not None:
            rst = self.layer_norm(rst)

        # Apply activation function if provided
        if self.activation is not None:
            rst = self.activation(rst)

        if get_attention:
            return rst, graph.edata['sa']
        else:
            return rst