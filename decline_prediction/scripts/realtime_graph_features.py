#!/usr/bin/env python3
"""
Lightweight graph features for real-time decline prediction
Optimized for < 100ms latency with minimal graph construction
"""

import os
import sys
import pandas as pd
import numpy as np
import dgl
import torch
from datetime import datetime, timedelta
import pickle
import json
from collections import defaultdict, deque
import redis
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

class RealtimeGraphFeatureEngine:
    """Fast graph feature computation for real-time decline prediction"""
    
    def __init__(self, config=None):
        self.config = config or {
            'max_card_history': 5,      # Last 5 transactions per card
            'max_ip_history': 10,       # Last 10 transactions per IP
            'max_merchant_history': 20,  # Last 20 transactions per merchant
            'time_window_hours': 1,     # 1 hour lookback
            'cache_ttl': 3600,          # 1 hour cache TTL
            'use_redis': False          # Redis for distributed caching
        }
        
        # Initialize cache
        self.local_cache = {}
        self.redis_client = None
        
        if self.config['use_redis']:
            try:
                self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
                self.redis_client.ping()
                print("Connected to Redis cache")
            except:
                print("Redis not available, using local cache")
                self.redis_client = None
    
    def build_lightweight_graph(self, transactions_df, target_trans_ids=None):
        """Build minimal graph for specific transactions"""
        print(f"Building lightweight graph for {len(target_trans_ids or [])} transactions...")
        
        if target_trans_ids is None:
            target_trans_ids = transactions_df['trans_id'].tolist()
        
        # Filter to recent transactions only
        cutoff_time = datetime.now() - timedelta(hours=self.config['time_window_hours'])
        recent_df = transactions_df[transactions_df['issue_date'] > cutoff_time].copy()
        
        # Build entity-to-transaction mappings
        entity_maps = {
            'card': defaultdict(list),
            'ip': defaultdict(list),
            'merchant': defaultdict(list),
            'email': defaultdict(list)
        }
        
        # Populate mappings
        for idx, row in recent_df.iterrows():
            trans_id = row['trans_id']
            entity_maps['card'][row['entity_card']].append(trans_id)
            entity_maps['ip'][row['entity_ip']].append(trans_id)
            entity_maps['merchant'][row['entity_merchant']].append(trans_id)
            entity_maps['email'][row['entity_email']].append(trans_id)
        
        # Create minimal edges for target transactions only
        edge_src, edge_dst = [], []
        edge_types = []
        
        for target_trans in target_trans_ids:
            if target_trans not in recent_df['trans_id'].values:
                continue
                
            target_row = recent_df[recent_df['trans_id'] == target_trans].iloc[0]
            
            # Card edges (last N transactions)
            card_trans = entity_maps['card'][target_row['entity_card']]
            card_trans = [t for t in card_trans if t != target_trans][-self.config['max_card_history']:]
            for t in card_trans:
                edge_src.append(t)
                edge_dst.append(target_trans)
                edge_types.append(0)  # Card edge
            
            # IP edges (recent transactions)
            ip_trans = entity_maps['ip'][target_row['entity_ip']]
            ip_trans = [t for t in ip_trans if t != target_trans][-self.config['max_ip_history']:]
            for t in ip_trans:
                edge_src.append(t)
                edge_dst.append(target_trans)
                edge_types.append(1)  # IP edge
            
            # Merchant edges (recent transactions)
            merchant_trans = entity_maps['merchant'][target_row['entity_merchant']]
            merchant_trans = [t for t in merchant_trans if t != target_trans][-self.config['max_merchant_history']:]
            for t in merchant_trans:
                edge_src.append(t)
                edge_dst.append(target_trans)
                edge_types.append(2)  # Merchant edge
        
        # Create DGL graph if edges exist
        if len(edge_src) > 0:
            # Create bidirectional edges
            edge_src_bi = edge_src + edge_dst
            edge_dst_bi = edge_dst + edge_src
            edge_types_bi = edge_types + edge_types
            
            # Map transaction IDs to node indices
            unique_trans = list(set(edge_src_bi + edge_dst_bi))
            trans_to_idx = {trans: idx for idx, trans in enumerate(unique_trans)}
            
            mapped_src = [trans_to_idx[t] for t in edge_src_bi]
            mapped_dst = [trans_to_idx[t] for t in edge_dst_bi]
            
            g = dgl.graph((mapped_src, mapped_dst), num_nodes=len(unique_trans))
            g.edata['edge_type'] = torch.tensor(edge_types_bi)
            
            return g, trans_to_idx
        else:
            # Return empty graph
            return dgl.graph(([], []), num_nodes=1), {}
    
    def compute_fast_neighborhood_features(self, g, trans_to_idx, target_trans_ids, 
                                         transactions_df):
        """Compute neighborhood features optimized for speed"""
        print("Computing fast neighborhood features...")
        
        features = {}
        
        # Initialize feature arrays
        for trans_id in target_trans_ids:
            features[trans_id] = {
                'neighbor_count': 0,
                'neighbor_decline_rate': 0.0,
                'neighbor_avg_amount': 0.0,
                'neighbor_velocity': 0.0,
                'card_decline_rate': 0.0,
                'ip_decline_rate': 0.0,
                'merchant_decline_rate': 0.0
            }
        
        if len(trans_to_idx) == 0:
            return features
        
        # Get node features
        trans_data = transactions_df.set_index('trans_id')
        
        for target_trans in target_trans_ids:
            if target_trans not in trans_to_idx:
                continue
                
            target_idx = trans_to_idx[target_trans]
            
            # Get neighbors
            if target_idx < g.num_nodes():
                # Find neighbors
                in_edges = g.in_edges(target_idx)
                neighbor_indices = in_edges[0].unique().numpy()
                neighbor_indices = neighbor_indices[neighbor_indices != target_idx]
                
                if len(neighbor_indices) > 0:
                    # Map back to transaction IDs
                    idx_to_trans = {idx: trans for trans, idx in trans_to_idx.items()}
                    neighbor_trans_ids = [idx_to_trans[idx] for idx in neighbor_indices 
                                        if idx in idx_to_trans]
                    
                    # Compute features
                    neighbor_data = trans_data.loc[neighbor_trans_ids]
                    
                    features[target_trans]['neighbor_count'] = len(neighbor_data)
                    features[target_trans]['neighbor_decline_rate'] = neighbor_data['is_declined'].mean()
                    features[target_trans]['neighbor_avg_amount'] = neighbor_data['amount'].mean()
                    
                    # Velocity (transactions per hour)
                    time_span = (neighbor_data['issue_date'].max() - 
                               neighbor_data['issue_date'].min()).total_seconds() / 3600
                    features[target_trans]['neighbor_velocity'] = len(neighbor_data) / max(time_span, 1)
            
            # Entity-specific features (from cache or computation)
            target_data = trans_data.loc[target_trans]
            
            features[target_trans]['card_decline_rate'] = self._get_entity_decline_rate(
                'entity_card', target_data['entity_card'], transactions_df
            )
            features[target_trans]['ip_decline_rate'] = self._get_entity_decline_rate(
                'entity_ip', target_data['entity_ip'], transactions_df
            )
            features[target_trans]['merchant_decline_rate'] = self._get_entity_decline_rate(
                'entity_merchant', target_data['entity_merchant'], transactions_df
            )
        
        return features
    
    def _get_entity_decline_rate(self, entity_type, entity_value, transactions_df):
        """Get cached entity decline rate"""
        cache_key = f"{entity_type}:{entity_value}"
        
        # Try cache first
        cached_rate = self._get_from_cache(cache_key)
        if cached_rate is not None:
            return cached_rate
        
        # Compute if not cached
        entity_data = transactions_df[transactions_df[entity_type] == entity_value]
        if len(entity_data) == 0:
            decline_rate = 0.0
        else:
            decline_rate = entity_data['is_declined'].mean()
        
        # Cache result
        self._set_cache(cache_key, decline_rate)
        
        return decline_rate
    
    def _get_from_cache(self, key):
        """Get value from cache (Redis or local)"""
        if self.redis_client:
            try:
                value = self.redis_client.get(key)
                return float(value) if value else None
            except:
                pass
        
        return self.local_cache.get(key)
    
    def _set_cache(self, key, value):
        """Set value in cache (Redis or local)"""
        if self.redis_client:
            try:
                self.redis_client.setex(key, self.config['cache_ttl'], str(value))
            except:
                pass
        
        self.local_cache[key] = value
    
    def extract_realtime_features(self, transaction_data, historical_df=None):
        """Extract features for a single transaction in real-time"""
        
        # Create temporary transaction record
        temp_trans = pd.DataFrame([transaction_data])
        temp_trans['trans_id'] = 0  # Temporary ID
        temp_trans['issue_date'] = pd.to_datetime(temp_trans['issue_date'])
        
        # Combine with recent historical data if available
        if historical_df is not None:
            recent_cutoff = datetime.now() - timedelta(hours=self.config['time_window_hours'])
            recent_historical = historical_df[historical_df['issue_date'] > recent_cutoff]
            combined_df = pd.concat([recent_historical, temp_trans], ignore_index=True)
        else:
            combined_df = temp_trans
        
        # Build minimal graph
        g, trans_to_idx = self.build_lightweight_graph(combined_df, [0])
        
        # Compute features
        features = self.compute_fast_neighborhood_features(g, trans_to_idx, [0], combined_df)
        
        return features.get(0, {})
    
    def precompute_entity_statistics(self, training_df, output_path):
        """Precompute entity statistics for fast lookup"""
        print("Precomputing entity statistics for real-time lookup...")
        
        entity_stats = {}
        
        for entity_type in ['entity_card', 'entity_ip', 'entity_merchant', 'entity_email', 'entity_bin']:
            if entity_type in training_df.columns:
                stats = training_df.groupby(entity_type).agg({
                    'is_declined': ['count', 'mean', 'sum'],
                    'amount': ['mean', 'std'],
                    'issue_date': ['min', 'max']
                }).reset_index()
                
                # Flatten columns
                stats.columns = [f'{entity_type}' if col[1] == '' else f'{entity_type}_{col[0]}_{col[1]}' 
                               for col in stats.columns]
                
                entity_stats[entity_type] = stats
        
        # Save to files for fast loading
        os.makedirs(output_path, exist_ok=True)
        
        for entity_type, stats in entity_stats.items():
            stats.to_csv(os.path.join(output_path, f'{entity_type}_realtime_stats.csv'), index=False)
        
        # Create lookup dictionaries
        lookup_dicts = {}
        for entity_type, stats in entity_stats.items():
            lookup_dict = {}
            for _, row in stats.iterrows():
                entity_value = row[entity_type]
                lookup_dict[entity_value] = {
                    'decline_rate': row[f'{entity_type}_is_declined_mean'],
                    'transaction_count': row[f'{entity_type}_is_declined_count'],
                    'avg_amount': row[f'{entity_type}_amount_mean']
                }
            lookup_dicts[entity_type] = lookup_dict
        
        # Save lookup dictionaries
        with open(os.path.join(output_path, 'entity_lookup_dicts.pkl'), 'wb') as f:
            pickle.dump(lookup_dicts, f)
        
        print(f"Entity statistics saved to {output_path}")
        
        return lookup_dicts

def create_training_graph_features(train_df, test_df, output_path):
    """Create graph features for training the decline prediction model"""
    print("\nCreating training graph features...")
    
    # Initialize feature engine
    engine = RealtimeGraphFeatureEngine({
        'max_card_history': 5,
        'max_ip_history': 10,
        'max_merchant_history': 20,
        'time_window_hours': 24,  # Larger window for training
        'use_redis': False
    })
    
    # Process training data
    print("Processing training data...")
    train_features = []
    
    # Process in batches to manage memory
    batch_size = 1000
    for i in range(0, len(train_df), batch_size):
        batch_df = train_df.iloc[i:i+batch_size]
        batch_trans_ids = batch_df['trans_id'].tolist()
        
        # Build graph for batch
        g, trans_to_idx = engine.build_lightweight_graph(train_df, batch_trans_ids)
        
        # Compute features
        batch_features = engine.compute_fast_neighborhood_features(
            g, trans_to_idx, batch_trans_ids, train_df
        )
        
        train_features.extend([batch_features.get(trans_id, {}) for trans_id in batch_trans_ids])
        
        if (i // batch_size + 1) % 10 == 0:
            print(f"  Processed {i + len(batch_df)} training transactions...")
    
    # Process test data
    print("Processing test data...")
    test_features = []
    
    # Combine train and test for graph context
    combined_df = pd.concat([train_df, test_df], ignore_index=True)
    
    for i in range(0, len(test_df), batch_size):
        batch_df = test_df.iloc[i:i+batch_size]
        batch_trans_ids = batch_df['trans_id'].tolist()
        
        # Adjust transaction IDs for combined dataframe
        adjusted_ids = [tid + len(train_df) for tid in batch_trans_ids]
        
        # Build graph
        g, trans_to_idx = engine.build_lightweight_graph(combined_df, adjusted_ids)
        
        # Compute features
        batch_features = engine.compute_fast_neighborhood_features(
            g, trans_to_idx, adjusted_ids, combined_df
        )
        
        test_features.extend([batch_features.get(trans_id, {}) for trans_id in adjusted_ids])
        
        if (i // batch_size + 1) % 10 == 0:
            print(f"  Processed {i + len(batch_df)} test transactions...")
    
    # Convert to DataFrames
    train_graph_features = pd.DataFrame(train_features)
    test_graph_features = pd.DataFrame(test_features)
    
    # Add transaction IDs
    train_graph_features['trans_id'] = train_df['trans_id'].values
    test_graph_features['trans_id'] = test_df['trans_id'].values
    
    # Save features
    os.makedirs(output_path, exist_ok=True)
    train_graph_features.to_csv(os.path.join(output_path, 'train_graph_features.csv'), index=False)
    test_graph_features.to_csv(os.path.join(output_path, 'test_graph_features.csv'), index=False)
    
    # Precompute entity statistics
    engine.precompute_entity_statistics(train_df, output_path)
    
    print(f"Graph features saved to {output_path}")
    
    return train_graph_features, test_graph_features

def main():
    # Paths
    data_path = '../data/processed'
    output_path = '../data/graph'
    
    # Load processed data
    print("Loading processed data...")
    train_df = pd.read_csv(os.path.join(data_path, 'train_decline_data.csv'))
    test_df = pd.read_csv(os.path.join(data_path, 'test_decline_data.csv'))
    
    # Convert dates
    for df in [train_df, test_df]:
        df['issue_date'] = pd.to_datetime(df['issue_date'])
    
    # Create graph features
    train_features, test_features = create_training_graph_features(train_df, test_df, output_path)
    
    print("\nReal-time graph feature generation complete!")
    print(f"Training features shape: {train_features.shape}")
    print(f"Test features shape: {test_features.shape}")

if __name__ == "__main__":
    main()