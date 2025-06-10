#!/usr/bin/env python3
"""
Redis-based caching system for real-time decline prediction
Optimizes performance by caching entity statistics and features
"""

import os
import sys
import json
import pickle
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import hashlib

import redis
import pandas as pd
import numpy as np
from dataclasses import dataclass
import logging

@dataclass
class CacheConfig:
    """Configuration for Redis cache"""
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    socket_timeout: float = 1.0
    connection_pool_kwargs: Dict = None
    
    # Cache TTL settings (in seconds)
    entity_stats_ttl: int = 3600 * 24  # 24 hours
    velocity_features_ttl: int = 3600  # 1 hour
    graph_features_ttl: int = 1800     # 30 minutes
    model_features_ttl: int = 300      # 5 minutes
    
    # Cache key prefixes
    entity_prefix: str = "entity"
    velocity_prefix: str = "velocity"
    graph_prefix: str = "graph"
    model_prefix: str = "model"

class DeclineCacheManager:
    """High-performance cache manager for decline prediction"""
    
    def __init__(self, config: CacheConfig = None):
        self.config = config or CacheConfig()
        self.redis_client = None
        self.local_cache = {}
        self.local_cache_timestamps = {}
        self.logger = logging.getLogger(__name__)
        
        # Connection retry settings
        self.max_retries = 3
        self.retry_delay = 0.1
        
        self._connect()
    
    def _connect(self):
        """Connect to Redis with retry logic"""
        try:
            pool = redis.ConnectionPool(
                host=self.config.host,
                port=self.config.port,
                db=self.config.db,
                password=self.config.password,
                socket_timeout=self.config.socket_timeout,
                **(self.config.connection_pool_kwargs or {})
            )
            
            self.redis_client = redis.Redis(connection_pool=pool)
            
            # Test connection
            self.redis_client.ping()
            self.logger.info("Connected to Redis cache")
            
        except Exception as e:
            self.logger.warning(f"Redis connection failed: {e}. Using local cache.")
            self.redis_client = None
    
    def _get_key(self, prefix: str, identifier: str) -> str:
        """Generate cache key"""
        return f"{prefix}:{identifier}"
    
    def _serialize_value(self, value: Any) -> bytes:
        """Serialize value for Redis storage"""
        if isinstance(value, (dict, list)):
            return json.dumps(value, default=str).encode()
        elif isinstance(value, (np.ndarray, pd.DataFrame)):
            return pickle.dumps(value)
        else:
            return str(value).encode()
    
    def _deserialize_value(self, value: bytes, value_type: str = 'json') -> Any:
        """Deserialize value from Redis"""
        try:
            if value_type == 'json':
                return json.loads(value.decode())
            elif value_type == 'pickle':
                return pickle.loads(value)
            else:
                return value.decode()
        except:
            return None
    
    def _redis_get(self, key: str, value_type: str = 'json') -> Optional[Any]:
        """Get value from Redis with retry logic"""
        if not self.redis_client:
            return None
        
        for attempt in range(self.max_retries):
            try:
                value = self.redis_client.get(key)
                if value is None:
                    return None
                return self._deserialize_value(value, value_type)
            except Exception as e:
                if attempt == self.max_retries - 1:
                    self.logger.warning(f"Redis get failed after {self.max_retries} attempts: {e}")
                    return None
                time.sleep(self.retry_delay)
        return None
    
    def _redis_set(self, key: str, value: Any, ttl: int, value_type: str = 'json') -> bool:
        """Set value in Redis with retry logic"""
        if not self.redis_client:
            return False
        
        for attempt in range(self.max_retries):
            try:
                serialized_value = self._serialize_value(value)
                return self.redis_client.setex(key, ttl, serialized_value)
            except Exception as e:
                if attempt == self.max_retries - 1:
                    self.logger.warning(f"Redis set failed after {self.max_retries} attempts: {e}")
                    return False
                time.sleep(self.retry_delay)
        return False
    
    def _local_cache_get(self, key: str) -> Optional[Any]:
        """Get value from local cache with TTL check"""
        if key not in self.local_cache:
            return None
        
        # Check TTL
        timestamp = self.local_cache_timestamps.get(key, 0)
        if time.time() - timestamp > 300:  # 5 minute local TTL
            del self.local_cache[key]
            del self.local_cache_timestamps[key]
            return None
        
        return self.local_cache[key]
    
    def _local_cache_set(self, key: str, value: Any):
        """Set value in local cache"""
        self.local_cache[key] = value
        self.local_cache_timestamps[key] = time.time()
        
        # Cleanup old entries
        if len(self.local_cache) > 1000:
            oldest_key = min(self.local_cache_timestamps.keys(), 
                           key=lambda k: self.local_cache_timestamps[k])
            del self.local_cache[oldest_key]
            del self.local_cache_timestamps[oldest_key]
    
    # Entity Statistics Caching
    def get_entity_stats(self, entity_type: str, entity_value: str) -> Optional[Dict]:
        """Get entity statistics from cache"""
        key = self._get_key(self.config.entity_prefix, f"{entity_type}:{entity_value}")
        
        # Try local cache first
        result = self._local_cache_get(key)
        if result is not None:
            return result
        
        # Try Redis
        result = self._redis_get(key, 'json')
        if result is not None:
            self._local_cache_set(key, result)
            return result
        
        return None
    
    def set_entity_stats(self, entity_type: str, entity_value: str, stats: Dict):
        """Set entity statistics in cache"""
        key = self._get_key(self.config.entity_prefix, f"{entity_type}:{entity_value}")
        
        # Set in local cache
        self._local_cache_set(key, stats)
        
        # Set in Redis
        self._redis_set(key, stats, self.config.entity_stats_ttl, 'json')
    
    def bulk_set_entity_stats(self, entity_stats: Dict[str, Dict]):
        """Bulk set entity statistics"""
        if not self.redis_client:
            return
        
        try:
            pipe = self.redis_client.pipeline()
            for entity_key, stats in entity_stats.items():
                key = self._get_key(self.config.entity_prefix, entity_key)
                serialized = self._serialize_value(stats)
                pipe.setex(key, self.config.entity_stats_ttl, serialized)
            pipe.execute()
            self.logger.info(f"Bulk cached {len(entity_stats)} entity statistics")
        except Exception as e:
            self.logger.warning(f"Bulk entity stats caching failed: {e}")
    
    # Velocity Features Caching
    def get_velocity_features(self, entity_type: str, entity_value: str) -> Optional[Dict]:
        """Get velocity features from cache"""
        key = self._get_key(self.config.velocity_prefix, f"{entity_type}:{entity_value}")
        
        # Try local cache first
        result = self._local_cache_get(key)
        if result is not None:
            return result
        
        # Try Redis
        result = self._redis_get(key, 'json')
        if result is not None:
            self._local_cache_set(key, result)
            return result
        
        return None
    
    def set_velocity_features(self, entity_type: str, entity_value: str, features: Dict):
        """Set velocity features in cache"""
        key = self._get_key(self.config.velocity_prefix, f"{entity_type}:{entity_value}")
        
        # Add timestamp for freshness check
        features_with_timestamp = {
            **features,
            'cached_at': time.time()
        }
        
        # Set in local cache
        self._local_cache_set(key, features_with_timestamp)
        
        # Set in Redis
        self._redis_set(key, features_with_timestamp, self.config.velocity_features_ttl, 'json')
    
    # Graph Features Caching
    def get_graph_features(self, transaction_hash: str) -> Optional[Dict]:
        """Get graph features for transaction"""
        key = self._get_key(self.config.graph_prefix, transaction_hash)
        
        # Try local cache first
        result = self._local_cache_get(key)
        if result is not None:
            return result
        
        # Try Redis
        result = self._redis_get(key, 'json')
        if result is not None:
            self._local_cache_set(key, result)
            return result
        
        return None
    
    def set_graph_features(self, transaction_hash: str, features: Dict):
        """Set graph features for transaction"""
        key = self._get_key(self.config.graph_prefix, transaction_hash)
        
        # Add timestamp
        features_with_timestamp = {
            **features,
            'cached_at': time.time()
        }
        
        # Set in local cache
        self._local_cache_set(key, features_with_timestamp)
        
        # Set in Redis
        self._redis_set(key, features_with_timestamp, self.config.graph_features_ttl, 'json')
    
    # Model Features Caching
    def get_model_features(self, feature_hash: str) -> Optional[Dict]:
        """Get preprocessed model features"""
        key = self._get_key(self.config.model_prefix, feature_hash)
        
        # Try local cache first
        result = self._local_cache_get(key)
        if result is not None:
            return result
        
        # Try Redis
        result = self._redis_get(key, 'json')
        if result is not None:
            self._local_cache_set(key, result)
            return result
        
        return None
    
    def set_model_features(self, feature_hash: str, features: Dict):
        """Set preprocessed model features"""
        key = self._get_key(self.config.model_prefix, feature_hash)
        
        # Set in local cache
        self._local_cache_set(key, features)
        
        # Set in Redis
        self._redis_set(key, features, self.config.model_features_ttl, 'json')
    
    # Utility Methods
    def generate_transaction_hash(self, transaction_data: Dict) -> str:
        """Generate hash for transaction caching"""
        # Create hash from key transaction attributes
        key_attrs = ['card_number', 'amount', 'merchant_id', 'timestamp']
        hash_string = '|'.join([str(transaction_data.get(attr, '')) for attr in key_attrs])
        return hashlib.md5(hash_string.encode()).hexdigest()
    
    def warm_cache_from_database(self, entity_stats_df: pd.DataFrame):
        """Warm cache with entity statistics from database"""
        print("Warming cache with entity statistics...")
        
        entity_stats = {}
        
        # Process each entity type
        for entity_type in ['entity_card', 'entity_ip', 'entity_merchant', 'entity_email']:
            if entity_type in entity_stats_df.columns:
                for _, row in entity_stats_df.iterrows():
                    entity_value = row[entity_type]
                    stats = {
                        'decline_rate': row.get(f'{entity_type}_decline_rate', 0.0),
                        'transaction_count': row.get(f'{entity_type}_transaction_count', 0),
                        'avg_amount': row.get(f'{entity_type}_avg_amount', 0.0),
                        'last_updated': time.time()
                    }
                    entity_stats[f"{entity_type}:{entity_value}"] = stats
        
        # Bulk set in cache
        self.bulk_set_entity_stats(entity_stats)
        print(f"Cache warmed with {len(entity_stats)} entity statistics")
    
    def get_cache_stats(self) -> Dict:
        """Get cache performance statistics"""
        stats = {
            'redis_connected': self.redis_client is not None,
            'local_cache_size': len(self.local_cache),
            'redis_info': None
        }
        
        if self.redis_client:
            try:
                info = self.redis_client.info()
                stats['redis_info'] = {
                    'used_memory': info.get('used_memory_human'),
                    'connected_clients': info.get('connected_clients'),
                    'keyspace_hits': info.get('keyspace_hits'),
                    'keyspace_misses': info.get('keyspace_misses')
                }
            except:
                pass
        
        return stats
    
    def clear_cache(self, pattern: str = None):
        """Clear cache entries"""
        # Clear local cache
        if pattern:
            keys_to_remove = [k for k in self.local_cache.keys() if pattern in k]
            for key in keys_to_remove:
                del self.local_cache[key]
                del self.local_cache_timestamps[key]
        else:
            self.local_cache.clear()
            self.local_cache_timestamps.clear()
        
        # Clear Redis cache
        if self.redis_client and pattern:
            try:
                keys = self.redis_client.keys(f"*{pattern}*")
                if keys:
                    self.redis_client.delete(*keys)
            except:
                pass
    
    def close(self):
        """Close cache connections"""
        if self.redis_client:
            try:
                self.redis_client.close()
            except:
                pass

def main():
    """Example usage and testing"""
    
    # Initialize cache manager
    cache_config = CacheConfig(
        host="localhost",
        port=6379,
        entity_stats_ttl=3600 * 24,
        velocity_features_ttl=3600
    )
    
    cache_manager = DeclineCacheManager(cache_config)
    
    # Test entity stats caching
    test_stats = {
        'decline_rate': 0.15,
        'transaction_count': 100,
        'avg_amount': 75.50
    }
    
    cache_manager.set_entity_stats('card', 'test_card_123', test_stats)
    retrieved_stats = cache_manager.get_entity_stats('card', 'test_card_123')
    
    print(f"Cached stats: {test_stats}")
    print(f"Retrieved stats: {retrieved_stats}")
    
    # Test velocity features
    velocity_features = {
        'hours_since_last': 2.5,
        'txn_per_day': 3.2,
        'amount_ratio': 1.1
    }
    
    cache_manager.set_velocity_features('card', 'test_card_123', velocity_features)
    retrieved_velocity = cache_manager.get_velocity_features('card', 'test_card_123')
    
    print(f"Cached velocity: {velocity_features}")
    print(f"Retrieved velocity: {retrieved_velocity}")
    
    # Get cache statistics
    cache_stats = cache_manager.get_cache_stats()
    print(f"Cache stats: {cache_stats}")
    
    # Close cache
    cache_manager.close()

if __name__ == "__main__":
    main()