#!/usr/bin/env python3
"""
Train RGTAN model for fraud/chargeback prediction
Adapted from main RGTAN implementation for fraud detection use case
"""

import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score, precision_recall_curve
from sklearn.preprocessing import StandardScaler
import pickle
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from methods.rgtan.rgtan_model import RGTAN
from methods.common.transformer_conv import TransformerConv

class FraudRGTAN(nn.Module):
    """RGTAN adapted for fraud detection with chargeback data"""
    
    def __init__(self, in_feats, hidden_dim, n_layers, n_classes, heads, 
                 activation, drop, device, nei_feats_dim=None):
        super(FraudRGTAN, self).__init__()
        self.n_layers = n_layers
        self.device = device
        self.hidden_dim = hidden_dim
        
        # Input projection
        self.input_proj = nn.Linear(in_feats, hidden_dim)
        
        # Neighborhood feature attention
        if nei_feats_dim:
            # Ensure embedding dimension is divisible by number of heads
            # If not, project to a compatible dimension first
            if nei_feats_dim % 4 != 0:
                # Project to nearest dimension divisible by 4
                projected_dim = ((nei_feats_dim // 4) + 1) * 4
                self.nei_input_proj = nn.Linear(nei_feats_dim, projected_dim)
                self.nei_att = nn.MultiheadAttention(
                    projected_dim, num_heads=4, dropout=drop[0], batch_first=True
                )
                self.nei_proj = nn.Linear(projected_dim, hidden_dim)
            else:
                self.nei_input_proj = None
                self.nei_att = nn.MultiheadAttention(
                    nei_feats_dim, num_heads=4, dropout=drop[0], batch_first=True
                )
                self.nei_proj = nn.Linear(nei_feats_dim, hidden_dim)
        else:
            self.nei_att = None
            self.nei_input_proj = None
        
        # Graph convolution layers
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for i in range(n_layers):
            in_hidden = hidden_dim
            out_hidden = hidden_dim
            
            self.layers.append(TransformerConv(
                in_hidden, out_hidden // heads[i], heads[i]
            ))
            self.norms.append(nn.LayerNorm(out_hidden))
        
        # Output layers
        self.dropout = nn.Dropout(drop[-1])
        self.activation = activation
        self.output = nn.Linear(hidden_dim, n_classes)
        
    def forward(self, blocks, x, nei_features=None):
        # Input projection
        h = self.input_proj(x)
        
        # Add neighborhood features if available
        if self.nei_att is not None and nei_features is not None:
            # Project to compatible dimension if needed
            if self.nei_input_proj is not None:
                nei_features = self.nei_input_proj(nei_features)
            
            nei_emb, _ = self.nei_att(nei_features, nei_features, nei_features)
            nei_emb = nei_emb.mean(dim=1)  # Average over sequence
            nei_emb = self.nei_proj(nei_emb)
            h = h + nei_emb
        
        # Graph convolution layers
        for i in range(self.n_layers):
            h = self.layers[i](blocks[i], h)
            h = self.activation(h)
            h = self.norms[i](h)
            h = self.dropout(h)
        
        # Output
        return self.output(h)

def prepare_features(train_df, test_df, train_neigh, test_neigh):
    """Prepare features for model training"""
    print("\nPreparing features...")
    
    # Select feature columns
    numeric_cols = ['amount_log', 'time_to_capture', 'hour', 'day_of_week', 
                   'is_weekend', 'is_night', 'is_round_amount', 'is_high_amount']
    
    categorical_cols = [col for col in train_df.columns if col.endswith('_encoded')]
    
    # Combine numeric and categorical features
    feature_cols = numeric_cols + categorical_cols
    
    # Extract features
    train_features = train_df[feature_cols].values
    test_features = test_df[feature_cols].values
    
    # Merge neighborhood features
    neigh_cols = [col for col in train_neigh.columns if col != 'trans_id']
    train_neigh_feat = train_neigh[neigh_cols].values
    test_neigh_feat = test_neigh[neigh_cols].values
    
    # Standardize features
    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_features)
    test_features = scaler.transform(test_features)
    
    neigh_scaler = StandardScaler()
    train_neigh_feat = neigh_scaler.fit_transform(train_neigh_feat)
    test_neigh_feat = neigh_scaler.transform(test_neigh_feat)
    
    # Get labels
    train_labels = train_df['is_fraud'].values
    test_labels = test_df['is_fraud'].values
    
    print(f"Feature dimensions: {train_features.shape[1]}")
    print(f"Neighborhood dimensions: {train_neigh_feat.shape[1]}")
    
    return (train_features, test_features, train_neigh_feat, test_neigh_feat, 
            train_labels, test_labels, scaler, neigh_scaler)

def create_mini_batch_sampler(g, train_idx, batch_size=512):
    """Create mini-batch sampler for training"""
    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
    dataloader = dgl.dataloading.DataLoader(
        g, train_idx, sampler,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0
    )
    return dataloader

def train_epoch(model, dataloader, features, nei_features, labels, optimizer, device):
    """Train one epoch"""
    model.train()
    total_loss = 0
    
    # Class weights for imbalanced data
    pos_weight = (labels == 0).sum() / (labels == 1).sum()
    
    for input_nodes, output_nodes, blocks in dataloader:
        blocks = [b.to(device) for b in blocks]
        
        # Get features for batch
        batch_features = features[input_nodes].to(device)
        batch_nei_features = nei_features[input_nodes].unsqueeze(1).to(device) if nei_features is not None else None
        batch_labels = labels[output_nodes].to(device)
        
        # Forward pass
        logits = model(blocks, batch_features, batch_nei_features)
        
        # Weighted loss
        loss = F.cross_entropy(logits, batch_labels, weight=torch.tensor([1.0, pos_weight]).to(device))
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

def evaluate(model, g, features, nei_features, labels, idx, device, batch_size=1024):
    """Evaluate model performance"""
    model.eval()
    
    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
    dataloader = dgl.dataloading.DataLoader(
        g, idx, sampler,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0
    )
    
    all_logits = []
    all_labels = []
    
    with torch.no_grad():
        for input_nodes, output_nodes, blocks in dataloader:
            blocks = [b.to(device) for b in blocks]
            
            batch_features = features[input_nodes].to(device)
            batch_nei_features = nei_features[input_nodes].unsqueeze(1).to(device) if nei_features is not None else None
            batch_labels = labels[output_nodes]
            
            logits = model(blocks, batch_features, batch_nei_features)
            
            all_logits.append(logits.cpu())
            all_labels.append(batch_labels)
    
    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    # Calculate metrics
    probs = F.softmax(all_logits, dim=1)[:, 1].numpy()
    preds = all_logits.argmax(dim=1).numpy()
    labels_np = all_labels.numpy()
    
    auc = roc_auc_score(labels_np, probs)
    f1 = f1_score(labels_np, preds, average='macro')
    ap = average_precision_score(labels_np, probs)
    
    # Calculate precision at different recall levels
    precision, recall, _ = precision_recall_curve(labels_np, probs)
    
    # Find precision at 50% recall
    idx_50_recall = np.argmin(np.abs(recall - 0.5))
    precision_at_50_recall = precision[idx_50_recall]
    
    return auc, f1, ap, precision_at_50_recall, probs

def train_fraud_model(config):
    """Main training function"""
    print("Starting RGTAN training for fraud detection...")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # GPU memory management
    if torch.cuda.is_available():
        print(f"GPU memory before loading: {torch.cuda.memory_allocated()/1024**2:.1f}MB")
        torch.cuda.empty_cache()
    
    # Load data
    data_path = config['data_path']
    print(f"Loading data from: {data_path}")
    
    try:
        train_df = pd.read_csv(os.path.join(data_path, 'processed/train_transactions.csv'))
        print(f"Train data loaded: {len(train_df):,} rows")
        
        test_df = pd.read_csv(os.path.join(data_path, 'processed/test_transactions.csv'))
        print(f"Test data loaded: {len(test_df):,} rows")
        
        train_neigh = pd.read_csv(os.path.join(data_path, 'graph/train_neigh_features.csv'))
        print(f"Train neighborhood features loaded: {train_neigh.shape}")
        
        test_neigh = pd.read_csv(os.path.join(data_path, 'graph/test_neigh_features.csv'))
        print(f"Test neighborhood features loaded: {test_neigh.shape}")
    except Exception as e:
        print(f"Error loading CSV files: {e}")
        return 0
    
    # Load graph
    try:
        print("Loading transaction graph...")
        graphs, _ = dgl.load_graphs(os.path.join(data_path, 'graph/transaction_graph.dgl'))
        g = graphs[0]
        print(f"Graph loaded: {g.num_nodes():,} nodes, {g.num_edges():,} edges")
        
        # Check if graph size matches data size
        total_transactions = len(train_df) + len(test_df)
        if g.num_nodes() != total_transactions:
            print(f"WARNING: Graph nodes ({g.num_nodes():,}) != Total transactions ({total_transactions:,})")
            print("The graph was created with sampled data. Adjusting data to match graph size...")
            
            # Use only the transactions that match the graph size
            if g.num_nodes() == len(train_neigh) + len(test_neigh):
                # Graph matches the neighborhood features (sampled data)
                print("Using sampled data that matches the graph...")
                
                # Load the combined transactions file that matches the graph
                combined_path = os.path.join(data_path, 'graph/combined_transactions.csv')
                if os.path.exists(combined_path):
                    combined_df = pd.read_csv(combined_path)
                    train_size = len(train_neigh)
                    train_df = combined_df.iloc[:train_size].copy()
                    test_df = combined_df.iloc[train_size:].copy()
                    print(f"Adjusted - Train: {len(train_df):,}, Test: {len(test_df):,}")
                else:
                    print("ERROR: Combined transactions file not found!")
                    return 0
            else:
                print("ERROR: Graph size doesn't match any expected data size!")
                return 0
        
        if torch.cuda.is_available():
            print(f"GPU memory after loading graph: {torch.cuda.memory_allocated()/1024**2:.1f}MB")
    except Exception as e:
        print(f"Error loading graph: {e}")
        return 0
    
    # Prepare features
    try:
        print("Preparing features...")
        (train_features, test_features, train_neigh_feat, test_neigh_feat,
         train_labels, test_labels, scaler, neigh_scaler) = prepare_features(
            train_df, test_df, train_neigh, test_neigh
        )
        print("Features prepared successfully")
    except Exception as e:
        print(f"Error preparing features: {e}")
        return 0
    
    # Combine features for graph
    try:
        print("Combining features...")
        all_features = np.vstack([train_features, test_features])
        all_nei_features = np.vstack([train_neigh_feat, test_neigh_feat])
        all_labels = np.concatenate([train_labels, test_labels])
        print(f"Combined features shape: {all_features.shape}")
        print(f"Combined neighbor features shape: {all_nei_features.shape}")
    except Exception as e:
        print(f"Error combining features: {e}")
        return 0
    
    # Convert to tensors with memory management
    try:
        print("Converting to tensors...")
        features = torch.FloatTensor(all_features)
        nei_features = torch.FloatTensor(all_nei_features)
        labels = torch.LongTensor(all_labels)
        
        if torch.cuda.is_available():
            print(f"GPU memory after tensor creation: {torch.cuda.memory_allocated()/1024**2:.1f}MB")
        
        print("Tensors created successfully")
    except Exception as e:
        print(f"Error creating tensors: {e}")
        return 0
    
    # Create train/test indices
    train_idx = torch.arange(len(train_df))
    test_idx = torch.arange(len(train_df), len(train_df) + len(test_df))
    print(f"Train indices: {len(train_idx):,}, Test indices: {len(test_idx):,}")
    
    # Final validation: ensure all sizes match
    total_data = len(train_df) + len(test_df)
    print(f"\nFinal validation:")
    print(f"  Graph nodes: {g.num_nodes():,}")
    print(f"  Total data: {total_data:,}")
    print(f"  Features: {all_features.shape[0]:,}")
    print(f"  Neighbor features: {all_nei_features.shape[0]:,}")
    print(f"  Max train index: {train_idx.max()}")
    print(f"  Max test index: {test_idx.max()}")
    
    if not (g.num_nodes() == total_data == all_features.shape[0] == all_nei_features.shape[0]):
        print("ERROR: Size mismatch detected!")
        print("This will cause index out-of-bounds errors during training.")
        return 0
    else:
        print("✓ All sizes match - proceeding with training")
    
    # Initialize model with smaller batch size for memory
    config['batch_size'] = min(config['batch_size'], 128)  # Reduce batch size
    print(f"Using batch size: {config['batch_size']}")
    
    try:
        print("Initializing model...")
        model = FraudRGTAN(
            in_feats=features.shape[1],
            hidden_dim=config['hidden_dim'],
            n_layers=config['n_layers'],
            n_classes=2,
            heads=config['heads'],
            activation=nn.PReLU(),
            drop=config['dropout'],
            device=device,
            nei_feats_dim=nei_features.shape[1]
        )
        print("Model initialized successfully")
        
        # Move to device with memory management
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        model = model.to(device)
        print(f"Model moved to {device}")
        
        if torch.cuda.is_available():
            print(f"GPU memory after model creation: {torch.cuda.memory_allocated()/1024**2:.1f}MB")
            
    except Exception as e:
        print(f"Error initializing model: {e}")
        return 0
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    
    # Training
    best_auc = 0
    best_model = None
    patience_counter = 0
    
    train_dataloader = create_mini_batch_sampler(g, train_idx, config['batch_size'])
    
    for epoch in range(config['epochs']):
        # Train
        train_loss = train_epoch(model, train_dataloader, features, nei_features, labels, optimizer, device)
        
        # Evaluate
        train_auc, train_f1, train_ap, train_p50, _ = evaluate(
            model, g, features, nei_features, labels, train_idx, device
        )
        test_auc, test_f1, test_ap, test_p50, test_probs = evaluate(
            model, g, features, nei_features, labels, test_idx, device
        )
        
        print(f"Epoch {epoch+1}/{config['epochs']}")
        print(f"  Train - Loss: {train_loss:.4f}, AUC: {train_auc:.4f}, F1: {train_f1:.4f}")
        print(f"  Test  - AUC: {test_auc:.4f}, F1: {test_f1:.4f}, AP: {test_ap:.4f}, P@50R: {test_p50:.4f}")
        
        # Early stopping
        if test_auc > best_auc:
            best_auc = test_auc
            best_model = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= config['patience']:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Save model and results
    save_model_and_results(best_model, model, test_probs, test_labels, test_df, 
                          scaler, neigh_scaler, config)
    
    return best_auc

def save_model_and_results(best_model_state, model, test_probs, test_labels, test_df, 
                          scaler, neigh_scaler, config):
    """Save trained model and evaluation results"""
    print("\nSaving model and results...")
    
    # Create model directory
    model_dir = os.path.join(config['output_path'], 'models')
    os.makedirs(model_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(model_dir, 'fraud_rgtan_model.pt')
    torch.save({
        'model_state_dict': best_model_state,
        'model_config': {
            'in_feats': model.input_proj.in_features,
            'hidden_dim': model.hidden_dim,
            'n_layers': model.n_layers,
            'n_classes': 2,
            'heads': [4, 4],  # From config
            'nei_feats_dim': model.nei_proj.in_features if model.nei_att else None
        },
        'scaler': scaler,
        'neigh_scaler': neigh_scaler,
        'training_date': datetime.now().isoformat()
    }, model_path)
    
    # Save predictions
    results_dir = os.path.join(config['output_path'], 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Create results dataframe
    results_df = test_df.copy()
    results_df['fraud_score'] = test_probs
    results_df['true_label'] = test_labels
    results_df['predicted_fraud'] = (test_probs > 0.5).astype(int)
    
    # Sort by fraud score
    results_df = results_df.sort_values('fraud_score', ascending=False)
    
    # Save top fraud predictions
    results_df.head(1000).to_csv(
        os.path.join(results_dir, 'top_fraud_predictions.csv'), 
        index=False
    )
    
    # Calculate and save metrics at different thresholds
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    threshold_metrics = []
    
    for threshold in thresholds:
        preds = (test_probs > threshold).astype(int)
        tp = ((preds == 1) & (test_labels == 1)).sum()
        fp = ((preds == 1) & (test_labels == 0)).sum()
        fn = ((preds == 0) & (test_labels == 1)).sum()
        tn = ((preds == 0) & (test_labels == 0)).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        threshold_metrics.append({
            'threshold': threshold,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'flagged_count': tp + fp,
            'flagged_rate': (tp + fp) / len(test_labels)
        })
    
    metrics_df = pd.DataFrame(threshold_metrics)
    metrics_df.to_csv(os.path.join(results_dir, 'threshold_metrics.csv'), index=False)
    
    print(f"Model saved to {model_path}")
    print(f"Results saved to {results_dir}")

def main():
    # Configuration with memory-safe defaults
    config = {
        'data_path': '/home/development/affdf/fraud_prevention/data',
        'output_path': '/home/development/affdf/fraud_prevention',
        'hidden_dim': 128,  # Reduced from 256 for memory
        'n_layers': 2,
        'heads': [4, 4],
        'dropout': [0.2, 0.1, 0.1],
        'lr': 0.001,
        'weight_decay': 1e-4,
        'batch_size': 64,  # Reduced from 512 for memory
        'epochs': 10,  # Reduced for faster testing
        'patience': 5   # Reduced for faster testing
    }
    
    # Train model
    best_auc = train_fraud_model(config)
    
    print(f"\nTraining complete! Best test AUC: {best_auc:.4f}")

if __name__ == "__main__":
    main()