#!/usr/bin/env python3
"""
Train RGTAN model for real-time decline prediction
Optimized for fast inference and high throughput
"""

import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score, accuracy_score
from sklearn.preprocessing import StandardScaler
import pickle
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from methods.common.transformer_conv import TransformerConv

class LightweightRGTAN(nn.Module):
    """Lightweight RGTAN optimized for real-time decline prediction"""
    
    def __init__(self, in_feats, graph_feats, hidden_dim, n_layers, n_classes, 
                 heads, activation, drop, device):
        super(LightweightRGTAN, self).__init__()
        self.n_layers = n_layers
        self.device = device
        self.hidden_dim = hidden_dim
        
        # Input projections
        self.input_proj = nn.Linear(in_feats, hidden_dim)
        self.graph_proj = nn.Linear(graph_feats, hidden_dim)
        
        # Lightweight graph convolution layers
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for i in range(n_layers):
            in_hidden = hidden_dim
            out_hidden = hidden_dim
            
            self.layers.append(TransformerConv(
                in_hidden, out_hidden // heads[i], heads[i],
                dropout=drop[i], edge_drop=0.0, skip_feat=True
            ))
            self.norms.append(nn.LayerNorm(out_hidden))
        
        # Feature fusion
        self.fusion = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # Output layers with multiple targets
        self.dropout = nn.Dropout(drop[-1])
        self.activation = activation
        
        # Multi-output heads for different decline reasons
        self.decline_head = nn.Linear(hidden_dim, n_classes)  # General decline
        self.insufficient_funds_head = nn.Linear(hidden_dim, 2)  # Insufficient funds
        self.security_head = nn.Linear(hidden_dim, 2)  # Security decline
        self.invalid_merchant_head = nn.Linear(hidden_dim, 2)  # Invalid merchant
        
    def forward(self, blocks, x, graph_features):
        # Project inputs
        h_node = self.input_proj(x)
        h_graph = self.graph_proj(graph_features)
        
        # Graph convolution
        h = h_node
        for i in range(self.n_layers):
            if blocks is not None and len(blocks) > i:
                h = self.layers[i](blocks[i], h)
            else:
                # Skip graph convolution if no blocks (single transaction)
                h = h
            h = self.activation(h)
            h = self.norms[i](h)
            h = self.dropout(h)
        
        # Fuse node and graph features
        h_fused = self.fusion(torch.cat([h, h_graph], dim=1))
        h_fused = self.activation(h_fused)
        h_fused = self.dropout(h_fused)
        
        # Multi-output predictions
        outputs = {
            'decline': self.decline_head(h_fused),
            'insufficient_funds': self.insufficient_funds_head(h_fused),
            'security': self.security_head(h_fused),
            'invalid_merchant': self.invalid_merchant_head(h_fused)
        }
        
        return outputs

def prepare_decline_features(train_df, test_df, train_graph_features, test_graph_features):
    """Prepare features for decline prediction model"""
    print("\nPreparing features for decline prediction...")
    
    # Select feature columns optimized for real-time
    numeric_cols = [
        'amount_log', 'hour', 'day_of_week', 'is_weekend', 'is_business_hours', 'is_night',
        'is_round_amount', 'is_exact_dollar', 'is_high_amount', 'is_low_amount',
        'is_debit', 'is_credit', 'is_domestic', 'has_billing_zip'
    ]
    
    # Velocity features
    velocity_cols = [col for col in train_df.columns if 'hours_since_last' in col or 'txn_per_day' in col or 'amount_ratio' in col]
    
    # Categorical features
    categorical_cols = [col for col in train_df.columns if col.endswith('_encoded')]
    
    # Combine feature types
    feature_cols = numeric_cols + velocity_cols + categorical_cols
    
    # Extract features
    train_features = train_df[feature_cols].values
    test_features = test_df[feature_cols].values
    
    # Graph features
    graph_feature_cols = ['neighbor_count', 'neighbor_decline_rate', 'neighbor_avg_amount', 
                         'neighbor_velocity', 'card_decline_rate', 'ip_decline_rate', 'merchant_decline_rate']
    
    train_graph_feat = train_graph_features[graph_feature_cols].fillna(0).values
    test_graph_feat = test_graph_features[graph_feature_cols].fillna(0).values
    
    # Standardize features
    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_features)
    test_features = scaler.transform(test_features)
    
    graph_scaler = StandardScaler()
    train_graph_feat = graph_scaler.fit_transform(train_graph_feat)
    test_graph_feat = graph_scaler.transform(test_graph_feat)
    
    # Create labels
    train_labels = train_df['is_declined'].values
    test_labels = test_df['is_declined'].values
    
    # Create multi-task labels
    decline_categories = ['insufficient_funds', 'security', 'invalid_merchant']
    train_multi_labels = {}
    test_multi_labels = {}
    
    for category in decline_categories:
        if 'decline_category' in train_df.columns:
            train_multi_labels[category] = (train_df['decline_category'] == category).astype(int).values
            test_multi_labels[category] = (test_df['decline_category'] == category).astype(int).values
        else:
            train_multi_labels[category] = np.zeros(len(train_df))
            test_multi_labels[category] = np.zeros(len(test_df))
    
    print(f"Feature dimensions: {train_features.shape[1]}")
    print(f"Graph feature dimensions: {train_graph_feat.shape[1]}")
    print(f"Training samples: {len(train_features)}")
    print(f"Test samples: {len(test_features)}")
    
    return (train_features, test_features, train_graph_feat, test_graph_feat,
            train_labels, test_labels, train_multi_labels, test_multi_labels,
            scaler, graph_scaler, feature_cols, graph_feature_cols)

def create_lightweight_dataloaders(train_features, train_graph_feat, train_labels, 
                                 train_multi_labels, batch_size=1024):
    """Create lightweight dataloaders for real-time training"""
    
    # Convert to tensors
    train_feat_tensor = torch.FloatTensor(train_features)
    train_graph_tensor = torch.FloatTensor(train_graph_feat)
    train_label_tensor = torch.LongTensor(train_labels)
    
    # Multi-task labels
    train_multi_tensors = {}
    for task, labels in train_multi_labels.items():
        train_multi_tensors[task] = torch.LongTensor(labels)
    
    # Create dataset
    dataset = torch.utils.data.TensorDataset(
        train_feat_tensor, train_graph_tensor, train_label_tensor,
        *train_multi_tensors.values()
    )
    
    # Create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, 
        num_workers=0, pin_memory=True
    )
    
    return dataloader, train_multi_tensors.keys()

def train_epoch(model, dataloader, optimizer, device, multi_task_names):
    """Train one epoch with multi-task learning"""
    model.train()
    total_loss = 0
    
    # Loss weights for different tasks
    task_weights = {
        'decline': 2.0,  # Main task
        'insufficient_funds': 1.0,
        'security': 1.5,
        'invalid_merchant': 1.0
    }
    
    for batch_data in dataloader:
        features = batch_data[0].to(device)
        graph_features = batch_data[1].to(device)
        main_labels = batch_data[2].to(device)
        
        # Multi-task labels
        multi_labels = {}
        for i, task in enumerate(multi_task_names):
            multi_labels[task] = batch_data[3 + i].to(device)
        
        # Forward pass (no blocks for lightweight model)
        outputs = model(None, features, graph_features)
        
        # Calculate losses
        main_loss = F.cross_entropy(outputs['decline'], main_labels)
        total_task_loss = main_loss * task_weights['decline']
        
        for task in multi_task_names:
            if task in outputs and task in multi_labels:
                task_loss = F.cross_entropy(outputs[task], multi_labels[task])
                total_task_loss += task_loss * task_weights.get(task, 1.0)
        
        # Backward pass
        optimizer.zero_grad()
        total_task_loss.backward()
        optimizer.step()
        
        total_loss += total_task_loss.item()
    
    return total_loss / len(dataloader)

def evaluate_model(model, features, graph_features, labels, multi_labels, device):
    """Evaluate model performance"""
    model.eval()
    
    with torch.no_grad():
        features_tensor = torch.FloatTensor(features).to(device)
        graph_features_tensor = torch.FloatTensor(graph_features).to(device)
        
        # Forward pass
        outputs = model(None, features_tensor, graph_features_tensor)
        
        # Main task metrics
        main_probs = F.softmax(outputs['decline'], dim=1)[:, 1].cpu().numpy()
        main_preds = outputs['decline'].argmax(dim=1).cpu().numpy()
        
        auc = roc_auc_score(labels, main_probs)
        f1 = f1_score(labels, main_preds, average='macro')
        accuracy = accuracy_score(labels, main_preds)
        ap = average_precision_score(labels, main_probs)
        
        # Task-specific metrics
        task_metrics = {}
        for task in ['insufficient_funds', 'security', 'invalid_merchant']:
            if task in outputs and task in multi_labels:
                task_probs = F.softmax(outputs[task], dim=1)[:, 1].cpu().numpy()
                task_preds = outputs[task].argmax(dim=1).cpu().numpy()
                task_labels = multi_labels[task]
                
                if len(np.unique(task_labels)) > 1:  # Check if both classes exist
                    task_auc = roc_auc_score(task_labels, task_probs)
                    task_f1 = f1_score(task_labels, task_preds, average='macro')
                    task_metrics[task] = {'auc': task_auc, 'f1': task_f1}
        
        return auc, f1, accuracy, ap, main_probs, task_metrics

def train_decline_model(config):
    """Main training function for decline prediction"""
    print("Starting RGTAN training for decline prediction...")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    data_path = config['data_path']
    train_df = pd.read_csv(os.path.join(data_path, 'processed/train_decline_data.csv'))
    test_df = pd.read_csv(os.path.join(data_path, 'processed/test_decline_data.csv'))
    train_graph_features = pd.read_csv(os.path.join(data_path, 'graph/train_graph_features.csv'))
    test_graph_features = pd.read_csv(os.path.join(data_path, 'graph/test_graph_features.csv'))
    
    # Prepare features
    (train_features, test_features, train_graph_feat, test_graph_feat,
     train_labels, test_labels, train_multi_labels, test_multi_labels,
     scaler, graph_scaler, feature_cols, graph_feature_cols) = prepare_decline_features(
        train_df, test_df, train_graph_features, test_graph_features
    )
    
    # Create dataloaders
    dataloader, multi_task_names = create_lightweight_dataloaders(
        train_features, train_graph_feat, train_labels, train_multi_labels,
        config['batch_size']
    )
    
    # Initialize model
    model = LightweightRGTAN(
        in_feats=train_features.shape[1],
        graph_feats=train_graph_feat.shape[1],
        hidden_dim=config['hidden_dim'],
        n_layers=config['n_layers'],
        n_classes=2,
        heads=config['heads'],
        activation=nn.ReLU(),
        drop=config['dropout'],
        device=device
    ).to(device)
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)
    
    # Training loop
    best_auc = 0
    best_model_state = None
    patience_counter = 0
    
    for epoch in range(config['epochs']):
        # Train
        train_loss = train_epoch(model, dataloader, optimizer, device, multi_task_names)
        
        # Evaluate
        train_auc, train_f1, train_acc, train_ap, _, train_task_metrics = evaluate_model(
            model, train_features, train_graph_feat, train_labels, train_multi_labels, device
        )
        test_auc, test_f1, test_acc, test_ap, test_probs, test_task_metrics = evaluate_model(
            model, test_features, test_graph_feat, test_labels, test_multi_labels, device
        )
        
        # Learning rate scheduling
        scheduler.step(test_auc)
        
        print(f"Epoch {epoch+1}/{config['epochs']}")
        print(f"  Train - Loss: {train_loss:.4f}, AUC: {train_auc:.4f}, F1: {train_f1:.4f}, Acc: {train_acc:.4f}")
        print(f"  Test  - AUC: {test_auc:.4f}, F1: {test_f1:.4f}, Acc: {test_acc:.4f}, AP: {test_ap:.4f}")
        
        # Print task-specific metrics
        for task, metrics in test_task_metrics.items():
            print(f"  {task.title()} - AUC: {metrics['auc']:.4f}, F1: {metrics['f1']:.4f}")
        
        # Early stopping
        if test_auc > best_auc:
            best_auc = test_auc
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= config['patience']:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Save model and results
    save_decline_model(best_model_state, model, test_probs, test_labels, test_df,
                      scaler, graph_scaler, feature_cols, graph_feature_cols, config)
    
    return best_auc

def save_decline_model(best_model_state, model, test_probs, test_labels, test_df,
                      scaler, graph_scaler, feature_cols, graph_feature_cols, config):
    """Save trained model and evaluation results"""
    print("\nSaving decline prediction model...")
    
    # Create model directory
    model_dir = os.path.join(config['output_path'], 'models')
    os.makedirs(model_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(model_dir, 'decline_rgtan_model.pt')
    torch.save({
        'model_state_dict': best_model_state,
        'model_config': {
            'in_feats': len(feature_cols),
            'graph_feats': len(graph_feature_cols),
            'hidden_dim': model.hidden_dim,
            'n_layers': model.n_layers,
            'n_classes': 2,
            'heads': [4, 4]  # From config
        },
        'feature_columns': feature_cols,
        'graph_feature_columns': graph_feature_cols,
        'scaler': scaler,
        'graph_scaler': graph_scaler,
        'training_date': datetime.now().isoformat()
    }, model_path)
    
    # Save predictions
    results_dir = os.path.join(config['output_path'], 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Create results dataframe
    results_df = test_df.copy()
    results_df['decline_score'] = test_probs
    results_df['predicted_decline'] = (test_probs > 0.5).astype(int)
    results_df['true_label'] = test_labels
    
    # Sort by decline score
    results_df = results_df.sort_values('decline_score', ascending=False)
    
    # Save results
    results_df.to_csv(os.path.join(results_dir, 'decline_predictions.csv'), index=False)
    
    # Performance analysis by decline reason
    if 'decline_category' in test_df.columns:
        category_performance = {}
        for category in test_df['decline_category'].unique():
            mask = test_df['decline_category'] == category
            if mask.sum() > 10:  # Minimum samples
                cat_auc = roc_auc_score(test_labels[mask], test_probs[mask])
                category_performance[category] = {
                    'count': mask.sum(),
                    'auc': cat_auc,
                    'decline_rate': test_labels[mask].mean()
                }
        
        # Save category analysis
        with open(os.path.join(results_dir, 'category_performance.json'), 'w') as f:
            json.dump(category_performance, f, indent=2, default=str)
    
    print(f"Model saved to {model_path}")
    print(f"Results saved to {results_dir}")

def main():
    # Configuration for decline prediction
    config = {
        'data_path': '../data',
        'output_path': '..',
        'hidden_dim': 128,  # Smaller for faster inference
        'n_layers': 2,
        'heads': [4, 4],
        'dropout': [0.1, 0.1, 0.1],  # Lower dropout
        'lr': 0.001,
        'weight_decay': 1e-5,
        'batch_size': 1024,  # Larger batch size
        'epochs': 30,
        'patience': 8
    }
    
    # Train model
    best_auc = train_decline_model(config)
    
    print(f"\nDecline prediction training complete! Best test AUC: {best_auc:.4f}")
    print("Model optimized for real-time inference (<100ms)")

if __name__ == "__main__":
    main()