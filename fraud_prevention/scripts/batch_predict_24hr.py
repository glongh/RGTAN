#!/usr/bin/env python3
"""
Batch prediction script for fraud detection on last 24 hours of transactions
Loads trained RGTAN model and scores new transactions
"""

import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from datetime import datetime, timedelta
import pickle
import json
import argparse
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fraud_prevention.scripts.train_fraud_rgtan import FraudRGTAN
from fraud_prevention.scripts.preprocess_chargebacks import preprocess_features, hash_sensitive_data
from fraud_prevention.scripts.generate_graph_features import build_transaction_graph, compute_neighborhood_features

def load_recent_transactions(data_path, hours=24):
    """Load transactions from the last N hours"""
    print(f"\nLoading transactions from last {hours} hours...")
    
    # In production, this would query your transaction database
    # For demo, we'll use test data and filter by date
    test_df = pd.read_csv(os.path.join(data_path, 'processed/test_transactions.csv'))
    test_df['issue_date'] = pd.to_datetime(test_df['issue_date'])
    
    # Get cutoff time
    max_date = test_df['issue_date'].max()
    cutoff_date = max_date - timedelta(hours=hours)
    
    # Filter recent transactions
    recent_df = test_df[test_df['issue_date'] >= cutoff_date].copy()
    
    print(f"Found {len(recent_df)} transactions since {cutoff_date}")
    print(f"Date range: {recent_df['issue_date'].min()} to {recent_df['issue_date'].max()}")
    
    return recent_df

def load_historical_context(data_path, days=90):
    """Load historical transactions for graph context"""
    print(f"\nLoading {days} days of historical context...")
    
    # Load training data as historical context
    train_df = pd.read_csv(os.path.join(data_path, 'processed/train_transactions.csv'))
    train_df['issue_date'] = pd.to_datetime(train_df['issue_date'])
    
    # In production, filter to last N days
    cutoff_date = train_df['issue_date'].max() - timedelta(days=days)
    historical_df = train_df[train_df['issue_date'] >= cutoff_date].copy()
    
    print(f"Loaded {len(historical_df)} historical transactions")
    
    return historical_df

def prepare_batch_features(recent_df, historical_df, label_encoders, scaler, neigh_scaler):
    """Prepare features for batch prediction"""
    print("\nPreparing features for prediction...")
    
    # Combine recent and historical for graph building
    combined_df = pd.concat([historical_df, recent_df], ignore_index=True)
    combined_df['trans_id'] = range(len(combined_df))
    combined_df['is_historical'] = [1] * len(historical_df) + [0] * len(recent_df)
    
    # Build graph
    print("Building transaction graph...")
    g = build_transaction_graph(combined_df)
    
    # Compute neighborhood features
    print("Computing neighborhood features...")
    neigh_features_df = compute_neighborhood_features(g, combined_df, hop_sizes=[1, 2])
    
    # Extract features for recent transactions
    recent_idx = combined_df[combined_df['is_historical'] == 0].index
    
    # Prepare feature columns
    numeric_cols = ['amount_log', 'time_to_capture', 'hour', 'day_of_week', 
                   'is_weekend', 'is_night', 'is_round_amount', 'is_high_amount']
    
    categorical_cols = [col for col in recent_df.columns if col.endswith('_encoded')]
    feature_cols = numeric_cols + categorical_cols
    
    # Extract and scale features
    recent_features = combined_df.loc[recent_idx, feature_cols].values
    recent_features = scaler.transform(recent_features)
    
    # Extract and scale neighborhood features
    neigh_cols = [col for col in neigh_features_df.columns if col != 'trans_id']
    recent_neigh = neigh_features_df.loc[recent_idx, neigh_cols].values
    recent_neigh = neigh_scaler.transform(recent_neigh)
    
    return recent_features, recent_neigh, g, recent_idx

def load_model(model_path):
    """Load trained RGTAN model"""
    print(f"\nLoading model from {model_path}")
    
    checkpoint = torch.load(model_path, map_location='cpu')
    model_config = checkpoint['model_config']
    
    # Initialize model
    model = FraudRGTAN(
        in_feats=model_config['in_feats'],
        hidden_dim=model_config['hidden_dim'],
        n_layers=model_config['n_layers'],
        n_classes=model_config['n_classes'],
        heads=model_config['heads'],
        activation=nn.PReLU(),
        drop=[0, 0, 0],  # No dropout for inference
        device='cpu',
        nei_feats_dim=model_config['nei_feats_dim']
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load scalers
    scaler = checkpoint['scaler']
    neigh_scaler = checkpoint['neigh_scaler']
    
    return model, scaler, neigh_scaler

def predict_fraud(model, g, features, nei_features, indices, device='cpu', batch_size=1024):
    """Run batch predictions"""
    print("\nRunning fraud predictions...")
    
    model = model.to(device)
    
    # Create dataloader for batch prediction
    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
    dataloader = dgl.dataloading.DataLoader(
        g, indices, sampler,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0
    )
    
    all_scores = []
    
    with torch.no_grad():
        for i, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
            blocks = [b.to(device) for b in blocks]
            
            # Get all features (including historical for graph context)
            all_features = torch.zeros((g.num_nodes(), features.shape[1]))
            all_features[indices] = torch.FloatTensor(features)
            
            all_nei_features = torch.zeros((g.num_nodes(), nei_features.shape[1]))
            all_nei_features[indices] = torch.FloatTensor(nei_features)
            
            batch_features = all_features[input_nodes].to(device)
            batch_nei_features = all_nei_features[input_nodes].unsqueeze(1).to(device)
            
            # Predict
            logits = model(blocks, batch_features, batch_nei_features)
            scores = F.softmax(logits, dim=1)[:, 1]  # Fraud probability
            
            all_scores.append(scores.cpu().numpy())
            
            if (i + 1) % 10 == 0:
                print(f"  Processed {(i + 1) * batch_size} transactions...")
    
    return np.concatenate(all_scores)

def generate_alerts(recent_df, fraud_scores, threshold=0.5, top_k=100):
    """Generate fraud alerts and reports"""
    print("\nGenerating fraud alerts...")
    
    # Add scores to dataframe
    results_df = recent_df.copy()
    results_df['fraud_score'] = fraud_scores
    results_df['is_high_risk'] = (fraud_scores > threshold).astype(int)
    
    # Sort by fraud score
    results_df = results_df.sort_values('fraud_score', ascending=False)
    
    # High risk transactions
    high_risk_count = results_df['is_high_risk'].sum()
    print(f"\nFound {high_risk_count} high-risk transactions (score > {threshold})")
    
    # Top fraud candidates
    top_frauds = results_df.head(top_k)
    
    # Calculate statistics
    stats = {
        'total_transactions': len(results_df),
        'high_risk_count': high_risk_count,
        'high_risk_rate': high_risk_count / len(results_df),
        'avg_fraud_score': fraud_scores.mean(),
        'max_fraud_score': fraud_scores.max(),
        'total_amount_at_risk': results_df[results_df['is_high_risk'] == 1]['amount'].sum(),
        'processing_time': datetime.now().isoformat()
    }
    
    return results_df, top_frauds, stats

def save_results(results_df, top_frauds, stats, output_path):
    """Save prediction results"""
    print(f"\nSaving results to {output_path}")
    
    os.makedirs(output_path, exist_ok=True)
    
    # Save all predictions
    results_df.to_csv(
        os.path.join(output_path, f'fraud_predictions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'),
        index=False
    )
    
    # Save top fraud alerts
    top_frauds.to_csv(
        os.path.join(output_path, f'fraud_alerts_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'),
        index=False
    )
    
    # Save statistics
    with open(os.path.join(output_path, f'batch_stats_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'), 'w') as f:
        json.dump(stats, f, indent=2)
    
    # Generate summary report
    report = f"""
Fraud Detection Batch Report
===========================
Processing Time: {stats['processing_time']}
Total Transactions: {stats['total_transactions']:,}
High Risk Transactions: {stats['high_risk_count']:,} ({stats['high_risk_rate']:.2%})
Average Fraud Score: {stats['avg_fraud_score']:.4f}
Maximum Fraud Score: {stats['max_fraud_score']:.4f}
Total Amount at Risk: ${stats['total_amount_at_risk']:,.2f}

Top 5 Highest Risk Transactions:
"""
    
    for idx, row in top_frauds.head(5).iterrows():
        report += f"\n- Transaction {row['trans_id']}: Score {row['fraud_score']:.4f}, Amount ${row['amount']:.2f}"
    
    with open(os.path.join(output_path, f'fraud_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'), 'w') as f:
        f.write(report)
    
    print(report)

def main():
    parser = argparse.ArgumentParser(description='Batch fraud prediction for last 24 hours')
    parser.add_argument('--data-path', default='../data', help='Path to data directory')
    parser.add_argument('--model-path', default='../models/fraud_rgtan_model.pt', help='Path to trained model')
    parser.add_argument('--output-path', default='../results/batch_predictions', help='Output directory')
    parser.add_argument('--hours', type=int, default=24, help='Hours to look back')
    parser.add_argument('--threshold', type=float, default=0.5, help='Fraud score threshold')
    parser.add_argument('--top-k', type=int, default=100, help='Number of top alerts to generate')
    
    args = parser.parse_args()
    
    # Load recent transactions
    recent_df = load_recent_transactions(args.data_path, args.hours)
    
    if len(recent_df) == 0:
        print("No recent transactions found!")
        return
    
    # Load historical context
    historical_df = load_historical_context(args.data_path, days=90)
    
    # Load model and scalers
    model, scaler, neigh_scaler = load_model(args.model_path)
    
    # Load label encoders (for preprocessing if needed)
    with open(os.path.join(args.data_path, 'processed/label_encoders.pkl'), 'rb') as f:
        label_encoders = pickle.load(f)
    
    # Prepare features
    features, nei_features, g, indices = prepare_batch_features(
        recent_df, historical_df, label_encoders, scaler, neigh_scaler
    )
    
    # Run predictions
    fraud_scores = predict_fraud(model, g, features, nei_features, indices)
    
    # Generate alerts
    results_df, top_frauds, stats = generate_alerts(
        recent_df, fraud_scores, args.threshold, args.top_k
    )
    
    # Save results
    save_results(results_df, top_frauds, stats, args.output_path)
    
    print("\nBatch prediction complete!")

if __name__ == "__main__":
    main()