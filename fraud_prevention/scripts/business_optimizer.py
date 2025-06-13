#!/usr/bin/env python3
"""
Business metrics optimizer for fraud detection
Finds optimal thresholds based on business costs, not just technical metrics
"""

import os
import sys
import pandas as pd
import numpy as np
import yaml
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve
import json

def load_business_config():
    """Load business configuration"""
    config_path = '../config/fraud_config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config['business']

def calculate_business_metrics(y_true, y_scores, threshold, business_config):
    """Calculate business metrics for a given threshold"""
    
    # Predictions based on threshold
    y_pred = (y_scores >= threshold).astype(int)
    
    # Confusion matrix components
    tp = ((y_pred == 1) & (y_true == 1)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()
    tn = ((y_pred == 0) & (y_true == 0)).sum()
    
    # Business costs
    chargeback_cost = business_config['average_chargeback_cost']
    review_cost = business_config['false_positive_review_cost']
    
    # Calculate costs
    fraud_prevented_value = tp * chargeback_cost  # Fraud caught
    fraud_missed_cost = fn * chargeback_cost      # Fraud missed
    review_cost_total = (tp + fp) * review_cost   # All flagged transactions
    
    # Net benefit = Fraud prevented - Review costs - Missed fraud
    net_benefit = fraud_prevented_value - review_cost_total - fraud_missed_cost
    
    # Other metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Flag rate (operational constraint)
    flag_rate = (tp + fp) / len(y_true)
    
    return {
        'threshold': threshold,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'true_positives': tp,
        'false_positives': fp,
        'false_negatives': fn,
        'true_negatives': tn,
        'flag_rate': flag_rate,
        'fraud_prevented_value': fraud_prevented_value,
        'fraud_missed_cost': fraud_missed_cost,
        'review_cost_total': review_cost_total,
        'net_benefit': net_benefit,
        'net_benefit_per_transaction': net_benefit / len(y_true)
    }

def optimize_business_threshold(y_true, y_scores, business_config):
    """Find optimal threshold based on business metrics"""
    
    print("Optimizing threshold for business value...")
    
    # Test range of thresholds
    thresholds = np.arange(0.1, 0.95, 0.05)
    results = []
    
    max_daily_reviews = business_config.get('max_daily_reviews', 1000)
    
    for threshold in thresholds:
        metrics = calculate_business_metrics(y_true, y_scores, threshold, business_config)
        
        # Check operational constraints
        daily_flags = metrics['flag_rate'] * len(y_true)  # Assuming this is daily data
        metrics['daily_flags'] = daily_flags
        metrics['operationally_feasible'] = daily_flags <= max_daily_reviews
        
        results.append(metrics)
    
    results_df = pd.DataFrame(results)
    
    # Find optimal thresholds based on different criteria
    
    # 1. Maximum net benefit
    optimal_net = results_df.loc[results_df['net_benefit'].idxmax()]
    
    # 2. Maximum net benefit with operational constraints
    feasible_df = results_df[results_df['operationally_feasible']]
    if len(feasible_df) > 0:
        optimal_feasible = feasible_df.loc[feasible_df['net_benefit'].idxmax()]
    else:
        optimal_feasible = optimal_net
        print("WARNING: No operationally feasible threshold found!")
    
    # 3. Target precision threshold
    target_precision = business_config.get('target_precision', 0.4)
    precision_mask = results_df['precision'] >= target_precision
    if precision_mask.any():
        optimal_precision = results_df[precision_mask].loc[results_df[precision_mask]['net_benefit'].idxmax()]
    else:
        # Find closest to target precision
        precision_diff = np.abs(results_df['precision'] - target_precision)
        optimal_precision = results_df.loc[precision_diff.idxmin()]
    
    return {
        'results_df': results_df,
        'optimal_net_benefit': optimal_net,
        'optimal_feasible': optimal_feasible,
        'optimal_precision': optimal_precision
    }

def create_business_dashboard(optimization_results, output_path):
    """Create business optimization dashboard"""
    
    results_df = optimization_results['results_df']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Business Optimization Dashboard', fontsize=16)
    
    # 1. Net Benefit vs Threshold
    ax1 = axes[0, 0]
    ax1.plot(results_df['threshold'], results_df['net_benefit'], 'b-', linewidth=2)
    ax1.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Threshold')
    ax1.set_ylabel('Net Benefit ($)')
    ax1.set_title('Net Benefit vs Threshold')
    ax1.grid(True, alpha=0.3)
    
    # Mark optimal point
    optimal = optimization_results['optimal_feasible']
    ax1.plot(optimal['threshold'], optimal['net_benefit'], 'ro', markersize=10, label='Optimal')
    ax1.legend()
    
    # 2. Precision-Recall Trade-off
    ax2 = axes[0, 1]
    ax2.plot(results_df['recall'], results_df['precision'], 'g-', linewidth=2)
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall Curve')
    ax2.grid(True, alpha=0.3)
    
    # Mark optimal point
    ax2.plot(optimal['recall'], optimal['precision'], 'ro', markersize=10, label='Optimal')
    ax2.legend()
    
    # 3. Flag Rate vs Threshold
    ax3 = axes[1, 0]
    ax3.plot(results_df['threshold'], results_df['flag_rate'], 'orange', linewidth=2)
    ax3.set_xlabel('Threshold')
    ax3.set_ylabel('Flag Rate')
    ax3.set_title('Flag Rate vs Threshold')
    ax3.grid(True, alpha=0.3)
    
    # Mark operational constraint
    max_flag_rate = 1000 / len(results_df)  # Assuming daily data
    ax3.axhline(y=max_flag_rate, color='r', linestyle='--', label='Max Capacity')
    ax3.plot(optimal['threshold'], optimal['flag_rate'], 'ro', markersize=10, label='Optimal')
    ax3.legend()
    
    # 4. Cost Breakdown
    ax4 = axes[1, 1]
    costs = [
        optimal['fraud_prevented_value'],
        -optimal['review_cost_total'],
        -optimal['fraud_missed_cost']
    ]
    labels = ['Fraud Prevented', 'Review Costs', 'Missed Fraud']
    colors = ['green', 'orange', 'red']
    
    bars = ax4.bar(labels, costs, color=colors, alpha=0.7)
    ax4.set_ylabel('Value ($)')
    ax4.set_title(f'Cost Breakdown (Threshold = {optimal["threshold"]:.2f})')
    ax4.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, cost in zip(bars, costs):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + (abs(height) * 0.01),
                f'${cost:,.0f}', ha='center', va='bottom' if height > 0 else 'top')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'business_optimization_dashboard.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Business dashboard saved to {output_path}")

def generate_business_report(optimization_results, business_config, output_path):
    """Generate business optimization report"""
    
    optimal = optimization_results['optimal_feasible']
    results_df = optimization_results['results_df']
    
    report = f"""
# Fraud Detection Business Optimization Report

## Executive Summary

Based on the analysis of fraud detection performance and business costs, we recommend:

**Optimal Threshold: {optimal['threshold']:.2f}**

### Key Metrics:
- **Net Benefit**: ${optimal['net_benefit']:,.0f} per day
- **Precision**: {optimal['precision']:.1%} (of flagged transactions are truly fraudulent)
- **Recall**: {optimal['recall']:.1%} (of actual fraud is caught)
- **Daily Reviews**: {optimal['daily_flags']:.0f} transactions need manual review

### Business Impact:
- **Fraud Prevented**: ${optimal['fraud_prevented_value']:,.0f} per day
- **Review Costs**: ${optimal['review_cost_total']:,.0f} per day
- **Missed Fraud**: ${optimal['fraud_missed_cost']:,.0f} per day

## Detailed Analysis

### Cost Assumptions:
- Average chargeback cost: ${business_config['average_chargeback_cost']:.2f}
- Review cost per transaction: ${business_config['false_positive_review_cost']:.2f}
- Maximum daily review capacity: {business_config.get('max_daily_reviews', 1000)} transactions

### Threshold Comparison:

| Threshold | Precision | Recall | Net Benefit | Daily Flags | Feasible |
|-----------|-----------|--------|-------------|-------------|----------|
"""
    
    # Add top 5 thresholds by net benefit
    top_5 = results_df.nlargest(5, 'net_benefit')
    for _, row in top_5.iterrows():
        feasible = "✓" if row['operationally_feasible'] else "✗"
        report += f"| {row['threshold']:.2f} | {row['precision']:.1%} | {row['recall']:.1%} | ${row['net_benefit']:,.0f} | {row['daily_flags']:.0f} | {feasible} |\n"
    
    report += f"""

## Recommendations

1. **Implement threshold {optimal['threshold']:.2f}** for optimal business value
2. **Plan for {optimal['daily_flags']:.0f} daily reviews** - ensure sufficient analyst capacity
3. **Monitor precision** - should maintain {optimal['precision']:.1%} accuracy
4. **Track net benefit** - expect ${optimal['net_benefit']:,.0f} daily value creation

## Risk Considerations

- False positive rate: {optimal['false_positives']/(optimal['false_positives']+optimal['true_negatives']):.1%}
- Missed fraud rate: {optimal['false_negatives']/(optimal['false_negatives']+optimal['true_positives']):.1%}
- Review capacity utilization: {optimal['daily_flags']/business_config.get('max_daily_reviews', 1000):.1%}

Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    # Save report
    report_path = os.path.join(output_path, 'business_optimization_report.md')
    with open(report_path, 'w') as f:
        f.write(report)
    
    # Save detailed results
    results_df.to_csv(os.path.join(output_path, 'threshold_optimization_results.csv'), index=False)
    
    # Save optimal configuration
    optimal_config = {
        'recommended_threshold': float(optimal['threshold']),
        'expected_precision': float(optimal['precision']),
        'expected_recall': float(optimal['recall']),
        'expected_daily_flags': int(optimal['daily_flags']),
        'expected_net_benefit': float(optimal['net_benefit']),
        'optimization_date': pd.Timestamp.now().isoformat()
    }
    
    with open(os.path.join(output_path, 'optimal_config.json'), 'w') as f:
        json.dump(optimal_config, f, indent=2)
    
    print(f"Business report saved to {report_path}")
    return optimal_config

def main():
    """Main business optimization function"""
    
    # Check if we have predictions to optimize
    results_path = '../results/top_fraud_predictions.csv'
    if not os.path.exists(results_path):
        print(f"Error: Predictions file not found: {results_path}")
        print("Run model training first to generate predictions")
        return 1
    
    # Load data
    print("Loading predictions and business configuration...")
    predictions_df = pd.read_csv(results_path)
    business_config = load_business_config()
    
    # Extract true labels and scores
    y_true = predictions_df['true_label'].values
    y_scores = predictions_df['fraud_score'].values
    
    print(f"Loaded {len(y_true)} predictions")
    print(f"Fraud rate: {y_true.mean():.2%}")
    
    # Optimize threshold
    optimization_results = optimize_business_threshold(y_true, y_scores, business_config)
    
    # Create output directory
    output_path = '../results'
    os.makedirs(output_path, exist_ok=True)
    
    # Generate reports and dashboard
    create_business_dashboard(optimization_results, output_path)
    optimal_config = generate_business_report(optimization_results, business_config, output_path)
    
    print("\n" + "="*50)
    print("BUSINESS OPTIMIZATION COMPLETE")
    print("="*50)
    print(f"Recommended threshold: {optimal_config['recommended_threshold']:.3f}")
    print(f"Expected net benefit: ${optimal_config['expected_net_benefit']:,.0f}/day")
    print(f"Expected daily reviews: {optimal_config['expected_daily_flags']}")
    print("="*50)
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)