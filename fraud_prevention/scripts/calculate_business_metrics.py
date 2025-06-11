#!/usr/bin/env python3
"""
Calculate business metrics from actual transaction data
Updates configuration with data-driven values
"""

import os
import sys
import pandas as pd
import numpy as np
import yaml
import json
from datetime import datetime

def load_transaction_data(data_path):
    """Load dispute and ok transaction data"""
    print("Loading transaction data...")
    
    # Load chargebacks
    chargeback_file = os.path.join(data_path, 'dispute_chargeback_20250606.csv')
    chargebacks = pd.read_csv(chargeback_file)
    print(f"Loaded {len(chargebacks)} chargeback transactions")
    
    # Load ok transactions
    ok_file = os.path.join(data_path, 'ok_transactions_20250606.csv')
    ok_transactions = pd.read_csv(ok_file)
    print(f"Loaded {len(ok_transactions)} legitimate transactions")
    
    return chargebacks, ok_transactions

def calculate_transaction_statistics(chargebacks, ok_transactions):
    """Calculate comprehensive transaction statistics"""
    print("\nCalculating transaction statistics...")
    
    # Basic statistics
    stats = {
        'chargeback_statistics': {
            'count': len(chargebacks),
            'average_amount': float(chargebacks['amount'].mean()),
            'median_amount': float(chargebacks['amount'].median()),
            'std_amount': float(chargebacks['amount'].std()),
            'min_amount': float(chargebacks['amount'].min()),
            'max_amount': float(chargebacks['amount'].max()),
            'percentiles': {
                'p25': float(chargebacks['amount'].quantile(0.25)),
                'p50': float(chargebacks['amount'].quantile(0.50)),
                'p75': float(chargebacks['amount'].quantile(0.75)),
                'p90': float(chargebacks['amount'].quantile(0.90)),
                'p95': float(chargebacks['amount'].quantile(0.95))
            }
        },
        'ok_transaction_statistics': {
            'count': len(ok_transactions),
            'average_amount': float(ok_transactions['amount'].mean()),
            'median_amount': float(ok_transactions['amount'].median()),
            'std_amount': float(ok_transactions['amount'].std()),
            'min_amount': float(ok_transactions['amount'].min()),
            'max_amount': float(ok_transactions['amount'].max()),
            'percentiles': {
                'p25': float(ok_transactions['amount'].quantile(0.25)),
                'p50': float(ok_transactions['amount'].quantile(0.50)),
                'p75': float(ok_transactions['amount'].quantile(0.75)),
                'p90': float(ok_transactions['amount'].quantile(0.90)),
                'p95': float(ok_transactions['amount'].quantile(0.95))
            }
        },
        'overall_statistics': {
            'total_transactions': len(chargebacks) + len(ok_transactions),
            'fraud_rate': len(chargebacks) / (len(chargebacks) + len(ok_transactions)),
            'average_transaction_value': float((ok_transactions['amount'].sum() + chargebacks['amount'].sum()) / 
                                              (len(ok_transactions) + len(chargebacks))),
            'total_chargeback_amount': float(chargebacks['amount'].sum()),
            'total_ok_amount': float(ok_transactions['amount'].sum()),
            'chargeback_to_ok_ratio': float(chargebacks['amount'].mean() / ok_transactions['amount'].mean())
        }
    }
    
    # Distribution analysis
    amount_bins = [0, 10, 25, 50, 100, 250, 500, 1000]
    
    chargeback_dist = pd.cut(chargebacks['amount'], bins=amount_bins, include_lowest=True)
    ok_dist = pd.cut(ok_transactions['amount'], bins=amount_bins, include_lowest=True)
    
    stats['amount_distribution'] = {
        'chargeback_distribution': chargeback_dist.value_counts(normalize=True).to_dict(),
        'ok_distribution': ok_dist.value_counts(normalize=True).to_dict()
    }
    
    return stats

def estimate_chargeback_costs(chargebacks, ok_transactions, stats):
    """Estimate chargeback costs based on transaction data"""
    print("\nEstimating chargeback costs...")
    
    # Average chargeback amount
    avg_chargeback_amount = stats['chargeback_statistics']['average_amount']
    
    # Industry standard fees and costs
    # Based on Visa/Mastercard chargeback fees ($15-25)
    base_chargeback_fee = 20.0
    
    # Processing costs (3-5% of transaction amount)
    processing_cost_rate = 0.04
    avg_processing_cost = avg_chargeback_amount * processing_cost_rate
    
    # Administrative costs (time, investigation, documentation)
    # Estimated 30-60 minutes of work at $30-50/hour
    admin_cost_estimate = 25.0  # 45 min @ $33/hour
    
    # Lost merchandise/service value
    # For physical goods, this is typically 40-60% of transaction amount
    # For digital/services, it's lower (10-30%)
    # We'll use a conservative 35% estimate
    lost_value_rate = 0.35
    avg_lost_value = avg_chargeback_amount * lost_value_rate
    
    # Total estimated chargeback cost
    estimated_chargeback_cost = (
        avg_chargeback_amount +  # Lost transaction amount
        base_chargeback_fee +    # Network fee
        avg_processing_cost +    # Processing costs
        admin_cost_estimate +    # Administrative overhead
        avg_lost_value          # Lost merchandise value
    )
    
    cost_breakdown = {
        'average_chargeback_amount': avg_chargeback_amount,
        'base_chargeback_fee': base_chargeback_fee,
        'processing_cost': avg_processing_cost,
        'administrative_cost': admin_cost_estimate,
        'lost_merchandise_value': avg_lost_value,
        'total_estimated_cost': estimated_chargeback_cost,
        'cost_components': {
            'transaction_amount': avg_chargeback_amount,
            'fees_and_overhead': base_chargeback_fee + avg_processing_cost + admin_cost_estimate,
            'lost_value': avg_lost_value
        }
    }
    
    return cost_breakdown

def calculate_review_costs(ok_transactions):
    """Estimate manual review costs based on transaction patterns"""
    print("\nEstimating manual review costs...")
    
    # Typical manual review takes 10-20 minutes
    avg_review_time_minutes = 15
    
    # Analyst hourly rate (including overhead)
    analyst_hourly_rate = 35.0  # $25-45/hour typical range
    
    # Cost per review
    review_cost = (avg_review_time_minutes / 60) * analyst_hourly_rate
    
    # Additional costs (system access, tools, supervision)
    overhead_multiplier = 1.3  # 30% overhead
    
    total_review_cost = review_cost * overhead_multiplier
    
    review_cost_breakdown = {
        'avg_review_time_minutes': avg_review_time_minutes,
        'analyst_hourly_rate': analyst_hourly_rate,
        'base_review_cost': review_cost,
        'overhead_multiplier': overhead_multiplier,
        'total_review_cost': total_review_cost
    }
    
    return review_cost_breakdown

def update_config_with_metrics(stats, chargeback_costs, review_costs, config_path):
    """Update configuration file with calculated metrics"""
    print("\nUpdating configuration with calculated metrics...")
    
    # Load existing config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update business section with calculated values
    config['business']['average_chargeback_cost'] = round(chargeback_costs['total_estimated_cost'], 2)
    config['business']['average_transaction_value'] = round(stats['overall_statistics']['average_transaction_value'], 2)
    config['business']['false_positive_review_cost'] = round(review_costs['total_review_cost'], 2)
    
    # Add new calculated metrics section
    config['calculated_metrics'] = {
        'last_updated': datetime.now().isoformat(),
        'data_source': 'dispute_chargeback_20250606.csv and ok_transactions_20250606.csv',
        'fraud_rate': round(stats['overall_statistics']['fraud_rate'], 4),
        'chargeback_statistics': {
            'average_amount': round(stats['chargeback_statistics']['average_amount'], 2),
            'median_amount': round(stats['chargeback_statistics']['median_amount'], 2),
            'p95_amount': round(stats['chargeback_statistics']['percentiles']['p95'], 2)
        },
        'cost_breakdown': {
            'chargeback': chargeback_costs['cost_components'],
            'review': review_costs
        }
    }
    
    # Save updated config
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"Configuration updated at {config_path}")
    
    return config

def generate_report(stats, chargeback_costs, review_costs):
    """Generate a report of calculated metrics"""
    
    report = f"""
CALCULATED BUSINESS METRICS REPORT
==================================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

TRANSACTION STATISTICS
---------------------
Total Transactions: {stats['overall_statistics']['total_transactions']:,}
Fraud Rate: {stats['overall_statistics']['fraud_rate']:.2%}

Chargeback Transactions:
- Count: {stats['chargeback_statistics']['count']:,}
- Average Amount: ${stats['chargeback_statistics']['average_amount']:.2f}
- Median Amount: ${stats['chargeback_statistics']['median_amount']:.2f}
- 95th Percentile: ${stats['chargeback_statistics']['percentiles']['p95']:.2f}

Legitimate Transactions:
- Count: {stats['ok_transaction_statistics']['count']:,}
- Average Amount: ${stats['ok_transaction_statistics']['average_amount']:.2f}
- Median Amount: ${stats['ok_transaction_statistics']['median_amount']:.2f}

COST CALCULATIONS
-----------------
Average Chargeback Cost: ${chargeback_costs['total_estimated_cost']:.2f}
  - Transaction Amount: ${chargeback_costs['average_chargeback_amount']:.2f}
  - Chargeback Fee: ${chargeback_costs['base_chargeback_fee']:.2f}
  - Processing Cost: ${chargeback_costs['processing_cost']:.2f}
  - Administrative Cost: ${chargeback_costs['administrative_cost']:.2f}
  - Lost Merchandise Value: ${chargeback_costs['lost_merchandise_value']:.2f}

Manual Review Cost: ${review_costs['total_review_cost']:.2f}
  - Review Time: {review_costs['avg_review_time_minutes']} minutes
  - Analyst Rate: ${review_costs['analyst_hourly_rate']:.2f}/hour
  - Overhead: {(review_costs['overhead_multiplier']-1)*100:.0f}%

RECOMMENDED CONFIG VALUES
------------------------
average_chargeback_cost: {chargeback_costs['total_estimated_cost']:.2f}
average_transaction_value: {stats['overall_statistics']['average_transaction_value']:.2f}
false_positive_review_cost: {review_costs['total_review_cost']:.2f}

Note: These values are calculated from actual transaction data and include
industry-standard costs for fees, processing, and administrative overhead.
"""
    
    return report

def main():
    # Paths
    data_path = '../../data'
    config_path = '../config/fraud_config.yaml'
    output_path = '../results'
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Load transaction data
    chargebacks, ok_transactions = load_transaction_data(data_path)
    
    # Calculate statistics
    stats = calculate_transaction_statistics(chargebacks, ok_transactions)
    
    # Estimate costs
    chargeback_costs = estimate_chargeback_costs(chargebacks, ok_transactions, stats)
    review_costs = calculate_review_costs(ok_transactions)
    
    # Update configuration
    updated_config = update_config_with_metrics(stats, chargeback_costs, review_costs, config_path)
    
    # Generate report
    report = generate_report(stats, chargeback_costs, review_costs)
    print(report)
    
    # Save report
    report_path = os.path.join(output_path, 'calculated_business_metrics.txt')
    with open(report_path, 'w') as f:
        f.write(report)
    
    # Save detailed statistics as JSON
    detailed_stats = {
        'transaction_statistics': stats,
        'chargeback_cost_analysis': chargeback_costs,
        'review_cost_analysis': review_costs,
        'generated_at': datetime.now().isoformat()
    }
    
    json_path = os.path.join(output_path, 'business_metrics_detailed.json')
    with open(json_path, 'w') as f:
        json.dump(detailed_stats, f, indent=2, default=str)
    
    print(f"\nReports saved to:")
    print(f"  - {report_path}")
    print(f"  - {json_path}")
    print(f"  - Config updated: {config_path}")

if __name__ == "__main__":
    main()