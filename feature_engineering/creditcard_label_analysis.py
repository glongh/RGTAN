#!/usr/bin/env python3
"""
Analyze auth_msg patterns in credit card dataset to understand decline reasons.
This helps in creating appropriate fraud labels.
"""

import pandas as pd
import argparse
from collections import Counter
import re


def analyze_auth_messages(df):
    """Analyze authorization messages to understand patterns"""
    
    print("=== Authorization Message Analysis ===\n")
    
    # Get value counts
    auth_counts = df['auth_msg'].value_counts()
    total = len(df)
    
    print(f"Total transactions: {total:,}")
    print(f"Unique auth messages: {len(auth_counts):,}\n")
    
    print("Top 20 Authorization Messages:")
    print("-" * 60)
    for msg, count in auth_counts.head(20).items():
        print(f"{msg:<40} {count:>8,} ({count/total*100:>6.2f}%)")
    
    # Categorize messages
    categories = {
        'approved': [],
        'insufficient_funds': [],
        'security_decline': [],
        'merchant_issues': [],
        'card_issues': [],
        'technical_issues': [],
        'other_declines': []
    }
    
    # Categorization rules
    for msg, count in auth_counts.items():
        msg_upper = str(msg).upper()
        
        if 'APPROVED' in msg_upper:
            categories['approved'].append((msg, count))
        elif any(x in msg_upper for x in ['INSUFF', 'FUNDS', 'NSF', 'BALANCE']):
            categories['insufficient_funds'].append((msg, count))
        elif any(x in msg_upper for x in ['FRAUD', 'STOLEN', 'LOST', 'SECURITY', 'BLOCKED', 'RESTRICTED']):
            categories['security_decline'].append((msg, count))
        elif any(x in msg_upper for x in ['MERCHANT', 'TERM ID', 'INVALID MERCHANT']):
            categories['merchant_issues'].append((msg, count))
        elif any(x in msg_upper for x in ['EXPIRED', 'INVALID CARD', 'CARD ERROR']):
            categories['card_issues'].append((msg, count))
        elif any(x in msg_upper for x in ['ERROR', 'SYSTEM', 'TIMEOUT', 'UNAVAILABLE']):
            categories['technical_issues'].append((msg, count))
        elif 'DECLINE' in msg_upper or 'CALL' in msg_upper:
            categories['other_declines'].append((msg, count))
    
    # Print categorized results
    print("\n\n=== Message Categories ===")
    for category, messages in categories.items():
        if messages:
            total_cat = sum(count for _, count in messages)
            print(f"\n{category.upper().replace('_', ' ')} ({total_cat:,} = {total_cat/total*100:.2f}%)")
            print("-" * 60)
            for msg, count in sorted(messages, key=lambda x: x[1], reverse=True)[:10]:
                print(f"{msg:<40} {count:>8,} ({count/total*100:>6.2f}%)")
    
    return categories


def create_label_recommendations(categories):
    """Provide recommendations for labeling strategy"""
    
    print("\n\n=== Labeling Recommendations ===\n")
    
    print("Based on the analysis, here are different labeling strategies:\n")
    
    print("1. BINARY CLASSIFICATION (Fraud/Not Fraud):")
    print("   - Label as FRAUD (1): security_decline + some suspicious patterns")
    print("   - Label as LEGITIMATE (0): approved + technical issues\n")
    
    print("2. BINARY CLASSIFICATION (Success/Failure):")
    print("   - Label as SUCCESS (0): approved only")
    print("   - Label as FAILURE (1): all declines\n")
    
    print("3. MULTI-CLASS CLASSIFICATION:")
    print("   - 0: Approved")
    print("   - 1: Insufficient Funds")
    print("   - 2: Security/Fraud")
    print("   - 3: Merchant Issues")
    print("   - 4: Card Issues")
    print("   - 5: Other\n")
    
    print("4. RISK-BASED BINARY:")
    print("   - HIGH RISK (1): security_decline + stolen/lost + fraud patterns")
    print("   - LOW RISK (0): everything else\n")


def main():
    parser = argparse.ArgumentParser(description='Analyze credit card auth messages')
    parser.add_argument('--input', default='data/vod_creditcard.csv', help='Input CSV file')
    parser.add_argument('--sample', type=int, help='Sample size for analysis')
    
    args = parser.parse_args()
    
    print(f"Loading data from {args.input}...")
    df = pd.read_csv(args.input)
    
    if args.sample and args.sample < len(df):
        df = df.sample(n=args.sample, random_state=42)
        print(f"Using sample of {args.sample} transactions")
    
    if 'auth_msg' not in df.columns:
        print("Error: auth_msg column not found!")
        return
    
    categories = analyze_auth_messages(df)
    create_label_recommendations(categories)
    
    # Additional statistics
    if 'amount' in df.columns:
        print("\n=== Amount Statistics by Category ===")
        
        # Create simple labels for analysis
        df['category'] = 'other'
        df.loc[df['auth_msg'].str.contains('APPROVED', case=False, na=False), 'category'] = 'approved'
        df.loc[df['auth_msg'].str.contains('DECLINE', case=False, na=False), 'category'] = 'declined'
        df.loc[df['auth_msg'].str.contains('FRAUD|STOLEN|LOST', case=False, na=False), 'category'] = 'fraud'
        
        print("\nAverage amounts by category:")
        print(df.groupby('category')['amount'].agg(['mean', 'median', 'std', 'count']))


if __name__ == "__main__":
    main()