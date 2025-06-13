#!/usr/bin/env python3
"""
Check and fix file path issues
"""

import os
import sys

def check_file_locations():
    """Check where files actually exist"""
    
    print("=== File Location Check ===")
    
    # Possible locations
    locations = [
        '/home/development/affdf/fraud_prevention/data',
        '/home/development/affdf/data', 
        '../data',
        'data'
    ]
    
    files_to_check = [
        'processed/train_transactions.csv',
        'processed/test_transactions.csv', 
        'graph/transaction_graph.dgl',
        'graph/train_neigh_features.csv',
        'graph/test_neigh_features.csv',
        'graph/combined_transactions.csv'
    ]
    
    found_files = {}
    
    for location in locations:
        if os.path.exists(location):
            print(f"\nChecking: {location}")
            for file_path in files_to_check:
                full_path = os.path.join(location, file_path)
                if os.path.exists(full_path):
                    size = os.path.getsize(full_path)
                    print(f"  ✓ {file_path} ({size:,} bytes)")
                    found_files[file_path] = full_path
                else:
                    print(f"  ✗ {file_path}")
    
    print(f"\n=== Summary ===")
    print(f"Found {len(found_files)} out of {len(files_to_check)} required files")
    
    if len(found_files) == len(files_to_check):
        print("✅ All files found!")
        
        # Determine the correct base path
        first_file = list(found_files.values())[0]
        if '/fraud_prevention/data/' in first_file:
            correct_path = '/home/development/affdf/fraud_prevention/data'
        else:
            correct_path = '/home/development/affdf/data'
        
        print(f"✅ Recommended data path: {correct_path}")
        return correct_path
    else:
        missing = set(files_to_check) - set(found_files.keys())
        print(f"❌ Missing files: {missing}")
        return None

if __name__ == "__main__":
    check_file_locations()