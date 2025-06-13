#!/usr/bin/env python3
"""
Find data files in the system
"""

import os
import sys

print("Current working directory:", os.getcwd())
print("\nSearching for data files...")

# Files we're looking for
target_files = [
    'dispute_chargeback_20250612.csv',
    'ok_transactions_20250612.csv',
    'denied_transactions_20250612.csv'
]

# Search in common locations
search_paths = [
    '.',
    '..',
    '../..',
    '../../..',
    'data',
    '../data',
    '../../data',
    '../../../data',
    '/home/development/affdf/data',
    '/home/development/data',
    os.path.expanduser('~/data'),
    os.path.expanduser('~/affdf/data')
]

found_files = {}

for search_path in search_paths:
    if os.path.exists(search_path):
        print(f"\nChecking {os.path.abspath(search_path)}...")
        
        # List all CSV files
        try:
            files = [f for f in os.listdir(search_path) if f.endswith('.csv')]
            if files:
                print(f"  Found CSV files: {', '.join(files[:5])}")
                if len(files) > 5:
                    print(f"  ... and {len(files) - 5} more")
            
            # Check for target files
            for target in target_files:
                filepath = os.path.join(search_path, target)
                if os.path.exists(filepath):
                    found_files[target] = os.path.abspath(filepath)
                    print(f"  ✓ Found {target}")
        except PermissionError:
            print(f"  Permission denied")
        except Exception as e:
            print(f"  Error: {e}")

print("\n" + "="*50)
if found_files:
    print("FOUND DATA FILES:")
    for filename, filepath in found_files.items():
        print(f"  {filename}: {filepath}")
    
    # Suggest the data directory
    if found_files:
        data_dir = os.path.dirname(list(found_files.values())[0])
        print(f"\nData directory appears to be: {data_dir}")
        print(f"\nTo fix the pipeline, update the data path in the scripts to:")
        print(f"  data_path = '{data_dir}'")
else:
    print("Could not find the required data files!")
    print("\nPlease ensure these files exist:")
    for f in target_files:
        print(f"  - {f}")
    print("\nOr update the filenames in the configuration if they have different dates.")