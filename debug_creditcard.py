#!/usr/bin/env python3
"""
Debug script to understand creditcard data structure
"""

import pandas as pd
import os

# Check what's in the preprocessed file
preprocessed_file = 'data/creditcard_preprocessed.csv'

if os.path.exists(preprocessed_file):
    print(f"Loading {preprocessed_file}...")
    df = pd.read_csv(preprocessed_file, nrows=5)  # Just load first 5 rows
    
    print("\nDataFrame shape:", df.shape)
    print("\nColumn names:")
    for i, col in enumerate(df.columns):
        print(f"{i:3d}: {col}")
    
    print("\nColumns containing 'encoded':")
    encoded_cols = [col for col in df.columns if 'encoded' in col]
    for col in encoded_cols:
        print(f"  {col}")
    
    print("\nPotential categorical columns:")
    cat_cols = ['trans_status_msg_id', 'site_tag_id', 'origin_id', 
                'currency_id', 'card_type_id', 'processor_id', 
                'trans_status_code', 'BRAND', 'DEBITCREDIT', 'CARDTYPE']
    
    for col in cat_cols:
        if col in df.columns:
            print(f"  {col}: EXISTS - dtype: {df[col].dtype}, sample values: {df[col].head(3).tolist()}")
        else:
            print(f"  {col}: NOT FOUND")
            if col + '_encoded' in df.columns:
                print(f"    -> {col}_encoded EXISTS")
    
    print("\nFirst few rows of key columns:")
    key_cols = ['Labels', 'amount'] + [col for col in cat_cols if col in df.columns][:3]
    print(df[key_cols].head())
    
else:
    print(f"File {preprocessed_file} not found!")
    print("Please run: python feature_engineering/preprocess_creditcard.py")