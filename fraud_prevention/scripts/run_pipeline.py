#!/usr/bin/env python3
"""
Complete fraud detection pipeline runner
Executes the full workflow from data preprocessing to model evaluation
"""

import os
import sys
import subprocess
import argparse
import yaml
import json
from datetime import datetime
import logging

def setup_logging():
    """Setup logging configuration"""
    # Create logs directory if it doesn't exist
    log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'pipeline.log')),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def load_config(config_path):
    """Load configuration file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def run_command(command, description, logger):
    """Run shell command and log output"""
    logger.info(f"Starting: {description}")
    logger.info(f"Command: {command}")
    
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            capture_output=True, 
            text=True,
            check=True
        )
        
        if result.stdout:
            logger.info(f"Output: {result.stdout}")
        
        logger.info(f"Completed: {description}")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed: {description}")
        logger.error(f"Error: {e.stderr}")
        return False

def create_directories(paths, logger):
    """Create necessary directories"""
    for path in paths:
        os.makedirs(path, exist_ok=True)
        logger.info(f"Created directory: {path}")

def main():
    parser = argparse.ArgumentParser(description='Run complete fraud detection pipeline')
    parser.add_argument('--config', default='../config/fraud_config.yaml', help='Configuration file')
    parser.add_argument('--data-path', default='../data', help='Data directory path')
    parser.add_argument('--skip-preprocessing', action='store_true', help='Skip data preprocessing')
    parser.add_argument('--skip-training', action='store_true', help='Skip model training')
    parser.add_argument('--skip-evaluation', action='store_true', help='Skip evaluation')
    parser.add_argument('--run-prediction', action='store_true', help='Run batch prediction')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    logger.info("="*50)
    logger.info("FRAUD DETECTION PIPELINE STARTED")
    logger.info("="*50)
    
    # Load configuration
    config = load_config(args.config)
    
    # Create necessary directories
    directories = [
        '../data/processed',
        '../data/graph', 
        '../models',
        '../results',
        '../logs'
    ]
    create_directories(directories, logger)
    
    # Pipeline execution
    success = True
    
    # Step 1: Data Preprocessing
    if not args.skip_preprocessing:
        logger.info("\n" + "="*30)
        logger.info("STEP 1: DATA PREPROCESSING")
        logger.info("="*30)
        
        success = run_command(
            "python preprocess_chargebacks.py",
            "Data preprocessing and feature engineering",
            logger
        )
        
        if not success:
            logger.error("Pipeline failed at preprocessing step")
            return 1
    
    # Step 2: Graph Feature Generation
    if success and not args.skip_preprocessing:
        logger.info("\n" + "="*30)
        logger.info("STEP 2: GRAPH FEATURE GENERATION")
        logger.info("="*30)
        
        # Use fast version for large datasets
        if os.path.exists('generate_graph_features_fast.py'):
            success = run_command(
                "python generate_graph_features_fast.py",
                "Fast graph construction and neighborhood feature computation",
                logger
            )
        else:
            success = run_command(
                "python generate_graph_features.py",
                "Graph construction and neighborhood feature computation",
                logger
            )
        
        if not success:
            logger.error("Pipeline failed at graph generation step")
            return 1
    
    # Step 3: Model Training
    if success and not args.skip_training:
        logger.info("\n" + "="*30)
        logger.info("STEP 3: MODEL TRAINING")
        logger.info("="*30)
        
        success = run_command(
            "python train_fraud_rgtan.py",
            "RGTAN model training for fraud detection",
            logger
        )
        
        if not success:
            logger.error("Pipeline failed at training step")
            return 1
    
    # Step 4: Model Evaluation
    if success and not args.skip_evaluation:
        logger.info("\n" + "="*30)
        logger.info("STEP 4: MODEL EVALUATION")
        logger.info("="*30)
        
        # Generate evaluation report
        eval_command = f"""
python -c "
import sys
sys.path.append('../utils')
from evaluation_tools import FraudEvaluator
import pandas as pd
import numpy as np
import torch
import pickle

# Load test results
results_df = pd.read_csv('../results/top_fraud_predictions.csv')
y_true = results_df['true_label'].values
y_scores = results_df['fraud_score'].values

# Generate comprehensive evaluation
evaluator = FraudEvaluator()
report = evaluator.generate_management_report(y_true, y_scores)
evaluator.save_evaluation_report(report, '../results/management_report.txt')
evaluator.create_evaluation_dashboard(y_true, y_scores, '../results/evaluation_dashboard.png')

print('Evaluation complete!')
"
"""
        
        success = run_command(
            eval_command,
            "Model evaluation and report generation",
            logger
        )
    
    # Step 5: Batch Prediction (Optional)
    if success and args.run_prediction:
        logger.info("\n" + "="*30)
        logger.info("STEP 5: BATCH PREDICTION")
        logger.info("="*30)
        
        threshold = config['business']['high_risk_threshold']
        top_k = config['evaluation']['top_k_alerts']
        
        success = run_command(
            f"python batch_predict_24hr.py --threshold {threshold} --top-k {top_k}",
            "24-hour batch fraud prediction",
            logger
        )
    
    # Pipeline completion
    if success:
        logger.info("\n" + "="*50)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("="*50)
        
        # Summary
        logger.info("\nOUTPUT FILES:")
        logger.info("- Models: ../models/fraud_rgtan_model.pt")
        logger.info("- Predictions: ../results/top_fraud_predictions.csv")
        logger.info("- Evaluation: ../results/management_report.txt")
        logger.info("- Dashboard: ../results/evaluation_dashboard.png")
        
        if args.run_prediction:
            logger.info("- Batch Results: ../results/batch_predictions/")
        
        logger.info("\nNext steps:")
        logger.info("1. Review evaluation dashboard and management report")
        logger.info("2. Adjust thresholds based on business requirements")
        logger.info("3. Deploy model for production use")
        logger.info("4. Set up monitoring and retraining schedule")
        
        return 0
    else:
        logger.error("\n" + "="*50)
        logger.error("PIPELINE FAILED!")
        logger.error("="*50)
        logger.error("Check logs above for error details")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)