#!/usr/bin/env python3
"""
Complete training pipeline for decline prediction
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
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('../logs/training_pipeline.log'),
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
    parser = argparse.ArgumentParser(description='Run complete decline prediction training pipeline')
    parser.add_argument('--config', default='../config/decline_config.yaml', help='Configuration file')
    parser.add_argument('--data-path', default='../data', help='Data directory path')
    parser.add_argument('--skip-preprocessing', action='store_true', help='Skip data preprocessing')
    parser.add_argument('--skip-training', action='store_true', help='Skip model training')
    parser.add_argument('--skip-evaluation', action='store_true', help='Skip evaluation')
    parser.add_argument('--start-api', action='store_true', help='Start API server after training')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    logger.info("="*50)
    logger.info("DECLINE PREDICTION TRAINING PIPELINE STARTED")
    logger.info("="*50)
    
    # Load configuration
    config = load_config(args.config)
    
    # Create necessary directories
    directories = [
        '../data/processed',
        '../data/graph', 
        '../models',
        '../results',
        '../logs',
        '../cache'
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
            "python preprocess_decline_data.py",
            "Decline data preprocessing and feature engineering",
            logger
        )
        
        if not success:
            logger.error("Pipeline failed at preprocessing step")
            return 1
    
    # Step 2: Graph Feature Generation
    if success and not args.skip_preprocessing:
        logger.info("\n" + "="*30)
        logger.info("STEP 2: LIGHTWEIGHT GRAPH FEATURES")
        logger.info("="*30)
        
        success = run_command(
            "python realtime_graph_features.py",
            "Real-time optimized graph feature generation",
            logger
        )
        
        if not success:
            logger.error("Pipeline failed at graph feature generation step")
            return 1
    
    # Step 3: Model Training
    if success and not args.skip_training:
        logger.info("\n" + "="*30)
        logger.info("STEP 3: RGTAN MODEL TRAINING")
        logger.info("="*30)
        
        success = run_command(
            "python train_decline_rgtan.py",
            "Lightweight RGTAN training for decline prediction",
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
from decline_evaluator import DeclineEvaluator
import pandas as pd
import numpy as np

# Load test results
try:
    results_df = pd.read_csv('../results/decline_predictions.csv')
    y_true = results_df['true_label'].values
    y_scores = results_df['decline_score'].values
    
    # Initialize evaluator
    evaluator = DeclineEvaluator(
        avg_decline_cost=15,
        avg_false_positive_cost=5,
        avg_transaction_value=75
    )
    
    # Generate comprehensive evaluation
    report, threshold_df = evaluator.generate_decline_report(y_true, y_scores)
    
    # Save results
    evaluator.save_decline_report(report, threshold_df, '../results/decline_evaluation_report.txt')
    evaluator.create_decline_dashboard(y_true, y_scores, '../results/decline_dashboard.png')
    
    print('Decline prediction evaluation complete!')
    print(f'Best AUC: {{report[\"model_performance\"][\"auc_roc\"]:.4f}}')
    print(f'Optimal Threshold: {{report[\"optimal_threshold\"]:.3f}}')
    print(f'False Decline Rate: {{report[\"business_impact\"][\"false_decline_rate\"]:.2%}}')
    
except Exception as e:
    print(f'Evaluation failed: {{e}}')
"
"""
        
        success = run_command(
            eval_command,
            "Model evaluation and business impact analysis",
            logger
        )
    
    # Step 5: Start API Server (Optional)
    if success and args.start_api:
        logger.info("\n" + "="*30)
        logger.info("STEP 5: STARTING API SERVER")
        logger.info("="*30)
        
        logger.info("Starting decline prediction API server...")
        logger.info("API will be available at: http://localhost:8000")
        logger.info("Health check: http://localhost:8000/health")
        logger.info("API docs: http://localhost:8000/docs")
        logger.info("Press Ctrl+C to stop the server")
        
        # Start API server (this will run indefinitely)
        try:
            subprocess.run(
                "cd ../api && python decline_api.py",
                shell=True,
                check=True
            )
        except KeyboardInterrupt:
            logger.info("API server stopped by user")
    
    # Pipeline completion
    if success:
        logger.info("\n" + "="*50)
        logger.info("TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("="*50)
        
        # Summary
        logger.info("\nOUTPUT FILES:")
        logger.info("- Processed Data: ../data/processed/")
        logger.info("- Graph Features: ../data/graph/")
        logger.info("- Trained Model: ../models/decline_rgtan_model.pt")
        logger.info("- Predictions: ../results/decline_predictions.csv")
        logger.info("- Evaluation: ../results/decline_evaluation_report.txt")
        logger.info("- Dashboard: ../results/decline_dashboard.png")
        
        logger.info("\nMODEL PERFORMANCE:")
        try:
            # Try to read evaluation results
            with open('../results/decline_evaluation_report.json', 'r') as f:
                report = json.load(f)
                logger.info(f"- AUC-ROC: {report['model_performance']['auc_roc']:.4f}")
                logger.info(f"- Optimal Threshold: {report['optimal_threshold']:.3f}")
                logger.info(f"- False Decline Rate: {report['business_impact']['false_decline_rate']:.2%}")
                logger.info(f"- Net Benefit: ${report['business_impact']['net_benefit']:,.2f}")
        except:
            logger.info("- See evaluation report for detailed metrics")
        
        logger.info("\nNEXT STEPS:")
        logger.info("1. Review evaluation dashboard and report")
        logger.info("2. Test API with sample requests")
        logger.info("3. Deploy to staging/production environment")
        logger.info("4. Set up monitoring and alerting")
        
        if not args.start_api:
            logger.info("\nTo start the API server:")
            logger.info("cd ../api && python decline_api.py")
            logger.info("Or use: python run_training_pipeline.py --start-api")
        
        return 0
    else:
        logger.error("\n" + "="*50)
        logger.error("TRAINING PIPELINE FAILED!")
        logger.error("="*50)
        logger.error("Check logs above for error details")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)