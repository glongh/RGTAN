#!/usr/bin/env python3
"""
Safe pipeline runner with comprehensive error handling
"""

import os
import sys
import subprocess
import logging
import traceback
from datetime import datetime

def setup_logging():
    """Setup comprehensive logging"""
    log_dir = '../logs'
    os.makedirs(log_dir, exist_ok=True)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # File handler
    file_handler = logging.FileHandler(
        os.path.join(log_dir, f'pipeline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    )
    file_handler.setFormatter(formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # Setup logger
    logger = logging.getLogger('pipeline')
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def run_step(step_name, command, logger, required=True):
    """Run a pipeline step with error handling"""
    logger.info(f"Starting: {step_name}")
    logger.info(f"Command: {command}")
    
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            check=True,
            timeout=3600  # 1 hour timeout
        )
        
        if result.stdout:
            logger.info(f"Output: {result.stdout[:1000]}...")  # Limit output
        
        logger.info(f"✅ Completed: {step_name}")
        return True
        
    except subprocess.TimeoutExpired:
        logger.error(f"❌ Timeout: {step_name} took more than 1 hour")
        return False
        
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Failed: {step_name}")
        logger.error(f"Return code: {e.returncode}")
        if e.stderr:
            logger.error(f"Error output: {e.stderr}")
        
        if required:
            logger.error(f"Pipeline stopped due to required step failure: {step_name}")
            return False
        else:
            logger.warning(f"Optional step failed, continuing: {step_name}")
            return True
            
    except Exception as e:
        logger.error(f"❌ Unexpected error in {step_name}: {e}")
        logger.error(traceback.format_exc())
        return False

def validate_environment(logger):
    """Validate environment before running pipeline"""
    logger.info("Validating environment...")
    
    # Check Python packages
    required_packages = ['pandas', 'numpy', 'torch', 'dgl', 'sklearn', 'yaml']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f"Missing packages: {missing_packages}")
        return False
    
    # Check CUDA availability
    try:
        import torch
        if torch.cuda.is_available():
            logger.info(f"CUDA available: {torch.cuda.get_device_name()}")
        else:
            logger.info("CUDA not available, using CPU")
    except:
        logger.warning("Could not check CUDA availability")
    
    # Check disk space (simplified)
    try:
        import shutil
        total, used, free = shutil.disk_usage('../data')
        free_gb = free // (1024**3)
        logger.info(f"Free disk space: {free_gb}GB")
        
        if free_gb < 5:
            logger.warning("Low disk space (< 5GB)")
    except:
        logger.warning("Could not check disk space")
    
    logger.info("✅ Environment validation passed")
    return True

def main():
    """Main pipeline execution with comprehensive error handling"""
    logger = setup_logging()
    
    logger.info("=" * 50)
    logger.info("FRAUD DETECTION PIPELINE - SAFE EXECUTION")
    logger.info("=" * 50)
    
    # Validate environment
    if not validate_environment(logger):
        logger.error("Environment validation failed")
        return 1
    
    # Step 1: Validate existing data
    logger.info("\n" + "=" * 30)
    logger.info("STEP 1: DATA VALIDATION")
    logger.info("=" * 30)
    
    if not run_step(
        "Data validation",
        "python validate_pipeline_data.py",
        logger,
        required=False  # Continue even if validation shows issues
    ):
        logger.warning("Data validation had issues, but continuing...")
    
    # Step 2: Generate graph features (if needed)
    logger.info("\n" + "=" * 30)
    logger.info("STEP 2: GRAPH GENERATION")
    logger.info("=" * 30)
    
    # Check if graph exists
    if os.path.exists('../data/graph/transaction_graph.dgl'):
        logger.info("Graph already exists, skipping generation")
        graph_success = True
    else:
        graph_success = run_step(
            "Graph feature generation",
            "python generate_graph_features_fast.py",
            logger,
            required=True
        )
    
    if not graph_success:
        logger.error("Graph generation failed")
        return 1
    
    # Step 3: Model training
    logger.info("\n" + "=" * 30)
    logger.info("STEP 3: MODEL TRAINING")
    logger.info("=" * 30)
    
    training_success = run_step(
        "Model training",
        "python train_fraud_rgtan.py",
        logger,
        required=True
    )
    
    if not training_success:
        logger.error("Model training failed")
        return 1
    
    # Step 4: Validation (optional)
    logger.info("\n" + "=" * 30)
    logger.info("STEP 4: POST-TRAINING VALIDATION")
    logger.info("=" * 30)
    
    # Check if model file was created
    model_path = '../models/fraud_rgtan_model.pt'
    if os.path.exists(model_path):
        logger.info(f"✅ Model file created: {model_path}")
        
        # Get model file size
        model_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
        logger.info(f"Model size: {model_size:.1f}MB")
    else:
        logger.error(f"❌ Model file not found: {model_path}")
        return 1
    
    # Check if results were created
    results_path = '../results/top_fraud_predictions.csv'
    if os.path.exists(results_path):
        logger.info(f"✅ Results file created: {results_path}")
        
        # Check results file
        try:
            import pandas as pd
            results_df = pd.read_csv(results_path)
            logger.info(f"Results contain {len(results_df)} predictions")
        except Exception as e:
            logger.warning(f"Could not validate results file: {e}")
    else:
        logger.warning(f"Results file not found: {results_path}")
    
    # Success
    logger.info("\n" + "=" * 50)
    logger.info("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
    logger.info("=" * 50)
    
    logger.info("\nOutput files:")
    logger.info(f"- Model: {model_path}")
    logger.info(f"- Results: {results_path}")
    logger.info(f"- Logs: {logger.handlers[0].baseFilename}")
    
    return 0

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n❌ Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Pipeline failed with unexpected error: {e}")
        traceback.print_exc()
        sys.exit(1)