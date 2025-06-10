#!/usr/bin/env python3
"""
Real-time API server for decline prediction
FastAPI-based service optimized for <100ms response time
"""

import os
import sys
import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json
import pickle
import hashlib

import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from contextlib import asynccontextmanager

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from decline_prediction.scripts.train_decline_rgtan import LightweightRGTAN
from decline_prediction.scripts.realtime_graph_features import RealtimeGraphFeatureEngine

# Global variables for model and cache
model = None
scaler = None
graph_scaler = None
feature_engine = None
entity_lookup = None
feature_columns = None
graph_feature_columns = None
model_config = None

# Performance tracking
prediction_times = []
MAX_TRACKING_SIZE = 1000

class TransactionRequest(BaseModel):
    """Request model for transaction decline prediction"""
    transaction_id: str = Field(..., description="Unique transaction identifier")
    card_number: str = Field(..., description="Card number (will be hashed)")
    amount: float = Field(..., gt=0, description="Transaction amount")
    currency_id: str = Field(default="USD", description="Currency code")
    merchant_id: str = Field(..., description="Merchant identifier")
    customer_email: str = Field(..., description="Customer email (will be hashed)")
    customer_ip: str = Field(..., description="Customer IP address (will be hashed)")
    bill_country: str = Field(default="US", description="Billing country")
    issuer_country: str = Field(default="US", description="Card issuer country")
    card_type: str = Field(default="CREDIT", description="DEBIT or CREDIT")
    card_brand: str = Field(default="VISA", description="Card brand")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat(), 
                          description="Transaction timestamp")
    
    # Optional fields
    site_tag_id: Optional[str] = Field(default="UNKNOWN", description="Site tag ID")
    processor_id: Optional[str] = Field(default="UNKNOWN", description="Processor ID")
    bill_zip: Optional[str] = Field(default=None, description="Billing ZIP code")
    BIN: Optional[str] = Field(default="UNKNOWN", description="Bank Identification Number")

class DeclineResponse(BaseModel):
    """Response model for decline prediction"""
    transaction_id: str
    decline_probability: float = Field(..., ge=0, le=1, description="Probability of decline (0-1)")
    predicted_decline: bool = Field(..., description="Predicted decline (True/False)")
    decline_reason: Optional[str] = Field(default=None, description="Most likely decline reason")
    confidence_score: float = Field(..., ge=0, le=1, description="Prediction confidence")
    risk_factors: List[str] = Field(default=[], description="Key risk factors")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    avg_response_time_ms: float
    predictions_served: int
    uptime_seconds: float

# Initialize FastAPI app
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await load_model()
    yield
    # Shutdown
    pass

app = FastAPI(
    title="Decline Prediction API",
    description="Real-time transaction decline prediction using RGTAN",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Startup time
startup_time = time.time()

def hash_sensitive_data(value: str) -> str:
    """Hash sensitive information for privacy"""
    if not value or value == '':
        return 'MISSING'
    return hashlib.sha256(str(value).encode()).hexdigest()[:16]

async def load_model():
    """Load trained model and dependencies"""
    global model, scaler, graph_scaler, feature_engine, entity_lookup
    global feature_columns, graph_feature_columns, model_config
    
    print("Loading decline prediction model...")
    
    try:
        # Load model
        model_path = '../models/decline_rgtan_model.pt'
        checkpoint = torch.load(model_path, map_location='cpu')
        
        model_config = checkpoint['model_config']
        feature_columns = checkpoint['feature_columns']
        graph_feature_columns = checkpoint['graph_feature_columns']
        scaler = checkpoint['scaler']
        graph_scaler = checkpoint['graph_scaler']
        
        # Initialize model
        model = LightweightRGTAN(
            in_feats=model_config['in_feats'],
            graph_feats=model_config['graph_feats'],
            hidden_dim=model_config['hidden_dim'],
            n_layers=model_config['n_layers'],
            n_classes=model_config['n_classes'],
            heads=model_config['heads'],
            activation=torch.nn.ReLU(),
            drop=[0, 0, 0],  # No dropout for inference
            device='cpu'
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Initialize feature engine
        feature_engine = RealtimeGraphFeatureEngine({
            'max_card_history': 3,      # Reduced for speed
            'max_ip_history': 5,
            'max_merchant_history': 10,
            'time_window_hours': 1,
            'use_redis': False          # Can be enabled for production
        })
        
        # Load entity lookup dictionaries
        try:
            with open('../data/graph/entity_lookup_dicts.pkl', 'rb') as f:
                entity_lookup = pickle.load(f)
        except:
            entity_lookup = {}
            print("Warning: Entity lookup dictionaries not found")
        
        print("Model loaded successfully!")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

def preprocess_transaction(request: TransactionRequest) -> Dict:
    """Preprocess transaction for real-time prediction"""
    
    # Parse timestamp
    try:
        issue_date = pd.to_datetime(request.timestamp)
    except:
        issue_date = datetime.now()
    
    # Create transaction record
    transaction = {
        'trans_id': 0,  # Temporary ID
        'issue_date': issue_date,
        'amount': request.amount,
        'currency_id': request.currency_id,
        'site_tag_id': request.merchant_id,
        'processor_id': request.processor_id or 'UNKNOWN',
        'DEBITCREDIT': request.card_type,
        'BRAND': request.card_brand,
        'ISSUERCOUNTRY': request.issuer_country,
        'bill_country': request.bill_country,
        'bill_zip': request.bill_zip,
        'BIN': request.BIN or 'UNKNOWN',
        
        # Hashed sensitive data
        'entity_card': hash_sensitive_data(request.card_number),
        'entity_email': hash_sensitive_data(request.customer_email),
        'entity_ip': hash_sensitive_data(request.customer_ip),
        'entity_merchant': request.merchant_id,
        'entity_bin': request.BIN or 'UNKNOWN'
    }
    
    # Compute basic features
    hour = issue_date.hour
    day_of_week = issue_date.dayofweek
    
    # Amount features
    amount_log = np.log1p(request.amount)
    is_round_amount = int(request.amount % 10 == 0)
    amount_cents = int((request.amount * 100) % 100)
    is_exact_dollar = int(amount_cents == 0)
    
    # Temporal features
    is_weekend = int(day_of_week >= 5)
    is_business_hours = int(9 <= hour <= 17)
    is_night = int(hour >= 22 or hour <= 6)
    
    # Card and geographic features
    is_debit = int(request.card_type == 'DEBIT')
    is_credit = int(request.card_type == 'CREDIT')
    is_domestic = int(request.issuer_country == request.bill_country)
    has_billing_zip = int(request.bill_zip is not None and request.bill_zip != '')
    
    # Add computed features
    transaction.update({
        'hour': hour,
        'day_of_week': day_of_week,
        'amount_log': amount_log,
        'is_round_amount': is_round_amount,
        'is_exact_dollar': is_exact_dollar,
        'is_weekend': is_weekend,
        'is_business_hours': is_business_hours,
        'is_night': is_night,
        'is_debit': is_debit,
        'is_credit': is_credit,
        'is_domestic': is_domestic,
        'has_billing_zip': has_billing_zip
    })
    
    return transaction

def get_entity_features(transaction: Dict) -> Dict:
    """Get entity-based features from cache/lookup"""
    
    entity_features = {}
    
    # Default values
    defaults = {
        'card_decline_rate': 0.0,
        'ip_decline_rate': 0.0,
        'merchant_decline_rate': 0.0,
        'neighbor_count': 0,
        'neighbor_decline_rate': 0.0,
        'neighbor_avg_amount': 0.0,
        'neighbor_velocity': 0.0
    }
    
    # Get entity statistics from lookup
    if entity_lookup:
        for entity_type in ['entity_card', 'entity_ip', 'entity_merchant']:
            entity_value = transaction.get(entity_type)
            if entity_value and entity_type in entity_lookup:
                stats = entity_lookup[entity_type].get(entity_value, {})
                decline_rate = stats.get('decline_rate', 0.0)
                entity_features[f'{entity_type.split("_")[1]}_decline_rate'] = decline_rate
    
    # Apply defaults for missing features
    for key, default_value in defaults.items():
        if key not in entity_features:
            entity_features[key] = default_value
    
    return entity_features

def extract_model_features(transaction: Dict, entity_features: Dict) -> np.ndarray:
    """Extract features in the format expected by the model"""
    
    # Prepare feature vector based on training columns
    feature_vector = []
    
    # Basic numeric features (in expected order)
    basic_features = [
        'amount_log', 'hour', 'day_of_week', 'is_weekend', 'is_business_hours', 'is_night',
        'is_round_amount', 'is_exact_dollar', 'is_debit', 'is_credit', 'is_domestic', 'has_billing_zip'
    ]
    
    for feature in basic_features:
        feature_vector.append(transaction.get(feature, 0))
    
    # Entity features (velocity, etc.)
    entity_velocity_features = [
        'entity_card_hours_since_last', 'entity_email_hours_since_last', 
        'entity_ip_hours_since_last', 'entity_merchant_hours_since_last'
    ]
    
    for feature in entity_velocity_features:
        # Default to 24 hours (normal interval)
        feature_vector.append(24.0)
    
    # Add other velocity features with defaults
    velocity_defaults = [1.0, 1.0, 1.0, 1.0]  # txn_per_day and amount_ratio features
    feature_vector.extend(velocity_defaults)
    
    return np.array(feature_vector, dtype=np.float32)

def extract_graph_features(entity_features: Dict) -> np.ndarray:
    """Extract graph features for the model"""
    
    graph_feature_order = [
        'neighbor_count', 'neighbor_decline_rate', 'neighbor_avg_amount',
        'neighbor_velocity', 'card_decline_rate', 'ip_decline_rate', 'merchant_decline_rate'
    ]
    
    graph_vector = []
    for feature in graph_feature_order:
        graph_vector.append(entity_features.get(feature, 0.0))
    
    return np.array(graph_vector, dtype=np.float32)

def predict_decline(transaction: Dict, entity_features: Dict) -> Dict:
    """Make decline prediction"""
    
    # Extract features
    features = extract_model_features(transaction, entity_features)
    graph_features = extract_graph_features(entity_features)
    
    # Standardize features
    features_scaled = scaler.transform(features.reshape(1, -1))
    graph_features_scaled = graph_scaler.transform(graph_features.reshape(1, -1))
    
    # Convert to tensors
    features_tensor = torch.FloatTensor(features_scaled)
    graph_features_tensor = torch.FloatTensor(graph_features_scaled)
    
    # Model prediction
    with torch.no_grad():
        outputs = model(None, features_tensor, graph_features_tensor)
        
        # Main decline prediction
        decline_probs = F.softmax(outputs['decline'], dim=1)
        decline_prob = decline_probs[0, 1].item()
        predicted_decline = decline_prob > 0.5
        
        # Task-specific predictions
        task_probs = {}
        for task in ['insufficient_funds', 'security', 'invalid_merchant']:
            if task in outputs:
                task_prob = F.softmax(outputs[task], dim=1)[0, 1].item()
                task_probs[task] = task_prob
    
    # Determine most likely decline reason
    decline_reason = None
    if predicted_decline:
        max_task = max(task_probs.items(), key=lambda x: x[1])
        if max_task[1] > 0.3:  # Threshold for confidence
            decline_reason = max_task[0].replace('_', ' ').title()
    
    # Risk factors
    risk_factors = []
    if decline_prob > 0.3:
        if entity_features.get('card_decline_rate', 0) > 0.1:
            risk_factors.append("High card decline rate")
        if entity_features.get('ip_decline_rate', 0) > 0.1:
            risk_factors.append("High IP decline rate")
        if transaction.get('is_night', 0):
            risk_factors.append("Night time transaction")
        if transaction.get('amount', 0) > 1000:
            risk_factors.append("High amount transaction")
        if not transaction.get('is_domestic', 1):
            risk_factors.append("International transaction")
    
    return {
        'decline_probability': decline_prob,
        'predicted_decline': predicted_decline,
        'decline_reason': decline_reason,
        'confidence_score': max(decline_prob, 1 - decline_prob),
        'risk_factors': risk_factors
    }

@app.post("/predict", response_model=DeclineResponse)
async def predict_transaction_decline(request: TransactionRequest):
    """Predict if a transaction will be declined"""
    
    start_time = time.time()
    
    try:
        # Check if model is loaded
        if model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        # Preprocess transaction
        transaction = preprocess_transaction(request)
        
        # Get entity features
        entity_features = get_entity_features(transaction)
        
        # Make prediction
        prediction = predict_decline(transaction, entity_features)
        
        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000
        
        # Track performance
        prediction_times.append(processing_time)
        if len(prediction_times) > MAX_TRACKING_SIZE:
            prediction_times.pop(0)
        
        # Create response
        response = DeclineResponse(
            transaction_id=request.transaction_id,
            decline_probability=prediction['decline_probability'],
            predicted_decline=prediction['predicted_decline'],
            decline_reason=prediction['decline_reason'],
            confidence_score=prediction['confidence_score'],
            risk_factors=prediction['risk_factors'],
            processing_time_ms=processing_time
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict/batch")
async def predict_batch_transactions(requests: List[TransactionRequest]):
    """Predict decline for multiple transactions"""
    
    if len(requests) > 100:
        raise HTTPException(status_code=400, detail="Batch size limited to 100 transactions")
    
    results = []
    for request in requests:
        try:
            result = await predict_transaction_decline(request)
            results.append(result)
        except Exception as e:
            # Add error response for failed predictions
            error_response = DeclineResponse(
                transaction_id=request.transaction_id,
                decline_probability=0.0,
                predicted_decline=False,
                confidence_score=0.0,
                risk_factors=["Prediction error"],
                processing_time_ms=0.0
            )
            results.append(error_response)
    
    return results

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    
    avg_response_time = np.mean(prediction_times) if prediction_times else 0.0
    uptime = time.time() - startup_time
    
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        avg_response_time_ms=avg_response_time,
        predictions_served=len(prediction_times),
        uptime_seconds=uptime
    )

@app.get("/metrics")
async def get_metrics():
    """Get detailed performance metrics"""
    
    if not prediction_times:
        return {"message": "No predictions served yet"}
    
    times = np.array(prediction_times)
    
    return {
        "predictions_served": len(prediction_times),
        "avg_response_time_ms": float(np.mean(times)),
        "median_response_time_ms": float(np.median(times)),
        "p95_response_time_ms": float(np.percentile(times, 95)),
        "p99_response_time_ms": float(np.percentile(times, 99)),
        "max_response_time_ms": float(np.max(times)),
        "min_response_time_ms": float(np.min(times)),
        "uptime_seconds": time.time() - startup_time
    }

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "service": "Decline Prediction API",
        "version": "1.0.0",
        "description": "Real-time transaction decline prediction using RGTAN",
        "endpoints": {
            "predict": "POST /predict - Predict single transaction decline",
            "batch": "POST /predict/batch - Predict multiple transactions",
            "health": "GET /health - Health check",
            "metrics": "GET /metrics - Performance metrics"
        },
        "model_loaded": model is not None
    }

if __name__ == "__main__":
    # Run the API server
    uvicorn.run(
        "decline_api:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1,  # Single worker for model consistency
        loop="asyncio"
    )