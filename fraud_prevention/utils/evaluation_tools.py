#!/usr/bin/env python3
"""
Evaluation and reporting utilities for fraud detection
Provides ROI analysis, model performance metrics, and management reports
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (roc_auc_score, f1_score, average_precision_score, 
                           precision_recall_curve, roc_curve, confusion_matrix)
import json
import yaml
from datetime import datetime
import os

def load_business_config(config_path=None):
    """Load business configuration from YAML file"""
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'fraud_config.yaml')
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        business_config = config.get('business', {})
        return {
            'avg_chargeback_cost': business_config.get('average_chargeback_cost', 85),
            'avg_transaction_value': business_config.get('average_transaction_value', 75),
            'false_positive_review_cost': business_config.get('false_positive_review_cost', 22)
        }
    except Exception as e:
        print(f"Warning: Could not load config file {config_path}: {e}")
        # Return defaults
        return {
            'avg_chargeback_cost': 85,
            'avg_transaction_value': 75,
            'false_positive_review_cost': 22
        }

class FraudEvaluator:
    """Comprehensive fraud detection evaluation toolkit"""
    
    def __init__(self, avg_chargeback_cost=None, avg_transaction_value=None, false_positive_review_cost=None):
        # Load from config if not provided
        if any(param is None for param in [avg_chargeback_cost, avg_transaction_value, false_positive_review_cost]):
            config = load_business_config()
            self.avg_chargeback_cost = avg_chargeback_cost or config['avg_chargeback_cost']
            self.avg_transaction_value = avg_transaction_value or config['avg_transaction_value']
            self.false_positive_review_cost = false_positive_review_cost or config['false_positive_review_cost']
        else:
            self.avg_chargeback_cost = avg_chargeback_cost
            self.avg_transaction_value = avg_transaction_value
            self.false_positive_review_cost = false_positive_review_cost
        
    def evaluate_model_performance(self, y_true, y_scores, y_pred=None):
        """Calculate comprehensive model performance metrics"""
        
        if y_pred is None:
            y_pred = (y_scores > 0.5).astype(int)
        
        # Basic metrics
        auc = roc_auc_score(y_true, y_scores)
        f1 = f1_score(y_true, y_pred, average='macro')
        ap = average_precision_score(y_true, y_scores)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # Calculate rates
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        # Precision-Recall analysis
        precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_true, y_scores)
        
        # Find optimal threshold (F1 score)
        f1_scores = 2 * (precision_curve * recall_curve) / (precision_curve + recall_curve)
        optimal_idx = np.nanargmax(f1_scores)
        optimal_threshold = pr_thresholds[optimal_idx] if optimal_idx < len(pr_thresholds) else 0.5
        
        metrics = {
            'auc_roc': auc,
            'f1_score': f1,
            'average_precision': ap,
            'precision': precision,
            'recall': recall,
            'specificity': specificity,
            'false_positive_rate': false_positive_rate,
            'true_positives': int(tp),
            'false_positives': int(fp),
            'true_negatives': int(tn),
            'false_negatives': int(fn),
            'optimal_threshold': optimal_threshold,
            'optimal_f1': f1_scores[optimal_idx] if not np.isnan(f1_scores[optimal_idx]) else 0
        }
        
        return metrics
    
    def calculate_business_impact(self, y_true, y_scores, threshold=0.5, 
                                transaction_amounts=None):
        """Calculate business impact and ROI of fraud detection"""
        
        y_pred = (y_scores > threshold).astype(int)
        
        # Confusion matrix components
        tp = ((y_pred == 1) & (y_true == 1)).sum()  # Correctly flagged fraud
        fp = ((y_pred == 1) & (y_true == 0)).sum()  # False alarms
        fn = ((y_pred == 0) & (y_true == 1)).sum()  # Missed fraud
        tn = ((y_pred == 0) & (y_true == 0)).sum()  # Correctly passed
        
        # Business calculations
        total_fraud_cases = y_true.sum()
        total_legitimate_cases = len(y_true) - total_fraud_cases
        
        # Cost calculations
        prevented_fraud_cost = tp * self.avg_chargeback_cost
        missed_fraud_cost = fn * self.avg_chargeback_cost
        false_positive_cost = fp * self.false_positive_review_cost  # Manual review cost per false alarm
        
        # Revenue impact
        if transaction_amounts is not None:
            # Use actual transaction amounts
            prevented_fraud_value = transaction_amounts[y_true == 1][y_pred[y_true == 1] == 1].sum()
            missed_fraud_value = transaction_amounts[y_true == 1][y_pred[y_true == 1] == 0].sum()
        else:
            # Use average values
            prevented_fraud_value = tp * self.avg_transaction_value
            missed_fraud_value = fn * self.avg_transaction_value
        
        # Total savings
        total_savings = prevented_fraud_cost - false_positive_cost
        
        # Fraud detection rate
        fraud_detection_rate = tp / total_fraud_cases if total_fraud_cases > 0 else 0
        false_alarm_rate = fp / total_legitimate_cases if total_legitimate_cases > 0 else 0
        
        business_metrics = {
            'total_transactions': len(y_true),
            'total_fraud_cases': int(total_fraud_cases),
            'fraud_rate': total_fraud_cases / len(y_true),
            'prevented_fraud_cases': int(tp),
            'missed_fraud_cases': int(fn),
            'false_alarms': int(fp),
            'fraud_detection_rate': fraud_detection_rate,
            'false_alarm_rate': false_alarm_rate,
            'prevented_fraud_cost': prevented_fraud_cost,
            'missed_fraud_cost': missed_fraud_cost,
            'false_positive_cost': false_positive_cost,
            'total_savings': total_savings,
            'prevented_fraud_value': prevented_fraud_value,
            'missed_fraud_value': missed_fraud_value,
            'net_value_protected': prevented_fraud_value - missed_fraud_value
        }
        
        return business_metrics
    
    def threshold_analysis(self, y_true, y_scores, thresholds=None):
        """Analyze performance across different thresholds"""
        
        if thresholds is None:
            thresholds = np.arange(0.1, 1.0, 0.1)
        
        threshold_results = []
        
        for threshold in thresholds:
            # Model metrics
            model_metrics = self.evaluate_model_performance(y_true, y_scores, 
                                                           (y_scores > threshold).astype(int))
            
            # Business metrics
            business_metrics = self.calculate_business_impact(y_true, y_scores, threshold)
            
            # Combine metrics
            result = {
                'threshold': threshold,
                **model_metrics,
                **business_metrics
            }
            
            threshold_results.append(result)
        
        return pd.DataFrame(threshold_results)
    
    def generate_management_report(self, y_true, y_scores, threshold=0.5, 
                                 transaction_amounts=None, current_baseline=None):
        """Generate comprehensive management report"""
        
        # Model performance
        model_metrics = self.evaluate_model_performance(y_true, y_scores)
        
        # Business impact
        business_metrics = self.calculate_business_impact(y_true, y_scores, threshold, 
                                                        transaction_amounts)
        
        # Comparison with baseline if provided
        baseline_comparison = {}
        if current_baseline:
            improvement = {}
            for key in ['fraud_detection_rate', 'false_alarm_rate', 'total_savings']:
                if key in current_baseline and key in business_metrics:
                    improvement[f'{key}_improvement'] = (
                        business_metrics[key] - current_baseline[key]
                    )
            baseline_comparison = improvement
        
        # Annual projections (assuming this is representative)
        annual_multiplier = 365 / 30  # Assuming 30 days of data
        annual_projections = {
            'annual_fraud_prevented': business_metrics['prevented_fraud_cases'] * annual_multiplier,
            'annual_savings': business_metrics['total_savings'] * annual_multiplier,
            'annual_value_protected': business_metrics['prevented_fraud_value'] * annual_multiplier
        }
        
        # Compile report
        report = {
            'report_date': datetime.now().isoformat(),
            'model_performance': model_metrics,
            'business_impact': business_metrics,
            'annual_projections': annual_projections,
            'baseline_comparison': baseline_comparison,
            'recommendations': self._generate_recommendations(model_metrics, business_metrics)
        }
        
        return report
    
    def _generate_recommendations(self, model_metrics, business_metrics):
        """Generate actionable recommendations"""
        
        recommendations = []
        
        # AUC recommendations
        if model_metrics['auc_roc'] > 0.85:
            recommendations.append("Excellent model performance (AUC > 0.85). Ready for production deployment.")
        elif model_metrics['auc_roc'] > 0.75:
            recommendations.append("Good model performance (AUC > 0.75). Consider feature engineering improvements.")
        else:
            recommendations.append("Model needs improvement (AUC < 0.75). Review feature quality and model architecture.")
        
        # Fraud detection rate
        if business_metrics['fraud_detection_rate'] < 0.6:
            recommendations.append("Low fraud detection rate. Consider lowering threshold or improving model sensitivity.")
        
        # False alarm rate
        if business_metrics['false_alarm_rate'] > 0.05:
            recommendations.append("High false alarm rate. Consider raising threshold or improving model specificity.")
        
        # ROI recommendations
        if business_metrics['total_savings'] < 0:
            recommendations.append("Negative ROI due to high false positive costs. Optimize threshold or improve precision.")
        elif business_metrics['total_savings'] > 100000:
            recommendations.append("Strong positive ROI. Consider expanding model coverage to more transaction types.")
        
        return recommendations
    
    def create_evaluation_dashboard(self, y_true, y_scores, output_path=None, 
                                  transaction_amounts=None):
        """Create visual evaluation dashboard"""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Fraud Detection Model Evaluation Dashboard', fontsize=16)
        
        # 1. ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        auc = roc_auc_score(y_true, y_scores)
        
        axes[0, 0].plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.3f})')
        axes[0, 0].plot([0, 1], [0, 1], 'k--', label='Random')
        axes[0, 0].set_xlabel('False Positive Rate')
        axes[0, 0].set_ylabel('True Positive Rate')
        axes[0, 0].set_title('ROC Curve')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # 2. Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        ap = average_precision_score(y_true, y_scores)
        
        axes[0, 1].plot(recall, precision, label=f'PR Curve (AP = {ap:.3f})')
        axes[0, 1].axhline(y=y_true.mean(), color='k', linestyle='--', 
                          label=f'Baseline ({y_true.mean():.3f})')
        axes[0, 1].set_xlabel('Recall')
        axes[0, 1].set_ylabel('Precision')
        axes[0, 1].set_title('Precision-Recall Curve')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # 3. Score Distribution
        axes[0, 2].hist(y_scores[y_true == 0], bins=30, alpha=0.7, label='Legitimate', 
                       density=True, color='blue')
        axes[0, 2].hist(y_scores[y_true == 1], bins=30, alpha=0.7, label='Fraud', 
                       density=True, color='red')
        axes[0, 2].set_xlabel('Fraud Score')
        axes[0, 2].set_ylabel('Density')
        axes[0, 2].set_title('Score Distribution')
        axes[0, 2].legend()
        axes[0, 2].grid(True)
        
        # 4. Threshold Analysis
        threshold_df = self.threshold_analysis(y_true, y_scores)
        
        axes[1, 0].plot(threshold_df['threshold'], threshold_df['fraud_detection_rate'], 
                       'g-', label='Fraud Detection Rate')
        axes[1, 0].plot(threshold_df['threshold'], threshold_df['false_alarm_rate'], 
                       'r-', label='False Alarm Rate')
        axes[1, 0].set_xlabel('Threshold')
        axes[1, 0].set_ylabel('Rate')
        axes[1, 0].set_title('Detection vs False Alarm Rate')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # 5. Business Impact
        axes[1, 1].plot(threshold_df['threshold'], threshold_df['total_savings'], 'b-')
        axes[1, 1].set_xlabel('Threshold')
        axes[1, 1].set_ylabel('Total Savings ($)')
        axes[1, 1].set_title('Business Impact by Threshold')
        axes[1, 1].grid(True)
        
        # 6. Confusion Matrix at optimal threshold
        optimal_threshold = threshold_df.loc[threshold_df['total_savings'].idxmax(), 'threshold']
        y_pred_optimal = (y_scores > optimal_threshold).astype(int)
        cm = confusion_matrix(y_true, y_pred_optimal)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 2])
        axes[1, 2].set_xlabel('Predicted')
        axes[1, 2].set_ylabel('Actual')
        axes[1, 2].set_title(f'Confusion Matrix (threshold={optimal_threshold:.2f})')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Dashboard saved to {output_path}")
        
        return fig
    
    def save_evaluation_report(self, report, output_path):
        """Save comprehensive evaluation report"""
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save JSON report
        json_path = output_path.replace('.txt', '.json')
        with open(json_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Generate text report
        text_report = self._format_text_report(report)
        
        with open(output_path, 'w') as f:
            f.write(text_report)
        
        print(f"Evaluation report saved to {output_path}")
        
        return text_report
    
    def _format_text_report(self, report):
        """Format report as readable text"""
        
        text = f"""
FRAUD DETECTION MODEL EVALUATION REPORT
======================================
Generated: {report['report_date']}

MODEL PERFORMANCE METRICS
------------------------
AUC-ROC Score: {report['model_performance']['auc_roc']:.4f}
F1 Score: {report['model_performance']['f1_score']:.4f}
Average Precision: {report['model_performance']['average_precision']:.4f}
Precision: {report['model_performance']['precision']:.4f}
Recall: {report['model_performance']['recall']:.4f}
Optimal Threshold: {report['model_performance']['optimal_threshold']:.4f}

BUSINESS IMPACT ANALYSIS
-----------------------
Total Transactions Analyzed: {report['business_impact']['total_transactions']:,}
Total Fraud Cases: {report['business_impact']['total_fraud_cases']:,}
Fraud Rate: {report['business_impact']['fraud_rate']:.2%}

Fraud Detection Performance:
- Prevented Fraud Cases: {report['business_impact']['prevented_fraud_cases']:,}
- Missed Fraud Cases: {report['business_impact']['missed_fraud_cases']:,}
- Detection Rate: {report['business_impact']['fraud_detection_rate']:.2%}

False Alarm Analysis:
- False Alarms: {report['business_impact']['false_alarms']:,}
- False Alarm Rate: {report['business_impact']['false_alarm_rate']:.2%}

FINANCIAL IMPACT
---------------
Prevented Fraud Cost: ${report['business_impact']['prevented_fraud_cost']:,.2f}
Missed Fraud Cost: ${report['business_impact']['missed_fraud_cost']:,.2f}
False Positive Cost: ${report['business_impact']['false_positive_cost']:,.2f}
Net Savings: ${report['business_impact']['total_savings']:,.2f}

Value Protection:
- Prevented Fraud Value: ${report['business_impact']['prevented_fraud_value']:,.2f}
- Missed Fraud Value: ${report['business_impact']['missed_fraud_value']:,.2f}
- Net Value Protected: ${report['business_impact']['net_value_protected']:,.2f}

ANNUAL PROJECTIONS
-----------------
Annual Fraud Cases Prevented: {report['annual_projections']['annual_fraud_prevented']:,.0f}
Annual Savings: ${report['annual_projections']['annual_savings']:,.2f}
Annual Value Protected: ${report['annual_projections']['annual_value_protected']:,.2f}

RECOMMENDATIONS
--------------
"""
        
        for i, rec in enumerate(report['recommendations'], 1):
            text += f"{i}. {rec}\n"
        
        if report['baseline_comparison']:
            text += "\nIMPROVEMENT OVER BASELINE\n"
            text += "------------------------\n"
            for key, value in report['baseline_comparison'].items():
                text += f"{key}: {value:+.2%}\n"
        
        return text

def main():
    """Example usage of evaluation tools"""
    
    # Example data (replace with actual predictions)
    np.random.seed(42)
    n_samples = 10000
    
    # Simulate fraud detection results
    y_true = np.random.binomial(1, 0.02, n_samples)  # 2% fraud rate
    fraud_scores = np.random.beta(2, 8, n_samples)  # Skewed towards low scores
    fraud_scores[y_true == 1] += 0.3  # Boost fraud scores
    fraud_scores = np.clip(fraud_scores, 0, 1)
    
    # Initialize evaluator (loads business costs from config)
    evaluator = FraudEvaluator()
    
    # Generate comprehensive report
    report = evaluator.generate_management_report(y_true, fraud_scores)
    
    # Save report
    evaluator.save_evaluation_report(report, '../results/evaluation_report.txt')
    
    # Create dashboard
    evaluator.create_evaluation_dashboard(y_true, fraud_scores, 
                                        '../results/evaluation_dashboard.png')
    
    print("Evaluation complete!")

if __name__ == "__main__":
    main()