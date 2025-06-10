#!/usr/bin/env python3
"""
Evaluation tools for decline prediction model
Provides business impact analysis and model performance metrics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (roc_auc_score, f1_score, accuracy_score, precision_score, 
                           recall_score, precision_recall_curve, roc_curve, confusion_matrix,
                           classification_report)
import json
from datetime import datetime
import os
from typing import Dict, List, Tuple, Optional

class DeclineEvaluator:
    """Comprehensive evaluation toolkit for decline prediction"""
    
    def __init__(self, avg_decline_cost=15, avg_false_positive_cost=5, avg_transaction_value=75):
        """
        Initialize evaluator with business costs
        
        Args:
            avg_decline_cost: Cost of incorrectly declining a good transaction
            avg_false_positive_cost: Cost of manual review for false positive
            avg_transaction_value: Average transaction value
        """
        self.avg_decline_cost = avg_decline_cost
        self.avg_false_positive_cost = avg_false_positive_cost
        self.avg_transaction_value = avg_transaction_value
        
    def evaluate_model_performance(self, y_true, y_scores, y_pred=None, decline_categories=None):
        """Calculate comprehensive model performance metrics"""
        
        if y_pred is None:
            y_pred = (y_scores > 0.5).astype(int)
        
        # Basic metrics
        auc = roc_auc_score(y_true, y_scores)
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='binary')
        recall = recall_score(y_true, y_pred, average='binary')
        f1 = f1_score(y_true, y_pred, average='binary')
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # Additional rates
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
        false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        # Precision-Recall analysis
        precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_true, y_scores)
        
        # Find optimal threshold (maximize F1 score)
        f1_scores = 2 * (precision_curve * recall_curve) / (precision_curve + recall_curve)
        f1_scores = np.nan_to_num(f1_scores)
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = pr_thresholds[optimal_idx] if optimal_idx < len(pr_thresholds) else 0.5
        
        metrics = {
            'auc_roc': auc,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'specificity': specificity,
            'false_positive_rate': false_positive_rate,
            'false_negative_rate': false_negative_rate,
            'true_positives': int(tp),
            'false_positives': int(fp),
            'true_negatives': int(tn),
            'false_negatives': int(fn),
            'optimal_threshold': optimal_threshold,
            'optimal_f1': f1_scores[optimal_idx]
        }
        
        # Category-specific metrics if available
        if decline_categories is not None:
            category_metrics = {}
            for category in decline_categories.unique():
                if category in ['insufficient_funds', 'security', 'invalid_merchant']:
                    mask = decline_categories == category
                    if mask.sum() > 10:  # Minimum samples
                        cat_auc = roc_auc_score(y_true[mask], y_scores[mask])
                        cat_f1 = f1_score(y_true[mask], y_pred[mask], average='binary')
                        category_metrics[category] = {
                            'auc': cat_auc,
                            'f1': cat_f1,
                            'count': mask.sum()
                        }
            metrics['category_metrics'] = category_metrics
        
        return metrics
    
    def calculate_business_impact(self, y_true, y_scores, threshold=0.5, 
                                transaction_amounts=None, decline_categories=None):
        """Calculate business impact of decline prediction"""
        
        y_pred = (y_scores > threshold).astype(int)
        
        # Confusion matrix components
        tp = ((y_pred == 1) & (y_true == 1)).sum()  # Correctly predicted declines
        fp = ((y_pred == 1) & (y_true == 0)).sum()  # False decline predictions
        fn = ((y_pred == 0) & (y_true == 1)).sum()  # Missed decline predictions
        tn = ((y_pred == 0) & (y_true == 0)).sum()  # Correctly predicted approvals
        
        total_declines = y_true.sum()
        total_approvals = len(y_true) - total_declines
        
        # Business impact calculations
        # Benefit: Correctly identifying transactions that would be declined
        # Cost: False positives that might cause unnecessary friction
        
        # Revenue impact (saved approvals)
        if transaction_amounts is not None:
            # Correctly predicted approvals (avoiding false declines)
            correctly_approved_value = transaction_amounts[
                (y_true == 0) & (y_pred == 0)
            ].sum()
            
            # False decline value (lost revenue)
            false_decline_value = transaction_amounts[
                (y_true == 0) & (y_pred == 1)
            ].sum()
        else:
            correctly_approved_value = tn * self.avg_transaction_value
            false_decline_value = fp * self.avg_transaction_value
        
        # Cost calculations
        false_decline_cost = fp * self.avg_decline_cost  # Cost of declining good transactions
        review_cost = fp * self.avg_false_positive_cost  # Cost of reviewing false positives
        
        # Decline prediction accuracy
        decline_prediction_accuracy = tp / total_declines if total_declines > 0 else 0
        approval_prediction_accuracy = tn / total_approvals if total_approvals > 0 else 0
        
        # False decline rate (most important business metric)
        false_decline_rate = fp / total_approvals if total_approvals > 0 else 0
        
        business_metrics = {
            'total_transactions': len(y_true),
            'total_declines': int(total_declines),
            'total_approvals': int(total_approvals),
            'decline_rate': total_declines / len(y_true),
            
            # Prediction accuracy
            'correctly_predicted_declines': int(tp),
            'missed_decline_predictions': int(fn),
            'false_decline_predictions': int(fp),
            'correctly_predicted_approvals': int(tn),
            
            # Rates
            'decline_prediction_accuracy': decline_prediction_accuracy,
            'approval_prediction_accuracy': approval_prediction_accuracy,
            'false_decline_rate': false_decline_rate,
            
            # Financial impact
            'correctly_approved_value': correctly_approved_value,
            'false_decline_value': false_decline_value,
            'false_decline_cost': false_decline_cost,
            'review_cost': review_cost,
            'net_benefit': correctly_approved_value - false_decline_value - false_decline_cost - review_cost,
            
            # Category analysis
            'category_breakdown': self._analyze_decline_categories(
                y_true, y_pred, decline_categories
            ) if decline_categories is not None else {}
        }
        
        return business_metrics
    
    def _analyze_decline_categories(self, y_true, y_pred, decline_categories):
        """Analyze performance by decline category"""
        
        category_analysis = {}
        
        for category in decline_categories.unique():
            mask = decline_categories == category
            if mask.sum() > 5:  # Minimum samples
                cat_true = y_true[mask]
                cat_pred = y_pred[mask]
                
                # Category-specific confusion matrix
                cat_tp = ((cat_pred == 1) & (cat_true == 1)).sum()
                cat_fp = ((cat_pred == 1) & (cat_true == 0)).sum()
                cat_fn = ((cat_pred == 0) & (cat_true == 1)).sum()
                cat_tn = ((cat_pred == 0) & (cat_true == 0)).sum()
                
                category_analysis[category] = {
                    'total_count': mask.sum(),
                    'actual_declines': cat_true.sum(),
                    'predicted_declines': cat_pred.sum(),
                    'true_positives': int(cat_tp),
                    'false_positives': int(cat_fp),
                    'false_negatives': int(cat_fn),
                    'true_negatives': int(cat_tn),
                    'accuracy': (cat_tp + cat_tn) / len(cat_true) if len(cat_true) > 0 else 0,
                    'precision': cat_tp / (cat_tp + cat_fp) if (cat_tp + cat_fp) > 0 else 0,
                    'recall': cat_tp / (cat_tp + cat_fn) if (cat_tp + cat_fn) > 0 else 0
                }
        
        return category_analysis
    
    def threshold_analysis(self, y_true, y_scores, thresholds=None, transaction_amounts=None):
        """Analyze performance across different thresholds"""
        
        if thresholds is None:
            thresholds = np.arange(0.1, 1.0, 0.05)
        
        threshold_results = []
        
        for threshold in thresholds:
            # Model metrics
            model_metrics = self.evaluate_model_performance(y_true, y_scores, 
                                                           (y_scores > threshold).astype(int))
            
            # Business metrics
            business_metrics = self.calculate_business_impact(y_true, y_scores, threshold,
                                                            transaction_amounts)
            
            # Combine metrics
            result = {
                'threshold': threshold,
                **model_metrics,
                **business_metrics
            }
            
            threshold_results.append(result)
        
        return pd.DataFrame(threshold_results)
    
    def latency_analysis(self, prediction_times: List[float]) -> Dict:
        """Analyze API response latency"""
        
        if not prediction_times:
            return {"error": "No prediction times provided"}
        
        times = np.array(prediction_times)
        
        return {
            'count': len(times),
            'mean_ms': float(np.mean(times)),
            'median_ms': float(np.median(times)),
            'p90_ms': float(np.percentile(times, 90)),
            'p95_ms': float(np.percentile(times, 95)),
            'p99_ms': float(np.percentile(times, 99)),
            'max_ms': float(np.max(times)),
            'min_ms': float(np.min(times)),
            'std_ms': float(np.std(times)),
            'sla_compliance': {
                '100ms': float(np.mean(times < 100)),
                '200ms': float(np.mean(times < 200)),
                '500ms': float(np.mean(times < 500))
            }
        }
    
    def generate_decline_report(self, y_true, y_scores, threshold=0.5, 
                              transaction_amounts=None, decline_categories=None,
                              prediction_times=None):
        """Generate comprehensive decline prediction report"""
        
        # Model performance
        model_metrics = self.evaluate_model_performance(y_true, y_scores, 
                                                       decline_categories=decline_categories)
        
        # Business impact
        business_metrics = self.calculate_business_impact(y_true, y_scores, threshold,
                                                        transaction_amounts, decline_categories)
        
        # Latency analysis
        latency_metrics = None
        if prediction_times:
            latency_metrics = self.latency_analysis(prediction_times)
        
        # Threshold optimization
        threshold_df = self.threshold_analysis(y_true, y_scores, 
                                             transaction_amounts=transaction_amounts)
        
        # Find optimal threshold for business metrics
        optimal_threshold_idx = threshold_df['net_benefit'].idxmax()
        optimal_threshold = threshold_df.loc[optimal_threshold_idx, 'threshold']
        optimal_metrics = threshold_df.loc[optimal_threshold_idx].to_dict()
        
        # Compile report
        report = {
            'report_date': datetime.now().isoformat(),
            'model_performance': model_metrics,
            'business_impact': business_metrics,
            'optimal_threshold': optimal_threshold,
            'optimal_metrics': optimal_metrics,
            'latency_analysis': latency_metrics,
            'recommendations': self._generate_decline_recommendations(
                model_metrics, business_metrics, latency_metrics
            )
        }
        
        return report, threshold_df
    
    def _generate_decline_recommendations(self, model_metrics, business_metrics, latency_metrics):
        """Generate actionable recommendations for decline prediction"""
        
        recommendations = []
        
        # Model performance recommendations
        if model_metrics['auc_roc'] > 0.85:
            recommendations.append("Excellent model performance (AUC > 0.85). Ready for production.")
        elif model_metrics['auc_roc'] > 0.75:
            recommendations.append("Good model performance (AUC > 0.75). Consider feature improvements.")
        else:
            recommendations.append("Model needs improvement (AUC < 0.75). Review training data and features.")
        
        # Business impact recommendations
        false_decline_rate = business_metrics['false_decline_rate']
        if false_decline_rate > 0.05:
            recommendations.append(f"High false decline rate ({false_decline_rate:.1%}). Consider raising threshold.")
        elif false_decline_rate < 0.01:
            recommendations.append(f"Very low false decline rate ({false_decline_rate:.1%}). Could lower threshold for better decline prediction.")
        
        # Net benefit analysis
        if business_metrics['net_benefit'] < 0:
            recommendations.append("Negative net benefit. Model causes more harm than good at current threshold.")
        elif business_metrics['net_benefit'] > 100000:
            recommendations.append("Strong positive net benefit. Consider expanding model usage.")
        
        # Latency recommendations
        if latency_metrics:
            p95_latency = latency_metrics['p95_ms']
            if p95_latency > 200:
                recommendations.append(f"High latency (P95: {p95_latency:.0f}ms). Optimize for real-time use.")
            elif p95_latency < 50:
                recommendations.append(f"Excellent latency (P95: {p95_latency:.0f}ms). Suitable for real-time processing.")
        
        # Category-specific recommendations
        if 'category_breakdown' in business_metrics:
            for category, metrics in business_metrics['category_breakdown'].items():
                if metrics['accuracy'] < 0.7:
                    recommendations.append(f"Low accuracy for {category} declines. Consider category-specific models.")
        
        return recommendations
    
    def create_decline_dashboard(self, y_true, y_scores, output_path=None, 
                               transaction_amounts=None, decline_categories=None):
        """Create visual dashboard for decline prediction evaluation"""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Decline Prediction Model Evaluation Dashboard', fontsize=16)
        
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
        
        axes[0, 1].plot(recall, precision, label='PR Curve')
        axes[0, 1].axhline(y=y_true.mean(), color='k', linestyle='--', 
                          label=f'Baseline ({y_true.mean():.3f})')
        axes[0, 1].set_xlabel('Recall')
        axes[0, 1].set_ylabel('Precision')
        axes[0, 1].set_title('Precision-Recall Curve')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # 3. Score Distribution
        axes[0, 2].hist(y_scores[y_true == 0], bins=30, alpha=0.7, label='Approved', 
                       density=True, color='green')
        axes[0, 2].hist(y_scores[y_true == 1], bins=30, alpha=0.7, label='Declined', 
                       density=True, color='red')
        axes[0, 2].set_xlabel('Decline Score')
        axes[0, 2].set_ylabel('Density')
        axes[0, 2].set_title('Score Distribution')
        axes[0, 2].legend()
        axes[0, 2].grid(True)
        
        # 4. Threshold Analysis - False Decline Rate
        threshold_df = self.threshold_analysis(y_true, y_scores, transaction_amounts=transaction_amounts)
        
        axes[1, 0].plot(threshold_df['threshold'], threshold_df['false_decline_rate'], 
                       'r-', label='False Decline Rate')
        axes[1, 0].plot(threshold_df['threshold'], threshold_df['decline_prediction_accuracy'], 
                       'g-', label='Decline Prediction Accuracy')
        axes[1, 0].set_xlabel('Threshold')
        axes[1, 0].set_ylabel('Rate')
        axes[1, 0].set_title('False Decline vs Prediction Accuracy')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # 5. Business Impact
        axes[1, 1].plot(threshold_df['threshold'], threshold_df['net_benefit'], 'b-')
        axes[1, 1].set_xlabel('Threshold')
        axes[1, 1].set_ylabel('Net Benefit ($)')
        axes[1, 1].set_title('Business Impact by Threshold')
        axes[1, 1].grid(True)
        
        # 6. Confusion Matrix at optimal threshold
        optimal_threshold = threshold_df.loc[threshold_df['net_benefit'].idxmax(), 'threshold']
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
        
        return fig, threshold_df
    
    def save_decline_report(self, report, threshold_df, output_path):
        """Save comprehensive decline prediction report"""
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save JSON report
        json_path = output_path.replace('.txt', '.json')
        with open(json_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Save threshold analysis
        csv_path = output_path.replace('.txt', '_thresholds.csv')
        threshold_df.to_csv(csv_path, index=False)
        
        # Generate text report
        text_report = self._format_decline_report(report)
        
        with open(output_path, 'w') as f:
            f.write(text_report)
        
        print(f"Decline prediction report saved to {output_path}")
        
        return text_report
    
    def _format_decline_report(self, report):
        """Format decline prediction report as readable text"""
        
        text = f"""
DECLINE PREDICTION MODEL EVALUATION REPORT
=========================================
Generated: {report['report_date']}

MODEL PERFORMANCE METRICS
------------------------
AUC-ROC Score: {report['model_performance']['auc_roc']:.4f}
Accuracy: {report['model_performance']['accuracy']:.4f}
Precision: {report['model_performance']['precision']:.4f}
Recall: {report['model_performance']['recall']:.4f}
F1 Score: {report['model_performance']['f1_score']:.4f}
Optimal Threshold: {report['model_performance']['optimal_threshold']:.4f}

BUSINESS IMPACT ANALYSIS
-----------------------
Total Transactions: {report['business_impact']['total_transactions']:,}
Total Declines: {report['business_impact']['total_declines']:,}
Decline Rate: {report['business_impact']['decline_rate']:.2%}

Prediction Performance:
- Correctly Predicted Declines: {report['business_impact']['correctly_predicted_declines']:,}
- Missed Decline Predictions: {report['business_impact']['missed_decline_predictions']:,}
- False Decline Predictions: {report['business_impact']['false_decline_predictions']:,}
- Correctly Predicted Approvals: {report['business_impact']['correctly_predicted_approvals']:,}

Key Metrics:
- Decline Prediction Accuracy: {report['business_impact']['decline_prediction_accuracy']:.2%}
- Approval Prediction Accuracy: {report['business_impact']['approval_prediction_accuracy']:.2%}
- False Decline Rate: {report['business_impact']['false_decline_rate']:.2%}

FINANCIAL IMPACT
---------------
Correctly Approved Value: ${report['business_impact']['correctly_approved_value']:,.2f}
False Decline Value (Lost): ${report['business_impact']['false_decline_value']:,.2f}
False Decline Cost: ${report['business_impact']['false_decline_cost']:,.2f}
Review Cost: ${report['business_impact']['review_cost']:,.2f}
Net Benefit: ${report['business_impact']['net_benefit']:,.2f}

OPTIMAL THRESHOLD ANALYSIS
--------------------------
Optimal Threshold: {report['optimal_threshold']:.3f}
Expected Performance at Optimal Threshold:
- AUC: {report['optimal_metrics']['auc_roc']:.4f}
- False Decline Rate: {report['optimal_metrics']['false_decline_rate']:.2%}
- Net Benefit: ${report['optimal_metrics']['net_benefit']:,.2f}
"""
        
        # Add latency analysis if available
        if report['latency_analysis']:
            text += f"""
LATENCY ANALYSIS
---------------
Predictions Served: {report['latency_analysis']['count']:,}
Mean Response Time: {report['latency_analysis']['mean_ms']:.1f}ms
P95 Response Time: {report['latency_analysis']['p95_ms']:.1f}ms
P99 Response Time: {report['latency_analysis']['p99_ms']:.1f}ms
SLA Compliance:
- <100ms: {report['latency_analysis']['sla_compliance']['100ms']:.1%}
- <200ms: {report['latency_analysis']['sla_compliance']['200ms']:.1%}
- <500ms: {report['latency_analysis']['sla_compliance']['500ms']:.1%}
"""
        
        text += "\nRECOMMENDATIONS\n"
        text += "--------------\n"
        for i, rec in enumerate(report['recommendations'], 1):
            text += f"{i}. {rec}\n"
        
        return text

def main():
    """Example usage of decline evaluation tools"""
    
    # Example data
    np.random.seed(42)
    n_samples = 10000
    
    # Simulate decline prediction results (higher decline rate than fraud)
    y_true = np.random.binomial(1, 0.35, n_samples)  # 35% decline rate
    decline_scores = np.random.beta(2, 3, n_samples)  # Scores between 0-1
    decline_scores[y_true == 1] += 0.2  # Boost decline scores
    decline_scores = np.clip(decline_scores, 0, 1)
    
    # Transaction amounts
    transaction_amounts = np.random.exponential(75, n_samples)
    
    # Initialize evaluator
    evaluator = DeclineEvaluator(
        avg_decline_cost=15, 
        avg_false_positive_cost=5, 
        avg_transaction_value=75
    )
    
    # Generate comprehensive report
    report, threshold_df = evaluator.generate_decline_report(
        y_true, decline_scores, transaction_amounts=transaction_amounts
    )
    
    # Save report
    evaluator.save_decline_report(report, threshold_df, '../results/decline_evaluation_report.txt')
    
    # Create dashboard
    evaluator.create_decline_dashboard(y_true, decline_scores, 
                                     '../results/decline_dashboard.png',
                                     transaction_amounts=transaction_amounts)
    
    print("Decline prediction evaluation complete!")

if __name__ == "__main__":
    main()