import numpy as np
from typing import Dict, Any, List
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

def compute_evaluation_metrics(y_true: List[int], y_pred: List[int], class_names: List[str]) -> Dict[str, Any]:
    """
    Computes rigorous multi-class classification metrics:
    - Test Accuracy
    - Macro-averaged Precision, Recall, and F1-score
    - Weighted F1-score
    - Per-class metrics
    - Confusion matrix
    """
    y_true_np = np.array(y_true)
    y_pred_np = np.array(y_pred)
    
    acc = float(accuracy_score(y_true_np, y_pred_np))
    macro_prec = float(precision_score(y_true_np, y_pred_np, average="macro", zero_division=0))
    macro_rec = float(recall_score(y_true_np, y_pred_np, average="macro", zero_division=0))
    macro_f1 = float(f1_score(y_true_np, y_pred_np, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true_np, y_pred_np, average="weighted", zero_division=0))
    
    # Classification Report
    report = classification_report(
        y_true_np,
        y_pred_np,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )
    
    per_class = []
    for cls in class_names:
        if cls in report:
            per_class.append({
                "class_name": cls,
                "precision": round(float(report[cls]["precision"]) * 100, 2),
                "recall": round(float(report[cls]["recall"]) * 100, 2),
                "f1_score": round(float(report[cls]["f1-score"]) * 100, 2),
                "support": int(report[cls]["support"])
            })
        else:
            per_class.append({
                "class_name": cls,
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0,
                "support": 0
            })
            
    # Confusion Matrix
    cm = confusion_matrix(y_true_np, y_pred_np, labels=list(range(len(class_names))))
    
    # Generate comprehensive academic report
    report_md = f"""# Food Image Classification: Model Evaluation Report

**Project Title:** ANN-Based Classification of Food-101 Alternative Image Set Using CNN  
**Student:** KANCHARLA POOJITHA (Reg: 2023002679, Roll: 10, Project: P35)  
**Syllabus Unit & Outcome:** Unit 3 / CO3  
**Optimization:** Mini-Batch Gradient Descent (Adam Optimizer)  

---

## 1. Executive Summary & Core Metrics

| Metric | Score | Definition & Interpretation |
| :--- | :--- | :--- |
| **Test Accuracy** | **{acc * 100:.2f}%** | Proportion of correctly classified food images over all test samples. |
| **Macro Precision** | **{macro_prec * 100:.2f}%** | Unweighted mean of per-class precision; penalizes false positives across all categories equally. |
| **Macro Recall** | **{macro_rec * 100:.2f}%** | Unweighted mean of per-class recall (sensitivity); assesses coverage across classes. |
| **Macro F1-Score** | **{macro_f1 * 100:.2f}%** | Harmonic mean of macro precision and recall, balancing both metrics. |
| **Weighted F1-Score**| **{weighted_f1 * 100:.2f}%** | F1-score weighted by support per class, accounting for any class imbalance. |

---

## 2. Multi-Class Averaging Methodology
In multi-class evaluation (Food-101 domain), **Macro Averaging** calculates the metric independently for each category and then averages them with equal weight. This ensures that frequent classes do not overpower rare classes in the overall evaluation benchmark.

## 3. Loss Function & Optimization Behavior
The model is trained using **Categorical Cross-Entropy Loss** with **Mini-Batch Gradient Descent**.
By computing gradients over batches of size 16 to 32, the optimization leverages vectorized GPU/CPU execution while maintaining stochastic noise that assists in escaping saddle points and local minima.
"""

    return {
        "test_accuracy": round(acc * 100, 2),
        "macro_precision": round(macro_prec * 100, 2),
        "macro_recall": round(macro_rec * 100, 2),
        "macro_f1_score": round(macro_f1 * 100, 2),
        "weighted_f1_score": round(weighted_f1 * 100, 2),
        "total_test_samples": len(y_true),
        "per_class_metrics": per_class,
        "confusion_matrix_labels": class_names,
        "confusion_matrix_matrix": cm.tolist(),
        "report_markdown": report_md
    }
