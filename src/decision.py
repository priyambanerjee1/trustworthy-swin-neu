import numpy as np
import pandas as pd
from src.config import Config

def compute_trust_score(calibrated_conf, normalized_uncertainty):
    return calibrated_conf * (1.0 - normalized_uncertainty)

def assign_decisions(trust_scores, high_t=Config.HIGH_TRUST_THRESH, low_t=Config.LOW_TRUST_THRESH):
    decisions = []
    for s in trust_scores:
        if s >= high_t:
            decisions.append("Accept")
        elif s >= low_t:
            decisions.append("Human Review")
        else:
            decisions.append("Reject")
    return decisions

def compute_risk_coverage(scores, true_labels, preds):
    sorted_indices = np.argsort(-scores)
    sorted_labels = true_labels[sorted_indices]
    sorted_preds = preds[sorted_indices]
    
    n = len(scores)
    coverages = np.arange(1, n + 1) / n
    risks = np.cumsum(sorted_preds != sorted_labels) / np.arange(1, n + 1)
    return coverages, risks

def evaluate_threshold_sweep(t_scores, preds, labels, thresh_list=[0.95, 0.90, 0.85, 0.80, 0.75]):
    sweep_records = []
    total = len(labels)
    for tau in thresh_list:
        accepted = t_scores >= tau
        acc_count = int(np.sum(accepted))
        cov = acc_count / total
        sel_acc = float(np.mean(preds[accepted] == labels[accepted])) if acc_count > 0 else 0.0
        sweep_records.append({
            "threshold": tau,
            "accepted": acc_count,
            "review": total - acc_count,
            "coverage": cov,
            "selective_accuracy": sel_acc
        })
    return pd.DataFrame(sweep_records)
