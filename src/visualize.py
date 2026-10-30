"""
Publication-Quality Visualization Suite for Uncertainty-Aware IDS.
Generates:
1. Fig 1: Uncertainty distributions under In-Distribution vs Unseen Attack vs Cross-Dataset shift
2. Fig 2: AUROC curves for Error Detection (distinguishing correct vs incorrect predictions)
3. Fig 3: Risk-Coverage tradeoff curves across uncertainty thresholds
4. Fig 4: Unseen attack detection ROC & separation analysis
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Styling for academic paper figures
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'font.family': 'sans-serif'
})

def plot_uncertainty_distributions(dist_dict: dict, save_path: str):
    """
    Fig 1: Density / Histogram comparing uncertainty distributions under:
    - In-Distribution Known Benign
    - In-Distribution Known Attack
    - Unseen Attack (Withheld)
    - Cross-Dataset Shift (CIC-IDS2017)
    """
    plt.figure(figsize=(10, 6))
    colors = {
        'In-Distribution Benign': '#2ecc71',
        'In-Distribution Attack': '#3498db',
        'Unseen Attack': '#e67e22',
        'Cross-Dataset Shift': '#e74c3c'
    }

    for label, scores in dist_dict.items():
        c = colors.get(label, '#95a5a6')
        # Histogram with KDE approximation
        plt.hist(scores, bins=35, density=True, alpha=0.45, label=f"{label} (Mean={np.mean(scores):.2f})", color=c)

    plt.xlabel('Estimated Uncertainty Score U(x)')
    plt.ylabel('Density')
    plt.title('Uncertainty Distribution Across Traffic Conditions', fontweight='bold')
    plt.legend(frameon=True, loc='upper center')
    plt.xlim(0.0, 1.0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Visualize] Saved: {save_path}")

def plot_error_detection_roc(methods_dict: dict, y_true: np.ndarray, y_pred: np.ndarray, save_path: str):
    """
    Fig 2: ROC curves comparing all uncertainty methods for error identification.
    Evaluates: Does higher uncertainty correspond to actual classifier errors?
    """
    plt.figure(figsize=(8, 7))
    errors = (y_true != y_pred).astype(int)
    
    colors = ['#34495e', '#2980b9', '#27ae60', '#8e44ad']
    linestyles = ['-', '--', '-.', ':']

    for i, (method_name, scores) in enumerate(methods_dict.items()):
        fpr, tpr, _ = roc_curve(errors, scores)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=colors[i % len(colors)], linestyle=linestyles[i % len(linestyles)],
                 lw=2.2, label=f"{method_name} (AUROC = {roc_auc:.3f})")

    plt.plot([0, 1], [0, 1], color='#bdc3c7', lw=1.5, linestyle='--', label='Random Chance (AUROC = 0.500)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (Correctly Classified Flagged as Uncertain)')
    plt.ylabel('True Positive Rate (Misclassified Flagged as Uncertain)')
    plt.title('AUROC for Prediction Error Identification', fontweight='bold')
    plt.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Visualize] Saved: {save_path}")

def plot_risk_coverage_curves(curves_dict: dict, save_path: str):
    """
    Fig 3: Selective Classification / Abstention Risk-Coverage Trade-off.
    Coverage on X-axis (from 0.3 to 1.0), Risk (Error Rate) on Y-axis.
    """
    plt.figure(figsize=(9, 6))
    colors = ['#34495e', '#2980b9', '#27ae60', '#8e44ad']
    markers = ['o', 's', '^', 'd']

    for i, (method_name, df_rc) in enumerate(curves_dict.items()):
        # Filter for coverage between 30% and 100%
        df_filtered = df_rc[(df_rc['coverage'] >= 0.35) & (df_rc['coverage'] <= 1.0)].sort_values('coverage')
        plt.plot(df_filtered['coverage'] * 100, df_filtered['risk'] * 100,
                 color=colors[i % len(colors)], lw=2.2, marker=markers[i % len(markers)],
                 markevery=max(1, len(df_filtered)//8), label=method_name)

    plt.xlabel('Coverage (% of Predictions Accepted)')
    plt.ylabel('Risk (% Error Rate in Accepted Predictions)')
    plt.title('Risk-Coverage Tradeoff Under Confidence-Based Abstention', fontweight='bold')
    plt.legend(frameon=True, loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Visualize] Saved: {save_path}")

def plot_unseen_attack_detection_roc(methods_dict: dict, id_scores: np.ndarray, unseen_scores: np.ndarray, save_path: str):
    """
    Fig 4: ROC curve evaluating detection of withheld unseen attacks (Open-Set Detection).
    """
    plt.figure(figsize=(8, 7))
    y_test_ood = np.concatenate([np.zeros(len(id_scores)), np.ones(len(unseen_scores))])
    colors = ['#34495e', '#2980b9', '#27ae60', '#8e44ad']

    for i, (method_name, _) in enumerate(methods_dict.items()):
        scores_id = methods_dict[method_name]['id']
        scores_unseen = methods_dict[method_name]['unseen']
        combined = np.concatenate([scores_id, scores_unseen])
        
        fpr, tpr, _ = roc_curve(y_test_ood, combined)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=colors[i % len(colors)], lw=2.2, label=f"{method_name} (AUROC = {roc_auc:.3f})")

    plt.plot([0, 1], [0, 1], color='#bdc3c7', lw=1.5, linestyle='--', label='Baseline Random (0.500)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Alarm Rate on Known Traffic')
    plt.ylabel('Detection Rate on Unseen Attacks')
    plt.title('Detection of Unseen Zero-Day Attacks via Uncertainty', fontweight='bold')
    plt.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Visualize] Saved: {save_path}")
