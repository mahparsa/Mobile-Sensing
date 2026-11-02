"""Metrics, summaries and plots."""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix,
    f1_score, recall_score, roc_auc_score, roc_curve,
)

METRICS = ["accuracy", "f1_score", "auc", "balanced_accuracy", "sensitivity", "specificity"]


def compute_metrics(y_true, y_pred, y_prob):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred, average="weighted"),
        "auc": roc_auc_score(y_true, y_prob),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "sensitivity": recall_score(y_true, y_pred, pos_label=1),
        "specificity": tn / (tn + fp) if (tn + fp) else 0.0,
    }


def format_metrics(m):
    return (f"Acc={m['accuracy']:.4f}, F1={m['f1_score']:.4f}, AUC={m['auc']:.4f}, "
            f"BalAcc={m['balanced_accuracy']:.4f}, Sens={m['sensitivity']:.4f}, "
            f"Spec={m['specificity']:.4f}")


def print_summary(results_df):
    print("\n======= Final Summary =======")
    for _, row in results_df.iterrows():
        print(f"Fold {row['fold']} - {row['model_name']}: {format_metrics(row)}")

    print()
    for metric in METRICS:
        values = results_df[metric].values
        print(f"Overall {metric}: {np.mean(values):.4f} ± {np.std(values):.4f}")


def plot_roc(roc_data, save_path=None):
    """roc_data: list of dicts with keys fold, model_name, y_true, y_prob, auc."""
    plt.figure(figsize=(8, 6))
    for d in roc_data:
        fpr, tpr, _ = roc_curve(d["y_true"], d["y_prob"])
        plt.plot(fpr, tpr, label=f"Fold {d['fold']} - {d['model_name']} (AUC={d['auc']:.2f})")

    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve per Outer Fold")
    plt.legend(loc="lower right", fontsize=8)
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
