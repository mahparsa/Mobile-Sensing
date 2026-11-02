"""Summary of outer-fold results (code from the notebook)."""
import numpy as np


def print_summary(outer_results):
    # Summary
    print("\n======= Final Summary =======")
    for res in outer_results:
        print(f"Fold {res['fold']} - {res['model_name']}: "
              f"Acc={res['accuracy']:.4f}, F1={res['f1_score']:.4f}, AUC={res['auc']:.4f}, "
              f"BalAcc={res['balanced_accuracy']:.4f}, Sens={res['sensitivity']:.4f}, Spec={res['specificity']:.4f}")

    # Mean ± Std Summary
    def summarize_metric(metric):
        values = [res[metric] for res in outer_results]
        return f"{np.mean(values):.4f} ± {np.std(values):.4f}"

    print(f"\nOverall Accuracy: {summarize_metric('accuracy')}")
    print(f"Overall F1 Score: {summarize_metric('f1_score')}")
    print(f"Overall AUC: {summarize_metric('auc')}")
    print(f"Overall Balanced Accuracy: {summarize_metric('balanced_accuracy')}")
    print(f"Overall Sensitivity: {summarize_metric('sensitivity')}")
    print(f"Overall Specificity: {summarize_metric('specificity')}")
