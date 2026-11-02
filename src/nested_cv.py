"""BLOCK 3: Nested CV (code from the notebook, wrapped in a function)."""
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             balanced_accuracy_score, recall_score,
                             confusion_matrix, roc_curve)
import matplotlib.pyplot as plt

from src.preprocessing import preprocess_fold


def run_nested_cv(X, y, classifiers, param_grids):
    # Outer Cross Validation
    best_cv_value = 10
    outer_cv = StratifiedKFold(n_splits=best_cv_value, shuffle=True, random_state=42)

    # Initialize lists and dictionaries
    outer_results = []
    best_models_per_fold = {}
    inner_best_models = {}

    for fold_idx, (train_index, test_index) in enumerate(outer_cv.split(X, y), 1):
        print(f"\n========= Fold {fold_idx} =========")

        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Preprocessing inside the fold
        X_train, y_train, X_test, fold_cv_value = preprocess_fold(X_train, X_test, y_train)

        # Inner CV
        inner_cv = StratifiedKFold(n_splits=fold_cv_value, shuffle=True, random_state=42)

        best_classifier = None
        best_params = None
        best_score = 0.0
        best_model_name = None

        for clf, params in zip(classifiers, param_grids):
            model_name = clf.__class__.__name__
            print(f"Evaluating {model_name}...")

            scoring_metrics = {'accuracy': 'accuracy', 'f1_weighted': 'f1_weighted'}
            grid_search = GridSearchCV(estimator=clf, param_grid=params, cv=inner_cv,
                                       scoring=scoring_metrics, refit='accuracy')
            grid_search.fit(X_train, y_train)

            best_model = grid_search.best_estimator_
            y_pred_inner = best_model.predict(X_test)
            acc = accuracy_score(y_test, y_pred_inner)
            f1 = f1_score(y_test, y_pred_inner, average='weighted')

            inner_best_models[(fold_idx, model_name)] = {
                'model': best_model,
                'params': grid_search.best_params_,
                'f1_score': f1,
                'accuracy': acc
            }

            if acc > best_score:
                best_score = acc
                best_classifier = best_model
                best_model_name = model_name
                best_params = grid_search.best_params_

        if best_classifier is None:
            best_classifier = classifiers[0]
            best_model_name = best_classifier.__class__.__name__
            print("No best model found, using default.")

        print(f"Best Hyperparameters from Inner CV for Fold {fold_idx}: "
              f"{best_model_name} with {best_params}")

        # Outer test evaluation
        y_pred_outer = best_classifier.predict(X_test)
        y_prob_outer = best_classifier.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred_outer)
        f1 = f1_score(y_test, y_pred_outer, average='weighted')
        auc_score = roc_auc_score(y_test, y_prob_outer)
        bal_acc = balanced_accuracy_score(y_test, y_pred_outer)
        sens = recall_score(y_test, y_pred_outer, pos_label=1)

        cm = confusion_matrix(y_test, y_pred_outer)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        spec = tn / (tn + fp) if (tn + fp) else 0.0

        print(f"Accuracy: {acc:.4f}, F1: {f1:.4f}, AUC: {auc_score:.4f}, "
              f"Balanced Acc: {bal_acc:.4f}, Sensitivity: {sens:.4f}, Specificity: {spec:.4f}")

        outer_results.append({
            'fold': fold_idx,
            'model_name': best_model_name,
            'accuracy': acc,
            'f1_score': f1,
            'auc': auc_score,
            'balanced_accuracy': bal_acc,
            'sensitivity': sens,
            'specificity': spec
        })

        best_models_per_fold[fold_idx] = {
            'model': best_classifier,
            'model_name': best_model_name,
            'params': best_params,
            'metrics': outer_results[-1]
        }

        # ROC Curve for the fold
        fpr, tpr, _ = roc_curve(y_test, y_prob_outer)
        plt.plot(fpr, tpr, label=f'Fold {fold_idx} - {best_model_name} (AUC={auc_score:.2f})')

    # Final ROC Plot
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Best Model in Each Fold')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return outer_results, best_models_per_fold, inner_best_models
