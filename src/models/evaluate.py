import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay,
                              RocCurveDisplay, PrecisionRecallDisplay)

def evaluate(model_path="../../mlflow/best_model.pkl",
             data_path="../../data/processed/data_splits.pkl"):

    model = joblib.load(model_path)
    _, X_test, _, y_test = joblib.load(data_path)

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Confusion Matrix
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred,
        display_labels=['Légitime', 'Fraude'],
        ax=axes[0], colorbar=False, cmap='Blues'
    )
    axes[0].set_title('Confusion Matrix')

    # 2. ROC Curve
    RocCurveDisplay.from_predictions(
        y_test, y_proba, ax=axes[1]
    )
    axes[1].set_title('ROC Curve')

    # 3. Precision-Recall Curve
    PrecisionRecallDisplay.from_predictions(
        y_test, y_proba, ax=axes[2]
    )
    axes[2].set_title('Precision-Recall Curve')

    plt.tight_layout()
    plt.savefig('../../data/processed/evaluation.png', dpi=150)
    plt.show()
    print("✅ Graphiques sauvegardés !")

if __name__ == "__main__":
    evaluate()