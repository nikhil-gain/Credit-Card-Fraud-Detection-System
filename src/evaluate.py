import joblib
import matplotlib.pyplot as plt
import shap
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, ConfusionMatrixDisplay
)

def evaluate(model, x_test, y_test):
    """Evaluate the trained model on the test set."""
    y_pred = model.predict(x_test)
    y_prob = model.predict_proba(x_test)[:, 1]
    
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Legit', 'Fraud']))
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob):.4f}")
    return y_pred, y_prob

def plot_confusion_matrix(y_test, y_pred):
    """Plot the confusion matrix."""
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Legit', 'Fraud'])
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix")
    plt.show()

def plot_roc_curve(y_test, y_prob):
    """Plot the ROC curve."""
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, label=f'AUC = {auc:.4f}')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess', color='red')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()

def compute_shap(model, x_test):
    """Compute and plot SHAP values for model interpretability."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x_test)
    shap.summary_plot(shap_values, x_test)
    shap.summary_plot(shap_values, x_test, plot_type="bar")
    return explainer, shap_values

def save_shap_data(explainer, x_test):
    """Save SHAP explainer and test data for dashboard use."""
    joblib.dump((explainer, x_test), '../data/shap_data.pkl')
    print("SHAP data saved successfully.")
    
def single_prediction_shap(model, explainer):
    """Get SHAP values for a single prediction."""
    input_data = x_test.iloc[[0]]  # Example input (first test sample)  
    shap_values = explainer.shap_values(input_data)
    shap.force_plot(explainer.expected_value, shap_values, input_data, matplotlib=True)
    plt.show()


if __name__ == "__main__":
    x_train_sm, x_test, y_train_sm, y_test = joblib.load('../data/preprocessed_splits.pkl')
    model = joblib.load('../models/fraud_model.pkl')
    y_pred, y_prob = evaluate(model, x_test, y_test)
    plot_confusion_matrix(y_test, y_pred)
    plot_roc_curve(y_test, y_prob)
    
    explainer, shap_values = compute_shap(model, x_test)
    save_shap_data(explainer, x_test)
    single_prediction_shap(model, explainer)
    print("Evaluation completed successfully.")