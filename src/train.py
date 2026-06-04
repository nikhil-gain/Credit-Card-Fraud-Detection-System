import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score

def train_xgboost(x_train_sm, y_train_sm):
    """Train XGBoost ith GridSearchCV for hyperparameter tuning."""
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [4, 6],
        'learning_rate': [0.05, 0.1]
    }
    
    grid = GridSearchCV(
        XGBClassifier(random_state=42, eval_metric='logloss'),
        param_grid=param_grid, 
        cv=3, 
        scoring='roc_auc',
        n_jobs=-1, 
        verbose=2
    )

    grid.fit(x_train_sm, y_train_sm)
    print(f"Best Hyperparameters: {grid.best_params_}")
    print(f"Best ROC-AUC: {grid.best_score_:.4f}")
    return grid.best_estimator_

def cross_validate(model, x_train_sm, y_train_sm):
    """Verify model generalizes with cross-validation."""
    cv_scores = cross_val_score(model, x_train_sm, y_train_sm, cv=5, scoring='roc_auc', n_jobs=-1)
    print(f"Mean ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")    
    return cv_scores

def save_model(model):
    """Save the trained model to disk."""
    joblib.dump(model, '../models/fraud_model.pkl')
    print("Model saved to ../models/fraud_model.pkl")


if __name__ == "__main__":
    x_train_sm, x_test, y_train_sm, y_test = joblib.load('../data/preprocessed_splits.pkl')
    model = train_xgboost(x_train_sm, y_train_sm)
    cross_validate(model, x_train_sm, y_train_sm)
    save_model(model)
    print("Model trained and saved successfully.")
