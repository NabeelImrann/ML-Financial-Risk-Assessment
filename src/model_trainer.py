import xgboost as xgb
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import joblib
import os

class ModelTrainer:
    def __init__(self):
        self.model = xgb.XGBClassifier(
            n_estimators=1000,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric='logloss'
        )

    def train(self, X_train, y_train, X_test, y_test):
        print("Training specialist XGBoost model...")
        self.model.fit(
            X_train, y_train,
            early_stopping_rounds=50,
            eval_set=[(X_test, y_test)],
            verbose=False
        )
        print("Training complete.")

    def evaluate(self, X_test, y_test):
        preds = self.model.predict(X_test)
        probs = self.model.predict_proba(X_test)[:, 1]
        
        print("\n--- Model Performance ---")
        print(classification_report(y_test, preds))
        print(f"ROC-AUC Score: {roc_auc_score(y_test, probs):.4f}")
        
    def save_model(self, path='models/risk_model.joblib'):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)
        print(f"Model saved to {path}")

if __name__ == "__main__":
    # Example execution flow
    print("XGBoost Specialist Trainer Module Loaded.")
