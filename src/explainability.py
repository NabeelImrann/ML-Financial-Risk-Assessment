import shap
import matplotlib.pyplot as plt

class ExplainabilityEngine:
    def __init__(self, model, feature_names):
        self.model = model
        self.feature_names = feature_names
        self.explainer = shap.TreeExplainer(model)

    def generate_explanations(self, X):
        """
        Generate SHAP values for model interpretability.
        """
        shap_values = self.explainer.shap_values(X)
        
        # Global Importance Plot
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X, feature_names=self.feature_names, show=False)
        plt.title("Global Feature Importance (SHAP)")
        plt.savefig("shap_summary.png")
        print("Global SHAP summary plot generated.")

    def explain_instance(self, instance):
        """
        Explain a single prediction (Local Explainability).
        """
        shap_value = self.explainer.shap_values(instance)
        # In a real app, this would return a JSON for a dashboard
        return shap_value
