import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc

class PAYGRiskModel:
    def __init__(self):
        self.model = LogisticRegression(random_state=42)
        self.scaler = StandardScaler()
        self.features = ["Distance_to_BSS_km", "Income_Volatility", "Daily_Income_KES", "Net_Savings_KES"]
        self.is_trained = False
        
    def train(self, data_path="data/rider_loans.csv"):
        """Trains the model on the rider dataset and computes importances."""
        df = pd.read_csv(data_path)
        
        X = df[self.features]
        y = df["Default_Indicator"]
        
        # Fit scaler and model
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_trained = True
        
        # Compute performance metrics
        y_pred = self.model.predict(X_scaled)
        y_prob = self.model.predict_proba(X_scaled)[:, 1]
        
        self.accuracy = accuracy_score(y, y_pred)
        self.report = classification_report(y, y_pred, output_dict=True)
        
        # ROC curve
        self.fpr, self.tpr, _ = roc_curve(y, y_prob)
        self.roc_auc = auc(self.fpr, self.tpr)
        
        # Calculate unscaled coefficients (for direct probability formula)
        # log(p/(1-p)) = w_scaled * (x - mean) / std + b
        #              = x * (w_scaled / std) + (b - sum(w_scaled * mean / std))
        w_scaled = self.model.coef_[0]
        std = self.scaler.scale_
        mean = self.scaler.mean_
        b = self.model.intercept_[0]
        
        self.unscaled_coef = w_scaled / std
        self.unscaled_intercept = b - np.sum(w_scaled * mean / std)
        
        # Odds ratios for standardized coefficients
        self.odds_ratios = np.exp(w_scaled)
        
        # Standardized feature importances
        # Higher absolute value of standardized coefficient means more influence
        self.importances = pd.DataFrame({
            "Feature": [
                "Distance to Nearest Swap Station (km)", 
                "Rider Income Volatility (%)", 
                "Daily Rider Income (KES)", 
                "Net Fuel Cost Savings (KES)"
            ],
            "Feature_Code": self.features,
            "Coefficient_Standardized": w_scaled,
            "Odds_Ratio": self.odds_ratios,
            "Direction": ["Increase Risk" if w > 0 else "Decrease Risk" for w in w_scaled],
            "Absolute_Importance": np.abs(w_scaled)
        }).sort_values(by="Absolute_Importance", ascending=False)
        
        return self
        
    def predict_probability(self, distance_km, volatility, income, net_savings):
        """Predict default probability for a single rider profile using the unscaled model."""
        if not self.is_trained:
            raise ValueError("Model is not trained yet!")
            
        x = np.array([distance_km, volatility, income, net_savings])
        z = np.dot(x, self.unscaled_coef) + self.unscaled_intercept
        prob = 1.0 / (1.0 + np.exp(-z))
        return prob

    def evaluate_portfolio_risk(self, riders_df, new_distances=None):
        """Evaluates portfolio default rates using the trained model and optional new BSS distances."""
        if not self.is_trained:
            raise ValueError("Model is not trained yet!")
            
        df_copy = riders_df.copy()
        if new_distances is not None:
            df_copy["Distance_to_BSS_km"] = new_distances
            
        # Extract features and scale
        X = df_copy[self.features]
        X_scaled = self.scaler.transform(X)
        
        # Predict default probability for the entire portfolio
        probs = self.model.predict_proba(X_scaled)[:, 1]
        
        # Calculate key metrics
        avg_default_prob = np.mean(probs)
        high_risk_count = np.sum(probs > 0.15) # 15% default threshold
        
        return {
            "expected_default_rate": np.round(avg_default_prob * 100.0, 2),
            "high_risk_riders_count": int(high_risk_count),
            "high_risk_riders_pct": np.round((high_risk_count / len(riders_df)) * 100.0, 1)
        }

def get_trained_model(data_path="data/rider_loans.csv"):
    model = PAYGRiskModel()
    model.train(data_path)
    return model

if __name__ == "__main__":
    # Test training
    import os
    if os.path.exists("data/rider_loans.csv"):
        model = get_trained_model()
        print("Model Trained successfully!")
        print("Accuracy:", model.accuracy)
        print("Feature Importances:")
        print(model.importances)
        
        # Test individual prediction
        test_prob = model.predict_probability(1.5, 0.15, 1100, 250)
        print(f"Low risk rider default prob: {test_prob*100:.2f}%")
        test_prob_high = model.predict_probability(8.0, 0.30, 800, 100)
        print(f"High risk rider default prob: {test_prob_high*100:.2f}%")
    else:
        print("rider_loans.csv not found. Run data_processor.py first.")
