"""
PAYG Credit Risk Model Module.

Trains a logistic regression model predicting default probability of riders
based on spatial and financial features. Includes support for portfolio risk evaluation.
"""

import logging
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# --- CUSTOM EXCEPTIONS ---
class ModelNotTrainedError(Exception):
    """Raised when predictions are requested from an untrained model instance."""
    pass

class DataImportError(Exception):
    """Raised when data loading fails due to missing files or corrupted schemas."""
    pass


class PAYGRiskModel:
    """Logistic Regression Credit Risk Model for pay-as-you-go e-boda boda financing."""

    def __init__(self) -> None:
        """Initializes model attributes and feature list."""
        self.model: LogisticRegression = LogisticRegression(random_state=42)
        self.scaler: StandardScaler = StandardScaler()
        self.features: List[str] = [
            "Distance_to_BSS_km", 
            "Income_Volatility", 
            "Daily_Income_KES", 
            "Net_Savings_KES"
        ]
        self.is_trained: bool = False
        self.accuracy: float = 0.0
        self.report: Dict[str, Any] = {}
        
        # Scaling conversion coefficients
        self.unscaled_coef: np.ndarray = np.array([])
        self.unscaled_intercept: float = 0.0
        self.odds_ratios: np.ndarray = np.array([])
        self.importances: pd.DataFrame = pd.DataFrame()

    def train(self, data_path: str = "data/rider_loans.csv") -> "PAYGRiskModel":
        """Fits scaler and logistic regression on historical rider loans.

        Calculates unscaled coefficients for algebraic model portability.

        Args:
            data_path (str): Relative path to the rider loans CSV training file.

        Returns:
            PAYGRiskModel: The trained model instance.

        Raises:
            DataImportError: If data path does not exist or headers are missing.
        """
        try:
            df = pd.read_csv(data_path)
        except Exception as e:
            raise DataImportError(f"Failed to read training data from {data_path}. Error: {e}")
            
        if not all(col in df.columns for col in self.features + ["Default_Indicator"]):
            raise DataImportError("Required columns are missing from the training dataset.")
            
        X = df[self.features]
        y = df["Default_Indicator"]
        
        # Fit scaler and model
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_trained = True
        
        # Performance analytics
        y_pred = self.model.predict(X_scaled)
        y_prob = self.model.predict_proba(X_scaled)[:, 1]
        
        self.accuracy = float(accuracy_score(y, y_pred))
        self.report = classification_report(y, y_pred, output_dict=True)
        
        # ROC metrics
        self.fpr, self.tpr, _ = roc_curve(y, y_prob)
        self.roc_auc = float(auc(self.fpr, self.tpr))
        
        # Calculate algebraic coefficients (unscaling)
        # log(p/(1-p)) = w_scaled * (x - mean) / std + b
        #              = x * (w_scaled / std) + (b - sum(w_scaled * mean / std))
        w_scaled = self.model.coef_[0]
        std = self.scaler.scale_
        mean = self.scaler.mean_
        b = self.model.intercept_[0]
        
        self.unscaled_coef = w_scaled / std
        self.unscaled_intercept = float(b - np.sum(w_scaled * mean / std))
        self.odds_ratios = np.exp(w_scaled)
        
        # Score feature importances
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
        
        logger.info("Successfully trained PAYG Credit Risk Model. Accuracy: %.4f, AUC: %.4f", self.accuracy, self.roc_auc)
        return self
        
    def predict_probability(self, distance_km: float, volatility: float, income: float, net_savings: float) -> float:
        """Predicts the default probability for an individual rider.

        Args:
            distance_km (float): Distance to nearest BSS in kilometers.
            volatility (float): Rider daily income coefficient of variation (0.0 to 1.0).
            income (float): Expected rider daily gross income in KES.
            net_savings (float): Net operating cost savings from using electric vehicle in KES.

        Returns:
            float: Predicted probability of payment default (0.0 to 1.0).

        Raises:
            ModelNotTrainedError: If the model has not been trained.
        """
        if not self.is_trained:
            raise ModelNotTrainedError("Predict probability called on an untrained model instance. Train model first.")
            
        x = np.array([distance_km, volatility, income, net_savings])
        z = np.dot(x, self.unscaled_coef) + self.unscaled_intercept
        prob = 1.0 / (1.0 + np.exp(-z))
        return float(prob)

    def evaluate_portfolio_risk(self, riders_df: pd.DataFrame, new_distances: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Calculates macro default rates and high-risk cohort sizes across the portfolio.

        Args:
            riders_df (pd.DataFrame): DataFrame of rider profiles.
            new_distances (Optional[np.ndarray]): Vector of recalculated distances post network expansion.

        Returns:
            Dict[str, Any]: Expected default rate (%), high-risk counts, and percentages.

        Raises:
            ModelNotTrainedError: If the model has not been trained.
        """
        if not self.is_trained:
            raise ModelNotTrainedError("Evaluate portfolio risk called on an untrained model instance. Train model first.")
            
        df_copy = riders_df.copy()
        if new_distances is not None:
            df_copy["Distance_to_BSS_km"] = new_distances
            
        X = df_copy[self.features]
        X_scaled = self.scaler.transform(X)
        
        probs = self.model.predict_proba(X_scaled)[:, 1]
        
        avg_default_prob = np.mean(probs)
        high_risk_count = np.sum(probs > 0.15) # Default threshold flag at 15%
        
        return {
            "expected_default_rate": float(np.round(avg_default_prob * 100.0, 2)),
            "high_risk_riders_count": int(high_risk_count),
            "high_risk_riders_pct": float(np.round((high_risk_count / len(riders_df)) * 100.0, 1))
        }

def get_trained_model(data_path: str = "data/rider_loans.csv") -> PAYGRiskModel:
    """Helper factory function to instantiate and train the model.

    Args:
        data_path (str): Relative path to the rider loans CSV file.

    Returns:
        PAYGRiskModel: A trained model instance.
    """
    model = PAYGRiskModel()
    model.train(data_path)
    return model

if __name__ == "__main__":
    import os
    if os.path.exists("data/rider_loans.csv"):
        model = get_trained_model()
        logger.info("Accuracy: %.4f", model.accuracy)
        logger.info("Importances:\n%s", model.importances)
        
        # Quick individual checks
        low_risk = model.predict_probability(1.5, 0.15, 1100, 250)
        high_risk = model.predict_probability(8.0, 0.30, 800, 100)
        logger.info("Low-risk profile default prob: %.2f%%", low_risk * 100.0)
        logger.info("High-risk profile default prob: %.2f%%", high_risk * 100.0)
    else:
        logger.warning("Rider loan CSV dataset not found. Execute data_processor.py to compile datasets.")
