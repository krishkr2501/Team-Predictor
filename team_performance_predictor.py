"""
Team Performance Prediction Model
==================================
A machine learning pipeline that predicts a sports team's league position
from season statistics (wins, losses, scoring, defense, etc.)

HOW TO USE IN GOOGLE COLAB
---------------------------
1. Upload this file OR just copy/paste its contents into a new Colab notebook.
2. Upload `teams_data.csv` (provided alongside this script) using the file
   upload cell below, OR mount Google Drive and point DATA_PATH to it.
3. Run all cells top to bottom (Runtime > Run all).
4. The trained model + scaler will be saved as .pkl files and offered for
   download at the end.

This script is written so it also runs unmodified as a normal .py file on
your own machine, as long as `teams_data.csv` sits in the same folder.
"""

# =============================================================================
# CELL 1: Install & Import dependencies
# =============================================================================
# In Colab, uncomment the line below if any package is missing (rare, most
# of these ship pre-installed on Colab):
# !pip install -q pandas numpy scikit-learn matplotlib seaborn joblib

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

print("✅ Libraries imported successfully")

# =============================================================================
# CELL 2: Get the dataset
# =============================================================================
# OPTION A (default): Upload teams_data.csv manually.
# If running in Google Colab, uncomment the block below to get an upload
# widget the first time you run the notebook:
#
# from google.colab import files
# uploaded = files.upload()   # choose teams_data.csv from your computer
#
# OPTION B: Generate the same synthetic dataset from scratch (useful if you
# don't have the CSV handy — this reproduces teams_data.csv exactly, since
# it uses the same random seed).

DATA_PATH = "teams_data.csv"


def generate_team_data(n_teams=60, seed=42, save_path=DATA_PATH):
    """Generate synthetic team performance data (deterministic via seed)."""
    np.random.seed(seed)
    data = {
        "team_name": [f"Team_{i}" for i in range(1, n_teams + 1)],
        "wins": np.random.randint(15, 68, n_teams),
        "points_for": np.round(np.random.uniform(100, 128, n_teams), 1),
        "points_against": np.round(np.random.uniform(100, 125, n_teams), 1),
        "rebounds": np.round(np.random.uniform(40, 53, n_teams), 1),
        "assists": np.round(np.random.uniform(22, 33, n_teams), 1),
        "field_goal_percent": np.round(np.random.uniform(42, 53, n_teams), 1),
        "three_point_percent": np.round(np.random.uniform(30, 43, n_teams), 1),
        "turnovers": np.round(np.random.uniform(11, 19, n_teams), 1),
    }
    df = pd.DataFrame(data)
    df["losses"] = 82 - df["wins"]

    # Assign a unique league position based on wins (1 = best record)
    df["league_position"] = df["wins"].rank(ascending=False, method="first").astype(int)
    df = df.sort_values("league_position").reset_index(drop=True)

    # Bucket standings into four tiers -- a more learnable classification
    # target than predicting the exact 1-of-60 finishing position.
    def tier(pos, n=n_teams):
        q = n / 4
        if pos <= q:
            return "Elite"
        elif pos <= 2 * q:
            return "Contender"
        elif pos <= 3 * q:
            return "Average"
        else:
            return "Struggling"

    df["performance_tier"] = df["league_position"].apply(tier)

    cols = [
        "team_name", "wins", "losses", "points_for", "points_against",
        "rebounds", "assists", "field_goal_percent", "three_point_percent",
        "turnovers", "league_position", "performance_tier",
    ]
    df = df[cols]
    df.to_csv(save_path, index=False)
    print(f"✅ Generated {n_teams} teams -> saved to {save_path}")
    return df


if not os.path.exists(DATA_PATH):
    print(f"⚠️  {DATA_PATH} not found locally — generating synthetic data instead.")
    generate_team_data(n_teams=60, seed=42, save_path=DATA_PATH)
else:
    print(f"✅ Found {DATA_PATH} — using it directly.")

# =============================================================================
# CELL 3: Model class
# =============================================================================
class TeamPerformancePredictor:
    """Predicts a team's league position from its season statistics."""

    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def load_data(self, csv_path):
        """Load team data from CSV."""
        self.df = pd.read_csv(csv_path)
        print(f"✅ Loaded {len(self.df)} teams")
        print(f"Dataset shape: {self.df.shape}")
        print(f"Columns: {self.df.columns.tolist()}")
        return self.df

    def prepare_data(self, target_column="performance_tier", test_size=0.2):
        """Split into train/test sets and scale features."""
        # Drop identifier columns and any column that would leak the target
        # (league_position is used to derive performance_tier, so it must
        # not be used as a predictive feature).
        drop_cols = {"team_name", target_column, "league_position"}
        drop_cols = [c for c in drop_cols if c in self.df.columns]
        X = self.df.drop(columns=drop_cols)
        y = self.df[target_column]
        self.feature_names = X.columns.tolist()

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)

        print("✅ Data prepared")
        print(f"Training set: {self.X_train.shape}")
        print(f"Test set: {self.X_test.shape}")

    def train_model(self, model_type="random_forest"):
        """Train the prediction model (random_forest or gradient_boosting)."""
        if model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
            )
        elif model_type == "gradient_boosting":
            self.model = GradientBoostingClassifier(
                n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42
            )
        else:
            raise ValueError("model_type must be 'random_forest' or 'gradient_boosting'")

        self.model.fit(self.X_train, self.y_train)
        print(f"✅ Model trained ({model_type})")

    def evaluate_model(self):
        """Evaluate model performance on the held-out test set."""
        y_pred = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        print("\n📊 Model Performance")
        print(f"Accuracy: {accuracy:.4f}")
        print("\nClassification Report:\n")
        print(classification_report(self.y_test, y_pred, zero_division=0))
        return y_pred

    def get_feature_importance(self, top_n=10):
        """Return a DataFrame of the most important features."""
        importances = self.model.feature_importances_
        feature_importance_df = pd.DataFrame(
            {"feature": self.feature_names, "importance": importances}
        ).sort_values("importance", ascending=False)
        print(f"\n🎯 Top {top_n} Important Features:")
        print(feature_importance_df.head(top_n))
        return feature_importance_df

    def visualize_importance(self, top_n=10):
        """Bar chart of feature importance."""
        feature_importance_df = self.get_feature_importance(top_n)
        plt.figure(figsize=(10, 6))
        sns.barplot(
            data=feature_importance_df.head(top_n),
            x="importance",
            y="feature",
            palette="viridis",
        )
        plt.title("Top Features Predicting Team Performance", fontsize=14, fontweight="bold")
        plt.xlabel("Importance")
        plt.tight_layout()
        plt.savefig("feature_importance.png", dpi=300, bbox_inches="tight")
        plt.show()

    def plot_confusion_matrix(self, y_pred):
        """Heatmap of predicted vs. actual league positions."""
        cm = confusion_matrix(self.y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title("Confusion Matrix")
        plt.ylabel("Actual")
        plt.xlabel("Predicted")
        plt.tight_layout()
        plt.savefig("confusion_matrix.png", dpi=300, bbox_inches="tight")
        plt.show()

    def predict_team_outcome(self, team_stats):
        """
        Predict the league position for a new team.
        team_stats: dict of the same feature columns used in training.
        """
        team_df = pd.DataFrame([team_stats])
        team_df = team_df[self.feature_names]
        team_scaled = self.scaler.transform(team_df)
        prediction = self.model.predict(team_scaled)[0]
        confidence = self.model.predict_proba(team_scaled).max()
        return {"predicted_position": prediction, "confidence": confidence}

    def save_model(self, filepath="team_predictor.pkl"):
        """Persist the trained model + scaler to disk."""
        joblib.dump(self.model, filepath)
        joblib.dump(self.scaler, "scaler.pkl")
        print(f"✅ Model saved to {filepath}")
        print("✅ Scaler saved to scaler.pkl")

    def load_model(self, filepath="team_predictor.pkl", scaler_path="scaler.pkl"):
        """Load a previously trained model + scaler."""
        self.model = joblib.load(filepath)
        self.scaler = joblib.load(scaler_path)
        print(f"✅ Model loaded from {filepath}")


# =============================================================================
# CELL 4: Exploratory Data Analysis (optional but useful in Colab)
# =============================================================================
_preview_df = pd.read_csv(DATA_PATH)
print(_preview_df.describe())

plt.figure(figsize=(12, 8))
sns.heatmap(_preview_df.corr(numeric_only=True), annot=True, cmap="coolwarm", center=0)
plt.title("Feature Correlations")
plt.tight_layout()
plt.savefig("correlation_heatmap.png", dpi=300, bbox_inches="tight")
plt.show()

# =============================================================================
# CELL 5: Train, evaluate, and save the model
# =============================================================================
predictor = TeamPerformancePredictor()

predictor.load_data(DATA_PATH)
predictor.prepare_data(target_column="performance_tier")
predictor.train_model(model_type="random_forest")

y_pred = predictor.evaluate_model()
predictor.visualize_importance(top_n=8)
predictor.plot_confusion_matrix(y_pred)
predictor.save_model(filepath="team_predictor.pkl")

# =============================================================================
# CELL 6: Predict on a brand-new team
# =============================================================================
new_team = {
    "wins": 50,
    "losses": 32,
    "points_for": 117.5,
    "points_against": 109.8,
    "rebounds": 46.5,
    "assists": 28.2,
    "field_goal_percent": 50.1,
    "three_point_percent": 37.5,
    "turnovers": 13.5,
}

result = predictor.predict_team_outcome(new_team)
print(
    f"\n🏀 Prediction: Position {result['predicted_position']} "
    f"(Confidence: {result['confidence']:.2%})"
)

# =============================================================================
# CELL 7: (Colab only) Download the trained artifacts to your computer
# =============================================================================
# Uncomment this block when running in Google Colab to download the model,
# scaler, and generated charts:
#
# from google.colab import files
# for f in ["team_predictor.pkl", "scaler.pkl", "feature_importance.png",
#           "confusion_matrix.png", "correlation_heatmap.png"]:
#     if os.path.exists(f):
#         files.download(f)
