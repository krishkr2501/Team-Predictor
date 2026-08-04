# 🏀 Team Performance Prediction Model

A machine learning model that predicts a sports team's **performance tier**
(Elite / Contender / Average / Struggling) from season statistics, using a
Random Forest classifier.

## 📊 What It Does

- **Analyzes** team performance metrics (wins, losses, scoring, defense, etc.)
- **Predicts** which performance tier a team belongs to
- **Identifies** which stats matter most for team success
- **Forecasts** the tier for a brand-new/hypothetical team

## 🛠 Tech Stack

- Python 3.8+
- Pandas & NumPy — data manipulation
- Scikit-learn — machine learning (Random Forest / Gradient Boosting)
- Matplotlib & Seaborn — visualization
- Google Colab / Jupyter — interactive execution

## 📁 Project Structure

```
team-performance-predictor/
├── team_performance_predictor.py   # All code: data, model, training, eval
├── teams_data.csv                  # Training dataset (60 synthetic teams)
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── team_predictor.pkl              # Trained model (generated after running)
├── scaler.pkl                      # Fitted feature scaler (generated)
├── feature_importance.png          # Chart (generated)
├── confusion_matrix.png            # Chart (generated)
└── correlation_heatmap.png         # Chart (generated)
```

## 🚀 Quick Start

### Option 1 — Run in Google Colab (easiest)

1. Go to [colab.research.google.com](https://colab.research.google.com) and
   create a new notebook.
2. Upload `team_performance_predictor.py` and `teams_data.csv`
   (File icon on the left sidebar → Upload), or paste the script's contents
   into a cell.
3. Run all cells (`Runtime` → `Run all`). The script will train the model,
   print metrics, show charts, and save `team_predictor.pkl` / `scaler.pkl`.
4. Optionally uncomment the final cell to download the trained files to
   your computer.

### Option 2 — Run locally

```bash
# Clone the repository
git clone https://github.com/<your-username>/team-performance-predictor.git
cd team-performance-predictor

# Install dependencies
pip install -r requirements.txt

# Run the full pipeline
python team_performance_predictor.py
```

### Make Predictions

```python
from team_performance_predictor import TeamPerformancePredictor

predictor = TeamPerformancePredictor()
predictor.load_data("teams_data.csv")
predictor.prepare_data(target_column="performance_tier")
predictor.train_model(model_type="random_forest")

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
print(result)
```

## 📈 Model Performance

- **Accuracy:** ~90% on held-out test data
- **Top features:** Wins, Losses, Rebounds, Three-point %
- **Algorithm:** Random Forest Classifier (Gradient Boosting also supported)

## 📚 Dataset

`teams_data.csv` contains **60 synthetic teams** generated with a fixed
random seed (so results are reproducible), with the following columns:

| Feature | Description |
|---|---|
| team_name | Team identifier |
| wins | Total wins in season |
| losses | Total losses in season |
| points_for | Average points scored |
| points_against | Average points allowed |
| rebounds | Rebounds per game |
| assists | Assists per game |
| field_goal_percent | Field goal accuracy % |
| three_point_percent | Three-point accuracy % |
| turnovers | Turnovers per game |
| league_position | Final standing (1 = best) — used only to derive the tier, **not** used as a model feature |
| performance_tier | Target label: Elite / Contender / Average / Struggling |

Want to use real data instead? Swap in stats from
[Basketball-Reference](https://www.basketball-reference.com/) or a
[Kaggle sports dataset](https://www.kaggle.com/datasets), as long as the
column names match (or update `feature_names` accordingly).

## 🎯 Key Insights

- Win/loss record is (unsurprisingly) the strongest predictor of tier
- Rebounding and three-point shooting help distinguish teams with similar records
- Turnovers have a measurable negative effect on tier

## 🔄 Improvements & Next Steps

- [ ] Swap in real historical season data
- [ ] Add a win/loss (next-game) prediction model
- [ ] Try clustering (unsupervised) instead of predefined tiers
- [ ] Build a small Streamlit/Flask app for live predictions
- [ ] Track performance across multiple seasons (time series)

## 🤝 Contributing

Have ideas? Fork the repo and submit a pull request!

## 📧 Contact

Questions? Open an issue on the repository.

---
*"Great teams are built on data-driven decisions."*
