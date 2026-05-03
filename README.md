# AI-Personalized-Meal-Planning-System
AI-Based Personalized Meal Planning and Dietary Recommendation System using Machine Learning and Generative AI
# AI Nutritionist - Personalized Meal Planning System

A Streamlit-based application that predicts daily calorie needs and generates a personalized meal plan using nutrition and health/activity data.

## Features

- Collects user profile data (age, sex, weight, height, activity level)
- Uses a trained ML model to estimate daily calorie requirements
- Calculates macro targets (protein, carbs, fat)
- Builds meal suggestions for breakfast, lunch, dinner, and snacks
- Provides health and lifestyle recommendations (blood pressure, sleep, exercise, diabetes, heart health)

## Project Structure

- `app.py`: Main Streamlit application
- `app-checkpoint.py`: Alternative/older app version with retraining control
- `foods.csv`: Food nutrient database used to build meal combinations
- `health_activity_data.csv`: Health/activity dataset used during model training
- `final_recommendations.csv`: Precomputed recommendations data
- `catboost_training.json`, `learn_error.tsv`, `time_left.tsv`, `events.out.tfevents*`: Training/log artifacts
- `nutrition.ipynb`: Notebook for exploration/experiments

## Requirements

- Python 3.9+
- pip

Install dependencies:

```bash
pip install streamlit pandas numpy scikit-learn joblib matplotlib
```

## How to Run

From this project directory:

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal (usually `http://localhost:8501`).

## Data and Model Files

The app expects these files in the same folder as `app.py`:

- `foods.csv`
- `health_activity_data.csv`
- `trained_model.pkl`
- `scaler.pkl`

If `trained_model.pkl` or `scaler.pkl` is missing, current `app.py` may stop at startup. In that case:

1. Use `app-checkpoint.py` retraining flow, or
2. Train and place `trained_model.pkl` and `scaler.pkl` in the project folder.

## Usage

1. Enter personal, health, and lifestyle details.
2. Click **Generate Personalized Meal Plan**.
3. Review calorie target, macro goals, meal suggestions, and health/lifestyle guidance.

## Notes

- Recommendations are informational and not a substitute for professional medical advice.
- Model quality depends on the quality and coverage of the training data.

## Future Improvements

- Add a `requirements.txt` for one-command setup
- Add input validation and error handling around model feature alignment
- Add automated tests for preprocessing and prediction paths
