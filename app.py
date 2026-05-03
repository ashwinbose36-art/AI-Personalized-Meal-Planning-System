# Streamlit app for AI Nutritionist with balanced meal planning
import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt


# Configuration
DATA_DIR = '.'
FOODS_CSV = os.path.join(DATA_DIR, 'foods.csv')
HEALTH_CSV = os.path.join(DATA_DIR, 'health_activity_data.csv')
MODEL_PATH = os.path.join(DATA_DIR, 'trained_model.pkl')
SCALER_PATH = os.path.join(DATA_DIR, 'scaler.pkl')
RECS_PATH = os.path.join(DATA_DIR, 'final_recommendations.csv')

# Page setup
st.set_page_config(page_title='AI Nutritionist', layout='wide')
st.title('AI Nutritionist — Personalized Meal Planning')

# Initialize session state
if 'model' not in st.session_state:
    st.session_state['model'] = None
if 'scaler' not in st.session_state:
    st.session_state['scaler'] = None
 # Sidebar: training controls
st.sidebar.header('Model & Data')
# if not os.path.exists(FOODS_CSV) or not os.path.exists(HEALTH_ACTIVITY_DATA.CSV):
#     st.sidebar.error('Missing data files. Place foods.csv and health_activity_data.csv in the app folder.')
HEALTH_ACTIVITY_DATA = os.path.join(DATA_DIR, 'health_activity_data.csv')

if not os.path.exists(FOODS_CSV) or not os.path.exists(HEALTH_ACTIVITY_DATA):
    st.sidebar.error('❌ Missing data files. Place foods.csv and health_activity_data.csv in the app folder.')
else:
    st.sidebar.success('✅ Data files found.')

# Load pre-trained model and scaler instead of retraining
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    st.session_state['model'] = model
    st.session_state['scaler'] = scaler
    st.sidebar.success('✅ Loaded saved model and scaler.')
except Exception as e:
    st.sidebar.error(f'⚠ Error loading model/scaler: {str(e)}')
    st.stop()

  
  
def compute_bmr(weight_kg, height_m, age, sex):
    try:
        weight_kg = float(weight_kg)
        height_m = float(height_m)
        age = float(age)
        h_cm = height_m * 100
        sex = str(sex).lower()
        
        if weight_kg <= 0 or height_m <= 0 or age <= 0:
            return 2000  # Default value if invalid inputs
            
        if sex in ['female','f','woman']:
            return 10*weight_kg + 6.25*h_cm - 5*age - 161
        if sex in ['male','m','man']:
            return 10*weight_kg + 6.25*h_cm - 5*age + 5
        return 10*weight_kg + 6.25*h_cm - 5*age  # Default formula for other
    except Exception as e:
        st.error(f"Error calculating BMR: {str(e)}")
        return 2000  # Default value if calculation fails
    
 
@st.cache_data
def train_and_save_model(force_retrain=False):
    foods, health = load_data()
    model_exists = os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH)
    if model_exists and not force_retrain:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        return model, scaler, None

    df, X, y, feature_cols = preprocess_health(health)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Linear baseline
    lr = LinearRegression()
    lr.fit(X_train, y_train)

    # Random Forest with a modest grid (balance time vs performance)
    rf = RandomForestRegressor(random_state=42)
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [8, 12, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    grid = GridSearchCV(rf, param_grid, cv=3, scoring='r2', n_jobs=-1, verbose=0)
    grid.fit(X_train, y_train)
    best_rf = grid.best_estimator_

    # Evaluate
    y_pred_lr = lr.predict(X_test)
    y_pred_rf = best_rf.predict(X_test)
    lr_mae = mean_absolute_error(y_test, y_pred_lr)
    rf_mae = mean_absolute_error(y_test, y_pred_rf)
    lr_r2 = r2_score(y_test, y_pred_lr)
    rf_r2 = r2_score(y_test, y_pred_rf)

    best_model = best_rf if rf_r2 >= lr_r2 else lr
    # ✅ Save model, scaler, and feature list
    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(list(X.columns), "feature_columns.pkl")  # Save feature names

    stats = {
        'lr_mae': lr_mae, 
        'rf_mae': rf_mae, 
        'lr_r2': lr_r2, 
        'rf_r2': rf_r2, 
        'best': 'RandomForest' if rf_r2 >= lr_r2 else 'LinearRegression'
    }
    return best_model, scaler, stats


# Main UI
st.header('Enter Your Information')

# Basic Information
col1, col2 = st.columns(2)
with col1:
    age = st.number_input('Age (years)', min_value=10, max_value=100, value=30)
    sex = st.selectbox('Sex', ['female', 'male', 'other'])
    weight = st.number_input('Weight (kg)', min_value=30.0, max_value=250.0, value=70.0)
with col2:
    height = st.number_input('Height (m)', min_value=1.0, max_value=2.5, value=1.75, format="%.2f")
    activity = st.selectbox('Activity level', ['sedentary','light','moderate','active','very active'])
    activity_map = {'sedentary':1.2,'light':1.375,'moderate':1.55,'active':1.725,'very active':1.9}
    activity_factor = activity_map.get(activity, 1.2)

# Health Parameters
st.subheader('Health Parameters')
col3, col4 = st.columns(2)
with col3:
    systolic = st.number_input('Systolic Blood Pressure (mmHg)', min_value=70, max_value=200, value=120)
    diastolic = st.number_input('Diastolic Blood Pressure (mmHg)', min_value=40, max_value=130, value=80)
    exercise_hrs = st.number_input('Exercise Hours per Week', min_value=0, max_value=40, value=3)
with col4:
    sleep_hrs = st.number_input('Hours of Sleep per Day', min_value=4, max_value=12, value=7)
    heart_disease = st.selectbox('Heart Disease History', ['No', 'Yes'])
    diabetes = st.selectbox('Diabetic Status', ['No', 'Yes'])
# Lifestyle Factors
st.subheader('Lifestyle Factors')
col5, col6 = st.columns(2)
with col5:
    smoking = st.selectbox('Smoking Status', ['Non-smoker', 'Occasional smoker', 'Regular smoker'])
with col6:
    alcohol = st.selectbox('Alcohol Consumption', ['None', 'Light (1-2 drinks/week)', 'Moderate (3-7 drinks/week)', 'Heavy (>7 drinks/week)'])

# Generate recommendations when button is clicked
if st.button('Generate Personalized Meal Plan'):
    # Load model and scaler
    if st.session_state['model'] is None:
        if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
            st.session_state['model'] = joblib.load(MODEL_PATH)
            st.session_state['scaler'] = joblib.load(SCALER_PATH)
        else:
            st.error('Model files not found. Please ensure the model is trained.')
            st.stop()
    
# Calculate BMR and predicted calories
    bmr = compute_bmr(weight, height, age, sex)
    tdee = bmr * activity_factor
    
    # Prepare input for model
    X_new = pd.DataFrame([{
        'age': age,
        'sex_cat': 1 if sex == 'male' else (0 if sex == 'female' else 2),
        'weight_kg': weight,
        'height_m': height,
        'BMI': weight / (height**2),
        'activity_factor': activity_factor,
        'BMR': bmr,
        'TDEE': tdee
    }])
    
    # Make prediction
    # ✅ Align input features with training feature order
if os.path.exists("feature_columns.pkl"):
    feature_cols = joblib.load("feature_columns.pkl")
    X_new = X_new.reindex(columns=feature_cols, fill_value=0)
else:
    st.warning("⚠ feature_columns.pkl not found — may cause mismatch with scaler.")

# -------------------------
# Create feature vector from user input
# -------------------------
X_new = pd.DataFrame([{
    'Age': age,
    'Weight': weight,
    'Height': height,
    'Gender': 1 if gender == 'Male' else 0,
    'Exercise Hours': exercise_hrs,
    'Sleep Hours': sleep_hrs,
    'Diabetes': 1 if diabetes == 'Yes' else 0,
    'Heart Disease': 1 if heart_disease == 'Yes' else 0
    # 🔹 Add any other fields that were used during model training
}])

X_scaled = st.session_state['scaler'].transform(X_new)

pred_cal = st.session_state['model'].predict(X_scaled)[0]

# Display daily targets
st.header('Your Daily Targets')
col_targets1, col_targets2 = st.columns(2)

with col_targets1:
    st.metric('Daily Calorie Need', f"{pred_cal:.0f} kcal")
    # Calculate macro targets
    protein_target = (pred_cal * 0.25) / 4  # 25% of calories from protein
    carbs_target = (pred_cal * 0.50) / 4    # 50% of calories from carbs
    fat_target = (pred_cal * 0.25) / 9      # 25% of calories from fat
    
    st.write("Macro Nutrient Targets:")
    st.write(f"- Protein: {protein_target:.0f}g")
    st.write(f"- Carbohydrates: {carbs_target:.0f}g")
    st.write(f"- Fat: {fat_target:.0f}g")
    
    # Load and process food data
    foods_df = pd.read_csv(FOODS_CSV)
    
    # Generate meal plan
    st.header('Your Personalized Meal Plan ')
    
    # Define meal distribution
    meals = {
        'Breakfast': {'calories': pred_cal * 0.25, 'protein': protein_target * 0.25},
        'Lunch': {'calories': pred_cal * 0.35, 'protein': protein_target * 0.35},
        'Dinner': {'calories': pred_cal * 0.30, 'protein': protein_target * 0.30},
        'Snacks': {'calories': pred_cal * 0.10, 'protein': protein_target * 0.10}
    }
    
    # Function to find suitable food combinations
    def find_meal_combination(available_foods, target_calories, target_protein, meal_type):
        suitable_foods = available_foods.copy()
        
        # Apply meal-specific filters
        if meal_type == "Breakfast":
            suitable_foods = suitable_foods[
                (suitable_foods['Energy kcal'] < target_calories * 0.6) &
                (suitable_foods['Fat(g)'] < 15)
            ]
        elif meal_type in ["Lunch", "Dinner"]:
            suitable_foods = suitable_foods[
                (suitable_foods['Energy kcal'] < target_calories * 0.5) &
                (suitable_foods['Protein(g)'] > 5)
            ]
        else:  # Snacks
            suitable_foods = suitable_foods[
                (suitable_foods['Energy kcal'] < target_calories) &
                (suitable_foods['Fat(g)'] < 10)
            ]
        
        best_combo = []
        current_cals = 0
        current_protein = 0
        
        # Try to find 2-3 items that meet the targets
        while len(best_combo) < 3 and len(suitable_foods) > 0:
            remaining_cals = target_calories - current_cals
            remaining_protein = target_protein - current_protein
            
            # Find next best item
            suitable_foods['score'] = abs(suitable_foods['Energy kcal'] - remaining_cals / 2)
            next_item = suitable_foods.nsmallest(1, 'score').iloc[0]
            
            if current_cals + next_item['Energy kcal'] > target_calories * 1.1:
                break
                
            best_combo.append(next_item)
            current_cals += next_item['Energy kcal']
            current_protein += next_item['Protein(g)']
            
            # Remove selected item
            suitable_foods = suitable_foods[suitable_foods.index != next_item.name]
        
        return best_combo
    
    # Generate and display meal plan
    daily_totals = {'calories': 0, 'protein': 0, 'carbs': 0, 'fat': 0}
    
    for meal_name, targets in meals.items():
        st.subheader(f"{meal_name} ({targets['calories']:.0f} kcal target)")
        
        meal_combo = find_meal_combination(
            foods_df, 
            targets['calories'],
            targets['protein'],
            meal_name
        )
        
        if meal_combo:
            col1, col2 = st.columns([2, 1])
            
            meal_totals = {
                'calories': sum(item['Energy kcal'] for item in meal_combo),
                'protein': sum(item['Protein(g)'] for item in meal_combo),
                'carbs': sum(item['Carbs'] for item in meal_combo),
                'fat': sum(item['Fat(g)'] for item in meal_combo)
            }
            
            with col1:
                for item in meal_combo:
                    # Calculate portion to meet targets
                    portion_factor = targets['calories'] / meal_totals['calories']
                    portion = 100 * portion_factor  # base portion is 100g
                    
                    with st.expander(f"🍽 {item['Food Items']}"):
                        st.write(f"Portion: {portion:.0f}g")
                        st.write("Nutritional Facts (for this portion):")
                        st.write(f"- Calories: {item['Energy kcal'] * portion_factor:.0f} kcal")
                        st.write(f"- Protein: {item['Protein(g)'] * portion_factor:.1f}g")
                        st.write(f"- Carbs: {item['Carbs'] * portion_factor:.1f}g")
                        st.write(f"- Fat: {item['Fat(g)'] * portion_factor:.1f}g")
            
            with col2:
                st.write("Meal Totals:")
                st.write(f"Calories: {meal_totals['calories']:.0f} / {targets['calories']:.0f}")
                st.progress(min(meal_totals['calories'] / targets['calories'], 1.0))
                st.write(f"Protein: {meal_totals['protein']:.1f}g")
                st.write(f"Carbs: {meal_totals['carbs']:.1f}g")
                st.write(f"Fat: {meal_totals['fat']:.1f}g")
            
            # Update daily totals
            for key in daily_totals:
                daily_totals[key] += meal_totals[key]
    
    # Display daily summary
    st.header('Daily Nutrition Summary')
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Total Calories", f"{daily_totals['calories']:.0f} / {pred_cal:.0f} kcal")
        st.metric("Total Protein", f"{daily_totals['protein']:.1f} / {protein_target:.0f}g")
    with col2:
        st.metric("Total Carbs", f"{daily_totals['carbs']:.1f} / {carbs_target:.0f}g")
        st.metric("Total Fat", f"{daily_totals['fat']:.1f} / {fat_target:.0f}g")
    
    # Separate Health and Lifestyle Recommendations
    st.header('Health Recommendations ')
    
    # Health recommendations - always show all categories
    st.subheader('Blood Pressure Status')
    if systolic >= 140 or diastolic >= 90:
        st.warning("🩺 Your blood pressure is high. Focus on low-sodium foods and consider consulting a healthcare provider.")
    elif systolic >= 120 or diastolic >= 80:
        st.warning("⚠ Your blood pressure is elevated. Monitor your sodium intake.")
    else:
        st.success("✅ Your blood pressure is in the normal range. Maintain a healthy diet to keep it that way.")
    
    st.subheader('Diabetes Management')
    if diabetes == 'Yes':
        st.warning("🍎 Important guidelines for diabetes management:")
        st.write("- Monitor your carbohydrate intake carefully")
        st.write("- Maintain regular meal times")
        st.write("- Choose foods with low glycemic index")
        st.write("- Include fiber-rich foods in your meals")
    else:
        st.success("✅ No diabetes indicated. Maintain a balanced diet to prevent diabetes.")
    
    st.subheader('Cardiovascular Health')
    if heart_disease == 'Yes':
        st.warning("❤ Heart health guidelines:")
        st.write("- Focus on heart-healthy foods low in saturated fat")
        st.write("- Limit cholesterol intake")
        st.write("- Include omega-3 rich foods")
        st.write("- Monitor sodium intake")
    else:
        st.success("✅ No heart disease indicated. Maintain heart-healthy habits.")

    st.header('Lifestyle Recommendations ')
    
    st.subheader('Physical Activity')
    if exercise_hrs < 2.5:
        st.warning("🏃‍♂ Exercise Recommendation:")
        st.write(f"- Current: {exercise_hrs:.1f} hours/week")
        st.write("- Target: At least 2.5 hours/week")
        st.write("- Consider adding moderate activities like brisk walking, swimming, or cycling")
    else:
        st.success("✅ Great job maintaining an active lifestyle! Keep up your exercise routine.")

    st.subheader('Sleep Habits')
    if sleep_hrs < 7:
        st.warning("😴 Sleep Recommendation:")
        st.write(f"- Current: {sleep_hrs:.1f} hours/day")
        st.write("- Target: 7-9 hours/day")
        st.write("- Consider improving sleep hygiene")
        st.write("- Maintain consistent sleep schedule")
    else:
        st.success("✅ You're getting adequate sleep. Maintain this healthy sleep pattern.")

    # Comprehensive Nutrition Summary
    st.header('Comprehensive Nutrition Summary ')
    
    # Daily Totals Explanation
    st.subheader('Understanding Your Daily Totals')
    st.write("Your daily nutrition targets and actuals are broken down as follows:")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Total Calories", 
            f"{daily_totals['calories']:.0f} kcal", 
            f"{(daily_totals['calories'] - pred_cal):.0f} kcal",
            delta_color="off"
        )
        st.write(f"Target: {pred_cal:.0f} kcal")
        st.progress(min(daily_totals['calories'] / pred_cal, 1.0))
        
    with col2:
        st.metric(
            "Total Protein", 
            f"{daily_totals['protein']:.1f}g",
            f"{(daily_totals['protein'] - protein_target):.1f}g",
            delta_color="off"
        )
        st.write(f"Target: {protein_target:.0f}g (25% of calories)")
        st.progress(min(daily_totals['protein'] / protein_target, 1.0))
        
    with col3:
        st.metric(
            "Total Carbs", 
            f"{daily_totals['carbs']:.1f}g",
            f"{(daily_totals['carbs'] - carbs_target):.1f}g",
            delta_color="off"
        )
        st.write(f"Target: {carbs_target:.0f}g (50% of calories)")
        st.progress(min(daily_totals['carbs'] / carbs_target, 1.0))
    
    st.subheader('Meal Distribution Analysis')
    meal_percentages = {
        'Breakfast': (meals['Breakfast']['calories'] / pred_cal) * 100,
        'Lunch': (meals['Lunch']['calories'] / pred_cal) * 100,
        'Dinner': (meals['Dinner']['calories'] / pred_cal) * 100,
        'Snacks': (meals['Snacks']['calories'] / pred_cal) * 100
    }
    
    st.write("Your daily calories are distributed as follows:")
    for meal, percentage in meal_percentages.items():
        st.write(f"- {meal}: {percentage:.1f}% of daily calories")
    
    st.subheader('Nutrition Balance')
    st.write("""
    Key points about your meal plan:
    - Protein: Helps maintain muscle mass and supports recovery
    - Carbohydrates: Provides energy for daily activities
    - Fats: Essential for nutrient absorption and hormone production
    
    Your meal plan is designed to provide:
    - Balanced nutrition across all meals
    - Appropriate portion sizes
    - Variety of nutrients
    - Consideration of your health conditions
    """)

