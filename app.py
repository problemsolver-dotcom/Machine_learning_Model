import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import warnings
from flask import Flask, request, render_template

# --- Configuration ---
MODEL_PATH = "plant_disease_model.pkl"
FEATURES_PATH = "feature_columns.pkl"
DATASET_PATH = "plant_disease_dataset.csv"

# --- Model Training Function ---
def train_and_save_model():
    """
    Trains the model using the dataset and saves it to .pkl files.
    This function contains the logic from your train_model.py script.
    """
    # Suppress warnings for a cleaner output
    warnings.filterwarnings('ignore')
    
    # Step 1: Load the dataset
    try:
        df = pd.read_csv(DATASET_PATH)
        print("✅ Dataset loaded successfully for training.")
    except FileNotFoundError:
        print(f"❌ Error: Dataset '{DATASET_PATH}' not found. Cannot train new model.")
        # Exit if the data isn't available to train
        exit()

    # Step 2: Define features (X) and target (y)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # Step 3: Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Step 4: Train the Random Forest model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    print("✅ Model training complete.")

    # Step 5: Evaluate the model
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"📈 New Model Accuracy: {acc * 100:.2f}%")

    # Step 6: Save the trained model and artifacts
    joblib.dump(model, MODEL_PATH)
    joblib.dump(X.columns.tolist(), FEATURES_PATH)
    print(f"✅ Model saved to '{MODEL_PATH}'")

# --- Main Application Setup ---

# Check if the model exists. If not, train it.
if not os.path.exists(MODEL_PATH):
    print(f"Model file not found at '{MODEL_PATH}'.")
    print("Starting new model training process...")
    train_and_save_model()
else:
    print(f"✅ Existing model found at '{MODEL_PATH}'. Loading it.")

# --- Flask App Initialization ---
app = Flask(__name__)

# --- Load the Model and Features for the App ---
try:
    model = joblib.load(MODEL_PATH)
    feature_columns = joblib.load(FEATURES_PATH)
    print("✅ Model and features loaded successfully for the web app.")
except Exception as e:
    print(f"❌ Critical Error: Could not load model files. {e}")
    model = None
    feature_columns = []

# --- Flask Routes ---
@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_result = None
    user_inputs = {}

    if request.method == 'POST':
        if model and feature_columns:
            try:
                # Get data from the form and convert to float
                input_data = [float(request.form[col]) for col in feature_columns]
                user_inputs = {col: request.form[col] for col in feature_columns}

                # Make a prediction
                prediction = model.predict([input_data])[0]

                # Format the result
                if prediction == 1:
                    prediction_result = "Yes, the plant is likely to have a disease."
                else:
                    prediction_result = "No, the plant is likely healthy."

            except Exception as e:
                print(f"Error during prediction: {e}")
                prediction_result = "Error: Could not process inputs. Please ensure all fields are numeric."
    
    return render_template('index.html', 
                           columns=feature_columns, 
                           result=prediction_result,
                           user_inputs=user_inputs)

# --- Run the App ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
