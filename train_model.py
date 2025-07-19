import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import warnings

# Suppress warnings for a cleaner output
warnings.filterwarnings('ignore')

# --- Training and Saving the Model ---

# Step 1: Load the dataset
try:
    df = pd.read_csv('plant_disease_dataset.csv')
    print("✅ Dataset loaded successfully.")
except FileNotFoundError:
    print("❌ Error: 'plant_disease_dataset.csv' not found. Please ensure the file is in the correct directory.")
    exit()

# Step 2: Define features (X) and target (y)
# All columns except the last one are features
X = df.iloc[:, :-1]
# The last column is the target
y = df.iloc[:, -1]

# Step 3: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train the Random Forest model
# Using n_estimators=100 is a good starting point
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("✅ Model training complete.")

# Step 5: Evaluate the model's performance on the test set
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"📈 Model Accuracy: {acc * 100:.2f}%")

# Step 6: Save the trained model and required artifacts
joblib.dump(model, "plant_disease_model.pkl")
# Save the column names for the user input prompt
joblib.dump(X.columns.tolist(), "feature_columns.pkl")
print("✅ Model and feature columns saved successfully.")
print("\n You can now run the Flask app using: python app.py")
