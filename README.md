#Machine_learning_Model

Plant Disease Predictor Web Application
A user-friendly web application built with Flask and Scikit-learn to predict the likelihood of plant disease based on environmental factors. The app features a clean interface that allows even non-technical users to get predictions by selecting simple, descriptive options.

🌟 Features
User-Friendly Interface: An intuitive web form designed for ease of use.

Simple Dropdown Inputs: Users can select options like "Normal" or "Very High" for temperature and humidity, which are then mapped to numerical values for the model.

Automatic Model Training: The application automatically trains and saves a machine learning model on the first run if one doesn't already exist.

Real-time Predictions: Get instant predictions from the trained Random Forest model.

Responsive Design: The interface is built with Bootstrap and is accessible on both desktop and mobile devices.

🛠️ Tech Stack
This project leverages a full-stack approach, combining a Python backend for machine learning and web serving with a standard HTML/CSS frontend.

Category

Technology

Language

Python

Backend

Flask

Machine Learning

Scikit-learn, Pandas, Joblib

Frontend

HTML5, Bootstrap 5

🚀 How the Tech is Used to the Fullest
This project is designed to use each technology for its core strengths, creating an efficient and cohesive application.

Flask: Serves as the lightweight yet powerful web framework. It handles routing (directing URL requests to the correct Python function), manages the request-response cycle, and uses the Jinja2 templating engine to dynamically generate the HTML form based on the dataset's columns. This makes the frontend adaptable to changes in the model's features.

Scikit-learn: The heart of the predictive power. We use its RandomForestClassifier for its robustness and high accuracy. The train_test_split function is crucial for creating a reliable evaluation set, and accuracy_score provides a clear metric for model performance.

Pandas: Essential for data manipulation. It's used to read the plant_disease_dataset.csv into a DataFrame, which is an ideal structure for cleanly separating features (X) from the target variable (y) using iloc.

Joblib: Optimized for saving and loading large NumPy arrays and Python objects, making it perfect for model persistence. It efficiently serializes the trained Scikit-learn model (.pkl), allowing the application to load a pre-trained model instantly on startup instead of retraining every time.

Bootstrap 5: Provides a responsive, mobile-first frontend without the need for extensive custom CSS. We leverage its grid system (row, col-md-6) for layout, form components (form-select, form-control) for a clean user interface, and utility classes (shadow-lg, text-center) for quick and consistent styling.

HTML5 & Jinja2: The structure of the website is built on standard HTML5. The Jinja2 templating engine, integrated within Flask, adds dynamic capabilities. It's used to loop through the feature columns to build the form, conditionally create dropdowns versus number inputs, and display the final prediction result, making the frontend code clean and reusable.

📁 Project Structure
The project is organized as follows:

.
├── app.py                  # Main Flask application with all logic
├── plant_disease_dataset.csv # The dataset used for training the model
├── templates/
│   └── index.html          # The HTML template for the web interface
├── static/
│   └── (empty)             # For CSS or JS files if needed
├── plant_disease_model.pkl   # (Generated automatically)
└── feature_columns.pkl     # (Generated automatically)

⚙️ Setup and Installation
To run this project locally, follow these steps:

1. Clone the Repository

git clone [https://github.com/your-username/your-repository-name.git](https://github.com/your-username/your-repository-name.git)
cd your-repository-name

2. Create a Virtual Environment (Recommended)

It's good practice to create a virtual environment to manage project dependencies.

On macOS/Linux:

python3 -m venv venv
source venv/bin/activate

On Windows:

python -m venv venv
.\venv\Scripts\activate

3. Install Required Libraries

Install all the necessary Python libraries using the following command:

pip install Flask pandas scikit-learn joblib

4. Add the Dataset

Place your `plant_disease_dataset
