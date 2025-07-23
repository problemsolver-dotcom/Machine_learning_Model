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

Language           -----------> Python

Backend            -----------> Flask

Machine Learning   -----------> Scikit-learn, Pandas, Joblib

Frontend           -----------> HTML5, Bootstrap 5

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

Place your plant_disease_dataset.csv file in the root directory of the project.

▶️ How to Run the Application
Once the setup is complete, you can start the application with a single command:

python app.py

On the very first run, the script will automatically train the machine learning model and save it as plant_disease_model.pkl. You will see the training progress in the terminal.

On subsequent runs, the script will find the existing model and load it directly.

After starting the script, open your web browser and navigate to:
https://www.google.com/search?q=http://127.0.0.1:5000

🤖 How It Works
Model Training: When app.py is executed, it first checks if a plant_disease_model.pkl file exists.

Automatic Training: If the model file is not found, the script reads plant_disease_dataset.csv, trains a RandomForestClassifier, and saves the trained model and feature columns as .pkl files.

Web Server: The Flask web server starts, ready to handle requests.

User Input: A user navigates to the website and selects values from the dropdown menus (e.g., Temperature: "Normal", Soil Ph: "Low").

Prediction: The Flask app receives the form data, maps the user-friendly text to the corresponding numerical values, and feeds
