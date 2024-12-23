from flask import Flask, render_template_string, request, jsonify
import pandas as pd
import numpy as np
import pickle
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score


app = Flask(__name__)

# Load and preprocess the dataset
def load_and_preprocess_data():
    car = pd.read_csv('car_data.csv')

    # Clean and preprocess data
    car = car[car['year'].str.isnumeric()]
    car['year'] = car['year'].astype(int)
    car = car[car['Price'] != 'Ask For Price']
    car['Price'] = car['Price'].str.replace(',', '').astype(int)

    car['kms_driven'] = car['kms_driven'].str.split().str.get(0).str.replace(',', '')
    car = car[car['kms_driven'].str.isnumeric()]
    car['kms_driven'] = car['kms_driven'].astype(int)

    # Adding new feature: Age of the car
    car['age'] = 2024 - car['year']

    # Drop unnecessary columns
    car = car[['name', 'company', 'age', 'kms_driven', 'fuel_type', 'Price']]
    car = car[~car['fuel_type'].isna()]
    car['name'] = car['name'].str.split().str.slice(start=0, stop=3).str.join(' ')
    car = car.reset_index(drop=True)

    return car

# Train the model
def train_model():
    car = load_and_preprocess_data()

    X = car[['name', 'company', 'age', 'kms_driven', 'fuel_type']]
    y = car['Price']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    # Define preprocessing and model pipeline
    column_trans = ColumnTransformer(
        transformers=[
            ('ohe', OneHotEncoder(handle_unknown='ignore'), ['name', 'company', 'fuel_type'])
        ],
        remainder='passthrough'
    )

    pipe = Pipeline([
        ('preprocessor', column_trans),
        ('regressor', XGBRegressor(random_state=42, n_estimators=100, learning_rate=0.1, max_depth=5, reg_alpha=0, reg_lambda=1))
    ])

    # Fit the model
    pipe.fit(X_train, y_train)

    # Validate the model
    prediction = pipe.predict(X_test)

    # Calculate performance metrics
    r2 = r2_score(y_test, prediction)

    print(f'R² = {r2}')  # Print R²

    pickle.dump(pipe, open('BestXGBoostModelWithRegularization.pkl', 'wb'))
    return pipe, car


# Load the trained model and data
model, car_data = train_model()

# Prepare unique values for dropdowns
car_companies = car_data['company'].unique()

# Create a dictionary for models by company
models_by_company = car_data.groupby('company')['name'].apply(list).to_dict()

# Define HTML templates
index_html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Car Price Predictor</title>
    <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
    <style>
        body {{ background: #120035; }}
        .container {{ margin-top: 50px; max-width: 600px; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 0 10px rgba(0, 0, 0, 0.1); }}
        .btn-primary {{ background: #008D09; border: none; }}
        .btn-primary:hover {{ background: #00CF0E; }}
        .form-group label {{ font-weight: bold; }}
        .btn {{ margin-top: 10px; }}
    </style>
</head>
<body>
    <div class="container">
        <h2 class="text-center mt-4">Car Price Predictor</h2>
        <form action="/predict" method="POST">
            <div class="form-group">
                <label for="company">Company</label>
                <select class="form-control" id="company" name="company" onchange="updateModels()" required>
                    <option value="" disabled selected>Select Company</option>
                    {"".join([f"<option value='{company}'>{company}</option>" for company in car_companies])}
                </select>
            </div>
            <div class="form-group">
                <label for="name">Car Name</label>
                <select class="form-control" id="name" name="name" required>
                    <option value="" disabled selected>Select Car Name</option>
                </select>
            </div>
            <div class="form-group">
                <label for="year">Year</label>
                <input type="number" class="form-control" id="year" name="year" required>
            </div>
            <div class="form-group">
                <label for="kms_driven">Kilometers Driven</label>
                <input type="number" class="form-control" id="kms_driven" name="kms_driven" required>
            </div>
            <div class="form-group">
                <label for="fuel_type">Fuel Type</label>
                <select class="form-control" id="fuel_type" name="fuel_type" required>
                    <option value="" disabled selected>Select Fuel Type</option>
                    <option value="Petrol">Petrol</option>
                    <option value="Diesel">Diesel</option>
                    <option value="CNG">CNG</option>
                    <option value="Electric">Electric</option>
                </select>
            </div>
            <button type="submit" class="btn btn-primary btn-block">Predict Price</button>
        </form>
    </div>

    <script>
        const modelsByCompany = {models_by_company};

        function updateModels() {{
            const companySelect = document.getElementById('company');
            const nameSelect = document.getElementById('name');
            const selectedCompany = companySelect.value;

            // Clear the car name dropdown
            nameSelect.innerHTML = '<option value="" disabled selected>Select Car Name</option>';

            if (selectedCompany) {{
                const models = modelsByCompany[selectedCompany];
                const uniqueModels = [...new Set(models)];  // Remove duplicates

                uniqueModels.forEach(model => {{
                    const option = document.createElement('option');
                    option.value = model;
                    option.textContent = model;
                    nameSelect.appendChild(option);
                }});
            }}
        }}
    </script>
</body>
</html>
"""

result_html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Car Price Prediction Result</title>
    <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
    <style>
        body { background: #200069; }
        .container { margin-top: 50px; max-width: 600px; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 0 10px rgba(0, 0, 0, 0.1); }
        .alert-info { font-size: 1.5rem; }
        .btn-secondary { background: #A90014; border: none; }
        .btn-secondary:hover { background: #B80000; }
    </style>
</head>
<body>
    <div class="container">
        <h2 class="text-center mt-4">Car Price Predictor</h2>
        <div class="alert alert-info text-center">
            {{ prediction_text }}
        </div>
        <div class="text-center">
            <a href="/" class="btn btn-secondary">Go Back</a>
        </div>
    </div>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(index_html)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        name = request.form['name']
        company = request.form['company']
        year = int(request.form['year'])
        kms_driven = int(request.form['kms_driven'])
        fuel_type = request.form['fuel_type']

        input_data = pd.DataFrame([[name, company, 2024 - year, kms_driven, fuel_type]],  # Age calculated here
                                   columns=['name', 'company', 'age', 'kms_driven', 'fuel_type'])

        # Predict price
        prediction = model.predict(input_data)

        return render_template_string(result_html, prediction_text=f'Estimated Price: ₹{int(prediction[0])}')

if __name__ == "__main__":
    app.run(debug=True)
