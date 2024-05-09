import pandas as pd
import openpyxl
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import joblib

def get_data_from_excel():
    file_path = "supermarkt_sales.xlsx"
    wb = openpyxl.load_workbook(file_path)
    sheet = wb.active
    
    # Skip 3 rows as empty, then read headers
    data = sheet.iter_rows(values_only=True)
    for _ in range(3):
        next(data)
    headers = next(data)

    # Read data starting from 5th row
    rows = []
    for row in data:
        rows.append(dict(zip(headers, row)))

    # Create DataFrame
    df = pd.DataFrame(rows)
    return df

def train_and_save_model():
    # Load data
    df = get_data_from_excel()

    # Select features and target variable
    features = ['City', 'Customer_type', 'Gender', 'Product line', 'Rating']
    target = 'Total'
    X = df[features]
    y = df[target]

    # Initialize label encoders for categorical features
    label_encoders = {}
    for feature in ['City', 'Customer_type', 'Gender', 'Product line']:
        label_encoders[feature] = LabelEncoder()
        X[feature] = label_encoders[feature].fit_transform(X[feature])

    # Train Random Forest Regression model
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X, y)

    # Save the model
    joblib.dump(rf_model, "model.joblib")

    # Save the label encoders
    for feature, encoder in label_encoders.items():
        joblib.dump(encoder, f"{feature}_encoder.joblib")

if __name__ == "__main__":
    train_and_save_model()
