
import pandas as pd
import numpy as np
import pickle
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import os

# Set file paths
DATASET_PATH = r'c:\Users\ayush\OneDrive\Desktop\project2\WA_Fn-UseC_-HR-Employee-Attrition.csv'
MODEL_PKL_PATH = r'c:\Users\ayush\OneDrive\Desktop\project2\model.pkl'
MODEL_JOBLIB_PATH = r'c:\Users\ayush\OneDrive\Desktop\project2\model.joblib'
ENCODERS_PATH = r'c:\Users\ayush\OneDrive\Desktop\project2\encoders.pkl'

def load_and_clean_data(filepath):
    print("Loading dataset...")
    df = pd.read_csv(filepath)
    
    # 1. Drop useless cols
    drop_cols = ['EmployeeCount', 'Over18', 'StandardHours', 'EmployeeNumber'] # EmployeeNumber is ID, not useful for prediction
    df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')
    
    # 2. Split Features and Target
    if 'Attrition' not in df.columns:
        raise ValueError("Target column 'Attrition' not found in dataset")
        
    X = df.drop('Attrition', axis=1)
    y = df['Attrition']
    
    return X, y

def train_model():
    X, y = load_and_clean_data(DATASET_PATH)
    
    # 3. Encoding
    encoders = {}
    X_encoded = X.copy()
    
    # Encode categorical features
    # Note: For production/Flask, we MUST save these encoders to handle future input
    cat_cols = X_encoded.select_dtypes(include=['object']).columns
    
    for col in cat_cols:
        le = LabelEncoder()
        X_encoded[col] = le.fit_transform(X_encoded[col])
        encoders[col] = le
        
    # Encode target
    y_le = LabelEncoder()
    y_encoded = y_le.fit_transform(y) # No -> 0, Yes -> 1 typically
    encoders['Target'] = y_le
    
    # Save encoders for later use in Flask app
    with open(ENCODERS_PATH, 'wb') as f:
        pickle.dump(encoders, f)
    print(f"Encoders saved to {ENCODERS_PATH}")

    # 4. Split Data
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_encoded, test_size=0.2, random_state=42)
    
    # 5. Train Random Forest
    print("Training Random Forest Model...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    # 6. Evaluate
    y_pred = rf_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # 7. Save Model
    # Pickle
    with open(MODEL_PKL_PATH, 'wb') as f:
        pickle.dump(rf_model, f)
    print(f"Model saved as pickle: {MODEL_PKL_PATH}")
    
    # Joblib
    joblib.dump(rf_model, MODEL_JOBLIB_PATH)
    print(f"Model saved as joblib: {MODEL_JOBLIB_PATH}")
    
    return rf_model, encoders, X_encoded.columns

def predict_single_input(data_dict, model, encoders, feature_names):
    """
    Simulates prediction for a single user (like from a Flask form).
    Also provides a basic explanation.
    """
    # Create DF
    input_df = pd.DataFrame([data_dict])
    
    # Encode
    for col, le in encoders.items():
        if col == 'Target': continue
        if col in input_df.columns:
            # Handle unknown labels delicately or just use simple transform for now
            # In a real app, you'd handle unseen labels
            try:
                input_df[col] = le.transform(input_df[col])
            except ValueError:
                # Fallback or error
                print(f"Warning: Unknown value in {col}")
                input_df[col] = -1 # Or some default
                
    # Ensure correct column order
    input_df = input_df[feature_names]
    
    # Predict
    pred_prob = model.predict_proba(input_df)[0]
    prediction = model.predict(input_df)[0]
    
    pred_label = encoders['Target'].inverse_transform([prediction])[0]
    
    # Explainability (Simple Feature Importance based)
    # We look at global importance and maybe just highlight the values of top features
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    print("\n--- Prediction Result ---")
    print(f"Prediction: {pred_label}")
    print(f"Confidence (Leave): {pred_prob[1]:.2f}")
    print(f"Confidence (Stay): {pred_prob[0]:.2f}")
    
    print("\n--- Why? (Top Influential Factors) ---")
    print("Based on the model, these are the most important factors globally:")
    for f in range(5):
        idx = indices[f]
        feature_name = feature_names[idx]
        val = data_dict.get(feature_name, "N/A")
        importance = importances[idx]
        print(f"{f+1}. {feature_name}: {val} (Importance: {importance:.4f})")
        
    # Heuristic suggestion (very basic)
    if pred_label == 'Yes':
        print("\n--- Suggestion ---")
        print("Employee is at risk of leaving. Check the top factors above.")
        print("Common retention strategies: Improve Work-Life Balance, Review Salary, Check OverTime.")

if __name__ == "__main__":
    if os.path.exists(DATASET_PATH):
        trained_model, trained_encoders, feature_cols = train_model()
        
        # Test with a sample from the dataset (first row)
        print("\nTesting with a dummy input (First row of dataset):")
        df_test = pd.read_csv(DATASET_PATH)
        sample_data = df_test.iloc[0].to_dict()
        sample_data.pop('Attrition', None) # Remove target
        
        # Clean sample data similarly (drop ignored cols)
        for junk in ['EmployeeCount', 'Over18', 'StandardHours', 'EmployeeNumber']:
            sample_data.pop(junk, None)
            
        predict_single_input(sample_data, trained_model, trained_encoders, feature_cols)
        
    else:
        print(f"Error: Dataset not found at {DATASET_PATH}")
