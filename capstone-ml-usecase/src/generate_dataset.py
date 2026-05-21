import pandas as pd
import numpy as np
import os

def generate_churn_dataset(n_samples=5000):
    np.random.seed(42)
    
    # Basic Demographics
    gender = np.random.choice(['Male', 'Female'], n_samples)
    senior = np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
    partner = np.random.choice(['Yes', 'No'], n_samples, p=[0.5, 0.5])
    
    # Account Details
    tenure = np.random.randint(1, 73, n_samples)
    contract = np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples, p=[0.5, 0.25, 0.25])
    
    # Services
    internet = np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples, p=[0.35, 0.45, 0.2])
    
    tech_support = []
    monthly_charges = []
    
    for i in range(n_samples):
        if internet[i] == 'No':
            tech_support.append('No internet service')
            monthly_charges.append(np.random.uniform(18, 25))
        else:
            ts = np.random.choice(['Yes', 'No'], p=[0.3, 0.7])
            tech_support.append(ts)
            base = 40 if internet[i] == 'DSL' else 75
            if ts == 'Yes': base += 15
            monthly_charges.append(base + np.random.uniform(-5, 15))
            
    # Churn Logic (to make it realistic for EDA/XAI)
    churn_prob = np.zeros(n_samples)
    churn_prob += np.where(contract == 'Month-to-month', 0.35, 0.05)
    churn_prob += np.where(np.array(internet) == 'Fiber optic', 0.15, 0.0)
    churn_prob -= np.where(np.array(tech_support) == 'Yes', 0.15, 0.0)
    churn_prob -= (tenure / 72) * 0.2
    
    churn_prob = np.clip(churn_prob, 0.05, 0.85)
    churn = [np.random.choice(['Yes', 'No'], p=[p, 1-p]) for p in churn_prob]
    
    df = pd.DataFrame({
        'CustomerID': [f"CUST_{1000+i}" for i in range(n_samples)],
        'Gender': gender,
        'SeniorCitizen': senior,
        'Partner': partner,
        'Tenure': tenure,
        'InternetService': internet,
        'TechSupport': tech_support,
        'Contract': contract,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': np.array(monthly_charges) * tenure,
        'Churn': churn
    })
    
    # Inject some missing values for preprocessing requirement
    idx_missing = np.random.choice(df.index, size=50, replace=False)
    df.loc[idx_missing, 'TotalCharges'] = np.nan
    
    # Save
    os.makedirs('../data/raw', exist_ok=True)
    df.to_csv('../data/raw/customer_churn.csv', index=False)
    print("customer_churn.csv dataset generated successfully.")

if __name__ == "__main__":
    generate_churn_dataset()
