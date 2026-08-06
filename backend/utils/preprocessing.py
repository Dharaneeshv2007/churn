import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# =========================
# LOAD DATASET
# =========================
def load_and_prepare_data(path):
    df = pd.read_csv(path)

    # clean TotalCharges
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df = df.dropna()

    return df


# =========================
# FEATURE SET (MATCH FRONTEND)
# =========================
NUM_COLS = [
    'SeniorCitizen',
    'tenure',
    'MonthlyCharges',
    'TotalCharges'
]

CAT_COLS = [
    'gender',
    'Partner',
    'Dependents',
    'InternetService',
    'Contract'
]


# =========================
# PREPROCESSING FUNCTION
# =========================
def preprocess_data(df, fit=True, scaler=None, encoder=None):

    df = df.copy()

    # Keep only required columns
    required_cols = NUM_COLS + CAT_COLS + (['Churn'] if 'Churn' in df.columns else [])
    df = df.reindex(columns=required_cols)

    # Target
    y = None
    if 'Churn' in df.columns:
        y = df['Churn'].map({'Yes': 1, 'No': 0})
        df = df.drop('Churn', axis=1)

    # Fill missing
    df[NUM_COLS] = df[NUM_COLS].fillna(0)
    df[CAT_COLS] = df[CAT_COLS].fillna("Unknown")

    # Order
    df = df[NUM_COLS + CAT_COLS]

    # Scale + Encode
    if fit:
        scaler = StandardScaler()
        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

        X_num = scaler.fit_transform(df[NUM_COLS])
        X_cat = encoder.fit_transform(df[CAT_COLS])
    else:
        X_num = scaler.transform(df[NUM_COLS])
        X_cat = encoder.transform(df[CAT_COLS])

    X = np.hstack([X_num, X_cat])

    feature_names = NUM_COLS + list(encoder.get_feature_names_out(CAT_COLS))

    return X, y, scaler, encoder, feature_names