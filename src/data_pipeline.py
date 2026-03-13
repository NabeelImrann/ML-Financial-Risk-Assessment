import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

class FinancialDataPipeline:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}

    def preprocess(self, df):
        """
        Specialist cleaning and feature engineering.
        """
        # Handling missing values
        df = df.fillna(df.median(numeric_only=True))
        
        # Specialist Feature Engineering
        if 'income' in df.columns and 'debt' in df.columns:
            df['debt_to_income_ratio'] = df['debt'] / (df['income'] + 1)
        
        # Categorical Encoding
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            self.label_encoders[col] = le
            
        return df

    def split_and_scale(self, df, target='target'):
        X = df.drop(columns=[target])
        y = df[target]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test, X.columns
