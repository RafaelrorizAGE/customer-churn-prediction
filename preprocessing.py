"""
Preprocessing utilities for customer churn data
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple, Dict
import joblib


class ChurnPreprocessor:
    """Preprocessador de dados para predição de churn"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = None
        
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ajusta e transforma os dados
        
        Args:
            df: DataFrame com dados brutos
            
        Returns:
            DataFrame processado
        """
        df_processed = df.copy()
        
        # Identificar colunas numéricas e categóricas
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df_processed.select_dtypes(include=['object']).columns.tolist()
        
        # Encodar variáveis categóricas
        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
            df_processed[col] = self.label_encoders[col].fit_transform(df_processed[col].astype(str))
        
        # Escalar features numéricas
        if numeric_cols:
            df_processed[numeric_cols] = self.scaler.fit_transform(df_processed[numeric_cols])
        
        self.feature_names = df_processed.columns.tolist()
        
        return df_processed
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transforma novos dados usando ajustes anteriores
        
        Args:
            df: DataFrame com dados novos
            
        Returns:
            DataFrame processado
        """
        df_processed = df.copy()
        
        # Aplicar label encoders
        for col, encoder in self.label_encoders.items():
            if col in df_processed.columns:
                df_processed[col] = encoder.transform(df_processed[col].astype(str))
        
        # Aplicar scaler
        numeric_cols = [col for col in self.feature_names 
                       if col not in self.label_encoders.keys()]
        
        if numeric_cols:
            df_processed[numeric_cols] = self.scaler.transform(df_processed[numeric_cols])
        
        return df_processed
    
    def save(self, path: str):
        """Salva preprocessador"""
        joblib.dump({
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names
        }, path)
    
    def load(self, path: str):
        """Carrega preprocessador"""
        data = joblib.load(path)
        self.scaler = data['scaler']
        self.label_encoders = data['label_encoders']
        self.feature_names = data['feature_names']
