"""
Customer Churn Prediction Model
Implementação do modelo XGBoost para predição de churn
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import joblib
from pathlib import Path
from loguru import logger

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    roc_auc_score,
    precision_recall_curve,
    f1_score
)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE


class ChurnPredictor:
    """Classe para predição de churn de clientes"""
    
    def __init__(self, model_path: str = None):
        """
        Inicializa o preditor
        
        Args:
            model_path: Caminho para modelo salvo (opcional)
        """
        self.model = None
        self.feature_names = None
        self.threshold = 0.5
        
        if model_path and Path(model_path).exists():
            self.load_model(model_path)
    
    def train(
        self, 
        X: pd.DataFrame, 
        y: pd.Series,
        use_smote: bool = True,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Dict:
        """
        Treina o modelo XGBoost
        
        Args:
            X: Features
            y: Target (churn)
            use_smote: Usar SMOTE para balanceamento
            test_size: Tamanho do conjunto de teste
            random_state: Seed para reprodutibilidade
            
        Returns:
            Dicionário com métricas de treino
        """
        logger.info(f"Iniciando treinamento com {len(X)} amostras")
        
        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Aplicar SMOTE se necessário
        if use_smote:
            logger.info("Aplicando SMOTE para balanceamento de classes")
            smote = SMOTE(random_state=random_state)
            X_train, y_train = smote.fit_resample(X_train, y_train)
        
        # Configurar modelo XGBoost
        self.model = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=1,
            reg_alpha=0.1,
            reg_lambda=1,
            scale_pos_weight=1,
            random_state=random_state,
            eval_metric='logloss',
            early_stopping_rounds=20
        )
        
        # Treinar modelo
        logger.info("Treinando modelo XGBoost...")
        self.model.fit(
            X_train, 
            y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )
        
        self.feature_names = list(X.columns)
        
        # Avaliar modelo
        metrics = self.evaluate(X_test, y_test)
        
        logger.info(f"Treinamento concluído. ROC-AUC: {metrics['roc_auc']:.4f}")
        
        return metrics
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """
        Avalia o modelo
        
        Args:
            X: Features de teste
            y: Target de teste
            
        Returns:
            Dicionário com métricas
        """
        # Predições
        y_pred = self.model.predict(X)
        y_pred_proba = self.model.predict_proba(X)[:, 1]
        
        # Calcular métricas
        metrics = {
            'roc_auc': roc_auc_score(y, y_pred_proba),
            'f1_score': f1_score(y, y_pred),
            'classification_report': classification_report(y, y_pred),
            'confusion_matrix': confusion_matrix(y, y_pred).tolist()
        }
        
        return metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Faz predições (classe binária)
        
        Args:
            X: Features
            
        Returns:
            Array de predições (0 ou 1)
        """
        if self.model is None:
            raise ValueError("Modelo não treinado. Execute train() primeiro.")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Retorna probabilidades de churn
        
        Args:
            X: Features
            
        Returns:
            Array de probabilidades
        """
        if self.model is None:
            raise ValueError("Modelo não treinado. Execute train() primeiro.")
        
        return self.model.predict_proba(X)[:, 1]
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Retorna importância das features
        
        Returns:
            DataFrame com features e importâncias
        """
        if self.model is None:
            raise ValueError("Modelo não treinado.")
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def save_model(self, path: str):
        """
        Salva o modelo treinado
        
        Args:
            path: Caminho para salvar o modelo
        """
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'threshold': self.threshold
        }
        
        joblib.dump(model_data, path)
        logger.info(f"Modelo salvo em {path}")
    
    def load_model(self, path: str):
        """
        Carrega modelo salvo
        
        Args:
            path: Caminho do modelo
        """
        model_data = joblib.load(path)
        
        self.model = model_data['model']
        self.feature_names = model_data['feature_names']
        self.threshold = model_data.get('threshold', 0.5)
        
        logger.info(f"Modelo carregado de {path}")


def create_sample_data(n_samples: int = 1000) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Cria dataset de exemplo para demonstração
    
    Args:
        n_samples: Número de amostras
        
    Returns:
        Tupla (X, y) com features e target
    """
    np.random.seed(42)
    
    # Features simuladas
    data = {
        'tenure': np.random.randint(0, 72, n_samples),
        'monthly_charges': np.random.uniform(20, 120, n_samples),
        'total_charges': np.random.uniform(100, 8000, n_samples),
        'contract_type': np.random.choice([0, 1, 2], n_samples),  # Month-to-month, One year, Two year
        'payment_method': np.random.choice([0, 1, 2, 3], n_samples),
        'internet_service': np.random.choice([0, 1, 2], n_samples),  # DSL, Fiber, No
        'online_security': np.random.choice([0, 1], n_samples),
        'tech_support': np.random.choice([0, 1], n_samples),
        'streaming_tv': np.random.choice([0, 1], n_samples),
        'streaming_movies': np.random.choice([0, 1], n_samples),
    }
    
    X = pd.DataFrame(data)
    
    # Criar target baseado em regras (para simular churn real)
    churn_probability = (
        0.3 * (X['tenure'] < 12) +
        0.2 * (X['contract_type'] == 0) +
        0.15 * (X['monthly_charges'] > 80) +
        0.1 * (X['internet_service'] == 1) +
        0.1 * (1 - X['tech_support']) +
        0.15 * np.random.random(n_samples)
    )
    
    y = (churn_probability > 0.5).astype(int)
    
    return X, y


if __name__ == "__main__":
    # Teste do modelo
    logger.info("Criando dados de exemplo...")
    X, y = create_sample_data(1000)
    
    logger.info(f"Dataset criado: {X.shape}")
    logger.info(f"Distribuição de churn: {y.value_counts().to_dict()}")
    
    # Treinar modelo
    predictor = ChurnPredictor()
    metrics = predictor.train(X, y)
    
    print("\nMétricas do modelo:")
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"F1-Score: {metrics['f1_score']:.4f}")
    print("\nClassification Report:")
    print(metrics['classification_report'])
    
    # Feature importance
    print("\nFeature Importance:")
    print(predictor.get_feature_importance().head(10))
