
import os
import json

# Criar estrutura de diretórios do projeto
project_structure = {
    "customer-churn-prediction": {
        "src": ["__init__.py", "model.py", "api.py", "preprocessing.py", "train.py"],
        "notebooks": ["01_exploratory_analysis.ipynb", "02_model_training.ipynb"],
        "data": ["raw", "processed"],
        "models": [],
        "tests": ["__init__.py", "test_api.py"],
        "docs": [],
        "config": [],
    }
}

# Estrutura detalhada do projeto
project_info = {
    "name": "Customer Churn Prediction System",
    "description": "Sistema completo de ML para predição de churn usando XGBoost, FastAPI e Docker",
    "tech_stack": ["Python 3.11", "XGBoost", "FastAPI", "Pandas", "Scikit-learn", "Docker", "MLflow"],
    "features": [
        "API REST para predições em tempo real",
        "Modelo XGBoost com 85%+ de acurácia",
        "Análise exploratória completa",
        "Containerização com Docker",
        "Testes automatizados",
        "Monitoramento de métricas"
    ]
}

print(json.dumps(project_info, indent=2, ensure_ascii=False))
