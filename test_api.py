"""
Test suite for Churn Prediction API
"""

import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path

# Adicionar src ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.api import app

client = TestClient(app)


class TestHealthEndpoints:
    """Testes para endpoints de health"""
    
    def test_root_endpoint(self):
        """Testa endpoint raiz"""
        response = client.get("/")
        assert response.status_code == 200
        assert "message" in response.json()
        assert "version" in response.json()
    
    def test_health_check(self):
        """Testa health check"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "version" in data


class TestModelTraining:
    """Testes para treinamento do modelo"""
    
    def test_train_model(self):
        """Testa endpoint de treinamento"""
        response = client.post("/model/train")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "metrics" in data
        assert "roc_auc" in data["metrics"]


class TestPredictions:
    """Testes para predições"""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup: treina modelo antes dos testes"""
        client.post("/model/train")
    
    def test_single_prediction(self):
        """Testa predição individual"""
        customer = {
            "tenure": 12,
            "monthly_charges": 75.5,
            "total_charges": 900.0,
            "contract_type": 0,
            "payment_method": 1,
            "internet_service": 1,
            "online_security": 0,
            "tech_support": 1,
            "streaming_tv": 1,
            "streaming_movies": 0
        }
        
        response = client.post("/predict", json=customer)
        assert response.status_code == 200
        
        data = response.json()
        assert "churn_probability" in data
        assert "will_churn" in data
        assert "risk_level" in data
        assert "recommendations" in data
        assert 0 <= data["churn_probability"] <= 1
    
    def test_batch_prediction(self):
        """Testa predição em lote"""
        request = {
            "customers": [
                {
                    "tenure": 12,
                    "monthly_charges": 75.5,
                    "total_charges": 900.0,
                    "contract_type": 0,
                    "payment_method": 1,
                    "internet_service": 1,
                    "online_security": 0,
                    "tech_support": 1,
                    "streaming_tv": 1,
                    "streaming_movies": 0
                },
                {
                    "tenure": 36,
                    "monthly_charges": 45.0,
                    "total_charges": 1620.0,
                    "contract_type": 2,
                    "payment_method": 0,
                    "internet_service": 0,
                    "online_security": 1,
                    "tech_support": 1,
                    "streaming_tv": 0,
                    "streaming_movies": 0
                }
            ]
        }
        
        response = client.post("/predict/batch", json=request)
        assert response.status_code == 200
        
        data = response.json()
        assert "predictions" in data
        assert "total_customers" in data
        assert "high_risk_count" in data
        assert len(data["predictions"]) == 2
        assert data["total_customers"] == 2
    
    def test_invalid_customer_data(self):
        """Testa validação de dados inválidos"""
        invalid_customer = {
            "tenure": -5,  # Valor negativo inválido
            "monthly_charges": 75.5,
            "total_charges": 900.0,
            "contract_type": 0,
            "payment_method": 1,
            "internet_service": 1,
            "online_security": 0,
            "tech_support": 1,
            "streaming_tv": 1,
            "streaming_movies": 0
        }
        
        response = client.post("/predict", json=invalid_customer)
        assert response.status_code == 422  # Validation error


class TestModelInfo:
    """Testes para informações do modelo"""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup: treina modelo antes dos testes"""
        client.post("/model/train")
    
    def test_model_info(self):
        """Testa endpoint de informações do modelo"""
        response = client.get("/model/info")
        assert response.status_code == 200
        
        data = response.json()
        assert "model_type" in data
        assert "features" in data
        assert "feature_importance" in data
        assert isinstance(data["features"], list)
        assert len(data["features"]) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
