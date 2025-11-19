"""
FastAPI Application for Churn Prediction
API REST para predição de churn em tempo real
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional
import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
import os

from src.model import ChurnPredictor

# Configuração
MODEL_PATH = os.getenv("MODEL_PATH", "models/xgboost_model.pkl")

# Inicializar FastAPI
app = FastAPI(
    title="Customer Churn Prediction API",
    description="API para predição de churn de clientes usando XGBoost",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Carregar modelo
predictor = ChurnPredictor()

# Schemas Pydantic
class CustomerData(BaseModel):
    """Schema para dados de um cliente"""
    tenure: int = Field(..., ge=0, description="Meses como cliente")
    monthly_charges: float = Field(..., ge=0, description="Cobrança mensal")
    total_charges: float = Field(..., ge=0, description="Total cobrado")
    contract_type: int = Field(..., ge=0, le=2, description="Tipo de contrato (0=mensal, 1=1 ano, 2=2 anos)")
    payment_method: int = Field(..., ge=0, le=3, description="Método de pagamento")
    internet_service: int = Field(..., ge=0, le=2, description="Serviço de internet")
    online_security: int = Field(..., ge=0, le=1, description="Segurança online (0=não, 1=sim)")
    tech_support: int = Field(..., ge=0, le=1, description="Suporte técnico")
    streaming_tv: int = Field(..., ge=0, le=1, description="Streaming TV")
    streaming_movies: int = Field(..., ge=0, le=1, description="Streaming filmes")
    
    class Config:
        schema_extra = {
            "example": {
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
        }


class PredictionRequest(BaseModel):
    """Schema para request de predição"""
    customers: List[CustomerData]


class PredictionResponse(BaseModel):
    """Schema para resposta de predição"""
    customer_id: int
    churn_probability: float
    will_churn: bool
    risk_level: str
    recommendations: List[str]


class BatchPredictionResponse(BaseModel):
    """Schema para resposta batch"""
    predictions: List[PredictionResponse]
    total_customers: int
    high_risk_count: int


class HealthResponse(BaseModel):
    """Schema para health check"""
    status: str
    model_loaded: bool
    version: str


# Funções auxiliares
def get_risk_level(probability: float) -> str:
    """Determina nível de risco baseado na probabilidade"""
    if probability < 0.3:
        return "Baixo"
    elif probability < 0.6:
        return "Médio"
    else:
        return "Alto"


def get_recommendations(data: Dict, probability: float) -> List[str]:
    """Gera recomendações baseadas nos dados do cliente"""
    recommendations = []
    
    if probability > 0.5:
        if data['contract_type'] == 0:
            recommendations.append("Oferecer contrato de longo prazo com desconto")
        
        if data['tech_support'] == 0:
            recommendations.append("Incluir suporte técnico gratuito por 3 meses")
        
        if data['monthly_charges'] > 70:
            recommendations.append("Revisar plano para reduzir custos mensais")
        
        if data['tenure'] < 12:
            recommendations.append("Programa de fidelidade para novos clientes")
        
        if data['online_security'] == 0:
            recommendations.append("Oferecer pacote de segurança como incentivo")
    else:
        recommendations.append("Cliente com baixo risco de churn - manter engajamento")
    
    return recommendations if recommendations else ["Monitorar satisfação regularmente"]


# Endpoints
@app.get("/", tags=["Root"])
async def root():
    """Endpoint raiz"""
    return {
        "message": "Customer Churn Prediction API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check da API"""
    return HealthResponse(
        status="healthy",
        model_loaded=predictor.model is not None,
        version="1.0.0"
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
async def predict_single(customer: CustomerData):
    """
    Prediz churn para um único cliente
    
    Args:
        customer: Dados do cliente
        
    Returns:
        Predição com probabilidade e recomendações
    """
    try:
        # Verificar se modelo está carregado
        if predictor.model is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Modelo não carregado. Execute o treinamento primeiro."
            )
        
        # Converter para DataFrame
        customer_dict = customer.dict()
        df = pd.DataFrame([customer_dict])
        
        # Fazer predição
        probability = float(predictor.predict_proba(df)[0])
        will_churn = bool(predictor.predict(df)[0])
        risk_level = get_risk_level(probability)
        recommendations = get_recommendations(customer_dict, probability)
        
        return PredictionResponse(
            customer_id=0,
            churn_probability=round(probability, 4),
            will_churn=will_churn,
            risk_level=risk_level,
            recommendations=recommendations
        )
        
    except Exception as e:
        logger.error(f"Erro na predição: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro ao processar predição: {str(e)}"
        )


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Predictions"])
async def predict_batch(request: PredictionRequest):
    """
    Prediz churn para múltiplos clientes
    
    Args:
        request: Lista de clientes
        
    Returns:
        Predições em lote
    """
    try:
        if predictor.model is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Modelo não carregado"
            )
        
        # Converter para DataFrame
        customers_data = [c.dict() for c in request.customers]
        df = pd.DataFrame(customers_data)
        
        # Fazer predições
        probabilities = predictor.predict_proba(df)
        predictions_binary = predictor.predict(df)
        
        # Criar respostas
        predictions = []
        high_risk_count = 0
        
        for idx, (prob, pred, customer_dict) in enumerate(zip(probabilities, predictions_binary, customers_data)):
            prob = float(prob)
            risk_level = get_risk_level(prob)
            
            if risk_level == "Alto":
                high_risk_count += 1
            
            predictions.append(
                PredictionResponse(
                    customer_id=idx,
                    churn_probability=round(prob, 4),
                    will_churn=bool(pred),
                    risk_level=risk_level,
                    recommendations=get_recommendations(customer_dict, prob)
                )
            )
        
        return BatchPredictionResponse(
            predictions=predictions,
            total_customers=len(predictions),
            high_risk_count=high_risk_count
        )
        
    except Exception as e:
        logger.error(f"Erro na predição batch: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro ao processar predições: {str(e)}"
        )


@app.get("/model/info", tags=["Model"])
async def model_info():
    """Retorna informações sobre o modelo"""
    if predictor.model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modelo não carregado"
        )
    
    feature_importance = predictor.get_feature_importance()
    
    return {
        "model_type": "XGBoost Classifier",
        "features": predictor.feature_names,
        "feature_importance": feature_importance.to_dict('records'),
        "threshold": predictor.threshold
    }


@app.post("/model/train", tags=["Model"])
async def train_model():
    """
    Treina o modelo com dados de exemplo
    (Em produção, isso seria feito offline)
    """
    try:
        from src.model import create_sample_data
        
        logger.info("Iniciando treinamento do modelo...")
        X, y = create_sample_data(2000)
        
        metrics = predictor.train(X, y)
        
        # Salvar modelo
        Path("models").mkdir(exist_ok=True)
        predictor.save_model(MODEL_PATH)
        
        return {
            "status": "success",
            "message": "Modelo treinado com sucesso",
            "metrics": {
                "roc_auc": float(metrics['roc_auc']),
                "f1_score": float(metrics['f1_score'])
            }
        }
        
    except Exception as e:
        logger.error(f"Erro no treinamento: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro ao treinar modelo: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
