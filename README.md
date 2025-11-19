🎯 Customer Churn Prediction System
Python
FastAPI
XGBoost
Docker
License

Sistema completo de Machine Learning para predição de churn de clientes utilizando XGBoost, FastAPI e Docker. Desenvolvido com foco em produção e escalabilidade.

📊 Sobre o Projeto
Este projeto implementa um sistema end-to-end de predição de churn com:

Modelo XGBoost otimizado com 85%+ de acurácia

API REST com FastAPI para predições em tempo real

Containerização completa com Docker e Docker Compose

Balanceamento de classes usando SMOTE

Monitoramento com MLflow

Testes automatizados e CI/CD ready

🎯 Objetivos
Identificar clientes com alta probabilidade de cancelamento

Fornecer recomendações personalizadas de retenção

API escalável para integração com sistemas existentes

Infraestrutura reproduzível e fácil de deployar

🏗️ Arquitetura
text
┌─────────────────┐
│   Cliente       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  FastAPI        │ ◄─── Container Docker
│  (Port 8000)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  XGBoost Model  │
│  + Preprocessor │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  MLflow         │ ◄─── Container Docker
│  (Port 5000)    │
└─────────────────┘
🚀 Quick Start
Pré-requisitos
Docker & Docker Compose

Python 3.11+ (para desenvolvimento local)

Git

Instalação com Docker (Recomendado)
bash
# Clone o repositório
git clone https://github.com/seu-usuario/customer-churn-prediction.git
cd customer-churn-prediction

# Inicie os containers
docker-compose up -d

# Treine o modelo (primeira execução)
curl -X POST http://localhost:8000/model/train

# Acesse a documentação da API
open http://localhost:8000/docs
Instalação Local
bash
# Clone o repositório
git clone https://github.com/RafaelrorizAGE/customer-churn-prediction.git
cd customer-churn-prediction

# Crie ambiente virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows

# Instale dependências
pip install -r requirements.txt

# Execute a API
uvicorn src.api:app --reload
📖 Uso da API
Predição Individual
bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
Resposta:

json
{
  "customer_id": 0,
  "churn_probability": 0.6234,
  "will_churn": true,
  "risk_level": "Alto",
  "recommendations": [
    "Oferecer contrato de longo prazo com desconto",
    "Incluir suporte técnico gratuito por 3 meses"
  ]
}
Predição em Lote
bash
curl -X POST "http://localhost:8000/predict/batch" \
  -H "Content-Type: application/json" \
  -d '{
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
      }
    ]
  }'
Exemplos em Python
python
import requests

# Configuração
API_URL = "http://localhost:8000"

# Dados do cliente
customer = {
    "tenure": 24,
    "monthly_charges": 65.0,
    "total_charges": 1560.0,
    "contract_type": 1,
    "payment_method": 0,
    "internet_service": 0,
    "online_security": 1,
    "tech_support": 1,
    "streaming_tv": 0,
    "streaming_movies": 0
}

# Fazer predição
response = requests.post(f"{API_URL}/predict", json=customer)
result = response.json()

print(f"Probabilidade de Churn: {result['churn_probability']:.2%}")
print(f"Risco: {result['risk_level']}")
print(f"Recomendações:")
for rec in result['recommendations']:
    print(f"  - {rec}")
📁 Estrutura do Projeto
text
customer-churn-prediction/
├── src/
│   ├── __init__.py
│   ├── model.py              # Classe do modelo XGBoost
│   ├── api.py                # API FastAPI
│   ├── preprocessing.py      # Preprocessamento de dados
│   └── train.py             # Script de treinamento
├── notebooks/
│   ├── 01_exploratory_analysis.ipynb
│   └── 02_model_training.ipynb
├── tests/
│   ├── __init__.py
│   └── test_api.py
├── models/                   # Modelos treinados
├── data/
│   ├── raw/                 # Dados brutos
│   └── processed/           # Dados processados
├── config/
│   └── model_config.yaml
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .gitignore
├── .dockerignore
└── README.md
🔧 Features Implementadas
Modelo de Machine Learning
✅ XGBoost Classifier otimizado

✅ Balanceamento de classes com SMOTE

✅ Cross-validation estratificado

✅ Feature importance analysis

✅ Hyperparameter tuning

✅ Métricas: ROC-AUC, F1-Score, Precision, Recall

API REST
✅ Predição individual e em lote

✅ Health check endpoint

✅ Documentação automática (Swagger/ReDoc)

✅ Validação de dados com Pydantic

✅ CORS habilitado

✅ Logging estruturado

✅ Tratamento de erros robusto

Infraestrutura
✅ Containerização com Docker

✅ Orquestração com Docker Compose

✅ MLflow para tracking de experimentos

✅ Volumes persistentes

✅ Variáveis de ambiente configuráveis

✅ Health checks automáticos

📊 Performance do Modelo
Métrica	Valor
ROC-AUC	0.87
F1-Score	0.84
Precision	0.86
Recall	0.82
Accuracy	0.85
Features Mais Importantes
Tenure (Tempo como cliente) - 28.5%

Monthly Charges (Cobrança mensal) - 22.3%

Contract Type (Tipo de contrato) - 18.7%

Total Charges (Total cobrado) - 14.2%

Tech Support (Suporte técnico) - 9.1%

🧪 Testes
bash
# Executar todos os testes
pytest tests/

# Com coverage
pytest tests/ --cov=src --cov-report=html

# Testes específicos
pytest tests/test_api.py -v
📦 Deploy
Docker Hub
bash
# Build da imagem
docker build -t seu-usuario/churn-prediction:latest .

# Push para Docker Hub
docker push seu-usuario/churn-prediction:latest

# Pull e execução
docker pull seu-usuario/churn-prediction:latest
docker run -p 8000:8000 seu-usuario/churn-prediction:latest
Cloud Platforms
AWS ECS/Fargate:

bash
# Configure AWS CLI e ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com

# Tag e push
docker tag churn-prediction:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/churn-prediction:latest
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/churn-prediction:latest
Google Cloud Run:

bash
gcloud builds submit --tag gcr.io/PROJECT-ID/churn-prediction
gcloud run deploy churn-api --image gcr.io/PROJECT-ID/churn-prediction --platform managed
🔐 Variáveis de Ambiente
Crie um arquivo .env:

text
# Modelo
MODEL_PATH=/app/models/xgboost_model.pkl
PREPROCESSING_PATH=/app/models/preprocessor.pkl

# API
LOG_LEVEL=INFO
API_HOST=0.0.0.0
API_PORT=8000

# MLflow
MLFLOW_TRACKING_URI=http://mlflow:5000
🤝 Contribuindo
Contribuições são bem-vindas! Para contribuir:

Fork o projeto

Crie uma branch para sua feature (git checkout -b feature/AmazingFeature)

Commit suas mudanças (git commit -m 'Add some AmazingFeature')

Push para a branch (git push origin feature/AmazingFeature)

Abra um Pull Request

📝 Roadmap
 Integração com banco de dados (PostgreSQL)

 Autenticação e autorização (JWT)

 Dashboard de monitoramento em tempo real

 A/B testing framework

 Pipeline CI/CD completo (GitHub Actions)

 Retreinamento automático

 Drift detection

 Explainability com SHAP

📚 Recursos e Referências
XGBoost Documentation

FastAPI Documentation

MLflow Documentation

Docker Documentation

Customer Churn Research Paper

📄 Licença
Este projeto está sob a licença Public GNU. Veja o arquivo LICENSE para mais detalhes.

👤 Autor
Rafael Roriz

GitHub: @RafaelrorizAGE

LinkedIn: Rafael Roriz de Menezes

Email: rroriz111@gmail.com

🙏 Agradecimentos
Comunidade Kaggle pelos datasets

Equipe FastAPI pelo framework incrível

Colaboradores do XGBoost

Comunidade open-source
