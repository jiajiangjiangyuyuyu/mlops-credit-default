# 端到端MLOps流水线：信用违约预测

一个从零到部署的完整MLOps Demo，覆盖实验追踪 → 模型服务 → 容器化 → K8s部署全链路。

## 技术栈
- Python / XGBoost
- MLflow Tracking（实验管理）
- FastAPI（推理服务）
- Docker（容器化）
- Kubernetes（编排部署）

## 架构
数据预处理 → XGBoost训练 → MLflow记录实验
  → FastAPI封装 /predict 接口
  → Docker打包镜像
  → K8s Deployment + Service 部署

## 快速开始
```bash
# 安装依赖
pip install -r requirements.txt

# 训练
python train.py

# 启动API
uvicorn predict:app --reload --port 8000

# Docker构建
docker build -t credit-default-api:v1 .

# K8s部署
kubectl apply -f k8s/

项目结构

├── train.py          # 训练脚本，接MLflow
├── predict.py        # FastAPI推理服务
├── generate_data.py  # 生成模拟数据集
├── Dockerfile
├── requirements.txt
├── k8s/
│   ├── deployment.yaml
│   └── service.yaml
└── data/

