from fastapi import FastAPI          # Web框架，用来写API接口
from pydantic import BaseModel       # 校验前端传进来的数据格式
import joblib                        # 用来读取pkl模型文件
import pandas as pd                  # 把请求数据转成模型认识的表格格式

app = FastAPI(title="信用违约预测API")  # 创建一个API应用

# 加载训练好的模型文件，启动时读一次，以后每次请求都用它
model = joblib.load("best_model.pkl")

# 定义前端要传什么数据进来，每个字段什么类型
class CreditFeatures(BaseModel):
    RevolvingUtilization: float
    age: int
    zip_transactions: float
    months_delay_30: int
    debt_ratio: float
    monthly_income: float
    open_credit_lines: float
    real_estate_loans: float
    months_delay_90: int
    dl_ratio: float
    months_delay_60: int
    dependents: int


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(req: CreditFeatures):
    df = pd.DataFrame([req.dict()])          # 把传进来的数据转成表格格式
    prob = float(model.predict_proba(df)[0][1])  # 模型预测违约概率
    return {
        "default_probability": round(prob, 4),
        "risk": "high" if prob > 0.5 else "low"
    }

