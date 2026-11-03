import mlflow
import mlflow.xgboost
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import joblib

# 1. 读数据
df = pd.read_csv("data/credit_data.csv")
X = df.drop("label", axis=1)
y = df["label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 2. 开始MLflow实验
mlflow.set_experiment("credit-default")

for depth in [3, 6, 10]:
    with mlflow.start_run(run_name=f"xgb-depth-{depth}"):
        params = {
            "max_depth": depth,
            "learning_rate": 0.1,
            "n_estimators": 200,
            "scale_pos_weight": 12,
            "use_label_encoder": False,
            "eval_metric": "auc",
            "verbosity": 0,
        }
        mlflow.log_params(params)

        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train)

        y_pred = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_pred)
        mlflow.log_metric("auc", auc)

        mlflow.xgboost.log_model(model, "model")
        print(f"depth={depth} -> AUC={auc:.4f}")
joblib.dump(model, "best_model.pkl")
print("模型已保存到 best_model.pkl")
