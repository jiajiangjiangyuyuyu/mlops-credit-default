import pandas as pd
from sklearn.datasets import make_classification

X, y = make_classification(
    n_samples=10000, n_features=11, n_informative=8,
    n_redundant=2, n_classes=2, weights=[0.93, 0.07], random_state=42,
)
feature_names = [
    "RevolvingUtilization","age","zip_transactions","months_delay_30",
    "debt_ratio","monthly_income","open_credit_lines","real_estate_loans",
    "months_delay_90","dl_ratio","months_delay_60",
]
df = pd.DataFrame(X, columns=feature_names)
df["label"] = y
df["age"] = (df["age"].abs()*10+25).round().clip(21,80).astype(int)
df["monthly_income"] = (df["monthly_income"].abs()*5000+2000).round(2)
df["dependents"] = (df["zip_transactions"].abs()*2).round().clip(0,5).astype(int)
df["months_delay_30"] = (df["months_delay_30"].abs()*3).round().clip(0,15).astype(int)
df["months_delay_60"] = (df["months_delay_60"].abs()*2).round().clip(0,10).astype(int)
df["months_delay_90"] = (df["months_delay_90"].abs()*1).round().clip(0,8).astype(int)
df.to_csv("data/credit_data.csv", index=False)
print(f"完成: {len(df)}条, 违约率{df['label'].mean():.2%}")
