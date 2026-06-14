import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path

model = joblib.load('models/xgboost_model.pkl')
X_tr = pd.read_csv('data/processed/X_train.csv', dtype='float32', nrows=5000)

explainer = shap.TreeExplainer(model)
sv = explainer.shap_values(X_tr)

if isinstance(sv, list):
    gi = np.mean([np.abs(s) for s in sv], axis=0).mean(axis=0)
elif hasattr(sv, 'ndim') and sv.ndim == 3:
    gi = np.abs(sv).mean(axis=(0, 2))
else:
    gi = np.abs(sv).mean(axis=0)

fi = pd.Series(gi, index=X_tr.columns).sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x=fi.values[:15], y=fi.index[:15], ax=ax, palette='viridis')
ax.set_title('ThreatSense - Top 15 Features by Mean |SHAP|')
fig.tight_layout()

Path('docs').mkdir(exist_ok=True)
fig.savefig('docs/shap_summary.png', dpi=150)
plt.close()

joblib.dump(explainer, 'models/shap_explainer.pkl')
print('SHAP done — shap_explainer.pkl and shap_summary.png saved')