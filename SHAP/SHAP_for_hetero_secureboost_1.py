import pandas as pd
import xgboost as xgb
import shap
import numpy as np
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")

model = lgb.Booster(model_file = '../Data/Models_exported_from_FATE/lgb_256_whole.txt')
model.params["objective"] = "binary"

df = pd.read_csv('../Data/Vertical_federated_experiment/After_train_test_split/Traditional/at.csv', encoding='utf-8')
X = df.iloc[:, 2:]


explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)
shap.summary_plot(shap_values, X, plot_type="bar", max_display=30)


