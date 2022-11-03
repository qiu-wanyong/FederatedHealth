import pandas as pd
import xgboost as xgb
import shap
import numpy as np
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")


model1 = lgb.Booster(model_file = '../Data/Models_exported_from_FATE/lgb_256_half_features.txt')
model1.params["objective"] = "binary"

model2 = lgb.Booster(model_file = '../Data/Models_exported_from_FATE/lgb_16_half_features.txt')
model2.params["objective"] = "binary"

test_1 = pd.read_csv('../Data/Vertical_federated_experiment/After_train_test_split/Vertically_federated/a1t.csv', encoding='utf-8')
test_2 = pd.read_csv('../Data/Vertical_federated_experiment/After_train_test_split/Vertically_federated/a2t.csv', encoding='utf-8')
test = pd.read_csv('../Data/Vertical_federated_experiment/After_train_test_split/Traditional/at.csv', encoding='utf-8')
X_test_1 = test_1.iloc[:,2:]
X_test_2 = test_2.iloc[:,1:]
X_test = test.iloc[:,2:]

explainer1 = shap.TreeExplainer(model1)
shap_values1 = explainer1.shap_values(X_test_1)
shap_values1 = np.array(shap_values1)



explainer2 = shap.TreeExplainer(model2)
shap_values2 = explainer2.shap_values(X_test_2)
shap_values2 = np.array(shap_values2)


shap_values = np.concatenate((shap_values1, shap_values2), axis=2)

s = list(shap_values)
shap.summary_plot(s, X_test, plot_type="bar", max_display=30)
