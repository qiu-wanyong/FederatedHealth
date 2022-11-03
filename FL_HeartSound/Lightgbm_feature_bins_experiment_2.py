import numpy as np
import shap
import pandas as pd
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")
print(lgb.__version__)

train_1 = pd.read_csv('../Data/Vertical_federated_experiment/After_train_test_split/Vertically_federated/a1.csv', encoding='utf-8')
train_2 = pd.read_csv('../Data/Vertical_federated_experiment/After_train_test_split/Vertically_federated/a2.csv', encoding='utf-8')
test_1 = pd.read_csv('../Data/Vertical_federated_experiment/After_train_test_split/Vertically_federated/a1t.csv', encoding='utf-8')
test_2 = pd.read_csv('../Data/Vertical_federated_experiment/After_train_test_split/Vertically_federated/a2t.csv', encoding='utf-8')
test = pd.read_csv('../Data/Vertical_federated_experiment/After_train_test_split/Traditional/at.csv', encoding='utf-8')

X_train_1 = train_1.iloc[:,2:]
X_train_2 = train_2.iloc[:,1:]
X_test_1 = test_1.iloc[:,2:]
X_test_2 = test_2.iloc[:,1:]
X_test = test.iloc[:,2:]


y_train, y_test = train_1['y'], test_1['y']


lgb_train1 = lgb.Dataset(X_train_1, y_train)
lgb_train2 = lgb.Dataset(X_train_2, y_train)

params1 = {
    'task': 'train',
    'boosting_type': 'gbdt',  
    'objective': 'binary',  
    'metric': {'binary_logloss'},  
    'learning_rate': 0.3,  
    'max_depth': 3,
    'lambda_l1':0,
    'lambda_l2':0.1,
    'max_bin':256
}

params2 = {
    'task': 'train',
    'boosting_type': 'gbdt',  
    'objective': 'binary', 
    'metric': {'binary_logloss'}, 
    'learning_rate': 0.3,  
    'max_depth': 3,
    'lambda_l1':0,
    'lambda_l2':0.1,
    'max_bin':16
}

model1 = lgb.train(params1, lgb_train1, num_boost_round=30)
explainer1 = shap.TreeExplainer(model1)
shap_values1 = explainer1.shap_values(X_test_1)

model2 = lgb.train(params2, lgb_train2, num_boost_round=30)
explainer2 = shap.TreeExplainer(model2)
shap_values2 = explainer2.shap_values(X_test_2)

shap_values = np.concatenate((shap_values1, shap_values2), axis=2)
s = list(shap_values)
shap.summary_plot(s, X_test, plot_type="bar", max_display=30)