import numpy as np
import shap
import pandas as pd
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")
print(lgb.__version__)

train = pd.read_csv('../Data/Vertical_federated_experiment/After_train_test_split/Traditional/a.csv', encoding='utf-8')
test = pd.read_csv('../Data/Vertical_federated_experiment/After_train_test_split/Traditional/at.csv', encoding='utf-8')
X_train, X_test = train.iloc[:, 2:], test.iloc[:, 2:]
y_train, y_test = train['y'], test['y']

lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_train, y_train, reference=lgb_train)


params = {
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

model = lgb.train(params, lgb_train, num_boost_round=30)

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test, plot_type="bar", max_display=30)