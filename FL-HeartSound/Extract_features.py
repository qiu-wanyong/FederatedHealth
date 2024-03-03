import numpy as np
import pandas as pd
import xgboost as xgb
import shap


params={        'max_depth':3,
                'n_estimators':40,
                'learning_rate':0.3,
                'nthread':4,
                'subsample':1.0,
                'colsample_bytree':1,
                'min_child_weight' : 3,
                # 'eval_metric' : ['logloss'],
                'seed':1301}

train = pd.read_csv('../Data/Raw_data/f.csv', encoding= 'utf-8') # Can be a,b,c,d,e,f
val = pd.read_csv('../Data/Raw_data/validate.csv', encoding= 'utf-8')

train_y, val_y = train['y'], val['y']
train_X, val_X = train.iloc[:,2:], val.iloc[:,2:]

xgtrain = xgb.DMatrix(train_X, label=train_y)
xgval = xgb.DMatrix(val_X, label=val_y)

model = xgb.train(params,
          dtrain=xgtrain,
          verbose_eval=True,
          evals=[(xgtrain, "train"), (xgval, "valid")],
          early_stopping_rounds=10,
          num_boost_round = 50
          )


explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(train_X)

global_importances = np.abs(shap_values).mean(0)
print(global_importances.shape)


inds = np.argsort(-global_importances)

index=[i for i in range(6373)]
df = pd.DataFrame({'Feature name':index, 'Shap value':global_importances})
df.sort_values(by="Shap value" , inplace=True, ascending=False)
print(df)
df.to_csv('../Data/SHAP_results/sort_shap_f.csv', index=False) # Can be a,b,c,d,e,f


