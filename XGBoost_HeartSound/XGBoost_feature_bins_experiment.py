import pandas as pd
import xgboost as xgb
import shap
import numpy as np
import warnings
warnings.filterwarnings("ignore")

print(xgb.__version__)


# ----------------------------------------------All features with 256bins------------------------------------------

train = pd.read_csv('../Data/Vertical_federated_experiment/After_train_test_split/Traditional/a.csv', encoding='utf-8')
test = pd.read_csv('../Data/Vertical_federated_experiment/After_train_test_split/Traditional/at.csv', encoding='utf-8')
X_train, X_test = train.iloc[:, 2:], test.iloc[:, 2:]
y_train, y_test = train['y'], test['y']

model = xgb.XGBClassifier(tree_method='approx', max_depth=3, n_estimators=30, learning_rate=0.3,reg_lambda=0.1,
                          reg_alpha=0, subsample=1.0, min_child_weight=0, max_bin=256, seed=1)

eval_set = [(X_train, y_train), (X_train, y_train)]
XGB = model.fit(X_train, y_train, eval_set=eval_set, verbose=True)

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

shap.summary_plot(shap_values, X_test, max_display=30)
shap.summary_plot(shap_values, X_test, plot_type="bar", max_display=30)

global_importances = np.abs(shap_values).mean(0)



index=[i for i in range(74)]
df = pd.DataFrame({'Feature name':index, 'Shap value':global_importances})
df.sort_values(by="Shap value" , inplace=True, ascending=False)
df.to_csv('256bins_a.csv', index=False)





# ------------------------------------------Half of the features with 256bins, the other half with 16bins---------------------------------------------

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




model1 = xgb.XGBClassifier(tree_method='approx', max_depth=3, n_estimators=30, learning_rate=0.3,reg_lambda=0.1,
                          reg_alpha=0, subsample=1.0, min_child_weight=0, max_bin=256, seed=1)

eval_set = [(X_train_1, y_train), (X_train_1, y_train)]
model1.fit(X_train_1, y_train, eval_set=eval_set,verbose=False)

explainer1 = shap.TreeExplainer(model1)
shap_values1 = explainer1.shap_values(X_test_1)
print(shap_values1.shape)
# shap.summary_plot(shap_values, X, plot_type="bar", max_display=10)
global_importances1 = np.abs(shap_values1).mean(0)
# print(global_importances1.shape) # (feature_num,)


model2 = xgb.XGBClassifier(tree_method='approx', max_depth=3, n_estimators=30, learning_rate=0.3,reg_lambda=0.1,
                          reg_alpha=0, subsample=1.0, min_child_weight=0, max_bin=16, seed=1)

eval_set = [(X_train_2, y_train), (X_train_2, y_train)]
model2.fit(X_train_2, y_train, eval_set=eval_set, verbose=False)

explainer2 = shap.TreeExplainer(model2)
shap_values2 = explainer2.shap_values(X_test_2)
print(shap_values2.shape)


global_importances2 = np.abs(shap_values2).mean(0)

# print(global_importances2.shape) # (feature_num,)

g1 = list(global_importances1)
g2 = list(global_importances2)
g1.extend(g2)
g = np.array(g1)

shap_values = np.hstack((shap_values1, shap_values2))
print(shap_values.shape)

shap.summary_plot(shap_values, X_test, max_display=30)

# shap.summary_plot(shap_values, X_test, plot_type="bar", max_display=30)


index=[i for i in range(74)]
df = pd.DataFrame({'Feature name':index, 'Shap value':g})
df.sort_values(by="Shap value" , inplace=True, ascending=False)
df.to_csv('256_16bins_a.csv', index=False