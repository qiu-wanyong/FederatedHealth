import pandas as pd
import xgboost as xgb
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, recall_score, f1_score


threshold = 0.5

params={'tree_method':'approx',
        'max_depth':3,   
        'learning_rate':0.3, 
        #'nthread':4,
        'reg_lambda':0.1,        
        'reg_alpha':0,         
        'subsample':1,  
         #'colsample_bytree':1, 
        'min_child_weight' : 0, 
        'eval_metric' : ['logloss'],
        'max_bin': 32,
        #'grow_policy':'lossguide',
        'seed':1}

test = pd.read_csv('../Data/74_features_data_sampled/a.csv', encoding='utf-8')
b = pd.read_csv('../Data/74_features_data_sampled/b.csv', encoding='utf-8')
c = pd.read_csv('../Data/74_features_data_sampled/c.csv', encoding='utf-8')
d = pd.read_csv('../Data/74_features_data_sampled/d.csv', encoding='utf-8')
e = pd.read_csv('../Data/74_features_data_sampled/e.csv', encoding='utf-8')
f = pd.read_csv('../Data/74_features_data_sampled/f.csv', encoding='utf-8')

train = pd.concat([b,c,d,e,f])

train_y= train['y']
train_X = train.iloc[:,2:]
xgtrain = xgb.DMatrix(train_X, label=train_y)

test_y= test['y']
test_X = test.iloc[:,2:]
xgtest = xgb.DMatrix(test_X)


model = xgb.train(params, dtrain=xgtrain, verbose_eval=True,
                  evals=[(xgtrain, "train"), (xgtrain, "valid")], num_boost_round = 30
          )


y_pred_prob = model.predict(xgtest).tolist()
pred = [int(i >= threshold) for i in y_pred_prob]
true = list(test['y'])

cm = confusion_matrix(true, pred)
recall0 = (cm[0][0]) / (cm[0][0] + cm[0][1])
recall1 = (cm[1][1]) / (cm[1][0] + cm[1][1])
precision0 = (cm[0][0]) / (cm[0][0] + cm[1][0])
precision1 = (cm[1][1]) / (cm[0][1] + cm[1][1])
f1_0 = 2 * recall0 * precision0 / (recall0 + precision0)
f1_1 = 2 * recall1 * precision1 / (recall1 + precision1)


# UAR = (recall0 + recall1) / 2
# UF1 = (f1_0 + f1_1) / 2

UAR = recall_score(true, pred, average='macro')
UF1 = f1_score(true, pred, average='macro')
Acc = accuracy_score(true, pred)


print()
#print("74个特征，医院e下采样，其他医院上采样，纵向secureboost，树的个数30，树的深度6，训练集是医院e，测试集(医院a,100个)结果如下: ")
print("传统xgboost，医院bcdef作训练集，参数与横向secureboost保持一致，测试集（医院a）的结果如下：")
print("敏感度 = ", recall1)
print("特异性 = ", recall0)
print("UAR = ", UAR)
print("UF1 = ", UF1)
print("准确率 = ", Acc)

