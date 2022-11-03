import pandas as pd
import xgboost as xgb
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, recall_score, f1_score
import matplotlib.pyplot as plt
import numpy as np

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

train = pd.read_csv('../Data/Vertical_federated_experiment/After_train_test_split/Traditional/f.csv', encoding='utf-8')
test = pd.read_csv('../Data/Vertical_federated_experiment/After_train_test_split/Traditional/ft.csv', encoding='utf-8')



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

names = ['Normal', 'Abnormal']

def plot_confusion_matrix(y_true, y_pred, classes, normalize=False, cmap=plt.cm.Blues):

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = cm * 100
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           ylabel='True label',
           xlabel='Predicted label')
    #ax.set_ylim(0,100)
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center",
             rotation_mode="anchor")
    plt.setp(ax.get_yticklabels(), rotation=90)
    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    #fig.tight_layout()
    return ax


np.set_printoptions(precision=2)

# Plot normalized confusion matrix
plt.rcParams['font.family'] = 'Times New Roman' 
plt.rcParams['font.size'] = '28' 
plot_confusion_matrix(true, pred, classes=names, normalize=True)

plt.show()