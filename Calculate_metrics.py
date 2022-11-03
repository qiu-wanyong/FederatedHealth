import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, recall_score, f1_score

threshold = 0.5


df= pd.read_csv('../Data/Predicted_data_of_homo_secureboost/data_3_30.csv', encoding= 'utf-8')


true = df.iloc[:,0].tolist()
pred_prob = df.iloc[:,2].tolist()
pred = [int(i >= threshold) for i in pred_prob]


print(true)
print(pred)

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
#print("纵向secureboost，医院c，树的个数30，树的深度为3，测试集（调过之后的validation）的结果如下：")
print("敏感度 = ", recall1)
print("特异性 = ", recall0)
print("UAR = ", UAR)
print("UF1 = ", UF1)
print("准确率 = ", Acc)


