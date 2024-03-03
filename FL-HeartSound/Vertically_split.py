import pandas as pd

df_train = pd.read_csv('../Data/Vertical_federated_experiment/After_train_test_split/Traditional/f.csv')
df_test = pd.read_csv('../Data/Vertical_federated_experiment/After_train_test_split/Traditional/ft.csv')

l = list(df_train.columns)
n = len(l)
p = int(n/2)

fea1 = l[0:p]
fea2 = l[p:]
fea2.insert(0,'id')

df_train_guest = df_train[fea1]
df_train_host = df_train[fea2]

df_test_guest = df_test[fea1]
df_test_host = df_test[fea2]

df_train_guest.to_csv('../Data/Vertical_federated_experiment/After_train_test_split/Vertically_federated/f1.csv', index=False)
df_train_host.to_csv('../Data/Vertical_federated_experiment/After_train_test_split/Vertically_federated/f2.csv', index=False)

df_test_guest.to_csv('../Data/Vertical_federated_experiment/After_train_test_split/Vertically_federated/f1t.csv', index=False)
df_test_host.to_csv('../Data/Vertical_federated_experiment/After_train_test_split/Vertically_federated/f2t.csv', index=False)