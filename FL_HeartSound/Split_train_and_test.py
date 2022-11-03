from sklearn.model_selection import train_test_split
import pandas as pd
df = pd.read_csv('../Data/Vertical_federated_experiment/Before_train_test_split/f.csv', encoding= 'utf-8')

X = df.iloc[:,2:]
y = df['y']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=69)

df1 = pd.concat([y_train, X_train], axis=1)
df1.index.name = 'id'
df2 = pd.concat([y_test, X_test], axis=1)
df2.index.name = 'id'

df1.to_csv('../Data/Vertical_federated_experiment/After_train_test_split/Traditional/f.csv', index=True)
df2.to_csv('../Data/Vertical_federated_experiment/After_train_test_split/Traditional/ft.csv', index=True)