import pandas as pd


df = pd.read_csv('../Data/SHAP_results/Combined_Shap_sort.csv', index_col=False)
df1 = df.iloc[:74,:]
print(df1)




fea = list(df1['Feature name'])
fea1 = [i+2 for i in fea]
fea1.append(0)
fea1.append(1)
fea1.sort()

new = pd.read_csv('../Data/Raw_data/a.csv', encoding= 'utf-8')
new1 = new.iloc[:,fea1]
new1.to_csv('../Data/74_features_data/a.csv', index=False)