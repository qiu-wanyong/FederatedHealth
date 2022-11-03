import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


a = pd.read_csv('../Data/SHAP_results/sort_shap_a.csv', encoding= 'utf-8')
b = pd.read_csv('../Data/SHAP_results/sort_shap_b.csv', encoding= 'utf-8')
c = pd.read_csv('../Data/SHAP_results/sort_shap_c.csv', encoding= 'utf-8')
d = pd.read_csv('../Data/SHAP_results/sort_shap_d.csv', encoding= 'utf-8')
e = pd.read_csv('../Data/SHAP_results/sort_shap_e.csv', encoding= 'utf-8')
f = pd.read_csv('../Data/SHAP_results/sort_shap_f.csv', encoding= 'utf-8')



whole1 = pd.concat([a,b,c,d,e,f], axis=0)

whole2 = pd.DataFrame(whole1.groupby('Feature name')['Shap value'].mean())

whole2.sort_values(by="Shap value", inplace=True, ascending=False)
print(whole2)
whole2.to_csv('../Data/SHAP_results/Combined_Shap_sort.csv', index=True)
