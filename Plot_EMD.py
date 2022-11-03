from scipy.stats import wasserstein_distance
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

df1 = pd.read_csv('../Data/74_features_data_sampled/a.csv', encoding= 'utf-8').to_numpy()
df2 = pd.read_csv('../Data/74_features_data_sampled/b.csv', encoding= 'utf-8').to_numpy()
df3 = pd.read_csv('../Data/74_features_data_sampled/c.csv', encoding= 'utf-8').to_numpy()
df4 = pd.read_csv('../Data/74_features_data_sampled/d.csv', encoding= 'utf-8').to_numpy()
df5 = pd.read_csv('../Data/74_features_data_sampled/e.csv', encoding= 'utf-8').to_numpy()
df6 = pd.read_csv('../Data/74_features_data_sampled/f.csv', encoding= 'utf-8').to_numpy()

df1 = df1.flatten().tolist()
df2 = df2.flatten().tolist()
df3 = df3.flatten().tolist()
df4 = df4.flatten().tolist()
df5 = df5.flatten().tolist()
df6 = df6.flatten().tolist()

'''
print('EMD Results:')
print('a-b',wasserstein_distance(df1, df2))
print('a-c',wasserstein_distance(df1, df3))
print('a-d',wasserstein_distance(df1, df4))
print('a-e',wasserstein_distance(df1, df5))
print('a-f',wasserstein_distance(df1, df6))

print('b-c',wasserstein_distance(df2, df3))
print('b-d',wasserstein_distance(df2, df4))
print('b-e',wasserstein_distance(df2, df5))
print('b-f',wasserstein_distance(df2, df6))

print('c-d',wasserstein_distance(df3, df4))
print('c-e',wasserstein_distance(df3, df5))
print('c-f',wasserstein_distance(df3, df6))

print('d-e',wasserstein_distance(df4, df5))
print('d-f',wasserstein_distance(df4, df6))

print('e-f',wasserstein_distance(df5, df6))
'''

matrix = [[-1, -1, -1, -1, -1, -1] for _ in range(6)]
for i in range(6):
    for j in range(6):
        if(i == j):
            matrix[j][j] = 0

matrix[0][1] = wasserstein_distance(df1, df2)
matrix[1][0] = wasserstein_distance(df1, df2)

matrix[0][2] = wasserstein_distance(df1, df3)
matrix[2][0] = wasserstein_distance(df1, df3)

matrix[0][3] = wasserstein_distance(df1, df4)
matrix[3][0] = wasserstein_distance(df1, df4)

matrix[0][4] = wasserstein_distance(df1, df5)
matrix[4][0] = wasserstein_distance(df1, df5)

matrix[0][5] = wasserstein_distance(df1, df6)
matrix[5][0] = wasserstein_distance(df1, df6)

matrix[1][2] = wasserstein_distance(df2, df3)
matrix[2][1] = wasserstein_distance(df2, df3)

matrix[1][3] = wasserstein_distance(df2, df4)
matrix[3][1] = wasserstein_distance(df2, df4)

matrix[1][4] = wasserstein_distance(df2, df5)
matrix[4][1] = wasserstein_distance(df2, df5)

matrix[1][5] = wasserstein_distance(df2, df6)
matrix[5][1] = wasserstein_distance(df2, df6)

matrix[2][3] = wasserstein_distance(df3, df4)
matrix[3][2] = wasserstein_distance(df3, df4)

matrix[2][4] = wasserstein_distance(df3, df5)
matrix[4][2] = wasserstein_distance(df3, df5)

matrix[2][5] = wasserstein_distance(df3, df6)
matrix[5][2] = wasserstein_distance(df3, df6)

matrix[3][4] = wasserstein_distance(df4, df5)
matrix[4][3] = wasserstein_distance(df5, df4)

matrix[3][5] = wasserstein_distance(df4, df6)
matrix[5][3] = wasserstein_distance(df6, df4)

matrix[4][5] = wasserstein_distance(df5, df6)
matrix[5][4] = wasserstein_distance(df5, df6)

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = '28'


ins = ['D$_a$(MIT)', 'D$_b$(AAD)', 'D$_c$(AUTH)', 'D$_d$(UHA)', 'D$_e$(DLUT)', 'D$_f$(SUA)']
matrix2 = pd.DataFrame(matrix, index=ins, columns=ins)


new_color = sns.color_palette("RdBu_r", 100000)[0:90000]
ax = sns.heatmap(matrix2, annot=True, center=3,
                 cmap=new_color, annot_kws={"fontsize": 28}, fmt='.2f')
ticks = np.array([0,100,200,230])
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=24)
plt.colorbar(ticks=ticks)
plt.xticks(fontsize=21, rotation=0)
plt.yticks(fontsize=21, rotation=0)
plt.show()