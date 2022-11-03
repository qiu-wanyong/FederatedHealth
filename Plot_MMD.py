import torch
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import itertools

class MMDLoss(nn.Module):

    #   计算源域数据和目标域数据的MMD距离
    #   Params:
    #   source: 源域数据（n * len(x))
    #   target: 目标域数据（m * len(y))
    #   kernel_mul:
    #   kernel_num: 取不同高斯核的数量
    #   fix_sigma: 不同高斯核的sigma值
    #   Return:
    #   loss: MMD loss

    def __init__(self, kernel_type='rbf', kernel_mul=2.0, kernel_num=5, fix_sigma=None, **kwargs):
        super(MMDLoss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        self.kernel_type = kernel_type

    def guassian_kernel(self, source, target, kernel_mul, kernel_num, fix_sigma):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0 - total1) ** 2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul ** i)
                          for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                      for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def linear_mmd2(self, f_of_X, f_of_Y):
        loss = 0.0
        delta = f_of_X.float().mean(0) - f_of_Y.float().mean(0)
        loss = delta.dot(delta.T)
        return loss

    def forward(self, source, target):
        if self.kernel_type == 'linear':
            return self.linear_mmd2(source, target)
        elif self.kernel_type == 'rbf':
            batch_size = int(source.size()[0])
            kernels = self.guassian_kernel(
                source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
            XX = torch.mean(kernels[:batch_size, :batch_size])
            YY = torch.mean(kernels[batch_size:, batch_size:])
            XY = torch.mean(kernels[:batch_size, batch_size:])
            YX = torch.mean(kernels[batch_size:, :batch_size])
            loss = torch.mean(XX + YY - XY - YX)
            return loss


def transpose(matrix):
    new_matrix = []
    for i in range(len(matrix[0])):
        matrix1 = []
        for j in range(len(matrix)):
            matrix1.append(matrix[j][i])
        new_matrix.append(matrix1)
    return new_matrix


df1 = pd.read_csv('../Data/74_features_data_sampled/a.csv', encoding= 'utf-8')
df2 = pd.read_csv('../Data/74_features_data_sampled/b.csv', encoding= 'utf-8')
df3 = pd.read_csv('../Data/74_features_data_sampled/c.csv', encoding= 'utf-8')
df4 = pd.read_csv('../Data/74_features_data_sampled/d.csv', encoding= 'utf-8')
df5 = pd.read_csv('../Data/74_features_data_sampled/e.csv', encoding= 'utf-8')
df6 = pd.read_csv('../Data/74_features_data_sampled/f.csv', encoding= 'utf-8')

dff = df1.iloc[:, 2:]
col = list(dff.columns)

a, b, c, d, e, f = [], [], [], [], [], []
for i in col:
    temp1 = df1[i].tolist()
    temp2 = df2[i].tolist()
    temp3 = df3[i].tolist()
    temp4 = df4[i].tolist()
    temp5 = df5[i].tolist()
    temp6 = df6[i].tolist()
    a.append(temp1)
    b.append(temp2)
    c.append(temp3)
    d.append(temp4)
    e.append(temp5)
    f.append(temp6)

aa = transpose(a)
bb = transpose(b)
cc = transpose(c)
dd = transpose(d)
ee = transpose(e)
ff = transpose(f)

aaa = torch.Tensor(aa)
bbb = torch.Tensor(bb)
ccc = torch.Tensor(cc)
ddd = torch.Tensor(dd)
eee = torch.Tensor(ee)
fff = torch.Tensor(ff)
aaa = Variable(aaa)
bbb = Variable(bbb)
ccc = Variable(ccc)
ddd = Variable(ddd)
eee = Variable(eee)
fff = Variable(fff)

index = [0, 1, 2, 3, 4, 5]
whole = []
whole.append(aaa)
whole.append(bbb)
whole.append(ccc)
whole.append(ddd)
whole.append(eee)
whole.append(fff)
dic = dict(zip(index, whole))

matrix = [[-1, -1, -1, -1, -1, -1] for _ in range(6)]
for i in range(6):
    for j in range(6):
        if(i == j):
            matrix[j][j] = 0


combine = list(itertools.combinations(index, 2)) 
for j in combine:
    MMD = MMDLoss()
    mmd_value = MMD(source=dic[j[0]], target=dic[j[1]]).numpy().tolist()
    matrix[j[0]][j[1]] = mmd_value
    matrix[j[1]][j[0]] = mmd_value

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = '24'


ins = ['D$_a$(MIT)', 'D$_b$(AAD)', 'D$_c$(AUTH)', 'D$_d$(UHA)', 'D$_e$(DLUT)', 'D$_f$(SUA)']
matrix2 = pd.DataFrame(matrix, index=ins, columns=ins)


new_color = sns.color_palette("RdBu_r", 1000)[200:1000]
ax = sns.heatmap(matrix2, annot=True, center=3,
                 cmap=new_color, annot_kws={"fontsize": 24}, fmt='.3f')
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=24)
plt.xticks(fontsize=21, rotation=0)
plt.yticks(fontsize=21, rotation=0)
plt.show()

