import pandas as pd

from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek

data = pd.read_csv('../Data/74_features_data/a.csv', encoding= 'utf-8')



X = data.iloc[:, 2:]
y = data['y']


'''
smo = SMOTETomek(sampling_strategy={0:150, 1:150} ,random_state=0)
X_sample, y_sample = smo.fit_resample(X, y)
y_sample = y_sample.to_frame()
new_data = pd.concat([y_sample, X_sample], axis=1)
'''

cc = RandomUnderSampler(sampling_strategy={0:7, 1:7},random_state=0)
X_sample, y_sample = cc.fit_resample(X, y)
y_sample = y_sample.to_frame()
new_data = pd.concat([y_sample, X_sample], axis=1)




new_data.index.name = 'id'

new_data.to_csv('../Data/74_features_data_sampled/a.csv', index=True)


