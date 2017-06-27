
#%%
import pandas as pd
d1 = pd.read_csv('./invasive_densenet161_32_0.1_True_2017-06-19 13:37:18.785533/result_39.csv')
d2 = pd.read_csv('./invasive_inception_v3_32_0.1_True_2017-06-20 19:08:56.505718/result_72.csv')
d3 = pd.read_csv('./invasive_vgg19_bn_16_0.1_True_2017-06-20 19:20:24.202969/result_28.csv')

result = (d1['invasive'] + d2['invasive'] + d3['invasive'])/3

d1['invasive'] = result

d1.to_csv('ensemble.csv')
