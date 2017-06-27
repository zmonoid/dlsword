import pandas as pd
from functools import reduce
import glob

fnames = glob.glob('./logs/*/result*.csv')
flist = [pd.read_csv(x) for x in fnames]

result = reduce((lambda x, y: x['invasive'] + y['invasive']),
                flist) / len(flist)
flist[0]['invasive'] = result
flist[0].to_csv('ensemble.csv')
