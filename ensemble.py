import pandas as pd
import glob

fnames = glob.glob('./logs/*/result*.csv')
print fnames
flist = [pd.read_csv(x) for x in fnames]

for item in flist[1:]:
    flist[0]['invasive'] += item['invasive']

flist[0]['invasive'] /= len(fnames)
flist[0].to_csv('ensemble.csv')
