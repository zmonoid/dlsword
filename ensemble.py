import pandas as pd
import glob
import argparse

parser = argparse.ArgumentParser(description='ensemble')
parser.add_argument('logs', type=str, help='logs folder prefix')
args = parser.parse_args()

fnames = glob.glob(args.logs + '*/result_*.csv')
fnames = sorted(fnames)
for name in fnames:
    print name

if len(fnames) == 0:
    print "found no csv"
else:
    flist = [pd.read_csv(x) for x in fnames]
    for item in flist[1:]:
        flist[0]['invasive'] += item['invasive']

    flist[0]['invasive'] /= len(fnames)
    flist[0].to_csv('logs/ensemble.csv', index=False)
