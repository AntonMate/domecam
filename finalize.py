import os
import getopt
import sys
import pandas as pd

optlist, args = getopt.getopt(sys.argv[1:], 'infile', ['infile='])
data_dir = optlist[0][1]

if not os.path.isdir(f'{data_dir}/together'):
    os.mkdir(f'{data_dir}/together')

for item in os.listdir(data_dir):
    if item.startswith('DC'):
        df1 = pd.read_csv(f'{data_dir}/{item}/{item}_2km_result.txt')
        df2 = pd.read_csv(f'{data_dir}/{item}/{item}_2km_info_from_logs.txt')
        df_res = pd.concat([df1, df2], axis=1)
        df_res.to_csv(f'{data_dir}/together/{item}.txt', index=False)

data_dir = f'{data_dir}/together'
for item in os.listdir(data_dir):
    if item.startswith('DC'):
        df_tmp = pd.read_csv(f'{data_dir}/{item}')
        break

df = pd.DataFrame().reindex_like(df_tmp)
    
for item in os.listdir(data_dir):
    if item.startswith('DC'):
        df1 = pd.read_csv(f'{data_dir}/{item}')
        df = pd.concat([df, df1])

df = df.tail(-1)
df.to_excel(f'{data_dir}/together.xlsx', index=False)
