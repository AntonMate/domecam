import os
import getopt
import sys

optlist, args = getopt.getopt(sys.argv[1:], 'infile', ['infile='])
data_dir = optlist[0][1]

for item in os.listdir(data_dir):
    if item.startswith('DC'):
        print('item', os.path.isfile(f'{data_dir}/{item}/{item}_2km_result.txt'))
        print('item', os.path.isfile(f'{data_dir}/{item}/{item}_2km_info_from_logs.txt'))