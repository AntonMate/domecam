import os
import getopt
import sys

optlist, args = getopt.getopt(sys.argv[1:], 'infile', ['infile='])
new_path = optlist[0][1]

indexes = [i for i in range(len(new_path)) if new_path[i] == "/"]
data_dir = new_path[:indexes[-1]]
data_dir = data_dir + '/results'
print(data_dir)