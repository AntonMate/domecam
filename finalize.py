import os
import getopt
import sys

optlist, args = getopt.getopt(sys.argv[1:], 'infile', ['infile='])
new_path = optlist[0][1]

data_dir = new_path
print(data_dir)