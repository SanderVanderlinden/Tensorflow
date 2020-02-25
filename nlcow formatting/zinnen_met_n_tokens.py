import sys
import os

dirpath = os.getcwd()
input_file = dirpath + '/' + sys.argv[1]

with open(input_file, 'r', errors='ignore') as input_file:
    for line in input_file:
        output_file_n = (dirpath + "/zinnen_met_n_tokens/output_file_{0}.txt".format(len(line.strip().split(" ")))).replace('\\', '/')
        with open(output_file_n, 'a', errors='ignore') as output_file:
            output_file.write(line.strip() + '\n')