#!/usr/bin/env python
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('infile', type=str, help='Name of input grid file in verbose format with sample index.')
parser.add_argument('-o', '--output', type=str, help='Name of output CSV file to write.')
args = parser.parse_args()

raw_data = np.genfromtxt(args.infile, delimiter=' ', skip_header=1)

# Trim the last column
trimmed_data = []
for row in raw_data:
    trimmed_data.append(row[:-1])

trimmed_data = np.array(trimmed_data)

if args.output:
    output = args.output
else:
    output = args.infile + '.csv'

np.savetxt(args.output, trimmed_data, delimiter=',', header='Blocker_scaling_factor,Reimers_scaling_factor,Final_CO_WD_Mass')
