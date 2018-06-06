#!/usr/bin/env python
"""
This script plots the grid status as colored dots in the parameter space.

Donald E. Willcox
"""
import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("gridstatus", type=str, help="Supply the grid status file from which to get the success/failure status.")
parser.add_argument('-o', '--outname', type=str, default='grid_status.png', help="Base name of the output plot file. Default is grid_status.png")
args = parser.parse_args()

f = open(args.gridstatus, 'r')
cols = f.readline().strip().split()
data = {}
for c in cols:
    data[c] = []
for l in f:
    ls = l.strip().split()
    for i, (xi, ci) in enumerate(zip(ls, cols)):
        if ci != 'status' and ci != 'index':
            x = float(xi)
        elif ci == 'status':
            x = (xi == 'success')
        else:
            x = int(xi)
        data[ci].append(x)
f.close()

numpts = len(data['index'])

area = [100 for i in range(numpts)]
colors = []
for i in range(numpts):
    if data['status'][i]:
        colors.append('blue')
    else:
        colors.append('red')

plt.scatter(data['Blocker_scaling_factor'], data['Reimers_scaling_factor'], s=area, c=colors, alpha=0.5)
plt.xlabel('Blocker')
plt.ylabel('Reimers')
plt.tight_layout()
plt.savefig(args.outname, dpi=300)
