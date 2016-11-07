from __future__ import print_function

import argparse
import os
import glob

parser = argparse.ArgumentParser()

parser.add_argument("--start_line", type = int)
parser.add_argument("-x", type = int)
parser.add_argument("-y", type = int)
parser.add_argument("-z", type = int)
parser.add_argument("-d", type = str)

args = vars(parser.parse_args())

start_line = args['start_line']
x_col = args['x']
y_col = args['y']
z_col = args['z']
data_file = args['d']

x = []
y = []
z = []
x_label = ''
y_label = ''
z_label = ''
group = {}
with open(data_file, "r") as f:
	nu = 0
	for line in f.readlines():
		nu += 1
		if nu < start_line:
                    continue
                l = line.split(",")
                if nu == start_line:
                    x_label = l[x_col]
                    if y_col:
                        y_label = l[y_col]
                    else:
                        y_label = 'MRR*recall'
                    if z_col:
                        z_label = l[z_col]
                    continue
		x.append(int(l[x_col]))
                if y_col:
                    y.append(float(l[y_col]))
                else:
                    y.append(float(l[2])*float(l[3]))

                if z_col:
                    z.append(float(l[z_col]))
                else:
                    x.append(int(l[x_col]))
                    if group.has_key(x[-1]):
                        group[x[-1]].append(y[-1])
                    else:
                        group[x[-1]] = [y[-1]]

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


fig = plt.figure()
boxes = [[]]
ax = fig.add_subplot(111, projection='3d') if z_col else fig.add_subplot(111)

ax.set_xlabel(x_label)
ax.set_ylabel(y_label)
if z_col:
    ax.scatter(x,y,z)
    ax.set_zlabel(z_label)
else:
    for k in sorted(group.keys()):
        boxes.append(group[k])
    ax.boxplot(boxes)
plt.show()
