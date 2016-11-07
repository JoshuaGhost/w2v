file = []
time = []
with open('lsinfo.txt') as f:
	for line in f.readlines():
		file.append(line.split()[8])
		time.append(float(line.split()[6])*24*60+float(line.split()[7].split(':')[0])*60+float(line.split()[7].split(':')[1]))

time = map(lambda x: x[0]-x[1], zip(time,time[1:]+[0]))
pairs = zip(file, time)
info = []
ebd = []
vvth = []
time = []
for pair in pairs:
	if pair[0][0]!='w':
		continue
	info.append({'ebd':int(pair[0].split('_')[1].split('d')[1]),
				 'vvth':int(pair[0].split('_')[2].split('h')[1].split('.')[0]),
				 'time':pair[1]})
	if info[-1]['time'] == 0:
		continue
	ebd.append(info[-1]['ebd'])
	vvth.append(info[-1]['vvth'])
	time.append(info[-1]['time'])

ebd_axis = list(set(ebd))
vvth_axis = list(set(vvth))
vvth_axis.sort()
ebd_axis.sort()
print ebd_axis
table = [[0 for j in range(len(ebd_axis))] for i in range(len(vvth_axis))]
print len(table), len(table[0])

with open('times.csv', 'w') as f:
	for (i, t) in enumerate(time):
		table[vvth_axis.index(vvth[i])][ebd_axis.index(ebd[i])] = t
	for e in ebd_axis:
		f.write( ','+str(e))
	f.write('\n')
	for (i, v) in enumerate(vvth_axis):
		f.write(str(v))
		for (j, _) in enumerate(ebd_axis):
			f.write(','+str(table[i][j]))
		f.write( '\n')

import numpy as np
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import random
from scipy import interpolate

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('embedding dimension')
ax.set_ylabel('vocabulary threshold')
ax.set_zlabel('time used(min)')
ax.scatter(ebd, vvth, time)
plt.show()
