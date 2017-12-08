#!./env/bin/python

for i in range(10):
    print 'aa'+str(i)+'dfe'+'\t'+''.join(chr(ord('a')+i) for j in range(3))
