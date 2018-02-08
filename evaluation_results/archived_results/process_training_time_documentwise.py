def time_elapse_in_sec(line):
    d = int(line.split()[0].split('-')[2])-7
    t = line.split()[1][:-1]
    h = d*24+int(t.split(':')[0])
    m = h*60+int(t.split(':')[1])
    s = m*60+int(t.split(':')[2].split(',')[0])
    ms = float(t.split(',')[1])
    return s+ms/1000

f = open('training_time_sub_model_documentwise.txt','r')
t = 0
for i in range(10):
    t1 = time_elapse_in_sec(f.readline())
    t2 = time_elapse_in_sec(f.readline())
    t = t+t2-t1
print t/10
