base_dir = "/home/zijian/workspace/master_thesis/w2v/"
division_strategy = 'sampling_100'
combine_method = 'concate'
arch = 'mapreduce'
folder_in = base_dir+'models/'+division_strategy+'/'+'subs/'
fname = 'part-'
ext = ''
ndim = 500
mean_corr = False
norm = False
folder_out = base_dir+'models/'+division_strategy+'/'+'combined/'
dump_name = division_strategy+'_'+combine_method
