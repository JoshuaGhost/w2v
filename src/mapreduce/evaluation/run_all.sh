#!/bin/bash
OUTPUT=comb_eval_o

#hdfs dfs -put comb_eval_i.txt 
#hdfs dfs -rm comb_eval_m.py
#hdfs dfs -put comb_eval_m.py
#hdfs dfs -put comb_eval_r.py
#hdfs dfs -put web.zip
#hdfs dfs -put submodels.zip
#hdfs dfs -put mine.zip
hdfs dfs -rm -r comb_eval_o

hadoop jar /opt/cloudera/parcels/CDH/jars/hadoop-streaming-2.6.0-cdh5.13.0.jar\
            -archives hdfs:///user/zijian/env.zip#env,hdfs:///user/zijian/web.zip#web,hdfs:///user/zijian/submodels.zip#submodels,hdfs:///user/zijian/mine.zip#mine\
            -Dmapreduce.job.maps=100\
            -Dmapreduce.job.reduces=100\
            -Dmapreduce.map.memory.mb=10240\
            -Dmapreduce.map.java.opts=-Xmx8192m\
            -file comb_eval_m.py\
            -file comb_eval_r.py\
            -file utils.py\
            -file evaluate_embeddings.py\
            -file evaluate_on_all.py\
            -file sampling.py\
            -input comb_eval_i.txt\
            -mapper 'comb_eval_m.py'\
            -reducer 'comb_eval_r.py'\
            -output comb_eval_o\
            1>logs/comb_eval.stdout 2>logs/comb_eval.stderr

#           -files hdfs:///user/zijian/comb_eval_i.txt,\
#                   hdfs:///user/zijian/comb_eval_m.py,\
#                   hdfs:///user/zijian/comb_eval_r.py,\
#                   hdfs:///user/zijian/evaluate_embeddings.py,\
#                   hdfs:///user/zijian/evaluate_on_all.py\
 
