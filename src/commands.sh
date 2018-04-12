#!/bin/bash

for dataset in {MEN,RW,Google,WS353}; do python main.py -t bench --dump_folder /tmp/zzj/ --cfolder ../two_10_pct_sub_corpora/ --cname_source c0 --cname_target c1 --vbfolder ../benchmark_vocabs/ --vbname $dataset 1>>stdout 2>>stderr; done

for dataset in {MEN,RW,Google,WS353}; do python main.py -t fixall -d /tmp/zzj/$dataset -f m -e csv -T csv --dump_folder /tmp/zzj/; done
