#!/bin/bash

DRY_RUN=$1
if [[ $DRY_RUN -ne 0 ]]; then
    debug=echo
fi

CORE_CONSTRAINT=4-21

NMODELS=10

BASE_DIR="/home/zijian/workspace/master_thesis/w2v/"

TMP_DIR="${BASE_DIR}temp/models/"
SKIP_DIR="${TMP_DIR}skip/"
BOOT_DIR="${TMP_DIR}bootstrap/"
COMB_SKIP_DIR="${BASE_DIR}output/models/combined/skip/"
$debug mkdir -pv $COMB_SKIP_DIR
COMB_BOOT_DIR="${BASE_DIR}output/models/combined/bootstrap/"
$debug mkdir -pv $COMB_BOOT_DIR

RESULTS_DIR="${BASE_DIR}output/results/temp/"
$debug mkdir -pv $RESULTS_DIR
FINAL_DIR="${BASE_DIR}output/results/final/"
$debug mkdir -pv $FINAL_DIR


CONFIGS=$( parallel --gnu printf "%s,%s,%s,%s,%s\ " ::: \
            {$SKIP_DIR,$BOOT_DIR} ::: \
            {'seq','dicht','MPE'} ::: \
            {'vadd','lint','pca'} ::: \
            {'Y','N'} ::: \
            {'Y','N'} )

counter=0

for cfg in $CONFIGS; do
    TEMP_IFS=$IFS
    IFS=,
    set -- $cfg
    if [ $1 == $SKIP_DIR ]; then
	COMB_DIR=$COMB_SKIP_DIR
    else
	COMB_DIR=$COMB_BOOT_DIR
    fi

    $debug taskset -c $CORE_CONSTRAINT python combine_pairwise.py $1 $NMODELS $2 $3 $4 $5 $COMB_DIR
done

#    counter=$[counter+1]
#    if [ $counter -eq 18 ]; then
#        $debug python evaluate_embeddings.py -j 18 -d $COMB_MODELS_DIR -o $TMP_RESULTS_DIR
#        >&2 echo python evaluate_embeddings.py -j 18 -d $COMB_MODELS_DIR -o $TMP_RESULTS_DIR
#        $debug mv "${COMB_MODELS_DIR}*" $ARCHIVED_MODELS_DIR
#        >&2 echo mv "${COMB_MODELS_DIR}*" $ARCHIVED_MODELS_DIR
#        $debug echo "${counter} combined models archived"| mail -s 'models archived' zhangzijian0523@gmail.com
#	counter=0
#    fi
#    IFS=$TEMP_IFS
#done
#
#if [ $counter -ne 0 ]; then
#    $debug python evaluate_embeddings.py -j 18 -d $COMB_MODELS_DIR -o $TMP_RESULTS_DIR
#    >&2 echo python evaluate_embeddings.py -j 18 -d $COMB_MODELS_DIR -o $TMP_RESULTS_DIR
#    $debug mv "${COMB_MODELS_DIR}*" $ARCHIVED_MODELS_DIR
#    >&2 echo mv "${COMB_MODELS_DIR}*" $ARCHIVED_MODELS_DIR
#    $debug echo "${counter} combined models archived"| mail -s 'models archived' zhangzijian0523@gmail.com
#fi

#$debug $[ find $TMP_RESULTS_DIR -type f -print -exec cat {} \;|perl -p -e 's/(csv)\n/\1/g'|sed -e 's/.*merge-2-\(.*\)\.csv/\1/g'>"${FINAL_RESULT_DIR}skip.csv" ]
