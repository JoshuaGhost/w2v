#!/bin/bash

DRY_RUN=$1
if [[ $DRY_RUN -ne 0 ]]; then
    debug=echo
fi

CORE_CONSTRAINT=4-21

NMODELS=10

BASE_DIR="/home/zijian/workspace/master_thesis/w2v/"

TMP_MODELS_DIR="${BASE_DIR}temp/models/"
SKIP_MODELS_DIR="${TMP_MODELS_DIR}skip/"
BOOT_MODELS_DIR="${TMP_MODELS_DIR}bootstrap/"
COMB_MODELS_DIR="${BASE_DIR}output/models/combined/"
ARCHIVED_MODELS_DIR="${BASE_DIR}output/models/archived/"

TMP_RESULTS_DIR="${BASE_DIR}output/results/temp/"
FINAL_RESULT_DIR="${BASE_DIR}output/results/ten_all/"

$debug mkdir $COMB_MODELS_DIR
$debug mkdir $TMP_RESULTS_DIR
$debug mkdir $FINAL_RESULT_DIR
$debug mkdir $ARCHIVED_MODELS_DIR

            #{'sequential','dichoto','min_procrustes_err'} ::: \
CONFIGS=$( parallel --gnu printf "%s,%s,%s,%s,%s\ " ::: \
            {$SKIP_MODELS_DIR,$BOOT_MODELS_DIR} ::: \
            'min_procrustes_err' ::: \
            {'vector_addition','linear_transform','pca_transbase'} ::: \
            {'Y','N'} ::: \
            {'Y','N'} )

counter=0

for cfg in $CONFIGS; do
    TEMP_IFS=$IFS
    IFS=,
    set -- $cfg
    $debug taskset -c $CORE_CONSTRAINT python combine_pairwise.py $1 $NMODELS $2 $3 $4 $5 $COMB_MODELS_DIR
    >&2 echo taskset -c $CORE_CONSTRAINT python combine_pairwise.py $1 $NMODELS $2 $3 $4 $5 $COMB_MODELS_DIR
    counter=$[counter+1]
    if [ $counter -eq 18 ]; then
        counter=0
        $debug python evaluate_embeddings.py -j 18 -d $COMB_MODELS_DIR -o $TMP_RESULTS_DIR
        >&2 echo python evaluate_embeddings.py -j 18 -d $COMB_MODELS_DIR -o $TMP_RESULTS_DIR
        $debug mv "${COMB_MODELS_DIR}*" $ARCHIVED_MODELS_DIR
        >&2 echo mv "${COMB_MODELS_DIR}*" $ARCHIVED_MODELS_DIR
        $debug echo "${counter} combined models archived"| mail -s 'models archived' zhangzijian0523@gmail.com
    fi
    IFS=$TEMP_IFS
done

if [ $counter -ne 0 ]; then
    $debug python evaluate_embeddings.py -j 18 -d $COMB_MODELS_DIR -o $TMP_RESULTS_DIR
    >&2 echo python evaluate_embeddings.py -j 18 -d $COMB_MODELS_DIR -o $TMP_RESULTS_DIR
    $debug mv "${COMB_MODELS_DIR}*" $ARCHIVED_MODELS_DIR
    >&2 echo mv "${COMB_MODELS_DIR}*" $ARCHIVED_MODELS_DIR
    $debug echo "${counter} combined models archived"| mail -s 'models archived' zhangzijian0523@gmail.com
fi

$debug $[ find $TMP_RESULTS_DIR -type f -print -exec cat {} \;|perl -p -e 's/(csv)\n/\1/g'|sed -e 's/.*merge-2-\(.*\)\.csv/\1/g'>"${FINAL_RESULT_DIR}skip.csv" ]

