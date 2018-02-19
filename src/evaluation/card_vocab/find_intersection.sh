#!/bin/bash

MODELS_VOCAB=models/
GT_VOCAB=ground_truth/
mkdir -pv comm

rm recall.csv
printf ' , '>>./recall.csv
for gt in $(ls $GT_VOCAB/*); do
    g=$(echo $gt|sed -E 's/.*\/(.*)\.txt/\1/g')
    printf "${g}($(wc -l $gt|cut -d' ' -f 1)),">>./recall.csv
done
printf '\n'>>./recall.csv

for model in $(ls $MODELS_VOCAB/*); do
    m=$(echo $model|sed -E 's/.*\/(.*)\.txt/\1/g')
    printf "${m}, ">>./recall.csv
    for gt in $(ls $GT_VOCAB/*); do
        g=$(echo $gt|sed -E 's/.*\/(.*)\.txt/\1/g')
        comm -1 -2 $model $gt > comm/${m}_${g}.txt
        printf $(echo "scale=3; $(wc -l comm/${m}_${g}.txt |cut -d' ' -f 1)/$(wc -l $gt |cut -d' ' -f 1)"|bc)',' >>./recall.csv
    done
    printf "\n">>./recall.csv
done
