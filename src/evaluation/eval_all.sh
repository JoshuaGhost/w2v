root=/home/zijian/workspace/master_thesis/w2v/models/

for model in {documentwise_100/combined/documentwise_100_pca.pkl,sampling_100/combined/sampling_100_concate.pkl,sampling_100/combined/sampling_100_pca.pkl}; do
    result_name=$(echo $model|rev|cut -d / -f 1|rev|cut -d . -f 1)_dist.csv
    division=$(echo $model|cut -d/ -f 1)
    echo $root/$division/results/$result_name 
    cat $root/$division/results/$result_name
    #python ./evaluate_on_all.py -f $root$model -p dict -o $root/$division/results/$result_name
done

