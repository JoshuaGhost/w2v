#!/bin/bash
parallel -j0 --results logs/drop_out python evaluate_on_demand.py /home/zijian/combined_models/{1}.pkl {2}.txt ::: partitioning_100_pca partitioning_10_pca sampling_100_pca sampling_10_pca sampling_100_pca_dim_500 sampling_10_pca_dim_500 ::: join intersection
#python evaluate_on_demand.py /home/zijian/combined_models/base_line.pkl join.txt
#python evaluate_on_demand.py /home/zijian/combined_models/partitioning_100_pca.pkl join.txt
#python evaluate_on_demand.py /home/zijian/combined_models/partitioning_10_pca.pkl join.txt
#python evaluate_on_demand.py /home/zijian/combined_models/sampling_100_pca.pkl join.txt
#python evaluate_on_demand.py /home/zijian/combined_models/sampling_10_pca.pkl join.txt
#python evaluate_on_demand.py /home/zijian/combined_models/sampling_100_pca_dim_500.pkl join.txt
#python evaluate_on_demand.py /home/zijian/combined_models/sampling_10_pca_dim_500.pkl join.txt

#python evaluate_on_demand.py /home/zijian/combined_models/base_line.pkl intersection.txt
#python evaluate_on_demand.py /home/zijian/combined_models/partitioning_100_pca.pkl intersection.txt
#python evaluate_on_demand.py /home/zijian/combined_models/partitioning_10_pca.pkl intersection.txt
#python evaluate_on_demand.py /home/zijian/combined_models/sampling_100_pca.pkl intersection.txt
#python evaluate_on_demand.py /home/zijian/combined_models/sampling_10_pca.pkl intersection.txt
#python evaluate_on_demand.py /home/zijian/combined_models/sampling_100_pca_dim_500.pkl intersection.txt
#python evaluate_on_demand.py /home/zijian/combined_models/sampling_10_pca_dim_500.pkl intersection.txt

