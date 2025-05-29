#!/bin/bash
dataset_list=('ABIDE')
batch_size_list=(16 32 64)
base_lr_list=(1e-4 1e-5)
target_lr_list=(1e-4 1e-5)
wd_list=(1e-4 1e-5 1e-3)
layers_list=(1 2 3)
#hidden_list=(16 32 64)
activation_list=('leaky_relu')
dropout_list=(0. 0.1 0.2 0.3 0.5)
pooling_list=(True)
cluster_num_list=(4 16 32)


for name in "${dataset_list[@]}"; do
  for pl in "${pooling_list[@]}"; do
    for acl in "${activation_list[@]}"; do
      for b_lrl in "${base_lr_list[@]}"; do
        for t_lrl in "${target_lr_list[@]}"; do
          for wdl in "${wd_list[@]}"; do
            for bzl in "${batch_size_list[@]}"; do
              for dl in "${dropout_list[@]}"; do
                for clus_numl in "${cluster_num_list[@]}"; do
                  for layersl in "${layers_list[@]}"; do
                    python main.py --dataset $name \
                        --batch_size $bzl \
                        --base_lr $b_lrl \
                        --target_lr $t_lrl \
                        --weight_decay $wdl \
                        --layers $layersl \
                        --activation $acl \
                        --dropout $dl \
                        --pooling $pl \
                        --cluster_num $clus_numl
                  done
                done
              done
            done
          done
        done
      done
    done
  done
done


