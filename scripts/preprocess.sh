#!/bin/bash

source "$(dirname "$0")/config.sh"

for data_name in deep100m 
do
    npts=1000000

    batch_npts=800000

    full_data=${DATA_PATH}/${data_name}/${data_name}_base.fbin
    query=${DATA_PATH}/${data_name}/${data_name}_query.fbin
    topk=10000
    gt=${DATA_PATH}/${data_name}/${data_name}_gt_K${topk}
    target_topk=10
    data_type=float

    cd $PROJECT_PATH/build
    cmake  ..
    make -j
    cd $PROJECT_PATH

    # ${PROJECT_PATH}/build/tests/utils/compute_groundtruth $data_type $full_data $query $topk $gt
    # for update_ratio in 0.001 0.002 0.004 0.008 0.01
    for update_ratio in 0.00125
    do
        truthset_prefix=${DATA_PATH}/${data_name}/${data_name}_gt_${update_ratio}
        # ${PROJECT_PATH}/build/tests/gt_update $gt $npts $batch_npts $target_topk $truthset_prefix $update_ratio 0
    done
    # exit 0
    # ${PROJECT_PATH}/build/tests/change_pts $data_type $full_data $batch_npts
    
    original_index_path=${DISKANN_INDEX_PATH}/$data_name/original_index
    mkdir -p ${original_index_path}
    data=$full_data$batch_npts
    index_path_prefix=${original_index_path}/${data_name}


    R=32
    B=64
    M=64
    Lbuild=75
    metric=l2
    single_file_index=0
    #/home/loujh/yq_code/Greator/src/aux_utils.cpp 987行控制pq大小
    # 搜索size_t num_pq_chunks =
    cmd="$PROJECT_PATH/build/tests/build_disk_index \
        ${data_type}  \
        ${data} \
        ${index_path_prefix} \
        ${R} \
        ${Lbuild} \
        ${B} \
        ${M} \
        32 \
        ${metric} \
        ${single_file_index}
        "
    echo "BUILD cmd:"
    echo $cmd
    # eval $cmd


    index_path=$original_index_path/$data_name"_disk.index"

    $PROJECT_PATH/build/tests/create_reverse_graph $index_path

done