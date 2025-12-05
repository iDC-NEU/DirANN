#!/bin/bash

source "$(dirname "$0")/config.sh"

log_path=$LOG_PATH/fig2

mkdir -p $log_path

batch_num=-1
K=10

for sys in greatorp odinann
do
    if [ "$sys" = "greatorp" ]; then
        flag="-DBG_IO_THREAD -DREORDER_COMPUTE_PQ -DINPLACE_DELETE -DREVERSE_GRAPH"
        search_mode=0 
        pipeline_width=4
        strategy=0
    else 
        flag="-DBG_IO_THREAD"
        search_mode=0
        pipeline_width=4
        strategy=0
    fi

    #     flag=${flag}" -DCOLLECT_INFO"
    
    echo "flag = "$flag

    cd $PROJECT_PATH/build
    export ADDITIONAL_DEFINITIONS="$flag"
    cmake ..
    make -j
    cd "$PROJECT_PATH"


    for update_ratio in 0.004 0.008 
    do 
        for data_name in sift100w_shuffled gist  
        do
            L='100'
            batch_size=1000
        
            if [ "$data_name" = "sift100w" ]  || [ "$data_name" = "sift100w_shuffled" ] ; then
                L="50"
            elif [ "$data_name" = "msong" ]; then
                L="100"
            elif [ "$data_name" = "gist" ]; then
                L="400"
            elif [ "$data_name" = "deep100m" ]; then
                L="200"
                batch_size=100000
            elif [ "$data_name" = "deep10m" ]; then
                L="150"
                batch_size=10000
            elif [ "$data_name" = "deep1m" ]; then
                L="50"
            elif [ "$data_name" = "deep1b" ]; then
                L="400"
                batch_size=1000000
            fi
            batch_size=$update_ratio
            data_type=float
            data="${DATA_PATH}/${data_name}/${data_name}_base.fbin"
            query="${DATA_PATH}/${data_name}/${data_name}_query.fbin"
            truthset_prefix=${DATA_PATH}/${data_name}/${data_name}_gt_${update_ratio}

            original_index_path=${DISKANN_INDEX_PATH}/${data_name}/original_index
            index_path_prefix=${OUTPUT_PATH}/${data_name}/${data_name}
            l_disk=75

            if [ "$sys" = "greatorp" ]; then
                cp $original_index_path/${data_name}_disk.index.in_graph* "$OUTPUT_PATH"/${data_name}/
            fi

            rm $OUTPUT_PATH/${data_name}/*.tags
            cp $original_index_path/${data_name}_disk.index "$OUTPUT_PATH"/${data_name}/
            cp $original_index_path/${data_name}_pq_compressed.bin "$OUTPUT_PATH"/${data_name}/
            cp $original_index_path/${data_name}_pq_pivots.bin "$OUTPUT_PATH"/${data_name}/

            sys_name=$(get_standard_sys_name "$sys")
            dataset_name=$(get_standard_dataset_name "$data_name")
            log_file_path=${log_path}/${sys_name}_${dataset_name}_${update_ratio}.log

            cmd="${PROJECT_PATH}/build/tests/overall_performance  \
                ${data_type} \
                ${data} \
                ${l_disk} \
                ${index_path_prefix} \
                ${query} \
                ${truthset_prefix} \
                ${K} \
                ${pipeline_width} \
                ${batch_num} \
                ${batch_size} \
                ${search_mode} \
                ${strategy} \
                ${L}
                "
            echo ${cmd}
            eval ${cmd} 2>&1 | tee ${log_file_path}
        done
    done
done