#!/bin/bash

source "$(dirname "$0")/config.sh"

batch_num=40

use_live_graph=0
collect_n_computation=OFF
for use_live_graph in 0
do
    if [ "$collect_n_computation" = "ON" ]; then
        log_path=$LOG_PATH/fig1_lg
        batch_num=10
    else
        log_path=$LOG_PATH/fig1
        batch_num=10
    fi

    mkdir -p $log_path

    for sys in greatorp 
    do
        K=10
        if [ "$sys" = "greatorp" ]; then
            flag=" -DBG_IO_THREAD -DREORDER_COMPUTE_PQ -DINPLACE_DELETE -DREVERSE_GRAPH"
            search_mode=0 
            pipeline_width=4
            strategy=0
        else 
            flag=" -DBG_IO_THREAD"
            search_mode=0
            pipeline_width=4
            strategy=0
        fi

        if [ "$collect_n_computation" = "ON" ]; then
            flag="$flag -DCOLLECT_INFO"
        fi

        if [ "$use_live_graph" = "1" ]; then
            flag="$flag -DUSE_LIVE_GRAPH"
        fi

        cd $PROJECT_PATH/build
        export ADDITIONAL_DEFINITIONS="$flag"
        cmake ..
        make -j
        cd "$PROJECT_PATH"

        for data_name in sift100w_shuffled
        do
            L='100'
            batch_size=1000
        
            if [ "$data_name" = "sift100w" ]  || [ "$data_name" = "sift100w_shuffled" ] ; then
                L="50"
                # L="10 15 20 25 30 50 55 60 65 70 75 80 85/ 90 95 100 105 110 115 120 125 130 135 145 150 155"
            elif [ "$data_name" = "msong" ]; then
                L="100"
            elif [ "$data_name" = "text" ]; then
                L="100"
                # L="20 30 40 50 60 70 80 90 100 110 120 130 140 150 160 170 180 190 200 210 220 230 240 250 260 270 280 290 300"
            elif [ "$data_name" = "gist" ]; then
                L="400"
                # L="160 200 230 260 300 360 420 500 600 700 800 900 1000 1100 1200 1300 1400 1500 1600 1700"
            elif [ "$data_name" = "deep100m" ]; then
                L="200"
                batch_size=100000
                # L="200 100 120 140 160 200 250 300 360 420 500 600 700 900 1100 1500"
            elif [ "$data_name" = "deep10m" ]; then
                L="150"
                batch_size=10000
            elif [ "$data_name" = "deep1m" ]; then
                L="50"
            elif [ "$data_name" = "deep1b" ]; then
                L="400"
                batch_size=1000000
            fi

            data_type=float
            data="${DATA_PATH}/${data_name}/${data_name}_base.fbin"
            query="${DATA_PATH}/${data_name}/${data_name}_query.fbin"
            truthset_prefix=${DATA_PATH}/${data_name}/${data_name}_gt_0.00125

            original_index_path=${DISKANN_INDEX_PATH}/${data_name}/original_index
            index_path_prefix=${OUTPUT_PATH}/${data_name}/${data_name}
            l_disk=75

            if [ "$sys" = "greatorp" ]; then
                if [ "$use_live_graph" = "1" ]; then
                    cp $original_index_path/${data_name}_disk.index.in_graph_lg* "$OUTPUT_PATH"/${data_name}/
                else
                    cp $original_index_path/${data_name}_disk.index.in_graph* "$OUTPUT_PATH"/${data_name}/
                fi
            fi

            rm $OUTPUT_PATH/${data_name}/*.tags
            cp $original_index_path/${data_name}_disk.index "$OUTPUT_PATH"/${data_name}/
            cp $original_index_path/${data_name}_pq_compressed.bin "$OUTPUT_PATH"/${data_name}/
            cp $original_index_path/${data_name}_pq_pivots.bin "$OUTPUT_PATH"/${data_name}/

            # log_file_path=${log_path}/_.log
            
            sys_name=$(get_standard_sys_name "$sys")
            dataset_name=$(get_standard_dataset_name "$data_name")
            log_file_path=${log_path}/${sys_name}_${dataset_name}.log

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