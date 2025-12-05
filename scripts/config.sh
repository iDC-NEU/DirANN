#!/bin/bash

set -u

PROJECT_PATH=/root/code/DirANN
DATA_PATH=/root/dataset
# DISKANN_INDEX_PATH=/root/diskann_index
DISKANN_INDEX_PATH=/root/work_folder/diskann_index
OUTPUT_PATH=/root/work_folder
LOG_PATH=/root/log


# 函数1：根据 sys 获取标准化 sys_name
get_standard_sys_name() {
    local sys="$1"  # 接收输入参数（sys 值）
    local sys_name=""  # 局部变量，避免污染外部

    if [ "$sys" = "odinann" ]; then
        sys_name='OdinANN'
    elif [ "$sys" = "dgai" ]; then
        sys_name='DGAI'
    elif [ "$sys" = "greatorp" ]; then
        sys_name='GreatorP'
    elif [ "$sys" = "fresh" ]; then
        sys_name='FreshDiskANN'
    elif [ "$sys" = "wolve" ]; then
        sys_name='Wolverine'
    else
        echo "警告：未找到 sys=$sys 对应的标准化名称！" >&2
        sys_name=""  # 未匹配时返回空（可根据需求改为 exit 1 终止脚本）
    fi

    echo "$sys_name"  # 通过 echo 输出结果，外部用 $() 捕获
}

# 函数2：根据 data_name 获取标准化 dataset_name
get_standard_dataset_name() {
    local data_name="$1"  # 接收输入参数（data_name 值）
    local dataset_name=""  # 局部变量

    if [ "$data_name" = "sift1b" ]; then
        dataset_name='SIFT1B'
    elif [ "$data_name" = "sift100w" ] || [ "$data_name" = "sift100w_shuffled" ] ; then
        dataset_name='SIFT'
    elif [ "$data_name" = "gist" ]; then
        dataset_name='GIST'
    elif [ "$data_name" = "msong" ]; then
        dataset_name='MSONG'
    elif [ "$data_name" = "glove" ]; then
        dataset_name='GLOVE'
    elif [ "$data_name" = "text" ]; then
        dataset_name='TEXT'
    elif [ "$data_name" = "deep100m" ]; then
        dataset_name='DEEP100M'
    elif [ "$data_name" = "deep1b" ]; then
        dataset_name='DEEP1B'
    elif [ "$data_name" = "deep10m" ]; then
        dataset_name='DEEP10M'
    elif [ "$data_name" = "deep1m" ]; then
        dataset_name='DEEP1M'
    else
        echo "警告：未找到 data_name=$data_name 对应的标准化名称！" >&2
        dataset_name=""
    fi

    echo "$dataset_name"  # 输出结果，外部捕获
}