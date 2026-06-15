#!/bin/bash

source "$(dirname "$0")/config.sh"

cd $PROJECT_PATH/build
cmake  ..
make -j16
cd $PROJECT_PATH

echo -e "\n----------------------------------------------------------------"
data_name=sift
npts=1000000
base_npts=500000

full_data=${DATA_PATH}/${data_name}/${data_name}_base.fbin
query=${DATA_PATH}/${data_name}/${data_name}_query.fbin
topk=10000
gt=${DATA_PATH}/${data_name}/${data_name}_gt_K${topk}
target_topk=10
data_type=float

if [ ! -f "$gt" ]; then
    echo "Computing ground truth for $data_name with topk=$topk..."
    cmd="${PROJECT_PATH}/build/tests/utils/compute_groundtruth $data_type $full_data $query $topk $gt"
    echo -e "\nGEN GT: $cmd"
    eval ${cmd}
else
    echo "Note: $gt already exists."
fi

step_gt_num=11
update_ratio=0.1
update_size=$(echo "$base_npts * $update_ratio" | bc -l | awk '{print int($1)}')
echo -e "\ndata_name: $data_name, update_ratio: $update_ratio, update_size: $update_size"

truthset_prefix=${DATA_PATH}/${data_name}/${data_name}_${base_npts}_gt_${update_ratio}
cmd="${PROJECT_PATH}/build/tests/gt_update $gt $base_npts $npts $update_size ${step_gt_num} $target_topk $truthset_prefix 0"
echo -e "\nGEN BATCH GT: $cmd"
eval $cmd

${PROJECT_PATH}/build/tests/change_pts $data_type $full_data $base_npts

original_index_path=${DISKANN_INDEX_STORE_PATH}/$data_name/original_index_${base_npts}
mkdir -p ${original_index_path}
data=$full_data$base_npts

R=32
pq_bytes=1
M=64
Lbuild=75
metric=l2
nbr_type=pq

index_path_prefix=${original_index_path}/${data_name}_R${R}_L${Lbuild}

# build_disk_index <type> <data> <prefix> <R> <L> <PQ_bytes> <M_GB> <threads> <metric> <nbr_type>
cmd="$PROJECT_PATH/build/tests/build_disk_index \
    ${data_type}  \
    ${data} \
    ${index_path_prefix} \
    ${R} \
    ${Lbuild} \
    ${pq_bytes} \
    ${M} \
    32 \
    ${metric} \
    ${nbr_type} \
    "
echo -e "\nBUILD INDEX:"
echo $cmd
eval $cmd