#!/bin/bash

source "$(dirname "$0")/config.sh"

log_path=$LOG_PATH/exp
mkdir -p $log_path

step=10
update_ratio=0.1
sed -i'' -E "s/(uint64_t vecs_per_step = index_npts \s*\* \s*)[0-9]+\.?[0-9]*(;)/\1$update_ratio\2/" $PROJECT_PATH/tests/overall_perf_mem.cpp

sys=DiskANN
data_name=sift
echo "[$(date '+%Y-%m-%d %H:%M:%S')] start"
sed -i'' -E "s/(std::string sys_name = \")[^\"]*(\";)/\1$sys\2/" "$PROJECT_PATH/tests/overall_perf_mem.cpp"

log=${log_path}/${data_name}_${sys}_saveAllUpdatesIndex_u${update_ratio}_s${step}.log
base_npts=500000

full_data=${DATA_PATH}/${data_name}/${data_name}_base.fbin
query_file=${DATA_PATH}/${data_name}/${data_name}_query.fbin
truthset_prefix=${DATA_PATH}/${data_name}/${data_name}_${base_npts}_gt_${update_ratio}

cd $PROJECT_PATH/build
export ADDITIONAL_DEFINITIONS="-DDIRANN"

echo "ADDITIONAL_DEFINITIONS=${ADDITIONAL_DEFINITIONS}"
cmake .. 
make -j16 
cd $PROJECT_PATH

# 1. <type> 2. <data_bin> 3. <L_disk> 4. <R_disk> 5. <alpha_disk>
L_disk=75 # L_build
R_disk=32 #unused
alpha_disk=1.2 #unused

# 6. <num_start> 7. <#nodes_to_cache> 8. <indice_path>
num_start=0 #unused
num_nodes_to_cache=0 #unused

# 9. <query_file> 10. <truthset_prefix> 11. <recall@>
recall_at=10

# 12. <#beam_width> 13. <step> 14. <Lsearch> 15. <L2>
beamwidth=2 #unused
Lsearch="100" 
data_type=float
R=32
index_prefix_path=${DISKANN_INDEX_PATH}/$data_name/original_index_${base_npts}/${data_name}_R${R}_L${L_disk}

cmd="
    $PROJECT_PATH/build/tests/overall_perf_mem \
    $data_type \
    $full_data \
    $L_disk \
    $R_disk \
    $alpha_disk \
    $num_start \
    $num_nodes_to_cache \
    $index_prefix_path \
    $query_file \
    $truthset_prefix \
    $recall_at \
    $beamwidth \
    $step \
    $Lsearch \
    > ${log}_R${R} 2>&1
"
echo $cmd
eval $cmd