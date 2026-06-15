#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <cmath>

#define SECTOR_LEN 4096ULL
#define DIV_ROUND_UP(X, Y) (((uint64_t)(X) + (Y) - 1) / (Y))
#define ROUND_UP(X, Y) ((((uint64_t)(X) / (Y)) + ((uint64_t)(X) % (Y) != 0)) * (Y))

struct MergedMeta {
    uint64_t npoints;
    uint64_t max_node_len;
    uint64_t range;
    uint64_t data_dim;
    uint64_t residual_bits;
    uint64_t padding[3];
};

struct InputResHeader {
    uint64_t npoints;
    uint64_t dim;
    uint64_t bits; 
};

bool file_exists(const std::string& name) {
    struct stat buffer;   
    return (stat(name.c_str(), &buffer) == 0); 
}

template <typename T>
int process_index(const std::string& prefix) {
    std::string disk_index_path = prefix + "_disk.index";
    std::string residual_bin_path = prefix + "_residual_lvq.bin";
    
    std::string out_pmem_path = prefix + "_pmem.index";
    std::string out_disk_path = prefix + "_disk.data";

    if (!file_exists(disk_index_path)) {
        std::cerr << "Error: Cannot find " << disk_index_path << std::endl;
        return 1;
    }
    if (!file_exists(residual_bin_path)) {
        std::cerr << "Error: Cannot find " << residual_bin_path << std::endl;
        return 1;
    }

    std::ifstream disk_reader(disk_index_path, std::ios::binary);
    
    int meta_num_uint64, meta_unused_dim;
    disk_reader.read((char *)&meta_num_uint64, sizeof(int));
    disk_reader.read((char *)&meta_unused_dim, sizeof(int));

    std::vector<uint64_t> disk_meta_vec(meta_num_uint64); 
    disk_reader.read((char *)disk_meta_vec.data(), meta_num_uint64 * sizeof(uint64_t));
    
    uint64_t npts = disk_meta_vec[0];
    uint64_t ndims = disk_meta_vec[1];
    uint64_t old_max_node_len = disk_meta_vec[3];
    uint64_t old_nnodes_per_sector = disk_meta_vec[4];
    
    uint64_t input_nsector_per_node = DIV_ROUND_UP(old_max_node_len, SECTOR_LEN);

    int fd_res = open(residual_bin_path.c_str(), O_RDONLY | O_LARGEFILE);
    InputResHeader res_header;
    pread(fd_res, &res_header, sizeof(InputResHeader), 0);
    uint64_t res_bits = res_header.bits;

    uint64_t new_disk_node_len = ndims * sizeof(T); 
    uint64_t new_disk_nnodes_per_sector = SECTOR_LEN / new_disk_node_len;

    std::vector<uint64_t> output_disk_meta = disk_meta_vec;
    
    output_disk_meta[3] = new_disk_node_len;          
    output_disk_meta[4] = new_disk_nnodes_per_sector; 
    
    std::cout << "--- Disk Data Metadata Changes ---" << std::endl;
    std::cout << "Node Len: " << old_max_node_len << " -> " << new_disk_node_len << std::endl;
    std::cout << "Nodes/Sector: " << old_nnodes_per_sector << " -> " << new_disk_nnodes_per_sector << std::endl;

    uint64_t code_len = DIV_ROUND_UP(ndims * res_bits, 8);
    uint64_t vec_blob_len = sizeof(float) + sizeof(float) + code_len; 
    uint64_t max_node_degree = (old_max_node_len - ndims * sizeof(T) - sizeof(uint32_t)) / sizeof(uint32_t);
    uint64_t topo_len = (max_node_degree + 1) * sizeof(uint32_t); 
    uint64_t pmem_node_len = vec_blob_len + topo_len;

    int fd_out_pmem = open(out_pmem_path.c_str(), O_CREAT | O_RDWR | O_LARGEFILE | O_TRUNC, 0666);
    int fd_out_disk = open(out_disk_path.c_str(), O_CREAT | O_RDWR | O_LARGEFILE | O_TRUNC, 0666);
    if (fd_out_pmem == -1 || fd_out_disk == -1) {
        perror("Failed to open output files");
        return 1;
    }

    MergedMeta final_pmem_meta;
    final_pmem_meta.npoints = npts;
    final_pmem_meta.max_node_len = pmem_node_len;
    final_pmem_meta.range = max_node_degree;
    final_pmem_meta.data_dim = ndims;
    final_pmem_meta.residual_bits = res_bits;
    final_pmem_meta.padding[0] = SECTOR_LEN / pmem_node_len; 

    
    pwrite(fd_out_pmem, &final_pmem_meta, sizeof(MergedMeta), 0);

    
    pwrite(fd_out_disk, &meta_num_uint64, sizeof(int), 0);
    pwrite(fd_out_disk, &meta_unused_dim, sizeof(int), sizeof(int));
    
    uint64_t meta_array_size = meta_num_uint64 * sizeof(uint64_t);
    pwrite(fd_out_disk, output_disk_meta.data(), meta_array_size, 2 * sizeof(int));

    uint64_t header_used_bytes = 2 * sizeof(int) + meta_array_size;
    if (header_used_bytes < SECTOR_LEN) {
        std::vector<char> disk_pad(SECTOR_LEN - header_used_bytes, 0);
        pwrite(fd_out_disk, disk_pad.data(), disk_pad.size(), header_used_bytes);
    }

    
    uint64_t pmem_cursor = sizeof(MergedMeta); 
    uint64_t disk_cursor = SECTOR_LEN; 

    std::vector<char> input_sector_buf(SECTOR_LEN * input_nsector_per_node);
    std::vector<char> pmem_node_buf(pmem_node_len, 0);
    std::vector<char> disk_vec_buf(new_disk_node_len, 0);

    disk_reader.seekg(SECTOR_LEN, std::ios::beg);

    uint64_t processed_pts = 0;
    while (processed_pts < npts) {
        disk_reader.read(input_sector_buf.data(), SECTOR_LEN * input_nsector_per_node);
        
        for (uint32_t i = 0; i < old_nnodes_per_sector && processed_pts < npts; ++i, ++processed_pts) {
            char* input_node_ptr = input_sector_buf.data() + i * old_max_node_len;
            
            
            memcpy(disk_vec_buf.data(), input_node_ptr, new_disk_node_len);

            if ((disk_cursor % SECTOR_LEN) + new_disk_node_len > SECTOR_LEN) {
                disk_cursor = (disk_cursor / SECTOR_LEN + 1) * SECTOR_LEN;
            }
            pwrite(fd_out_disk, disk_vec_buf.data(), new_disk_node_len, disk_cursor);
            disk_cursor += new_disk_node_len;

            
            uint32_t nnbrs = *reinterpret_cast<uint32_t*>(input_node_ptr + new_disk_node_len);
            uint32_t* nbrs = reinterpret_cast<uint32_t*>(input_node_ptr + new_disk_node_len + sizeof(uint32_t));

            std::vector<char> res_blob(vec_blob_len);
            uint64_t res_offset = sizeof(InputResHeader) + processed_pts * vec_blob_len;
            pread(fd_res, res_blob.data(), vec_blob_len, res_offset);

            memset(pmem_node_buf.data(), 0, pmem_node_len);
            memcpy(pmem_node_buf.data(), res_blob.data(), vec_blob_len); 
            memcpy(pmem_node_buf.data() + vec_blob_len, &nnbrs, sizeof(uint32_t)); 
            memcpy(pmem_node_buf.data() + vec_blob_len + sizeof(uint32_t), nbrs, nnbrs * sizeof(uint32_t));

            
            pwrite(fd_out_pmem, pmem_node_buf.data(), pmem_node_len, pmem_cursor);
            pmem_cursor += pmem_node_len;
        }

        if (processed_pts % 100000 == 0 || processed_pts == npts) {
            std::cout << "\rProgress: " << (processed_pts * 100 / npts) << "%" << std::flush;
        }
    }

    ftruncate(fd_out_disk, ROUND_UP(disk_cursor, SECTOR_LEN));
    ftruncate(fd_out_pmem, pmem_cursor);

    std::cout << "\nFinished!" << std::endl;
    std::cout << "PMEM Index: " << out_pmem_path << " (Compact, offset 64)" << std::endl;
    std::cout << "Disk Data : " << out_disk_path << " (Aligned 4K)" << std::endl;

    close(fd_res);
    close(fd_out_pmem);
    close(fd_out_disk);
    disk_reader.close();
    return 0;
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <prefix> <dtype: float/uint8>" << std::endl;
        return 1;
    }
    std::string prefix = argv[1];
    std::string dtype = argv[2];
    if (dtype == "float") return process_index<float>(prefix);
    else if (dtype == "uint8") return process_index<uint8_t>(prefix);
    else std::cerr << "Wrong data type!" << std::endl;
    return 1;
}