#include "log.h"
#include "v2/DynGraphDisk.hpp"
#include "v2/DynGraphDisk_lg.hpp"
#include "query_buf.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>
#include <mutex>
#include <atomic>
#include <omp.h>
#include <sys/mman.h>
#include <sys/stat.h>
// 补充必要头文件（原代码依赖但未包含）
#include <fcntl.h>       // open/O_RDONLY 所需
#include <cstdint>       // uint64_t/uint32_t 等类型定义
#include <cassert>       // assert 宏所需
#include <cerrno>        // strerror 依赖 errno 定义
#include <cstdlib>       // exit 函数所需

#define SECTORS_PER_MERGE 65536ULL
#define SECTOR_LEN 4096ULL

#define USE_LIVE_GRAPH

template<typename T>
struct DiskNode {
    uint32_t id = 0;
    T *coords = nullptr;
    uint32_t nnbrs;
    uint32_t *nbrs;

    // id : id of node
    // sector_buf : sector buf containing `id` data
    
    DiskNode(uint32_t id, T *coords, uint32_t *nhood) : id(id) {
        this->coords = coords;
        this->nnbrs = *nhood;
        this->nbrs = nhood + 1;
    }
    DiskNode(uint32_t id, T *coords) : id(id) {
        this->coords = coords;
    }
    DiskNode(uint32_t id, uint32_t *nhood) : id(id) {
        this->nnbrs = *nhood;
        this->nbrs = nhood + 1;
    }
    
};
// structs for DiskNode
template struct DiskNode<float>;
template struct DiskNode<uint8_t>;
template struct DiskNode<int8_t>;


// 获取topk值（返回值为前k大的元素，按从大到小排序）
std::vector<uint32_t> get_topk_values_heap(const std::vector<uint32_t>& degrees, size_t k) {
    if (k == 0 || k > degrees.size()) {
        return {};
    }
    
    // 小顶堆：堆顶是当前k个元素中最小的（方便替换）
    std::priority_queue<uint32_t, std::vector<uint32_t>, std::greater<uint32_t>> min_heap;
    
    for (uint32_t d : degrees) {
        if (min_heap.size() < k) {
            // 堆未满，直接加入
            min_heap.push(d);
        } else if (d > min_heap.top()) {
            // 当前元素比堆顶大，替换堆顶（保证堆内是前k大）
            min_heap.pop();
            min_heap.push(d);
        }
    }
    
    // 从堆中提取元素（此时堆内元素是前k大，但顺序是从小到大）
    std::vector<uint32_t> topk;
    while (!min_heap.empty()) {
        topk.push_back(min_heap.top());
        min_heap.pop();
    }
    // 反转得到从大到小的顺序
    std::reverse(topk.begin(), topk.end());
    
    return topk;
}

template <typename T>
int CreatReverseGraph(std::string index_path) {
    
    std::ifstream reader(index_path, std::ios::binary);
    if (!reader) {
        std::cerr << "Failed to open: " << index_path << std::endl;
        return 1;
    }

    int meta_npts, meta_ndims;
    reader.read((char *)&meta_npts, sizeof(int));
    reader.read((char *)&meta_ndims, sizeof(int));

    std::vector<uint64_t> meta(meta_npts);
    for (auto &x : meta) {
        reader.read((char *)&x, sizeof(uint64_t));
    }

    uint64_t num_points = meta[0];
    uint64_t ndims = meta[1];
    uint64_t max_node_len = meta[3];
    uint64_t nnodes_per_sector = meta[4];

    uint64_t range = (max_node_len - ndims * sizeof(T) - sizeof(uint32_t)) / sizeof(uint32_t);

    reader.close();

    size_t file_size;
    if (num_points <= 1e6 + 5) {
        file_size = 1ULL << 30;
    } else if (num_points <= 1e8 + 5) {
        file_size = 1ULL << 36;
    } else if (num_points <= 1e9 + 5) {
        file_size = 1ULL << 38;
    }
    LOG(INFO) << "Make reverse graph";

    // 修正：BlockGraph 是指针类型（new 返回指针，原代码少 *）
#ifdef USE_LIVE_GRAPH
    livegraph::BlockGraph* disk_in_graph = new livegraph::BlockGraph (index_path + ".in_graph_lg", num_points * 1.05, range, 4, false, file_size);
#else
    DynGraphDisk::BlockGraph* disk_in_graph = new DynGraphDisk::BlockGraph (index_path + ".in_graph", num_points * 1.05, range, 4, 100000, false, file_size);
#endif
    // uint64_t max_node_len = (ndims * sizeof(T)) + sizeof(uint32_t) + (range * sizeof(uint32_t));
    // uint64_t nnodes_per_sector = SECTOR_LEN / max_node_len;
    
    std::atomic<int> cnt {0};
    uint64_t n_sectors = (num_points + nnodes_per_sector - 1) / nnodes_per_sector;

    LOG(INFO) << "\nmax_node_len = " << max_node_len << "\n" 
    << "nnodes_per_sector = " << nnodes_per_sector << "\n"
    << "n_sectors = " << n_sectors << "\n"
    << "ndims = " << ndims << "\n"
    << "range = " << range ;
    
    int ntds = omp_get_max_threads();
    std::vector<std::mutex> mutexs(num_points);
    std::vector<uint32_t> degrees(num_points, 0);  // 补充初始化：度数默认0
    
    int fd = open(index_path.c_str(), O_RDONLY);
    if (fd == -1) {
        std::cerr << "open failed: " << strerror(errno) << std::endl;
        exit(-1);
    }
    struct stat st;
    if (fstat(fd, &st) == -1) {
        perror("fstat");
        close(fd);  // 补充：fstat失败后关闭文件，避免资源泄漏
        exit(-1);
    }
    if (st.st_size == 0) {
        std::cerr << "file size is 0\n";
        close(fd);
        exit(-1);
    }

    std::cout << "begin init mem_in_graph\n";
    size_t disk_npts = num_points;
    uint64_t sz = (1ULL * st.st_size + SECTOR_LEN - 1) / SECTOR_LEN  * SECTOR_LEN;
    uint64_t nhood_len = sizeof(uint32_t) + (range * sizeof(uint32_t));

    char * buf = (char *)malloc(SECTORS_PER_MERGE * SECTOR_LEN);
    if (buf == nullptr) {
        std::cerr << "malloc failed: " << strerror(errno) << std::endl;
        close(fd);
        exit(-1);
    }

    LOG(INFO) << "Begin degree counting.";
    constexpr uint64_t CHUNK = SECTORS_PER_MERGE * SECTOR_LEN; // 1 GiB
    uint64_t offset = SECTOR_LEN;
    uint64_t start_id = 0;

    while (offset < sz && start_id < disk_npts) {
        uint64_t this_len = std::min(CHUNK, sz - offset);

        ssize_t bytes_read = pread(fd, buf, this_len, offset);
        if (bytes_read != this_len) {
            std::cerr << "pread failed: read " << bytes_read << " bytes (expected " << this_len << ")\n";
            exit(-1);
        }

        #pragma omp parallel for num_threads(ntds)
        for (uint64_t id = start_id; id < std::min((uint64_t) disk_npts, (uint64_t) (start_id + SECTORS_PER_MERGE * nnodes_per_sector)); id++) {
            
            uint64_t page_offset = (((uint64_t)(id)) / nnodes_per_sector + 1) * SECTOR_LEN; 
            char *node_buf = (char *)(buf + page_offset - offset) + (((uint64_t)id) % nnodes_per_sector) * max_node_len; 
            unsigned* nhood_ptr = (unsigned *)((char *)node_buf + ndims * sizeof(T)); 
            DiskNode<T> disk_node(id, nhood_ptr);

            if (disk_node.nnbrs == 0) {
                std::cerr << "nbrs is 0\n";
                std::cerr << "id = " << id << "\n";
                // exit(-1);
            }
            if (disk_node.nnbrs > range) {
                std::cerr << "id " << disk_node.id << "'s nnbrs is " << disk_node.nnbrs << ", but is not in [0, " << range << "]\n";
                exit(-1);
            }
            
            for (uint32_t i = 0; i < disk_node.nnbrs; i ++ ) {
                if (disk_node.nbrs[i] >= num_points || disk_node.nbrs[i] < 0) {
                    std::cerr << "nbrs[" << i << "] is " << disk_node.nbrs[i] << ", but is not in [0, " << num_points << ")\n";
                    exit(-1);
                }
                std::lock_guard<std::mutex> lock(mutexs[disk_node.nbrs[i]]);
                degrees[disk_node.nbrs[i]]++;
            }
        }

        std::cerr << start_id << " / " << num_points << " nodes processed.\n";
        offset += this_len;
        start_id += SECTORS_PER_MERGE * nnodes_per_sector;
    }

    LOG(INFO) << "Degree counting finished, begin alloc memory for mem_in_graph";

    
    // USE CSR
    std::vector<uint64_t> offsets(num_points + 1); 
    std::vector<uint64_t> idxs(num_points + 1, 0); 
    std::vector<uint32_t> edges; 

    offsets[0] = 0;
    for (uint64_t i = 0; i < num_points; ++i)
        offsets[i + 1] = offsets[i] + degrees[i];

    uint64_t total_edges = offsets[num_points];
    edges.resize(total_edges);

    uint32_t max_degree_id = 0;
    for (size_t id = 0; id < num_points; ++id) {
        if (degrees[id] > degrees[max_degree_id]) {
            max_degree_id = id;
        }
    }
    std::cout << "Max degree id: " << max_degree_id << ", degree: " << degrees[max_degree_id] << std::endl;
    auto topk_degree = get_topk_values_heap(degrees, 20);
    size_t s = 0;
    for (size_t i = 0; i < topk_degree.size(); ++i) {
        std::cout << "Top " << i + 1 << " degree: " << topk_degree[i] << std::endl;
        s += topk_degree[i];
    }
    std::cout << "Top 10000 sum degree: " << s << std::endl;

    std::cout << "CSR graph allocated. Total edges = "
                << total_edges << " (" << (total_edges * sizeof(uint32_t) / (1024.0 * 1024 * 1024))
                << " GB)" << std::endl;

    offset = SECTOR_LEN;
    start_id = 0;
    LOG(INFO) << "Begin read disk again";
    while (offset < sz && start_id < disk_npts) {
        uint64_t this_len = std::min(CHUNK, sz - offset);

        ssize_t bytes_read = pread(fd, buf, this_len, offset);
        if (bytes_read != this_len) {
            std::cerr << "pread failed: read " << bytes_read << " bytes (expected " << this_len << ")\n";
            exit(-1);
        }

        #pragma omp parallel for num_threads(ntds)
        for (uint64_t id = start_id; id < std::min((uint64_t) disk_npts, (uint64_t) (start_id + SECTORS_PER_MERGE * nnodes_per_sector)); id++) {
            
            uint64_t page_offset = (((uint64_t)(id)) / nnodes_per_sector + 1) * SECTOR_LEN;
            char *node_buf = (char *)(buf + page_offset - offset) + (((uint64_t)id) % nnodes_per_sector) * max_node_len;
            unsigned* nhood_ptr = (unsigned *)((char *)node_buf + ndims * sizeof(T));
            DiskNode<T> disk_node(id, nhood_ptr);

            if (disk_node.nnbrs > range) {
                std::cerr << "nbrs[" << disk_node.nnbrs << "] is " << disk_node.nnbrs << ", but is not in [0, " << range << ")\n";
                exit(-1);
            }
            for (uint32_t i = 0; i < disk_node.nnbrs; i ++ ) {
                if (disk_node.nbrs[i] >= num_points || disk_node.nbrs[i] < 0) {
                    std::cerr << "nbrs[" << i << "] is " << disk_node.nbrs[i] << ", but is not in [0, " << num_points << ")\n";
                    exit(-1);
                }
                std::lock_guard<std::mutex> lock(mutexs[disk_node.nbrs[i]]);
                edges[offsets[disk_node.nbrs[i]] + idxs[disk_node.nbrs[i]]++] = disk_node.id;
            }
        }

        std::cerr << start_id << " / " << num_points << " nodes processed.\n";
        offset += this_len;
        start_id += SECTORS_PER_MERGE * nnodes_per_sector;
    }

    LOG(INFO) << "Begin add edge to disk_in_graph";

    // 计算总边数（基于 degrees 数组）
    uint64_t total_edges_from_degrees = 0;
    for (uint32_t i = 0; i < num_points; ++i) {
        total_edges_from_degrees += degrees[i];
    }

    LOG(INFO) << "Total edges from degrees: " << total_edges_from_degrees;
    // 计算总边数（基于 offsets 数组，offsets 最后一个元素应为总偏移量）
    uint64_t total_edges_from_offsets = offsets[num_points];  // 假设 offsets 长度为 num_points + 1

    // 校验 degrees 和 offsets 计算的总边数是否一致
    if (total_edges_from_degrees != total_edges_from_offsets) {
        std::cerr << "Total edges mismatch: degrees sum=" << total_edges_from_degrees 
                    << ", offsets total=" << total_edges_from_offsets << "\n";
        free(buf);
        close(fd);
        exit(-1);
    }

    #pragma omp parallel for num_threads(ntds)
    for (uint32_t i = 0; i < num_points; ++i) {
        if (degrees[i] != idxs[i]) {
            std::cerr << "Node " << i << " degrees mismatch: degrees[i]=" << degrees[i] 
                    << ", idxs[i]=" << idxs[i] << "\n";
            free(buf);
            close(fd);
            exit(-1);
        }
    }
    LOG(INFO) << "Degrees and idxs validation passed.";

    #pragma omp parallel for num_threads(ntds)
    for (uint32_t i = 0; i < num_points; ++i) {
        if (i == max_degree_id) continue;
        for (uint32_t j = 0; j < degrees[i]; ++j) {
            uint64_t edge_idx = offsets[i] + j;
            // 校验 edges 索引是否合法
            assert(edge_idx < total_edges_from_offsets && "edge index out of edges array range");
            uint32_t neighbor = edges[edge_idx];
            // 校验邻居节点 ID 是否合法
            assert(neighbor < num_points && neighbor >= 0 && "neighbor ID is invalid");
            disk_in_graph->add_edge(i, neighbor);
        }
    }
    LOG(INFO) << "Add edge to disk_in_graph finished";

    // 补充：释放资源（避免内存泄漏和文件句柄泄漏）
    delete disk_in_graph;
    free(buf);
    close(fd);

    // 补充：返回值（函数声明为 int，需返回有效值）
    return 0;
}
    
int main(int argc, char* argv[]){
    if (argc!= 2) {
        std::cerr << "Usage: " << argv[0] << " <input_file> \n";
        return 1;
    }
    
    std::string index_path  = argv[1];

    CreatReverseGraph<float>(index_path);

    {
        std::ifstream reader(index_path, std::ios::binary);
        if (!reader) {
            std::cerr << "Failed to open: " << index_path << std::endl;
            return 1;
        }

        int meta_npts, meta_ndims;
        reader.read((char *)&meta_npts, sizeof(int));
        reader.read((char *)&meta_ndims, sizeof(int));

        std::vector<uint64_t> meta(meta_npts);
        for (auto &x : meta) {
            reader.read((char *)&x, sizeof(uint64_t));
        }

        uint64_t num_points = meta[0];
        uint64_t ndims = meta[1];
        uint64_t max_node_len = meta[3];
        uint64_t nnodes_per_sector = meta[4];

        uint64_t range = (max_node_len - ndims * sizeof(float) - sizeof(uint32_t)) / sizeof(uint32_t);

        reader.close();
        std::cerr << "num_points = " << num_points << "\n"
                << "ndims = " << ndims << "\n"
                << "max_node_len = " << max_node_len << "\n"
                << "nnodes_per_sector = " << nnodes_per_sector << "\n"
                << "range = " << range << "\n";

        size_t file_size;
        if (num_points <= 1e6 + 5) {
            file_size = 1ULL << 30;
        } else if (num_points <= 1e8 + 5) {
            file_size = 1ULL << 36;
        } else if (num_points <= 1e9 + 5) {
            file_size = 1ULL << 38;
        }
        
#ifdef USE_LIVE_GRAPH
        livegraph::BlockGraph * disk_in_graph = new livegraph::BlockGraph (
                    index_path + ".in_graph_lg",
                    (size_t)(num_points * 1.25), 32, 4, true,
                    file_size);

#else
        DynGraphDisk::BlockGraph * disk_in_graph = new DynGraphDisk::BlockGraph (
            index_path + ".in_graph",
            num_points * 1.25, 32, 4, 20, true,
            file_size);
#endif
        LOG(INFO) << "Reverse graph loaded";
    }
    return 0;
}