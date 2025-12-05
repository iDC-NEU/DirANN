#include "aligned_file_reader.h"
#include "linux_aligned_file_reader.h"
#include "ssd_index.h"
#include <malloc.h>

#include <omp.h>
#include <cmath>
#include "parameters.h"
#include "query_buf.h"
#include "timer.h"
#include "utils.h"

#include <unistd.h>
#include <sys/syscall.h>
#include "tsl/robin_set.h"

#include "global_stats.h"

namespace dirann {
  template<typename T>
  DiskNode<T>::DiskNode(uint32_t id, T *coords, uint32_t *nhood) : id(id) {
    this->coords = coords;
    this->nnbrs = *nhood;
    this->nbrs = nhood + 1;
  }
  template<typename T>
  DiskNode<T>::DiskNode(uint32_t id, T *coords) : id(id) {
    this->coords = coords;
  }
  template<typename T>
  DiskNode<T>::DiskNode(uint32_t id, uint32_t *nhood) : id(id) {
    this->nnbrs = *nhood;
    this->nbrs = nhood + 1;
  }

  // structs for DiskNode
  template struct DiskNode<float>;
  template struct DiskNode<uint8_t>;
  template struct DiskNode<int8_t>;

  template<typename T, typename TagT>
  SSDIndex<T, TagT>::SSDIndex(dirann::Metric m, std::shared_ptr<AlignedFileReader> &fileReader, bool single_file_index,
                              bool tags, Parameters *params)
      : reader(fileReader), data_is_normalized(false), enable_tags(tags) {
    if (m == dirann::Metric::COSINE) {
      if (std::is_floating_point<T>::value) {
        LOG(INFO) << "Cosine metric chosen for (normalized) float data."
                     "Changing distance to L2 to boost accuracy.";
        m = dirann::Metric::L2;
        data_is_normalized = true;

      } else {
        LOG(ERROR) << "WARNING: Cannot normalize integral data types."
                   << " This may result in erroneous results or poor recall."
                   << " Consider using L2 distance with integral data types.";
      }
    }

    if (0)
    {
      
      std::string shuffled_map_path = "/data/dataset_vec_public/sift100w/sift100w_base.fbin800000.shuffled_map";
      int dim;
      std::ifstream map_reader(shuffled_map_path, std::ios::binary);
      if (map_reader.is_open()) {
          // 先读取维度大小，再读取映射数组
          map_reader.read((char*)&dim, sizeof(int));
          dim_shuffler.resize(dim);
          map_reader.read((char*)this->dim_shuffler.data(), dim * sizeof(int));
          std::cout << "维度映射已加载到: " << shuffled_map_path << std::endl;
          std::cerr << "dim_shuffler size: " << this->dim_shuffler.size() << std::endl;
          for(auto i : this->dim_shuffler) {
            std::cerr << i << " ";
          }
          std::cerr << std::endl;
      } else {
          std::cerr << "无法打开映射文件: " << shuffled_map_path << std::endl;
      }
    }

    if (0)
    {
        std::string ortho_matrix_path = "/data/dataset_vec_public/sift100w/sift100w_base.fbin800000.shuffled_matrix";
        int dim;
        std::ifstream matrix_reader(ortho_matrix_path, std::ios::binary);
        
        if (matrix_reader.is_open()) {
            // 步骤1：读取矩阵维度
            matrix_reader.read((char*)&dim, sizeof(int));
            
            // 步骤2：调整vector大小并读取矩阵数据
            ortho_matrix.resize(dim * dim);  // 分配dim×dim个float的空间
            matrix_reader.read((char*)ortho_matrix.data(), dim * dim * sizeof(float));
            
            std::cout << "正交矩阵已加载: " << ortho_matrix_path 
                      << " (" << dim << "x" << dim << ")" << std::endl;
        } else {
            std::cerr << "无法打开正交矩阵文件: " << ortho_matrix_path << std::endl;
            // 错误处理：根据业务需求处理（如返回、抛出异常等）
        }
    }

    this->dist_cmp.reset(dirann::get_distance_function<T>(m));

    // this->pq_reader = new LinuxAlignedFileReader();
    if (params != nullptr) {
      this->beam_width = params->Get<uint32_t>("beamwidth");
      this->l_index = params->Get<uint32_t>("L");
      this->range = params->Get<uint32_t>("R");
      this->maxc = params->Get<uint32_t>("C");
      this->alpha = params->Get<float>("alpha");
      LOG(INFO) << "Beamwidth: " << this->beam_width << ", L: " << this->l_index << ", R: " << this->range
                << ", C: " << this->maxc;
    }
  }

  template<typename T, typename TagT>
  SSDIndex<T, TagT>::~SSDIndex() {
    LOG(INFO) << "Lock table size: " << this->idx_lock_table.size();
    LOG(INFO) << "Page cache size: " << v2::cache.cache.size();

    if (load_flag) {
      this->destroy_thread_data();
      reader->close();
    }

    if (medoids != nullptr) {
      delete[] medoids;
    }
  }

  template<typename T, typename TagT>
  void SSDIndex<T, TagT>::init_buffers(_u64 n_threads) {
    _u64 n_buffers = n_threads * 2;
    LOG(INFO) << "Init buffers for " << n_threads << " threads, setup " << n_buffers << " buffers.";
    for (uint64_t i = 0; i < n_buffers; i++) {
      QueryBuffer<T> *data = new QueryBuffer<T>();
      this->init_query_buf(*data);
      // std::cerr << "init query buf " << (void *)data << "\n";
      this->thread_data_bufs.push_back(data);
      this->thread_data_queue.push(data);
    }

    for (uint64_t i = 0; i < n_buffers; ++i) {
      uint8_t *thread_pq_buf;
      dirann::alloc_aligned((void **) &thread_pq_buf, 16ul << 20, 256);
      thread_pq_bufs.push_back(thread_pq_buf);
    }

#ifndef READ_ONLY_TESTS
    // background thread.
    LOG(INFO) << "Setup " << kBgIOThreads << " background I/O threads for insert...";
    for (int i = 0; i < kBgIOThreads; ++i) {
      bg_io_thread_[i] = new std::thread(&SSDIndex<T, TagT>::bg_io_thread, this);
      bg_io_thread_[i]->detach();
    }
#endif
    load_flag = true;
  }

  template<typename T, typename TagT>
  void SSDIndex<T, TagT>::destroy_thread_data() {
    // TODO(gh): destruct thread_queue and other readers.
    for (auto &buf : this->thread_data_bufs) {
      dirann::aligned_free((void *) buf->coord_scratch);
      dirann::aligned_free((void *) buf->sector_scratch);
      dirann::aligned_free((void *) buf->aligned_pq_coord_scratch);
      dirann::aligned_free((void *) buf->aligned_pqtable_dist_scratch);
      dirann::aligned_free((void *) buf->aligned_dist_scratch);
      dirann::aligned_free((void *) buf->aligned_query_T);
      dirann::aligned_free((void *) buf->aligned_query_T_2);
      dirann::aligned_free((void *) buf->update_buf);
    }
  }

  template<typename T, typename TagT>
  void SSDIndex<T, TagT>::load_mem_index(Metric metric, const size_t query_dim, const std::string &mem_index_path) {
    if (mem_index_path.empty()) {
      LOG(ERROR) << "mem_index_path is needed";
      exit(1);
    }
    mem_index_ = std::make_unique<dirann::Index<T, uint32_t>>(metric, query_dim, 0, false, false, true);
    mem_index_->load(mem_index_path.c_str());
  }

  template<typename T, typename TagT>
  int SSDIndex<T, TagT>::load(const char *index_prefix, _u32 num_threads, bool new_index_format, bool use_page_search) {
    std::string pq_table_bin, pq_compressed_vectors, disk_index_file, centroids_file;

    int strategy = ((LinuxAlignedFileReader *)this->reader.get())->strategy;
    int use_double_pq = (strategy >> 2) & 0x1;
    int use_triple_pq = (strategy >> 7) & 0x1;
    std::string iprefix = std::string(index_prefix);
    
    pq_table_bin = iprefix + "_pq_pivots.bin";
    pq_compressed_vectors = iprefix + "_pq_compressed.bin";
    
    disk_index_file = iprefix + "_disk.index";
    this->_disk_index_file = disk_index_file;
    centroids_file = disk_index_file + "_centroids.bin";

    std::ifstream index_metadata(disk_index_file, std::ios::binary);

    size_t tags_offset = 0;
    size_t pq_pivots_offset = 0;
    size_t pq_vectors_offset = 0;
    _u64 disk_nnodes;
    _u64 disk_ndims;
    size_t medoid_id_on_file;
    _u64 file_frozen_id;

    if (new_index_format) {
      _u32 nr, nc;

      READ_U32(index_metadata, nr);
      READ_U32(index_metadata, nc);

      READ_U64(index_metadata, disk_nnodes);
      READ_U64(index_metadata, disk_ndims);

      READ_U64(index_metadata, medoid_id_on_file);
      READ_U64(index_metadata, max_node_len);
      READ_U64(index_metadata, nnodes_per_sector);
      data_dim = disk_ndims;
      max_degree = ((max_node_len - data_dim * sizeof(T)) / sizeof(unsigned)) - 1;
      
      coord_len = data_dim * sizeof(T);
      topo_len = (max_degree + 1) * sizeof(_u32);
      // topo_len = (std::ceil(max_degree * GRAPH_SLACK_FACTOR) + 1) * sizeof(_u32);
      ncoord_per_sector = SECTOR_LEN / coord_len;
      ntopo_per_sector = SECTOR_LEN / topo_len;

      if (max_degree != this->range) {
        LOG(ERROR) << "Range mismatch: " << max_degree << " vs " << this->range << ", setting range to " << max_degree;
        this->range = max_degree;
      }

      LOG(INFO) << "Meta-data: # nodes per sector: " << nnodes_per_sector << ", max node len (bytes): " << max_node_len
                << ", max node degree: " << max_degree << ", npts: " << nr << ", dim: " << nc
                << " disk_nnodes: " << disk_nnodes << " disk_ndims: " << disk_ndims;

      if (nnodes_per_sector > this->kMaxElemInAPage) {
        LOG(ERROR) << "nnodes_per_sector: " << nnodes_per_sector << " is greater than " << this->kMaxElemInAPage
                   << ". Please recompile with a higher value of kMaxElemInAPage.";
        return -1;
      }

      READ_U64(index_metadata, this->num_frozen_points);
      READ_U64(index_metadata, file_frozen_id);
      if (this->num_frozen_points == 1) {
        this->frozen_location = file_frozen_id;
        // if (this->num_frozen_points == 1) {
        LOG(INFO) << " Detected frozen point in index at location " << this->frozen_location
                  << ". Will not output it at search time.";
      }
      READ_U64(index_metadata, tags_offset);
      READ_U64(index_metadata, pq_pivots_offset);
      READ_U64(index_metadata, pq_vectors_offset);

      LOG(INFO) << "Tags offset: " << tags_offset << " PQ Pivots offset: " << pq_pivots_offset
                << " PQ Vectors offset: " << pq_vectors_offset;
    } else {  // old index file format
      size_t actual_index_size = get_file_size(disk_index_file);
      size_t expected_file_size;
      READ_U64(index_metadata, expected_file_size);
      if (actual_index_size != expected_file_size) {
        LOG(INFO) << "File size mismatch for " << disk_index_file << " (size: " << actual_index_size << ")"
                  << " with meta-data size: " << expected_file_size;
        return -1;
      }

      READ_U64(index_metadata, disk_nnodes);
      READ_U64(index_metadata, medoid_id_on_file);
      READ_U64(index_metadata, max_node_len);
      READ_U64(index_metadata, nnodes_per_sector);
      max_degree = ((max_node_len - data_dim * sizeof(T)) / sizeof(unsigned)) - 1;

      LOG(INFO) << "Disk-Index File Meta-data: # nodes per sector: " << nnodes_per_sector;
      LOG(INFO) << ", max node len (bytes): " << max_node_len;
      LOG(INFO) << ", max node degree: " << max_degree;
    }

    size_per_io = SECTOR_LEN * (nnodes_per_sector > 0 ? 1 : DIV_ROUND_UP(max_node_len, SECTOR_LEN));
    LOG(INFO) << "Size per IO: " << size_per_io;

    index_metadata.close();

    pq_pivots_offset = 0;
    pq_vectors_offset = 0;

    LOG(INFO) << "After single file index check, Tags offset: " << tags_offset
              << " PQ Pivots offset: " << pq_pivots_offset << " PQ Vectors offset: " << pq_vectors_offset;

    size_t npts_u64, nchunks_u64;
    dirann::load_bin<_u8>(pq_compressed_vectors, data, npts_u64, nchunks_u64, pq_vectors_offset);
    this->n_chunks = nchunks_u64;

    this->num_points = this->init_num_pts = npts_u64;

    this->cur_id = this->num_points;

    LOG(INFO) << "Load compressed vectors from file: " << pq_compressed_vectors << " offset: " << pq_vectors_offset
              << " num points: " << npts_u64 << " n_chunks: " << nchunks_u64;

    pq_table.load_pq_centroid_bin(pq_table_bin.c_str(), this->n_chunks, pq_pivots_offset);

    if (disk_nnodes != num_points) {
      LOG(INFO) << "Mismatch in #points for compressed data file and disk "
                   "index file: "
                << disk_nnodes << " vs " << num_points;
      return -1;
    }

    this->data_dim = pq_table.get_dim();
    this->aligned_dim = ROUND_UP(this->data_dim, 8);

    LOG(INFO) << "Loaded PQ centroids and in-memory compressed vectors. #points: " << num_points
              << " #dim: " << data_dim << " #aligned_dim: " << aligned_dim << " #chunks: " << n_chunks;
              
    if(ncoord_per_sector <= 1){
      ((LinuxAlignedFileReader *)this->reader.get())->strategy = strategy & ~(0x1 << 3);
    }

    // read index metadata
    // open AlignedFileReader handle to index_file
    std::string index_fname(disk_index_file);
    reader->open(index_fname, true, false);
    this->init_buffers(num_threads);
    this->max_nthreads = num_threads;

    // load page layout and set cur_loc
    this->use_page_search_ = use_page_search;
    this->load_page_layout(index_prefix, nnodes_per_sector, num_points);

    // load tags
    if (this->enable_tags) {
      std::string tag_file = disk_index_file + ".tags";
      LOG(INFO) << "Loading tags from " << tag_file;
      this->load_tags(tag_file);
    }

    num_medoids = 1;
    medoids = new uint32_t[1];
    medoids[0] = (_u32) (medoid_id_on_file);
    LOG(INFO) << "Medoid id is " << medoids[0];
    
    #ifdef USE_NHOOD_CACHE
    // cache bfs levels
    std::vector<uint32_t> node_list;

    int topo_len = (this->range + 1) * sizeof(uint32_t);
    int coord_len = this->data_dim * sizeof(T);
    double cache_ratio = 0.01 * ((double)coord_len + topo_len) / topo_len;
    int cache_size = this->num_points * cache_ratio;
    LOG(INFO) << "topo_len is " << topo_len;
    LOG(INFO) << "coord_len is " << coord_len;
    LOG(INFO) << "cache_ratio is " << cache_ratio;
    LOG(INFO) << "cache_size is " << cache_size;
    cache_bfs_levels(cache_size, node_list);
    load_cache_list(node_list);
    node_list.clear();
    node_list.shrink_to_fit();
    #endif

    #ifdef REVERSE_GRAPH
    LOG(INFO) << "Loading reverse graph";
      
    size_t file_size;
    if (num_points <= 1e6 + 5) {
        file_size = 1ULL << 30;
    } else if (num_points <= 1e8 + 5) {
        file_size = 1ULL << 36;
    } else if (num_points <= 1e9 + 5) {
        file_size = 1ULL << 38;
    }
    
      #ifdef USE_LIVE_GRAPH
        this->disk_in_graph = new livegraph::BlockGraph (
            disk_index_file + ".in_graph_lg",
            this->num_points * 1.25, 32, 4, true,
            file_size);
      #else
        this->disk_in_graph = new DynGraphDisk::BlockGraph (
            disk_index_file + ".in_graph",
            this->num_points * 1.25, 32, 4, 20, true,
            file_size);
      #endif  

    std::vector<int> in_nbrs;
    this->disk_in_graph->get_edges(medoids[0], in_nbrs);
    // LOG(INFO) << "Medoids in graph has " << in_nbrs.size() << " neighbors";
    LOG(INFO) << "Reverse graph loaded";
    // exit(-1);
    return 0;
    #endif
    LOG(INFO) << "SSDIndex loaded successfully.";
    return 0;
  }

  template<typename T, typename TagT>
  _u64 SSDIndex<T, TagT>::return_nd() {
    return this->num_points;
  }

  template<typename T, typename TagT>
  void SSDIndex<T, TagT>::compute_pq_dists(const _u32 src, const _u32 *ids, float *fp_dists, const _u32 count,
                                           uint8_t *aligned_scratch) {
    const _u8 *src_ptr = this->data.data() + (this->n_chunks * src);
    if (unlikely(aligned_scratch == nullptr || count >= 32768)) {
      LOG(ERROR) << "Aligned scratch buffer is null or count is too large: " << count
                 << ". This will lead to memory issues.";
      crash();
    }
#ifdef COLLECT_INFO
    if (gs != nullptr) {
      gs->pq_calc_count += count;
    }
#endif
    // aggregate PQ coords into scratch
    ::aggregate_coords(ids, count, this->data.data(), this->n_chunks, aligned_scratch);
    // compute distances
    this->pq_table.compute_distances_alltoall(src_ptr, aligned_scratch, fp_dists, count);
  }

  template<typename T, typename TagT>
  std::vector<_u8> SSDIndex<T, TagT>::deflate_vector(const T *vec) {
    std::vector<_u8> pq_coords(this->n_chunks);
    std::vector<float> fp_vec(this->data_dim);
    for (uint32_t i = 0; i < this->data_dim; i++) {
      fp_vec[i] = (float) vec[i];
    }
    this->pq_table.deflate_vec(fp_vec.data(), pq_coords.data());
    return pq_coords;
  }

  template<typename T, typename TagT>
  std::vector<_u8> SSDIndex<T, TagT>::deflate_vector_2(const T *vec) {
    std::vector<_u8> pq_coords(this->n_chunks);
    std::vector<float> fp_vec(this->data_dim);
    for (uint32_t i = 0; i < this->data_dim; i++) {
      fp_vec[i] = (float) vec[i];
    }
    this->pq_table_2.deflate_vec(fp_vec.data(), pq_coords.data());
    return pq_coords;
  }

  template<>
  std::vector<_u8> SSDIndex<float>::deflate_vector(const float *vec) {
    std::vector<_u8> pq_coords(this->n_chunks);
    this->pq_table.deflate_vec(vec, pq_coords.data());
    return pq_coords;
  }

  template<typename T, typename TagT>
  void SSDIndex<T, TagT>::load_tags(const std::string &tag_file_name, size_t offset) {
    size_t tag_num, tag_dim;
    std::vector<TagT> tag_v;
    this->tags.clear();

    if (!file_exists(tag_file_name)) {
      LOG(INFO) << "Tags file not found. Using equal mapping";
      // Equal mapping are by default eliminated in tags map.
    } else {
      LOG(INFO) << "Load tags from existing file: " << tag_file_name;
      dirann::load_bin<TagT>(tag_file_name, tag_v, tag_num, tag_dim, offset);
      tags.reserve(tag_v.size());
      id2loc_.reserve(tag_v.size());

#pragma omp parallel for num_threads(max_nthreads)
      for (size_t i = 0; i < tag_num; ++i) {
        tags.insert_or_assign(i, tag_v[i]);
      }
    }
    LOG(INFO) << "Loaded " << tags.size() << " tags";
  }

  template<typename T, typename TagT>
  int SSDIndex<T, TagT>::get_vector_by_id(const uint32_t &id, T *vector_coords) {
    if (!enable_tags) {
      LOG(INFO) << "Tags are disabled, cannot retrieve vector";
      return -1;
    }
    uint32_t pos = id;
    size_t num_sectors = node_sector_no(pos);
    std::ifstream disk_reader(_disk_index_file.c_str(), std::ios::binary);
    std::unique_ptr<char[]> sector_buf = std::make_unique<char[]>(size_per_io);
    disk_reader.seekg(SECTOR_LEN * num_sectors, std::ios::beg);
    disk_reader.read(sector_buf.get(), size_per_io);
    char *node_coords = (offset_to_node(sector_buf.get(), pos));
    memcpy((void *) vector_coords, (void *) node_coords, data_dim * sizeof(T));
    return 0;
  }
  
  template <typename T, typename TagT>
  void SSDIndex<T, TagT>::cache_bfs_levels(_u64 num_nodes_to_cache,
                                               std::vector<uint32_t> &node_list) {
    // random_shuffle() is deprecated.
    std::random_device rng;
    std::mt19937 urng(rng());
  
    node_list.clear();
  
    // Do not cache more than 10% of the nodes in the index
    _u64 tenp_nodes = (_u64)(std::round(this->num_points * 0.1));
    if (num_nodes_to_cache > tenp_nodes) {
      std::cout << "Reducing nodes to cache from: " << num_nodes_to_cache
                    << " to: " << tenp_nodes
                    << "(10 percent of total nodes:" << this->num_points << ")"
                    << std::endl;
      num_nodes_to_cache = tenp_nodes == 0 ? 1 : tenp_nodes;
    }
    std::cout << "Caching " << num_nodes_to_cache << "..." << std::endl;
  
    void *ctx = reader->get_ctx();
  
    std::unique_ptr<tsl::robin_set<unsigned> > cur_level, prev_level;
    cur_level = std::make_unique<tsl::robin_set<unsigned> >();
    prev_level = std::make_unique<tsl::robin_set<unsigned> >();
  
    std::unordered_set<uint32_t> node_set;
    for (_u64 miter = 0; miter < num_medoids; miter++) {
      cur_level->insert(medoids[miter]);
    }

    Timer timer;
    _u64 lvl = 1;
    uint64_t prev_node_list_size = 0;
    while ((node_list.size() + cur_level->size() < num_nodes_to_cache) &&
           cur_level->size() != 0) {
      // swap prev_level and cur_level
      std::swap(prev_level, cur_level);
      // clear cur_level
      cur_level->clear();
  
      std::vector<unsigned> nodes_to_expand;
  
      for (const unsigned &id : *prev_level) {
        if (node_set.find(id) != node_set.end()) {
          continue;
        }
        node_set.insert(id);
        node_list.push_back(id);
        nodes_to_expand.push_back(id);
      }
  
      // random_shuffle() is deprecated.
      std::shuffle(nodes_to_expand.begin(), nodes_to_expand.end(), urng);
  
      std::cout << "Level: " << lvl << std::flush;
      bool finish_flag = false;
  
      uint64_t _BLOCK_SIZE = 1024;
      uint64_t nblocks = DIV_ROUND_UP(nodes_to_expand.size(), _BLOCK_SIZE);
      for (size_t block = 0; block < nblocks && !finish_flag; block++) {
        std::cout << "." << std::flush;
        size_t start = block * _BLOCK_SIZE;
        size_t end = (std::min)((block + 1) * _BLOCK_SIZE, nodes_to_expand.size());
        std::vector<IORequest> read_reqs;
        std::vector<std::pair<_u32, char *> > nhoods;
        // Timer timer;
        for (size_t cur_pt = start; cur_pt < end; cur_pt++) {
          char *buf = nullptr;
          alloc_aligned((void **)&buf, SECTOR_LEN, SECTOR_LEN);
          nhoods.push_back(std::make_pair(nodes_to_expand[cur_pt], buf));
          IORequest read;
          read.len = SECTOR_LEN;
          read.buf = buf;
          uint32_t id = nodes_to_expand[cur_pt];
          read.offset = loc_sector_no(id) * SECTOR_LEN;
          read_reqs.push_back(read);
        }
        // std::cout << " read time1: " << timer.elapsed() / 1e6 << " s" << std::endl;
        // issue read requests
        reader->read(read_reqs, ctx);
        // process each nhood buf
        // timer.reset();
        for (auto &nhood : nhoods) {
          // insert node coord into coord_cache
          char *node_buf = offset_to_loc(nhood.second, nhood.first);
          unsigned *node_nhood = offset_to_node_nhood(node_buf);
          _u64 nnbrs = (_u64) * node_nhood;
          unsigned *nbrs = node_nhood + 1;
          // explore next level
          for (_u64 j = 0; j < nnbrs && !finish_flag; j++) {
            if (node_set.find(nbrs[j]) == node_set.end()) {
              cur_level->insert(nbrs[j]);
            }
            if (cur_level->size() + node_list.size() >= num_nodes_to_cache) {
              finish_flag = true;
            }
          }
          aligned_free(nhood.second);
        }
        // std::cout << " read time2: " << timer.elapsed() / 1e6 << " s" << std::endl;
      }
  
      std::cout << ". #nodes: " << node_list.size() - prev_node_list_size
                    << ", #nodes thus far: " << node_list.size() << std::endl;
      prev_node_list_size = node_list.size();
      lvl++;
    }
  
    std::vector<uint32_t> cur_level_node_list;
    for (const unsigned &p : *cur_level)
      cur_level_node_list.push_back(p);
  
    // random_shuffle() is deprecated
    std::shuffle(cur_level_node_list.begin(), cur_level_node_list.end(), urng);
    size_t residual = num_nodes_to_cache - node_list.size();
  
    for (size_t i = 0; i < (std::min)(residual, cur_level_node_list.size()); i++)
      node_list.push_back(cur_level_node_list[i]);
  
    std::cout << "Level: " << lvl << std::flush;
    std::cout << ". #nodes: " << node_list.size() - prev_node_list_size
                  << ", #nodes thus far: " << node_list.size() << std::endl;

    LOG(INFO) << "Cache node cost " << timer.elapsed() / 1e6 << " s";
  }
  
  template <typename T, typename TagT>
  void SSDIndex<T, TagT>::load_cache_list(std::vector<uint32_t> &node_list) {
    //    std::cout << "Loading the cache list into memory.." << std::flush;
    _u64 num_cached_nodes = node_list.size();

    void *ctx = reader->get_ctx();

    nhood_cache_buf = new unsigned[num_cached_nodes * (max_degree + 1)];
    memset(nhood_cache_buf, 0, num_cached_nodes * (max_degree + 1));

    // _u64 coord_cache_buf_len = num_cached_nodes * aligned_dim;
    // dirann::alloc_aligned((void **)&coord_cache_buf,
    //                       coord_cache_buf_len * sizeof(T), 8 * sizeof(T));
    // memset(coord_cache_buf, 0, coord_cache_buf_len * sizeof(T));

    size_t _BLOCK_SIZE = 1024;
    size_t num_blocks = DIV_ROUND_UP(num_cached_nodes, _BLOCK_SIZE);

    for (_u64 block = 0; block < num_blocks; block++) {
      _u64 start_idx = block * _BLOCK_SIZE;
      _u64 end_idx = (std::min)(num_cached_nodes, (block + 1) * _BLOCK_SIZE);
      std::vector<IORequest> read_reqs;
      std::vector<std::pair<_u32, char *> > nhoods;
      for (_u64 node_idx = start_idx; node_idx < end_idx; node_idx++) {
        IORequest read;
        char *buf = nullptr;
        alloc_aligned((void **)&buf, SECTOR_LEN, SECTOR_LEN);
        nhoods.push_back(std::make_pair(node_list[node_idx], buf));
        read.len = SECTOR_LEN;
        read.buf = buf;
        uint32_t id = node_list[node_idx];
        read.offset = loc_sector_no(id) * SECTOR_LEN;
        read_reqs.push_back(read);
      }

      reader->read(read_reqs, ctx);

      _u64 node_idx = start_idx;
      for (auto &nhood : nhoods) {
        char *node_buf = offset_to_loc(nhood.second, nhood.first);
        // T *node_coords = offset_to_node_coords(node_buf);
        // T *cached_coords = coord_cache_buf + node_idx * aligned_dim;
        // memcpy(cached_coords, node_coords, data_dim * sizeof(T));
        // coord_cache.insert(std::make_pair(nhood.first, cached_coords));

        // insert node nhood into nhood_cache
        unsigned *node_nhood = offset_to_node_nhood(node_buf);
        auto nnbrs = *node_nhood;
        unsigned *nbrs = node_nhood + 1;
        // std::cout << "CACHE: nnbrs = " << nnbrs << "\n";
        _u32 * buf = nhood_cache_buf + node_idx * (max_degree + 1);
        memcpy(buf, node_nhood, (nnbrs + 1) * sizeof(unsigned));
        nhood_cache.insert(std::make_pair(nhood.first, buf));
        aligned_free(nhood.second);
        node_idx++;
      }
    }
    
    std::cout << "..done." << std::endl;
  }


  template class SSDIndex<float>;
  template class SSDIndex<_s8>;
  template class SSDIndex<_u8>;
}  // namespace dirann
