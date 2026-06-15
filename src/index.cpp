#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <fstream>
#include <iostream>
#include <limits>
#include <thread>
#include <omp.h>
#include <shared_mutex>
#include <sstream>
#include <string>
#include "utils/percentile_stats.h"
#include "utils/tsl/robin_set.h"
#include <unordered_map>
#include <map>
#include <queue>

#include <fcntl.h>
#include <sys/stat.h>
#include <time.h>

#include "index.h"
#include "utils/timer.h"
#include "utils.h"
#include "ssd_index_defs.h"
#include "utils/lock_table.h"
#include <unordered_set>

// #define USE_FLAT_GRAPH
// #define USE_DATA_PMEM
// #define USE_DATA_PMEM_NUMA
// #define USE_GRAPH_PMEM
#define PMEM_STACK_FACTOR 1.5
#define N_DELETE_THREADS 20
#define ANGLE_DEGREES 75

#if defined(USE_DATA_PMEM) && defined(USE_GRAPH_PMEM)
#include <libpmem.h>
    #ifdef USE_DATA_PMEM_NUMA
    #include <numa.h>
    #include <numaif.h>
    #endif
#endif

// #define DISKANN_WITH_INCREMENTAL_PRUNING

// only L2 implemented. Need to implement inner product search

namespace pipeann {
  // Initialize an index with metric m, load the data of type T with filename
  // (bin), and initialize max_points
  template<typename T, typename TagT>

  Index<T, TagT>::Index(Metric m, const size_t dim, const size_t max_points, const bool dynamic_index,
                        const bool save_index_in_one_file, const bool enable_tags)
      : _dist_metric(m), _dim(dim), _max_points(max_points), _save_as_one_file(save_index_in_one_file),
        _dynamic_index(dynamic_index), _enable_tags(enable_tags) {
    if (dynamic_index && !enable_tags) {
      LOG(ERROR) << "WARNING: Dynamic Indices must have tags enabled. Auto-enabling.";
      _enable_tags = true;
    }
    // data is stored to _nd * aligned_dim matrix with necessary
    // zero-padding
    _aligned_dim = ROUND_UP(_dim, 8);

    if (dynamic_index)
      _num_frozen_pts = 1;

    if (_max_points == 0) {
      _max_points = 1;
    }

#ifndef USE_DATA_PMEM
    alloc_aligned(((void **) &_data), (_max_points + 1) * _aligned_dim * sizeof(T), 8 * sizeof(T));
#endif
    // std::memset(_data, 0, (_max_points + 1) * _aligned_dim * sizeof(T));

    _ep = (unsigned) _max_points;

    _final_graph.reserve(_max_points + _num_frozen_pts);
    _final_graph.resize(_max_points + _num_frozen_pts);

    for (size_t i = 0; i < _max_points + _num_frozen_pts; i++)
      _final_graph[i].clear();

    constexpr uint64_t kLockTableEntries = 131072;  // ~1MB lock table.
    this->_locks = new v2::LockTable(kLockTableEntries);
    LOG(INFO) << "Getting distance function for metric: " << (m == pipeann::Metric::COSINE ? "cosine" : "l2");
    this->_distance = get_distance_function<T>(m);
    _width = 0;
  }

  template<typename T, typename TagT>
  Index<T, TagT>::~Index() {
    delete this->_distance;
    delete this->_locks;
#ifdef USE_DATA_PMEM
  #ifdef USE_DATA_PMEM_NUMA
      for (size_t i = 0; i < _numa_nodes; i++) {
        if (_numa_data_ptrs[i] == nullptr) continue;
        pmem_unmap(_numa_data_ptrs[i], _numa_mmap_lens[i]); 
      }
  #else
      if (_data != nullptr)
        pmem_unmap(_data, (_max_points + 1) * _aligned_dim
                    * sizeof(T)); 
  #endif
#else
    if (_data != nullptr)
      aligned_free(_data);
#endif
  }

  template<typename T, typename TagT>
  void Index<T, TagT>::clear_index() {
    memset(_data, 0, _aligned_dim * (_max_points + _num_frozen_pts) * sizeof(T));
    _nd = 0;
    for (size_t i = 0; i < _final_graph.size(); i++)
      _final_graph[i].clear();

    _tag_to_location.clear();
    _location_to_tag.clear();

    _delete_set.clear();
    _empty_slots.clear();
  }

  template<typename T, typename TagT>
  uint64_t Index<T, TagT>::save_tags(std::string tags_file, size_t offset, bool frozen) {
    if (!_enable_tags) {
      LOG(INFO) << "Not saving tags as they are not enabled.";
      return 0;
    }
    size_t tag_bytes_written;
    TagT *tag_data = new TagT[_nd + _num_frozen_pts];
    for (uint32_t i = 0; i < _nd; i++) {
      if (_location_to_tag.find(i) != _location_to_tag.end()) {
        tag_data[i] = _location_to_tag[i];
      } else {
        // catering to future when tagT can be any type.
        std::memset((char *) &tag_data[i], 0, sizeof(TagT));
      }
    }
    if (_num_frozen_pts > 0) {
      std::memset((char *) &tag_data[_ep], 0, sizeof(TagT));
    }
    size_t n = frozen ? (_nd + _num_frozen_pts) : _nd;
    tag_bytes_written = save_bin<TagT>(tags_file, tag_data, n, 1, offset);
    delete[] tag_data;
    return tag_bytes_written;
  }

  template<typename T, typename TagT>
  uint64_t Index<T, TagT>::save_data(std::string data_file, size_t offset, bool frozen) {
    size_t n = (frozen ? (_nd + _num_frozen_pts) : _nd);
    return save_data_in_base_dimensions(data_file, _data, n, _dim, _aligned_dim, offset);
  }

  // save the graph index on a file as an adjacency list. For each point,
  // first store the number of neighbors, and then the neighbor list (each as
  // 4 byte unsigned)
  template<typename T, typename TagT>
  uint64_t Index<T, TagT>::save_graph(std::string graph_file, size_t offset) {
    std::ofstream out;
    open_file_to_write(out, graph_file);

    out.seekp(offset, out.beg);
    uint64_t index_size = 24;
    uint32_t max_degree = 0;
    out.write((char *) &index_size, sizeof(uint64_t));
    out.write((char *) &_width, sizeof(unsigned));
    unsigned ep_u32 = _ep;
    out.write((char *) &ep_u32, sizeof(unsigned));
    out.write((char *) &_num_frozen_pts, sizeof(uint64_t));
    for (unsigned i = 0; i < _nd + _num_frozen_pts; i++) {
      unsigned GK = (unsigned) _final_graph[i].size();
      out.write((char *) &GK, sizeof(unsigned));
      out.write((char *) _final_graph[i].data(), GK * sizeof(unsigned));
      max_degree = _final_graph[i].size() > max_degree ? (uint32_t) _final_graph[i].size() : max_degree;
      index_size += (uint64_t) (sizeof(unsigned) * (GK + 1));
    }
    out.seekp(offset, out.beg);
    out.write((char *) &index_size, sizeof(uint64_t));
    out.write((char *) &max_degree, sizeof(uint32_t));
    out.close();
    return index_size;  // number of bytes written
  }

  template<typename T, typename TagT>
  uint64_t Index<T, TagT>::save_delete_list(const std::string &filename, uint64_t file_offset) {
    if (_delete_set.size() == 0) {
      return 0;
    }
    std::unique_ptr<uint32_t[]> delete_list = std::make_unique<uint32_t[]>(_delete_set.size());
    uint32_t i = 0;
    for (auto &del : _delete_set) {
      delete_list[i++] = del;
    }
    return save_bin<uint32_t>(filename, delete_list.get(), _delete_set.size(), 1, file_offset);
  }

  template<typename T, typename TagT>
  void Index<T, TagT>::save(const char *filename) {
    // first check if no thread is inserting
    auto start = std::chrono::high_resolution_clock::now();
    std::unique_lock<std::shared_timed_mutex> lock(_update_lock);
    _change_lock.lock();

    // compact_data();
    compact_frozen_point();
    if (!_save_as_one_file) {
      std::string graph_file = std::string(filename);
      std::string tags_file = std::string(filename) + ".tags";
      std::string data_file = std::string(filename) + ".data";
      std::string delete_list_file = std::string(filename) + ".del";

      // Because the save_* functions use append mode, ensure that
      // the files are deleted before save. Ideally, we should check
      // the error code for delete_file, but will ignore now because
      // delete should succeed if save will succeed.
      delete_file(graph_file);
      save_graph(graph_file);
      delete_file(data_file);
      save_data(data_file);
      delete_file(tags_file);
      save_tags(tags_file);
      delete_file(delete_list_file);
      save_delete_list(delete_list_file);
    } else {
      delete_file(filename);
      std::vector<size_t> cumul_bytes(5, 0);
      cumul_bytes[0] = METADATA_SIZE;
      cumul_bytes[1] = cumul_bytes[0] + save_graph(std::string(filename), cumul_bytes[0]);
      cumul_bytes[2] = cumul_bytes[1] + save_data(std::string(filename), cumul_bytes[1]);
      cumul_bytes[3] = cumul_bytes[2] + save_tags(std::string(filename), cumul_bytes[2]);
      cumul_bytes[4] = cumul_bytes[3] + save_delete_list(filename, cumul_bytes[3]);
      pipeann::save_bin<uint64_t>(filename, cumul_bytes.data(), cumul_bytes.size(), 1, 0);

      LOG(INFO) << "Saved index as one file to " << filename << " of size " << cumul_bytes[cumul_bytes.size() - 1]
                << "B.";
    }

    reposition_frozen_point_to_end();

    _change_lock.unlock();
    auto stop = std::chrono::high_resolution_clock::now();
    auto timespan = std::chrono::duration_cast<std::chrono::duration<double>>(stop - start);
    LOG(INFO) << "Time taken for save: " << timespan.count() << "s.";
  }

  template<typename T, typename TagT>
  size_t Index<T, TagT>::load_tags(const std::string tag_filename, size_t offset) {
    if (_enable_tags && !file_exists(tag_filename)) {
      LOG(ERROR) << "Tag file provided does not exist!";
      crash();
    }

    if (!_enable_tags) {
      LOG(INFO) << "Tags not loaded as tags not enabled.";
      return 0;
    }

    size_t file_dim, file_num_points;
    TagT *tag_data;
    load_bin<TagT>(std::string(tag_filename), tag_data, file_num_points, file_dim, offset);

    if (file_dim != 1) {
      LOG(ERROR) << "ERROR: Loading " << file_dim << " dimensions for tags,"
                 << "but tag file must have 1 dimension.";
      crash();
    }

    size_t num_data_points = _num_frozen_pts > 0 ? file_num_points - 1 : file_num_points;
    for (uint32_t i = 0; i < (uint32_t) num_data_points; i++) {
      TagT tag = *(tag_data + i);
      if (_delete_set.find(i) == _delete_set.end()) {
        _location_to_tag[i] = tag;
        _tag_to_location[tag] = (uint32_t) i;
      }
    }
    LOG(INFO) << "Tags loaded.";
    delete[] tag_data;
    return file_num_points;
  }

  template<typename T, typename TagT>
  size_t Index<T, TagT>::load_data(std::string filename, size_t offset) {
    LOG(INFO) << "Loading data from " << filename << " offset " << offset;
    if (!file_exists(filename)) {
      LOG(ERROR) << "ERROR: data file " << filename << " does not exist.";
      aligned_free(_data);
      crash();
    }
    size_t file_dim, file_num_points;
    pipeann::get_bin_metadata(filename, file_num_points, file_dim, offset);

    // since we are loading a new dataset, _empty_slots must be cleared
    _empty_slots.clear();

    if (file_dim != _dim) {
      LOG(ERROR) << "ERROR: Driver requests loading " << _dim << " dimension,"
                 << "but file has " << file_dim << " dimension.";
      aligned_free(_data);
      crash();
    }

    if (file_num_points > _max_points + _num_frozen_pts) {
      //_change_lock is already locked in load()
      std::unique_lock<std::shared_timed_mutex> tl(_tag_lock);
      std::unique_lock<std::shared_timed_mutex> growth_lock(_update_lock);

      resize(file_num_points);
    }

    copy_aligned_data_from_file<T>(std::string(filename), _data, file_num_points, file_dim, _aligned_dim, offset);
    return file_num_points;
  }

  template<typename T, typename TagT>
  size_t Index<T, TagT>::load_delete_set(const std::string &filename, size_t offset) {
    std::unique_ptr<uint32_t[]> delete_list;
    uint64_t npts, ndim;
    load_bin<uint32_t>(filename, delete_list, npts, ndim, offset);
    assert(ndim == 1);
    for (size_t i = 0; i < npts; i++) {
      _delete_set.insert(delete_list[i]);
    }
    return npts;
  }

  // load the index from file and update the width (max_degree), ep (navigating
  // node id), and _final_graph (adjacency list)
  template<typename T, typename TagT>
  void Index<T, TagT>::load(const char *filename) {
    _change_lock.lock();

    size_t tags_file_num_pts = 0, graph_num_pts = 0, data_file_num_pts = 0;

    if (!_save_as_one_file) {
      std::string data_file = std::string(filename) + ".data";
      std::string tags_file = std::string(filename) + ".tags";
      std::string delete_set_file = std::string(filename) + ".del";
      std::string graph_file = std::string(filename);
      data_file_num_pts = load_data(data_file);
      if (file_exists(delete_set_file)) {
        load_delete_set(delete_set_file);
      }
      if (_enable_tags) {
        tags_file_num_pts = load_tags(tags_file);
      }
      graph_num_pts = load_graph(graph_file, data_file_num_pts);

    } else {
      uint64_t nr, nc;
      std::unique_ptr<uint64_t[]> file_offset_data;

      std::string index_file(filename);

      pipeann::load_bin<uint64_t>(index_file, file_offset_data, nr, nc, 0);
      // Loading data first so that we know how many points to expect.
      data_file_num_pts = load_data(index_file, file_offset_data[1]);
      graph_num_pts = load_graph(index_file, data_file_num_pts, file_offset_data[0]);
      if (file_offset_data[3] != file_offset_data[4]) {
        load_delete_set(index_file, file_offset_data[3]);
      }
      if (_enable_tags) {
        tags_file_num_pts = load_tags(index_file, file_offset_data[2]);
      }
    }

    if (data_file_num_pts != graph_num_pts || (data_file_num_pts != tags_file_num_pts && _enable_tags)) {
      LOG(ERROR) << "ERROR: When loading index, loaded " << data_file_num_pts << " points from datafile, "
                 << graph_num_pts << " from graph, and " << tags_file_num_pts
                 << " tags, with num_frozen_pts being set to " << _num_frozen_pts << " in constructor.";
      aligned_free(_data);
      crash();
    }

    _nd = data_file_num_pts - _num_frozen_pts;
    _empty_slots.clear();
    for (uint32_t i = _nd; i < _max_points; i++) {
      _empty_slots.insert(i);
    }

    _lazy_done = _delete_set.size() != 0;

    reposition_frozen_point_to_end();
    LOG(INFO) << "Num frozen points:" << _num_frozen_pts << " _nd: " << _nd << " _ep: " << _ep
              << " size(_location_to_tag): " << _location_to_tag.size()
              << " size(_tag_to_location):" << _tag_to_location.size() << " Max points: " << _max_points;

    _change_lock.unlock();
  }

  template<typename T, typename TagT>
  void Index<T, TagT>::load_from_disk_index(const std::string &filename) {
    // only load V and E.
    std::ifstream in(filename + "_disk.index", std::ios::binary);
    uint32_t nr, nc;
    uint64_t disk_nnodes, disk_ndims, medoid_id_on_file, max_node_len, nnodes_per_sector;

    in.read((char *) &nr, sizeof(uint32_t));
    in.read((char *) &nc, sizeof(uint32_t));

    in.read((char *) &disk_nnodes, sizeof(uint64_t));
    in.read((char *) &disk_ndims, sizeof(uint64_t));

    in.read((char *) &medoid_id_on_file, sizeof(uint64_t));
    in.read((char *) &max_node_len, sizeof(uint64_t));
    in.read((char *) &nnodes_per_sector, sizeof(uint64_t));

    LOG(INFO) << "Loading disk index from " << filename << "_disk.index";
    LOG(INFO) << "Disk index has " << disk_nnodes << " nodes and " << disk_ndims << " dimensions.";
    LOG(INFO) << "Medoid id on file: " << medoid_id_on_file << " Max node len: " << max_node_len
              << " Nodes per sector: " << nnodes_per_sector;

    _ep = medoid_id_on_file;
    uint64_t data_dim = disk_ndims;
    range = ((max_node_len - data_dim * sizeof(T)) / sizeof(unsigned)) - 1;
    nhood_len = (range + 1) * sizeof(uint32_t);

    #ifdef USE_FLAT_GRAPH
  #ifdef USE_GRAPH_PMEM
    {
      size_t mmap_len = 0;
      int is_pmem = 0;
      size_t requested_len = (_max_points + _num_frozen_pts) * nhood_len;
      std::string fname = "/mnt/pmem1/yq_data/pmemann_output/_flat_graph";
      _flat_graph = (uint8_t *)pmem_map_file(fname.c_str(), requested_len, PMEM_FILE_CREATE, 0666, &mmap_len, &is_pmem);

      if (!is_pmem) {
        LOG(ERROR) << "ERROR: flat graph is not mapped to PMEM!";
        crash();
      }
      LOG(PMEM) << "Pmem Mapped. Size: " << mmap_len;
      LOG(PMEM) << "Mapped File: " << fname << " File Size: " << requested_len;
    }
  #else
    _flat_graph = new uint8_t[(_max_points + _num_frozen_pts) * nhood_len];
  #endif
    _final_graph.clear();
    #endif

    #ifdef USE_DATA_PMEM
  #ifdef USE_DATA_PMEM_NUMA
    if (numa_available() < 0) {
        LOG(ERROR) << "NUMA is not available on this system.";
        crash();
    }
    
    const std::string cluster_map_path = "/data/dataset_vec_public/sift100w/sift100w_cluster_map.bin";
    
    int32_t num_points = 0;
    int32_t num_clusters = 0;
    std::vector<int32_t> cluster_labels;

    {
        std::ifstream cluster_file(cluster_map_path, std::ios::binary);
        if (!cluster_file) {
            LOG(ERROR) << "Could not open cluster file: " << cluster_map_path;
        }
        cluster_file.read((char*)&num_points, sizeof(int32_t));
        cluster_file.read((char*)&num_clusters, sizeof(int32_t));
        
        cluster_labels.resize(num_points);
        cluster_file.read((char*)cluster_labels.data(), num_points * sizeof(int32_t));
        _numa_cluster_ids.resize(num_clusters);
    
        cluster_file.read((char*)_numa_cluster_ids.data(), num_clusters * sizeof(uint32_t));

        LOG(NUMA) << "Loaded cluster map. Points: " << num_points << ", Clusters: " << num_clusters;
        for (size_t i = 0; i < _numa_cluster_ids.size(); i++) {
            LOG(NUMA) << "Cluster " << i << " Nearest point ID: " << _numa_cluster_ids[i];
        }
    }

    _numa_nodes = num_clusters;
    if (_numa_nodes > 4) LOG(ERROR) << "Too many clusters for 4-slot NUMA system.";

    std::vector<uint64_t> node_counts(_numa_nodes, 0);
    for (int i = 0; i < num_points; i++) {
        node_counts[cluster_labels[i]]++;
    }

    for (uint8_t i = 0; i < _numa_nodes; i++) {
        if (node_counts[i] == 0) continue;

        size_t mmap_len = 0;
        int is_pmem = 0;
        size_t requested_len = node_counts[i] * _aligned_dim * sizeof(T);
        std::string fname = "/mnt/pmem" + std::to_string(i) + "/yq_data/pmemann_output/_data_n" + std::to_string(i);

        _numa_data_ptrs[i] = (T*)pmem_map_file(
            fname.c_str(), requested_len, PMEM_FILE_CREATE, 0666, &mmap_len, &is_pmem
        );
        _numa_mmap_lens.push_back(mmap_len);

        if (!_numa_data_ptrs[i] || !is_pmem) {
            LOG(ERROR) << "PMEM mapping failed on Node " << i << " (Path: " << fname << ")";
            crash();
        }
        LOG(NUMA) << "Node " << (int)i << " mapped " << node_counts[i] << " points. Size: " << mmap_len;
    }

    _numa_cluster_map.clear();
    _numa_cluster_map.resize(num_points);

    std::vector<std::atomic<uint64_t>> current_offsets(_numa_nodes);
    for (size_t i = 0; i < _numa_nodes; ++i) {
        current_offsets[i].store(0);
    }

    LOG(NUMA) << "NUMA-aware data distribution complete.";
  #else
    {
      size_t mmap_len = 0;
      int is_pmem = 0;
      size_t requested_len = (_max_points + 1) * _aligned_dim * sizeof(T);
      std::string fname = "/mnt/pmem1/yq_data/pmemann_output/_data";
      _data = (T *)pmem_map_file(fname.c_str(), requested_len, PMEM_FILE_CREATE, 0666, &mmap_len, &is_pmem);
      
      if (!is_pmem) {
        LOG(ERROR) << "ERROR: flat graph is not mapped to PMEM!";
        crash();
      }
      LOG(PMEM) << "Pmem Mapped. Size: " << mmap_len;
      LOG(PMEM) << "Mapped File: " << fname << " File Size: " << requested_len;
      if (!_data) {
          perror("pmem_map_file");
          exit(1);
      }
    }
  #endif
    #endif

    constexpr int kSectorsPerRead = 65536;
    constexpr int kSectorLen = 4096;
    char *buf;
    pipeann::alloc_aligned((void **) &buf, kSectorsPerRead * kSectorLen, kSectorLen);
    if (nnodes_per_sector > 0) {
      uint64_t n_sectors = ROUND_UP(disk_nnodes, nnodes_per_sector) / nnodes_per_sector;
      in.seekg(4096, in.beg);
      for (uint64_t in_sector = 0; in_sector < n_sectors; in_sector += kSectorsPerRead) {
        uint64_t st_sector = in_sector, ed_sector = std::min(in_sector + kSectorsPerRead, n_sectors);
        uint64_t loc_st = st_sector * nnodes_per_sector, loc_ed = std::min(disk_nnodes, ed_sector * nnodes_per_sector);
        uint64_t n_sectors_to_read = ed_sector - st_sector;
        in.read(buf, n_sectors_to_read * kSectorLen);

        #pragma omp parallel for
        for (uint64_t loc = loc_st; loc < loc_ed; ++loc) {
          uint64_t id = loc;
          #pragma omp critical
          {
            _location_to_tag[id] = id;
            _tag_to_location[id] = id;
          }

          auto page_rbuf = buf + (loc / nnodes_per_sector - st_sector) * kSectorLen;
          auto node_rbuf = page_rbuf + (nnodes_per_sector == 0 ? 0 : ((uint64_t) loc % nnodes_per_sector) * max_node_len);
          DiskNode<T> node(id, (T *) node_rbuf, (unsigned *) (node_rbuf + data_dim * sizeof(T)));

          #ifdef USE_DATA_PMEM_NUMA
          uint8_t nid = static_cast<uint8_t>(cluster_labels[id]);
          uint64_t offset = current_offsets[nid].fetch_add(1);

          _numa_cluster_map[id] = (offset << 2) | nid;

          T* dest = _numa_data_ptrs[nid] + (offset * _aligned_dim);
          memcpy(dest, node.coords, data_dim * sizeof(T));
          #else
          // load data and nhood.
          memcpy(_data + id * _aligned_dim, node.coords, data_dim * sizeof(T));
          #endif

         #ifdef USE_FLAT_GRAPH
          memcpy(_flat_graph + id * nhood_len, &node.nnbrs, sizeof(uint32_t));
          memcpy(_flat_graph + id * nhood_len + sizeof(uint32_t), node.nbrs, node.nnbrs * sizeof(uint32_t));
          #else
          std::vector<uint32_t> nhood;
          for (uint32_t i = 0; i < node.nnbrs; ++i) {
            nhood.push_back(node.nbrs[i]);
          }
          _final_graph[id] = nhood;
          #endif
        }
      }
      
      #ifdef USE_DATA_PMEM_NUMA
      for (size_t i = 0; i < _numa_nodes; i++) {
        if (node_counts[i] != current_offsets[i].load()) {
          LOG(ERROR) << "Mismatch in node count for cluster " << i << ": expected "
                    << node_counts[i] << ", got " << current_offsets[i].load();
        }
      }
      #endif
    } else {
      // Logic for Large Nodes (Node size > Sector size)
      // Reference from writer: nsectors_per_node = DIV_ROUND_UP(max_node_len, SECTOR_LEN);
      uint64_t nsectors_per_node = (max_node_len + kSectorLen - 1) / kSectorLen;
      
      // Calculate how many full nodes fit into our reading buffer
      uint64_t nodes_per_read = kSectorsPerRead / nsectors_per_node;
      if (nodes_per_read == 0) nodes_per_read = 1; // Ensure we read at least one node

      in.seekg(4096, in.beg); // Skip the file header

      for (uint64_t start_node = 0; start_node < disk_nnodes; start_node += nodes_per_read) {
        uint64_t end_node = std::min(disk_nnodes, start_node + nodes_per_read);
        uint64_t nodes_in_batch = end_node - start_node;
        uint64_t sectors_to_read = nodes_in_batch * nsectors_per_node;

        // Read the batch of nodes into the buffer
        // Layout in file: [Node 0 Pad] [Node 1 Pad] ... where Pad aligns to sector boundaries
        in.read(buf, sectors_to_read * kSectorLen);

        #pragma omp parallel for
        for (uint64_t i = 0; i < nodes_in_batch; ++i) {
          uint64_t id = start_node + i;
          
          #pragma omp critical
          {
            _location_to_tag[id] = id;
            _tag_to_location[id] = id;
          }

          // Calculate the offset of the current node within the read buffer
          // Each node occupies 'nsectors_per_node' sectors
          char *node_rbuf = buf + (i * nsectors_per_node * kSectorLen);
          
          // Construct DiskNode (pointers to buffer)
          // Data layout matches Writer: [Coords] [Num Neighbors] [Neighbors]
          DiskNode<T> node(id, (T *) node_rbuf, (unsigned *) (node_rbuf + data_dim * sizeof(T)));

          #ifdef USE_DATA_PMEM_NUMA
          uint8_t nid = static_cast<uint8_t>(cluster_labels[id]);
          uint64_t offset = current_offsets[nid].fetch_add(1);

          _numa_cluster_map[id] = (offset << 2) | nid;

          T* dest = _numa_data_ptrs[nid] + (offset * _aligned_dim);
          memcpy(dest, node.coords, data_dim * sizeof(T));
          #else
          // load data and nhood.
          memcpy(_data + id * _aligned_dim, node.coords, data_dim * sizeof(T));
          #endif

          #ifdef USE_FLAT_GRAPH
          memcpy(_flat_graph + id * nhood_len, &node.nnbrs, sizeof(uint32_t));
          memcpy(_flat_graph + id * nhood_len + sizeof(uint32_t), node.nbrs, node.nnbrs * sizeof(uint32_t));
          #else
          std::vector<uint32_t> nhood;
          nhood.reserve(node.nnbrs);
          for (uint32_t j = 0; j < node.nnbrs; ++j) {
            nhood.push_back(node.nbrs[j]);
          }
          _final_graph[id] = nhood;
          #endif
        }
      }

      #ifdef USE_DATA_PMEM_NUMA
      for (size_t i = 0; i < _numa_nodes; i++) {
        if (node_counts[i] != current_offsets[i].load()) {
          LOG(ERROR) << "Mismatch in node count for cluster " << i << ": expected "
                    << node_counts[i] << ", got " << current_offsets[i].load();
        }
      }
      #endif
    }

    disk_npts = disk_nnodes;
    _nd = disk_nnodes - _num_frozen_pts;
    _empty_slots.clear();
    for (uint32_t i = _nd; i < _max_points; i++) {
      _empty_slots.insert(i);
    }
    reposition_frozen_point_to_end();
  }

  template<typename T, typename TagT>
  void Index<T, TagT>::save_to_disk_index(const std::string &filename) {
    const std::string disk_index_file = filename + "_disk.index";
    std::ofstream out(disk_index_file, std::ios::binary | std::ios::trunc);
    if (!out.is_open()) {
      LOG(ERROR) << "Failed to open disk index file for writing: " << disk_index_file;
      crash();
    }

    const uint64_t disk_nnodes = _nd + _num_frozen_pts;
    const uint64_t disk_ndims = _dim;
    uint64_t medoid_id_on_file = _ep;
    const uint64_t max_degree = _width;
    const uint64_t max_node_len = disk_ndims * sizeof(T) + (max_degree + 1) * sizeof(uint32_t);
    const uint64_t nnodes_per_sector = max_node_len <= SECTOR_LEN ? SECTOR_LEN / max_node_len : 0;

    // Dynamic index keeps a frozen point at _max_points as entry point.
    // Disk index stores only [0, disk_nnodes), so remap entry point if needed.
    if (medoid_id_on_file >= disk_nnodes) {
      if (_ep < _final_graph.size() && !_final_graph[_ep].empty()) {
        medoid_id_on_file = _final_graph[_ep][0];
      } else {
        medoid_id_on_file = 0;
      }
      if (medoid_id_on_file >= disk_nnodes) {
        medoid_id_on_file = 0;
      }
      LOG(INFO) << "Remapped frozen entry point " << _ep << " to disk medoid " << medoid_id_on_file;
    }

    uint32_t nr = 6;
    uint32_t nc = 1;
    out.write((char *) &nr, sizeof(uint32_t));
    out.write((char *) &nc, sizeof(uint32_t));
    out.write((char *) &disk_nnodes, sizeof(uint64_t));
    out.write((char *) &disk_ndims, sizeof(uint64_t));
    out.write((char *) &medoid_id_on_file, sizeof(uint64_t));
    out.write((char *) &max_node_len, sizeof(uint64_t));
    out.write((char *) &nnodes_per_sector, sizeof(uint64_t));

    const size_t header_bytes = sizeof(uint32_t) * 2 + sizeof(uint64_t) * 5;
    if (header_bytes < SECTOR_LEN) {
      std::vector<char> padding(SECTOR_LEN - header_bytes, 0);
      out.write(padding.data(), padding.size());
    }

    auto write_node = [&](char *node_buf, uint64_t node_id) {
      std::memset(node_buf, 0, max_node_len);
      const uint64_t coord_bytes = disk_ndims * sizeof(T);
      std::memcpy(node_buf, _data + (size_t) node_id * _aligned_dim, coord_bytes);

      const uint32_t nnbrs = (uint32_t) _final_graph[node_id].size();
      std::memcpy(node_buf + coord_bytes, &nnbrs, sizeof(uint32_t));
      if (nnbrs > 0) {
        std::memcpy(node_buf + coord_bytes + sizeof(uint32_t), _final_graph[node_id].data(),
                    (size_t) nnbrs * sizeof(uint32_t));
      }
    };

    if (disk_nnodes == 0) {
      out.close();
      LOG(INFO) << "Saved empty disk index to " << disk_index_file;
      return;
    }

    if (nnodes_per_sector > 0) {
      const uint64_t n_sectors = ROUND_UP(disk_nnodes, nnodes_per_sector) / nnodes_per_sector;
      std::vector<char> sector_buf(SECTOR_LEN, 0);
      for (uint64_t sector_idx = 0; sector_idx < n_sectors; ++sector_idx) {
        std::memset(sector_buf.data(), 0, sector_buf.size());
        const uint64_t sector_start = sector_idx * nnodes_per_sector;
        const uint64_t sector_end = std::min(disk_nnodes, sector_start + nnodes_per_sector);
        for (uint64_t node_id = sector_start; node_id < sector_end; ++node_id) {
          const uint64_t node_offset = (node_id - sector_start) * max_node_len;
          write_node(sector_buf.data() + node_offset, node_id);
        }
        out.write(sector_buf.data(), sector_buf.size());
      }
    } else {
      const uint64_t nsectors_per_node = (max_node_len + SECTOR_LEN - 1) / SECTOR_LEN;
      const uint64_t node_bytes = nsectors_per_node * SECTOR_LEN;
      std::vector<char> node_buf(node_bytes, 0);
      for (uint64_t node_id = 0; node_id < disk_nnodes; ++node_id) {
        std::memset(node_buf.data(), 0, node_buf.size());
        write_node(node_buf.data(), node_id);
        out.write(node_buf.data(), node_buf.size());
      }
    }

    out.close();
    LOG(INFO) << "Saved disk index to " << disk_index_file << " with " << disk_nnodes << " nodes.";
  }

  template<typename T, typename TagT>
  size_t Index<T, TagT>::load_graph(std::string filename, size_t expected_num_points, size_t offset) {
    std::ifstream in(filename, std::ios::binary);
    in.seekg(offset, in.beg);
    size_t expected_file_size;
    uint64_t file_frozen_pts;
    in.read((char *) &expected_file_size, sizeof(uint64_t));
    in.read((char *) &_width, sizeof(unsigned));
    in.read((char *) &_ep, sizeof(unsigned));
    in.read((char *) &file_frozen_pts, sizeof(uint64_t));

    if (file_frozen_pts != _num_frozen_pts) {
      if (file_frozen_pts == 1) {
        LOG(ERROR) << "ERROR: When loading index, detected dynamic index, but "
                      "constructor asks for static index. Exitting.";
      } else {
        LOG(ERROR) << "ERROR: When loading index, detected static index, but "
                      "constructor asks for dynamic index. Exitting.";
      }
      aligned_free(_data);
      crash();
    }
    LOG(INFO) << "Loading vamana index " << filename << "...";

    // Sanity check. In case the user gave us fewer points as max_points than
    // the number
    // of points in the dataset, resize the _final_graph to the larger size.
    if (_max_points < (expected_num_points - _num_frozen_pts)) {
      LOG(INFO) << "Number of points in data: " << expected_num_points
                << " is more than max_points argument: " << _final_graph.size()
                << " Setting max points to: " << expected_num_points;
      _final_graph.resize(expected_num_points);
      _max_points = expected_num_points - _num_frozen_pts;
      // changed expected_num to expected_num - frozen_num
    }

    size_t bytes_read = 24;
    size_t cc = 0;
    unsigned nodes = 0;
    while (bytes_read != expected_file_size) {
      unsigned k;
      in.read((char *) &k, sizeof(unsigned));
      if (k == 0) {
        LOG(ERROR) << "ERROR: Point found with no out-neighbors, point#" << nodes;
      }
      //      if (in.eof())
      //        break;
      cc += k;
      ++nodes;
      std::vector<unsigned> tmp(k);
      tmp.reserve(k);
      in.read((char *) tmp.data(), k * sizeof(unsigned));
      _final_graph[nodes - 1].swap(tmp);
      bytes_read += sizeof(uint32_t) * ((uint64_t) k + 1);
      if (nodes % 10000000 == 0)
        LOG(INFO) << ".";
    }

    LOG(INFO) << "done. Index has " << nodes << " nodes and " << cc << " out-edges, _ep is set to " << _ep;
    return nodes;
  }

  template<typename T, typename TagT>
  int Index<T, TagT>::get_vector_by_tag(TagT &tag, T *vec) {
    std::shared_lock<std::shared_timed_mutex> lock(_tag_lock);
    if (_tag_to_location.find(tag) == _tag_to_location.end()) {
      LOG(INFO) << "Tag " << tag << " does not exist";
      return -1;
    }
    unsigned location = _tag_to_location[tag];
    // memory should be allocated for vec before calling this function
    memcpy((void *) vec, (void *) (_data + (size_t) (location * _aligned_dim)), (size_t) _aligned_dim * sizeof(T));
    return 0;
  }

  /**************************************************************
   *      Support for Static Index Building and Searching
   **************************************************************/

  /* This function finds out the navigating node, which is the medoid node
   * in the graph.
   */
  template<typename T, typename TagT>
  unsigned Index<T, TagT>::calculate_entry_point() {
    // allocate and init centroid
    std::vector<float> center(_aligned_dim, 0.0f);

    for (size_t i = 0; i < _nd; i++)
      for (size_t j = 0; j < _aligned_dim; j++)
        center[j] += (float) _data[i * _aligned_dim + j];

    for (size_t j = 0; j < _aligned_dim; j++)
      center[j] /= (float) _nd;

    // compute all to one distance, updating the atomic variables should not be the bottleneck.
    constexpr uint64_t kDistNum = 256;
    struct alignas(64) AtomicDistance {
      unsigned idx = 0;
      float dist = std::numeric_limits<float>::max();
      std::mutex lk;

      void update(unsigned i, float d) {
        std::lock_guard<std::mutex> guard(lk);
        if (d < dist) {
          dist = d;
          idx = i;
        }
      }
    };
    AtomicDistance atomic_dists[kDistNum];

#pragma omp parallel for schedule(static, 65536)
    for (int64_t i = 0; i < (int64_t) _nd; i++) {
      // extract point and distance reference
      float dist = 0;
      const T *cur_vec = _data + (i * (size_t) _aligned_dim);
      for (size_t j = 0; j < _aligned_dim; j++) {
        dist += (center[j] - (float) cur_vec[j]) * (center[j] - (float) cur_vec[j]);
      }
      atomic_dists[(i / 65536) % kDistNum].update(i, dist);
    }

    unsigned min_idx = 0;
    float min_dist = std::numeric_limits<float>::max();
    for (unsigned i = 0; i < kDistNum; i++) {
      if (atomic_dists[i].dist < min_dist) {
        min_idx = atomic_dists[i].idx;
        min_dist = atomic_dists[i].dist;
      }
    }
    return min_idx;
  }
  
  template<typename T, typename TagT>
  inline void Index<T, TagT>::get_coord(size_t id, T* buffer) const {
    #ifdef USE_DATA_PMEM_NUMA
        
        uint8_t nid = static_cast<uint8_t>(_numa_cluster_map[id] & 0x03);
        uint64_t offset = _numa_cluster_map[id] >> 2;
        
        
        const T* src = _numa_data_ptrs[nid] + (offset * _aligned_dim);
        std::memcpy(buffer, src, _aligned_dim * sizeof(T));
    #else
        
        std::memcpy(buffer, _data + (_aligned_dim * id), _aligned_dim * sizeof(T));
    #endif
  }

  template<typename T, typename TagT>
  uint8_t Index<T, TagT>::get_target_numa_node(const T *query_coords, T *coord_scratch) {
    if (_numa_nodes <= 1) return 0;

    int8_t min_nid = -1;
    float min_dist = std::numeric_limits<float>::max();

    for (uint8_t i = 0; i < (uint8_t)_numa_nodes; ++i) {
        
        size_t centroid_id = _numa_cluster_ids[i];
        
        
        
        get_coord(centroid_id, coord_scratch);
        
        
        float d = _distance->compare(coord_scratch, query_coords, (unsigned)_aligned_dim);
        
        
        if (min_nid == -1 || d < min_dist) {
            min_dist = d;
            min_nid = i;
        }
    }

    if (unlikely(min_nid == -1)) {
        LOG(ERROR) << "Critical: No NUMA nodes found during routing!";
        return 0; 
    }

    return (uint8_t)min_nid;
  }

  /* iterate_to_fixed_point():
   * node_coords : point whose neighbors to be found.
   * init_ids : ids of initial search list.
   * Lsize : size of list.
   * beam_width: beam_width when performing indexing
   * expanded_nodes_info: will contain all the node ids and distances from
   * query that are expanded
   * expanded_nodes_ids : will contain all the nodes that are expanded during
   * search.
   * best_L_nodes: ids of closest L nodes in list
   */
  template<typename T, typename TagT>
  std::pair<uint32_t, uint32_t> Index<T, TagT>::iterate_to_fixed_point(const T *node_coords, const unsigned Lsize,
                                                                       const std::vector<unsigned> &init_ids,
                                                                       std::vector<Neighbor> &expanded_nodes_info,
                                                                       tsl::robin_set<unsigned> &expanded_nodes_ids,
                                                                       std::vector<Neighbor> &best_L_nodes,
                                                                       bool ret_frozen, QueryStats *stats) {
    // struct alignas(64) DistBuffer {
    //   T buf[1024];
    // } dist_buf_aligned;
    
    alignas(4096) T coord_buf[2048];
    alignas(4096) unsigned nhood_buf[512];

//     auto get_coord = [&](size_t id, T* buffer) {
// #ifdef USE_DATA_PMEM_NUMA
//       uint8_t nid = static_cast<uint8_t>(_numa_cluster_map[id] & 0x3);
//       uint64_t offset = _numa_cluster_map[id] >> 2;
//       T* src = _numa_data_ptrs[nid] + (offset * _aligned_dim);
//       memcpy(coord_buf, (const char *) src,  _aligned_dim * sizeof(T));
// #else
//       memcpy(coord_buf, (const char *) (_data + _aligned_dim * (size_t) id),  _aligned_dim * sizeof(T));
// #endif
//     };

    uint32_t hops = 0;
    uint32_t cmps = 0;
    best_L_nodes.resize(Lsize + 1);
    for (unsigned i = 0; i < Lsize + 1; i++) {
      best_L_nodes[i].distance = std::numeric_limits<float>::max();
    }
    expanded_nodes_info.reserve(10 * Lsize);
    expanded_nodes_ids.reserve(10 * Lsize);

    unsigned l = 0;
    Neighbor nn;
    tsl::robin_set<unsigned> inserted_into_pool;
    inserted_into_pool.reserve(Lsize * 20);

    for (auto id : init_ids) {
      assert(id < _max_points + _num_frozen_pts);
      get_coord(id, coord_buf);
      nn = Neighbor(id, _distance->compare(coord_buf, node_coords, (unsigned) _aligned_dim),
                    true);
      cmps ++;
      if (inserted_into_pool.find(id) == inserted_into_pool.end()) {
        inserted_into_pool.insert(id);
        best_L_nodes[l++] = nn;
      }
      if (l == Lsize)
        break;
    }
    // exit(0);

    Timer query_timer, io_timer, cpu_timer;

#ifdef _USE_DATA_PMEM_NUMA
#endif

    /* sort best_L_nodes based on distance of each point to node_coords */
    std::sort(best_L_nodes.begin(), best_L_nodes.begin() + l);
    unsigned k = 0;

    while (k < l) {
      unsigned nk = l;

      if (best_L_nodes[k].flag) {
        hops++;
        io_timer.reset();
        best_L_nodes[k].flag = false;
        auto n = best_L_nodes[k].id;
        if (!(best_L_nodes[k].id == _ep && _num_frozen_pts > 0 && !ret_frozen)) {
          expanded_nodes_info.emplace_back(best_L_nodes[k]);
          expanded_nodes_ids.insert(n);
        }
        std::vector<unsigned> des;

        {
          // v2::SparseReadLockGuard<uint64_t> guard(&_locks, n);
          v2::LockGuard guard(_locks->rdlock(n));
#ifdef USE_FLAT_GRAPH
          memcpy(nhood_buf, (uint32_t *)(_flat_graph + (uint64_t)n * this->nhood_len),  (range + 1) * sizeof(uint32_t));
          uint32_t* nbr = nhood_buf;

          uint32_t nnbr = *nbr;
          if (unlikely( nnbr > (uint32_t)range)) {
            LOG(ERROR) << "Wrong number of neighbors found: " << nnbr;
            crash();
          }
          nbr ++;

          for (uint32_t m = 0; m < nnbr; ++m) {
            if (nbr[m] >= _max_points + _num_frozen_pts) {
              LOG(ERROR) << "Wrong id found: " << nbr[m];
              crash();
            }
            des.emplace_back(nbr[m]);
          }
#else
          for (unsigned m = 0; m < _final_graph[n].size(); m++) {
            if (_final_graph[n][m] >= _max_points + _num_frozen_pts) {
              LOG(ERROR) << "Wrong id found: " << _final_graph[n][m];
              crash();
            }
            des.emplace_back(_final_graph[n][m]);
          }
#endif
        }

        if (stats != nullptr) {
          stats->io_us += io_timer.elapsed();  // read vec
        }

        cpu_timer.reset();

        for (unsigned m = 0; m < des.size(); ++m) {
          unsigned id = des[m];
          if (inserted_into_pool.find(id) == inserted_into_pool.end()) {
            inserted_into_pool.insert(id);

            // io_timer.reset();
            // if ((m + 1) < des.size()) {
            //   auto nextn = des[m + 1];
            //   pipeann::prefetch_vector((const char *) _data + _aligned_dim * (size_t) nextn, sizeof(T) * _aligned_dim);
            // }
            cmps++;
            // memcpy((void *) dist_buf_aligned.buf, (const void *) (_data + _aligned_dim * (size_t) id),
            //        sizeof(T) * _aligned_dim);
            // if (stats != nullptr) {
            //   stats->io_us1 += io_timer.elapsed();  // read nbrs
            // }

            // float dist = _distance->compare(node_coords, dist_buf_aligned.buf, (unsigned) _aligned_dim);
            get_coord(id, coord_buf);
            float dist = _distance->compare(node_coords, coord_buf, (unsigned) _aligned_dim);

            if (dist >= best_L_nodes[l - 1].distance && (l == Lsize))
              continue;

            Neighbor nn(id, dist, true);
            unsigned r = InsertIntoPool(best_L_nodes.data(), l, nn);
            if (l < Lsize)
              ++l;
            if (r < nk)
              nk = r;
          }
        }
        if (stats != nullptr) {
          stats->cpu_us += cpu_timer.elapsed();  // compute + read nbr
        }

        if (nk <= k)
          k = nk;
        else
          ++k;
      } else
        k++;
    }

    // for (auto nbr : expanded_nodes_info) {
    //   std::cerr << "nbr: " << nbr.id << " dist: " << nbr.distance << std::endl;
    // }
    if (stats != nullptr) {
      stats->n_hops = (double) hops;
      stats->n_cmps = (double) cmps;
    }
    return std::make_pair(hops, cmps);
  }

  template<typename T, typename TagT>
  void Index<T, TagT>::iterate_to_fixed_point(const T *node_coords, const unsigned Lindex,
                                              std::vector<Neighbor> &expanded_nodes_info,
                                              tsl::robin_map<uint32_t, T *> &coord_map, bool return_frozen_pt) {
    std::vector<uint32_t> init_ids;
    init_ids.push_back(this->_ep);
    std::vector<Neighbor> best_L_nodes;
    tsl::robin_set<uint32_t> expanded_nodes_ids;
    this->iterate_to_fixed_point(node_coords, Lindex, init_ids, expanded_nodes_info, expanded_nodes_ids, best_L_nodes,
                                 return_frozen_pt);
    for (Neighbor &einf : expanded_nodes_info) {
      T *coords = this->_data + (uint64_t) einf.id * (uint64_t) this->_aligned_dim;
      coord_map.insert(std::make_pair(einf.id, coords));
    }
  }

  template<typename T, typename TagT>
  std::pair<uint32_t, uint32_t> Index<T, TagT>::get_expanded_nodes(const size_t node_id, const unsigned Lindex, std::vector<unsigned> init_ids,
                                          std::vector<Neighbor> &expanded_nodes_info,
                                          tsl::robin_set<unsigned> &expanded_nodes_ids) {
    const T *node_coords = _data + _aligned_dim * node_id;
    std::vector<Neighbor> best_L_nodes;

    if (init_ids.size() == 0)
      init_ids.emplace_back(_ep);

    return iterate_to_fixed_point(node_coords, Lindex, init_ids, expanded_nodes_info, expanded_nodes_ids, best_L_nodes);
  }

  template<typename T, typename TagT>
  int Index<T, TagT>::occlude_list(std::vector<Neighbor> &pool, const float alpha, const unsigned degree,
                                    const unsigned maxc, std::vector<Neighbor> &result) {
    auto pool_size = (uint32_t) pool.size();
    std::vector<float> occlude_factor(pool_size, 0);
    return occlude_list(pool, alpha, degree, maxc, result, occlude_factor);
  }

  template<typename T, typename TagT>
  int Index<T, TagT>::occlude_list(std::vector<Neighbor> &pool, const float alpha, const unsigned degree,
                                    const unsigned maxc, std::vector<Neighbor> &result,
                                    std::vector<float> &occlude_factor, bool skip) {
    if (pool.empty())
      return 0;
    assert(std::is_sorted(pool.begin(), pool.end()));
    assert(!pool.empty());

    float cur_alpha = 1;
    int cmp = 0;
    // Timer timer;
    while (cur_alpha <= alpha && result.size() < degree) {
      unsigned start = 0;

      while (result.size() < degree && (start) < pool.size() && start < maxc) {
        auto &p = pool[start];
        if (occlude_factor[start] > cur_alpha) {
          start++;
          continue;
        }
        occlude_factor[start] = std::numeric_limits<float>::max();
        result.push_back(p);
        for (unsigned t = start + 1; t < pool.size() && t < maxc; t++) {
          if (occlude_factor[t] > alpha)
            continue;
            
          if (skip && pool[t].flag == p.flag) {// skip
            continue;
          }
          float djk = _distance->compare(_data + _aligned_dim * (size_t) pool[t].id,
                                         _data + _aligned_dim * (size_t) p.id, (unsigned) _aligned_dim);
          cmp ++;
          occlude_factor[t] = (std::max)(occlude_factor[t], pool[t].distance / djk);
        }
        start++;
      }
      cur_alpha *= 1.2f;
    }
    return cmp;
  }

  template<typename T, typename TagT>
  int Index<T, TagT>::prune_neighbors(const unsigned location, std::vector<Neighbor> &pool,
                                       const Parameters &parameter, std::vector<unsigned> &pruned_list) {
    unsigned range = parameter.R;
    unsigned maxc = parameter.C;
    float alpha = parameter.alpha;

    if (pool.size() == 0) {
      crash();
    }

    _width = (std::max)(_width, range);

    // sort the pool based on distance to query
    std::sort(pool.begin(), pool.end());

    std::vector<Neighbor> result;
    result.reserve(range);
    std::vector<float> occlude_factor(pool.size(), 0);

    auto ret = occlude_list(pool, alpha, range, maxc, result, occlude_factor);

    /* Add all the nodes in result into a variable called cut_graph
     * So this contains all the neighbors of id location
     */
    pruned_list.clear();
    assert(result.size() <= range);
    for (auto iter : result) {
      if (iter.id != location)
        pruned_list.emplace_back(iter.id);
    }

    if (_saturate_graph && alpha > 1) {
      for (uint32_t i = 0; i < pool.size() && pruned_list.size() < range; i++) {
        if ((std::find(pruned_list.begin(), pruned_list.end(), pool[i].id) == pruned_list.end()) &&
            pool[i].id != location)
          pruned_list.emplace_back(pool[i].id);
      }
    }

    return ret;
  }

  // *****copy from PipeANN*****
  // ============================================================================
  // Unified delta_prune_neighbors implementation
  // ============================================================================
  // Delta prune neighbors: when inserting a new point into a full neighbor list.
  // nhood: current neighbors + target_id (size = R + 1), will be modified in place to output result (size = R).
  // NOTE: The last element of nhood should be target_id!
  // center_id: the center node whose neighbors are being pruned.
  // target_id: ID of the newly inserted point.
  // params: build parameters (R, alpha).
  // metric: distance metric.
  // compute_distances: callable to compute distances from center_id to multiple IDs.
  //   void(uint32_t center_id, const uint32_t *ids, uint32_t n, float *dists_out)
  template<typename T, typename TagT>
  void Index<T, TagT>::delta_prune_neighbors(const Parameters &parameter, std::vector<uint32_t> &nhood, uint32_t center_id, uint32_t target_id) {
    struct TriangleNeighbor {
      uint32_t id;
      float tgt_dis;   // distance to target
      float distance;  // distance to center
      inline bool operator<(const TriangleNeighbor &other) const {
        return (distance < other.distance) || (distance == other.distance && id < other.id);
      }
    };

    auto get_occlude_factor = [](float dik, float djk) {
      return (djk == 0.0f) ? std::numeric_limits<float>::max() : (dik / djk);
    };
    
    auto compute_distances = [&](uint32_t q_id, const uint32_t *ids, uint32_t n, float *dists_out) {
      const T* vec_q = _data + (size_t)q_id * _aligned_dim;
      for (uint32_t i = 0; i < n; ++i) {
        if (ids[i] == q_id) {
          dists_out[i] = 0.0f;
        } else {
          const T* vec_target = _data + (size_t)ids[i] * _aligned_dim;
          dists_out[i] = _distance->compare(vec_q, vec_target, (unsigned)_aligned_dim);
        }
      }
    };

    constexpr uint32_t kInvalidID = std::numeric_limits<uint32_t>::max();

    unsigned range = parameter.R;
    unsigned maxc = parameter.C;
    float alpha = parameter.alpha;

    if (unlikely(nhood.size() != range + 1)) {
      LOG(ERROR) << "nhood size " << nhood.size() << " != R + 1 (" << range + 1 << ")";
    }

    // Compute distances from center and target to all neighbors.
    std::vector<float> center_dists(nhood.size()), target_dists(nhood.size());
    compute_distances(center_id, nhood.data(), nhood.size(), center_dists.data());
    compute_distances(target_id, nhood.data(), nhood.size(), target_dists.data());

    assert(nhood.back() == target_id);
    float target_center_dist = center_dists.back();

    // Build pool with both distances and sort by distance to center.
    std::vector<TriangleNeighbor> pool(nhood.size());
    for (uint32_t i = 0; i < nhood.size(); i++) {
      pool[i] = {nhood[i], target_dists[i], center_dists[i]};
    }
    std::sort(pool.begin(), pool.end());

    uint32_t to_evict = kInvalidID;
    uint32_t tgt_idx = kInvalidID;

    // Fast path: try to find a point to evict using triangle inequality with target.
    // From farthest to nearest.
    float cur_alpha = alpha;
    while (cur_alpha >= (1.0f - 1e-5f) && to_evict == kInvalidID) {
      for (int i = (int) pool.size() - 1; i >= 0; --i) {
        if (pool[i].id == target_id) {
          tgt_idx = i;
          continue;
        }
        // Check if target occludes pool[i] or vice versa.
        if (pool[i].distance > target_center_dist) {
          // pool[i] -> center is the longest edge.
          if (get_occlude_factor(pool[i].distance, pool[i].tgt_dis) > cur_alpha) {
            to_evict = (uint32_t) i;
            break;
          }
        } else {
          // target -> center is the longest edge.
          if (get_occlude_factor(target_center_dist, pool[i].tgt_dis) > cur_alpha) {
            to_evict = tgt_idx;
            break;
          }
        }
      }
      cur_alpha /= 1.2f;
    }
    
    // bool evict_tgt = false;
    // std::vector<uint32_t> evicted_ids;
    // while (cur_alpha >= (1.0f - 1e-5f) && to_evict == kInvalidID) {
    //   // for (int i = (int) pool.size() - 1; i >= 0; --i) {
    //   for (int i = 0; i < (int) pool.size(); ++i) {

    //     if (pool[i].id == target_id) {
    //       tgt_idx = i;
    //       if (evict_tgt == true) {
    //         to_evict = tgt_idx;
    //         break;
    //       }
    //       continue;
    //     }
    //     if (pool[i].distance < target_center_dist && evict_tgt == false) {
    //       // target -> center is the longest edge.
    //       if (get_occlude_factor(target_center_dist, pool[i].tgt_dis) > cur_alpha) {
    //         // to_evict = tgt_idx;
    //         evict_tgt = true;
    //         // break;
    //       }
    //     } else {
          
    //       if (get_occlude_factor(pool[i].distance, pool[i].tgt_dis) > cur_alpha) {
    //         // to_evict = (uint32_t) i;
    //         // break;
    //         evicted_ids.push_back(pool[i].id);
    //       }
    //     }

    //   }
    //   cur_alpha /= 1.2f;
    // }


    auto finish = [&]() {
      nhood.clear();
      nhood.reserve(range);
      for (uint32_t i = 0; i < pool.size(); i++) {
        if (i != to_evict) {
          nhood.emplace_back(pool[i].id);
        }
      }
    };

    if (to_evict != kInvalidID) {
      finish();
      return;
    }

    // Fast path failed. The target is high quality.
    // Seek another low quality point to evict using full alpha-RNG check.
    // Copy sorted IDs for batch distance computation.
    std::vector<uint32_t> ids(pool.size());
    for (uint32_t i = 0; i < pool.size(); i++) {
      ids[i] = pool[i].id;
    }

    for (uint32_t start = 0; start < pool.size() - 1; ++start) {
      if (start == tgt_idx) {
        continue;
      }
      // Batch compute distances from pool[start] to all points after it.
      compute_distances(ids[start], ids.data() + start + 1, pool.size() - start - 1, center_dists.data() + start + 1);
      for (uint32_t t = start + 1; t < pool.size(); t++) {
        if (get_occlude_factor(pool[t].distance, center_dists[t]) > alpha) {
          to_evict = t;
          finish();
          return;
        }
      }
    }

    // All points satisfy alpha-RNG, evict the farthest.
    to_evict = pool.size() - 1;
    finish();
  }

  /* inter_insert():
   * This function tries to add reverse links from all the visited nodes to
   * the current node n.
   */
  template<typename T, typename TagT>
  std::pair<int, size_t> Index<T, TagT>::inter_insert(unsigned n, std::vector<unsigned> &pruned_list, const Parameters &parameter) {
    const auto range = parameter.R;
    const auto maxc = parameter.C;
    const auto alpha = parameter.alpha;

    assert(n >= 0 && n < _nd + _num_frozen_pts);

    const auto &src_pool = pruned_list;

    assert(!src_pool.empty());

    int cmp = 0;
    size_t occ_time = 0;
    for (auto des : src_pool) {
      /* des.id is the id of the neighbors of n */
      assert(des >= 0 && des < _max_points + _num_frozen_pts);
      /* des_pool contains the neighbors of the neighbors of n */
      auto &des_pool = _final_graph[des];
      std::vector<unsigned> copy_of_neighbors;
      bool prune_needed = false;
// #define DELTA_PRUNE
#ifdef DELTA_PRUNE
      {
        v2::LockGuard guard(_locks->wrlock(des));
        if (std::find(des_pool.begin(), des_pool.end(), n) == des_pool.end()) {
          // if (des_pool.size() > range) {
          // delta_prune_neighbors(parameter, des_pool, (uint32_t)des, (uint32_t)n);
          // }
          std::vector<Neighbor> result;
          des_pool.push_back(n);
          std::vector<float> occlude_factor(des_pool.size(), 0.0f);
          std::vector<Neighbor> pool;
          
          for (auto &id : des_pool) {
            float dist = _distance->compare(_data + _aligned_dim * (size_t) des,
            _data + _aligned_dim * (size_t) id, (unsigned) _aligned_dim);
            cmp ++;
            pool.emplace_back(Neighbor(id, dist, false));
          }
          pool.back().flag = true; 
            
          std::sort(pool.begin(), pool.end());
          cmp += occlude_list(pool, alpha, range, maxc, result, occlude_factor, true);

          des_pool.clear();
          for (auto &nbr : result) {
            des_pool.emplace_back(nbr.id);
          }
        }
      }
#else 
      {
        // v2::SparseWriteLockGuard<uint64_t> guard(&_locks, des);
        v2::LockGuard guard(_locks->wrlock(des));
        #ifdef DIRANN
        
        if (std::find(des_pool.begin(), des_pool.end(), n) == des_pool.end()) {
          // if (des_pool.size() < (uint64_t) (SLACK_FACTOR * range)) {
            des_pool.emplace_back(n);
            prune_needed = true;
          // } else {
            copy_of_neighbors = des_pool;
          //   prune_needed = true;
          // }
        }
        #else
        if (std::find(des_pool.begin(), des_pool.end(), n) == des_pool.end()) {
          if (des_pool.size() < (uint64_t) (SLACK_FACTOR * range)) {
            des_pool.emplace_back(n);
            prune_needed = false;
          } else {
            copy_of_neighbors = des_pool;
            prune_needed = true;
          }
        }
        #endif
      }  // des lock is released by this point

      if (prune_needed) {
        copy_of_neighbors.push_back(n);
        tsl::robin_set<unsigned> dummy_visited(0);
        std::vector<Neighbor> dummy_pool(0);

        size_t reserveSize = (size_t) (std::ceil(1.05 * SLACK_FACTOR * range));
        dummy_visited.reserve(reserveSize);
        dummy_pool.reserve(reserveSize);
        Timer timer;
        for (auto cur_nbr : copy_of_neighbors) {
          if (dummy_visited.find(cur_nbr) == dummy_visited.end() && cur_nbr != des) {
            float dist = _distance->compare(_data + _aligned_dim * (size_t) des,
                                            _data + _aligned_dim * (size_t) cur_nbr, (unsigned) _aligned_dim);
            cmp ++;
            dummy_pool.emplace_back(Neighbor(cur_nbr, dist, true));
            dummy_visited.insert(cur_nbr);
          }
        }
        std::vector<unsigned> new_out_neighbors;
        cmp += prune_neighbors(des, dummy_pool, parameter, new_out_neighbors);
        occ_time += timer.elapsed();
        {
          // v2::SparseWriteLockGuard<uint64_t> guard(&_locks, des);
          v2::LockGuard guard(_locks->wrlock(des));
          _final_graph[des].assign(new_out_neighbors.begin(), new_out_neighbors.end());
        }
      }
#endif
    }

    return {cmp, occ_time};
  }

  // one-pass graph building.
  template<typename T, typename TagT>
  void Index<T, TagT>::link(Parameters &parameters) {
    unsigned num_threads = parameters.num_threads;
    _saturate_graph = parameters.saturate_graph;
    unsigned L = parameters.L;  // Search list size
    const unsigned range = parameters.R;

    LOG(INFO) << "Parameters: " << "L: " << L << ", R: " << range
              << ", saturate_graph: " << (_saturate_graph ? "true" : "false") << ", num_threads: " << num_threads
              << ", alpha: " << parameters.alpha;
    if (num_threads != 0)
      omp_set_num_threads(num_threads);

    int64_t n_vecs_to_visit = _nd + _num_frozen_pts;
    _ep = _num_frozen_pts > 0 ? _max_points : calculate_entry_point();

    std::vector<unsigned> init_ids;
    init_ids.emplace_back(_ep);

    pipeann::Timer link_timer;
#pragma omp parallel for schedule(dynamic)
    for (int64_t node = 0; node < n_vecs_to_visit; node++) {
      // search.
      std::vector<Neighbor> pool;
      tsl::robin_set<unsigned> visited;
      pool.reserve(2 * L);
      visited.reserve(2 * L);
      get_expanded_nodes(node, L, init_ids, pool, visited);
      // remove the node itself from pool.
      for (auto it = pool.begin(); it != pool.end();) {
        if (it->id == node) {
          it = pool.erase(it);
        } else {
          ++it;
        }
      }
      // prune neighbors.
      std::vector<unsigned> pruned_list;
      prune_neighbors(node, pool, parameters, pruned_list);

      {
        // v2::SparseWriteLockGuard<uint64_t> guard(&_locks, node);
        v2::LockGuard guard(_locks->wrlock(node));
        _final_graph[node].assign(pruned_list.begin(), pruned_list.end());
      }

      inter_insert(node, pruned_list, parameters);

      if (node % 100000 == 0) {
        std::cerr << "\r" << (100.0 * node) / (n_vecs_to_visit) << "% of index build completed.";
      }
    }

    if (_nd > 0) {
      LOG(INFO) << "Starting final cleanup..";
    }
#pragma omp parallel for schedule(dynamic, 65536)
    for (int64_t node_ctr = 0; node_ctr < n_vecs_to_visit; node_ctr++) {
      auto node = node_ctr;
      if (_final_graph[node].size() > range) {
        tsl::robin_set<unsigned> dummy_visited(0);
        std::vector<Neighbor> dummy_pool(0);
        std::vector<unsigned> new_out_neighbors;

        for (auto cur_nbr : _final_graph[node]) {
          if (dummy_visited.find(cur_nbr) == dummy_visited.end() && cur_nbr != node) {
            float dist = _distance->compare(_data + _aligned_dim * (size_t) node,
                                            _data + _aligned_dim * (size_t) cur_nbr, (unsigned) _aligned_dim);
            dummy_pool.emplace_back(Neighbor(cur_nbr, dist, true));
            dummy_visited.insert(cur_nbr);
          }
        }
        prune_neighbors(node, dummy_pool, parameters, new_out_neighbors);

        _final_graph[node].clear();
        for (auto id : new_out_neighbors)
          _final_graph[node].emplace_back(id);
      }
    }
    if (_nd > 0) {
      LOG(INFO) << "done. Link time: " << ((double) link_timer.elapsed() / (double) 1000000) << "s";
    }
  }

  template<typename T, typename TagT>
  void Index<T, TagT>::build(const char *filename, const size_t num_points_to_load, Parameters &parameters,
                             const std::vector<TagT> &tags) {
    if (!file_exists(filename)) {
      LOG(ERROR) << "Data file " << filename << " does not exist!!! Exiting....";
      crash();
    }

    size_t file_num_points, file_dim;
    if (filename == nullptr) {
      LOG(INFO) << "Starting with an empty index.";
      _nd = 0;
    } else {
      pipeann::get_bin_metadata(filename, file_num_points, file_dim);
      if (file_num_points > _max_points || num_points_to_load > file_num_points) {
        LOG(ERROR) << "ERROR: Driver requests loading " << num_points_to_load << " points and file has "
                   << file_num_points << " points, but "
                   << "index can support only " << _max_points << " points as specified in constructor.";
        crash();
      }
      if (file_dim != _dim) {
        LOG(ERROR) << "ERROR: Driver requests loading " << _dim << " dimension,"
                   << "but file has " << file_dim << " dimension.";
        crash();
      }

      copy_aligned_data_from_file<T>(std::string(filename), _data, file_num_points, file_dim, _aligned_dim);

      LOG(INFO) << "Loading only first " << num_points_to_load << " from file.. ";
      _nd = num_points_to_load;

      if (_enable_tags && tags.size() != num_points_to_load) {
        LOG(ERROR) << "ERROR: Driver requests loading " << num_points_to_load << " points from file,"
                   << "but tags vector is of size " << tags.size() << ".";
        crash();
      }
      if (_enable_tags) {
        for (size_t i = 0; i < tags.size(); ++i) {
          _tag_to_location[tags[i]] = (unsigned) i;
          _location_to_tag[(unsigned) i] = tags[i];
        }
      }
    }

    generate_frozen_point();
    link(parameters);  // Primary func for creating nsg graph

    size_t max = 0, min = 1 << 30, total = 0, cnt = 0;
    for (size_t i = 0; i < _nd; i++) {
      auto &pool = _final_graph[i];
      max = (std::max)(max, pool.size());
      min = (std::min)(min, pool.size());
      total += pool.size();
      if (pool.size() < 2)
        cnt++;
    }
    if (min > max)
      min = max;
    if (_nd > 0) {
      LOG(INFO) << "Index built with degree: max:" << max << "  avg:" << (float) total / (float) (_nd + _num_frozen_pts)
                << "  min:" << min << "  count(deg<2):" << cnt;
    }
    _width = (std::max)((unsigned) max, _width);
    _has_built = true;
  }

  template<typename T, typename TagT>
  void Index<T, TagT>::build(const char *filename, const size_t num_points_to_load, Parameters &parameters,
                             const char *tag_filename) {
    if (!file_exists(filename)) {
      LOG(ERROR) << "Data file provided " << filename << " does not exist.";
      crash();
    }

    size_t file_num_points, file_dim;
    if (filename == nullptr) {
      LOG(INFO) << "Starting with an empty index.";
      _nd = 0;
    } else {
      pipeann::get_bin_metadata(filename, file_num_points, file_dim);
      if (file_num_points > _max_points || num_points_to_load > file_num_points) {
        LOG(ERROR) << "ERROR: Driver requests loading " << num_points_to_load << " points and file has "
                   << file_num_points << " points, but "
                   << "index can support only " << _max_points << " points as specified in constructor.";
        crash();
      }
      if (file_dim != _dim) {
        LOG(ERROR) << "ERROR: Driver requests loading " << _dim << " dimension,"
                   << "but file has " << file_dim << " dimension.";
        crash();
      }

      copy_aligned_data_from_file<T>(std::string(filename), _data, file_num_points, file_dim, _aligned_dim);

      LOG(INFO) << "Loading only first " << num_points_to_load << " from file.. ";
      _nd = num_points_to_load;
      if (_enable_tags) {
        if (tag_filename == nullptr) {
          for (unsigned i = 0; i < num_points_to_load; i++) {
            _tag_to_location[i] = i;
            _location_to_tag[i] = i;
          }
        } else {
          if (file_exists(tag_filename)) {
            LOG(INFO) << "Loading tags from " << tag_filename << " for vamana index build";
            TagT *tag_data = nullptr;
            size_t npts, ndim;
            pipeann::load_bin(tag_filename, tag_data, npts, ndim);
            if (npts != num_points_to_load) {
              std::stringstream sstream;
              sstream << "Loaded " << npts << " tags instead of expected number: " << num_points_to_load;
              LOG(ERROR) << sstream.str();
              crash();
            }
            for (size_t i = 0; i < npts; i++) {
              _tag_to_location[tag_data[i]] = (unsigned) i;
              _location_to_tag[(unsigned) i] = tag_data[i];
            }
            delete[] tag_data;
          } else {
            LOG(ERROR) << "Tag file " << tag_filename << " does not exist. Exiting...";
            crash();
          }
        }
      }
    }

    generate_frozen_point();
    link(parameters);  // Primary func for creating nsg graph

    size_t max = 0, min = 1 << 30, total = 0, cnt = 0;
    for (size_t i = 0; i < _nd; i++) {
      auto &pool = _final_graph[i];
      max = (std::max)(max, pool.size());
      min = (std::min)(min, pool.size());
      total += pool.size();
      if (pool.size() < 2)
        cnt++;
    }
    if (min > max)
      min = max;
    if (_nd > 0) {
      LOG(INFO) << "Index built with degree: max:" << max << "  avg:" << (float) total / (float) (_nd + _num_frozen_pts)
                << "  min:" << min << "  count(deg<2):" << cnt;
    }
    _width = (std::max)((unsigned) max, _width);
    _has_built = true;
  }

  template<typename T, typename TagT>
  std::pair<uint32_t, uint32_t> Index<T, TagT>::search(const T *query, const size_t K, const unsigned L,
                                                       std::vector<NeighborTag<TagT>> &best_K_tags) {
    std::shared_lock<std::shared_timed_mutex> ulock(_update_lock);
    assert(best_K_tags.size() == 0);
    std::vector<unsigned> init_ids;
    tsl::robin_set<unsigned> visited(10 * L);
    std::vector<Neighbor> best, expanded_nodes_info;
    tsl::robin_set<unsigned> expanded_nodes_ids;

    if (init_ids.size() == 0) {
      init_ids.emplace_back(_ep);
    }

    T *aligned_query;
    size_t allocSize = _aligned_dim * sizeof(T);
    alloc_aligned(((void **) &aligned_query), allocSize, 8 * sizeof(T));
    memset(aligned_query, 0, _aligned_dim * sizeof(T));
    memcpy(aligned_query, query, _dim * sizeof(T));
    auto retval =
        iterate_to_fixed_point(aligned_query, L, init_ids, expanded_nodes_info, expanded_nodes_ids, best, false);

    std::shared_lock<std::shared_timed_mutex> lock(_tag_lock);
    for (auto iter : best) {
      if (_location_to_tag.find(iter.id) != _location_to_tag.end())
        best_K_tags.emplace_back(NeighborTag<TagT>(_location_to_tag[iter.id], iter.distance));
      if (best_K_tags.size() == K)
        break;
    }
    aligned_free(aligned_query);
    return retval;
  }

  template<typename T, typename TagT>
  std::pair<uint32_t, uint32_t> Index<T, TagT>::search(const T *query, const size_t K, const unsigned L,
                                                       unsigned *indices, float *distances, QueryStats *stats) {
    std::vector<unsigned> init_ids;
    tsl::robin_set<unsigned> visited(10 * L);
    std::vector<Neighbor> best_L_nodes, expanded_nodes_info;
    tsl::robin_set<unsigned> expanded_nodes_ids;

    std::shared_lock<std::shared_timed_mutex> lock(_update_lock);

    if (init_ids.size() == 0) {
      init_ids.emplace_back(_ep);
    }
    T *aligned_query;
    size_t allocSize = _aligned_dim * sizeof(T);
    alloc_aligned(((void **) &aligned_query), allocSize, 8 * sizeof(T));
    memset(aligned_query, 0, _aligned_dim * sizeof(T));
    memcpy(aligned_query, query, _dim * sizeof(T));
    auto retval = iterate_to_fixed_point(aligned_query, L, init_ids, expanded_nodes_info, expanded_nodes_ids,
                                         best_L_nodes, true, stats);

    size_t pos = 0;
    for (auto it : best_L_nodes) {
      if (it.id < _max_points) {
        indices[pos] = it.id;
        if (distances != nullptr)
          distances[pos] = it.distance;
        pos++;
      }
      if (pos == K)
        break;
    }
    aligned_free(aligned_query);
    return retval;
  }

  template<typename T, typename TagT>
  std::pair<uint32_t, uint32_t> Index<T, TagT>::search(const T *query, const uint64_t K, const unsigned L,
                                                       std::vector<unsigned> init_ids, uint64_t *indices,
                                                       float *distances) {
    tsl::robin_set<unsigned> visited(10 * L);
    std::vector<Neighbor> best_L_nodes, expanded_nodes_info;
    tsl::robin_set<unsigned> expanded_nodes_ids;

    std::shared_lock<std::shared_timed_mutex> lock(_update_lock);

    if (init_ids.size() == 0) {
      init_ids.emplace_back(_ep);
    }
    T *aligned_query;
    size_t allocSize = _aligned_dim * sizeof(T);
    alloc_aligned(((void **) &aligned_query), allocSize, 8 * sizeof(T));
    memset(aligned_query, 0, _aligned_dim * sizeof(T));
    memcpy(aligned_query, query, _dim * sizeof(T));
    auto retval = iterate_to_fixed_point(aligned_query, (unsigned) L, init_ids, expanded_nodes_info, expanded_nodes_ids,
                                         best_L_nodes);

    size_t pos = 0;
    for (auto it : best_L_nodes) {
      indices[pos] = it.id;
      distances[pos] = it.distance;
      pos++;
      if (pos == K)
        break;
    }
    aligned_free(aligned_query);
    return retval;
  }

  template<typename T, typename TagT>
  size_t Index<T, TagT>::search_with_tags(const T *query, const uint64_t K, const unsigned L, TagT *tags,
                                          float *distances, std::vector<T *> &res_vectors, QueryStats *stats) {
    uint32_t *indices = new unsigned[L];
    float *dist_interim = new float[L];
    search(query, L, L, indices, dist_interim, stats);

    std::shared_lock<std::shared_timed_mutex> ulock(_update_lock);
    std::shared_lock<std::shared_timed_mutex> lock(_tag_lock);
    size_t pos = 0;
    for (int i = 0; i < (int) L; ++i)
      if (_location_to_tag.find(indices[i]) != _location_to_tag.end()) {
        tags[pos] = _location_to_tag[indices[i]];
        res_vectors[i] = _data + indices[i] * _aligned_dim;

        if (distances != nullptr)
          distances[pos] = dist_interim[i];
        pos++;
        if (pos == K)
          break;
      }
    delete[] indices;
    delete[] dist_interim;
    return pos;
  }

  template<typename T, typename TagT>
  size_t Index<T, TagT>::search_with_tags(const T *query, const size_t K, const unsigned L, TagT *tags,
                                          float *distances, QueryStats *stats) {
    uint32_t *indices = new unsigned[L];
    float *dist_interim = new float[L];
    search(query, L, L, indices, dist_interim, stats);

    std::shared_lock<std::shared_timed_mutex> ulock(_update_lock);
    std::shared_lock<std::shared_timed_mutex> lock(_tag_lock);
    size_t pos = 0;
    for (int i = 0; i < (int) L; ++i) {
      if (_location_to_tag.find(indices[i]) != _location_to_tag.end()) {
        tags[pos] = _location_to_tag[indices[i]];
        if (distances != nullptr)
          distances[pos] = dist_interim[i];
        pos++;
        if (pos == K)
          break;
      }
    }
    delete[] indices;
    delete[] dist_interim;
    return pos;
  }

  template<typename T, typename TagT>
  uint32_t Index<T, TagT>::search_with_tags_fast(const T *node_coords, const unsigned Lsize, TagT *tags, float *dists) {
    std::vector<Neighbor> best_L_nodes(Lsize + 1);
    for (unsigned i = 0; i < Lsize + 1; i++) {
      best_L_nodes[i].distance = std::numeric_limits<float>::max();
    }

    unsigned l = 0;
    Neighbor nn;
    tsl::robin_set<unsigned> inserted_into_pool;
    inserted_into_pool.reserve(Lsize * 20);

    auto id = _ep;
    nn = Neighbor(id, _distance->compare(_data + _aligned_dim * (size_t) id, node_coords, _aligned_dim), true);
    inserted_into_pool.insert(id);
    best_L_nodes[l++] = nn;

    unsigned k = 0, cmps = 0;

    while (k < l) {
      unsigned nk = l;

      if (best_L_nodes[k].flag) {
        best_L_nodes[k].flag = false;
        auto n = best_L_nodes[k].id;

        auto &cur_v = _final_graph[n];
        for (unsigned m = 0; m < cur_v.size(); ++m) {
          unsigned id = cur_v[m];
          if (inserted_into_pool.find(id) == inserted_into_pool.end()) {
            inserted_into_pool.insert(id);

            if ((m + 1) < cur_v.size()) {
              auto nextn = cur_v[m + 1];
              pipeann::prefetch_vector((const char *) _data + _aligned_dim * (size_t) nextn, sizeof(T) * _aligned_dim);
            }

            float dist = _distance->compare(node_coords, _data + _aligned_dim * (size_t) id, (unsigned) _aligned_dim);
            cmps++;

            if (dist >= best_L_nodes[l - 1].distance && (l == Lsize))
              continue;

            Neighbor nn(id, dist, true);
            unsigned r = InsertIntoPool(best_L_nodes.data(), l, nn);
            if (l < Lsize)
              ++l;
            if (r < nk)
              nk = r;
          }
        }

        if (nk <= k)
          k = nk;
        else
          ++k;
      } else {
        k++;
      }
    }
    for (uint32_t i = 0; i < Lsize; ++i) {
      tags[i] = _location_to_tag[best_L_nodes[i].id];
      dists[i] = best_L_nodes[i].distance;
    }
    return cmps;
  }

  template<typename T, typename TagT>
  size_t Index<T, TagT>::get_num_points() {
    return _nd;
  }

  template<typename T, typename TagT>
  T *Index<T, TagT>::get_data() {
    if (_num_frozen_pts > 0) {
      T *ret_data = nullptr;
      size_t allocSize = _nd * _aligned_dim * sizeof(T);
      alloc_aligned(((void **) &ret_data), allocSize, 8 * sizeof(T));
      memset(ret_data, 0, _nd * _aligned_dim * sizeof(T));
      memcpy(ret_data, _data, _nd * _aligned_dim * sizeof(T));
      return ret_data;
    }
    return _data;
  }

  /*************************************************
   *      Support for Incremental Update
   *************************************************/

  // in case we add ''frozen'' auxiliary points to the dataset, these are not
  // visible to external world, we generate them here and update our dataset
  template<typename T, typename TagT>
  int Index<T, TagT>::generate_frozen_point() {
    if (_num_frozen_pts == 0)
      return 0;

    if (_nd == 0) {
      memset(_data + (_max_points) *_aligned_dim, 0, _aligned_dim * sizeof(T));
      return 1;
    }
    size_t res = calculate_entry_point();
    memcpy(_data + _max_points * _aligned_dim, _data + res * _aligned_dim, _aligned_dim * sizeof(T));
    return 0;
  }

  template<typename T, typename TagT>
  int Index<T, TagT>::enable_delete() {
    assert(_enable_tags);

    if (!_enable_tags) {
      LOG(ERROR) << "Tags must be instantiated for deletions";
      return -2;
    }

    if (_data_compacted) {
      for (unsigned slot = (unsigned) _nd; slot < _max_points; ++slot) {
        _empty_slots.insert(slot);
      }
    }

    _lazy_done = false;
    _eager_done = false;
    return 0;
  }

  template<typename T, typename TagT>
  size_t Index<T, TagT>::consolidate_deletes(const Parameters &parameters) {
    if (_eager_done) {
      LOG(INFO) << "In consolidate_deletes(), _eager_done is true. So exiting.";
      return 0;
    }

    omp_set_num_threads(N_DELETE_THREADS);
    LOG(INFO) << "Empty slots size: " << _empty_slots.size() << " _nd: " << _nd << " max_points: " << _max_points;
    assert(_enable_tags);
    assert(_delete_set.size() <= _nd);
    assert(_empty_slots.size() + _nd == _max_points);

    const unsigned range = parameters.R;
    const unsigned maxc = parameters.C;
    const float alpha = parameters.alpha;

    uint64_t total_pts = _max_points + _num_frozen_pts;
    unsigned block_size = 1 << 4;
    int64_t total_blocks = DIV_ROUND_UP(total_pts, block_size);
    std::atomic<size_t> total_cmp(0), total_occ(0), total_occ_cnt(0);

    auto start = std::chrono::high_resolution_clock::now();
#pragma omp parallel for schedule(dynamic)
    for (int64_t block = 0; block < total_blocks; ++block) {

    size_t cmp = 0;
    size_t occ_time = 0;
#ifdef DISKANN_WITH_INCREMENTAL_PRUNING
      tsl::robin_set<unsigned> candidate_set0, candidate_set1;
      std::vector<Neighbor> expanded_nghrs;
      std::vector<Neighbor> result;

      for (int64_t i = block * block_size;
           i < (int64_t) ((block + 1) * block_size) && i < (int64_t) (_max_points + _num_frozen_pts); i++) {
        if ((_delete_set.find((uint32_t) i) == _delete_set.end()) &&
            (_empty_slots.find((uint32_t) i) == _empty_slots.end())) {
          candidate_set0.clear();
          candidate_set1.clear();
          expanded_nghrs.clear();
          result.clear();

          bool modify = false;
          for (auto ngh : _final_graph[(uint32_t) i]) {
            if (_delete_set.find(ngh) != _delete_set.end()) {
              modify = true;

              // Add outgoing links from
              for (auto j : _final_graph[ngh])
                if (_delete_set.find(j) == _delete_set.end())
                  candidate_set1.insert(j);
            } else {
              candidate_set0.insert(ngh);
            }
          }
          if (modify) {
            Timer timer;
            for (auto j : candidate_set1) {
              expanded_nghrs.push_back(
                  Neighbor(j,
                           _distance->compare(_data + _aligned_dim * i, _data + _aligned_dim * (size_t) j,
                                              (unsigned) _aligned_dim),
                           true));
              cmp ++;
            }

            std::sort(expanded_nghrs.begin(), expanded_nghrs.end());
            std::vector<float> occlude_factor(expanded_nghrs.size(), 0);

            cmp += occlude_list(expanded_nghrs, alpha, range, maxc, result, occlude_factor);
            total_occ_cnt++;

            expanded_nghrs.swap(result);
            result.clear();

            for (auto j : candidate_set0) {
              expanded_nghrs.push_back(
                  Neighbor(j,
                           _distance->compare(_data + _aligned_dim * i, _data + _aligned_dim * (size_t) j,
                                              (unsigned) _aligned_dim),
                           false));
              cmp ++;
            }
            std::sort(expanded_nghrs.begin(), expanded_nghrs.end());
            occlude_factor.assign(expanded_nghrs.size(), 0);

            cmp += occlude_list(expanded_nghrs, alpha, range, maxc, result, occlude_factor, true);
            
            occ_time += timer.elapsed();

            _final_graph[(uint32_t) i].clear();
            for (auto j : result) {
              if (j.id != (uint32_t) i && (_delete_set.find(j.id) == _delete_set.end()))
                _final_graph[(uint32_t) i].push_back(j.id);
            }
          }
        }
      }
      total_cmp += cmp;
      total_occ += occ_time;
#else
      tsl::robin_set<unsigned> candidate_set;
      std::vector<Neighbor> expanded_nghrs;
      std::vector<Neighbor> result;

      for (int64_t i = block * block_size;
           i < (int64_t) ((block + 1) * block_size) && i < (int64_t) (_max_points + _num_frozen_pts); i++) {
        if ((_delete_set.find((uint32_t) i) == _delete_set.end()) &&
            (_empty_slots.find((uint32_t) i) == _empty_slots.end())) {
          candidate_set.clear();
          expanded_nghrs.clear();
          result.clear();
          
          bool modify = false;
          for (auto ngh : _final_graph[(uint32_t) i]) {
            if (_delete_set.find(ngh) != _delete_set.end()) {
              modify = true;

              // Add outgoing links from
              for (auto j : _final_graph[ngh])
                if (_delete_set.find(j) == _delete_set.end())
                  candidate_set.insert(j);
            } else {
              candidate_set.insert(ngh);
            }
          }

          if (modify) {
            Timer timer;
            for (auto j : candidate_set) {
              expanded_nghrs.push_back(
                  Neighbor(j,
                           _distance->compare(_data + _aligned_dim * i, _data + _aligned_dim * (size_t) j,
                                              (unsigned) _aligned_dim),
                           true));
              cmp ++;
            }

            std::sort(expanded_nghrs.begin(), expanded_nghrs.end());

            cmp += occlude_list(expanded_nghrs, alpha, range, maxc, result);
            total_occ_cnt++;

            occ_time += timer.elapsed();

            _final_graph[(uint32_t) i].clear();
            for (auto j : result) {
              if (j.id != (uint32_t) i && (_delete_set.find(j.id) == _delete_set.end()))
                _final_graph[(uint32_t) i].push_back(j.id);
            }
          }
        }
      }
      total_cmp += cmp;
      total_occ += occ_time;
#endif
    }

    for (auto iter : _delete_set) {
      _empty_slots.insert(iter);
    }
    _nd -= _delete_set.size();

    _data_compacted = _delete_set.size() == 0;

    auto stop = std::chrono::high_resolution_clock::now();
    LOG(INFO) << "Time taken for consolidate_deletes() "
              << std::chrono::duration_cast<std::chrono::duration<double>>(stop - start).count() << "s.";

    return _nd;
  }

#define IP_DISKANN_DEFER_PRUNE
  // ip-diskann
  template<typename T, typename TagT>
  size_t Index<T, TagT>::consolidate_deletes_ip_diskann(const Parameters &parameters) {
    if (_eager_done) {
      LOG(INFO) << "In consolidate_deletes(), _eager_done is true. So exiting.";
      return 0;
    }

    omp_set_num_threads(N_DELETE_THREADS);
    LOG(INFO) << "Consolidate deletes threads num " << omp_get_max_threads();
    LOG(INFO) << "Inside Index::consolidate_deletes()";
    LOG(INFO) << "Empty slots size: " << _empty_slots.size() << " _nd: " << _nd << " max_points: " << _max_points;
    
    assert(_enable_tags);
    assert(_delete_set.size() <= _nd);
    assert(_empty_slots.size() + _nd == _max_points);

    const unsigned range = parameters.R;
    const unsigned Lindex = parameters.L;
    const unsigned c = 3;

    std::atomic<size_t> total_cmp(0), total_occ(0), total_occ_cnt(0);
    std::atomic<size_t> total_search_time(0);
    uint64_t total_pts = _max_points + _num_frozen_pts;
  #ifdef IP_DISKANN_DEFER_PRUNE
    std::vector<uint8_t> need_prune_nodes(total_pts, 0);
  #endif

    auto start = std::chrono::high_resolution_clock::now();
    Timer delete_timer;

    std::vector<uint32_t> deleted_nodes(_delete_set.begin(), _delete_set.end());

    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < deleted_nodes.size(); ++i) {
      uint32_t p = deleted_nodes[i];

      std::vector<Neighbor> pool; // candidates
      tsl::robin_set<unsigned> visited;
      std::vector<unsigned> init_ids;

      size_t cmp = 0;
      size_t occ_time = 0;
      size_t search_time = 0;
      Timer timer;
      auto retval = get_expanded_nodes(p, 75, init_ids, pool, visited);
      cmp += retval.second;
      search_time += timer.elapsed();

      std::vector<unsigned> N_in_p;
      for (auto z : visited) {
        if (z == p) continue;
        bool has_p = false;
        {
          v2::LockGuard guard(_locks->rdlock(z));
          for (auto ngh : _final_graph[z]) {
            if (ngh == p) { has_p = true; break; }
          }
        }
        if (has_p) N_in_p.push_back(z);
      }
      
      for (auto z : N_in_p) {
        std::vector<Neighbor> C_z;
        for (auto cand : pool) {
          if (cand.id != z && cand.id != p && _delete_set.find(cand.id) == _delete_set.end()) {
            float dist = _distance->compare(_data + _aligned_dim * z, _data + _aligned_dim * cand.id, (unsigned) _aligned_dim);
            C_z.push_back(Neighbor(cand.id, dist, true));
            cmp++;
          }
        }
        std::sort(C_z.begin(), C_z.end());
        
#ifdef IP_DISKANN_DEFER_PRUNE
        {
          v2::LockGuard guard(_locks->wrlock(z));
          tsl::robin_set<unsigned> unique_nghrs;
          for (auto ngh : _final_graph[z]) {
            if (ngh != p) unique_nghrs.insert(ngh);
          }
          // N_out(z) <- N_out(z) U C_z \ {p}
          for (size_t k = 0; k < std::min((size_t)c, C_z.size()); ++k) {
            unique_nghrs.insert(C_z[k].id);
          }

          _final_graph[z].assign(unique_nghrs.begin(), unique_nghrs.end());
          if (_final_graph[z].size() > range) {
#pragma omp atomic write
            need_prune_nodes[z] = 1;
          }
        }
#else
        std::vector<Neighbor> expanded_nghrs;
        bool need_prune = false;
        {
          v2::LockGuard guard(_locks->wrlock(z));
          tsl::robin_set<unsigned> unique_nghrs;
          for (auto ngh : _final_graph[z]) {
            if (ngh != p) unique_nghrs.insert(ngh);
          }
          for (size_t k = 0; k < std::min((size_t)c, C_z.size()); ++k) {
            unique_nghrs.insert(C_z[k].id);
          }

          if (unique_nghrs.size() > range) {
            need_prune = true;
            for (auto ngh : unique_nghrs) {
              float dist = _distance->compare(_data + _aligned_dim * z, _data + _aligned_dim * ngh, (unsigned) _aligned_dim);
              expanded_nghrs.push_back(Neighbor(ngh, dist, true));
              cmp++;
            }
          } else {
            _final_graph[z].assign(unique_nghrs.begin(), unique_nghrs.end());
          }
        }

        if (need_prune) {
          std::sort(expanded_nghrs.begin(), expanded_nghrs.end());
          std::vector<unsigned> pruned_list;
          Timer timer;
          cmp += prune_neighbors(z, expanded_nghrs, parameters, pruned_list);
          
          total_occ_cnt ++;
          occ_time += timer.elapsed();

          v2::LockGuard guard(_locks->wrlock(z));
          _final_graph[z] = pruned_list;
        }
#endif
      }
      
      std::vector<unsigned> N_out_p;
      {
        v2::LockGuard guard(_locks->rdlock(p));
        N_out_p = _final_graph[p];
      }

      for (auto w : N_out_p) {
        if (w == p || _delete_set.find(w) != _delete_set.end()) continue;
        
        std::vector<Neighbor> C_w;
        for (auto cand : pool) {
          if (cand.id != w && cand.id != p && _delete_set.find(cand.id) == _delete_set.end()) {
            float dist = _distance->compare(_data + _aligned_dim * w, _data + _aligned_dim * cand.id, (unsigned) _aligned_dim);
            C_w.push_back(Neighbor(cand.id, dist, true));
            cmp++;
          }
        }
        std::sort(C_w.begin(), C_w.end());

        for (size_t k = 0; k < std::min((size_t)c, C_w.size()); ++k) {
          unsigned y = C_w[k].id;
#ifdef IP_DISKANN_DEFER_PRUNE
          {
            v2::LockGuard guard(_locks->wrlock(y));
            bool has_w = false;
            for (auto ngh : _final_graph[y]) {
              if (ngh == w) { has_w = true; break; }
            }
            if (!has_w) {
              _final_graph[y].push_back(w);
            }
            if (_final_graph[y].size() > range) {
#pragma omp atomic write
              need_prune_nodes[y] = 1;
            }
          }
#else
          std::vector<Neighbor> expanded_nghrs;
          bool need_prune = false;
          
          {
            v2::LockGuard guard(_locks->wrlock(y));
            bool has_w = false;
            for (auto ngh : _final_graph[y]) {
              if (ngh == w) { has_w = true; break; }
            }
            if (!has_w) {
              _final_graph[y].push_back(w);
              if (_final_graph[y].size() > range) {
                need_prune = true;
                for (auto ngh : _final_graph[y]) {
                  float dist = _distance->compare(_data + _aligned_dim * y, _data + _aligned_dim * ngh, (unsigned) _aligned_dim);
                  expanded_nghrs.push_back(Neighbor(ngh, dist, true));
                  cmp++;
                }
              }
            }
          }

          if (need_prune) {
            std::sort(expanded_nghrs.begin(), expanded_nghrs.end());
            std::vector<unsigned> pruned_list;
            Timer timer;
            cmp += prune_neighbors(y, expanded_nghrs, parameters, pruned_list);
            total_occ_cnt ++;

            occ_time += timer.elapsed();

            v2::LockGuard guard(_locks->wrlock(y));
            _final_graph[y] = pruned_list;
          }
#endif
        }
      }

      total_cmp += cmp;
      total_occ += occ_time;
      total_search_time += search_time;
    }

    LOG(INFO) << "Time 1 (IP-DiskANN In-place patch) elapsed " << (delete_timer.elapsed() / 1e6) << "s";
    delete_timer.reset();

#ifdef IP_DISKANN_DEFER_PRUNE
    // Deferred global prune: prune all nodes marked in Stage 1 once patching is complete.
#pragma omp parallel for schedule(dynamic, 1024)
    for (uint64_t i = 0; i < total_pts; ++i) {
      if (need_prune_nodes[i] == 0) {
        continue;
      }
      if (_delete_set.find((uint32_t) i) != _delete_set.end()) {
        continue;
      }

      std::vector<unsigned> cur_nghrs;
      {
        v2::LockGuard guard(_locks->rdlock((uint32_t) i));
        if (_final_graph[i].size() <= range) {
          continue;
        }
        cur_nghrs = _final_graph[i];
      }

      if (cur_nghrs.size() <= range) {
        continue;
      }

      std::vector<Neighbor> expanded_nghrs;
      expanded_nghrs.reserve(cur_nghrs.size());

      size_t cmp = 0;
      for (auto ngh : cur_nghrs) {
        float dist = _distance->compare(_data + _aligned_dim * i, _data + _aligned_dim * ngh, (unsigned) _aligned_dim);
        expanded_nghrs.push_back(Neighbor(ngh, dist, true));
        cmp++;
      }

      std::sort(expanded_nghrs.begin(), expanded_nghrs.end());
      std::vector<unsigned> pruned_list;
      Timer timer;
      cmp += prune_neighbors((unsigned) i, expanded_nghrs, parameters, pruned_list);
      size_t occ_time = timer.elapsed();

      {
        v2::LockGuard guard(_locks->wrlock((uint32_t) i));
        _final_graph[i] = pruned_list;
      }

      total_cmp += cmp;
      total_occ += occ_time;
      total_occ_cnt++;
    }

    LOG(INFO) << "Time 1.5 (IP-DiskANN deferred prune) elapsed " << (delete_timer.elapsed() / 1e6) << "s";
    delete_timer.reset();
#endif

    #pragma omp parallel for schedule(dynamic, 1024)
    for (uint64_t i = 0; i < total_pts; i++) {
      if (_delete_set.find((uint32_t)i) != _delete_set.end()) {
        _final_graph[i].clear();
        continue;
      }

      auto& nbrs = _final_graph[i];
      if (nbrs.empty()) continue;

      size_t j = 0;
      for (size_t k = 0; k < nbrs.size(); k++) {
        if (_delete_set.find(nbrs[k]) == _delete_set.end()) {
          nbrs[j++] = nbrs[k];
        }
      }

      if (j != nbrs.size()) {
        nbrs.resize(j);
      }
    }

    for (auto iter : _delete_set) {
      _empty_slots.insert(iter);
    }
    _nd -= _delete_set.size();

    _data_compacted = _delete_set.size() == 0;

    auto stop = std::chrono::high_resolution_clock::now();
    LOG(INFO) << "Time taken for consolidate_deletes() "
              << std::chrono::duration_cast<std::chrono::duration<double>>(stop - start).count() << "s.";

    return _nd;
  }

  //Greator
  template<typename T, typename TagT>
  size_t Index<T, TagT>::consolidate_deletes_greator(const Parameters &parameters) {
    if (_eager_done) {
      LOG(INFO) << "In consolidate_deletes(), _eager_done is true. So exiting.";
      return 0;
    }

    omp_set_num_threads(N_DELETE_THREADS);
    LOG(INFO) << "Empty slots size: " << _empty_slots.size() << " _nd: " << _nd << " max_points: " << _max_points;
    assert(_enable_tags);
    assert(_delete_set.size() <= _nd);
    assert(_empty_slots.size() + _nd == _max_points);

    const unsigned range = parameters.R;
    const unsigned maxc = parameters.C;
    const float alpha = parameters.alpha;
    
    const unsigned T_threshold = 2; 
    LOG(INFO) << "T_threshold: " << T_threshold;

    uint64_t total_pts = _max_points + _num_frozen_pts;
    unsigned block_size = 1 << 10;
    int64_t total_blocks = DIV_ROUND_UP(total_pts, block_size);
    std::atomic<size_t> total_cmp(0), total_occ(0), total_occ_cnt(0);

    auto start = std::chrono::high_resolution_clock::now();
#pragma omp parallel for schedule(dynamic)
    for (int64_t block = 0; block < total_blocks; ++block) {

      size_t cmp = 0;
      size_t occ_time = 0;
      std::vector<Neighbor> expanded_nghrs;
      std::vector<Neighbor> result;

      for (int64_t i = block * block_size;
           i < (int64_t) ((block + 1) * block_size) && i < (int64_t) (_max_points + _num_frozen_pts); i++) {
        if ((_delete_set.find((uint32_t) i) == _delete_set.end()) &&
            (_empty_slots.find((uint32_t) i) == _empty_slots.end())) {
          
          std::vector<unsigned> D;
          std::vector<unsigned> C;
          
          for (auto ngh : _final_graph[(uint32_t) i]) {
            if (_delete_set.find(ngh) != _delete_set.end()) {
              D.push_back(ngh);
            } else {
              C.push_back(ngh);
            }
          }

          if (!D.empty()) {
            Timer timer;
            
            if (D.size() < T_threshold) {
              unsigned slot = range - C.size();
              unsigned k_slot = std::max((unsigned)(slot / (D.size() + C.size())), (unsigned)1);
              
              tsl::robin_set<unsigned> C_set(C.begin(), C.end());
              
              for (auto v : D) {
                std::vector<Neighbor> v_candidates;
                for (auto v_ngh : _final_graph[v]) {
                  if (_delete_set.find(v_ngh) == _delete_set.end() && v_ngh != (uint32_t)i && C_set.find(v_ngh) == C_set.end()) {
                     float dist = _distance->compare(_data + _aligned_dim * i, _data + _aligned_dim * (size_t) v_ngh, (unsigned) _aligned_dim);
                     v_candidates.push_back(Neighbor(v_ngh, dist, true));
                     cmp++;
                  }
                }
                
                std::sort(v_candidates.begin(), v_candidates.end());
                for (size_t k = 0; k < std::min((size_t)k_slot, v_candidates.size()); ++k) {
                  C_set.insert(v_candidates[k].id);
                }
              }
              
              _final_graph[(uint32_t) i].clear();
              for (auto j : C_set) {
                _final_graph[(uint32_t) i].push_back(j);
              }
              
            } else {
              tsl::robin_set<unsigned> candidate_set(C.begin(), C.end());
              
              for (auto v : D) {
                for (auto v_ngh : _final_graph[v]) {
                  if (_delete_set.find(v_ngh) == _delete_set.end() && v_ngh != (uint32_t)i) {
                    candidate_set.insert(v_ngh);
                  }
                }
              }
              
              if (candidate_set.size() > range) {
                expanded_nghrs.clear();
                result.clear();
                for (auto j : candidate_set) {
                  expanded_nghrs.push_back(
                      Neighbor(j,
                               _distance->compare(_data + _aligned_dim * i, _data + _aligned_dim * (size_t) j,
                                                  (unsigned) _aligned_dim),
                               true));
                  cmp ++;
                }

                std::sort(expanded_nghrs.begin(), expanded_nghrs.end());
                cmp += occlude_list(expanded_nghrs, alpha, range, maxc, result);
                total_occ_cnt++;

                _final_graph[(uint32_t) i].clear();
                for (auto j : result) {
                  _final_graph[(uint32_t) i].push_back(j.id);
                }
              } else {
                _final_graph[(uint32_t) i].clear();
                for (auto j : candidate_set) {
                  _final_graph[(uint32_t) i].push_back(j);
                }
              }
            }
            occ_time += timer.elapsed();
          }
        }
      }
      total_cmp += cmp;
      total_occ += occ_time;
    }

    for (auto iter : _delete_set) {
      _empty_slots.insert(iter);
    }
    _nd -= _delete_set.size();

    _data_compacted = _delete_set.size() == 0;

    auto stop = std::chrono::high_resolution_clock::now();
    LOG(INFO) << "total_cmp :" << total_cmp.load();
    LOG(INFO) << "total_occ :" << total_occ.load() / 1e6 / N_DELETE_THREADS << "s.";
    LOG(INFO) << "total_occ_cnt :" << total_occ_cnt.load();

    LOG(INFO) << "Time taken for consolidate_deletes() "
              << std::chrono::duration_cast<std::chrono::duration<double>>(stop - start).count() << "s.";

    return _nd;
  }

  
  // DirANN 
  template<typename T, typename TagT>
  size_t Index<T, TagT>::consolidate_deletes_dirann(const Parameters &parameters) {
    if (_eager_done) {
      LOG(INFO) << "In consolidate_deletes(), _eager_done is true. So exiting.";
      return 0;
    }

    omp_set_num_threads(N_DELETE_THREADS);
    LOG(INFO) << "Consolidate deletes threads num " << omp_get_max_threads();

    bool OPEN_DEGREE = true;

    LOG(INFO) << "Empty slots size: " << _empty_slots.size() << " _nd: " << _nd << " max_points: " << _max_points;
    assert(_enable_tags);
    assert(_delete_set.size() <= _nd);
    assert(_empty_slots.size() + _nd == _max_points);

    const unsigned range = parameters.R;
    const unsigned maxc = parameters.C;
    const float alpha = parameters.alpha;

    std::atomic<size_t> total_hit(0), total_miss(0);
    std::atomic<size_t> total_hit2(0);
    std::atomic<size_t> total_hit3(0);
    std::atomic<size_t> total_hit4(0);
    std::atomic<size_t> total_cmp(0), total_occ(0), total_occ_cnt(0);

    uint64_t total_pts = _max_points + _num_frozen_pts;

    auto cached_occlude_list = [&](const std::vector<Neighbor> &pool, 
      std::vector<Neighbor> &result,
      std::vector<float> &occlude_factor,
      std::vector<float> &dist_cache,
      std::vector<uint64_t> &dist_gen,
      uint64_t current_gen,
      size_t Vin_size
    ) -> int {

      if (pool.empty())
        return 0;
      size_t degree = range;

      int cmp = 0;
      float cur_alpha = 1;
      while (cur_alpha <= alpha && result.size() < degree) {
        unsigned start = 0;

        while (result.size() < degree && (start) < pool.size() && start < maxc) {
          auto &p = pool[start];
          if (occlude_factor[start] > cur_alpha) {
            start++;
            continue;
          }
          occlude_factor[start] = std::numeric_limits<float>::max();
          result.push_back(p);
          const int idx_p = p.idx;
          for (unsigned t = start + 1; t < pool.size() && t < maxc; t++) {
            if (occlude_factor[t] > alpha)
              continue;
            int idx_t = pool[t].idx;
            // Symmetric cache: ensure a <= b
            int a = idx_p <= idx_t ? idx_p : idx_t;
            int b = idx_p <= idx_t ? idx_t : idx_p;
            size_t cache_pos = (size_t)a * Vin_size + (size_t)b;
            uint64_t &gen = dist_gen[cache_pos];
            float &djk = dist_cache[cache_pos];
            if (__builtin_expect(gen != current_gen, 0)) {
              gen = current_gen;
              djk = _distance->compare(_data + _aligned_dim * (size_t) pool[t].id,
                                        _data + _aligned_dim * (size_t) p.id, (unsigned) _aligned_dim);
              cmp ++;
            }
            occlude_factor[t] = (std::max)(occlude_factor[t], pool[t].distance / djk);
          }
          start++;
        }
        cur_alpha *= 1.2f;
      }
      return cmp;
    };
    
    auto start = std::chrono::high_resolution_clock::now();
    Timer delete_timer;
    // 1. get reverse neighbors of deleted points
    tsl::robin_map<uint32_t, std::vector<uint32_t>> reverse_nbrs; // store deleted_id's incoming
    // tsl::robin_map<uint32_t, std::vector<uint32_t>> deleted_nbrs_map; // store survivor_id -> removed deleted neighbors
    tsl::robin_map<uint32_t, uint32_t> id_idx_map; // 
    std::vector<std::vector<Neighbor>> candidate_lists; // 
    std::vector<std::atomic<uint32_t>> reverse_nbrs_num(total_pts);
    std::vector<std::atomic<uint32_t>> deleted_reverse_nbrs_num(total_pts);

    for (auto vd : _delete_set) {
      reverse_nbrs[vd];
    }

    uint32_t need_patch_pts = 0;
    #pragma omp parallel
    {
      std::vector<uint32_t> local_v;
      local_v.reserve(128);
      // tsl::robin_map<uint32_t, std::vector<uint32_t>> local_deleted_nbrs_map;

      #pragma omp for schedule(dynamic, 1024)
      for (uint64_t i = 0; i < total_pts; i++) {
        if (_delete_set.find((uint32_t)i) != _delete_set.end()) continue;

        auto& nbrs = _final_graph[i];
        if (nbrs.empty()) continue;

        // std::vector<uint32_t> local_deleted_nbrs;
        size_t j = 0;
        for (size_t k = 0; k < nbrs.size(); k++) {
          if (_delete_set.find(nbrs[k]) != _delete_set.end()) {
            v2::LockGuard guard(_locks->wrlock(nbrs[k]));
            reverse_nbrs[nbrs[k]].push_back((uint32_t)i);
            // local_deleted_nbrs.push_back(nbrs[k]);
            deleted_reverse_nbrs_num[nbrs[k]].fetch_add(1, std::memory_order_relaxed);
          } else {
            nbrs[j++] = nbrs[k];
          }
          reverse_nbrs_num[nbrs[k]].fetch_add(1, std::memory_order_relaxed);
        }

        if (j != nbrs.size()) {
          nbrs.resize(j);
          local_v.push_back((uint32_t)i);
        }
      }

      #pragma omp critical
      {
        for (auto id : local_v) {
          id_idx_map[id] = need_patch_pts ++;
        }
      }
    }
    candidate_lists.resize(need_patch_pts);

    LOG(INFO) << "Time 1 elapsed " << (delete_timer.elapsed() / 1e6) << "s";
    std::vector<uint32_t> deleted_ids(_delete_set.begin(), _delete_set.end());
    #pragma omp parallel
    {
#ifndef DIRANN_DISABLE_DISTANCE_CACHE
      size_t cache_threshold = range * 30;
      std::vector<float> dist_cache(cache_threshold * cache_threshold);
      std::vector<uint64_t> dist_gen(cache_threshold * cache_threshold, 0);
      uint64_t gen_counter = 1;
#endif
      std::vector<Neighbor> pool, result;
      std::vector<float> occlude_factor; 
      std::vector<float> d_vo_vd, d_vi_vd;

      #pragma omp for schedule(dynamic)
      for (uint32_t k = 0; k < deleted_ids.size(); k ++) {
        uint32_t vd = deleted_ids[k];
        auto &Vin = reverse_nbrs[vd];
        auto &Vout = _final_graph[vd];
        
        size_t Vin_size = Vin.size();
        total_hit2 += Vin_size * Vout.size();
        
        d_vi_vd.clear();
        d_vo_vd.clear();
        pool.reserve(Vin_size);
        result.reserve(range);
#ifndef DIRANN_DISABLE_DISTANCE_CACHE
        bool use_cache = Vin_size < cache_threshold;
        if (use_cache) {
          gen_counter++;
        }
#else
        const bool use_cache = false;
#endif

        T* vec_d = _data + (size_t)vd * _aligned_dim;
        size_t cmp = 0;
        size_t occ_time = 0;
        int hit = 0;
        for (auto vi : Vin) {
          T* vec_i = _data + (size_t)vi * _aligned_dim;
          d_vi_vd.push_back( _distance->compare(vec_i, vec_d, (unsigned)_aligned_dim));
          cmp ++;
        }
        for (auto vo : Vout) {
          T* vec_o = _data + (size_t)vo * _aligned_dim;
          d_vo_vd.push_back( _distance->compare(vec_o, vec_d, (unsigned)_aligned_dim));
          cmp ++;
        }
        
        for (uint32_t j = 0; j < Vout.size(); j ++) {
          uint32_t vo = Vout[j];
          if (_delete_set.find(vo) != _delete_set.end()) continue;
          
          pool.clear();
          result.clear();

          T* vec_o = _data + (size_t)vo * _aligned_dim;

          for (uint32_t i = 0; i < Vin.size(); i ++) {
            uint32_t vi = Vin[i];
            if (vi == vo) continue;
            T* vec_i = _data + (size_t)vi * _aligned_dim;
            float d_vo_vi = _distance->compare(vec_o, vec_i, (unsigned)_aligned_dim);
            cmp ++;

            float cosin = cosf((float)ANGLE_DEGREES * 3.1415926535f / 180.0f);
            if ((d_vo_vi*d_vo_vi + d_vo_vd[j]*d_vo_vd[j] - d_vi_vd[i]*d_vi_vd[i] > 2 * d_vo_vi * d_vo_vd[j] * cosin) || (OPEN_DEGREE && (reverse_nbrs_num[vo] < 5 || (deleted_reverse_nbrs_num[vo]*1.0/reverse_nbrs_num[vo] >= 0.1) || (Vin.size() <= 5)))) { // alpha
              pool.emplace_back(vi, d_vo_vi, true);
              pool.back().idx = i;
            } else { // Hardly ever triggers
              hit ++;
            }
          }

          if (pool.empty()) continue;
          Timer timer;
          std::sort(pool.begin(), pool.end());
          occlude_factor.assign(pool.size(), 0.0f);
          
#ifndef DIRANN_DISABLE_DISTANCE_CACHE
          if (use_cache) {
            cmp += cached_occlude_list(pool, result, occlude_factor, dist_cache, dist_gen, gen_counter, Vin_size);
          } else
#endif
          {
            cmp += occlude_list(pool, alpha, range, maxc, result, occlude_factor);
          }

          total_occ_cnt++;
          total_hit4 += result.size();
          occ_time += timer.elapsed();

          for (auto & vi_nbr: result) {
            v2::LockGuard guard(_locks->wrlock(vi_nbr.id));
            candidate_lists[id_idx_map[vi_nbr.id]].emplace_back(Neighbor(vo, vi_nbr.distance, true));
          }
        }
        total_cmp += cmp;
        total_occ += occ_time;
        total_hit3 += hit;
      }
    }

    std::atomic<size_t> hoe_cnt(0);
    std::vector<uint32_t> need_patch_points;
    for (auto [vi, idx] : id_idx_map) {
      need_patch_points.push_back(vi);
    }

    size_t total_occ_node_cnt = 0;

    #pragma omp parallel for schedule(dynamic) reduction(+:total_occ_node_cnt)
    for (uint32_t i = 0; i < need_patch_points.size(); i ++) {
      size_t cmp = 0, occ_time = 0;

      auto vi = need_patch_points[i];
      auto & pool = candidate_lists[id_idx_map[vi]];
      if (pool.empty()) {
        total_cmp += cmp;
        continue;
      }

      total_occ_node_cnt ++;
      auto & nbrs = _final_graph[vi];

      tsl::robin_set<uint32_t> new_nbrs_set;
      for (auto &nbr : pool) {
        new_nbrs_set.insert(nbr.id);
      }
      for (size_t j = 0; j < nbrs.size(); j++) {
        uint32_t nbr = nbrs[j];
        if (new_nbrs_set.find(nbr) != new_nbrs_set.end()) {
          nbrs[j] = nbrs.back();
          nbrs.pop_back();
          j--;
        }
      }

      std::vector<Neighbor> result;
      std::vector<float> occlude_factor;
      Timer timer;
      occlude_factor.assign(pool.size(), 0.0f); 
      std::sort(pool.begin(), pool.end());
      cmp += occlude_list(pool, alpha, range, maxc, result, occlude_factor);
      total_occ_cnt++;
      pool = std::move(result);
      result.clear();
      
      const size_t sorted_end = pool.size();
      pool.reserve(sorted_end + nbrs.size());
      for (auto &id : nbrs) {
        float dist = _distance->compare(_data + _aligned_dim * (size_t) vi,
                                        _data + _aligned_dim * (size_t) id, (unsigned) _aligned_dim);
        cmp ++;
        pool.emplace_back(Neighbor(id, dist, false));
      }
      std::sort(pool.begin() + sorted_end, pool.end());
      std::inplace_merge(pool.begin(), pool.begin() + sorted_end, pool.end());

      occlude_factor.assign(pool.size(), 0.0f); 
      cmp += occlude_list(pool, alpha, range, maxc, result, occlude_factor, true);
      total_occ_cnt++;

      occ_time += timer.elapsed();

      _final_graph[vi].clear();
      for (auto & nbr : result) {
        _final_graph[vi].emplace_back(nbr.id);
      }
      total_cmp += cmp;
      total_occ += occ_time;
    }
    LOG(INFO) << "Time 4 elapsed " << (delete_timer.elapsed() / 1e6) << "s";

    #ifdef DIRANN_CONNECTIVITY_REPAIR
    {
      // 5. connectivity check and repair
      repair_connectivity_after_delete_dirann(parameters);
      LOG(INFO) << "Time 5 elapsed " << (delete_timer.elapsed() / 1e6) << "s";
    }
    #endif

    for (auto iter : _delete_set) {
      _empty_slots.insert(iter);
    }
    _nd -= _delete_set.size();

    _data_compacted = _delete_set.size() == 0;

    auto stop = std::chrono::high_resolution_clock::now();
    LOG(INFO) << "Time taken for consolidate_deletes() "
              << std::chrono::duration_cast<std::chrono::duration<double>>(stop - start).count() << "s.";

    return _nd;
  }

  template<typename T, typename TagT>
  void Index<T, TagT>::repair_connectivity_after_delete_dirann(const Parameters &parameters) {
    const unsigned range = parameters.R;
    const unsigned maxc = parameters.C;
    const float alpha = parameters.alpha;

    // anchor search parameters
    const unsigned anchor_c = 4;  // number of anchor candidates to keep
    const unsigned Lanchor = std::max<unsigned>(16, std::max<unsigned>(maxc, 4 * anchor_c));

    uint64_t total_pts = _max_points + _num_frozen_pts;
    if (_nd == 0) {
      LOG(INFO) << "Connectivity repair skipped: index is empty.";
      return;
    }

    uint32_t ep = find_valid_ep_after_delete();
    if (ep == std::numeric_limits<uint32_t>::max()) {
      LOG(INFO) << "Connectivity repair skipped: no valid entry point found.";
      return;
    }

    
    _ep = ep;

    // ------------------------------------------------------------
    
    // ------------------------------------------------------------
    std::vector<uint8_t> visited(total_pts, 0);

    auto bfs_from_ep = [&](uint32_t seed) {
      if (seed >= total_pts) return;
      if (_delete_set.find(seed) != _delete_set.end()) return;
      if (_empty_slots.find(seed) != _empty_slots.end()) return;

      std::queue<uint32_t> q;
      visited[seed] = 1;
      q.push(seed);

      while (!q.empty()) {
        uint32_t u = q.front();
        q.pop();

        for (auto v : _final_graph[u]) {
          if (v >= total_pts) continue;
          if (_delete_set.find(v) != _delete_set.end()) continue;
          if (_empty_slots.find(v) != _empty_slots.end()) continue;
          if (!visited[v]) {
            visited[v] = 1;
            q.push(v);
          }
        }
      }
    };

    bfs_from_ep(ep);

    // ------------------------------------------------------------
    
    
    // ------------------------------------------------------------
    size_t isolated_cnt_before = 0;
    size_t no_visited = 0;
    std::vector<uint32_t> repair_nodes; 
    repair_nodes.reserve(_nd);

    for (uint32_t i = 0; i < total_pts; i++) {
      if (_delete_set.find(i) != _delete_set.end()) continue;
      if (_empty_slots.find(i) != _empty_slots.end()) continue;

      if (_final_graph[i].empty()) {
        isolated_cnt_before++;
      }

      if (!visited[i]) {
        repair_nodes.push_back(i);
        no_visited++;
      }
    }

    LOG(INFO) << "ep=" << ep;
    LOG(INFO) << "no_visited=" << no_visited;
    LOG(INFO) << "isolated_cnt_before=" << isolated_cnt_before;
    LOG(INFO) << "repair_unreachable_nodes=" << repair_nodes.size();

    if (repair_nodes.empty()) {
      LOG(INFO) << "Connectivity repair skipped: no unreachable isolated nodes.";
      return;
    }

    // ------------------------------------------------------------
    
    
    
    // ------------------------------------------------------------
    std::atomic<size_t> total_cmp(0);
    std::atomic<size_t> repaired_cnt(0);
    std::atomic<size_t> skipped_no_anchor(0);

    std::vector<std::pair<uint32_t, uint32_t>> reverse_requests; // (anchor, u)

    #pragma omp parallel
    {
      alignas(4096) T coord_u[2048];

      std::vector<std::pair<uint32_t, uint32_t>> local_reverse_requests;
      std::vector<unsigned> init_ids;
      std::vector<Neighbor> expanded_nodes_info;
      tsl::robin_set<unsigned> expanded_nodes_ids;
      std::vector<Neighbor> best_L_nodes;
      std::vector<Neighbor> anchor_candidates;

      #pragma omp for schedule(dynamic, 64)
      for (int64_t idx = 0; idx < (int64_t) repair_nodes.size(); idx++) {
        uint32_t u = repair_nodes[(size_t)idx];

        get_coord(u, coord_u);

        init_ids.clear();
        init_ids.push_back(_ep);

        expanded_nodes_info.clear();
        expanded_nodes_ids.clear();
        best_L_nodes.clear();

        iterate_to_fixed_point(coord_u, Lanchor, init_ids,
                              expanded_nodes_info, expanded_nodes_ids, best_L_nodes,
                              false, nullptr);

        anchor_candidates.clear();
        anchor_candidates.reserve(anchor_c);

        for (auto &nbr : best_L_nodes) {
          if (nbr.distance == std::numeric_limits<float>::max()) continue;
          if (nbr.id == u) continue;
          if (nbr.id >= total_pts) continue;
          if (_delete_set.find(nbr.id) != _delete_set.end()) continue;
          if (_empty_slots.find(nbr.id) != _empty_slots.end()) continue;
          if (!visited[nbr.id]) continue;  

          bool duplicate = false;
          for (auto &x : anchor_candidates) {
            if (x.id == nbr.id) {
              duplicate = true;
              break;
            }
          }
          if (duplicate) continue;

          anchor_candidates.push_back(nbr);
          if (anchor_candidates.size() >= anchor_c) break;
        }

        if (anchor_candidates.empty()) {
          skipped_no_anchor++;
          continue;
        }

        for (auto &cand : anchor_candidates) {
          local_reverse_requests.emplace_back(cand.id, u);
        }

        repaired_cnt++;
      }

      #pragma omp critical
      {
        reverse_requests.insert(reverse_requests.end(),
                                local_reverse_requests.begin(),
                                local_reverse_requests.end());
      }
    }

    if (reverse_requests.empty()) {
      LOG(INFO) << "Connectivity repair finished. repaired_cnt: 0"
                << " skipped_no_anchor: " << skipped_no_anchor.load()
                << " isolated_before: " << isolated_cnt_before
                << " isolated_after: " << isolated_cnt_before
                << " unreachable_after: " << no_visited
                << " total_cmp: " << total_cmp.load();
      return;
    }

    
    std::sort(reverse_requests.begin(), reverse_requests.end(),
              [](const auto &a, const auto &b) {
                if (a.first != b.first) return a.first < b.first;
                return a.second < b.second;
              });
    reverse_requests.erase(std::unique(reverse_requests.begin(), reverse_requests.end()),
                          reverse_requests.end());

    // ------------------------------------------------------------
    
    
    // ------------------------------------------------------------
    std::vector<uint32_t> unique_anchors;
    std::vector<size_t> group_starts;
    unique_anchors.reserve(reverse_requests.size());
    group_starts.reserve(reverse_requests.size());

    for (size_t i = 0; i < reverse_requests.size();) {
      size_t j = i + 1;
      while (j < reverse_requests.size() && reverse_requests[j].first == reverse_requests[i].first) {
        j++;
      }
      unique_anchors.push_back(reverse_requests[i].first);
      group_starts.push_back(i);
      i = j;
    }
    group_starts.push_back(reverse_requests.size());

    #pragma omp parallel
    {
      alignas(4096) T coord_anchor[2048];
      T coord_tmp[2048];
      std::vector<Neighbor> pool, result;
      std::vector<float> occlude_factor;

      #pragma omp for schedule(dynamic, 64)
      for (int64_t g = 0; g < (int64_t) unique_anchors.size(); g++) {
        uint32_t anchor = unique_anchors[(size_t)g];
        if (anchor >= total_pts) continue;
        if (_delete_set.find(anchor) != _delete_set.end()) continue;
        if (_empty_slots.find(anchor) != _empty_slots.end()) continue;

        get_coord(anchor, coord_anchor);

        pool.clear();
        result.clear();

        auto &nbrs = _final_graph[anchor];
        pool.reserve(nbrs.size() + (group_starts[(size_t)g + 1] - group_starts[(size_t)g]));

        
        for (auto id : nbrs) {
          if (id >= total_pts || id == anchor) continue;
          if (_delete_set.find(id) != _delete_set.end()) continue;
          if (_empty_slots.find(id) != _empty_slots.end()) continue;

          get_coord(id, coord_tmp);
          float dist = _distance->compare(coord_anchor, coord_tmp, (unsigned)_aligned_dim);
          total_cmp++;
          pool.emplace_back(id, dist, false);
        }

        
        for (size_t p = group_starts[(size_t)g]; p < group_starts[(size_t)g + 1]; p++) {
          uint32_t u = reverse_requests[p].second;
          if (u >= total_pts || u == anchor) continue;
          if (_delete_set.find(u) != _delete_set.end()) continue;
          if (_empty_slots.find(u) != _empty_slots.end()) continue;

          bool found = false;
          for (auto &x : pool) {
            if (x.id == u) {
              found = true;
              break;
            }
          }
          if (!found) {
            get_coord(u, coord_tmp);
            float dist = _distance->compare(coord_anchor, coord_tmp, (unsigned)_aligned_dim);
            total_cmp++;
            pool.emplace_back(u, dist, true);
          }
        }

        if (!pool.empty()) {
          std::sort(pool.begin(), pool.end());
          occlude_factor.assign(pool.size(), 0.0f);
          total_cmp += occlude_list(pool, alpha, range, maxc, result, occlude_factor);

          v2::LockGuard guard(_locks->wrlock(anchor));
          _final_graph[anchor].clear();
          for (auto &nbr : result) {
            if (nbr.id != anchor) {
              _final_graph[anchor].push_back(nbr.id);
            }
          }
        }
      }
    }

    // ------------------------------------------------------------
    
    // ------------------------------------------------------------
    std::fill(visited.begin(), visited.end(), 0);
    bfs_from_ep(_ep);

    size_t isolated_cnt_after = 0;
    size_t unreachable_cnt = 0;

    for (uint32_t i = 0; i < total_pts; i++) {
      if (_delete_set.find(i) != _delete_set.end()) continue;
      if (_empty_slots.find(i) != _empty_slots.end()) continue;

      if (_final_graph[i].empty()) isolated_cnt_after++;
      if (!visited[i]) unreachable_cnt++;
    }

    LOG(INFO) << "Connectivity repair finished. repaired_cnt: " << repaired_cnt.load()
              << " skipped_no_anchor: " << skipped_no_anchor.load()
              << " isolated_before: " << isolated_cnt_before
              << " isolated_after: " << isolated_cnt_after
              << " unreachable_after: " << unreachable_cnt
              << " total_cmp: " << total_cmp.load();
  }  
  template<typename T, typename TagT>
  uint32_t Index<T, TagT>::find_valid_ep_after_delete() const {
    uint64_t total_pts = _max_points + _num_frozen_pts;

    if (_ep < total_pts && _delete_set.find(_ep) == _delete_set.end()) {
      return _ep;
    }

    for (uint32_t i = 0; i < total_pts; i++) {
      if (_delete_set.find(i) == _delete_set.end()) {
        return i;
      }
    }

    return std::numeric_limits<uint32_t>::max();
  }

  template<typename T, typename TagT>
  void Index<T, TagT>::find_anchor_candidates_by_search(
      uint32_t u,
      const std::vector<uint8_t> &visited,
      const Parameters &parameters,
      size_t c,
      std::vector<Neighbor> &anchors) {

    anchors.clear();

    uint32_t ep = find_valid_ep_after_delete();
    if (ep == std::numeric_limits<uint32_t>::max()) return;

    
    alignas(4096) T query_buf[2048];
    get_coord(u, query_buf);

    
    unsigned Lanchor = std::max<unsigned>((unsigned)(4 * c), parameters.C);
    Lanchor = std::max<unsigned>(Lanchor, (unsigned)16);

    std::vector<unsigned> init_ids;
    init_ids.push_back(ep);

    std::vector<Neighbor> expanded_nodes_info;
    tsl::robin_set<unsigned> expanded_nodes_ids;
    std::vector<Neighbor> best_L_nodes;

    iterate_to_fixed_point(
        query_buf, Lanchor, init_ids,
        expanded_nodes_info, expanded_nodes_ids, best_L_nodes,
        false, nullptr);

    
    
    for (auto &nbr : best_L_nodes) {
      if (nbr.distance == std::numeric_limits<float>::max()) continue;
      if (nbr.id == u) continue;
      if (_delete_set.find(nbr.id) != _delete_set.end()) continue;

      
      if (!visited[nbr.id]) continue;

      anchors.push_back(nbr);
      if (anchors.size() >= c) break;
    }
  }

  // Woloverine
  template<typename T, typename TagT>
  size_t Index<T, TagT>::consolidate_deletes_wolverine(const Parameters &parameters) {
    if (_eager_done) {
      LOG(INFO) << "In consolidate_deletes(), _eager_done is true. So exiting.";
      return 0;
    }

    omp_set_num_threads(N_DELETE_THREADS);
    LOG(INFO) << "Empty slots size: " << _empty_slots.size() << " _nd: " << _nd << " max_points: " << _max_points;
    assert(_enable_tags);
    assert(_delete_set.size() <= _nd);
    assert(_empty_slots.size() + _nd == _max_points);

    const unsigned range = parameters.R;
    const unsigned maxc = parameters.C;
    const float alpha = parameters.alpha;

    std::atomic<size_t> total_hit(0), total_miss(0), total_cmp(0);
    std::atomic<size_t> total_occ(0), total_occ_cnt(0);

    uint64_t total_pts = _max_points + _num_frozen_pts;

    auto start = std::chrono::high_resolution_clock::now();
    Timer delete_timer;

    #pragma omp parallel for schedule(dynamic, 1024)
    for (uint64_t i = 0; i < total_pts; i++) {

      auto& nbrs = _final_graph[i];
      if (nbrs.empty()) continue;

      size_t j = 0;
      for (size_t k = 0; k < nbrs.size(); k++) {
        if (_delete_set.find(nbrs[k]) == _delete_set.end()) {
          nbrs[j++] = nbrs[k];
        }
      }

      if (j != nbrs.size()) {
        nbrs.resize(j);
      }
    }

    LOG(INFO) << "Time 1 elapsed " << (delete_timer.elapsed() / 1e6) << "s";
    
    std::vector<uint32_t> deleted_ids(_delete_set.begin(), _delete_set.end());

    std::vector<std::vector<Neighbor>> new_links(total_pts); //

    #pragma omp parallel for schedule(dynamic)
    for (uint32_t k = 0; k < deleted_ids.size(); k ++) { // for each deleted point p
      uint32_t p = deleted_ids[k];
      auto &pout_list = _final_graph[p];
      std::vector<Neighbor> pool, result;
      std::vector<float> occlude_factor;

      T * vec_p = _data + (size_t)p * _aligned_dim;

      size_t cmp = 0, occ_time = 0;
      for (auto &pout : pout_list) { // for pout
        std::unordered_set<uint32_t> predict_list;
        std::unordered_map<uint32_t, float> dist_map;
        auto &one_hop_list = _final_graph[p];
        T * vec_pout = _data + (size_t)pout * _aligned_dim;

        #if defined(WOLVERINE_TWOHOP_DELETE)
          // ref: https://github.com/LDW2020/Wolverine/blob/main/hnsw_Wolverine/hnswalg.h#L1599
          for (auto &one_hop : one_hop_list) { // for one-hop for p
            auto & two_hop_list = _final_graph[one_hop]; // one-hof of p, \ie pout
            for (auto & two_hop : two_hop_list) { // for two-hop for p
              predict_list.insert(two_hop);
              if(predict_list.size()>5*range)break;
            }
            if(predict_list.size()>5*range)break;
          }
          pool.clear();
          for (auto predict : predict_list) {
            pool.emplace_back(predict, _distance->compare(vec_pout, _data + (size_t)predict * _aligned_dim, (unsigned)_aligned_dim), true); // calc d(pout, predict)
            cmp ++;
          }
        #else
          // ref: https://github.com/LDW2020/Wolverine/blob/main/hnsw_Wolverine/hnswalg.h#L1620
          float d_pout_p = _distance->compare(vec_pout, vec_p, (unsigned)_aligned_dim); // calc d(pout, p)
          cmp ++;

          for (auto &one_hop : one_hop_list) { // for one-hop for p
            T * vec_one_hop = _data + (size_t)one_hop * _aligned_dim;
            float d_pout_onehop = _distance->compare(vec_pout, vec_one_hop, (unsigned)_aligned_dim); // calc d(pout, one-hop of p)
            dist_map[one_hop] = d_pout_onehop;
            cmp ++;
            if (d_pout_onehop < d_pout_p) {
              predict_list.insert(one_hop);
              auto & two_hop_list = _final_graph[one_hop];

              for (auto & two_hop : two_hop_list) { // for two-hop for p
                T * vec_two_hop = _data + (size_t)two_hop * _aligned_dim;
                float d_p_twohop = _distance->compare(vec_p, vec_two_hop, (unsigned)_aligned_dim); // calc d(p, two-hop of p)
                float d_pout_twohop = _distance->compare(vec_pout, vec_two_hop, (unsigned)_aligned_dim); // calc d(pout, two-hop of p)
                dist_map[two_hop] = d_pout_twohop;
                cmp += 2;
                if (d_p_twohop > d_pout_p &&  // d(p, two-hop of p) > d(pout, two-hop of p)
                  d_pout_twohop < d_pout_p && // d(pout, two-hop of p) < d(pout, p)
                  d_pout_twohop + d_pout_p > d_p_twohop) { // d(pout, two-hop of p)^2 + d(pout, p)^2 > d(p, two-hop of p)^2
                    predict_list.insert(two_hop);
                    if (predict_list.size() > range * 2) break;
                }
              }
              if (predict_list.size() > range * 2) break;
            }
          }
          pool.clear();
          for (auto predict : predict_list) {
            if (dist_map.find(predict) == dist_map.end()) {
              // LOG(ERROR) << "Predict " << predict << " not found in dist_map. This should not happen.";
              // exit(0);
              pool.emplace_back(predict, _distance->compare(vec_pout, _data + (size_t)predict * _aligned_dim, (unsigned)_aligned_dim), true); // calc d(pout, predict)
              cmp ++;
            } else {
              float dist = dist_map[predict];
              pool.emplace_back(predict, dist, true);
            }
          }
        #endif

        Timer timer;

        result.clear();
        std::sort(pool.begin(), pool.end());
        occlude_factor.assign(pool.size(), 0.0f);
        
        cmp += occlude_list(pool, alpha, range, maxc, result, occlude_factor);
        total_occ_cnt++;
        occ_time += timer.elapsed();

        for (auto & nbr: result) {
          v2::LockGuard guard(_locks->wrlock(nbr.id));
          new_links[nbr.id].push_back(Neighbor(pout, nbr.distance, true));
        }
      }
      total_cmp += cmp;
      total_occ += occ_time;
    }

    LOG(INFO) << "Time 1.5 elapsed " << (delete_timer.elapsed() / 1e6) << "s";
    LOG(INFO) << "total_cmp1 :" << (total_cmp.load());
    LOG(INFO) << "total_occ time1:" << (total_occ.load() / 1e6) << "s";
    LOG(INFO) << "total_occ_cnt1 :" << (total_occ_cnt.load());

    size_t total_occ_node_cnt = 0;
    size_t total_occ_node_cnt_exc = 0;

    #pragma omp parallel for schedule(dynamic) reduction(+:total_occ_node_cnt, total_occ_node_cnt_exc)
    for (uint32_t id = 0; id < new_links.size(); id ++) {
      if (new_links[id].empty()) continue;

      total_occ_node_cnt ++;

      auto &new_nbrs = new_links[id];
      std::vector<Neighbor> pool, result;
      std::vector<float> occlude_factor;
      pool.clear();
      result.clear();
      size_t cmp = 0, occ_time = 0;

      v2::LockGuard guard(_locks->wrlock(id));
      auto & old_nbrs = _final_graph[id];
      Timer timer;

      if (old_nbrs.size() + new_nbrs.size() <= range) {
        for (auto & nbr: new_nbrs) {
          if (std::find(old_nbrs.begin(), old_nbrs.end(), nbr.id) == old_nbrs.end()) {
            old_nbrs.emplace_back(nbr.id);
          }
        }
      } else {
        total_occ_node_cnt_exc++;
        std::unordered_set<uint32_t> new_nbrs_set;
        for (auto & old_nbr : old_nbrs) {
          if (new_nbrs_set.find(old_nbr) != new_nbrs_set.end()) continue;
          new_nbrs_set.insert(old_nbr);
          pool.emplace_back(old_nbr, 
            _distance->compare(_data + (size_t)id * _aligned_dim, _data + (size_t)old_nbr * _aligned_dim, (unsigned)_aligned_dim),
            true);
          cmp ++;
        }

        for (auto & nbr: new_nbrs) {
          if (new_nbrs_set.find(nbr.id) != new_nbrs_set.end()) continue;
          new_nbrs_set.insert(nbr.id);
          pool.emplace_back(nbr);
        }

        std::sort(pool.begin(), pool.end());
        occlude_factor.assign(pool.size(), 0.0f);
        
        cmp += occlude_list(pool, alpha, range, maxc, result, occlude_factor);
        occ_time += timer.elapsed();
        total_occ_cnt++;

        old_nbrs.clear();
        for (auto & nbr: result) {
          old_nbrs.emplace_back(nbr.id);
        }
      }
      total_cmp += cmp;
      total_occ += occ_time;
    }
        
    for (auto iter : _delete_set) {
      _empty_slots.insert(iter);
    }
    _nd -= _delete_set.size();

    _data_compacted = _delete_set.size() == 0;

    auto stop = std::chrono::high_resolution_clock::now();
    LOG(INFO) << "Time taken for consolidate_deletes() "
              << std::chrono::duration_cast<std::chrono::duration<double>>(stop - start).count() << "s.";

    return _nd;
  }

  // hnsw
  template<typename T, typename TagT>
  size_t Index<T, TagT>::consolidate_deletes_hnsw(const Parameters &parameters) {
    if (_eager_done) {
      LOG(INFO) << "In consolidate_deletes(), _eager_done is true. So exiting.";
      return 0;
    }

    omp_set_num_threads(N_DELETE_THREADS);
    LOG(INFO) << "Empty slots size: " << _empty_slots.size() << " _nd: " << _nd << " max_points: " << _max_points;
    assert(_enable_tags);
    assert(_delete_set.size() <= _nd);
    assert(_empty_slots.size() + _nd == _max_points);

    const unsigned range = parameters.R;
    const unsigned maxc = parameters.C;
    const float alpha = parameters.alpha;
    unsigned Lindex = parameters.L;
    LOG(INFO) << "Lindex=" << Lindex;

    std::atomic<size_t> total_cmp(0), total_occ(0);

    uint64_t total_pts = _max_points + _num_frozen_pts;

    auto start = std::chrono::high_resolution_clock::now();
    Timer delete_timer;
    std::vector<uint8_t> need_patch(total_pts, 0);

    #pragma omp parallel for schedule(dynamic, 1024)
    for (uint64_t i = 0; i < total_pts; i++) {

      auto& nbrs = _final_graph[i];
      if (nbrs.empty()) continue;

      size_t j = 0;
      for (size_t k = 0; k < nbrs.size(); k++) {
        if (_delete_set.find(nbrs[k]) == _delete_set.end()) {
          nbrs[j++] = nbrs[k];
        }
      }

      if (j != nbrs.size()) {
        nbrs.resize(j);
        need_patch[i] = 1;
      }
    }

    LOG(INFO) << "Time 1 elapsed " << (delete_timer.elapsed() / 1e6) << "s";
    
    #pragma omp parallel for schedule(dynamic)
    for (uint64_t location = 0; location < total_pts; location++) {
      if (need_patch[location] == 0) continue;
      if (_delete_set.find((uint32_t) location) != _delete_set.end()) continue;
      
      std::vector<Neighbor> pool;
      tsl::robin_set<unsigned> visited;
      std::vector<unsigned> pruned_list;
      std::vector<unsigned> init_ids;

      size_t cmp = 0;
      size_t occ_time = 0;
      auto retval = get_expanded_nodes(location, Lindex, init_ids, pool, visited);
      cmp += retval.second;

      for (unsigned i = 0; i < pool.size(); i++)
        if (pool[i].id == (unsigned) location) {
          pool.erase(pool.begin() + i);
          visited.erase((unsigned) location);
          break;
        }
  
      Timer timer;
      cmp += prune_neighbors(location, pool, parameters, pruned_list);
      occ_time += timer.elapsed();

      assert(_final_graph.size() == _max_points + _num_frozen_pts);
  
      if (unlikely(pruned_list.empty())) {
        LOG(INFO) << "pruned_list.size(): " << pruned_list.size();
      }
  
      {
        v2::LockGuard guard(_locks->wrlock(location));
        _final_graph[location].clear();
        for (auto link : pruned_list) {
          _final_graph[location].emplace_back(link);
        }
      }
      
      auto ret = inter_insert(location, pruned_list, parameters);
      cmp += ret.first;
      occ_time += ret.second;

      total_cmp += cmp;
      total_occ += occ_time;
    }
        
    LOG(INFO) << "Time 2 elapsed " << (delete_timer.elapsed() / 1e6) << "s";

    LOG(INFO) << "total_cmp :" << total_cmp.load();
    LOG(INFO) << "total_occ :" << total_occ.load() / 1e6 / N_DELETE_THREADS << "s.";

    for (auto iter : _delete_set) {
      _empty_slots.insert(iter);
    }
    _nd -= _delete_set.size();

    _data_compacted = _delete_set.size() == 0;

    auto stop = std::chrono::high_resolution_clock::now();
    LOG(INFO) << "Time taken for consolidate_deletes() "
              << std::chrono::duration_cast<std::chrono::duration<double>>(stop - start).count() << "s.";

    return _nd;
  }

  template<typename T, typename TagT>
  void Index<T, TagT>::full_pruning(Parameters &parameters) {
    {
      const unsigned range = parameters.R;
      const uint64_t total_pts = _max_points + _num_frozen_pts;
      std::atomic<size_t> total_prune_cmp(0);
      std::atomic<size_t> total_pruned_nodes(0);

#pragma omp parallel for schedule(dynamic, 1024)
      for (int64_t i = 0; i < (int64_t) total_pts; ++i) {
        const uint32_t node = (uint32_t) i;

        if (_delete_set.find(node) != _delete_set.end() ||
            _empty_slots.find(node) != _empty_slots.end()) {
          continue;
        }

        std::vector<unsigned> cur_nghrs;
        {
          v2::LockGuard guard(_locks->rdlock(node));
          cur_nghrs = _final_graph[node];
        }

        if (cur_nghrs.empty()) {
          continue;
        }

        tsl::robin_set<unsigned> unique_nghrs;
        unique_nghrs.reserve(cur_nghrs.size());
        std::vector<Neighbor> pool;
        pool.reserve(cur_nghrs.size());

        size_t cmp = 0;
        for (auto ngh : cur_nghrs) {
          if (ngh == node ||
              _delete_set.find((uint32_t) ngh) != _delete_set.end() ||
              _empty_slots.find((uint32_t) ngh) != _empty_slots.end()) {
            continue;
          }
          if (!unique_nghrs.insert(ngh).second) {
            continue;
          }

          float dist = _distance->compare(_data + (size_t) _aligned_dim * node,
                                          _data + (size_t) _aligned_dim * ngh,
                                          (unsigned) _aligned_dim);
          pool.emplace_back(ngh, dist, true);
          cmp++;
        }

        if (pool.empty()) {
          v2::LockGuard guard(_locks->wrlock(node));
          _final_graph[node].clear();
          continue;
        }

        std::vector<unsigned> pruned_list;
        cmp += prune_neighbors(node, pool, parameters, pruned_list);

        if (pruned_list.size() > range) {
          pruned_list.resize(range);
        }

        {
          v2::LockGuard guard(_locks->wrlock(node));
          _final_graph[node] = pruned_list;
        }

        total_prune_cmp += cmp;
        total_pruned_nodes++;
      }

      LOG(INFO) << "Full pruning before delete consolidation finished. nodes="
                << total_pruned_nodes.load()
                << ", cmp=" << total_prune_cmp.load();
    }
  }

  template<typename T, typename TagT>
  void Index<T, TagT>::consolidate(Parameters &parameters) {
    #ifdef WOLVERINE
    consolidate_deletes_wolverine(parameters);
    #elif IP
    consolidate_deletes_ip_diskann(parameters);
    #elif GREATOR
    consolidate_deletes_greator(parameters);
    #elif DIRANN
    consolidate_deletes_dirann(parameters);
    #elif HNSW
    consolidate_deletes_hnsw(parameters);
    #else
    consolidate_deletes(parameters);
    #endif
    compact_data();
  }

  template<typename T, typename TagT>
  void Index<T, TagT>::compact_frozen_point() {
    if (_nd < _max_points) {
      if (_num_frozen_pts > 0) {
        // set new _ep to be frozen point
        _ep = (uint32_t) _nd;
        if (!_final_graph[_max_points].empty()) {
          for (unsigned i = 0; i < _nd; i++)
            for (unsigned j = 0; j < _final_graph[i].size(); j++)
              if (_final_graph[i][j] == _max_points)
                _final_graph[i][j] = (uint32_t) _nd;

          _final_graph[_nd].clear();
          for (unsigned k = 0; k < _final_graph[_max_points].size(); k++)
            _final_graph[_nd].emplace_back(_final_graph[_max_points][k]);

          _final_graph[_max_points].clear();

          memcpy((void *) (_data + (size_t) _aligned_dim * _nd), _data + (size_t) _aligned_dim * _max_points,
                 sizeof(T) * _dim);
          memset((_data + (size_t) _aligned_dim * _max_points), 0, sizeof(T) * _aligned_dim);
        }
      }
    }
  }

  template<typename T, typename TagT>
  void Index<T, TagT>::compact_data() {
    if (!_dynamic_index)
      return;

    if (!_lazy_done && !_eager_done)
      return;

    if (_data_compacted) {
      LOG(ERROR) << "Warning! Calling compact_data() when _data_compacted is true!";
      return;
    }

    auto start = std::chrono::high_resolution_clock::now();
    auto fnstart = start;

    std::vector<unsigned> new_location = std::vector<unsigned>(_max_points + _num_frozen_pts, (uint32_t) _max_points);

    uint32_t new_counter = 0;

    for (uint32_t old_counter = 0; old_counter < _max_points + _num_frozen_pts; old_counter++) {
      if (_location_to_tag.find(old_counter) != _location_to_tag.end()) {
        new_location[old_counter] = new_counter;
        new_counter++;
      }
    }

    auto stop = std::chrono::high_resolution_clock::now();
    LOG(INFO) << "Time taken for initial setup: "
              << std::chrono::duration_cast<std::chrono::duration<double>>(stop - start).count() << "s.";
    // If start node is removed, replace it.
    if (_delete_set.find(_ep) != _delete_set.end()) {
      LOG(ERROR) << "Replacing start node which has been deleted... ";
      auto old_ep = _ep;
      // First active neighbor of old start node is new start node
      for (auto iter : _final_graph[_ep])
        if (_delete_set.find(iter) != _delete_set.end()) {
          _ep = iter;
          break;
        }
      if (_ep == old_ep) {
        LOG(ERROR) << "ERROR: Did not find a replacement for start node.";
        crash();
      } else {
        assert(_delete_set.find(_ep) == _delete_set.end());
      }
    }

    start = std::chrono::high_resolution_clock::now();
    double copy_time = 0;
    for (unsigned old = 0; old <= _max_points; ++old) {
      if ((new_location[old] < _max_points) || (old == _max_points)) {  // If point continues to exist

        // Renumber nodes to compact the order
        for (size_t i = 0; i < _final_graph[old].size(); ++i) {
          if (new_location[_final_graph[old][i]] > _final_graph[old][i]) {
            std::stringstream sstream;
            sstream << "Error in compact_data(). Found point: " << old << " whose " << i
                    << "th neighbor has new location " << new_location[_final_graph[old][i]]
                    << " that is greater than its old location: " << _final_graph[old][i];
            if (_delete_set.find(_final_graph[old][i]) != _delete_set.end()) {
              sstream << " Point: " << old << " index: " << i << " neighbor: " << _final_graph[old][i]
                      << " found in delete set of size: " << _delete_set.size();
            } else {
              sstream << " Point: " << old << " neighbor: " << _final_graph[old][i]
                      << " NOT found in delete set of size: " << _delete_set.size();
            }

            LOG(ERROR) << sstream.str();
            crash();
          }
          _final_graph[old][i] = new_location[_final_graph[old][i]];
        }

        // Move the data and adj list to the correct position
        auto c_start = std::chrono::high_resolution_clock::now();
        if (new_location[old] != old) {
          assert(new_location[old] < old);
          _final_graph[new_location[old]].swap(_final_graph[old]);
          memcpy((void *) (_data + _aligned_dim * (size_t) new_location[old]),
                 (void *) (_data + _aligned_dim * (size_t) old), _aligned_dim * sizeof(T));
        }
        auto c_stop = std::chrono::high_resolution_clock::now();
        copy_time += std::chrono::duration_cast<std::chrono::duration<double>>(c_stop - c_start).count();

      } else {
        _final_graph[old].clear();
      }
    }
    stop = std::chrono::high_resolution_clock::now();
    LOG(INFO) << "Time taken for moving data around: "
              << std::chrono::duration_cast<std::chrono::duration<double>>(stop - start).count()
              << "s. Of which copy_time: " << copy_time << "s.";

    start = std::chrono::high_resolution_clock::now();
    _tag_to_location.clear();
    for (auto iter : _location_to_tag) {
      _tag_to_location[iter.second] = new_location[iter.first];
    }
    _location_to_tag.clear();
    for (auto iter : _tag_to_location) {
      _location_to_tag[iter.second] = iter.first;
    }

    for (uint64_t old = _nd; old < _max_points; ++old) {
      _final_graph[old].clear();
    }
    _delete_set.clear();
    _empty_slots.clear();
    for (uint32_t i = _nd; i < _max_points; i++) {
      _empty_slots.insert(i);
    }

    _lazy_done = false;
    _eager_done = false;
    _data_compacted = true;

    size_t total_degree = 0;
    for (uint32_t i = 0; i < _nd; ++i) {
      total_degree += _final_graph[i].size();
    }
    const double avg_degree = (_nd == 0) ? 0.0 : static_cast<double>(total_degree) / static_cast<double>(_nd);

    stop = std::chrono::high_resolution_clock::now();
    LOG(INFO) << "Time taken for tag<->index consolidation: "
              << std::chrono::duration_cast<std::chrono::duration<double>>(stop - start).count() << "s.";
    LOG(INFO) << "Average degree after compact_data: " << avg_degree;
    LOG(INFO) << "Time taken for compact_data(): "
              << std::chrono::duration_cast<std::chrono::duration<double>>(stop - fnstart).count() << "s.";
  }

  // Do not call reserve_location() if you have not locked _change_lock.
  // It is not thread safe.
  template<typename T, typename TagT>
  int Index<T, TagT>::reserve_location() {
    std::lock_guard<std::mutex> guard(_change_lock);
    if (_nd >= _max_points) {
      return -1;
    }
    unsigned location;
    if (_data_compacted) {
      location = (unsigned) _nd;
      _empty_slots.erase(location);
    } else {
      // no need of delete_lock here, _change_lock will ensure no other thread
      // executes this block of code
      assert(_empty_slots.size() != 0);
      assert(_empty_slots.size() + _nd == _max_points);

      auto iter = _empty_slots.begin();
      location = *iter;
      _empty_slots.erase(iter);
      _delete_set.erase(location);
    }

    ++_nd;
    return location;
  }

  template<typename T, typename TagT>
  void Index<T, TagT>::reposition_point(unsigned old_location, unsigned new_location) {
    for (unsigned i = 0; i < _nd; i++)
      for (unsigned j = 0; j < _final_graph[i].size(); j++)
        if (_final_graph[i][j] == old_location)
          _final_graph[i][j] = (unsigned) new_location;

    _final_graph[new_location].clear();
    for (unsigned k = 0; k < _final_graph[_nd].size(); k++)
      _final_graph[new_location].emplace_back(_final_graph[old_location][k]);

    _final_graph[old_location].clear();

    memcpy((void *) (_data + (size_t) _aligned_dim * new_location), _data + (size_t) _aligned_dim * old_location,
           sizeof(T) * _aligned_dim);
    memset((_data + (size_t) _aligned_dim * old_location), 0, sizeof(T) * _aligned_dim);
  }

  template<typename T, typename TagT>
  void Index<T, TagT>::reposition_frozen_point_to_end() {
    if (_num_frozen_pts == 0)
      return;

    if (_nd == _max_points) {
      LOG(INFO) << "Not repositioning frozen point as it is already at the end.";
      return;
    }
    reposition_point(_nd, _max_points);
    _ep = (uint32_t) _max_points;
  }

  template<typename T, typename TagT>
  void Index<T, TagT>::resize(uint32_t new_max_points) {
    // TODO: Check if the _change_lock and _update_lock are both locked.

    auto start = std::chrono::high_resolution_clock::now();
    assert(_empty_slots.size() == 0);  // should not resize if there are empty slots.

    T *new_data;
    alloc_aligned((void **) &new_data, (new_max_points + 1) * _aligned_dim * sizeof(T), 8 * sizeof(T));
    LOG(INFO) << "Resize to " << new_max_points << " " << _max_points << " with ptr " << (void *) _data << " "
              << (void *) new_data;
    memcpy(new_data, _data, (_max_points + 1) * _aligned_dim * sizeof(T));
    aligned_free(_data);
    _data = new_data;

    _final_graph.resize(new_max_points + 1);

    reposition_point(_max_points, new_max_points);
    _max_points = new_max_points;
    _ep = new_max_points;

    for (uint32_t i = _nd; i < _max_points; i++) {
      _empty_slots.insert(i);
    }

    auto stop = std::chrono::high_resolution_clock::now();
    LOG(INFO) << "Resizing took: " << std::chrono::duration<double>(stop - start).count() << "s";
  }

  template<typename T, typename TagT>
  int Index<T, TagT>::insert_point(const T *point, const Parameters &parameters, const TagT tag) {
    std::shared_lock<std::shared_timed_mutex> lock(_update_lock);
    unsigned range = parameters.R;
    //    assert(_has_built);
    std::vector<Neighbor> pool;
    std::vector<Neighbor> tmp;
    tsl::robin_set<unsigned> visited;

    {
      std::shared_lock<std::shared_timed_mutex> lock(_update_lock);
      std::shared_lock<std::shared_timed_mutex> tsl(_tag_lock);
      if (_enable_tags && (_tag_to_location.find(tag) != _tag_to_location.end())) {
        // LOG(INFO) << "Into Locking" ;
        // TODO! This is a repeat of lazy_delete, but we can't call
        // that function because we are taking many locks here. Hence
        // the repeated code.
        tsl.unlock();
        std::unique_lock<std::shared_timed_mutex> tdl(_delete_lock);
        std::unique_lock<std::shared_timed_mutex> tul(_tag_lock);
        _lazy_done = true;
        _delete_set.insert(_tag_to_location[tag]);
        _location_to_tag.erase(_tag_to_location[tag]);
        _tag_to_location.erase(tag);
        // LOG(INFO) << "Out Locking" ;
      }
    }

    auto location = reserve_location();
    if (location == -1) {
      DLOG(INFO) << "Thread: " << std::this_thread::get_id() << " location  == -1. Waiting for unique_lock. ";
      lock.unlock();
      std::unique_lock<std::shared_timed_mutex> growth_lock(_update_lock);

      DLOG(INFO) << "Thread: " << std::this_thread::get_id() << " Obtained unique_lock. ";
      if (_nd >= _max_points) {
        auto new_max_points = (size_t) (_max_points * INDEX_GROWTH_FACTOR);
        LOG(INFO) << "Thread: " << std::this_thread::get_id() << ": Increasing _max_points from " << _max_points
                  << " to " << new_max_points << " _nd is: " << _nd;
        resize(new_max_points);
      }
      growth_lock.unlock();
      lock.lock();
      location = reserve_location();
      // TODO: Consider making this a while/do_while loop so that we retry
      // instead of terminating.
      if (location == -1) {
        crash();
      }
    }

    {
      std::unique_lock<std::shared_timed_mutex> lock(_tag_lock);

      _tag_to_location[tag] = location;
      _location_to_tag[location] = tag;
    }

    auto offset_data = _data + (size_t) _aligned_dim * location;
    memset((void *) offset_data, 0, sizeof(T) * _aligned_dim);
    memcpy((void *) offset_data, point, sizeof(T) * _dim);

    pool.clear();
    tmp.clear();
    visited.clear();
    std::vector<unsigned> pruned_list;
    unsigned Lindex = parameters.L;

    std::vector<unsigned> init_ids;
    get_expanded_nodes(location, Lindex, init_ids, pool, visited);

    for (unsigned i = 0; i < pool.size(); i++)
      if (pool[i].id == (unsigned) location) {
        pool.erase(pool.begin() + i);
        visited.erase((unsigned) location);
        break;
      }

    prune_neighbors(location, pool, parameters, pruned_list);
    assert(_final_graph.size() == _max_points + _num_frozen_pts);

    _final_graph[location].clear();
    _final_graph[location].shrink_to_fit();
    _final_graph[location].reserve((uint64_t) (range * SLACK_FACTOR * 1.05));

    if (pruned_list.empty()) {
      LOG(INFO) << "Thread: " << std::this_thread::get_id() << "Tag id: " << tag
                << " pruned_list.size(): " << pruned_list.size();
    }

    assert(!pruned_list.empty());
    {
      // v2::SparseWriteLockGuard<uint64_t> guard(&_locks, location);
      v2::LockGuard guard(_locks->wrlock(location));
      for (auto link : pruned_list) {
        _final_graph[location].emplace_back(link);
      }
    }

    assert(_final_graph[location].size() <= range);
    inter_insert(location, pruned_list, parameters);
    return 0;
  }

  template<typename T, typename TagT>
  int Index<T, TagT>::lazy_delete(const TagT &tag) {
    if ((_eager_done) && (!_data_compacted)) {
      LOG(ERROR) << "Eager delete requests were issued but data was not "
                    "compacted, cannot proceed with lazy_deletes";
      return -2;
    }
    std::shared_lock<std::shared_timed_mutex> lock(_update_lock);
    _lazy_done = true;

    {
      std::shared_lock<std::shared_timed_mutex> l(_tag_lock);

      if (_tag_to_location.find(tag) == _tag_to_location.end()) {
        //        LOG(ERROR) << "Delete tag not found";
        return -1;
      }
      assert(_tag_to_location[tag] < _max_points);
    }

    {
      std::unique_lock<std::shared_timed_mutex> l(_delete_lock);
      std::shared_lock<std::shared_timed_mutex> tl(_tag_lock);
      _delete_set.insert(_tag_to_location[tag]);
    }

    {
      std::unique_lock<std::shared_timed_mutex> l(_tag_lock);
      _location_to_tag.erase(_tag_to_location[tag]);
      _tag_to_location.erase(tag);
    }

    return 0;
  }

  // TODO: Check if this function needs a shared_lock on _tag_lock.
  template<typename T, typename TagT>
  int Index<T, TagT>::lazy_delete(const tsl::robin_set<TagT> &tags, std::vector<TagT> &failed_tags) {
    if (failed_tags.size() > 0) {
      LOG(ERROR) << "failed_tags should be passed as an empty list";
      return -3;
    }
    if ((_eager_done) && (!_data_compacted)) {
      LOG(INFO) << "Eager delete requests were issued but data was not "
                   "compacted, cannot proceed with lazy_deletes";
      return -2;
    }
    std::shared_lock<std::shared_timed_mutex> lock(_update_lock);
    _lazy_done = true;

    for (auto tag : tags) {
      //      assert(_tag_to_location[tag] < _max_points);
      if (_tag_to_location.find(tag) == _tag_to_location.end()) {
        failed_tags.push_back(tag);
      } else {
        _delete_set.insert(_tag_to_location[tag]);
        _location_to_tag.erase(_tag_to_location[tag]);
        _tag_to_location.erase(tag);
      }
    }

    return 0;
  }

  template<typename T, typename TagT>
  void Index<T, TagT>::get_active_tags(tsl::robin_set<TagT> &active_tags) {
    active_tags.clear();
    for (auto iter : _tag_to_location) {
      active_tags.insert(iter.first);
    }
  }

  /*  Internals of the library */
  // EXPORTS
  template class Index<float, uint32_t>;
  template class Index<int8_t, uint32_t>;
  template class Index<uint8_t, uint32_t>;
}  // namespace pipeann
