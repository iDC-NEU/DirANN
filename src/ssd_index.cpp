#include "aligned_file_reader.h"
#include "ssd_index.h"
#include <malloc.h>
#include <filesystem>

#include <omp.h>
#include <cmath>
#include "nbr/abstract_nbr.h"
#include "ssd_index_defs.h"
#include "utils/timer.h"
#include "utils.h"

#include <unistd.h>
#include <sys/syscall.h>
#include "utils/tsl/robin_set.h"

namespace pipeann {
  template<typename T>
  DiskNode<T>::DiskNode(uint32_t id, T *coords, uint32_t *nhood) : id(id) {
    this->coords = coords;
    this->nnbrs = *nhood;
    this->nbrs = nhood + 1;
  }

  // structs for DiskNode
  template struct DiskNode<float>;
  template struct DiskNode<uint8_t>;
  template struct DiskNode<int8_t>;
  
  PmemNode::PmemNode(uint32_t id, uint8_t *codes, uint32_t *nhood) : id(id) {
    this->codes = codes;
    this->nnbrs = *nhood;
    this->nbrs = nhood + 1;
  }

  template<typename T, typename TagT>
  SSDIndex<T, TagT>::SSDIndex(pipeann::Metric m, std::shared_ptr<AlignedFileReader> fileReader, std::shared_ptr<UnalignedFileReader> u_fileReader,
                              AbstractNeighbor<T> *nbr_handler, bool tags, Parameters *params)
      : reader(fileReader), u_reader(u_fileReader), nbr_handler(nbr_handler), data_is_normalized(false), enable_tags(tags) {
    if (m == pipeann::Metric::COSINE) {
      if (std::is_floating_point<T>::value) {
        LOG(INFO) << "Cosine metric chosen for (normalized) float data."
                      "Changing distance to L2 to boost accuracy.";
        m = pipeann::Metric::L2;
        data_is_normalized = true;

      } else {
        LOG(ERROR) << "WARNING: Cannot normalize integral data types."
                    << " This may result in erroneous results or poor recall."
                    << " Consider using L2 distance with integral data types.";
      }
    }

    this->dist_cmp.reset(pipeann::get_distance_function<T>(m));

    if (params != nullptr) {
      this->beam_width = params->beam_width;
      this->l_index = params->L;
      this->range = params->R;
      this->maxc = params->C;
      this->alpha = params->alpha;
      LOG(INFO) << "Beamwidth: " << this->beam_width << ", L: " << this->l_index << ", R: " << this->range
                << ", C: " << this->maxc;
    }
    LOG(INFO) << "Use " << nbr_handler->get_name() << " as neighbor handler.";
  }

  template<typename T, typename TagT>
  SSDIndex<T, TagT>::SSDIndex(pipeann::Metric m, std::shared_ptr<AlignedFileReader> fileReader,
                              AbstractNeighbor<T> *nbr_handler, bool tags, Parameters *params)
      : reader(fileReader), nbr_handler(nbr_handler), data_is_normalized(false), enable_tags(tags) {
    if (m == pipeann::Metric::COSINE) {
      if (std::is_floating_point<T>::value) {
        LOG(INFO) << "Cosine metric chosen for (normalized) float data."
                     "Changing distance to L2 to boost accuracy.";
        m = pipeann::Metric::L2;
        data_is_normalized = true;

      } else {
        LOG(ERROR) << "WARNING: Cannot normalize integral data types."
                   << " This may result in erroneous results or poor recall."
                   << " Consider using L2 distance with integral data types.";
      }
    }

    this->dist_cmp.reset(pipeann::get_distance_function<T>(m));

    if (params != nullptr) {
      this->beam_width = params->beam_width;
      this->l_index = params->L;
      this->range = params->R;
      this->maxc = params->C;
      this->alpha = params->alpha;
      LOG(INFO) << "Beamwidth: " << this->beam_width << ", L: " << this->l_index << ", R: " << this->range
                << ", C: " << this->maxc;
    }
    LOG(INFO) << "Use " << nbr_handler->get_name() << " as neighbor handler.";
  }

  template<typename T, typename TagT>
  SSDIndex<T, TagT>::~SSDIndex() {
    LOG(INFO) << "Lock table size: " << this->idx_lock_table.size();
    LOG(INFO) << "Page cache size: " << v2::cache.cache.size();

    if (this->coord_buffer != nullptr) {
      this->coord_buffer->flush();
    }
    if (load_flag) {
      this->destroy_buffers();
      reader->close();
    }
  }

  template<typename T, typename TagT>
  void SSDIndex<T, TagT>::copy_index(const std::string &prefix_in, const std::string &prefix_out) {
    LOG(INFO) << "Copying disk index from " << prefix_in << " to " << prefix_out;
    std::filesystem::copy(prefix_in + "_disk.index", prefix_out + "_disk.index",
                          std::filesystem::copy_options::overwrite_existing);
    if (std::filesystem::exists(prefix_in + "_disk.index.tags")) {
      std::filesystem::copy(prefix_in + "_disk.index.tags", prefix_out + "_disk.index.tags",
                            std::filesystem::copy_options::overwrite_existing);
    } else {
      // remove the original tags.
      std::filesystem::remove(prefix_out + "_disk.index.tags");
    }

    // nbr.
    this->nbr_handler->load(prefix_in.c_str());
    this->nbr_handler->save(prefix_out.c_str());

    // partition data
    if (std::filesystem::exists(prefix_in + "_partition.bin.aligned")) {
      std::filesystem::copy(prefix_in + "_partition.bin.aligned", prefix_out + "_partition.bin.aligned",
                            std::filesystem::copy_options::overwrite_existing);
    }
  }

  template<typename T, typename TagT>
  void SSDIndex<T, TagT>::init_buffers(uint64_t n_threads) {
    uint64_t n_buffers = n_threads * 2;
    LOG(INFO) << "Init buffers for " << n_threads << " threads, setup " << n_buffers << " buffers.";
    this->thread_data_queue.null_T = nullptr;
    for (uint64_t i = 0; i < n_buffers; i++) {
      QueryBuffer<T> *data = new QueryBuffer<T>();
      this->init_query_buf(*data);
      this->thread_data_bufs.push_back(data);
      this->thread_data_queue.push(data);
      this->reader->register_buf(data->sector_scratch, MAX_N_SECTOR_READS * SECTOR_LEN, 0);
    }

#ifndef READ_ONLY_TESTS
    // background thread.
    LOG(INFO) << "Setup " << kBgIOThreads << " background I/O threads for insert...";
    for (int i = 0; i < kBgIOThreads; ++i) {
      bg_io_thread_[i] = new std::thread(&SSDIndex<T, TagT>::bg_io_thread, this);
      bg_io_thread_[i]->detach();
    }
#endif
  }

  template<typename T, typename TagT>
  void SSDIndex<T, TagT>::destroy_buffers() {
#ifndef READ_ONLY_TESTS
    for (int i = 0; i < kBgIOThreads; ++i) {
      if (bg_io_thread_[i] != nullptr) {
        auto bg_task = new BgTask{
            .thread_data = nullptr, .writes = {}, .pages_to_unlock = {}, .pages_to_deref = {}, .terminate = true};
        bg_tasks.push(bg_task);
        bg_tasks.push_notify_all();
        bg_io_thread_[i] = nullptr;
      }
    }
#endif

    while (!this->thread_data_bufs.empty()) {
      auto buf = this->thread_data_bufs.back();
      pipeann::aligned_free((void *) buf->coord_scratch);
      pipeann::aligned_free((void *) buf->sector_scratch);
      pipeann::aligned_free((void *) buf->nbr_vec_scratch);
      pipeann::aligned_free((void *) buf->nbr_ctx_scratch);
      pipeann::aligned_free((void *) buf->aligned_dist_scratch);
      pipeann::aligned_free((void *) buf->aligned_query_T);
      this->thread_data_bufs.pop_back();
      this->thread_data_queue.pop();
      delete buf;
    }
  }

  template<typename T, typename TagT>
  void SSDIndex<T, TagT>::load_mem_index(Metric metric, const size_t query_dim, const std::string &mem_index_path) {
    if (mem_index_path.empty()) {
      LOG(ERROR) << "mem_index_path is needed";
      exit(-1);
    }
    mem_index_ = std::make_unique<pipeann::Index<T, uint32_t>>(metric, query_dim, 0, false, false, true);
    mem_index_->load(mem_index_path.c_str());
  }

  template<typename T, typename TagT>
  int SSDIndex<T, TagT>::load(const char *index_prefix, uint32_t num_threads, bool new_index_format,
                              bool use_page_search) {
    std::string disk_index_file = std::string(index_prefix) + "_disk.index";
    this->_disk_index_file = disk_index_file;

    SSDIndexMetadata<T> meta;
    meta.load_from_disk_index(disk_index_file);
    this->init_metadata(meta);

    // load nbrs (e.g., PQ)
    nbr_handler->load(index_prefix);

    // read index metadata
    // open AlignedFileReader handle to index_file
    if (!std::filesystem::exists(disk_index_file)) {
      LOG(ERROR) << "Index file " << disk_index_file << " does not exist!";
      exit(-1);
    }

    this->destroy_buffers();  // in case of re-init.
    reader->open(disk_index_file, true, false);
    this->init_buffers(num_threads);
    this->max_nthreads = num_threads;

    // load page layout.
    this->use_page_search_ = use_page_search;
    this->load_page_layout(index_prefix, nnodes_per_sector, num_points);

    // load tags
    if (this->enable_tags) {
      std::string tag_file = disk_index_file + ".tags";
      LOG(INFO) << "Loading tags from " << tag_file;
      this->load_tags(tag_file);
    }

    load_flag = true;
    LOG(INFO) << "SSDIndex loaded successfully.";
    return 0;
  }

  template<typename T, typename TagT>
  int SSDIndex<T, TagT>::load(const char *index_prefix, const char *pmem_index_prefix, uint32_t num_threads, bool new_index_format,
                              bool use_page_search) {

    std::string disk_index_file = std::string(index_prefix) + "_disk.data";
    std::string pmem_index_file = std::string(pmem_index_prefix) + "_pmem.index";
    
    this->_disk_index_file = disk_index_file;

    SSDIndexMetadata<T> meta;
    PmemIndexMetadata<T> pmem_meta;
    meta.load_from_disk_index(disk_index_file);
    pmem_meta.load_from_pmem_index(pmem_index_file);
    
    if (!std::filesystem::exists(pmem_index_file)) {
      LOG(ERROR) << "Pmem Index file " << pmem_index_file << " does not exist!";
      exit(-1);
    }
    this->init_metadata(meta, pmem_meta);

    // load nbrs (e.g., PQ)
    nbr_handler->load(index_prefix);

    // read index metadata
    // open AlignedFileReader handle to index_file
    if (!std::filesystem::exists(disk_index_file)) {
      LOG(ERROR) << "Index file " << disk_index_file << " does not exist!";
      exit(-1);
    }

    this->destroy_buffers();  // in case of re-init.
    reader->open(disk_index_file, true, false);
    u_reader->open(pmem_index_file);
    this->init_buffers(num_threads);
    this->max_nthreads = num_threads;

    // load page layout.
    this->use_page_search_ = use_page_search;
    this->load_page_layout(index_prefix, nnodes_per_sector, num_points);

    // load tags
    if (this->enable_tags) {
      std::string tag_file = disk_index_file + ".tags";
      LOG(INFO) << "Loading tags from " << tag_file;
      this->load_tags(tag_file);
    }

    coord_buffer.reset(new CoordBuffer(reader, coord_len, 256, num_points));

    load_flag = true;
    LOG(INFO) << "Pmem-SSDIndex loaded successfully.";
    return 0;
  }

  template<typename T, typename TagT>
  uint64_t SSDIndex<T, TagT>::return_nd() {
    return this->num_points;
  }

  template<typename T, typename TagT>
  void SSDIndex<T, TagT>::load_page_layout(const std::string &index_prefix, const uint64_t nnodes_per_sector,
                                           const uint64_t num_points) {
    std::string partition_file = index_prefix + "_partition.bin.aligned";
    id2loc_.resize(num_points);  // pre-allocate space first.
    loc2id_.resize(cur_loc);     // pre-allocate space first.

    if (std::filesystem::exists(partition_file)) {
      LOG(INFO) << "Loading partition file " << partition_file;
      std::ifstream part(partition_file);
      uint64_t C, partition_nums, nd;
      part.read((char *) &C, sizeof(uint64_t));
      part.read((char *) &partition_nums, sizeof(uint64_t));
      part.read((char *) &nd, sizeof(uint64_t));
      if (nnodes_per_sector <= 1 || C != nnodes_per_sector) {
        // graph reordering is useful only when nnodes_per_sector > 1.
        LOG(ERROR) << "partition information not correct.";
        exit(-1);
      }
      LOG(INFO) << "Partition meta: C: " << C << " partition_nums: " << partition_nums;

      uint64_t page_offset = loc_sector_no(0);
      auto st = std::chrono::high_resolution_clock::now();

      constexpr uint64_t n_parts_per_read = 1024 * 1024;
      std::vector<unsigned> part_buf(n_parts_per_read * (1 + nnodes_per_sector));
      for (uint64_t p = 0; p < partition_nums; p += n_parts_per_read) {
        uint64_t nxt_p = std::min(p + n_parts_per_read, partition_nums);
        part.read((char *) part_buf.data(), sizeof(unsigned) * n_parts_per_read * (1 + nnodes_per_sector));
#pragma omp parallel for schedule(dynamic)
        for (uint64_t i = p; i < nxt_p; ++i) {
          uint32_t base = (i - p) * (1 + nnodes_per_sector);
          uint32_t s = part_buf[base];  // size of this partition
          for (uint32_t j = 0; j < s; ++j) {
            uint64_t id = part_buf[base + 1 + j];
            uint64_t loc = i * nnodes_per_sector + j;
            id2loc_[id] = loc;
            loc2id_[loc] = id;
          }
          for (uint32_t j = s; j < nnodes_per_sector; ++j) {
            loc2id_[i * nnodes_per_sector + j] = kInvalidID;
          }
        }
      }
      auto et = std::chrono::high_resolution_clock::now();
      LOG(INFO) << "Page layout loaded in " << std::chrono::duration_cast<std::chrono::milliseconds>(et - st).count()
                << " ms";
    } else {
      LOG(INFO) << partition_file << " does not exist, use equal partition mapping";
// use equal mapping for id2loc and page_layout.
#ifndef NO_MAPPING
#pragma omp parallel for
      for (size_t i = 0; i < this->num_points; ++i) {
        id2loc_[i] = i;
        loc2id_[i] = i;
      }
      for (size_t i = this->num_points; i < this->cur_loc; ++i) {
        loc2id_[i] = kInvalidID;
      }
#endif
    }
    LOG(INFO) << "Page layout loaded.";
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
      pipeann::load_bin<TagT>(tag_file_name, tag_v, tag_num, tag_dim, offset);
      tags.reserve(tag_v.size());

#pragma omp parallel for num_threads(max_nthreads)
      for (size_t i = 0; i < tag_num; ++i) {
        tags.insert_or_assign(i, tag_v[i]);
      }
      LOG(INFO) << "Loaded " << tags.size() << " tags";
    }
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

  template class SSDIndex<float>;
  template class SSDIndex<int8_t>;
  template class SSDIndex<uint8_t>;
}  // namespace pipeann
