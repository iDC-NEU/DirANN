#include "aligned_file_reader.h"
#include "utils/libcuckoo/cuckoohash_map.hh"
#include "ssd_index.h"
#include <malloc.h>
#include <algorithm>
#include <filesystem>

#include <omp.h>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <tuple>
#include "utils/timer.h"
#include "utils/tsl/robin_map.h"
#include "utils.h"
#include "utils/page_cache.h"
#include "nbr/lvq_nbr.h"

#include <unistd.h>
#include <sys/syscall.h>
#include "linux_aligned_file_reader.h"

#include "v3/global_stats.h"

// #define USE_RESIDUAL

namespace pipeann {
  template<typename T, typename TagT>
  void SSDIndex<T, TagT>::do_rerank_search(const T *query1, uint32_t mem_L, uint32_t l_search, const uint32_t beam_width,
                                         std::vector<Neighbor> &expanded_nodes_info,
                                         tsl::robin_map<uint32_t, T *> *coord_map, T *coord_buf, QueryStats *stats,
                                         tsl::robin_set<uint32_t> *exclude_nodes /* tags */, bool dyn_search_l,
                                         std::vector<uint64_t> *passthrough_page_ref) {
    uint32_t original_l_search = l_search;
    auto diskSearchBegin = std::chrono::high_resolution_clock::now();

    auto query_buf = pop_query_buf(query1);
    void *ctx = reader->get_ctx();

    const T *query = query_buf->aligned_query_T;

    // reset query
    query_buf->reset();

    // pointers to current vector for comparison
    T *data_buf = query_buf->coord_scratch;
    _mm_prefetch((char *) data_buf, _MM_HINT_T1);

    // sector scratch
    char *sector_scratch = query_buf->sector_scratch;
    uint64_t &sector_scratch_idx = query_buf->sector_idx;

    auto *lvq_handler = static_cast<LVQNeighbor<T, PRIMARY_BITS, RESIDUAL_BITS>*>(nbr_handler);

    lvq_handler->initialize_query(query, query_buf);
    float *dist_scratch = query_buf->aligned_dist_scratch;

    Timer query_timer, io_timer, cpu_timer;
    std::vector<Neighbor> retset;
    std::vector<Neighbor> half_retset;
    retset.resize(mem_L + 10 * l_search);
    half_retset.reserve(l_search * 2);
    tsl::robin_set<uint64_t> visited(4096);

    // re-naming `expanded_nodes_info` to not change rest of the code
    std::vector<Neighbor> &full_retset = expanded_nodes_info;
    full_retset.reserve(10 * l_search);

    unsigned cur_list_size = 0;
    auto compute_and_add_to_retset = [&](const unsigned *node_ids, const uint64_t n_ids) {
      cpu_timer.reset();
      lvq_handler->compute_dists_only_primary(query_buf, node_ids, n_ids);
      if (stats != nullptr) {
        stats->primary_4bit_cpu_us += (double) cpu_timer.elapsed();
      }
      for (uint64_t i = 0; i < n_ids; ++i) {
        retset[cur_list_size].id = node_ids[i];
        retset[cur_list_size].distance = dist_scratch[i];
        retset[cur_list_size++].flag = true;
        visited.insert(node_ids[i]);
      }
    };

    if (mem_L) {
      std::vector<unsigned> mem_tags(mem_L);
      std::vector<float> mem_dists(mem_L);
      mem_index_->search_with_tags(query, mem_L, mem_L, mem_tags.data(), mem_dists.data());
      compute_and_add_to_retset(mem_tags.data(), std::min((unsigned) mem_L, (unsigned) l_search));
    } else {
      // Do not use optimized start point.
      compute_and_add_to_retset(&medoid, 1);
    }

    std::sort(retset.begin(), retset.begin() + cur_list_size);

    unsigned cmps = 0;
    unsigned hops = 0;
    unsigned num_ios = 0;
    unsigned k = 0;

    // cleared every iteration
    std::vector<unsigned> frontier;
    using fnhood_t = std::tuple<unsigned, unsigned, char *>;
    std::vector<fnhood_t> frontier_nhoods;
    std::vector<IORequest> frontier_read_reqs;
    std::vector<uint32_t> vec_rdlocks;

    std::vector<uint64_t> new_page_ref{};
    std::vector<uint64_t> &page_ref = passthrough_page_ref ? *passthrough_page_ref : new_page_ref;

    uint64_t coord_buf_idx = 0;  // used by coord_map
    while (k < cur_list_size) {
      auto nk = cur_list_size;
      // clear iteration state
      frontier.clear();
      frontier_nhoods.clear();
      frontier_read_reqs.clear();
      vec_rdlocks.clear();
      sector_scratch_idx = 0;
      // find new beam
      // WAS: uint64_t marker = k - 1;
      uint32_t marker = k;
      uint32_t num_seen = 0;
      while (marker < cur_list_size && frontier.size() < beam_width && num_seen < beam_width) {
        if (retset[marker].flag) {
          num_seen++;
          frontier.push_back(retset[marker].id);
          #ifndef USE_RESIDUAL
          half_retset.push_back(Neighbor(retset[marker].id, retset[marker].distance, true)); // add lvq/pq distance to half retset
          #endif
          retset[marker].flag = false;
        }
        marker++;
      }

      // read nhoods of frontier ids
      std::vector<uint32_t> locked;
      if (!frontier.empty()) {
        if (stats != nullptr)
          stats->n_hops++;
        locked = this->lock_idx(idx_lock_table, kInvalidID, frontier, true);
        for (uint64_t i = 0; i < frontier.size(); i++) {
          uint32_t id = frontier[i];
          uint32_t loc = this->id2loc(id);
          
          char * pmem_ptr = (char *)u_reader->get_addr(pmem_u_loc_offset(loc)); // zero-copy, TODO(yq) : add prefetch?          
          fnhood_t fnhood = std::make_tuple(id, loc, pmem_ptr);
          
          frontier_nhoods.push_back(fnhood);
          
          if (stats != nullptr) {
            stats->n_4k++;
            stats->n_ios++;
          }
          num_ios++;
        }
        this->unlock_idx(idx_lock_table, locked);
      }

      for (auto &frontier_nhood : frontier_nhoods) {
        auto [id, loc, pmem_ptr] = frontier_nhood;
        unsigned *nhood_ptr = pmem_offset_to_node_nhood(pmem_ptr);
        io_timer.reset();
        pipeann::prefetch_vector((const char *) pmem_ptr, pmem_node_len);
        uint64_t nnbrs = (uint64_t) (*nhood_ptr);

        unsigned *node_nbrs = (nhood_ptr + 1);
        if (stats != nullptr) {
          stats->pmem_io_us += (double) io_timer.elapsed();
        }
        // compute node_nbrs <-> query dist in PQ space
        cpu_timer.reset();
        lvq_handler->compute_dists_only_primary(query_buf, node_nbrs, nnbrs);
        if (stats != nullptr) {
          stats->n_cmps += (double) nnbrs;
          stats->cpu_us += (double) cpu_timer.elapsed();
          stats->primary_4bit_cpu_us += (double) cpu_timer.elapsed();
        }   

        #ifdef USE_RESIDUAL
        cpu_timer.reset();
        lvq_handler->compute_dists_with_residual(query_buf, &id, 1, pmem_ptr);
        if (stats != nullptr) {
          stats->cpu_us += (double) cpu_timer.elapsed();
          stats->residual_8bit_io_cpu_us += (double) cpu_timer.elapsed();
        }
        float dist = dist_scratch[0];
        half_retset.push_back(Neighbor(id, dist, true)); // add lvq/pq distance to half retset
        #endif

        cpu_timer.reset();
        // std::cerr << "nnbrs " << nnbrs << std::endl;
        // process prefetch-ed nhood
        for (uint64_t m = 0; m < nnbrs; ++m) {
          unsigned id = node_nbrs[m];
          // std::cerr << "id " << id << std::endl; 
          if (unlikely(id > this->cur_id)) {
            LOG(ERROR) << "ID is larger than current ID, " << id << " vs " << this->cur_id;
            crash();
          }
          if (visited.find(id) != visited.end()) {
            continue;
          } else {
            visited.insert(id);
            cmps++;
            float dist = dist_scratch[m];
            if (stats != nullptr) {
              stats->n_cmps++;
            }
            if (dist >= retset[cur_list_size - 1].distance && (cur_list_size == l_search))
              continue;
            Neighbor nn(id, dist, true);
            // variable search_L for deleted nodes.
            // Return position in sorted list where nn inserted.

            auto r = InsertIntoPool(retset.data(), cur_list_size, nn);

            if (cur_list_size < l_search) {
              ++cur_list_size;
              if (unlikely(cur_list_size >= retset.size())) {
                retset.resize(2 * cur_list_size);
              }
            }

            if (r < nk)
              nk = r;  // nk logs the best position in the retset that was
                       // updated due to neighbors of n.
          }
        }

        if (dyn_search_l) {
          // TODO(gh): contention still exists in id2tag(x)
          // O(n), but it is not slow as L is typically smaller than 300.
          // l_search monotonically increases to handle deleted nodes.
          uint32_t tot = 0, cur = 0;
          for (cur = 0; cur < cur_list_size; ++cur) {
            uint32_t tag = id2tag(retset[cur].id);
            if (exclude_nodes->find(tag) == exclude_nodes->end()) {
              ++tot;
              if (tot == original_l_search) {
                break;
              }
            }
          }
          // cur is the stopped index (cur + 1 is the length it should be)
          l_search = std::max(original_l_search, cur + 1);
        }

        if (stats != nullptr) {
          stats->cpu_us += (double) cpu_timer.elapsed();
        }
      }

      // update best inserted position
      //

      if (nk <= k)
        k = nk;  // k is the best position in retset updated in this round.
      else
        ++k;

      hops++;
      if (stats != nullptr && stats->n_current_used != 0) {
        auto diskSearchEnd = std::chrono::high_resolution_clock::now();
        double elapsedSeconds =
            std::chrono::duration_cast<std::chrono::milliseconds>(diskSearchEnd - diskSearchBegin).count();
        if (elapsedSeconds >= stats->n_current_used)
          break;
      }
    }

    Timer rerank_timer;
    if (1)
    {
      // sort by distance
      std::sort(half_retset.begin(), half_retset.end(),
                [](const Neighbor &left, const Neighbor &right) { return left < right; });
      tsl::robin_map<unsigned, char *> page_buf_map;
      uint32_t data_buf_idx = 0;
      half_retset.resize(l_search);
      for(auto node : half_retset){ // remove deleted nodes
        TagT tag = id2tag(node.id);
        if (exclude_nodes != nullptr && exclude_nodes->find(tag) != exclude_nodes->end()) {
          continue;
        }
        full_retset.push_back(node);
      }
      full_retset.resize(std::min(l_search, 25u));
    
      std::vector<std::pair<int, char *>> idx_buf;
      std::vector<IORequest> read_reqs;
      idx_buf.reserve(full_retset.size());
      read_reqs.reserve(full_retset.size());
      page_buf_map.clear();

      std::vector<uint32_t> to_lock(full_retset.size());
      for(size_t i = 0; i < full_retset.size(); i++){
        to_lock[i] = full_retset[i].id;
      }
      sort(to_lock.begin(), to_lock.end());

      auto last = std::unique(to_lock.begin(), to_lock.end());
      if(unlikely(last != to_lock.end())){
        LOG(ERROR) << "Rerank, some id are duplicated.";
        exit(-1);
      }
      
      for (auto &id : to_lock) {
        vec_lock_table.rdlock(id);
      }
      
      for(size_t i = 0; i < full_retset.size(); i++){
        auto id = full_retset[i].id;
        
        uint64_t loc = id2loc(id);
        uint64_t pid = loc_sector_no(loc);
        
        if(unlikely(loc == kInvalidID)){
          LOG(ERROR) << "id " << id << " not found in id2loc.";
          exit(-1);
        }
        
        if (page_buf_map.find(pid) != page_buf_map.end()) {
          idx_buf.push_back({i, offset_to_loc(page_buf_map[pid], loc)});
          continue;
        }
        uint64_t &cur_buf_idx = query_buf->sector_idx;
        auto buf = sector_scratch + cur_buf_idx * size_per_io;
        
#ifdef USE_COORD_BUFFER
        if (coord_buffer != nullptr) {
          if(coord_buffer->get(loc, (uint8_t *)offset_to_loc(buf, loc))){
            idx_buf.push_back({i, offset_to_loc(buf, loc)});
            cur_buf_idx = (cur_buf_idx + 1) % MAX_N_SECTOR_READS;
            continue;
          };
        }
#endif
        
        read_reqs.push_back(IORequest(static_cast<uint64_t>(pid) * SECTOR_LEN, size_per_io, buf, 0, 0));
        page_buf_map.insert({pid, buf});
        idx_buf.push_back({i, offset_to_loc(buf, loc)});
        cur_buf_idx = (cur_buf_idx + 1) % MAX_N_SECTOR_READS;
      }
      
      io_timer.reset();
      reader->read(read_reqs, ctx);
      if (stats != nullptr) {
        stats->rerank_io_us += (double) io_timer.elapsed();
      }

      for (auto &id : to_lock) {
        vec_lock_table.unlock(id);
      }

      Timer timer;
      for(auto [idx, node_buf] : idx_buf){
        unsigned id = full_retset[idx].id;
        T *node_fp_coords_copy = data_buf;
        memcpy(node_fp_coords_copy, node_buf, data_dim * sizeof(T));
        
        if (coord_map != nullptr) {
          if (unlikely(coord_buf == nullptr || coord_buf_idx > this->l_index * 10)) {
            LOG(ERROR) << "Please allocate larger coord_buf.";
            crash();
          }
          T *coord_ptr = coord_buf + coord_buf_idx * aligned_dim;
          memcpy(coord_ptr, node_fp_coords_copy, data_dim * sizeof(T));
          coord_map->insert(std::make_pair(id, node_fp_coords_copy));
          coord_buf_idx++;
        }
        
        float cur_expanded_dist = dist_cmp->compare(query, node_fp_coords_copy, (unsigned) aligned_dim);
        full_retset[idx].distance = cur_expanded_dist;
      }
    } else {
      full_retset = half_retset;
    }
    
    if (stats != nullptr) {
      stats->rerank_us += (double) rerank_timer.elapsed();
    }

    // re-sort by distance
    std::sort(full_retset.begin(), full_retset.end(),
              [](const Neighbor &left, const Neighbor &right) { return left < right; });

    if (passthrough_page_ref == nullptr) {
      reader->deref(&page_ref, ctx);
    }

    push_query_buf(query_buf);

    if (stats != nullptr) {
      stats->total_us = (double) query_timer.elapsed();
    }
  }

  template<typename T, typename TagT>
  size_t SSDIndex<T, TagT>::rerank_search(const T *query, const uint64_t k_search, const uint32_t mem_L,
                                        const uint64_t l_search, TagT *res_tags, float *distances,
                                        const uint64_t beam_width, QueryStats *stats,
                                        tsl::robin_set<uint32_t> *deleted_nodes, bool dyn_search_l) {
    // iterate to fixed point
    std::shared_lock lk(merge_lock);
    std::vector<Neighbor> expanded_nodes_info;
    this->do_rerank_search(query, mem_L, (uint32_t) l_search, (uint32_t) beam_width, expanded_nodes_info, nullptr,
                         nullptr, stats, deleted_nodes, dyn_search_l);
    uint64_t res_count = 0;
    for (uint32_t i = 0; i < l_search && res_count < k_search && i < expanded_nodes_info.size(); i++) {
      res_tags[res_count] = id2tag(expanded_nodes_info[i].id);
      distances[res_count] = expanded_nodes_info[i].distance;
      res_count++;
    }
    return res_count;
  }

  template class SSDIndex<float>;
  template class SSDIndex<int8_t>;
  template class SSDIndex<uint8_t>;
}  // namespace pipeann
