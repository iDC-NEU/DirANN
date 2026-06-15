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

namespace pipeann {
  template<typename T, typename TagT>
  int SSDIndex<T, TagT>::insert_in_place_pmemann(const T *point1, const TagT &tag, tsl::robin_set<uint32_t> *deletion_set) {
    if (unlikely(size_per_io != SECTOR_LEN)) {
      LOG(ERROR) << "Insert not supported for size_per_io == " << size_per_io;
    }

    QueryBuffer<T> *read_data = this->pop_query_buf(point1);
    T *point = read_data->aligned_query_T;  // normalized point for cosine.
    void *ctx = reader->get_ctx();

    uint32_t target_id = cur_id++;

    auto *lvq_handler = static_cast<LVQNeighbor<T, PRIMARY_BITS, RESIDUAL_BITS>*>(nbr_handler);

    // write neighbor (e.g., PQ).
    LVQPointLevel residual_data = lvq_handler->insert_primary_and_get_residual(point, target_id);

    std::vector<Neighbor> exp_node_info;
    tsl::robin_map<uint32_t, T *> coord_map;
    coord_map.reserve(10 * this->l_index);
    
    // Dynamic alloc and not using MAX_N_CMPS to reduce memory footprint.
    T *coord_buf = nullptr;
    alloc_aligned((void **) &coord_buf, 10 * this->l_index * this->aligned_dim, 256);
    
    // re-normalize point1 to support inner_product search (it adds one more dimension, so not idempotent).
    // TODO(yq) : reduce more disk ios for PmemANN.
    this->do_rerank_search(point1, 0, l_index, beam_width, exp_node_info, &coord_map, coord_buf, nullptr, deletion_set,
                         false, nullptr);
    std::vector<uint32_t> new_nhood;
    prune_neighbors(coord_map, exp_node_info, new_nhood);
    
    // lock the pages to write
    aligned_free(coord_buf);

    set_id2loc(target_id, target_id);

    // update loc2id, target_id <-> target_id.
    cur_loc++;  // for target ID, atomic update.
    set_loc2id(target_id, target_id);

    // update the target node.
    uint8_t * pmem_ptr = u_reader->get_addr(pmem_u_loc_offset(target_id));
    PmemNode target_node(target_id, (uint8_t *)pmem_ptr, (uint32_t *)(pmem_ptr + residual_data_len));

    // insert residual data
    memcpy(target_node.codes, &residual_data.bias, sizeof(float));
    memcpy(target_node.codes + sizeof(float), &residual_data.scale, sizeof(float));
    memcpy(target_node.codes + 2 * sizeof(float), residual_data.codes.data(), residual_data.codes.size() * sizeof(uint8_t));

    // update insert point nhood
    target_node.nnbrs = new_nhood.size();
    *(target_node.nbrs - 1) = target_node.nnbrs;
    memcpy(target_node.nbrs, new_nhood.data(), new_nhood.size() * sizeof(uint32_t));
    tags.insert_or_assign(target_id, tag);
    
#ifdef USE_COORD_BUFFER
    this->coord_buffer->put(loc, point1);
#else
    // insert coord to ssd
    alignas(SECTOR_LEN) char sector_buf[SECTOR_LEN];
    // alignas(SECTOR_LEN) static thread_local char sector_buf[SECTOR_LEN];
    uint64_t loc = id2loc(target_id);
    uint64_t pid = loc_sector_no(loc);
    std::vector<IORequest> req(1);
    
    req[0].buf = sector_buf;
    req[0].len = SECTOR_LEN;
    req[0].offset = pid * SECTOR_LEN;
    char * coord_ptr = offset_to_loc(sector_buf, loc);

    page_lock_table.wrlock(pid);
    reader->read(req, ctx, false);

    memcpy(coord_ptr, point1, data_dim * sizeof(T));
    reader->write(req, ctx, false);
    
    page_lock_table.unlock(pid);
#endif
    
    // update the neighbors
    for (uint32_t i = 0; i < new_nhood.size(); ++i) {
      idx_lock_table.wrlock(new_nhood[i]);

      uint8_t *pmem_ptr = u_reader->get_addr(pmem_u_loc_offset_nbr(new_nhood[i]));
      PmemNode nbr_node(new_nhood[i], nullptr, (unsigned *)pmem_ptr); // (perf) : reduce pmem read ios?
      
      std::vector<uint32_t> nhood(nbr_node.nnbrs + 1);
      nhood.assign(nbr_node.nbrs, nbr_node.nbrs + nbr_node.nnbrs);
      nhood.emplace_back(target_id);

      if (nhood.size() > this->range) {  // delta prune neighbors
        auto &thread_pq_buf = read_data->nbr_vec_scratch;
        std::vector<float> tgt_dists(nhood.size(), 0.0f), nbr_dists(nhood.size(), 0.0f);
        nbr_handler->compute_dists(target_id, nhood.data(), nhood.size(), tgt_dists.data(), thread_pq_buf); // just lvq primary
        nbr_handler->compute_dists(nbr_node.id, nhood.data(), nhood.size(), nbr_dists.data(), thread_pq_buf);
        std::vector<TriangleNeighbor> tri_pool(nhood.size());

        for (uint32_t k = 0; k < nhood.size(); k++) {
          tri_pool[k].id = nhood[k];
          tri_pool[k].tgt_dis = tgt_dists[k];
          tri_pool[k].distance = nbr_dists[k];
        }
        std::sort(tri_pool.begin(), tri_pool.end());

        int tgt_idx = -1;
        for (int k = 0; k < (int) nhood.size(); ++k) {
          if (tri_pool[k].id == target_id) {
            tgt_idx = k;
            break;
          }
        }
        if (unlikely(tgt_idx == -1)) {
          LOG(ERROR) << "Target ID " << target_id << " not found in tri_pool";
          exit(-1);
        }
        this->delta_prune_neighbors_pq(tri_pool, nhood, thread_pq_buf, tgt_idx);
      }

      *(nbr_node.nbrs - 1) = (uint32_t) nhood.size();
      memcpy(nbr_node.nbrs, nhood.data(), nbr_node.nnbrs * sizeof(uint32_t));
      idx_lock_table.unlock(new_nhood[i]);
    }

    this->push_query_buf(read_data);
    return target_id;
  }

  template class SSDIndex<float>;
  template class SSDIndex<int8_t>;
  template class SSDIndex<uint8_t>;
}  // namespace pipeann
