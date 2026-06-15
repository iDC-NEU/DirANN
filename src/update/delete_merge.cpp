#include "aligned_file_reader.h"
#include "utils/libcuckoo/cuckoohash_map.hh"
#include "nbr/pq_nbr.h"
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

#include <unistd.h>
#include <sys/syscall.h>
#include "linux_aligned_file_reader.h"

namespace pipeann {
#define SECTORS_PER_MERGE 65536
  template<typename T, typename TagT>
  void SSDIndex<T, TagT>::merge_deletes(const std::string &in_path_prefix, const std::string &out_path_prefix,
                                        const std::vector<TagT> &deleted_nodes,
                                        const tsl::robin_set<TagT> &deleted_nodes_set, uint32_t nthreads,
                                        const uint32_t &n_sampled_nbrs) {
    if (nthreads == 0) {
      nthreads = this->max_nthreads;
    }

    void *ctx = reader->get_ctx();

    while (!bg_tasks.empty()) {
      sleep(5);  // simple way to wait for background IO thread.
    }
    std::string disk_index_out = out_path_prefix + "_disk.index";
    // Note that the index is immutable currently.
    // Step 1: populate neighborhoods, allocate IDs.
    libcuckoo::cuckoohash_map<uint32_t, uint32_t> id_map, rev_id_map;           // old_id -> new_id & new_id -> old_id
    libcuckoo::cuckoohash_map<uint32_t, std::vector<uint32_t>> deleted_nhoods;  // id -> nhood
    std::atomic<uint64_t> new_npoints = 0;
    Timer delete_timer;

    char *rbuf = nullptr, *wbuf = nullptr;
    alloc_aligned((void **) &rbuf, SECTORS_PER_MERGE * SECTOR_LEN, SECTOR_LEN);
    alloc_aligned((void **) &wbuf, 2 * SECTORS_PER_MERGE * SECTOR_LEN, SECTOR_LEN);  // sliding window buffer.
    uint64_t n_sectors = (cur_loc + nnodes_per_sector - 1) / nnodes_per_sector;
    LOG(INFO) << "Cur loc: " << cur_loc.load() << ", cur ID: " << cur_id << ", n_sectors: " << n_sectors
              << ", nnodes_per_sector: " << nnodes_per_sector;

    constexpr int SECTORS_PER_POPULATE = 128;             // small to avoid blocking search threads.
    uint32_t populate_nthreads = std::min(nthreads, 4u);  // restrict the flow.

    for (uint64_t in_sector = 0; in_sector < n_sectors; in_sector += SECTORS_PER_POPULATE) {
      uint64_t st_sector = in_sector, ed_sector = std::min(in_sector + SECTORS_PER_POPULATE, n_sectors);
      uint64_t loc_st = st_sector * nnodes_per_sector, loc_ed = std::min(cur_loc.load(), ed_sector * nnodes_per_sector);
      uint64_t n_sectors_to_read = ed_sector - st_sector;
      std::vector<IORequest> read_reqs;
      read_reqs.push_back(IORequest(loc_sector_no(loc_st) * SECTOR_LEN, n_sectors_to_read * size_per_io, rbuf, 0, 0));
      reader->read(read_reqs, ctx, false);

#pragma omp parallel for num_threads(populate_nthreads)
      for (uint64_t loc = loc_st; loc < loc_ed; ++loc) {
        // populate nhood.
        uint64_t id = loc2id(loc);
        if (id == kInvalidID) {
          continue;
        }

        uint64_t tag = id2tag(id);
        if (deleted_nodes_set.find(tag) == deleted_nodes_set.end()) {  // 2. not deleted, alloc ID.
          // allocate ID.
          uint64_t new_id = new_npoints.fetch_add(1);
          id_map.insert(id, new_id);
          rev_id_map.insert(new_id, id);
          continue;
        }

        // 3. deleted, populate nhoods.
        auto page_rbuf = rbuf + (loc / nnodes_per_sector - st_sector) * SECTOR_LEN;
        auto node_rbuf = offset_to_loc(page_rbuf, loc);
        DiskNode<T> node(id, offset_to_node_coords(node_rbuf), offset_to_node_nhood(node_rbuf));
        std::vector<uint32_t> nhood;
        for (uint32_t i = 0; i < node.nnbrs; ++i) {
          uint32_t nbr_tag = id2tag(node.nbrs[i]);
          if (deleted_nodes_set.find(nbr_tag) == deleted_nodes_set.end()) {
            nhood.push_back(node.nbrs[i]);  // filtered neighborhoods.
          }
        }
        // sample for less space consumption.
        if (nhood.size() > n_sampled_nbrs) {
          // std::shuffle(nhood.begin(), nhood.end(), std::default_random_engine());
          nhood.resize(n_sampled_nbrs);  // nearest.
        }
        deleted_nhoods.insert(id, nhood);
      }
    }
    LOG(INFO) << "Finished populating neighborhoods, totally elapsed: " << delete_timer.elapsed() / 1e3
              << "ms, new npoints: " << new_npoints.load() << " " << "id_map size: " << id_map.size();
    LOG(INFO) << "Deleted nodes size: " << deleted_nodes.size() << ", deleted_nhoods size: " << deleted_nhoods.size();

    // Step 2: prune neighbors, populate PQ and tags.
    int fd = open(disk_index_out.c_str(), O_DIRECT | O_LARGEFILE | O_RDWR | O_CREAT, 0755);
    const uint64_t kVecInWBuf = 2 * SECTORS_PER_MERGE * nnodes_per_sector;
    uint64_t wb_id = 0;
    std::atomic<uint64_t> n_used_id = 0;
    auto write_back = [&]() {
      // write one buffer.
      uint64_t buf_id = (wb_id % kVecInWBuf) / (nnodes_per_sector * SECTORS_PER_MERGE);
      auto b = wbuf + buf_id * SECTORS_PER_MERGE * SECTOR_LEN;
      std::vector<IORequest> write_reqs;
      uint64_t id_delta = std::min((uint64_t) SECTORS_PER_MERGE * nnodes_per_sector, n_used_id - wb_id);
      write_reqs.push_back(IORequest(loc_sector_no(wb_id) * SECTOR_LEN,
                                     ROUND_UP(id_delta, nnodes_per_sector) / nnodes_per_sector * size_per_io, b, 0, 0));
      reader->write_fd(fd, write_reqs, ctx);
      wb_id += id_delta;
      LOG(INFO) << "Write back " << wb_id << "/" << n_used_id << " IDs.";
    };

    std::vector<TagT> new_tags(new_npoints);

    for (uint64_t in_sector = 0; in_sector < n_sectors; in_sector += SECTORS_PER_MERGE) {
      uint64_t st_sector = in_sector, ed_sector = std::min(in_sector + SECTORS_PER_MERGE, n_sectors);
      uint64_t loc_st = st_sector * nnodes_per_sector, loc_ed = std::min(cur_loc.load(), ed_sector * nnodes_per_sector);
      uint64_t n_sectors_to_read = ed_sector - st_sector;
      std::vector<IORequest> read_reqs;
      read_reqs.push_back(IORequest(loc_sector_no(loc_st) * SECTOR_LEN, n_sectors_to_read * size_per_io, rbuf, 0, 0));
      reader->read(read_reqs, ctx, false);  // read in fd

#pragma omp parallel for num_threads(nthreads)
      for (uint64_t loc = loc_st; loc < loc_ed; ++loc) {
        uint64_t id = loc2id(loc);
        if (id == kInvalidID) {
          continue;
        }

        uint64_t tag = id2tag(id);
        if (deleted_nodes_set.find(tag) != deleted_nodes_set.end()) {  // deleted.
          continue;
        }

        auto page_rbuf = rbuf + (loc / nnodes_per_sector - st_sector) * SECTOR_LEN;
        auto loc_rbuf = offset_to_loc(page_rbuf, loc);
        DiskNode<T> node(id, offset_to_node_coords(loc_rbuf), offset_to_node_nhood(loc_rbuf));
        // prune neighbors.
        std::unordered_set<uint32_t> nhood_set;
        for (uint32_t i = 0; i < node.nnbrs; ++i) {
          uint32_t nbr_tag = id2tag(node.nbrs[i]);
          if (deleted_nodes_set.find(nbr_tag) != deleted_nodes_set.end()) {
            // deleted, insert neighbors.
            const auto &nhoods = deleted_nhoods.find(node.nbrs[i]);
            nhood_set.insert(nhoods.begin(), nhoods.end());
          } else {
            nhood_set.insert(node.nbrs[i]);
            // LOG(INFO) << id << " insert " << node.nbrs[i];
          }
        }
        nhood_set.erase(id);  // remove self.
        std::vector<uint32_t> nhood(nhood_set.begin(), nhood_set.end());

        if (nhood.size() > this->range) {
          std::vector<float> dists(nhood.size(), 0.0f);
          std::vector<Neighbor> pool(nhood.size());
          // Use dynamic buffer instead of pre-initialized buffer to save space.
          uint8_t *pq_buf = nullptr;
          pipeann::alloc_aligned((void **) &pq_buf, nhood.size() * AbstractNeighbor<T>::MAX_BYTES_PER_NBR, 256);
          nbr_handler->compute_dists(id, nhood.data(), nhood.size(), dists.data(), pq_buf);

          for (uint32_t k = 0; k < nhood.size(); k++) {
            pool[k].id = nhood[k];
            pool[k].distance = dists[k];
          }
          std::sort(pool.begin(), pool.end());
          if (pool.size() > this->maxc) {
            pool.resize(this->maxc);
          }
          nhood.clear();
          this->prune_neighbors_pq(pool, nhood, pq_buf);
          pipeann::aligned_free(pq_buf);
        }

        // map to new IDs.
        for (auto &nbr : nhood) {
          nbr = id_map.find(nbr);
          if (unlikely(nbr > new_npoints)) {
            LOG(ERROR) << "Invalid neighbor ID: " << nbr << ", new_npoints: " << new_npoints;
          }
        }

        // write neighbors.
        uint64_t new_id = id_map.find(id);
        uint64_t off = new_id % kVecInWBuf;
        auto page_wbuf = wbuf + (off / nnodes_per_sector) * SECTOR_LEN;
        auto loc_wbuf = offset_to_loc(page_wbuf, off);
        DiskNode<T> w_node(new_id, offset_to_node_coords(loc_wbuf), offset_to_node_nhood(loc_wbuf));
        memcpy(w_node.coords, node.coords, data_dim * sizeof(T));
        w_node.nnbrs = nhood.size();
        *(w_node.nbrs - 1) = w_node.nnbrs;
        memcpy(w_node.nbrs, nhood.data(), w_node.nnbrs * sizeof(uint32_t));
        ++n_used_id;
        // copy tags.
        new_tags[new_id] = id2tag(id);
      }

      LOG(INFO) << "Processed " << ed_sector << "/" << n_sectors << " sectors, n_used_id: " << n_used_id << ".";
      if (n_used_id - wb_id >= SECTORS_PER_MERGE * nnodes_per_sector) {
        write_back();
      }
    }

    while (wb_id < n_used_id) {
      write_back();
    }
    LOG(INFO) << "Write nhoods finished, totally elapsed " << delete_timer.elapsed() / 1e3 << "ms.";

    // duplicate neighbor handler.
    auto new_nbr_handler = this->nbr_handler->shuffle(rev_id_map, new_npoints, nthreads);

    while (deleted_nodes_set.find(id2tag(medoid)) != deleted_nodes_set.end()) {
      LOG(INFO) << "Medoid deleted. Choosing another start node. Medoid ID: " << medoid << " tag: " << id2tag(medoid);
      const auto &nhoods = deleted_nhoods.find(medoid);
      medoid = nhoods[0];
    }
    close(fd);
    // free buf
    aligned_free((void *) rbuf);
    aligned_free((void *) wbuf);

    // set metadata, PQ and tags.
    merge_lock.lock();  // unlock in reload().
    // metadata.
    this->num_points = new_npoints;
    this->medoid = id_map.find(medoid);
    // PQ.
    auto tmp = this->nbr_handler;
    this->nbr_handler = new_nbr_handler;
    delete tmp;
    // tags.
    tags.clear();
    // no need to clear id2loc & loc2id as they are arrays.
    // out-of-bound loc2id is initialized in reload().
#pragma omp parallel for num_threads(nthreads)
    for (size_t i = 0; i < new_tags.size(); ++i) {
      tags.insert_or_assign(i, new_tags[i]);
      set_id2loc(i, i);
      set_loc2id(i, i);
    }

    this->write_metadata_and_pq(in_path_prefix, out_path_prefix, new_npoints, medoid, &new_tags);
    LOG(INFO) << "Write metadata and PQ finished, totally elapsed " << delete_timer.elapsed() / 1e3 << "ms.";
  }

  template<typename T, typename TagT>
  void SSDIndex<T, TagT>::write_metadata_and_pq(const std::string &in_path_prefix, const std::string &out_path_prefix,
                                                const uint64_t &new_npoints, const uint64_t &new_medoid,
                                                std::vector<TagT> *new_tags) {
    SSDIndexMetadata<T> meta(new_npoints, this->data_dim, new_medoid, this->max_node_len, this->nnodes_per_sector);
    meta.print();
    std::string disk_index_out = out_path_prefix + "_disk.index";
    meta.save_to_disk_index(disk_index_out);

    // Step 3. Write tags and PQ.
    std::vector<TagT> tags_vec;
    if (new_tags == nullptr) {
      tags_vec.resize(new_npoints);
      for (uint64_t i = 0; i < new_npoints; ++i) {
        tags_vec[i] = id2tag(i);
      }
      new_tags = &tags_vec;
    }
    pipeann::save_bin<TagT>(out_path_prefix + "_disk.index.tags", new_tags->data(), new_npoints, 1, 0);
    nbr_handler->save(out_path_prefix.c_str());
  }

  template<typename T, typename TagT>
  void SSDIndex<T, TagT>::reload(const char *index_prefix, uint32_t num_threads) {
    std::string iprefix = std::string(index_prefix);
    std::string disk_index_file = iprefix + "_disk.index";
    this->_disk_index_file = disk_index_file;
    this->max_nthreads = num_threads;

    reader->close();
    reader->open(disk_index_file, true, false);

    // reload metadata.
    SSDIndexMetadata<T> meta;
    meta.load_from_disk_index(disk_index_file);
    this->init_metadata(meta);

    // No need to reload PQ, as it is already reloaded in merge_deletes.
    for (uint32_t i = this->num_points; i < this->cur_loc; ++i) {
      set_loc2id(i, kInvalidID);  // reset loc2id.
    }
    while (!this->empty_pages.empty()) {
      this->empty_pages.pop();
    }
    merge_lock.unlock();
    LOG(INFO) << "Reload finished, cur_id: " << this->cur_id << ", cur_loc: " << this->cur_loc;
    return;
  }

  template class SSDIndex<float>;
  template class SSDIndex<int8_t>;
  template class SSDIndex<uint8_t>;
}  // namespace pipeann
