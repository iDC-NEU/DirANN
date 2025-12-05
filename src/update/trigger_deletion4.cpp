#include "aligned_file_reader.h"
#include "libcuckoo/cuckoohash_map.hh"
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
#include "timer.h"
#include "tsl/robin_map.h"
#include "utils.h"
#include "v2/page_cache.h"
#include "set"

#include <unistd.h>
#include <sys/syscall.h>
#include "linux_aligned_file_reader.h"

#include "global_stats.h"

// #define USE_TOPO_DISK    // 取消宏定义
namespace dirann {
  #define SECTORS_PER_MERGE 65536
  template<typename T, typename TagT>
  void SSDIndex<T, TagT>::trigger_deletion_in_place4(const tsl::robin_set<TagT> &deleted_nodes_set, uint32_t nthreads,
                                        const uint32_t &n_sampled_nbrs) {
    if (nthreads == 0) {
      nthreads = this->max_nthreads;
    }

    Timer delete_timer;

    Timer delete_io_timer;
    double delete_io = 0;
    double reverse_graph_time_0 = 0;

    void *ctx = reader->get_ctx();

    // Note that the index is immutable currently.
    // Step 1: populate neighborhoods, allocate IDs.
    libcuckoo::cuckoohash_map<uint32_t, std::vector<uint32_t>> deleted_nhoods;  // id -> nhood
    libcuckoo::cuckoohash_map<uint32_t, std::vector<float>> deleted_nhoods_dist;  // id -> nhood_dist
    std::vector<uint32_t> deleted_ids;  // log deleted ids

    auto p_reader = (LinuxAlignedFileReader *)this->reader.get();
    int strategy = p_reader->strategy;

    std::mutex mu1, mu2, mu3;
    std::vector<uint32_t> affected_ids;
    LOG(INFO) << "Merge thread: " << nthreads;
    uint64_t n_sectors = (cur_loc + nnodes_per_sector - 1) / nnodes_per_sector;

    if (1) {
      delete_io_timer.reset();
      #pragma omp parallel for num_threads(nthreads) schedule(dynamic, 1024)
      for (uint32_t loc = 0; loc < cur_loc; loc++) {
        uint32_t id = loc2id(loc);
        if (id == kInvalidID) {
          continue;
        }
        uint64_t tag = id2tag(id);
        if (deleted_nodes_set.find(tag) != deleted_nodes_set.end()) {  // deleted, continue
          {
            std::unique_lock<std::mutex> lock(mu2);
            deleted_ids.push_back(id);
          }
          std::vector<int32_t> in_nbrs;
          if (id != this->medoids[0]) 
            this->disk_in_graph->get_edges(id, in_nbrs);
          for (int32_t id : in_nbrs) {
            uint64_t tag = id2tag(id);
            if (deleted_nodes_set.find(tag) == deleted_nodes_set.end()) {  // not deleted, add to affected_ids
              // #pragma omp 
              std::unique_lock<std::mutex> lock(mu1);
              affected_ids.push_back(id);
            }
          }
          // affected_ids.insert(affected_ids.end(), in_nbrs.begin(), in_nbrs.end());
        }
      }
      reverse_graph_time_0 += delete_io_timer.elapsed();
      delete_io += delete_io_timer.elapsed();
      std::sort(affected_ids.begin(), affected_ids.end());
      affected_ids.erase(std::unique(affected_ids.begin(), affected_ids.end()), affected_ids.end());
    } 
    LOG(INFO) << "reverse_graph_time_0 " << reverse_graph_time_0 / 1e6 << " s.";
    LOG(INFO) << "Finished 1, totally elapsed: " << delete_timer.elapsed() / 1e3
              << "ms";
    if(gs != nullptr){
      gs->in_graph_count+=deleted_nodes_set.size();
    }

    LOG(INFO) << "affected ids size: " << affected_ids.size();
    // for (int i = 0; i < test.size(); ++i) {
    //   std::cout << test[i] << "\n";
    //   // if (i > 30) break;
    // }

    tsl::robin_set<uint32_t> affected_page_ids;
    tsl::robin_map<uint32_t, uint32_t> page2idx;
    for (uint32_t id : affected_ids) {
      uint32_t pid = loc_sector_no(id2loc(id));
      affected_page_ids.insert(pid);
    }

    std::vector<std::vector<Neighbor>> cand_nhoods(cur_loc.load());
    std::vector<std::shared_mutex> mu4(cur_loc.load());

    char *buf = nullptr;
    alloc_aligned((void **) &buf, std::max((uint64_t)affected_page_ids.size(), (uint64_t)SECTORS_PER_MERGE) * SECTOR_LEN, SECTOR_LEN);

    int buf_idx = 0;
    std::vector<IORequest> read_reqs;
    for(auto page_id : affected_page_ids){
      read_reqs.emplace_back(IORequest(page_id * SECTOR_LEN, SECTOR_LEN, buf + buf_idx * SECTOR_LEN, 0, 0));
      page2idx[page_id] = buf_idx;
      buf_idx++;
    }
    
    LOG(INFO) << "Cur loc: " << cur_loc.load() << " Cur topo loc: " << cur_loc_topo.load() 
              << " Cur coord loc: " << cur_loc_coord.load() 
              << ", cur ID: " << cur_id << ", n_sectors: " << n_sectors
              << ", nnodes_per_sector: " << nnodes_per_sector << ", ntopo_per_sector: " << ntopo_per_sector;
    LOG(INFO) << "Tags size: " << tags.size() << " layout size: " << page_layout.size() << " id2loc_ size: " << id2loc_.size();

    if (1) {
      
      // 2、读取删除点邻居
      double reverse_graph_time_1 = 0;
      libcuckoo::cuckoohash_map<uint64_t, char *> buf_map;
      {
        std::vector<IORequest> read_reqs;
        std::atomic<uint32_t> buf_idx {0};
        // #pragma omp parallel for num_threads(nthreads) schedule(dynamic, 128)
        for (size_t i = 0; i < deleted_ids.size(); ++i) {
          uint32_t id = deleted_ids[i];
          uint32_t loc = id2loc(id);
          if (unlikely(loc == kInvalidID)) {
            LOG(ERROR) << "Invalid loc: " << id;
            exit(-1);
          }
          uint32_t idx = buf_idx.fetch_add(1);
          char * sector_buf = buf + idx * SECTOR_LEN;
          uint64_t pid = loc_sector_no(loc);
          
          {
            // std::unique_lock<std::shared_mutex> lock(mu1);
            read_reqs.push_back(IORequest(pid * SECTOR_LEN, SECTOR_LEN, sector_buf, 0, 0));
          }
          buf_map.insert(id, sector_buf);
        }
        delete_io_timer.reset();
        reader->read(read_reqs, ctx, false);
        if (gs != nullptr) {
          gs->delete_io += delete_io_timer.elapsed();
          gs->update_ios += read_reqs.size() * SECTOR_LEN;
          gs->delete_ios += read_reqs.size() * SECTOR_LEN;
        }
        delete_io += delete_io_timer.elapsed();
      }
      LOG(INFO) << "Finished 2, totally elapsed: " << delete_timer.elapsed() / 1e3
                << "ms";

      #pragma omp parallel for num_threads(nthreads) schedule(dynamic, 10)
      for (size_t i = 0; i < deleted_ids.size(); ++i) {
        uint32_t z_id = deleted_ids[i];
        uint32_t loc = id2loc(z_id);
        if (unlikely(loc == kInvalidID)) {
          LOG(ERROR) << "Invalid loc: " << z_id;
          exit(-1);
        }
        
        char * sector_buf = buf_map.find(z_id);
        char * node_buf = offset_to_loc(sector_buf, loc);
        DiskNode<T> node(z_id, offset_to_node_coords(node_buf), offset_to_node_nhood(node_buf));
        // std::vector<uint32_t> nhood;
        // nhood.reserve(node.nnbrs);
        auto &thread_pq_buf = thread_pq_bufs[omp_get_thread_num()];
        std::vector<int32_t> tmp;
        std::vector<uint32_t> x_ids;
        this->disk_in_graph->get_edges(z_id, tmp);
        
        x_ids.reserve(tmp.size());
        for (int32_t x_id : tmp) {
          uint64_t tag = id2tag(x_id);
          if (deleted_nodes_set.find(tag) == deleted_nodes_set.end()) {  // not deleted, add to affected_ids
            x_ids.push_back(x_id);
          }
        }
        std::vector<float> xy_dists(x_ids.size(), 0.0f);
        // std::vector<float> xz_dists(x_ids.size(), 0.0f);
        // std::vector<float> yz_dists(node.nnbrs, 0.0f);

        // compute_pq_dists(z_id, x_ids.data(), xz_dists.data(), (_u32)x_ids.size(), thread_pq_buf);
        // compute_pq_dists(z_id, node.nbrs, yz_dists.data(), (_u32)node.nnbrs, thread_pq_buf);

        tsl::robin_map<uint32_t, tsl::robin_map<uint32_t, float>> dists_map;
        std::vector<Neighbor> pool;
        for (uint32_t i = 0; i < node.nnbrs; ++i) {
          uint64_t y_id = node.nbrs[i];
          TagT nbr_tag = id2tag(y_id);
          if (deleted_nodes_set.find(nbr_tag) == deleted_nodes_set.end()) {
            // nhood.push_back(y_id);  // filtered neighborhoods.
            // if (nhood.size() > 512) continue;
            pool.clear();
            compute_pq_dists(y_id, x_ids.data(), xy_dists.data(), (_u32)x_ids.size(), thread_pq_buf); // 可能因为thread_pq_buf出问题
            
            // for (int j = 0; j < x_ids.size(); ++j) {
            //   uint32_t x_id = x_ids[j];
            //   std::unique_lock<std::shared_mutex> lock(mu4[x_id]);
            //   cand_nhoods[x_id].emplace_back(Neighbor(y_id, xy_dists[j], true));
            // }
            
            for (uint32_t j = 0; j < x_ids.size(); ++j) {
              pool.push_back(Neighbor(x_ids[j], xy_dists[j], true));
            }
            std::sort(pool.begin(), pool.end());
            std::vector<float> occlude_factor(pool.size(), 0.0f);
            std::vector<Neighbor> pruned_list;
            // occlude_list_pq(pool, pruned_list, occlude_factor, thread_pq_buf);

            if (1)
            {
              auto &result = pruned_list;
              auto &scratch = thread_pq_buf;
              std::set<Neighbor> result_set;  // deduplication, and keep distance sorted.
              #ifdef REORDER_COMPUTE_PQ
                  std::vector<uint32_t> idxs(pool.size()), ids(pool.size());
                  std::vector<float> dijks(pool.size());
              #endif
                  float cur_alpha = 1;
                  while (cur_alpha <= alpha && result_set.size() < range) {
                    uint32_t start = 0;
                    while (result_set.size() < range && (start) < pool.size() && start < maxc) {
                      auto &p = pool[start];
                      if (occlude_factor[start] > cur_alpha) {
                        start++;
                        continue;
                      }
                      occlude_factor[start] = std::numeric_limits<float>::max();
                      result_set.insert(p);
              #ifdef REORDER_COMPUTE_PQ
                      idxs.clear();
                      ids.clear();
                      for (uint32_t t = start + 1; t < pool.size() && t < maxc; t++) {
                          if (occlude_factor[t] > alpha)
                              continue;
                          idxs.push_back(t);
                          ids.push_back(pool[t].id);
                      }
              
                      compute_pq_dists(p.id, ids.data(), dijks.data(), ids.size(), scratch);
              
                      for (uint32_t i = 0; i < idxs.size(); i ++) {
                          auto t = idxs[i];
                          occlude_factor[t] = (std::max)(occlude_factor[t], pool[t].distance / dijks[i]);
                      }
              #else
                      // dynamic programming, if p (current) is included,
                      // then D(t, p0) / D(t, p) should be updated.
                      for (uint32_t t = start + 1; t < pool.size() && t < maxc; t++) {
                        if (occlude_factor[t] > alpha)
                          continue;
                        // djk = dist(p.id, pool[t.id])
                        float djk;
                        auto &a = p.id;
                        auto &b = pool[t].id;
                        if (a > b) {
                          std::swap(a, b);
                        }
                        if (dists_map.find(a) != dists_map.end() && dists_map[a].find(b) != dists_map[a].end()) {
                          djk = dists_map[a][b];
                        } else {
                          compute_pq_dists(a, &b, &djk, 1, scratch);
                          dists_map[a][b] = djk;
                        }
                        // LOG(INFO) << pool[t].distance << " " << djk << " " << alpha << " " << result_set.size();
                        occlude_factor[t] = (std::max)(occlude_factor[t], pool[t].distance / djk);
                      }
              #endif
                      start++;
                    }
                    cur_alpha *= 1.2f;
                  }
                  for (auto &x : result_set) {
                    result.push_back(x);
                  }
            }
            
            for (auto &nbr : pruned_list) {
              auto x_id = nbr.id;
              std::unique_lock<std::shared_mutex> lock(mu4[x_id]);
              cand_nhoods[x_id].emplace_back(Neighbor(y_id, nbr.distance, true));
            }
          }

          Timer timer;
          if (y_id != this->medoids[0]) {
            disk_in_graph->del_edge(y_id, z_id);
            // std::cerr << "del " << id << " " << id << " #\n";
          }
          reverse_graph_time_1 += timer.elapsed();
        }

        
        if (gs != nullptr) {
          gs->in_graph_count += node.nnbrs;
        }
      }
    } else {
      
      double reverse_graph_time_1 = 0;
      for (uint64_t in_sector = 0; in_sector < n_sectors; in_sector += SECTORS_PER_MERGE) {
        uint64_t st_sector = in_sector, ed_sector = std::min(in_sector + SECTORS_PER_MERGE, n_sectors);
        uint64_t loc_st = st_sector * nnodes_per_sector, loc_ed = std::min(cur_loc.load(), ed_sector * nnodes_per_sector);
        uint64_t n_sectors_to_read = ed_sector - st_sector;
        std::vector<IORequest> read_reqs;
        tsl::robin_set<uint64_t> page_set;
        std::vector<std::pair<uint32_t, unsigned *>> deleted_ids_buf;

        #pragma omp parallel for num_threads(nthreads) schedule(dynamic, 1024)
        for (uint64_t loc = loc_st; loc < loc_ed; ++loc) {
          // populate nhood.
          uint64_t id = loc2id(loc);
          if (id == kInvalidID) {
            continue;
          }

          uint64_t tag = id2tag(id);
          if (deleted_nodes_set.find(tag) == deleted_nodes_set.end()) {  // 2. not deleted, alloc ID.
            // allocate ID.
            continue;
          }

          char * page_buf = buf + (loc / nnodes_per_sector - st_sector) * SECTOR_LEN;
          uint64_t pid = loc_sector_no(loc);

          // #pragma omp critical
          {
            std::unique_lock<std::mutex> lock(mu1);
            deleted_ids.push_back(id);
            deleted_ids_buf.emplace_back(id, (unsigned *)offset_to_loc(page_buf, loc));
          }

          // #pragma omp critical
          std::unique_lock<std::mutex> lock(mu2);
          if (page_set.find(pid) == page_set.end()) {
            read_reqs.push_back(IORequest(pid * SECTOR_LEN, SECTOR_LEN, page_buf, 0, 0));
            page_set.insert(pid);
          }
        }

        delete_io_timer.reset();
        reader->read(read_reqs, ctx, false);
        if (gs != nullptr) {
          gs->delete_io += delete_io_timer.elapsed();
          gs->update_ios += read_reqs.size() * SECTOR_LEN;
          gs->delete_ios += read_reqs.size() * SECTOR_LEN;
        }
        delete_io += delete_io_timer.elapsed();

        #pragma omp parallel for num_threads(nthreads) schedule(dynamic, 1024)
        for (size_t i = 0; i < deleted_ids_buf.size(); ++i) {
          auto [z_id, nhood_buf] = deleted_ids_buf[i];
          
          DiskNode<T> node(z_id, offset_to_node_coords((char *)nhood_buf), offset_to_node_nhood((char *)nhood_buf));
          
          
          // std::vector<uint32_t> nhood;
          // nhood.reserve(node.nnbrs);
          auto &thread_pq_buf = thread_pq_bufs[omp_get_thread_num()];
          std::vector<int32_t> tmp;
          std::vector<uint32_t> x_ids;
          this->disk_in_graph->get_edges(z_id, tmp);
          
          x_ids.reserve(tmp.size());
          for (int32_t x_id : tmp) {
            uint64_t tag = id2tag(x_id);
            if (deleted_nodes_set.find(tag) == deleted_nodes_set.end()) {  // not deleted, add to affected_ids
              x_ids.push_back(x_id);
            }
          }
          std::vector<float> xy_dists(x_ids.size(), 0.0f);
          std::vector<float> xz_dists(x_ids.size(), 0.0f);
          std::vector<float> yz_dists(node.nnbrs, 0.0f);

          // compute_pq_dists(z_id, x_ids.data(), xz_dists.data(), (_u32)x_ids.size(), thread_pq_buf);
          // compute_pq_dists(z_id, node.nbrs, yz_dists.data(), (_u32)node.nnbrs, thread_pq_buf);

          std::vector<Neighbor> pool;
          for (uint32_t i = 0; i < node.nnbrs; ++i) {
            uint64_t y_id = node.nbrs[i];
            TagT nbr_tag = id2tag(y_id);
            if (deleted_nodes_set.find(nbr_tag) == deleted_nodes_set.end()) {
              // nhood.push_back(y_id);  // filtered neighborhoods.
              // if (nhood.size() > 512) continue;
              pool.clear();
              compute_pq_dists(y_id, x_ids.data(), xy_dists.data(), (_u32)x_ids.size(), thread_pq_buf); // 可能因为thread_pq_buf出问题
              
              // for (int j = 0; j < x_ids.size(); ++j) {
              //   uint32_t x_id = x_ids[j];
              //   std::unique_lock<std::shared_mutex> lock(mu4[x_id]);
              //   cand_nhoods[x_id].emplace_back(Neighbor(y_id, xy_dists[j], true));
              // }
              
              for (uint32_t j = 0; j < x_ids.size(); ++j) {
                pool.push_back(Neighbor(x_ids[j], xy_dists[j], true));
              }
              std::sort(pool.begin(), pool.end());
              std::vector<float> occlude_factor(pool.size(), 0.0f);
              std::vector<Neighbor> pruned_list;
              occlude_list_pq(pool, pruned_list, occlude_factor, thread_pq_buf);
              
              for (auto &nbr : pruned_list) {
                auto x_id = nbr.id;
                std::unique_lock<std::shared_mutex> lock(mu4[x_id]);
                cand_nhoods[x_id].emplace_back(Neighbor(y_id, nbr.distance, true));
              }
            }
          }


  // #ifdef COLLECT_INFO 
          if (gs != nullptr) {
            gs->in_graph_count += node.nnbrs;
          }
  // #endif
        }
      }
    }
    LOG(INFO) << "Finished populating neighborhoods 3, totally elapsed: " << delete_timer.elapsed() / 1e3
              << "ms";
              // , new npoints: " << new_npoints.load() << " " << "id_map size: " << id_map.size();

    // LOG(INFO) << "reverse_graph_time_1 " << reverse_graph_time_1 / 1e6 / nthreads << " s";
    // Step 2: prune neighbors, populate PQ and tags.
    
    delete_io_timer.reset();
    reader->read(read_reqs, ctx, false);
    delete_io += delete_io_timer.elapsed();
    if (gs != nullptr) {
      gs->update_ios += read_reqs.size() * SECTOR_LEN;
      gs->delete_ios += read_reqs.size() * SECTOR_LEN;
    }

    LOG(INFO) << "Prune neighbors begin, totally elapsed: " << delete_timer.elapsed() / 1e3 << "ms";

    LOG(ERROR) << "affected ids size: " << affected_ids.size();
    LOG(ERROR) << "affected page ids size: " << affected_page_ids.size();
    
    double reverse_graph_time_2 = 0.0;
#pragma omp parallel for num_threads(nthreads) schedule(dynamic, 1024)
    for (uint32_t k = 0 ; k < (uint32_t) affected_ids.size(); k++) {
      uint32_t id = affected_ids[k];
      uint32_t loc = id2loc(id);
      uint32_t pid = loc_sector_no(loc);
      char * page_buf = buf + page2idx[pid] * SECTOR_LEN;
      char * node_buf = offset_to_loc(page_buf, loc);
      DiskNode<T> node(id, offset_to_node_nhood(node_buf));

      // if (0)
      {
      uint32_t x_id = node.id;
      // std::unordered_set<uint32_t> nhood_set0, nhood_set1;
      std::vector<Neighbor> final_pool;
      final_pool.reserve(this->range * 2);
      std::unordered_map<uint32_t, int> nhood_id_map;

      auto &thread_pq_buf = thread_pq_bufs[omp_get_thread_num()];
      std::vector<float> xz_dists(node.nnbrs, 0.0f);
      compute_pq_dists(x_id, node.nbrs, xz_dists.data(), node.nnbrs, thread_pq_buf);

      if (gs != nullptr) {
        gs->compute_pq_dists += node.nnbrs;
      }


      for (uint32_t i = 0; i < node.nnbrs; ++i) {
        uint32_t nbr_tag;
        uint32_t z_id = node.nbrs[i];

        nbr_tag = id2tag(z_id);
        if (deleted_nodes_set.find(nbr_tag) != deleted_nodes_set.end()) {
          // deleted, insert neighbors.
          // const auto &y_ids = deleted_nhoods.find(z_id);
          // const auto &zy_dists = deleted_nhoods_dist.find(z_id);

          // std::vector<float> xy_dists(y_ids.size(), 0.0f);
          // compute_pq_dists(x_id, y_ids.data(), xy_dists.data(), (_u32) y_ids.size(), thread_pq_buf);
          
          // std::vector<Neighbor> pool;
          // pool.reserve(y_ids.size());
          // for(uint32_t j = 0; j < y_ids.size(); j++){
          //   if (y_ids[j] == node.id) continue;
          //   // if (xy_dists[j] > zy_dists[j]){
          //     // nhood_set1.insert(y_ids[j]);
          //     pool.push_back(Neighbor(y_ids[j], xy_dists[j], true));
          //   // }
          // }

          // std::sort(pool.begin(), pool.end());
          // std::vector<float> occlude_factor(pool.size(), 0.0f);
          // std::vector<Neighbor> pruned_list;
          // occlude_list_pq(pool, pruned_list, occlude_factor, thread_pq_buf);
          // for (auto &nbr : pool) {
          //   nhood_id_map[nbr.id] = 1;
          //   final_pool.push_back(nbr);
          //   // nhood_set1.insert(nbr.id);
          // }
        } else {
          if (z_id == node.id) continue;
          // nhood_set0.insert(z_id);
          nhood_id_map[z_id] = 0;
          final_pool.push_back(Neighbor(z_id, xz_dists[i], false));
        }
      }

      for (auto &y : cand_nhoods[x_id]) {
        if (node.id == y.id) continue;
        nhood_id_map[y.id] = 1;
        final_pool.push_back(y);
      }
      // Timer prune_timer;

      // nhood_set.erase(id);  // remove self.
      if (final_pool.empty()) { // deep1b 会大量触发此分支
        // LOG(ERROR) << "No neighbors after prune!";
        *(node.nbrs - 1) = 0;
        continue;
      }
      
      std::vector<uint32_t> original_nbrs(node.nbrs, node.nbrs + node.nnbrs);
      auto &pool = final_pool;
      std::sort(pool.begin(), pool.end());
      pool.erase(std::unique(pool.begin(), pool.end()), pool.end());
      auto &scratch = thread_pq_buf;
      std::vector<float> occlude_factor(pool.size(), 0.0f);
      std::set<Neighbor> result_set;  // deduplication, and keep distance sorted.
  #ifdef REORDER_COMPUTE_PQ
      std::vector<uint32_t> idxs(pool.size()), ids(pool.size());
      std::vector<float> dijks(pool.size());
  #endif
      float cur_alpha = 1;
      while (cur_alpha <= alpha && result_set.size() < range) {
        uint32_t start = 0;
        while (result_set.size() < range && (start) < pool.size() && start < maxc) {
          auto &p = pool[start];
          if (occlude_factor[start] > cur_alpha) {
            start++;
            continue;
          }
          occlude_factor[start] = std::numeric_limits<float>::max();
          result_set.insert(p);
  #ifdef REORDER_COMPUTE_PQ
          idxs.clear();
          ids.clear();
          for (uint32_t t = start + 1; t < pool.size() && t < maxc; t++) {
              if (occlude_factor[t] > alpha) continue;
              // if (nhood_id_map[p.id] == nhood_id_map[pool[t] .id]) continue; // same group, skip
              idxs.push_back(t);
              ids.push_back(pool[t].id);
          }

          compute_pq_dists(p.id, ids.data(), dijks.data(), ids.size(), scratch);

          for (uint32_t i = 0; i < idxs.size(); i ++) {
              auto t = idxs[i];
              occlude_factor[t] = (std::max)(occlude_factor[t], pool[t].distance / dijks[i]);
          }
  #else
          // dynamic programming, if p (current) is included,
          // then D(t, p0) / D(t, p) should be updated.
          for (uint32_t t = start + 1; t < pool.size() && t < maxc; t++) {
            if (occlude_factor[t] > alpha)
              continue;
            // djk = dist(p.id, pool[t.id])
            float djk;
            if (nhood_id_map[p.id] == nhood_id_map[pool[t] .id]) continue; // same group, skip
            compute_pq_dists(p.id, &(pool[t].id), &djk, 1, scratch);
            // LOG(INFO) << pool[t].distance << " " << djk << " " << alpha << " " << result_set.size();
            occlude_factor[t] = (std::max)(occlude_factor[t], pool[t].distance / djk);
          }
  #endif
          start++;
        }
        cur_alpha *= 1.2f;
      }

      std::vector<uint32_t> pruned_list;
      
      assert(result_set.size() <= range);
      for (auto iter : result_set) {
        pruned_list.emplace_back(iter.id);
      }
      if (alpha > 1) {
        for (uint32_t i = 0; i < pool.size() && pruned_list.size() < range; i++) {
          if (std::find(pruned_list.begin(), pruned_list.end(), pool[i].id) == pruned_list.end())
            pruned_list.emplace_back(pool[i].id);
        }
      }

      auto &nhood = pruned_list;
      std::sort(original_nbrs.begin(), original_nbrs.end()); // 原始邻居
      std::sort(nhood.begin(), nhood.end()); // 新邻居
      
      auto &a = original_nbrs;
      auto &b = nhood;
      std::vector<uint32_t> diff_a, diff_b;

      std::set_difference(a.begin(), a.end(),
                          b.begin(), b.end(),
                          std::back_inserter(diff_a)); // A - B

      std::set_difference(b.begin(), b.end(),
                          a.begin(), a.end(),
                          std::back_inserter(diff_b)); // B - A

      Timer timer;
      for (auto &z_id : diff_a) {
        if (z_id != this->medoids[0]) 
          disk_in_graph->del_edge(z_id, x_id);
        // std::cerr << "del " << z_id << " " << x_id << "\n";
      }
      for (auto &z_id : diff_b) {
        if (z_id != this->medoids[0]) 
          disk_in_graph->add_edge(z_id, x_id);
        // std::cerr << "add " << z_id << " " << x_id << "\n";
      }

      reverse_graph_time_2 += timer.elapsed();

      if (gs != nullptr) {
        gs->in_graph_count += diff_a.size() + diff_b.size();
      }

      // write neighbors.
      node.nnbrs = nhood.size();
      *(node.nbrs - 1) = node.nnbrs;
      memcpy(node.nbrs, nhood.data(), node.nnbrs * sizeof(uint32_t));
      
      }
      
      // if (gs != nullptr){
      //   gs->occlude_list_pq += prune_timer.elapsed();
      // }
      // if (gs != nullptr) {
      //   #pragma omp critical
      //   gs->consolidate_deletes += t.elapsed();
      // }
    }
    LOG(INFO) << "Prune finished, totally elapsed " << delete_timer.elapsed() / 1e3 << "ms.";

    delete_io_timer.reset();
    //TODO : double buffer
    reader->write(read_reqs, ctx, false);  // write back
    delete_io += delete_io_timer.elapsed();
    
    if (gs != nullptr) {
      gs->delete_io += delete_io_timer.elapsed();
      gs->delete_ios += read_reqs.size() * SECTOR_LEN;
      gs->update_ios += read_reqs.size() * SECTOR_LEN;
    }

    LOG(INFO) << "reverse_graph_time_2 " << reverse_graph_time_2 / nthreads /  1e6 << " s.";
    LOG(INFO) << "Write nhoods finished, totally elapsed " << delete_timer.elapsed() / 1e3 << "ms.";
    LOG(INFO) << "Delete io cost " << delete_io / 1e6 << " s.";

    if (gs != nullptr) {
      std::cerr << "in_graph_count: " << gs->in_graph_count << std::endl;
      gs->delete_ios += gs->in_graph_count * SECTOR_LEN;
      gs->update_ios += gs->in_graph_count * SECTOR_LEN;
    }
      // std::cerr << "cpu time: " << cpu_time / 1e6/ thread_num << "s" << std::endl;
      // std::cerr << "cpu2 time: " << cpu2_time / 1e6/ thread_num << "s" << std::endl;
      // std::cerr << "cpu3 time: " << cpu3_time / 1e6 << "s" << std::endl;
      // std::cerr << "prune time: " << (tmp.elapsed() / 1e6) << "s" << std::endl;
    uint32_t medoid = this->medoids[0];
    while (deleted_nodes_set.find(id2tag(medoid)) != deleted_nodes_set.end()) {
      LOG(INFO) << "Medoid deleted. Choosing another start node.";
      const auto &nhoods = deleted_nhoods.find(medoid);
      medoid = nhoods[0];
    }
    // free buf
    aligned_free((void *) buf);

    // set metadata, PQ and tags.
    // merge_lock.lock();  // unlock in reload().
    // metadata.
    this->medoids[0] = medoid;
    // PQ.
    // tags.
    // std::cerr << "cnt0: " << cnt0 << std::endl;
    // std::cerr << "cnt1: " << cnt1 << std::endl;
#pragma omp parallel for num_threads(nthreads)
    for (auto id: deleted_ids) { // TODO : use topo disk.
      tags.erase(id);
      erase_loc2id(id2loc(id));
      id2loc_.erase(id);
    }
    
    // LOG(INFO) << "empty_locs_topo size is " << empty_locs_topo.size();
    // LOG(INFO) << "empty_locs_coord size is " << empty_locs_coord.size();
    // LOG(INFO) << "empty_pages_topo size is " << empty_pages_topo.size();
    // LOG(INFO) << "empty_pages_coord size is " << empty_pages_coord.size();
    LOG(INFO) << "Write metadata finished, totally elapsed " << delete_timer.elapsed() / 1e3 << "ms.";
    std::cerr << "$E2E delete cost " << delete_timer.elapsed() / 1e6 << " s.\n";
    // exit(0);
  }

  template class SSDIndex<float>;
  template class SSDIndex<_s8>;
  template class SSDIndex<_u8>;
}  // namespace dirann
