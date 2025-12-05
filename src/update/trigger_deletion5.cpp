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
  void SSDIndex<T, TagT>::trigger_deletion_in_place5(const tsl::robin_set<TagT> &deleted_nodes_set, uint32_t nthreads,
                                        const uint32_t & _) {
    if (nthreads == 0) {
      nthreads = this->max_nthreads;
    }

    std::ignore = _;

    auto print_vec = [](const std::vector<uint32_t> &vec, std::string name) {
      std::cerr << name << " " << vec.size() << " : ";
      for (auto &x : vec) {
        std::cerr << x << " ";
      }
      std::cerr << "\n";
    };

    Timer delete_timer;

    Timer delete_io_timer;
    Timer timer;
    double delete_io = 0;
    double reverse_graph_time_0 = 0;

    void *ctx = reader->get_ctx();

    libcuckoo::cuckoohash_map<uint32_t, std::pair<std::vector<uint32_t>, std::vector<float>>> deleted_nhoods;  // id -> nhood
    std::atomic<uint64_t> new_npoints {0};

    auto p_reader = (LinuxAlignedFileReader *)this->reader.get();

    std::vector<uint64_t> deleted_ids;
    std::vector<uint64_t> affected_pages;
    tsl::robin_set<uint64_t> affected_ids;
    std::shared_mutex mu1, mu2, mu3;
    LOG(INFO) << "Delete-Merge thread: " << nthreads;
    
    // 1、找出受影响点，和受影响的块
    
    {
      delete_io_timer.reset();
      #pragma omp parallel for num_threads(nthreads) schedule(dynamic, 1024)
      for (uint64_t loc = 0; loc < cur_loc; loc++) {     
        uint64_t id = loc2id(loc);
        
        if (id == kInvalidID) {
          uint64_t pid = loc_sector_no(loc);
          {
            std::unique_lock<std::shared_mutex> lock(mu1);
            affected_pages.push_back(pid);
          }
          continue;
        }
        
        TagT tag = id2tag(id);
        if (deleted_nodes_set.find(tag) != deleted_nodes_set.end()) {  // deleted, continue
          {
            std::unique_lock<std::shared_mutex> lock(mu2);
            deleted_ids.push_back(id);
          }

          std::vector<int32_t> in_nbrs;
          #ifdef COLLECT_INFO
          timer.reset();
          #endif
          if (id != this->medoids[0]) this->disk_in_graph->get_edges(id, in_nbrs);
          
          #ifdef COLLECT_INFO
          if (gs!= nullptr) {
            gs->reverse_graph_op_time.fetch_add(timer.elapsed());
          }
          #endif
          for (int32_t id : in_nbrs) {
            uint64_t tag = id2tag(id);
            if (deleted_nodes_set.find(tag) == deleted_nodes_set.end()) {  // not deleted, add to affected_ids
              uint32_t pid = loc_sector_no(id2loc(id));
              
              {
                std::unique_lock<std::shared_mutex> lock(mu1);
                affected_pages.push_back(pid);
              }
              {
                std::unique_lock<std::shared_mutex> lock(mu3);
                affected_ids.insert(id);
              }
            }
          }
          uint32_t pid = loc_sector_no(id2loc(id));
          {
            std::unique_lock<std::shared_mutex> lock(mu1);
            affected_pages.push_back(pid);
          }
        } else {
          new_npoints.fetch_add(1);
        }
      }
      std::sort(affected_pages.begin(), affected_pages.end());
      affected_pages.erase(std::unique(affected_pages.begin(), affected_pages.end()), affected_pages.end());
      reverse_graph_time_0 += delete_io_timer.elapsed();
      delete_io += delete_io_timer.elapsed();
      LOG(INFO) << "reverse_graph_time_0 " << reverse_graph_time_0 / 1e6 << " s.";
    } 
    LOG(INFO) << "1.Finished find all affected page, totally elapsed: " << delete_timer.elapsed() / 1e3
              << "ms";
    
    if(gs != nullptr){
      gs->in_graph_count += deleted_nodes_set.size();
    }

    uint64_t n_sectors = (cur_loc.load() + nnodes_per_sector - 1) / nnodes_per_sector;
    uint64_t new_n_sectors = (new_npoints.load() + nnodes_per_sector - 1) / nnodes_per_sector;
    
    char *buf = nullptr;
    alloc_aligned((void **) &buf, (uint64_t)deleted_ids.size() * SECTOR_LEN, SECTOR_LEN);
    
    LOG(INFO) << "Cur loc: " << cur_loc.load()
              << ", cur ID: " << cur_id.load() << ", n_sectors: " << n_sectors
              << ", new_n_sectors: " << new_n_sectors << ", nnodes_per_sector: " << nnodes_per_sector;
    LOG(INFO) << "Tags size: " << tags.size() << " layout size: " << page_layout.size() << " id2loc_ size: " << id2loc_.size()
              << " new_npoints: " << new_npoints.load();

    // 2、读取删除点邻居
    double reverse_graph_time_1 = 0;
    libcuckoo::cuckoohash_map<uint64_t, char *> buf_map;
    {
      std::vector<IORequest> read_reqs;
      std::atomic<uint64_t> buf_idx {0};
      // #pragma omp parallel for num_threads(nthreads) schedule(dynamic, 128)
      for (size_t i = 0; i < deleted_ids.size(); ++i) { // perf: add omp for deep1b
        uint32_t id = deleted_ids[i];
        uint32_t loc = id2loc(id);
        if (unlikely(loc == kInvalidID)) {
          LOG(ERROR) << "Invalid loc: " << id;
          exit(-1);
        }
        uint64_t idx = buf_idx.fetch_add(1);
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

    #pragma omp parallel for num_threads(nthreads) schedule(dynamic, 128)
    for (size_t i = 0; i < deleted_ids.size(); ++i) {
      uint32_t id = deleted_ids[i];
      uint32_t loc = id2loc(id);
      if (unlikely(loc == kInvalidID)) {
        LOG(ERROR) << "Invalid loc: " << id;
        exit(-1);
      }
      char * sector_buf = buf_map.find(id);
      char * node_buf = offset_to_loc(sector_buf, loc);
      DiskNode<T> node(id, offset_to_node_coords(node_buf), offset_to_node_nhood(node_buf));
      std::vector<uint32_t> nhood;
      nhood.reserve(node.nnbrs);
      for (uint32_t i = 0; i < node.nnbrs; ++i) {
        TagT nbr_tag = id2tag(node.nbrs[i]);
        if (deleted_nodes_set.find(nbr_tag) == deleted_nodes_set.end()) {
          nhood.push_back(node.nbrs[i]);  // filtered neighborhoods.
        }
        Timer timer;
        if (node.nbrs[i] != this->medoids[0]) {
          disk_in_graph->del_edge(node.nbrs[i], id);
          // std::cerr << "del " << node.nbrs[i] << " " << id << " #\n";
        }
        reverse_graph_time_1 += timer.elapsed();
        
        #ifdef COLLECT_INFO
        if (gs!= nullptr) {
          gs->reverse_graph_op_time.fetch_add(timer.elapsed());
        }
        #endif
      }

      std::vector<float> dists(nhood.size(), 0.0f);
      std::vector<Neighbor> pool(nhood.size());
      auto &thread_pq_buf = thread_pq_bufs[omp_get_thread_num()];
      compute_pq_dists(id, nhood.data(), dists.data(), (_u32)nhood.size(), thread_pq_buf);
      deleted_nhoods.insert(id, std::make_pair(nhood, dists));
      // deleted_nhoods_dist.insert(id, dists);
      
      if (gs != nullptr) {
        gs->in_graph_count += node.nnbrs;
      }
    }

    LOG(INFO) << "2.Finished populating neighborhoods, totally elapsed: " << delete_timer.elapsed() / 1e3
              << "ms";
    LOG(INFO) << "reverse_graph_time_1 " << reverse_graph_time_1 / 1e6 / nthreads << " s";
    
    aligned_free((void *) buf);
    buf = nullptr;
#define sqr(x) ((x)*(x))
    double reverse_graph_time_2 = 0;

    auto consolidate_deletes = [&](DiskNode<T> &node, uint8_t *thread_pq_buf) {
      uint32_t x_id = node.id;
      // std::unordered_set<uint32_t> nhood_set0, nhood_set1;
      std::vector<Neighbor> final_pool;
      final_pool.reserve(this->range * 2);
      tsl::robin_map<uint32_t, int> nhood_id_map;

      std::vector<float> xz_dists(node.nnbrs, 0.0f);
      compute_pq_dists(x_id, node.nbrs, xz_dists.data(), node.nnbrs, thread_pq_buf);

      if (gs != nullptr) {
        gs->compute_pq_dists += node.nnbrs;
      }

      for (uint32_t i = 0; i < node.nnbrs; ++i) {
        uint32_t z_id = node.nbrs[i];
        TagT nbr_tag = id2tag(z_id);
        if (deleted_nodes_set.find(nbr_tag) != deleted_nodes_set.end()) {
          // deleted, insert neighbors.
          const auto &[y_ids, zy_dists] = deleted_nhoods.find(z_id);
          // const auto &zy_dists = deleted_nhoods_dist.find(z_id);

          std::vector<float> xy_dists(y_ids.size(), 0.0f);
          compute_pq_dists(x_id, y_ids.data(), xy_dists.data(), (_u32) y_ids.size(), thread_pq_buf);
          
          std::vector<Neighbor> pool;
          pool.reserve(y_ids.size());
          for(uint32_t j = 0; j < y_ids.size(); j++){
            if (y_ids[j] == node.id) continue;
            // if (xy_dists[j] > zy_dists[j]){
            #ifndef DONT_USE_OPT1
            if (sqr(zy_dists[j]) + sqr(xy_dists[j]) <= sqr(xz_dists[i])) continue;
            #endif
              // nhood_set1.insert(y_ids[j]);
            pool.push_back(Neighbor(y_ids[j], xy_dists[j], true));
          }

          #ifndef DONT_USE_OPT2
          std::sort(pool.begin(), pool.end());
          std::vector<float> occlude_factor(pool.size(), 0.0f);
          std::vector<Neighbor> pruned_list;
          occlude_list_pq(pool, pruned_list, occlude_factor, thread_pq_buf);
          for (auto &nbr : pruned_list) {
            nhood_id_map[nbr.id] = 1;
            final_pool.push_back(nbr);
          }
          #else
          for (auto &nbr : pool) {
            nhood_id_map[nbr.id] = 0;
            final_pool.push_back(nbr);
          }
          #endif
        } else {
          if (z_id == node.id) continue;
          nhood_id_map[z_id] = 0;
          final_pool.push_back(Neighbor(z_id, xz_dists[i], true));
        }
      }

      // nhood_set.erase(id);  // remove self.
      if (final_pool.empty()) { // deep1b 会大量触发此分支
        // LOG(ERROR) << "No neighbors after prune!";
        *(node.nbrs - 1) = 0;
        return;
      }
      
      std::vector<uint32_t> original_nbrs(node.nbrs, node.nbrs + node.nnbrs);
      auto &pool = final_pool;
      std::sort(pool.begin(), pool.end());
      auto &scratch = thread_pq_buf;
      std::vector<float> occlude_factor(pool.size(), 0.0f);
      std::set<Neighbor> result_set;  // deduplication, and keep distance sorted.
      std::vector<uint32_t> idxs(pool.size()), ids(pool.size());
      std::vector<float> dijks(pool.size());
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
          // TODO : 仿照delta_prune_neighbors_pq，复用ids
          idxs.clear();
          ids.clear();
          for (uint32_t t = start + 1; t < pool.size() && t < maxc; t++) {
              if (occlude_factor[t] > alpha) continue;
              // #ifndef DONT_USE_OPT2
              if (nhood_id_map[p.id] == nhood_id_map[pool[t].id]) continue; // same group, skip
              // #endif
              idxs.push_back(t);
              ids.push_back(pool[t].id);
          }

          compute_pq_dists(p.id, ids.data(), dijks.data(), ids.size(), scratch);

          for (uint32_t i = 0; i < idxs.size(); i ++) {
              auto t = idxs[i];
              occlude_factor[t] = (std::max)(occlude_factor[t], pool[t].distance / dijks[i]);
          }
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
        if (z_id != this->medoids[0]) {
          disk_in_graph->del_edge(z_id, x_id);
          // std::cerr << "del " << z_id << " " << x_id << " *\n";
        }
      }
      for (auto &z_id : diff_b) {
        if (z_id != this->medoids[0]) {
          disk_in_graph->add_edge(z_id, x_id);
          // std::cerr << "add " << z_id << " " << x_id << "\n";
        }
      }

      reverse_graph_time_2 += timer.elapsed();
      #ifdef COLLECT_INFO
      if (gs!= nullptr) {
        gs->reverse_graph_op_time.fetch_add(timer.elapsed());
      }
      #endif

      if (gs != nullptr) {
        gs->in_graph_count += diff_a.size() + diff_b.size();
      }

      // write neighbors.
      node.nnbrs = nhood.size();
      *(node.nbrs - 1) = node.nnbrs;
      memcpy(node.nbrs, nhood.data(), node.nnbrs * sizeof(uint32_t));
    };
    
    // 3、读取需要回收的空间上的非删除点
    char * buf2;
    alloc_aligned((void **) &buf2, (n_sectors - new_n_sectors + 1) * SECTOR_LEN, SECTOR_LEN); // TODO : 减少memory的使用

    std::vector<IORequest> read_reqs(1);
    read_reqs[0].offset = (new_n_sectors + loc_sector_no(0) - 1) * SECTOR_LEN;   
    read_reqs[0].len = (n_sectors - new_n_sectors + 1) * SECTOR_LEN;
    read_reqs[0].buf = buf2;

    delete_io_timer.reset();
    reader->read(read_reqs, ctx, false);
    delete_io += delete_io_timer.elapsed();
    if (gs != nullptr) {
      gs->delete_io += delete_io_timer.elapsed();
      gs->delete_ios += read_reqs[0].len;
      gs->update_ios += read_reqs[0].len;
    }


    std::cerr << "loc begin form " << new_n_sectors * nnodes_per_sector << " to " << cur_loc.load() - 1 << "\n";
    std::cerr << "affected page begin form " << 1 << " to " << new_n_sectors + loc_sector_no(0) << "\n";
    std::vector<DiskNode<T>> free_nodes;
    if (unlikely(new_n_sectors <= 0)) {
      LOG(ERROR) << "new_n_sectors <= 0";
      exit(-1);
    }
    #pragma omp parallel for num_threads(nthreads) schedule(dynamic, 128)
    for (uint64_t loc = (new_n_sectors - 1) * nnodes_per_sector; loc < cur_loc.load(); loc++) {
      if (loc < new_npoints.load()) {
        continue;
      }
      uint64_t id = loc2id(loc);
      if (id == kInvalidID) { // 空位
        continue;
      }
      TagT tag = id2tag(id);
      if (deleted_nodes_set.find(tag) == deleted_nodes_set.end()) {  // 非删除点
        uint64_t pid = loc_sector_no(loc);
        if (unlikely(pid - new_n_sectors < 0)) {
          LOG(ERROR) << "loc_sector_no(loc) < new_n_sectors";
          exit(-1);
        }
        if (unlikely(pid - new_n_sectors >= n_sectors - new_n_sectors + 1)) {
          LOG(ERROR) << "loc_sector_no(loc) - new_n_sectors >= n_sectors - new_n_sectors";
          exit(-1);
        }
        char * sector_buf = buf2 + (pid - new_n_sectors) * SECTOR_LEN;
        char * node_buf = offset_to_loc(sector_buf, loc);
        
        DiskNode<T> node(id, offset_to_node_coords(node_buf), offset_to_node_nhood(node_buf));
        if (affected_ids.find(id) != affected_ids.end()) { // 受影响点
          auto &thread_pq_buf = thread_pq_bufs[omp_get_thread_num()];
          consolidate_deletes(node, thread_pq_buf);
        }
        {
          std::unique_lock<std::shared_mutex> lock(mu1);
          free_nodes.push_back(node);
        }
      }
    }

    LOG(INFO) << "affected_ids size: " << affected_ids.size();
    LOG(INFO) << "affected_pages size: " << affected_pages.size();
    LOG(INFO) << "free_nodes size: " << free_nodes.size();
    LOG(INFO) << "3. Collect free nodes. Prune neighbors begin, totally elapsed: " << delete_timer.elapsed() / 1e3 << "ms";
    LOG(INFO) << "Delete io cost " << delete_io / 1e6 << " s.";

    // 4、开始处理受影响点，同时回收空间
    // TODO : pq空间没有回收，pq的使用只用到了逻辑id，没有用到loc；反向图的空间也没有回收。
    // 可以调整
    while(affected_pages.size() > 0 && affected_pages.back() > new_n_sectors){
      affected_pages.pop_back();
    }

    std::atomic<uint64_t> free_node_idx {0};
    const size_t BUF3_TOTAL_SIZE = (affected_pages.size() ) * SECTOR_LEN;
    char * buf3;
    alloc_aligned((void **) &buf3, BUF3_TOTAL_SIZE, SECTOR_LEN); // TODO : 减少memory的使用

    std::vector<IORequest> reqs;
    {
      buf_map.clear();
      std::atomic<uint64_t> buf_idx {0};
      // #pragma omp parallel for num_threads(nthreads) schedule(dynamic, 1024)
      for(auto pid : affected_pages){ // perf: add omp for deep1b
        auto idx = buf_idx.fetch_add(1);
        char * sector_buf = buf3 + idx * SECTOR_LEN;
        {
          // std::unique_lock<std::shared_mutex> lock(mu1);
          reqs.emplace_back(IORequest(pid * SECTOR_LEN, SECTOR_LEN, sector_buf, 0, 0));
        }
        buf_map.insert(pid, sector_buf);
      }

      LOG(INFO) << "3.1 totally elapsed: " << delete_timer.elapsed() / 1e3 << "ms";
      
      delete_io_timer.reset();
      reader->read(reqs, ctx, false);
      delete_io += delete_io_timer.elapsed();
      if (gs != nullptr) {
        gs->delete_io += delete_io_timer.elapsed();
        gs->update_ios += reqs.size() * SECTOR_LEN;
        gs->delete_ios += reqs.size() * SECTOR_LEN;
      }
    }

    std::cerr << "affected_pages size: " << affected_pages.size() << "\n";
    std::cerr << "reqs size: " << reqs.size() << "\n";
    LOG(INFO) << "3.2 totally elapsed: " << delete_timer.elapsed() / 1e3 << "ms";
    
    #pragma omp parallel for num_threads(nthreads) schedule(dynamic, 1024)
    for (uint64_t i = 0; i < affected_pages.size(); i++) {
      uint64_t pid = affected_pages[i];

      char * sectro_buf = buf_map.find(pid);
      for (uint64_t j = 0; j < nnodes_per_sector; j ++) {
        
        uint64_t loc = sector_to_loc(pid, j);
        if (loc >= new_npoints.load()) continue;
        uint32_t id = loc2id(loc);
        
        char * node_buf = offset_to_loc(sectro_buf, loc);
        DiskNode<T> node(id, offset_to_node_coords(node_buf), offset_to_node_nhood(node_buf));
        
        if (id == kInvalidID || deleted_nodes_set.find(id2tag(id)) != deleted_nodes_set.end()) { 
          // 空位 或者是删除点，拿一个新点填充
          
          uint64_t idx = free_node_idx.fetch_add(1);
          if (unlikely(idx >= free_nodes.size())) {
            LOG(INFO) << "Free nodes empty!";
            continue;
          }

          auto &free_node = free_nodes[idx];
          memcpy(node.coords, free_node.coords, data_dim * sizeof(T));
          *(node.nbrs - 1) = free_node.nnbrs;
          memcpy(node.nbrs, free_node.nbrs, free_node.nnbrs * sizeof(uint32_t));

          erase_loc2id(id2loc(free_node.id));
          id2loc_.insert_or_assign(free_node.id, loc);
          set_loc2id(loc, free_node.id);
          
        } else if (affected_ids.find(id) != affected_ids.end()) { 
          auto &thread_pq_buf = thread_pq_bufs[omp_get_thread_num()];

          consolidate_deletes(node, thread_pq_buf);
        }
      }
    }
    LOG(INFO) << "3.3 totally elapsed: " << delete_timer.elapsed() / 1e3 << "ms";

    delete_io_timer.reset();
    reader->write(reqs, ctx, false);  // write back
    delete_io += delete_io_timer.elapsed();
    if (gs != nullptr) {
      gs->delete_io += delete_io_timer.elapsed();
      gs->delete_ios += reqs.size() * SECTOR_LEN;
      gs->update_ios += reqs.size() * SECTOR_LEN;
    }

    LOG(INFO) << "reverse_graph_time_2 " << reverse_graph_time_2 / nthreads /  1e6 << " s.";
    LOG(INFO) << "Write nhoods finished, totally elapsed " << delete_timer.elapsed() / 1e3 << "ms.";
    LOG(INFO) << "Delete io cost " << delete_io / 1e6 << " s.";

    if (gs != nullptr) {
      std::cerr << "in_graph_count: " << gs->in_graph_count << std::endl;
      gs->delete_ios += gs->in_graph_count * SECTOR_LEN;
      gs->update_ios += gs->in_graph_count * SECTOR_LEN;
    }
    
    uint32_t medoid = this->medoids[0];
    while (deleted_nodes_set.find(id2tag(medoid)) != deleted_nodes_set.end()) {
      LOG(INFO) << "Medoid deleted. Choosing another start node.";
      const auto &[nhoods, _] = deleted_nhoods.find(medoid);
      medoid = nhoods[0];
    }

    this->num_points = new_npoints.load();
    this->cur_loc = new_npoints.load();
    if (this->num_points % nnodes_per_sector != 0) {
      this->cur_loc += nnodes_per_sector - (num_points % nnodes_per_sector);
    }

    // free bufs
    aligned_free((void *) buf2);  
    aligned_free((void *) buf3);

    // set metadata, PQ and tags.
    // merge_lock.lock();  // unlock in reload().

    for (auto id : deleted_ids) {
      tags.erase(id);
      id2loc_.erase(id);
    }
    for (uint64_t pid = new_n_sectors + 1; pid < n_sectors; pid ++) { // TODO : 删除文件末尾的点
      page_layout.erase(pid);
    }
    for (uint32_t i = 0; i < nnodes_per_sector; i ++) {
      erase_loc2id(cur_loc.load() + i);
    }
    
    // metadata.
    this->medoids[0] = medoid;
    LOG(INFO) << "tags size: " << tags.size() << " id2loc size: " << id2loc_.size() << " page_layout size: " << page_layout.size();
    LOG(INFO) << "new_npoints: " << new_npoints.load() << " cur_loc: " << cur_loc.load() << " num_points: " << num_points;
    LOG(INFO) << "Medoid: " << medoid;
    // PQ.
    // tags.

    while (!this->empty_pages.empty()) {
      this->empty_pages.pop();
    }

    LOG(INFO) << "Write metadata finished, totally elapsed " << delete_timer.elapsed() / 1e3 << "ms.";
    std::cerr << "$E2E delete cost " << delete_timer.elapsed() / 1e6 << " s.\n";
    
  }

  template class SSDIndex<float>;
  template class SSDIndex<_s8>;
  template class SSDIndex<_u8>;
}  // namespace dirann
