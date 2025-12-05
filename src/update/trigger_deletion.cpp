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
  void SSDIndex<T, TagT>::trigger_deletion_in_place(const tsl::robin_set<TagT> &deleted_nodes_set, uint32_t nthreads,
                                        const uint32_t &n_sampled_nbrs) {
    if (nthreads == 0) {
      nthreads = this->max_nthreads;
    }

    void *ctx = reader->get_ctx();

    // Note that the index is immutable currently.
    // Step 1: populate neighborhoods, allocate IDs.
    libcuckoo::cuckoohash_map<uint32_t, std::vector<uint32_t>> deleted_nhoods;  // id -> nhood
    std::vector<uint32_t> deleted_ids;  // log deleted ids
    Timer delete_timer;
    
    auto p_reader = (LinuxAlignedFileReader *)this->reader.get();
    int strategy = p_reader->strategy;
    FileHandle coord_fd = p_reader->coord_file_desc;
    FileHandle topo_fd = p_reader->topo_file_desc;
    auto & block_cache = p_reader->block_cache;
    int use_topo_reorder = (strategy >> 1) & 0x1;
    int use_coord_reorder = (strategy >> 3) & 0x1;
    int use_topo_buffer = (strategy >> 4) & 0x1;

    LOG(INFO) << "Merge thread: " << nthreads;

    char *buf = nullptr, *wbuf = nullptr;
    alloc_aligned((void **) &buf, SECTORS_PER_MERGE * SECTOR_LEN, SECTOR_LEN);//搞一个双缓冲区
    
#ifdef USE_TOPO_DISK
    uint64_t n_sectors = (cur_loc_topo + ntopo_per_sector - 1) / ntopo_per_sector;
#else
    uint64_t n_sectors = (cur_loc + nnodes_per_sector - 1) / nnodes_per_sector;
#endif
    LOG(INFO) << "Cur loc: " << cur_loc.load() << " Cur topo loc: " << cur_loc_topo.load() 
              << " Cur coord loc: " << cur_loc_coord.load() 
              << ", cur ID: " << cur_id << ", n_sectors: " << n_sectors
              << ", nnodes_per_sector: " << nnodes_per_sector << ", ntopo_per_sector: " << ntopo_per_sector;
    LOG(INFO) << "Tags size: " << tags.size() << " layout size: " << page_layout.size() << " id2loc_ size: " << id2loc_.size();

    Timer delete_io_timer;
    double delete_io = 0;
    for (uint64_t in_sector = 0; in_sector < n_sectors; in_sector += SECTORS_PER_MERGE) {
      uint64_t st_sector = in_sector, ed_sector = std::min(in_sector + SECTORS_PER_MERGE, n_sectors);
#ifdef USE_TOPO_DISK
      uint64_t loc_st = st_sector * ntopo_per_sector, loc_ed = std::min(cur_loc_topo.load(), ed_sector * ntopo_per_sector);
#else
      uint64_t loc_st = st_sector * nnodes_per_sector, loc_ed = std::min(cur_loc.load(), ed_sector * nnodes_per_sector);
#endif
      uint64_t n_sectors_to_read = ed_sector - st_sector;
      std::vector<IORequest> read_reqs;
      tsl::robin_set<uint64_t> page_set;
      std::vector<std::pair<uint32_t, unsigned *>> deleted_ids_buf;

#pragma omp parallel for num_threads(nthreads) schedule(dynamic, 1024)
      for (uint64_t loc = loc_st; loc < loc_ed; ++loc) {
        // populate nhood.
#ifdef USE_TOPO_DISK
        uint64_t id = loc2id_topo(loc);
#else
        uint64_t id = loc2id(loc);
#endif
        if (id == kInvalidID) {
          continue;
        }

        uint64_t tag = id2tag(id);
        if (deleted_nodes_set.find(tag) == deleted_nodes_set.end()) {  // 2. not deleted, alloc ID.
          // allocate ID.
          continue;
        }

#ifdef USE_TOPO_DISK
        char * page_buf = buf + (loc / ntopo_per_sector - st_sector) * SECTOR_LEN;
        uint64_t pid = loc_sector_no_topo(loc);

        #pragma omp critical
        deleted_ids.push_back(id);
        #pragma omp critical
        deleted_ids_buf.emplace_back(id, (unsigned *)offset_to_loc_topo(page_buf, loc));

#else
        char * page_buf = buf + (loc / nnodes_per_sector - st_sector) * SECTOR_LEN;
        uint64_t pid = loc_sector_no(loc);

        #pragma omp critical
        deleted_ids.push_back(id);
        #pragma omp critical
        deleted_ids_buf.emplace_back(id, (unsigned *)offset_to_loc(page_buf, loc));

#endif
        #pragma omp critical
        if (page_set.find(pid) == page_set.end()) {
          read_reqs.push_back(IORequest(pid * SECTOR_LEN, SECTOR_LEN, page_buf, 0, 0));
          page_set.insert(pid);
        }
      }

      delete_io_timer.reset();
#ifdef USE_TOPO_DISK
      reader->read_fd(topo_fd, read_reqs, ctx);
#else
      reader->read(read_reqs, ctx, false);
#endif
      delete_io += delete_io_timer.elapsed();

#pragma omp parallel for num_threads(nthreads) schedule(dynamic, 1024)
      for (size_t i = 0; i < deleted_ids_buf.size(); ++i) {
        auto [id, nhood_buf] = deleted_ids_buf[i];
        
#ifdef USE_TOPO_DISK
        DiskNode<T> node(id, (unsigned *)(nhood_buf));
#else
        DiskNode<T> node(id, offset_to_node_coords((char *)nhood_buf), offset_to_node_nhood((char *)nhood_buf));
#endif
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
              << "ms";
              // , new npoints: " << new_npoints.load() << " " << "id_map size: " << id_map.size();

    // Step 2: prune neighbors, populate PQ and tags.
    // std::atomic<uint64_t> n_used_id = 0;

    // std::cerr << "deleted_nhoods size: " << deleted_nhoods.size() << std::endl;
    // std::cerr << "deleted_ids size: " << deleted_ids.size() << std::endl;
    
    if(0)
    {
      std::string filename = "disk_deleted_nhoods_test_ssd.txt";
      std::ofstream out(filename);
      if (!out) {
          throw std::runtime_error("Failed to open file for writing: " + filename);
      }

      auto lt = deleted_nhoods.lock_table();
      for (const auto& [key, vec] : lt) {
          out << key << " " << vec.size() << " ";
          for (uint32_t val : vec) {
              out << ' ' << val;
          }
          out << '\n';  // 每个key对应一行
      }

      out.close();
      std::cout << "success write " << filename << std::endl;
      exit(0);
    }
    LOG(INFO) << "Prune neighbors begin, totally elapsed: " << delete_timer.elapsed() / 1e3 << "ms";

    std::atomic<int> affected_cnt {0};
    for (uint64_t in_sector = 0; in_sector < n_sectors; in_sector += SECTORS_PER_MERGE) {
      uint64_t st_sector = in_sector, ed_sector = std::min(in_sector + SECTORS_PER_MERGE, n_sectors);
#ifdef USE_TOPO_DISK
      uint64_t loc_st = st_sector * ntopo_per_sector, loc_ed = std::min(cur_loc_topo.load(), ed_sector * ntopo_per_sector);
#else
      uint64_t loc_st = st_sector * nnodes_per_sector, loc_ed = std::min(cur_loc.load(), ed_sector * nnodes_per_sector);
#endif
      uint64_t n_sectors_to_read = ed_sector - st_sector;
      std::vector<IORequest> read_reqs;
#ifdef USE_TOPO_DISK
      read_reqs.push_back(IORequest(loc_sector_no_topo(loc_st) * SECTOR_LEN, n_sectors_to_read * size_per_io, buf, 0, 0));
      delete_io_timer.reset();
      reader->read_fd(topo_fd, read_reqs, ctx);  // read in fd
#else
      read_reqs.push_back(IORequest(loc_sector_no(loc_st) * SECTOR_LEN, n_sectors_to_read * size_per_io, buf, 0, 0));
      delete_io_timer.reset();
      reader->read(read_reqs, ctx, false);  // read in fd
#endif

      delete_io += delete_io_timer.elapsed();
      
      if (gs != nullptr) {
        gs->delete_io += delete_io_timer.elapsed();
        gs->update_ios += read_reqs[0].len;
      }
      // std::cerr << loc_st << " , " << loc_ed << std::endl;
      
      // Timer tmp;
#pragma omp parallel for num_threads(nthreads) schedule(dynamic, 1024)
      for (uint64_t loc = loc_st; loc < loc_ed; ++loc) {
      // Timer t;
      try{
        // Timer scan_timer;
#ifdef USE_TOPO_DISK
        uint64_t id = loc2id_topo(loc);
#else
        uint64_t id = loc2id(loc);
#endif
        if (id == kInvalidID) {
          // if (gs != nullptr) {
          //   gs->delete_scan1 += scan_timer.elapsed();
          // }
          continue;
        }
        
        uint64_t tag = id2tag(id);

#ifdef USE_TOPO_DISK
        auto page_buf = buf + (loc / ntopo_per_sector - st_sector) * SECTOR_LEN;
        auto loc_buf = offset_to_loc_topo(page_buf, loc);
        DiskNode<T> node(id, (unsigned *)(loc_buf));
#else
        auto page_buf = buf + (loc / nnodes_per_sector - st_sector) * SECTOR_LEN;
        auto loc_buf = offset_to_loc(page_buf, loc);
        DiskNode<T> node(id, offset_to_node_nhood(loc_buf));
#endif

        if (deleted_nodes_set.find(tag) != deleted_nodes_set.end()) {  // deleted.
          node.nnbrs = 0;
          *(node.nbrs - 1) = 0;
#ifdef USE_NHOOD_CACHE
          if(nhood_cache.find(id) != nhood_cache.end()){
            *nhood_cache[id] = 0;
          }
#endif
          // if (gs != nullptr) {
          //   gs->delete_scan1 += scan_timer.elapsed();
          // }

          continue;
        }
        
        // prune neighbors.
        std::unordered_set<uint32_t> nhood_set;
        bool change = false;
        for (uint32_t i = 0; i < node.nnbrs; ++i) {
          // std::cerr << "3\n";
          
          uint32_t nbr_tag;
          nbr_tag = id2tag(node.nbrs[i]);
          if (deleted_nodes_set.find(nbr_tag) != deleted_nodes_set.end()) {
            // deleted, insert neighbors.
            const auto &nhoods = deleted_nhoods.find(node.nbrs[i]);
            nhood_set.insert(nhoods.begin(), nhoods.end());
            change = true;
          } else {
            nhood_set.insert(node.nbrs[i]);
            // LOG(INFO) << id << " insert " << node.nbrs[i];
          }
        }
        // if (gs != nullptr) {
        //   gs->delete_scan1 += scan_timer.elapsed();
        // }

        if(!change) continue;
        // Timer prune_timer;

        affected_cnt ++;
        nhood_set.erase(id);  // remove self.
        std::vector<uint32_t> nhood(nhood_set.begin(), nhood_set.end());

        if (nhood.size() > this->range) {
          std::vector<float> dists(nhood.size(), 0.0f);
          std::vector<Neighbor> pool(nhood.size());
          auto &thread_pq_buf = thread_pq_bufs[omp_get_thread_num()];
          compute_pq_dists(id, nhood.data(), dists.data(), (_u32) nhood.size(), thread_pq_buf);

          for (uint32_t k = 0; k < nhood.size(); k++) {
            pool[k].id = nhood[k];
            pool[k].distance = dists[k];
          }
          std::sort(pool.begin(), pool.end());
          if (pool.size() > this->maxc) {
            pool.resize(this->maxc);
          }
          nhood.clear();
          // prune_cnt ++;
          this->prune_neighbors_pq(pool, nhood, thread_pq_buf);
        // std::cerr << "shouyingx id is" << id << std::endl;
        //   for(auto x : nhood){
        //     std::cerr << x << " ";
        //   }
        //   std::cerr << "\n";
        }

        // write neighbors.
        node.nnbrs = nhood.size();
        *(node.nbrs - 1) = node.nnbrs;
        memcpy(node.nbrs, nhood.data(), node.nnbrs * sizeof(uint32_t));
        
#ifdef USE_NHOOD_CACHE
        if(nhood_cache.find(id) != nhood_cache.end()){
          memcpy(nhood_cache[id], node.nbrs - 1, (node.nnbrs + 1) * sizeof(uint32_t));
        }
#endif
        // if (gs != nullptr){
        //   gs->occlude_list_pq += prune_timer.elapsed();
        // }
      }     
      catch (...) {
        LOG(ERROR) << "Unknown exception!";
        // exit(-1);
      }
      // if (gs != nullptr) {
      //   #pragma omp critical
      //   gs->consolidate_deletes += t.elapsed();
      // }
      }
      
      // LOG(INFO) << "tmp timer " << tmp.elapsed() / 1e6 << " s.";
      // sum_time += tmp.elapsed();
      LOG(INFO) << "Processed " << ed_sector << "/" << n_sectors << " sectors.";
      delete_io_timer.reset();
//       //TODO : double buffer
#ifdef USE_TOPO_DISK  
      reader->write_fd(topo_fd, read_reqs, ctx);  // write back
#else
      //TODO : double buffer
      reader->write(read_reqs, ctx, false);  // write back
#endif
      delete_io += delete_io_timer.elapsed();
      
      if (gs != nullptr) {
        gs->delete_io += delete_io_timer.elapsed();
        gs->update_ios += read_reqs[0].len;
      }
    }

    LOG(ERROR) << "Affected id count " << affected_cnt;
    LOG(INFO) << "Write nhoods finished, totally elapsed " << delete_timer.elapsed() / 1e3 << "ms.";
    LOG(INFO) << "Delete io cost " << delete_io / 1e6 << " s.";

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
#ifdef USE_TOPO_DISK
      // erase_loc2id_topo(id2loc_topo(id));
      // erase_loc2id_coord(id2loc_coord(id));
#else
      erase_loc2id(id2loc(id));
      id2loc_.erase(id);
#endif
    }
    
    
    if(use_topo_buffer){
      if(block_cache == nullptr){
        LOG(ERROR) << "block_cache is nullptr";
        exit(-1);
      }
      block_cache->clear();

    }
    if(use_topo_reorder){
      while(empty_locs_topo.pop() != kInvalidID);
    } else {
      while(empty_pages_topo.pop() != kInvalidID);
    }

    if(use_coord_reorder){
      while(empty_locs_coord.pop() != kInvalidID);
    }else {
      while(empty_pages_coord.pop() != kInvalidID);
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
