#pragma once

#include <vector>
#include <queue>
#include <mutex>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <fstream>
#include <sstream>
#include <thread>
#include <atomic>
#include <limits>
#include <fcntl.h>
#include <sys/mman.h>
#include <stdexcept>
#include <algorithm>
#include <utility>

namespace DynGraphDisk
{

using edge_t = int32_t;
using vertex_t = int32_t;
using offset_t = long long int;

template <typename T>
constexpr T round_up_to_next_power_of_two_constexpr(T x) {
    return x == 0 ? 1 : (T(1) << (sizeof(T) * 8 - __builtin_clz(x)));
}


struct Block {
  struct Head {
    offset_t next_offset_ = 0;
    size_t max_edge_num_ = 0;
    int nbr_num_ = 0;
    int last_value_id_ = 0;
    edge_t min_value_ = std::numeric_limits<edge_t>::max();
    edge_t max_value_ = 0;
  };

  
  char* data_ = nullptr; 
  Head* head_ = nullptr;

  
  Block(char* ptr, const int max_edge_num, const offset_t next_offset) 
    : data_(ptr) {
    head_ = reinterpret_cast<Head*>(data_);
    set_max_edge_num(max_edge_num);
    set_next_offset(next_offset);
    set_nbr_num(0);
    set_last_value_id_(0);
    set_min_value_(std::numeric_limits<edge_t>::max());
    set_max_value_(0);
    
    for (int i = 0; i < max_edge_num; ++i) {
      get_edge_data()[i] = std::numeric_limits<edge_t>::max();
    }
  }

  
  Block(char* ptr) : data_(ptr) {
    head_ = reinterpret_cast<Head*>(data_);
  }

  void set_max_edge_num(const int max_edge_num) {
    head_->max_edge_num_ = max_edge_num;
  }

  void set_next_offset(const offset_t next_offset) {
    head_->next_offset_ = next_offset;
  }

  void set_nbr_num(const int nbr_num_) {
    head_->nbr_num_ = nbr_num_;
  }

  void set_min_value_(const int min_value_) {
    head_->min_value_ = min_value_;
  }

  void set_max_value_(const edge_t max_value) {
    head_->max_value_ = max_value;
  }

  void set_last_value_id_(const int loc) {
    head_->last_value_id_ = loc;
  }

  int get_max_edge_num() {
    return head_->max_edge_num_;
  }

  offset_t get_next_offset() {
    return head_->next_offset_;
  }

  int get_nbr_num() {
    return head_->nbr_num_;
  }

  edge_t get_min_value() {
    return head_->min_value_;
  }

  edge_t get_max_value() {
    return head_->max_value_;
  }

  edge_t get_last_value_id() {
    return head_->last_value_id_;
  }

  edge_t* get_edge_data() {
    return reinterpret_cast<edge_t*>(data_ + sizeof(Head));
  }

  char* get_data() {
    return data_;
  }

  void print() {
    std::cout << "print vertex info:" << std::endl;
    std::cout << "  max_edge_num=" << get_max_edge_num() 
              << ", next_offset=" << get_next_offset()
              << ", nbr_num=" << get_nbr_num()
              << ", min_value=" << get_min_value()
              << ", max_value=" << get_max_value()
              << ", last_value_id=" << get_last_value_id()
              << std::endl;
    for (int i = 0; i < get_max_edge_num(); ++i) {
      std::cout << "  " << get_edge_data()[i];
    }
    std::cout << std::endl;
  }

};

class BlockUpdater {
public:
  BlockUpdater() {}

  void set_next_offset(Block* block, const offset_t next_offset) {
    block->set_next_offset(next_offset);
  }

  void add_edge(Block* block, edge_t dst) {
    if (block->get_max_edge_num() <= block->get_nbr_num()) {
      std::cout << "Error: block is full, get_max_edge_num=" 
                << block->get_max_edge_num() << std::endl;
      exit(0);
    }
    edge_t* nbrs = block->get_edge_data();
    nbrs[block->get_nbr_num()] = dst;
    block->set_nbr_num(block->get_nbr_num() + 1);
    if (dst < block->get_min_value()) {
      block->set_min_value_(dst);
    }
    if (dst > block->get_max_value()) {
      block->set_max_value_(dst);
    }
  }

  bool del_edge(Block* block, edge_t dst) {
  
    edge_t* nbrs = block->get_edge_data();
    for (int i = 0; i < block->get_nbr_num(); ++i) {
      if (nbrs[i] == dst) {
        
        if (i != block->get_nbr_num() - 1) {
          nbrs[i] = nbrs[block->get_nbr_num()-1];
          nbrs[block->get_nbr_num()-1] = set_deleted(nbrs[block->get_nbr_num()-1]);
        } else {
          nbrs[i] = set_deleted(dst);
        }
        block->set_nbr_num(block->get_nbr_num() - 1);

        if (dst == block->get_min_value()) {
          edge_t min = std::numeric_limits<edge_t>::max();
          for (int j = 0; j < block->get_nbr_num(); ++j) {
            if (is_deleted(nbrs[j]) == false) {
              min = std::min(min, nbrs[j]);
            }
          }
          block->set_min_value_(min);
        }
        if (dst == block->get_max_value()) {
          edge_t max = 0;
          for (int j = 0; j < block->get_nbr_num(); ++j) {
            if (is_deleted(nbrs[j]) == false) {
              max = std::max(max, nbrs[j]);
            }
          }
          block->set_max_value_(max);
        }
        
        
        return true;
      }
    }
    return false;
  }

  void get_edges(Block* block, std::vector<edge_t> &edge) {
    if (block->get_nbr_num() == 0) {
      return;
    }
    edge_t* nbrs = block->get_edge_data();
    for (int i = 0; i < block->get_nbr_num(); ++i) {
      if (is_deleted(nbrs[i]) == false) {
        edge.push_back(nbrs[i]);
      }
    }
  }

  
  bool is_deleted(edge_t dst) {
    
    
    return dst == std::numeric_limits<edge_t>::max();
  }

  
  edge_t set_deleted(edge_t val) {
    
    return std::numeric_limits<edge_t>::max();
  }

};

class BlockManager {
public:
    constexpr static uintptr_t NULLPOINTER = 0; 

    
    BlockManager(std::string path, bool load_old_data, size_t _capacity = 1ul << 40)
        : capacity(_capacity),
          mutex(),
          load_old_data_(load_old_data),
          path_(path),
          file_size(0),
          used_size(0),
          fd(EMPTY_FD)
    {
        if (path.empty())
        {
            fd = EMPTY_FD;
            data =
                mmap(nullptr, capacity, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
            if (data == MAP_FAILED)
                throw std::runtime_error("mmap block error.");
        }
        else
        {
            std::cout << "\nopen file path=" << path << std::endl;
            std::cout << "  _capacity=" << _capacity << std::endl;
            size_t FILE_TRUNC_SIZE = _capacity;
            std::cout << "  FILE_TRUNC_SIZE=" << FILE_TRUNC_SIZE << std::endl;

            if (load_old_data_ == true) {
              load_meta();
            }

            if (load_old_data_ == false || file_size == 0) {
              std::cout << "  clear file" << std::endl;
              fd = open(path.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0640); 
            } else {
              std::cout << "  read old file" << std::endl;
              
              fd = open(path.c_str(), O_RDWR | O_CREAT, 0640);  
            }
            if (fd == EMPTY_FD)
                throw std::runtime_error("open block file error.");

            if (file_size == 0) {
              if (ftruncate(fd, FILE_TRUNC_SIZE) != 0)
                    throw std::runtime_error("ftruncate block file error.");
              file_size = FILE_TRUNC_SIZE;
            } else {
              if (ftruncate(fd, file_size) != 0)
                    throw std::runtime_error("ftruncate block file error.");
            }

            data = mmap(nullptr, capacity, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);

            if (data == MAP_FAILED)
                throw std::runtime_error("mmap block error.");
        }

        if (madvise(data, capacity, MADV_RANDOM) != 0)
            throw std::runtime_error("madvise block error.");
    }

    void save_meta() {
      
      std::cout << "wirte to file " << (path_+".mmap.meta") << std::endl;
      
      std::ofstream outfile(path_+".mmap.meta", std::ios::binary);
      if (!outfile) {
        std::cerr << "Error: cannot open file " << path_ << std::endl;
        exit(1);
      }
  
      
      outfile.write(reinterpret_cast<const char*>(&used_size), sizeof(used_size));
      outfile.write(reinterpret_cast<const char*>(&file_size), sizeof(file_size));
      std::cout << "  used_size: " << used_size << ", file_size: " << file_size << std::endl;
      size_t free_block_count = free_blocks.size();
      outfile.write(reinterpret_cast<const char*>(&free_block_count), sizeof(free_block_count));
      std::cout << "  free_block_count: " << free_block_count << std::endl;
      for (auto& pair : free_blocks) {
        int block_size = pair.first;
        outfile.write(reinterpret_cast<const char*>(&block_size), sizeof(block_size));
        size_t block_count = pair.second.size();
        outfile.write(reinterpret_cast<const char*>(&block_count), sizeof(block_count));
        
        for (auto& block : pair.second) {
            outfile.write(reinterpret_cast<const char*>(&block), sizeof(block));
        }
      }
    }

    void load_meta() {
      std::cout << "  read to file " << (path_+".mmap.meta") << std::endl;
      std::ifstream infile(path_+".mmap.meta", std::ios::binary);
      if (!infile) {
        std::cout << "Note: cannot open file " << (path_+".mmap.meta") << std::endl;
        
        return;
      }

      
      infile.read(reinterpret_cast<char*>(&used_size), sizeof(used_size));
      infile.read(reinterpret_cast<char*>(&file_size), sizeof(file_size));
      std::cout << "  used_size: " << used_size << ", file_size: " << file_size << std::endl;
      size_t free_block_count;
      infile.read(reinterpret_cast<char*>(&free_block_count), sizeof(free_block_count));
      std::cout << "  free_block_count: " << free_block_count << std::endl;
      for (size_t i = 0; i < free_block_count; ++i) {
        int block_size;
        infile.read(reinterpret_cast<char*>(&block_size), sizeof(block_size));
        size_t block_count;
        infile.read(reinterpret_cast<char*>(&block_count), sizeof(block_count));
        
        for (size_t j = 0; j < block_count; ++j) {
          uintptr_t block;
          infile.read(reinterpret_cast<char*>(&block), sizeof(block));
          free_blocks[block_size].push_back(block);
        }
      }
    }

    ~BlockManager()
    {
        std::cout << "  used_size: " << used_size 
                  << ", file_size: " << file_size << std::endl;
        // if (load_old_data_ == true) {
          save_meta();
        // }
        msync(data, capacity, MS_SYNC);
        munmap(data, capacity);
        if (fd != EMPTY_FD)
            close(fd);
    }

    char * alloc(size_t block_size)
    {
        uintptr_t pointer = NULLPOINTER;
        {
            std::lock_guard<std::mutex> lock(mutex);
            pointer = pop(block_size);
        }

        if (pointer == NULLPOINTER) {
            pointer = used_size.fetch_add(block_size);

            if (pointer + block_size >= file_size) {
                auto new_file_size = ((pointer + block_size) / FILE_TRUNC_SIZE + 1) * FILE_TRUNC_SIZE;
                std::lock_guard<std::mutex> lock(mutex);
                if (pointer + block_size < file_size) {
                  std::cout << "  return, file_size=" << file_size << std::endl;
                  return reinterpret_cast<char *>(data) + pointer;
                }
                if (new_file_size >= file_size) {
                    std::cout << "  resize file new_file_size=" << new_file_size << std::endl;
                    if (fd != EMPTY_FD) {
                        if (ftruncate(fd, new_file_size) != 0)
                            throw std::runtime_error("ftruncate block file error.");
                    }
                    file_size = new_file_size;
                }
            }
        }

        return reinterpret_cast<char *>(data) + pointer;
    }

    void free(uintptr_t block, int order) {
        {
            std::lock_guard<std::mutex> lock(mutex);
            push(order, block);
        }
    }

    offset_t convert_to_offset(char* ptr) {
        return ptr - reinterpret_cast<char *>(data);
    }

    char* convert_to_prt(offset_t offset) {
        return reinterpret_cast<char *>(data) + offset;
    }

private:
    const size_t capacity;
    int fd;
    void *data;
    std::mutex mutex;
    std::unordered_map<int, std::vector<uintptr_t>> free_blocks;
    std::atomic<size_t> used_size, file_size;
    uintptr_t null_holder;
    std::string path_;
    bool load_old_data_;

    uintptr_t pop(int order) {
        uintptr_t pointer = NULLPOINTER;
        if (free_blocks[order].size())
        {
            pointer = free_blocks[order].back();
            free_blocks[order].pop_back();
        }
        return pointer;
    }

    void push(int order, uintptr_t pointer) {
      free_blocks[order].push_back(pointer);
    }

    constexpr static int EMPTY_FD = -1;
    constexpr static int MAX_ORDER = 1024;
    constexpr static size_t FILE_TRUNC_SIZE = (1ul << 30) * 1; 
};

class BlockGraph {
public:
  BlockGraph(std::string path, size_t max_vertex, int edge_num_per_block, 
             int merge_block_num,
             int sample_num,
             bool load_old_data,
             size_t capacity = 1ul << 30) 
    : blockmanager(path, load_old_data, capacity),
      writelock_(max_vertex),
      edge_num_per_block_(edge_num_per_block),
      graph_(max_vertex, nullptr),
      merge_block_num_(merge_block_num),
      max_vertex_(max_vertex),
      sample_num_(sample_num),
      filepath_(path),
      load_old_data_(load_old_data),
      curr_max_vertex_id_(0) {
    std::cout << "Oursys BlockGraph constructed." << std::endl;
    std::cout << "  max_vertex: " << max_vertex << std::endl;
    std::cout << "  capacity: " << capacity << std::endl;
    std::cout << "  edge_num_per_block: " << edge_num_per_block << std::endl;
    std::cout << "  merge_block_num_: " << merge_block_num_ << std::endl;
    std::cout << "  sample_num_: " << sample_num_ << std::endl;

    std::cout << "  graph_.size(): " << graph_.size() << std::endl;

    if (load_old_data_ == true) {
      std::cout << "====Use old data, load old in_graph!===" << std::endl;
      load_meta(); 
    } else {
      std::cout << "===Not use old data, create new in_graph!===" << std::endl;
    }
  }

  
  void save_meta() {
    std::cout << "wirte to file " << (filepath_+".meta") << std::endl;
    
    std::ofstream outfile(filepath_+".meta", std::ios::binary);
    if (!outfile) {
      std::cerr << "Error: cannot open file " << filepath_ << std::endl;
      exit(1);
    }

    
    outfile.write(reinterpret_cast<const char*>(&max_vertex_), sizeof(max_vertex_));
    outfile.write(reinterpret_cast<const char*>(&curr_max_vertex_id_), sizeof(curr_max_vertex_id_));
    outfile.write(reinterpret_cast<const char*>(&sample_num_), sizeof(sample_num_));
    outfile.write(reinterpret_cast<const char*>(&edge_num_per_block_), sizeof(edge_num_per_block_));
    outfile.write(reinterpret_cast<const char*>(&merge_block_num_), sizeof(merge_block_num_));
    std::cout << "  max_vertex_:" << max_vertex_ << std::endl;
    std::cout << "  curr_max_vertex_id_:" << curr_max_vertex_id_ << std::endl;
    std::cout << "  sample_num_:" << sample_num_ << std::endl;
    std::cout << "  edge_num_per_block_:" << edge_num_per_block_ << std::endl;
    for (size_t i = 0; i < max_vertex_; ++i) {
      offset_t offset = -1;
      if (nullptr != graph_[i]) {
        auto block = *(graph_[i]);
        offset = blockmanager.convert_to_offset(block.get_data());
      }

      outfile.write(reinterpret_cast<const char*>(&offset), sizeof(offset));
    }

    
    outfile.close();
  }

  void load_meta() {
    
    std::cout << "read to file " << (filepath_+".meta") << std::endl;
    std::ifstream infile((filepath_+".meta").c_str(), std::ios::binary);
    if (!infile) {
      std::cout << "Note: cannot open file " << (filepath_+".meta") << std::endl;
      return;
    }

    
    infile.read(reinterpret_cast<char*>(&max_vertex_), sizeof(max_vertex_));
    infile.read(reinterpret_cast<char*>(&curr_max_vertex_id_), sizeof(curr_max_vertex_id_));
    infile.read(reinterpret_cast<char*>(&sample_num_), sizeof(sample_num_));
    infile.read(reinterpret_cast<char*>(&edge_num_per_block_), sizeof(edge_num_per_block_));
    infile.read(reinterpret_cast<char*>(&merge_block_num_), sizeof(merge_block_num_));
    std::cout << "  max_vertex_:" << max_vertex_ << std::endl;
    std::cout << "  curr_max_vertex_id_:" << curr_max_vertex_id_ << std::endl;
    std::cout << "  sample_num_:" << sample_num_ << std::endl;
    std::cout << "  edge_num_per_block_:" << edge_num_per_block_ << std::endl;

    if (max_vertex_ > graph_.size()) {
      std::cout << "Error: max_vertex_ " << max_vertex_ << " is out of range." 
                << " graph_.size()=" << graph_.size()
                << std::endl;
      exit(0);
    }

    for (size_t i = 0; i < max_vertex_; ++i) {
      offset_t offset = -1;
      infile.read(reinterpret_cast<char*>(&offset), sizeof(offset));
      if (i < 10) {
        std::cout << i << "  offset = " << offset << " ";
      }
      if (offset != -1) {
        graph_[i] = new Block(blockmanager.convert_to_prt(offset));
        if (i < 10) { 
          std::vector<edge_t> t;
          get_edges(i, t);
          std::cout << "edges.sizet = " << t.size() << std::endl;
        }
      } else {
        graph_[i] = nullptr;
        if (i < 10) { 
          std::cout << "edges.sizet = 0" << std::endl;
        }
      }
    }
  }

  void add_edge(vertex_t src, edge_t dst) {
    if (src >= max_vertex_) {
      std::cout << "Error: src " << src << " is out of range." << std::endl;
      exit(0);
    }
    
    
    writelock_[src].lock();
    Block* block = graph_[src];
    if (block == nullptr) {
      
      int block_size = sizeof(Block::Head) + sizeof(edge_t) * edge_num_per_block_;
      char* new_pointer = blockmanager.alloc(block_size);

      if (graph_[src] != nullptr) {
        delete graph_[src];
      }
      graph_[src] = new Block(new_pointer, edge_num_per_block_, -1);
      block = graph_[src];
    }
    
    if (block->get_max_edge_num() <= block->get_nbr_num()) { 
      char* new_pointer = nullptr;

      
      if (merge_block_num_ > 0) {
        int block_num = 1;
        std::vector<std::pair<offset_t, int>> recycle_blocks;
        Block block_tmpt = *(graph_[src]);
        recycle_blocks.emplace_back(blockmanager.convert_to_offset(block_tmpt.get_data()), block_tmpt.get_max_edge_num());
        offset_t next_offset = block_tmpt.get_next_offset();

        while (next_offset != -1) {
          block_num++;
          block_tmpt = Block(blockmanager.convert_to_prt(next_offset)); 
          recycle_blocks.emplace_back(blockmanager.convert_to_offset(block_tmpt.get_data()), block_tmpt.get_max_edge_num());
          next_offset = block_tmpt.get_next_offset();
        }
        if (block_num != recycle_blocks.size()) {
          std::cout << "Error: block_num != recycle_blocks.size()" << std::endl;
          exit(0);
        }
        
        if (block_num >= merge_block_num_) {
          
          
          std::vector<edge_t> old_edges;
          

          for (auto& pair : recycle_blocks) {
            Block block_tmpt = Block(blockmanager.convert_to_prt(pair.first));
            blockupdater.get_edges(&block_tmpt, old_edges);
            blockmanager.free(pair.first, pair.second);
          }

          int new_edge_num_per_block = round_up_to_next_power_of_two_constexpr<vertex_t>(old_edges.size());
          if (new_edge_num_per_block <= old_edges.size()) {
            std::cout << "errror: old_edges.size()=" << old_edges.size() 
                      << " new_edge_num_per_block=" << new_edge_num_per_block
                      << std::endl;
            std::exit(0);
          }
          
          
          

          int block_size = sizeof(Block::Head) + sizeof(edge_t) * new_edge_num_per_block;
          new_pointer = blockmanager.alloc(block_size);
          if (graph_[src] != nullptr) {
            delete graph_[src];
          }
          graph_[src] = new Block(new_pointer, new_edge_num_per_block, -1);
          block = graph_[src];
          for (auto& edge : old_edges) {
            blockupdater.add_edge(block, edge);
          }
        }
      }

      
      if (new_pointer == nullptr) {
        int block_size = sizeof(Block::Head) + sizeof(edge_t) * edge_num_per_block_;
        new_pointer = blockmanager.alloc(block_size);
        char* old_pointer = graph_[src]->get_data();
        if (graph_[src] != nullptr) {
          delete graph_[src];
        }
        graph_[src] = new Block(new_pointer, edge_num_per_block_, 
          blockmanager.convert_to_offset(old_pointer));
      }

      block = graph_[src];
      
    }
    
    
    
    blockupdater.add_edge(block, dst);
    writelock_[src].unlock();
  }

  int del_edge(vertex_t src, edge_t dst) {
    if (src >= max_vertex_) {
      std::cout << "Error: src " << src << " is out of range." << std::endl;
      exit(0);
      return 1;
    }

    writelock_[src].lock();

    if(graph_[src] == nullptr) {
      std::cout << "Error: src " << src << " not found in graph. 3" << std::endl;
      writelock_[src].unlock();
      return 1;
    }
    Block block = *(graph_[src]);
    bool result = blockupdater.del_edge(&block, dst);
    offset_t next_offset = block.get_next_offset();
    while (result == false && next_offset != -1) {
      block = Block(blockmanager.convert_to_prt(next_offset));
      result = blockupdater.del_edge(&block, dst);
      next_offset = block.get_next_offset();
    }
    writelock_[src].unlock();
    if (result == false) {
      std::cout << "Error: dst " << dst << " not found in graph. 4 and src = " << src << std::endl;
      // return 1;
      exit(0);
    }
    return 0;
  }

  void get_edges(vertex_t src, std::vector<edge_t> &edge) {
    if (src >= max_vertex_) {
      std::cout << "Error: src " << src << " is out of range." << std::endl;
      exit(0);
    }

    writelock_[src].lock();

    if(graph_[src] == nullptr) {
      writelock_[src].unlock();
      return;
    }

    
    Block block = *(graph_[src]);
    blockupdater.get_edges(&block, edge);
    offset_t next_offset = block.get_next_offset();
    while (next_offset != -1) {
      
      block = Block(blockmanager.convert_to_prt(next_offset)); 
      blockupdater.get_edges(&block, edge);
      next_offset = block.get_next_offset();
    }
    writelock_[src].unlock();
  }

  void sample_degree() {
    int min_dgree = std::numeric_limits<int>::max();
    std::vector<size_t> degree_count;
    degree_count.reserve(sample_num_);
    std::cout << "Note: from 0-" << max_vertex_ << " sample " << sample_num_ << " vertexes" << std::endl;
    for (vertex_t key = 0; key < max_vertex_; key++) {
      if (graph_[key] != nullptr) {
        size_t degree = 0;
        Block block = *(graph_[key]);
        degree += block.get_nbr_num();
        offset_t next_offset = block.get_next_offset();
        while (next_offset != -1) {
          block = Block(blockmanager.convert_to_prt(next_offset));
          degree += block.get_nbr_num();
          next_offset = block.get_next_offset();
        }
        degree_count.push_back(degree);
        if (degree < min_dgree) {
          min_dgree = degree;
        }
        if (degree == 0) {
          std::cout << "degree is 0" << std::endl;
          std::cout << "key=" << key << std::endl;
          std::cout << "degree=" << degree << std::endl;
          std::cout << "block->get_nbr_num()=" << block.get_nbr_num() << std::endl;
        }
      }
      if (degree_count.size() >= sample_num_) {
        break;
      }
    }
    std::sort( degree_count.begin(), degree_count.end() );
    if (degree_count.size() != 0) {
      std::cout << "degree_count.size()=" << degree_count.size() 
                << " middle_degree=" << degree_count[degree_count.size()/2]
                << " min_degree=" << degree_count[0]
                << " max_degree=" << degree_count[degree_count.size()-1]
                << std::endl;
      edge_num_per_block_ = degree_count[degree_count.size()/2];
      std::cout << "update edge_num_per_block_=" << edge_num_per_block_ << std::endl;
    } else {
      std::cout << "degree_count.size()=" << degree_count.size() << std::endl;
    }
  }

  ~BlockGraph() {
    
    // if (load_old_data_ == true) {
    save_meta();
    // }
    std::cout << "\n~BlockGraph will be destructed." << std::endl;
    std::vector<int> block_num_count;
    bool is_count_block_num = 1;
    size_t vertex_num = 0;

    for (vertex_t key = 0; key < max_vertex_; key++) {
      if (graph_[key] != nullptr) {
          vertex_num++;
          
          if (is_count_block_num) {
            
            int tmpt_block_num = 1;
            Block block = *(graph_[key]);
            offset_t next_offset = block.get_next_offset();
            
            while (next_offset != -1) {
              
              tmpt_block_num++;
              block = Block(blockmanager.convert_to_prt(next_offset));
              next_offset = block.get_next_offset();
            }

            if (tmpt_block_num >= block_num_count.size()) {
              block_num_count.resize(tmpt_block_num+1);
            }
            block_num_count[tmpt_block_num]++;
          }

          delete graph_[key]; 
          graph_[key] = nullptr; 
      }
    }
    std::cout << " vertex_num:" << vertex_num << std::endl;
    if (is_count_block_num) {
      int block_num_sum = 0;
      for (int i = 0; i < block_num_count.size(); i++) {
        block_num_sum += block_num_count[i] * i;
        if (block_num_count[i] != 0) {
          std::cout << "   block_num:" << i << " count:" << block_num_count[i] << std::endl;
        }
      }
      std::cout << " block_num_sum:" << block_num_sum << std::endl;
      std::cout << " ave_block_num_sum:" << block_num_sum*1.0/vertex_num << std::endl;
    }
  }

private:
  BlockManager blockmanager;
  BlockUpdater blockupdater;
  std::vector<std::mutex>  writelock_; 
  
  std::vector<Block*> graph_;
  size_t max_vertex_ = 0;
  int edge_num_per_block_;
  int merge_block_num_; 
  int sample_num_; 
  int curr_max_vertex_id_;
  std::string filepath_;
  bool load_old_data_;
};

}
