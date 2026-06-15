#pragma once

#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <atomic>
#include <memory>
#include <numa.h>
#include <iostream>
#include <stdexcept>

class NumaThreadPool {
public:
    enum class OverflowStrategy { 
        WAIT, 
        SPILLOVER 
    };

    NumaThreadPool(int num_nodes, int threads_per_node) 
        : _num_nodes(num_nodes), _stop(false), _active_tasks(0) {
        
        if (numa_available() < 0) {
            throw std::runtime_error("NUMA not available on this system.");
        }

        
        _node_pools.reserve(num_nodes);
        for (int nid = 0; nid < num_nodes; ++nid) {
            _node_pools.emplace_back(std::make_unique<NodePool>());
        }

        
        for (int nid = 0; nid < num_nodes; ++nid) {
            for (int t = 0; t < threads_per_node; ++t) {
                _node_pools[nid]->workers.emplace_back(&NumaThreadPool::worker_routine, this, nid);
            }
        }
    }

    ~NumaThreadPool() {
        _stop = true;
        for (auto& pool : _node_pools) {
            pool->cv.notify_all();
            for (auto& worker : pool->workers) {
                if (worker.joinable()) worker.join();
            }
        }
    }

    void submit(int preferred_nid, std::function<void()> task, OverflowStrategy strategy = OverflowStrategy::WAIT) {
        int target_nid = preferred_nid;

        
        if (strategy == OverflowStrategy::SPILLOVER && 
            _node_pools[preferred_nid]->queue_size.load(std::memory_order_relaxed) > 20) {
            
            int min_load = _node_pools[preferred_nid]->queue_size.load(std::memory_order_relaxed);
            for (int i = 0; i < _num_nodes; ++i) {
                int current_load = _node_pools[i]->queue_size.load(std::memory_order_relaxed);
                if (current_load < min_load) {
                    min_load = current_load;
                    target_nid = i;
                }
            }
        }

        
        auto& pool = *_node_pools[target_nid];
        {
            std::lock_guard<std::mutex> lock(pool.mtx);
            pool.tasks.push(std::move(task));
            pool.queue_size.fetch_add(1, std::memory_order_relaxed);
        }
        _active_tasks.fetch_add(1, std::memory_order_release);
        pool.cv.notify_one();
    }

    void wait_all() {
        
        while (_active_tasks.load(std::memory_order_acquire) > 0) {
            std::this_thread::yield();
        }
    }

private:
    
    struct alignas(64) NodePool {
        std::vector<std::thread> workers;
        std::queue<std::function<void()>> tasks;
        std::mutex mtx;
        std::condition_variable cv;
        std::atomic<int> queue_size{0}; 
    };

    std::vector<std::unique_ptr<NodePool>> _node_pools;
    int _num_nodes;
    std::atomic<bool> _stop;
    std::atomic<int> _active_tasks;

    void worker_routine(int nid) {
        
        this->bind_thread_to_node(nid);
        
        
        auto& pool = *_node_pools[nid];
        
        while (true) {
            std::function<void()> task;
            {
                std::unique_lock<std::mutex> lock(pool.mtx);
                pool.cv.wait(lock, [&] { return _stop || !pool.tasks.empty(); });
                
                if (_stop && pool.tasks.empty()) return;
                
                task = std::move(pool.tasks.front());
                pool.tasks.pop();
                pool.queue_size.fetch_sub(1, std::memory_order_relaxed);
            }
            
            task(); 
            _active_tasks.fetch_sub(1, std::memory_order_acq_rel);
        }
    }

    void bind_thread_to_node(int nid) {
        if (numa_available() < 0) return;
        struct bitmask* mask = numa_allocate_nodemask();
        numa_bitmask_setbit(mask, nid);
        numa_bind(mask);
        numa_free_nodemask(mask);
    }
};