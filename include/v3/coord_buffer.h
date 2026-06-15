#pragma once
#include <atomic>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <vector>
#include <memory>
#include <stdexcept>
#include <algorithm>

#include "ssd_index_defs.h"
#include "aligned_file_reader.h"

class CoordBuffer {
public:
    
    CoordBuffer(std::shared_ptr<AlignedFileReader> reader,
                uint64_t bytes_per_elem,
                uint64_t batch_sectors = 256,
                uint64_t meta_sectors = 1,
                uint64_t init_loc_bytes = 0)
        : reader_(std::move(reader)),
          bytes_per_elem_(bytes_per_elem),
          batch_sectors_(batch_sectors),
          meta_sectors_(meta_sectors) {

        if (!reader_) throw std::runtime_error("CoordBuffer: reader is null");
        if (bytes_per_elem_ == 0 || bytes_per_elem_ > SECTOR_LEN)
            throw std::runtime_error("CoordBuffer: invalid bytes_per_elem");
        if (batch_sectors_ == 0)
            throw std::runtime_error("CoordBuffer: batch_sectors == 0");
        if (meta_sectors_ == 0)
            throw std::runtime_error("CoordBuffer: meta_sectors == 0");

        vecs_per_sector_ = SECTOR_LEN / bytes_per_elem_;
        if (vecs_per_sector_ == 0)
            throw std::runtime_error("CoordBuffer: bytes_per_elem too large to fit in a sector");

        data_base_ = meta_sectors_ * SECTOR_LEN;

        // tail_sector_
        {
            void* p = nullptr;
            int rc = ::posix_memalign(&p, SECTOR_LEN, SECTOR_LEN);
            if (rc != 0) throw std::bad_alloc();
            tail_sector_ = p;
        }

        // batch_buf_
        {
            void* p = nullptr;
            int rc = ::posix_memalign(&p, SECTOR_LEN, batch_sectors_ * SECTOR_LEN);
            if (rc != 0) {
                ::free(tail_sector_);
                tail_sector_ = nullptr;
                throw std::bad_alloc();
            }
            batch_buf_ = p;
        }

        std::memset(tail_sector_, 0, SECTOR_LEN);

        
        init_from_byte_offset(init_loc_bytes);

        
        flushed_upto_bytes_.store(flushed_upto_bytes_init_, std::memory_order_release);
        next_loc_.store(next_loc_init_, std::memory_order_release);
    }

    ~CoordBuffer() {
        ::free(tail_sector_);
        ::free(batch_buf_);
        tail_sector_ = nullptr;
        batch_buf_ = nullptr;
    }

    CoordBuffer(const CoordBuffer&) = delete;
    CoordBuffer& operator=(const CoordBuffer&) = delete;

    
    uint64_t put(const uint8_t* coord) {
        if (!coord) throw std::runtime_error("CoordBuffer::put coord is null");

        std::lock_guard<std::mutex> lk(mtx_);

        
        if (tail_used_vecs_ == vecs_per_sector_) {
            seal_tail_locked();
        }

        const uint64_t loc = next_loc_.load(std::memory_order_relaxed);

        
        const uint64_t off = tail_used_vecs_ * bytes_per_elem_;
        std::memcpy(static_cast<uint8_t*>(tail_sector_) + off, coord, bytes_per_elem_);
        tail_used_vecs_++;

        next_loc_.store(loc + 1, std::memory_order_release);

        
        if (tail_used_vecs_ == vecs_per_sector_) {
            seal_tail_locked();
        }

        return loc;
    }

    
    bool get(uint64_t loc, uint8_t* out) {
        if (!out) return false;

        const uint64_t next = next_loc_.load(std::memory_order_acquire);
        if (loc >= next) return false;

        std::lock_guard<std::mutex> lk(mtx_);

        
        const uint64_t off_bytes = byte_offset_of_loc(loc);
        const uint64_t flushed = flushed_upto_bytes_.load(std::memory_order_acquire);
        if (off_bytes < flushed) return false;

        const uint64_t sector_base = align_down(off_bytes, SECTOR_LEN);

        
        
        
        const uint64_t batch_start = flushed;
        const uint64_t batch_end = batch_start + batch_used_sectors_ * SECTOR_LEN;

        if (sector_base >= batch_start && sector_base < batch_end) {
            const uint64_t in_batch = sector_base - batch_start;
            const uint64_t slot = loc % vecs_per_sector_;
            const uint64_t in_sector = slot * bytes_per_elem_;
            std::memcpy(out,
                        static_cast<uint8_t*>(batch_buf_) + in_batch + in_sector,
                        bytes_per_elem_);
            return true;
        }

        
        if (sector_base == tail_sector_base_bytes_) {
            const uint64_t slot = loc % vecs_per_sector_;
            if (slot >= tail_used_vecs_) return false; 
            const uint64_t in_sector = slot * bytes_per_elem_;
            std::memcpy(out,
                        static_cast<uint8_t*>(tail_sector_) + in_sector,
                        bytes_per_elem_);
            return true;
        }

        return false;
    }

    
    void flush() {
        std::lock_guard<std::mutex> lk(mtx_);
        flush_batch_locked();
    }

    
    void flush_tail_with_padding() {
        std::lock_guard<std::mutex> lk(mtx_);

        flush_batch_locked();

        if (tail_used_vecs_ == 0) return;

        void* ctx = reader_->get_ctx(0);

        std::vector<IORequest> req(1);
        req[0].buf = tail_sector_;
        req[0].len = SECTOR_LEN;
        req[0].offset = tail_sector_base_bytes_;
        reader_->write(req, ctx, false);

        flushed_upto_bytes_.store(tail_sector_base_bytes_ + SECTOR_LEN, std::memory_order_release);

        
        tail_sector_base_bytes_ += SECTOR_LEN;
        tail_used_vecs_ = 0;
        std::memset(tail_sector_, 0, SECTOR_LEN);
    }

    
    uint64_t byte_offset_of_loc(uint64_t loc) const {
        const uint64_t sector_idx = loc / vecs_per_sector_;
        const uint64_t slot = loc % vecs_per_sector_;
        return data_base_ + sector_idx * SECTOR_LEN + slot * bytes_per_elem_;
    }

private:
    static uint64_t align_down(uint64_t x, uint64_t a) { return (x / a) * a; }

    void init_from_byte_offset(uint64_t init_loc_bytes) {
        
        if (init_loc_bytes < data_base_) init_loc_bytes = data_base_;

        const uint64_t rel = init_loc_bytes - data_base_;
        uint64_t sector = rel / SECTOR_LEN;
        uint64_t in_sector = rel % SECTOR_LEN;

        uint64_t slot = in_sector / bytes_per_elem_;
        uint64_t rem  = in_sector % bytes_per_elem_;

        
        if (slot >= vecs_per_sector_) {
            sector += 1;
            slot = 0;
            rem = 0;
            in_sector = 0;
        }

        
        
        if (rem != 0) {
            slot += 1;
            if (slot >= vecs_per_sector_) {
                sector += 1;
                slot = 0;
            }
            rem = 0;
        }

        tail_sector_base_bytes_ = data_base_ + sector * SECTOR_LEN;
        tail_used_vecs_ = slot;

        
        next_loc_init_ = sector * vecs_per_sector_ + slot;

        
        flushed_upto_bytes_init_ = tail_sector_base_bytes_;

        
        std::memset(tail_sector_, 0, SECTOR_LEN);
        if (tail_used_vecs_ > 0) {
            void* ctx = reader_->get_ctx(0);
            std::vector<IORequest> req(1);
            req[0].buf = tail_sector_;
            req[0].len = SECTOR_LEN;
            req[0].offset = tail_sector_base_bytes_;

            
            reader_->read(req, ctx, false);
        }
    }

    void seal_tail_locked() {
        
        std::memcpy(static_cast<uint8_t*>(batch_buf_) + batch_used_sectors_ * SECTOR_LEN,
                    tail_sector_, SECTOR_LEN);
        batch_used_sectors_++;

        
        tail_sector_base_bytes_ += SECTOR_LEN;
        tail_used_vecs_ = 0;
        std::memset(tail_sector_, 0, SECTOR_LEN);

        if (batch_used_sectors_ == batch_sectors_) flush_batch_locked();
    }

    void flush_batch_locked() {
        if (batch_used_sectors_ == 0) return;

        void* ctx = reader_->get_ctx(0);
        const uint64_t flushed = flushed_upto_bytes_.load(std::memory_order_acquire);

        std::vector<IORequest> req(1);
        req[0].buf = batch_buf_;
        req[0].len = batch_used_sectors_ * SECTOR_LEN;
        req[0].offset = flushed;

        reader_->write(req, ctx, false);

        flushed_upto_bytes_.store(flushed + req[0].len, std::memory_order_release);
        batch_used_sectors_ = 0;
    }

private:
    std::shared_ptr<AlignedFileReader> reader_;
    uint64_t bytes_per_elem_;
    uint64_t batch_sectors_;
    uint64_t meta_sectors_;

    uint64_t vecs_per_sector_{0};
    uint64_t data_base_{0};

    std::atomic<uint64_t> next_loc_{0};              
    std::atomic<uint64_t> flushed_upto_bytes_{0};    

    std::mutex mtx_;

    void* tail_sector_{nullptr};
    uint64_t tail_sector_base_bytes_{0}; 
    uint64_t tail_used_vecs_{0};         

    void* batch_buf_{nullptr};
    uint64_t batch_used_sectors_{0};

    
    uint64_t next_loc_init_{0};
    uint64_t flushed_upto_bytes_init_{0};
};
