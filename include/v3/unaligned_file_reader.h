#pragma once

#include "aligned_file_reader.h"
#include <libpmem.h>
#include <sys/mman.h>
#include <fcntl.h>

// #define USE_DRAM

class UnalignedFileReader {
private:
    uint8_t *base = nullptr;
    size_t mmap_len = 0;
    int is_pmem = 0;

public:
    UnalignedFileReader() = default;
    ~UnalignedFileReader() { close(); }

    void open(const std::string &fname) {
        
        size_t requested_len = 0;
        struct stat st;
        if (stat(fname.c_str(), &st) == 0) {
            requested_len = st.st_size;
        } else {
            perror("stat");
            exit(1);
        }

        requested_len *= 2;
        #ifdef USE_DRAM

        int fd = ::open(fname.c_str(), O_RDONLY);
        if (fd < 0) { perror("open"); exit(1); }
        mmap_len = requested_len;
        base = (uint8_t *)mmap(NULL, mmap_len, 
                            PROT_READ | PROT_WRITE, 
                            MAP_PRIVATE | MAP_POPULATE, 
                            fd, 0);
        ::close(fd);

        if (base == MAP_FAILED) {
            perror("mmap"); exit(1);
        }
        is_pmem = 0;
        #else
        base = (uint8_t *)pmem_map_file(fname.c_str(), requested_len, PMEM_FILE_CREATE, 0666, &mmap_len, &is_pmem);
        #endif
        LOG(INFO) << (is_pmem ? "PMEM" : "DRAM") << " Mapped. Size: " << mmap_len;
        LOG(INFO) << "Mapped File: " << fname << " File Size: " << requested_len;
        if (!base) {
            perror("pmem_map_file");
            LOG(ERROR) << "Failed to map file: " << fname;
            exit(1);
        }
    }

    void close() {
        if (base) {
            #ifdef USE_DRAM
            munmap(base, mmap_len);
            #else
            pmem_unmap(base, mmap_len);
            #endif
            base = nullptr;
            mmap_len = 0;
        }
    }

    // for zero-copy
    inline uint8_t *get_addr(uint64_t offset) { 
        assert(base != nullptr);
        assert(offset < mmap_len);
        return base + offset;  
    }

    void read(std::vector<IORequest> &read_reqs) {
        for (const auto &req : read_reqs) {
            if (likely(req.u_offset + req.u_len <= mmap_len)) {
                
                // (perf) : Add _mm_prefetch?
                std::memcpy(req.buf, base + req.u_offset, req.u_len);
            } else {
                LOG(ERROR) << "Read out of range at offset " << req.u_offset;
            }
        }
    }

    void write(std::vector<IORequest> &write_reqs) {
        for (const auto &req : write_reqs) {
            if (likely( req.u_offset + req.u_len <= mmap_len)) {
                
                std::memcpy(base + req.u_offset, req.buf, req.u_len);
                
                
                
                // if (is_pmem) {
                //     pmem_persist(base + req.offset, req.len);
                // } else {
                //     pmem_msync(base + req.offset, req.len);
                // }
            } else {
                LOG(ERROR) << "Write out of range at offset " << req.u_offset;
            }
        }
    }
};