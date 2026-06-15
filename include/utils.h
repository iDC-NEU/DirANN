#pragma once
#include <fcntl.h>
#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <memory>
#include <random>
#include <sstream>
#include <malloc.h>
#include <unistd.h>
#include <sys/stat.h>

#include "distance.h"
#include "utils/log.h"

// taken from
// https://github.com/Microsoft/BLAS-on-flash/blob/master/include/utils.h
// round up X to the nearest multiple of Y
#define ROUND_UP(X, Y) ((((uint64_t) (X) / (Y)) + ((uint64_t) (X) % (Y) != 0)) * (Y))

#define DIV_ROUND_UP(X, Y) (((uint64_t) (X) / (Y)) + ((uint64_t) (X) % (Y) != 0))

// round down X to the nearest multiple of Y
#define ROUND_DOWN(X, Y) (((uint64_t) (X) / (Y)) * (Y))

// alignment tests
#define IS_ALIGNED(X, Y) ((uint64_t) (X) % (uint64_t) (Y) == 0)
#define IS_512_ALIGNED(X) IS_ALIGNED(X, 512)
#define METADATA_SIZE \
  4096  // all metadata of individual sub-component files is written in first
        // 4KB for unified files

/* Used to replace exception. */
inline void crash() {
  // #ifndef NDEBUG
  __builtin_trap();
  // #endif
}

static inline bool file_exists(const std::string &name, bool dirCheck = false) {
  int val;
  struct stat buffer;
  val = stat(name.c_str(), &buffer);

  DLOG(INFO) << " Stat(" << name.c_str() << ") returned: " << val;
  if (val != 0) {
    switch (errno) {
      case EINVAL:
        DLOG(INFO) << "Invalid argument passed to stat()";
        break;
      case ENOENT:
        DLOG(INFO) << "File " << name.c_str() << " does not exist";
        break;
      default:
        DLOG(INFO) << "Unexpected error in stat():" << errno;
        break;
    }
    return false;
  } else {
    // the file entry exists. If reqd, check if this is a directory.
    return dirCheck ? buffer.st_mode & S_IFDIR : true;
  }
}

inline void open_file_to_write(std::ofstream &writer, const std::string &filename) {
  writer.exceptions(std::ofstream::failbit | std::ofstream::badbit);
  if (!file_exists(filename))
    writer.open(filename, std::ios::binary | std::ios::out);
  else
    writer.open(filename, std::ios::binary | std::ios::in | std::ios::out);
  if (writer.fail()) {
    LOG(ERROR) << std::string("Failed to open file") + filename + " for write because " << std::strerror(errno);
    crash();
  }
}

inline uint64_t get_file_size(const std::string &fname) {
  std::ifstream reader(fname, std::ios::binary | std::ios::ate);
  if (!reader.fail() && reader.is_open()) {
    uint64_t end_pos = reader.tellg();
    reader.close();
    return end_pos;
  } else {
    LOG(ERROR) << "Could not open file: " << fname;
    return 0;
  }
}

inline int delete_file(const std::string &fileName) {
  if (file_exists(fileName)) {
    auto rc = ::remove(fileName.c_str());
    if (rc != 0) {
      LOG(ERROR) << "Could not delete file: " << fileName
                 << " even though it exists. This might indicate a permissions issue. "
                    "If you see this message, please contact the pipeann team.";
    }
    return rc;
  } else {
    return 0;
  }
}

namespace pipeann {
  enum Metric { L2 = 0, INNER_PRODUCT = 1, COSINE = 2 };

  inline void alloc_aligned(void **ptr, size_t size, size_t align) {
    *ptr = nullptr;
    assert(IS_ALIGNED(size, align));
    *ptr = ::aligned_alloc(align, size);
    assert(*ptr != nullptr);
  }

  inline void aligned_free(void *ptr) {
    if (ptr == nullptr) {
      return;
    }
    free(ptr);
  }

  inline void get_bin_metadata(const std::string &bin_file, size_t &nrows, size_t &ncols, size_t offset = 0) {
    std::ifstream reader(bin_file.c_str(), std::ios::binary);
    int nrows_32, ncols_32;
    reader.seekg(offset, reader.beg);
    reader.read((char *) &nrows_32, sizeof(int));
    reader.read((char *) &ncols_32, sizeof(int));
    nrows = nrows_32;
    ncols = ncols_32;
  }

  template<typename T>
  inline void load_bin_impl(std::basic_istream<char> &reader, T *&data, size_t &npts, size_t &dim,
                            size_t file_offset = 0) {
    int npts_i32, dim_i32;

    reader.seekg(file_offset, reader.beg);
    reader.read((char *) &npts_i32, sizeof(int));
    reader.read((char *) &dim_i32, sizeof(int));
    npts = (unsigned) npts_i32;
    dim = (unsigned) dim_i32;

    DLOG(INFO) << "Metadata: #pts = " << npts << ", #dims = " << dim << "...";

    data = new T[npts * dim];
    reader.read((char *) data, npts * dim * sizeof(T));
  }

  template<typename T>
  inline void load_bin_impl(std::basic_istream<char> &reader, std::vector<T> &data, size_t &npts, size_t &dim,
                            size_t file_offset = 0) {
    int npts_i32, dim_i32;

    reader.seekg(file_offset, reader.beg);
    reader.read((char *) &npts_i32, sizeof(int));
    reader.read((char *) &dim_i32, sizeof(int));
    npts = (unsigned) npts_i32;
    dim = (unsigned) dim_i32;

    DLOG(INFO) << "Metadata: #pts = " << npts << ", #dims = " << dim << "...";

    data.resize(npts * dim);
    reader.read((char *) (data.data()), npts * dim * sizeof(T));
  }

  template<typename T>
  inline void load_bin(const std::string &bin_file, T *&data, size_t &npts, size_t &dim, size_t offset = 0) {
    DLOG(INFO) << "Reading bin file " << bin_file.c_str() << " ...";
    std::ifstream reader(bin_file, std::ios::binary | std::ios::ate);
    reader.seekg(0);

    load_bin_impl<T>(reader, data, npts, dim, offset);
  }

  template<typename T>
  inline void load_bin(const std::string &bin_file, std::vector<T> &data, size_t &npts, size_t &dim,
                       size_t offset = 0) {
    DLOG(INFO) << "Reading bin file " << bin_file.c_str() << " ...";
    std::ifstream reader(bin_file, std::ios::binary | std::ios::ate);
    reader.seekg(0);

    load_bin_impl<T>(reader, data, npts, dim, offset);
  }

  inline void load_truthset(const std::string &bin_file, uint32_t *&ids, float *&dists, size_t &npts, size_t &dim,
                            uint32_t **tags = nullptr) {
    std::ifstream reader(bin_file, std::ios::binary);
    LOG(INFO) << "Reading truthset file " << bin_file.c_str() << "...";
    size_t actual_file_size = get_file_size(bin_file);

    int npts_i32, dim_i32;
    reader.read((char *) &npts_i32, sizeof(int));
    reader.read((char *) &dim_i32, sizeof(int));
    npts = (unsigned) npts_i32;
    dim = (unsigned) dim_i32;

    LOG(INFO) << "Metadata: #pts = " << npts << ", #dims = " << dim << "...";

    int truthset_type = -1;  // 1 means truthset has ids and distances, 2 means
                             // only ids, -1 is error
    size_t expected_file_size_with_dists = 2 * npts * dim * sizeof(uint32_t) + 2 * sizeof(uint32_t);

    if (actual_file_size == expected_file_size_with_dists)
      truthset_type = 1;

    size_t expected_file_size_just_ids = npts * dim * sizeof(uint32_t) + 2 * sizeof(uint32_t);

    size_t with_tags_actual_file_size = 3 * npts * dim * sizeof(uint32_t) + 2 * sizeof(uint32_t);

    if (actual_file_size == expected_file_size_just_ids)
      truthset_type = 2;

    if (actual_file_size == with_tags_actual_file_size)
      truthset_type = 3;

    if (truthset_type == -1) {
      std::stringstream stream;
      stream << "Error. File size mismatch. File should have bin format, with "
                "npts followed by ngt followed by npts*ngt ids and optionally "
                "followed by npts*ngt distance values; actual size: "
             << actual_file_size << ", expected: " << expected_file_size_with_dists << " or "
             << expected_file_size_just_ids;
      LOG(INFO) << stream.str();
      crash();
    }

    ids = new uint32_t[npts * dim];
    reader.read((char *) ids, npts * dim * sizeof(uint32_t));

    if ((truthset_type == 1) || (truthset_type == 3)) {
      dists = new float[npts * dim];
      reader.read((char *) dists, npts * dim * sizeof(float));
    }
    if (truthset_type == 3) {
      *tags = new uint32_t[npts * dim];
      reader.read((char *) *tags, npts * dim * sizeof(uint32_t));
    }
  }

  template<typename T>
  inline void load_bin(const std::string &bin_file, std::unique_ptr<T[]> &data, size_t &npts, size_t &dim,
                       size_t offset = 0) {
    T *ptr;
    load_bin<T>(bin_file, ptr, npts, dim, offset);
    data.reset(ptr);
  }

  template<typename T>
  inline uint64_t save_bin(const std::string &filename, T *data, size_t npts, size_t ndims, size_t offset = 0) {
    std::ofstream writer;
    open_file_to_write(writer, filename);

    DLOG(INFO) << "Writing bin: " << filename.c_str();
    writer.seekp(offset, writer.beg);
    int npts_i32 = (int) npts, ndims_i32 = (int) ndims;
    size_t bytes_written = npts * ndims * sizeof(T) + 2 * sizeof(uint32_t);
    writer.write((char *) &npts_i32, sizeof(int));
    writer.write((char *) &ndims_i32, sizeof(int));
    DLOG(INFO) << "bin: #pts = " << npts << ", #dims = " << ndims << ", size = " << bytes_written << "B";

    writer.write((char *) data, npts * ndims * sizeof(T));
    writer.close();
    DLOG(INFO) << "Finished writing bin.";
    return bytes_written;
  }

  template<typename InType, typename OutType>
  void convert_types(const InType *srcmat, OutType *destmat, size_t npts, size_t dim) {
#pragma omp parallel for schedule(static, 65536)
    for (int64_t i = 0; i < (int64_t) npts; i++) {
      for (uint64_t j = 0; j < dim; j++) {
        destmat[i * dim + j] = (OutType) srcmat[i * dim + j];
      }
    }
  }

  template<typename T>
  inline uint64_t save_data_in_base_dimensions(const std::string &filename, T *data, size_t npts, size_t ndims,
                                               size_t aligned_dim, size_t offset = 0) {
    std::ofstream writer;  //(filename, std::ios::binary | std::ios::out);
    open_file_to_write(writer, filename);
    int npts_i32 = (int) npts, ndims_i32 = (int) ndims;
    uint64_t bytes_written = 2 * sizeof(uint32_t) + npts * ndims * sizeof(T);
    writer.seekp(offset, writer.beg);
    writer.write((char *) &npts_i32, sizeof(int));
    writer.write((char *) &ndims_i32, sizeof(int));
    for (size_t i = 0; i < npts; i++) {
      writer.write((char *) (data + i * aligned_dim), ndims * sizeof(T));
    }
    writer.close();
    return bytes_written;
  }

  template<typename T>
  inline void copy_aligned_data_from_file(const std::string bin_file, T *&data, size_t &npts, size_t &dim,
                                          const size_t &rounded_dim, size_t offset = 0) {
    if (data == nullptr) {
      LOG(INFO) << "Memory was not allocated for " << data << " before calling the load function. Exiting...";
      exit(-1);
    }
    std::ifstream reader(bin_file, std::ios::binary);
    reader.seekg(offset, reader.beg);

    int npts_i32, dim_i32;
    reader.read((char *) &npts_i32, sizeof(int));
    reader.read((char *) &dim_i32, sizeof(int));
    npts = (unsigned) npts_i32;
    dim = (unsigned) dim_i32;

    for (size_t i = 0; i < npts; i++) {
      reader.read((char *) (data + i * rounded_dim), dim * sizeof(T));
      memset(data + i * rounded_dim + dim, 0, (rounded_dim - dim) * sizeof(T));
    }
  }

  // NOTE :: good efficiency when total_vec_size is integral multiple of 64
  inline void prefetch_vector(const char *vec, size_t vecsize) {
    size_t max_prefetch_size = (vecsize / 64) * 64;
    for (size_t d = 0; d < max_prefetch_size; d += 64)
      _mm_prefetch((const char *) vec + d, _MM_HINT_T0);
  }

  void normalize_data_file(const std::string &inFileName, const std::string &outFileName);

  template<typename T>
  Distance<T> *get_distance_function(Metric m);

  struct Parameters {
    uint32_t R = 0;              // maximum out-neighbors.
    uint32_t L = 0;              // build L.
    uint32_t C = 384;            // delete pruning capacity.
    float alpha = 1.2f;          // alpha for Vamana.
    uint32_t num_threads = 0;    // num_threads used.
    bool saturate_graph = true;  // saturate graph during build (using kNN neighbors).
    uint32_t beam_width = 4;     // insert beam width. (for SSD)

    void set(uint32_t R, uint32_t L, uint32_t C, float alpha = 1.2, uint32_t num_threads = 0,
             bool saturate_graph = true, uint32_t beam_width = 4) {
      this->R = R;
      this->L = L;
      this->C = C;
      this->alpha = alpha;
      this->num_threads = num_threads;
      this->beam_width = beam_width;
      this->saturate_graph = saturate_graph;
    }
  };
}  // namespace pipeann

struct PivotContainer {
  PivotContainer() = default;

  PivotContainer(size_t pivo_id, float pivo_dist) : piv_id{pivo_id}, piv_dist{pivo_dist} {
  }

  bool operator<(const PivotContainer &p) const {
    return p.piv_dist < piv_dist;
  }

  bool operator>(const PivotContainer &p) const {
    return p.piv_dist > piv_dist;
  }

  size_t piv_id;
  float piv_dist;
};

template<typename T>
pipeann::Distance<T> *get_distance_function(pipeann::Metric m);
