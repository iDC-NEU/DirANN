#include "utils.h"

#include <stdio.h>

namespace pipeann {
  // Get the right distance function for the given metric.
  template<>
  pipeann::Distance<float> *get_distance_function(pipeann::Metric m) {
    if (m == pipeann::Metric::L2) {
      return new pipeann::DistanceL2();  // compile-time dispatch
    } else if (m == pipeann::Metric::COSINE) {
      return new pipeann::DistanceCosineFloat();
    } else {
      LOG(ERROR) << "Only L2 and cosine metric supported as of now.";
      crash();
      return nullptr;
    }
  }

  template<>
  pipeann::Distance<int8_t> *get_distance_function(pipeann::Metric m) {
    if (m == pipeann::Metric::L2) {
      return new pipeann::DistanceL2Int8();
    } else if (m == pipeann::Metric::COSINE) {
      return new pipeann::DistanceCosineInt8();
    } else {
      LOG(ERROR) << "Only L2 and cosine metric supported as of now";
      crash();
      return nullptr;
    }
  }

  template<>
  pipeann::Distance<uint8_t> *get_distance_function(pipeann::Metric m) {
    if (m == pipeann::Metric::L2) {
      return new pipeann::DistanceL2UInt8();
    } else if (m == pipeann::Metric::COSINE) {
      LOG(INFO) << "AVX/AVX2 distance function not defined for Uint8. Using slow version.";
      return new pipeann::SlowDistanceCosineUInt8();
    } else {
      LOG(ERROR) << "Only L2 and Cosine metric supported as of now.";
      crash();
      return nullptr;
    }
  }

  void normalize_data_file(const std::string &inFileName, const std::string &outFileName) {
    std::ifstream readr(inFileName, std::ios::binary);
    std::ofstream writr(outFileName, std::ios::binary);

    int npts_s32, ndims_s32;
    readr.read((char *) &npts_s32, sizeof(int32_t));
    readr.read((char *) &ndims_s32, sizeof(int32_t));

    writr.write((char *) &npts_s32, sizeof(int32_t));
    writr.write((char *) &ndims_s32, sizeof(int32_t));

    uint64_t npts = (uint64_t) npts_s32, ndims = (uint64_t) ndims_s32;
    LOG(INFO) << "Normalizing FLOAT vectors in file: " << inFileName;
    LOG(INFO) << "Dataset: #pts = " << npts << ", # dims = " << ndims;

    uint64_t blk_size = 131072;
    uint64_t nblks = ROUND_UP(npts, blk_size) / blk_size;
    LOG(INFO) << "# blks: " << nblks;

    float *read_buf = new float[blk_size * ndims];
    for (uint64_t i = 0; i < nblks; i++) {
      uint64_t cblk_size = std::min(npts - i * blk_size, blk_size);

      readr.read((char *) read_buf, cblk_size * ndims * sizeof(float));
      uint32_t ndims_u32 = (uint32_t) ndims;
#pragma omp parallel for
      for (int64_t i = 0; i < (int64_t) cblk_size; i++) {
        float norm_pt = std::numeric_limits<float>::epsilon();
        for (uint32_t dim = 0; dim < ndims_u32; dim++) {
          norm_pt += *(read_buf + i * ndims + dim) * *(read_buf + i * ndims + dim);
        }
        norm_pt = std::sqrt(norm_pt);
        for (uint32_t dim = 0; dim < ndims_u32; dim++) {
          *(read_buf + i * ndims + dim) = *(read_buf + i * ndims + dim) / norm_pt;
        }
      }
      writr.write((char *) read_buf, cblk_size * ndims * sizeof(float));
    }
    delete[] read_buf;

    LOG(INFO) << "Wrote normalized points to file: " << outFileName;
  }
}  // namespace pipeann
