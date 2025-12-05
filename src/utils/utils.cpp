#include "utils.h"

#include <stdio.h>

namespace dirann {
  // Get the right distance function for the given metric.
  template<>
  dirann::Distance<float> *get_distance_function(dirann::Metric m) {
    if (m == dirann::Metric::L2) {
      return new dirann::DistanceL2();  // compile-time dispatch
    } else if (m == dirann::Metric::COSINE) {
      return new dirann::DistanceCosineFloat();
    } else {
      LOG(ERROR) << "Only L2 and cosine metric supported as of now.";
      crash();
      return nullptr;
    }
  }

  template<>
  dirann::Distance<int8_t> *get_distance_function(dirann::Metric m) {
    if (m == dirann::Metric::L2) {
      return new dirann::DistanceL2Int8();
    } else if (m == dirann::Metric::COSINE) {
      return new dirann::DistanceCosineInt8();
    } else {
      LOG(ERROR) << "Only L2 and cosine metric supported as of now";
      crash();
      return nullptr;
    }
  }

  template<>
  dirann::Distance<uint8_t> *get_distance_function(dirann::Metric m) {
    if (m == dirann::Metric::L2) {
      return new dirann::DistanceL2UInt8();
    } else if (m == dirann::Metric::COSINE) {
      LOG(INFO) << "AVX/AVX2 distance function not defined for Uint8. Using slow version.";
      return new dirann::SlowDistanceCosineUInt8();
    } else {
      LOG(ERROR) << "Only L2 and Cosine metric supported as of now.";
      crash();
      return nullptr;
    }
  }

  void block_convert(std::ofstream &writr, std::ifstream &readr, float *read_buf, _u64 npts, _u64 ndims) {
    readr.read((char *) read_buf, npts * ndims * sizeof(float));
    _u32 ndims_u32 = (_u32) ndims;
#pragma omp parallel for
    for (_s64 i = 0; i < (_s64) npts; i++) {
      float norm_pt = std::numeric_limits<float>::epsilon();
      for (_u32 dim = 0; dim < ndims_u32; dim++) {
        norm_pt += *(read_buf + i * ndims + dim) * *(read_buf + i * ndims + dim);
      }
      norm_pt = std::sqrt(norm_pt);
      for (_u32 dim = 0; dim < ndims_u32; dim++) {
        *(read_buf + i * ndims + dim) = *(read_buf + i * ndims + dim) / norm_pt;
      }
    }
    writr.write((char *) read_buf, npts * ndims * sizeof(float));
  }

  void normalize_data_file(const std::string &inFileName, const std::string &outFileName) {
    std::ifstream readr(inFileName, std::ios::binary);
    std::ofstream writr(outFileName, std::ios::binary);

    int npts_s32, ndims_s32;
    readr.read((char *) &npts_s32, sizeof(_s32));
    readr.read((char *) &ndims_s32, sizeof(_s32));

    writr.write((char *) &npts_s32, sizeof(_s32));
    writr.write((char *) &ndims_s32, sizeof(_s32));

    _u64 npts = (_u64) npts_s32, ndims = (_u64) ndims_s32;
    LOG(INFO) << "Normalizing FLOAT vectors in file: " << inFileName;
    LOG(INFO) << "Dataset: #pts = " << npts << ", # dims = " << ndims;

    _u64 blk_size = 131072;
    _u64 nblks = ROUND_UP(npts, blk_size) / blk_size;
    LOG(INFO) << "# blks: " << nblks;

    float *read_buf = new float[blk_size * ndims];
    for (_u64 i = 0; i < nblks; i++) {
      _u64 cblk_size = std::min(npts - i * blk_size, blk_size);
      block_convert(writr, readr, read_buf, cblk_size, ndims);
    }
    delete[] read_buf;

    LOG(INFO) << "Wrote normalized points to file: " << outFileName;
  }
}  // namespace dirann
