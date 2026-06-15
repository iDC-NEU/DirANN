#pragma once

#include "nbr/abstract_nbr.h"
#include "ssd_index_defs.h"
#include "utils/libcuckoo/cuckoohash_map.hh"
#include "utils.h"
#include <immintrin.h>
#include <vector>
#include "partition.h"
#include "math_utils.h"
#include "utils/lock_table.h"

#ifndef USE_AVX512
#define USE_AVX512
#endif

#ifdef USE_AVX512
#define PRIMARY_BITS 4
#define RESIDUAL_BITS 8
// LVQNeighbor is a neighbor index that uses LVQ (Locally Vector Quantized) to quantize the vectors.
// Here we use the implementation of LVQ with radius-based multi-level quantization.
namespace pipeann {
  struct LVQPointLevel {
    float bias = 0.f;            // low
    float scale = 1.f;           // (max - min) / (2^bits - 1)
    std::vector<uint8_t> codes;  

    LVQPointLevel() = default;
    LVQPointLevel(float bias, float scale, std::vector<uint8_t> codes)
    : bias(bias), scale(scale), codes(std::move(codes)) {
    }
    LVQPointLevel(float bias, float scale, uint8_t *codes_ptr, size_t len)
    : bias(bias), scale(scale), codes(codes_ptr, codes_ptr + len) {
    }
  };
  /// GlobalMinMax is a class that stores the minimum and maximum values of a set of vectors.
  class GlobalMinMax {
   private:
    float min_ = std::numeric_limits<float>::max();
    float max_ = std::numeric_limits<float>::lowest();

   public:
    // Constructors
    GlobalMinMax() = default;
    explicit GlobalMinMax(float min, float max) : min_{min}, max_{max} {
    }

    float min() const {
      return min_;
    }
    float max() const {
      return max_;
    }

    /// Compute the two-constant scale for the given minimum and maximum.
    float scale(size_t nbits) const {
      return (max() - min()) / (std::pow(2.0f, static_cast<float>(nbits)) - 1);
    }

    // Update
    void update(float v) {
      min_ = std::min(v, min_);
      max_ = std::max(v, max_);
    }

    void update(GlobalMinMax other) {
      min_ = std::min(min(), other.min());
      max_ = std::max(max(), other.max());
    }
    std::pair<float, float> get() const {
      return {min_, max_};
    }
  };

  template<typename T, size_t PrimaryBits, size_t ResidualBits = 0>
  class LVQNeighbor : public AbstractNeighbor<T> {
    using Base = AbstractNeighbor<T>;

    static constexpr uint8_t kNumCenters = 1;     
    static constexpr uint32_t kBlockMax = 16384;  

    
    uint64_t dim_ = 0;

    
    std::vector<std::vector<float>> centroids_;  // [kNumCenters][dim_]

    
    v2::ReaderOptSharedMutex lvq_mu;

    

    
    std::vector<LVQPointLevel> primary_;
    std::vector<LVQPointLevel> residual_;  

   public:
    static constexpr bool UseResidualBits = ResidualBits > 0 && false;
    LVQNeighbor() = default;
    ~LVQNeighbor() override = default;

    std::string get_name() {
      return "LVQNeighbor";
    }
    LVQPointLevel get_primary(uint32_t id) {
      return primary_[id];
    }
    LVQPointLevel get_residual(uint32_t id) {
      return residual_[id];
    }
    std::vector<std::vector<float>> get_centroids() {
      return centroids_;
    }
    uint64_t query_ctx_size() {
      return sizeof(float) * this->dim_;
    }
    void initialize_query(const T *query, QueryBuffer<T> *query_buf) override {
      
      float *q_f32 = reinterpret_cast<float *>(query_buf->nbr_ctx_scratch);
      for (uint64_t i = 0; i < this->dim_; ++i) {
        q_f32[i] = static_cast<float>(query[i]);
      }
    }
    void compute_dists(QueryBuffer<T> *query_buf, const uint32_t *ids, const uint64_t n_ids) override {
      lvq_mu.lock_shared();
      const float *q_f32 = reinterpret_cast<const float *>(query_buf->nbr_ctx_scratch);

      
      for (auto &c : centroids_) {
        pipeann::prefetch_vector(reinterpret_cast<const char *>(c.data()), c.size() * sizeof(float));
      }

      
      const size_t primary_code_bytes = (this->dim_ * PrimaryBits + 7) / 8;
      const size_t residual_code_bytes = (UseResidualBits) ? (this->dim_ * ResidualBits + 7) / 8 : 0;

      for (int64_t i = 0; i < static_cast<int64_t>(n_ids); ++i) {
        
        if (i + 1 < static_cast<int64_t>(n_ids)) {
          const uint32_t next_id = ids[i + 1];

          if (next_id < primary_.size()) {
            const auto &Pn = primary_[next_id];
            pipeann::prefetch_vector(reinterpret_cast<const char *>(Pn.codes.data()), primary_code_bytes);
          }
          if constexpr (UseResidualBits) {
            if (next_id < residual_.size()) {
              const auto &Rn = residual_[next_id];
              pipeann::prefetch_vector(reinterpret_cast<const char *>(Rn.codes.data()), residual_code_bytes);
            }
          }
        }

        
        query_buf->aligned_dist_scratch[i] = compute_distance_avx512(q_f32, ids[i]);
      }
      lvq_mu.unlock_shared();
    }

    float single_compute_dists(const T *query, uint32_t id) {
      lvq_mu.lock_shared();
      const float *q_f32 = reinterpret_cast<const float *>(query);
      for (auto &c : centroids_) {
        pipeann::prefetch_vector(reinterpret_cast<const char *>(c.data()), c.size() * sizeof(float));
      }
      float dist = compute_distance_avx512(q_f32, id);
      lvq_mu.unlock_shared();
      return dist;
    }

    void compute_dists_only_primary(QueryBuffer<T> *query_buf, const uint32_t *ids, const uint64_t n_ids)  {
      lvq_mu.lock_shared();
      const float *q_f32 = reinterpret_cast<const float *>(query_buf->nbr_ctx_scratch);
      
      for (auto &c : centroids_) {
        pipeann::prefetch_vector(reinterpret_cast<const char *>(c.data()), c.size() * sizeof(float));
      }
      
      const size_t primary_code_bytes = (this->dim_ * PrimaryBits + 7) / 8;
      for (int64_t i = 0; i < static_cast<int64_t>(n_ids); ++i) {
        
        if (i + 1 < static_cast<int64_t>(n_ids)) {
          const uint32_t next_id = ids[i + 1];

          if (next_id < primary_.size()) {
            const auto &Pn = primary_[next_id];
            pipeann::prefetch_vector(reinterpret_cast<const char *>(Pn.codes.data()), primary_code_bytes);
          }
        }
        
        auto dist= compute_distance_avx512_only_primary(q_f32, ids[i]);
        query_buf->aligned_dist_scratch[i] = dist;
      }
      lvq_mu.unlock_shared();
    }
    
    void compute_dists_with_residual(QueryBuffer<T> *query_buf, const uint32_t *ids, const uint64_t n_ids, const char *residual_data) {
      lvq_mu.lock_shared();
      const float *q_f32 = reinterpret_cast<const float *>(query_buf->nbr_ctx_scratch);

      
      for (auto &c : centroids_) {
        pipeann::prefetch_vector(reinterpret_cast<const char *>(c.data()), c.size() * sizeof(float));
      }

      
      const size_t primary_code_bytes = (this->dim_ * PrimaryBits + 7) / 8;
      const size_t residual_code_bytes = (this->dim_ * ResidualBits + 7) / 8;

      for (int64_t i = 0; i < static_cast<int64_t>(n_ids); ++i) {
        
        const char * ptr = residual_data;
        LVQPointLevel Rn(*(float *)ptr, *(float *)(ptr + 4), (uint8_t *)(ptr + 8), residual_code_bytes);
        
        if (i + 1 < static_cast<int64_t>(n_ids)) {
          const uint32_t next_id = ids[i + 1];

          if (next_id < primary_.size()) {
            const auto &Pn = primary_[next_id];
            pipeann::prefetch_vector(reinterpret_cast<const char *>(Pn.codes.data()), primary_code_bytes);
          }

          pipeann::prefetch_vector(reinterpret_cast<const char *>(Rn.codes.data()), residual_code_bytes);
        }

        
        query_buf->aligned_dist_scratch[i] = compute_distance_avx512(q_f32, ids[i], primary_[ids[i]], &Rn);
      }
      lvq_mu.unlock_shared();
    };

    
    void build(const std::string &index_prefix, const std::string &data_bin,
               uint32_t /*bytes_per_nbr_unused*/) override {
      
      size_t npts, dim;
      pipeann::get_bin_metadata(data_bin, npts, dim);
      this->npoints = static_cast<uint64_t>(npts);
      dim_ = static_cast<uint64_t>(dim);

      
      float *train_data = nullptr;
      size_t train_size = 0, train_dim = 0;
      gen_random_slice<T>(data_bin, this->get_sample_p(), train_data, train_size, train_dim);

      centroids_.assign(kNumCenters, std::vector<float>(dim_, 0.f));
      generate_centroids(train_data, train_size, train_dim, centroids_);
      delete[] train_data;
      train_data = nullptr;

      primary_.clear();
      primary_.resize(this->npoints);
      residual_.clear();
      if constexpr (UseResidualBits)
        residual_.resize(this->npoints);

      
      std::ifstream reader(data_bin, std::ios::binary);
      if (!reader)
        throw std::runtime_error("Failed to open data_bin: " + data_bin);
      reader.seekg(sizeof(uint32_t) * 2);  

      const size_t BLOCK = std::min<size_t>(kBlockMax, npts);
      std::unique_ptr<T[]> buf_T(new T[BLOCK * dim_]);
      std::unique_ptr<float[]> buf_f(new float[BLOCK * dim_]);

      const size_t nblocks = DIV_ROUND_UP(npts, BLOCK);
      for (size_t b = 0; b < nblocks; ++b) {
        const size_t start = b * BLOCK;
        const size_t end = std::min((b + 1) * BLOCK, npts);
        const size_t cur = end - start;

        reader.read(reinterpret_cast<char *>(buf_T.get()), sizeof(T) * cur * dim_);
        pipeann::convert_types<T, float>(buf_T.get(), buf_f.get(), cur, dim_);

#pragma omp parallel for
        for (int64_t j = 0; j < static_cast<int64_t>(cur); ++j) {
          const float *v = buf_f.get() + j * dim_;
          const uint32_t id = static_cast<uint32_t>(start + j);

          
          encode_level(v, dim_,
                       centroids_[0].data(),  
                       PrimaryBits, primary_[id].bias, primary_[id].scale, primary_[id].codes);

          
          if constexpr (UseResidualBits) {
            std::vector<float> residual(dim_);
            for (size_t d = 0; d < dim_; ++d) {
              const float code_val = decode_scalar(primary_[id].codes, PrimaryBits, d);
              const float recon_p = primary_[id].bias + primary_[id].scale * code_val;
              const float recon0 = centroids_[0][d] + recon_p;
              residual[d] = v[d] - recon0;
            }
            encode_level(residual.data(), dim_,
                         nullptr,  
                         ResidualBits, residual_[id].bias, residual_[id].scale, residual_[id].codes);
          }
        }
      }

      
      save_primary(index_prefix.c_str());
      if constexpr (UseResidualBits)
        save_residual(index_prefix.c_str());
      reconstruct_compressed_data(index_prefix.c_str());
      // exit(-1);
    }
    void reconstruct_compressed_data(const std::string &index_prefix) {
      const std::string out_path = index_prefix + "_compress_data.bin";
      LOG(INFO) << "Writing reconstructed (un-normalized) LVQ vectors to " << out_path;

      std::ofstream writer(out_path, std::ios::binary);
      if (!writer)
        throw std::runtime_error("Failed to open " + out_path);

      
      int32_t npts_s32 = static_cast<int32_t>(this->npoints);
      int32_t ndims_s32 = static_cast<int32_t>(dim_);
      writer.write(reinterpret_cast<char *>(&npts_s32), sizeof(int32_t));
      writer.write(reinterpret_cast<char *>(&ndims_s32), sizeof(int32_t));

      
#pragma omp parallel
      {
        std::vector<float> local_recon(dim_);
        std::vector<float> local_res(dim_);

#pragma omp for schedule(dynamic, 128)
        for (int64_t i = 0; i < (int64_t) this->npoints; i++) {
          const auto &P = primary_[i];
          const auto &center = centroids_[0];

          
          for (size_t d = 0; d < dim_; d++) {
            float code_val = decode_scalar(P.codes, PrimaryBits, d);
            float val = P.bias + P.scale * code_val;
            local_recon[d] = center[d] + val;
          }

          
          if constexpr (UseResidualBits) {
            const auto &R = residual_[i];
            for (size_t d = 0; d < dim_; d++) {
              float code_val = decode_scalar(R.codes, ResidualBits, d);
              float val = R.bias + R.scale * code_val;
              local_res[d] = val;
            }
            for (size_t d = 0; d < dim_; d++) {
              local_recon[d] += local_res[d];
            }
          }

#pragma omp critical
          {
            writer.write(reinterpret_cast<char *>(local_recon.data()), sizeof(float) * dim_);
          }
        }
      }

      writer.close();
      LOG(INFO) << "Reconstruction finished. Output written to: " << out_path;
    }

    
    void load(const char *index_prefix) override {
      
      load_primary(index_prefix);

      
      if constexpr (UseResidualBits) {
        load_residual(index_prefix);
      } else {
        residual_.clear();
      }
    }

    
    
    // recon = center + (bias_p + scale_p * code_p) + (bias_r + scale_r * code_r)
    float compute_distance_sample(const float *query, uint32_t id) const {
      const auto &P = primary_[id];
      const bool has_res = (UseResidualBits);// && (id < residual_.size());
      const LVQPointLevel *R = has_res ? &residual_[id] : nullptr;

      float dist = 0.0f;
      for (size_t d = 0; d < dim_; ++d) {
        const float code_p = decode_scalar(P.codes, PrimaryBits, d);
        float recon = centroids_[0][d] + (P.bias + P.scale * code_p);

        if (R) {
          const float code_r = decode_scalar(R->codes, ResidualBits, d);
          recon += (R->bias + R->scale * code_r);
        }

        const float diff = query[d] - recon;
        dist += diff * diff;
      }
      return dist;
    }

    __attribute__((target("avx512vbmi")))
    float compute_distance_avx512(const float *query, uint32_t id) const {
      const LVQPointLevel &P = primary_[id];
      const bool has_res = (UseResidualBits) && (id < residual_.size());
      const LVQPointLevel *R = has_res ? &residual_[id] : nullptr;
      return compute_distance_avx512(query, id, P, R);
    }

    __attribute__((target("avx512vbmi")))
    float compute_distance_avx512(const float *query, uint32_t id, const LVQPointLevel &P, const LVQPointLevel *R = nullptr) const {

      const float *centroid = centroids_[0].data();
      const size_t dim = this->dim_;

      __m512 acc = _mm512_setzero_ps();
      const size_t step = 16;  
      const size_t n_blk = dim / step;
      const size_t tail = dim % step;

      const __m512 scale_p = _mm512_set1_ps(P.scale);
      const __m512 bias_p = _mm512_set1_ps(P.bias);
      __m512 scale_r, bias_r;
      if (R != nullptr) {
        scale_r = _mm512_set1_ps(R->scale);
        bias_r = _mm512_set1_ps(R->bias);
      }

      const __m512i low_mask = _mm512_set1_epi8(0x0F);
      const __m512i shuf_idx = _mm512_set_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5,
                                               6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                                               12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);

      for (size_t b = 0; b < n_blk; ++b) {
        const size_t off_p = (PrimaryBits == 8) ? b * 16 : b * 8;
        const size_t off_r = (ResidualBits == 8) ? b * 16 : b * 8;

        // -------- Decode primary --------
        __m512 codes_p;
        if constexpr (PrimaryBits == 8) {
          __m128i raw = _mm_loadu_si128((__m128i *) (P.codes.data() + off_p));   
          __m512i epi32 = _mm512_cvtepu8_epi32(raw);                             
          codes_p = _mm512_cvtepi32_ps(epi32);                                   
        } else if constexpr (PrimaryBits == 4) {                                 
          
          __m128i raw8 = _mm_loadl_epi64((__m128i const*)(P.codes.data() + off_p));

          
          const __m128i low_mask_128 = _mm_set1_epi8(0x0F);

          
          __m128i lo = _mm_and_si128(raw8, low_mask_128);

          
          // (B0>>4), (B1>>4), ... 
          
          
          __m128i hi = _mm_and_si128(_mm_srli_epi16(raw8, 4), low_mask_128);

          
          // lo = [L0, L1, L2, L3, ...]
          // hi = [H0, H1, H2, H3, ...]
          
          // [L0, H0, L1, H1, L2, H2, ...]
          
          __m128i codes_u8 = _mm_unpacklo_epi8(lo, hi);

          
          __m512i epi32 = _mm512_cvtepu8_epi32(codes_u8);

          
          codes_p = _mm512_cvtepi32_ps(epi32);
        }

        // reconstruct primary
        __m512 recon = _mm512_fmadd_ps(scale_p, codes_p, bias_p);  

        // -------- Decode residual --------
        if (R != nullptr) {
          __m512 codes_r;
          if constexpr (ResidualBits == 8) {
            __m128i raw = _mm_loadu_si128((__m128i *) (R->codes.data() + off_r));
            __m512i epi32 = _mm512_cvtepu8_epi32(raw);
            codes_r = _mm512_cvtepi32_ps(epi32);
          } else if constexpr (ResidualBits == 4) {
            
            __m128i raw8 = _mm_loadl_epi64((__m128i const*)(R->codes.data() + off_r));
  
            
            const __m128i low_mask_128 = _mm_set1_epi8(0x0F);
  
            
            __m128i lo = _mm_and_si128(raw8, low_mask_128);
  
            
            // (B0>>4), (B1>>4), ... 
            
            
            __m128i hi = _mm_and_si128(_mm_srli_epi16(raw8, 4), low_mask_128);
  
            
            // lo = [L0, L1, L2, L3, ...]
            // hi = [H0, H1, H2, H3, ...]
            
            // [L0, H0, L1, H1, L2, H2, ...]
            
            __m128i codes_u8 = _mm_unpacklo_epi8(lo, hi);
  
            
            __m512i epi32 = _mm512_cvtepu8_epi32(codes_u8);
  
            
            codes_p = _mm512_cvtepi32_ps(epi32);
          }
          __m512 recon_r = _mm512_fmadd_ps(scale_r, codes_r, bias_r);
          recon = _mm512_add_ps(recon, recon_r);
        }

        
        __m512 c = _mm512_loadu_ps(centroid + b * 16);
        recon = _mm512_add_ps(recon, c);

        __m512 q = _mm512_loadu_ps(query + b * 16);
        __m512 diff = _mm512_sub_ps(q, recon);
        acc = _mm512_fmadd_ps(diff, diff, acc);
      }

      
      float tail_sum = 0.0f;
      for (size_t d = n_blk * step; d < dim; ++d) {
        float recon = centroids_[0][d] + (P.bias + P.scale * decode_scalar(P.codes, PrimaryBits, d));
        if (R != nullptr) {
          recon += (R->bias + R->scale * decode_scalar(R->codes, ResidualBits, d));
        }
        float df = query[d] - recon;
        tail_sum += df * df;
      }

      return _mm512_reduce_add_ps(acc) + tail_sum;
    }
    __attribute__((target("avx512vbmi")))
    float compute_distance_avx512_only_primary(const float *query, uint32_t id) const {
      const LVQPointLevel &P = primary_[id];

      const float *centroid = centroids_[0].data();
      const size_t dim = this->dim_;

      __m512 acc = _mm512_setzero_ps();
      const size_t step = 16;  
      const size_t n_blk = dim / step;
      const size_t tail = dim % step;

      const __m512 scale_p = _mm512_set1_ps(P.scale);
      const __m512 bias_p = _mm512_set1_ps(P.bias);


      const __m512i low_mask = _mm512_set1_epi8(0x0F);
      const __m512i shuf_idx = _mm512_set_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5,
                                               6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                                               12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);

      for (size_t b = 0; b < n_blk; ++b) {
        const size_t off_p = (PrimaryBits == 8) ? b * 16 : b * 8;

        // -------- Decode primary --------
        __m512 codes_p;
        if constexpr (PrimaryBits == 8) {
          __m128i raw = _mm_loadu_si128((__m128i *) (P.codes.data() + off_p));   
          __m512i epi32 = _mm512_cvtepu8_epi32(raw);                             
          codes_p = _mm512_cvtepi32_ps(epi32);                                   
        } else if constexpr (PrimaryBits == 4) {                                 
          
          __m128i raw8 = _mm_loadl_epi64((__m128i const*)(P.codes.data() + off_p));

          
          const __m128i low_mask_128 = _mm_set1_epi8(0x0F);

          
          __m128i lo = _mm_and_si128(raw8, low_mask_128);

          
          // (B0>>4), (B1>>4), ... 
          
          
          __m128i hi = _mm_and_si128(_mm_srli_epi16(raw8, 4), low_mask_128);

          
          // lo = [L0, L1, L2, L3, ...]
          // hi = [H0, H1, H2, H3, ...]
          
          // [L0, H0, L1, H1, L2, H2, ...]
          
          __m128i codes_u8 = _mm_unpacklo_epi8(lo, hi);

          
          __m512i epi32 = _mm512_cvtepu8_epi32(codes_u8);

          
          codes_p = _mm512_cvtepi32_ps(epi32);
        }

        // reconstruct primary
        __m512 recon = _mm512_fmadd_ps(scale_p, codes_p, bias_p);  


        
        __m512 c = _mm512_loadu_ps(centroid + b * 16);
        recon = _mm512_add_ps(recon, c);

        __m512 q = _mm512_loadu_ps(query + b * 16);
        __m512 diff = _mm512_sub_ps(q, recon);
        acc = _mm512_fmadd_ps(diff, diff, acc);
      }

      
      float tail_sum = 0.0f;
      for (size_t d = n_blk * step; d < dim; ++d) {
        float recon = centroids_[0][d] + (P.bias + P.scale * decode_scalar(P.codes, PrimaryBits, d));
        float df = query[d] - recon;
        tail_sum += df * df;
      }

      return _mm512_reduce_add_ps(acc) + tail_sum;
    }
    
    void compute_dists(const uint32_t query_id, const uint32_t *ids, const uint64_t n_ids, float *dists_out,
                       uint8_t * /*aligned_scratch*/) override {
      
      lvq_mu.lock_shared();
      
      auto reconstruct_point = [&](uint32_t idx, float *out) {
        const auto &P = primary_[idx];
        const auto &C0 = centroids_[0];

        
        for (size_t d = 0; d < dim_; ++d) {
          float code = decode_scalar(P.codes, PrimaryBits, d);
          float val = P.bias + P.scale * code;
          out[d] = C0[d] + val;
        }

        
        if constexpr (UseResidualBits) {
          const auto &R = residual_[idx];
          for (size_t d = 0; d < dim_; ++d) {
            float code = decode_scalar(R.codes, ResidualBits, d);
            out[d] += (R.bias + R.scale * code);
          }
        } 
      };
      std::vector<float> query(dim_);
      reconstruct_point(query_id, query.data());
      
      for (int64_t i = 0; i < (int64_t) n_ids; i++) {
        uint32_t id = ids[i];
        
        dists_out[i] = compute_distance_avx512(query.data(), id);
      }
      lvq_mu.unlock_shared();
    }
    void compute_dists_only_primary(const uint32_t query_id, const uint32_t *ids, const uint64_t n_ids, float *dists_out,
                       uint8_t * /*aligned_scratch*/) {
      
      lvq_mu.lock_shared();
      
      auto reconstruct_point = [&](uint32_t idx, float *out) {
        const auto &P = primary_[idx];
        const auto &C0 = centroids_[0];

        
        for (size_t d = 0; d < dim_; ++d) {
          float code = decode_scalar(P.codes, PrimaryBits, d);
          float val = P.bias + P.scale * code;
          out[d] = C0[d] + val;
        }
      };
      std::vector<float> query(dim_);
      reconstruct_point(query_id, query.data());
      
      for (int64_t i = 0; i < (int64_t) n_ids; i++) {
        uint32_t id = ids[i];
        dists_out[i] = compute_distance_avx512_only_primary(query.data(), id);
      }
      lvq_mu.unlock_shared();
    }
    AbstractNeighbor<T> *shuffle(const libcuckoo::cuckoohash_map<uint32_t, uint32_t> &rev_id_map, uint64_t new_npoints,
                                 uint32_t nthreads) override {
      auto *new_handler = new LVQNeighbor<T, PRIMARY_BITS, RESIDUAL_BITS>();
      new_handler->npoints = new_npoints;
      new_handler->dim_ = this->dim_;
      new_handler->centroids_ = this->centroids_;

      new_handler->primary_.resize(new_npoints);
      if constexpr (UseResidualBits)
        new_handler->residual_.resize(new_npoints);

#pragma omp parallel for num_threads(nthreads)
      for (uint64_t i = 0; i < new_npoints; i++) {
        auto old_id = rev_id_map.find(i);
        new_handler->primary_[i] = this->primary_[old_id];
        if constexpr (UseResidualBits)
          new_handler->residual_[i] = this->residual_[old_id];
      }

      return reinterpret_cast<AbstractNeighbor<T> *>(new_handler);
    }

    void insert(T *point, uint32_t loc) override {
      std::vector<float> fp_vec(dim_);
      for (uint32_t i = 0; i < dim_; i++)
        fp_vec[i] = static_cast<float>(point[i]);

      const auto &center = centroids_[0];  

      
      LVQPointLevel P_temp;
      encode_level(fp_vec.data(), dim_, center.data(), PrimaryBits, P_temp.bias, P_temp.scale, P_temp.codes);

      LVQPointLevel R_temp;
      if constexpr (UseResidualBits) {
        std::vector<float> residual(dim_);
        for (size_t d = 0; d < dim_; d++) {
          float code_val = decode_scalar(P_temp.codes, PrimaryBits, d);
          float recon_p = P_temp.bias + P_temp.scale * code_val;
          residual[d] = fp_vec[d] - (center[d] + recon_p);
        }
        encode_level(residual.data(), dim_, nullptr, ResidualBits, R_temp.bias, R_temp.scale, R_temp.codes);
      }

      
      {
        lvq_mu.lock();
        if (loc >= primary_.size())
          primary_.resize(static_cast<size_t>(1.5 * (loc + 1)));
        primary_[loc] = std::move(P_temp);

        if constexpr (UseResidualBits) {
          if (loc >= residual_.size())
            residual_.resize(static_cast<size_t>(1.5 * (loc + 1)));
          residual_[loc] = std::move(R_temp);
        }

        this->npoints = std::max<uint64_t>(this->npoints, loc + 1);
        lvq_mu.unlock();
      }
    }

    LVQPointLevel insert_primary_and_get_residual(T *point, uint32_t loc) {
      std::vector<float> fp_vec(dim_);
      for (uint32_t i = 0; i < dim_; i++)
        fp_vec[i] = static_cast<float>(point[i]);

      const auto &center = centroids_[0];  

      
      LVQPointLevel P_temp;
      encode_level(fp_vec.data(), dim_, center.data(), PrimaryBits, P_temp.bias, P_temp.scale, P_temp.codes);

      LVQPointLevel R_temp;
      
      std::vector<float> residual(dim_);
      for (size_t d = 0; d < dim_; d++) {
        float code_val = decode_scalar(P_temp.codes, PrimaryBits, d);
        float recon_p = P_temp.bias + P_temp.scale * code_val;
        residual[d] = fp_vec[d] - (center[d] + recon_p);
      }
      encode_level(residual.data(), dim_, nullptr, ResidualBits, R_temp.bias, R_temp.scale, R_temp.codes);

      
      {
        lvq_mu.lock();
        if (loc >= primary_.size())
          primary_.resize(static_cast<size_t>(1.5 * (loc + 1)));
        primary_[loc] = std::move(P_temp);

        this->npoints = std::max<uint64_t>(this->npoints, loc + 1);
        lvq_mu.unlock();
      }
      return R_temp;
    }

    
    void batch_insert(const std::vector<T *> &points, const std::vector<uint32_t> &locs) {
      if (points.size() != locs.size())
        throw std::runtime_error("batch_insert: points and locs size mismatch.");

      const size_t n_batch = points.size();
      const uint32_t max_loc = *std::max_element(locs.begin(), locs.end());
      const auto &center = centroids_[0];
      lvq_mu.lock();
      
      if (primary_.size() < max_loc + 1) {
        primary_.resize(max_loc + 1);
        if constexpr (UseResidualBits)
          residual_.resize(max_loc + 1);
      }
      this->npoints = std::max<uint64_t>(this->npoints, max_loc + 1);

      
#pragma omp parallel for schedule(dynamic, 64)
      for (int64_t i = 0; i < (int64_t) n_batch; i++) {
        const T *point = points[i];
        const uint32_t loc = locs[i];

        
        std::vector<float> fp_vec(dim_);
        for (uint32_t d = 0; d < dim_; d++)
          fp_vec[d] = static_cast<float>(point[d]);

        
        encode_level(fp_vec.data(), dim_, center.data(), PrimaryBits, primary_[loc].bias, primary_[loc].scale,
                     primary_[loc].codes);

        
        if constexpr (UseResidualBits) {
          std::vector<float> residual(dim_);
          for (size_t d = 0; d < dim_; d++) {
            float code_val = decode_scalar(primary_[loc].codes, PrimaryBits, d);
            float recon_p = primary_[loc].bias + primary_[loc].scale * code_val;
            residual[d] = fp_vec[d] - (center[d] + recon_p);
          }
          encode_level(residual.data(), dim_, nullptr, ResidualBits, residual_[loc].bias, residual_[loc].scale,
                       residual_[loc].codes);
        }
      }
      lvq_mu.unlock();
    }

   private:
    
    static void encode_level(const float *vec, size_t dim,
                             const float *center,  
                             size_t bits, float &bias_out, float &scale_out, std::vector<uint8_t> &codes_out) {
      float minv = std::numeric_limits<float>::max();
      float maxv = std::numeric_limits<float>::lowest();

      for (size_t d = 0; d < dim; ++d) {
        const float shifted = vec[d] - (center ? center[d] : 0.f);
        minv = std::min(minv, shifted);
        maxv = std::max(maxv, shifted);
      }

      bias_out = minv;
      const float max_code = static_cast<float>((1u << bits) - 1u);
      scale_out = (maxv == minv) ? 1.0f : ((maxv - minv) / max_code);

      const size_t total_bytes = (dim * bits + 7) / 8;
      codes_out.assign(total_bytes, 0);

      for (size_t d = 0; d < dim; ++d) {
        const float shifted = vec[d] - (center ? center[d] : 0.f);
        const float norm = (shifted - bias_out) / scale_out;
        uint32_t code = static_cast<uint32_t>(std::round(std::clamp(norm, 0.0f, max_code)));

        const size_t bit_pos = d * bits;
        const size_t byte_pos = bit_pos / 8;
        const size_t bit_off = bit_pos % 8;

        codes_out[byte_pos] |= static_cast<uint8_t>(code << bit_off);
        if (bit_off + bits > 8) {
          codes_out[byte_pos + 1] |= static_cast<uint8_t>(code >> (8 - bit_off));
        }
      }
    }

    
    static inline float decode_scalar(const std::vector<uint8_t> &codes, size_t bits, size_t idx) {
      const size_t bit_pos = idx * bits;
      const size_t byte_pos = bit_pos / 8;
      const size_t bit_off = bit_pos % 8;

      uint32_t code = static_cast<uint32_t>(codes[byte_pos] >> bit_off);
      if (bit_off + bits > 8) {
        code |= static_cast<uint32_t>(codes[byte_pos + 1]) << (8 - bit_off);
      }
      code &= ((1u << bits) - 1u);
      return static_cast<float>(code);
    }

    
    static void generate_centroids(const float *train_data, size_t ntrain, size_t dim,
                                   std::vector<std::vector<float>> &cents) {
      if (ntrain == 0) {
        for (auto &c : cents)
          c.assign(dim, 0.f);
        return;
      }
      for (auto &c : cents)
        c.assign(dim, 0.f);

      for (size_t k = 0; k < ntrain; ++k) {
        const float *v = train_data + k * dim;
        for (size_t d = 0; d < dim; ++d)
          cents[0][d] += v[d];
      }
      for (size_t d = 0; d < dim; ++d)
        cents[0][d] /= static_cast<float>(ntrain);
    }

    
    static std::string primary_path(const char *prefix) {
      return std::string(prefix) + "_primary_lvq.bin";
    }
    static std::string residual_path(const char *prefix) {
      return std::string(prefix) + "_residual_lvq.bin";
    }

    
    void save_primary(const char *prefix) const {
      const std::string path = primary_path(prefix);
      std::ofstream w(path, std::ios::binary);
      if (!w)
        throw std::runtime_error("open fail: " + path);

      // header
      w.write((const char *) &this->npoints, sizeof(uint64_t));
      w.write((const char *) &dim_, sizeof(uint64_t));
      const size_t bits = PrimaryBits;
      w.write((const char *) &bits, sizeof(size_t));

      
      for (size_t d = 0; d < dim_; ++d)
        w.write((const char *) &centroids_[0][d], sizeof(float));

      
      const size_t code_bytes = (dim_ * PrimaryBits + 7) /
                                8;  
      for (const auto &p : primary_) {
        w.write((const char *) &p.bias, sizeof(float));
        w.write((const char *) &p.scale, sizeof(float));
        w.write((const char *) p.codes.data(), code_bytes);
      }
      w.close();
      LOG(INFO) << "Saved PRIMARY<" << PrimaryBits << "> LVQ: " << path;
    }

    void load_primary(const char *prefix) {
      const std::string path = primary_path(prefix);
      std::ifstream r(path, std::ios::binary);
      if (!r)
        throw std::runtime_error("open fail: " + path);

      uint64_t npts = 0, dim = 0;
      size_t bits = 0;
      r.read((char *) &npts, sizeof(uint64_t));
      r.read((char *) &dim, sizeof(uint64_t));
      r.read((char *) &bits, sizeof(size_t));
      if (bits != PrimaryBits){
        LOG(ERROR) << "PrimaryBits mismatch in " << path << " " << bits << " " << PrimaryBits;
        throw std::runtime_error("PrimaryBits mismatch in " + path);
      }

      this->npoints = npts;
      dim_ = dim;

      centroids_.assign(kNumCenters, std::vector<float>(dim_, 0.f));
      for (size_t d = 0; d < dim_; ++d)
        r.read((char *) &centroids_[0][d], sizeof(float));

      primary_.assign(this->npoints, LVQPointLevel{});
      const size_t code_bytes = (dim_ * PrimaryBits + 7) / 8;
      for (auto &p : primary_) {
        r.read((char *) &p.bias, sizeof(float));
        r.read((char *) &p.scale, sizeof(float));
        p.codes.resize(code_bytes);
        r.read((char *) p.codes.data(), code_bytes);
      }
      r.close();
      LOG(INFO) << "Loaded PRIMARY LVQ: " << path;
    }

    
    void save_residual(const char *prefix) const {
      static_assert(ResidualBits > 0, "ResidualBits must be > 0 to save residual.");
      const std::string path = residual_path(prefix);
      std::ofstream w(path, std::ios::binary);
      if (!w)
        throw std::runtime_error("open fail: " + path);

      // header
      w.write((const char *) &this->npoints, sizeof(uint64_t));
      w.write((const char *) &dim_, sizeof(uint64_t));
      const size_t bits = ResidualBits;
      w.write((const char *) &bits, sizeof(size_t));

      
      const size_t code_bytes = (dim_ * ResidualBits + 7) / 8;
      for (const auto &p : residual_) {
        w.write((const char *) &p.bias, sizeof(float));
        w.write((const char *) &p.scale, sizeof(float));
        w.write((const char *) p.codes.data(), code_bytes);
      }
      w.close();
      LOG(INFO) << "Saved RESIDUAL<" << ResidualBits << "> LVQ: " << path;
    }

    void load_residual(const char *prefix) {
      static_assert(ResidualBits > 0, "ResidualBits must be > 0 to load residual.");
      const std::string path = residual_path(prefix);
      std::ifstream r(path, std::ios::binary);
      if (!r)
        throw std::runtime_error("open fail: " + path);

      uint64_t npts = 0, dim = 0;
      size_t bits = 0;
      r.read((char *) &npts, sizeof(uint64_t));
      r.read((char *) &dim, sizeof(uint64_t));
      r.read((char *) &bits, sizeof(size_t));
      if (bits != ResidualBits)
        throw std::runtime_error("ResidualBits mismatch in " + path);

      if (npts != this->npoints || dim != dim_)
        throw std::runtime_error("Residual header mismatches primary header.");

      residual_.assign(this->npoints, LVQPointLevel{});
      const size_t code_bytes = (dim_ * ResidualBits + 7) / 8;
      for (auto &p : residual_) {
        r.read((char *) &p.bias, sizeof(float));
        r.read((char *) &p.scale, sizeof(float));
        p.codes.resize(code_bytes);
        r.read((char *) p.codes.data(), code_bytes);
      }
      r.close();
      LOG(INFO) << "Loaded RESIDUAL LVQ: " << path;
    }
  };
}  // namespace pipeann
#else
namespace pipeann {
  template<typename T>
  class LVQNeighbor : public AbstractNeighbor<T> {
   public:
    LVQNeighbor<T>() {
      LOG(ERROR) << "LVQNeighbor requires AVX512 support.";
      exit(-1);
    }

    static std::string get_name() {
      return "LVQNeighbor";
    }
    // rev_id_map: new_id -> old_id.
    AbstractNeighbor<T> *shuffle(const libcuckoo::cuckoohash_map<uint32_t, uint32_t> &rev_id_map, uint64_t new_npoints,
                                 uint32_t nthreads) {
      return this;
    }
    void initialize_query(const T *query, QueryBuffer<T> *query_buf) {
    }
    // Compute dists using assymetric distance computation.
    void compute_dists(QueryBuffer<T> *query_buf, const uint32_t *ids, const uint64_t n_ids) {
    }
    // Compute dists using PQ all-to-all.
    void compute_dists(const uint32_t query_id, const uint32_t *ids, const uint64_t n_ids, float *dists_out,
                       uint8_t *aligned_scratch) {
    }
    // Load the neighbor data (e.g., PQ) from disk.
    void load(const char *index_prefix) {
    }
    // Save the neighbor data (e.g., PQ) to disk.
    void save(const char *index_prefix) {
    }
    // Call load after build to load the neighbors.
    void build(const std::string &index_prefix, const std::string &data_bin, uint32_t bytes_per_nbr) {
    }
    void insert(T *point, uint32_t loc) {
    }

    uint64_t npoints = 0;
  };
}  // namespace pipeann
#endif
