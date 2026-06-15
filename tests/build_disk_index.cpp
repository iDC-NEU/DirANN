#include <cstdint>
#include "nbr/abstract_nbr.h"
#include "nbr/pq_nbr.h"
// #include "nbr/rabitq_nbr.h"
#include "nbr/lvq_nbr.h"
#include "omp.h"

#include "aux_utils.h"
#include "utils.h"

int main(int argc, char **argv) {
  if (argc < 11) {
    std::cout << "Usage: " << argv[0]
              << " <data_type (float/int8/uint8)>  <data_file.bin>"
                 " <index_prefix_path> <R>  <L>  <PQ_bytes>  <M>  <T>"
                 " <similarity metric (cosine/l2) case sensitive> <nbr_type (pq/rabitq/lvq)>."
                 " See README for more information on parameters."
              << std::endl;
  } else {
    std::string dist_metric(argv[9]);

    pipeann::Metric m = dist_metric == "cosine" ? pipeann::Metric::COSINE : pipeann::Metric::L2;
    if (dist_metric != "l2" && m == pipeann::Metric::L2) {
      std::cout << "Metric " << dist_metric << " is not supported. Using L2" << std::endl;
    }

    std::string nbr_type = argv[10];

    if (std::string(argv[1]) == std::string("float")) {
      pipeann::AbstractNeighbor<float> *nbr_handler = nullptr;
      if (nbr_type == "rabitq") {
        // nbr_handler = new pipeann::RaBitQNeighbor<float>();
      } else if(nbr_type == "lvq") {
        nbr_handler = new pipeann::LVQNeighbor<float, PRIMARY_BITS, RESIDUAL_BITS>();
      } else {
        nbr_handler = new pipeann::PQNeighbor<float>();
      }
      pipeann::build_disk_index<float>(argv[2], argv[3], std::stoi(argv[4]), std::stoi(argv[5]), std::stoi(argv[7]),
                                       std::stoi(argv[8]), std::stoi(argv[6]), m, nullptr, nbr_handler);
    } else if (std::string(argv[1]) == std::string("int8")) {
      pipeann::AbstractNeighbor<int8_t> *nbr_handler = nullptr;
      if (nbr_type == "rabitq") {
        // nbr_handler = new pipeann::RaBitQNeighbor<int8_t>();
      } else if(nbr_type == "lvq") {
        nbr_handler = new pipeann::LVQNeighbor<int8_t, PRIMARY_BITS, RESIDUAL_BITS>();
      } else {
        nbr_handler = new pipeann::PQNeighbor<int8_t>();
      }
      pipeann::build_disk_index<int8_t>(argv[2], argv[3], std::stoi(argv[4]), std::stoi(argv[5]), std::stoi(argv[7]),
                                        std::stoi(argv[8]), std::stoi(argv[6]), m, nullptr, nbr_handler);
    } else if (std::string(argv[1]) == std::string("uint8")) {
      pipeann::AbstractNeighbor<uint8_t> *nbr_handler = nullptr;
      if (nbr_type == "rabitq") {
        // nbr_handler = new pipeann::RaBitQNeighbor<uint8_t>();
      } else if(nbr_type == "lvq") {
        nbr_handler = new pipeann::LVQNeighbor<uint8_t, PRIMARY_BITS, RESIDUAL_BITS>();
      } else {
        nbr_handler = new pipeann::PQNeighbor<uint8_t>();
      }
      pipeann::build_disk_index<uint8_t>(argv[2], argv[3], std::stoi(argv[4]), std::stoi(argv[5]), std::stoi(argv[7]),
                                         std::stoi(argv[8]), std::stoi(argv[6]), m, nullptr, nbr_handler);
    } else {
      std::cout << "Error. wrong file type" << std::endl;
    }
  }
}
