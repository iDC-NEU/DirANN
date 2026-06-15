#include <omp.h>
#include <cstring>
#include <ctime>
#include "utils/timer.h"
#include "utils/log.h"
#include "utils.h"

#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include "distance.h"

template<class T>
void convert(char *input_path, char *output_path) {
  int num_i32, dim_i32;
  std::ifstream reader(input_path, std::ios::binary | std::ios::ate);
  reader.seekg(0, std::ios::beg);
  reader.read((char *) &num_i32, sizeof(int));
  reader.read((char *) &dim_i32, sizeof(int));
  printf("num_i32=%d, dim_i32=%d\n", num_i32, dim_i32);
  size_t num = (size_t)num_i32;
  size_t dim = (size_t)dim_i32;
  size_t total = num * dim;
  T *data = new T[total];
  reader.read((char *) data, total * sizeof(T));
  for (size_t i = 0; i < num; ++i) {
    auto norm = pipeann::compute_l2_norm(data + (i * dim), dim);
    for (size_t j = 0; j < dim; ++j) {
      data[i * dim + j] /= norm;
    }
  }
  pipeann::save_bin<T>((std::string(output_path)).c_str(), data, num, dim);
  printf("output_path=%s\n", output_path);
}

int main(int argc, char **argv) {
  if (argc < 4) {
    std::cout << "Correct usage: " << argv[0] << " <type[uint8/float]> <originfile> <outputfile>" << std::endl;
    exit(-1);
  }

  int arg_no = 1;
  char *type = argv[arg_no++];
  char *base_data_file = argv[arg_no++];
  char *output_file = argv[arg_no++];

  if (strcmp(type, "uint8") == 0) {
    convert<uint8_t>(base_data_file, output_file);
  } else if (strcmp(type, "int8") == 0) {
    convert<int8_t>(base_data_file, output_file);
  } else if (strcmp(type, "float") == 0) {
    convert<float>(base_data_file, output_file);
  } else {
    std::cout << "Unknown type: " << type << std::endl;
    exit(-1);
  }
}