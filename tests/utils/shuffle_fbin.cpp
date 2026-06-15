#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <algorithm>
#include <cstdint>

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cout << "Usage: shuffle_fbin <input.fbin> <output.fbin>" << std::endl;
        return -1;
    }

    const char* input_file = argv[1];
    const char* output_file = argv[2];

    std::ifstream in(input_file, std::ios::binary);
    if (!in) {
        std::cerr << "Cannot open input file\n";
        return -1;
    }

    int32_t num_points, dim;

    
    in.read(reinterpret_cast<char*>(&num_points), sizeof(int32_t));
    in.read(reinterpret_cast<char*>(&dim), sizeof(int32_t));

    std::cout << "N=" << num_points << ", D=" << dim << std::endl;

    
    std::vector<float> data(num_points * dim);
    in.read(reinterpret_cast<char*>(data.data()), data.size() * sizeof(float));
    in.close();

    
    std::vector<int> indices(num_points);
    for (int i = 0; i < num_points; i++) {
        indices[i] = i;
    }

    
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(indices.begin(), indices.end(), g);

    
    std::ofstream out(output_file, std::ios::binary);
    if (!out) {
        std::cerr << "Cannot open output file\n";
        return -1;
    }

    
    out.write(reinterpret_cast<char*>(&num_points), sizeof(int32_t));
    out.write(reinterpret_cast<char*>(&dim), sizeof(int32_t));

    
    std::vector<float> buffer(dim);
    for (int i = 0; i < num_points; i++) {
        int idx = indices[i];
        float* src = data.data() + idx * dim;

        
        std::copy(src, src + dim, buffer.begin());

        out.write(reinterpret_cast<char*>(buffer.data()), dim * sizeof(float));
    }

    out.close();

    std::cout << "Shuffle done." << std::endl;
    return 0;
}