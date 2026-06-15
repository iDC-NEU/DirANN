#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <queue>
#include <stdexcept>
#include <string>
#include <vector>

struct GraphHeader {
  uint64_t graph_bytes = 0;
  uint32_t width = 0;
  uint32_t entry_point = 0;
  uint64_t frozen_points = 0;
};

struct GraphData {
  GraphHeader header;
  std::vector<std::vector<uint32_t>> adj;
  uint64_t edge_count = 0;
  uint64_t invalid_edge_count = 0;
};

static uint64_t get_file_size(std::ifstream &in) {
  const std::streampos cur = in.tellg();
  in.seekg(0, std::ios::end);
  const std::streampos end = in.tellg();
  in.seekg(cur, std::ios::beg);
  if (end < 0) {
    throw std::runtime_error("failed to get file size");
  }
  return static_cast<uint64_t>(end);
}

static bool read_exact(std::ifstream &in, char *ptr, size_t n) {
  in.read(ptr, static_cast<std::streamsize>(n));
  return static_cast<size_t>(in.gcount()) == n;
}

static GraphData load_graph(const std::string &path, uint64_t offset) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("cannot open file: " + path);
  }

  const uint64_t file_size = get_file_size(in);
  if (offset >= file_size) {
    throw std::runtime_error("offset is beyond file size");
  }

  in.seekg(static_cast<std::streamoff>(offset), std::ios::beg);

  GraphData g;
  if (!read_exact(in, reinterpret_cast<char *>(&g.header.graph_bytes), sizeof(uint64_t)) ||
      !read_exact(in, reinterpret_cast<char *>(&g.header.width), sizeof(uint32_t)) ||
      !read_exact(in, reinterpret_cast<char *>(&g.header.entry_point), sizeof(uint32_t)) ||
      !read_exact(in, reinterpret_cast<char *>(&g.header.frozen_points), sizeof(uint64_t))) {
    throw std::runtime_error("failed to read graph header");
  }

  if (g.header.graph_bytes < 24) {
    throw std::runtime_error("invalid graph size in header (must be >= 24)");
  }
  if (offset + g.header.graph_bytes > file_size) {
    throw std::runtime_error("header graph size exceeds physical file size");
  }

  uint64_t bytes_read = 24;
  while (bytes_read < g.header.graph_bytes) {
    uint32_t degree = 0;
    if (!read_exact(in, reinterpret_cast<char *>(&degree), sizeof(uint32_t))) {
      throw std::runtime_error("unexpected EOF while reading degree");
    }
    bytes_read += sizeof(uint32_t);

    const uint64_t nbr_bytes = static_cast<uint64_t>(degree) * sizeof(uint32_t);
    if (bytes_read + nbr_bytes > g.header.graph_bytes) {
      throw std::runtime_error("corrupted graph: degree section exceeds declared graph bytes");
    }

    std::vector<uint32_t> nbrs(degree);
    if (degree > 0 &&
        !read_exact(in, reinterpret_cast<char *>(nbrs.data()), static_cast<size_t>(nbr_bytes))) {
      throw std::runtime_error("unexpected EOF while reading neighbor list");
    }

    g.edge_count += degree;
    g.adj.emplace_back(std::move(nbrs));
    bytes_read += nbr_bytes;
  }

  if (bytes_read != g.header.graph_bytes) {
    throw std::runtime_error("parsed bytes do not match graph bytes in header");
  }

  for (const auto &nbrs : g.adj) {
    for (uint32_t v : nbrs) {
      if (v >= g.adj.size()) {
        g.invalid_edge_count++;
      }
    }
  }

  return g;
}

static void print_usage(const char *prog) {
  std::cerr << "Usage: " << prog << " <graph_file> [offset]\n"
            << "  <graph_file> : index graph file path\n"
            << "  [offset]     : graph start offset in bytes (default: 0)\n";
}

int main(int argc, char **argv) {
  if (argc < 2 || argc > 3) {
    print_usage(argv[0]);
    return 1;
  }

  const std::string graph_path = argv[1];
  uint64_t offset = 0;

  if (argc == 3) {
    try {
      size_t pos = 0;
      const std::string s = argv[2];
      offset = std::stoull(s, &pos);
      if (pos != s.size()) {
        throw std::runtime_error("invalid offset");
      }
    } catch (const std::exception &) {
      std::cerr << "Invalid offset: " << argv[2] << "\n";
      return 1;
    }
  }

  try {
    GraphData g = load_graph(graph_path, offset);
    const uint64_t n = g.adj.size();

    if (n == 0) {
      std::cout << "nodes: 0\n";
      std::cout << "edges: 0\n";
      std::cout << "entry_point: " << g.header.entry_point << "\n";
      std::cout << "unreachable_nodes: 0\n";
      std::cout << "unreachable_ratio: 0\n";
      return 0;
    }

    if (g.header.entry_point >= n) {
      std::cerr << "Error: entry point " << g.header.entry_point
                << " is out of range [0, " << (n - 1) << "]\n";
      return 2;
    }

    std::vector<uint8_t> visited(n, 0);
    std::queue<uint32_t> q;

    visited[g.header.entry_point] = 1;
    q.push(g.header.entry_point);

    uint64_t reached = 0;
    while (!q.empty()) {
      const uint32_t u = q.front();
      q.pop();
      reached++;

      for (uint32_t v : g.adj[u]) {
        if (v >= n) {
          continue;
        }
        if (!visited[v]) {
          visited[v] = 1;
          q.push(v);
        }
      }
    }

    const uint64_t unreachable = n - reached;
    const double ratio = static_cast<double>(unreachable) / static_cast<double>(n);

    std::cout << "graph_file: " << graph_path << "\n";
    std::cout << "offset: " << offset << "\n";
    std::cout << "nodes: " << n << "\n";
    std::cout << "edges: " << g.edge_count << "\n";
    std::cout << "header_width: " << g.header.width << "\n";
    std::cout << "entry_point: " << g.header.entry_point << "\n";
    std::cout << "frozen_points(header): " << g.header.frozen_points << "\n";
    std::cout << "invalid_edges(ignored): " << g.invalid_edge_count << "\n";
    std::cout << "reachable_nodes: " << reached << "\n";
    std::cout << "unreachable_nodes: " << unreachable << "\n";
    std::cout << std::fixed << std::setprecision(8)
              << "unreachable_ratio: " << ratio << "\n";
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 2;
  }

  return 0;
}

// fresh:
// /tmp/check_graph_connectivity /root/work_folder/diskann_index/tiny5m/original_index_500000/tiny5m_after_step10_batchsize50000_DiskANN
// /tmp/check_graph_connectivity /root/work_folder/diskann_index/gist/original_index_500000/gist_after_step10_batchsize50000_DiskANN
// /tmp/check_graph_connectivity /root/work_folder/diskann_index/sift100w_shuffled/original_index_500000/sift100w_shuffled_after_step10_batchsize50000_DiskANN

// DirANN
// /tmp/check_graph_connectivity /root/work_folder/diskann_index/tiny5m/original_index_500000/tiny5m_after_step10_batchsize50000_DirANN
// /tmp/check_graph_connectivity /root/work_folder/diskann_index/gist/original_index_500000/gist_after_step10_batchsize50000_DirANN
// /tmp/check_graph_connectivity /root/work_folder/diskann_index/sift100w_shuffled/original_index_500000/sift100w_shuffled_after_step10_batchsize50000_DirANN