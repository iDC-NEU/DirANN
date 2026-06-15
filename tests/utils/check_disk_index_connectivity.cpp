#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <queue>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr uint64_t kSectorLen = 4096;
constexpr size_t kCoordBytesPerDim = sizeof(float);

struct ParsedHeader {
	std::string format;
	uint64_t node_count = 0;
	uint64_t ndims = 0;
	uint64_t entry_point = 0;
	uint64_t max_node_len = 0;
	uint64_t nnodes_per_sector = 0;
	uint64_t width = 0;
	uint64_t frozen_points = 0;
	uint64_t file_size_field = 0;
	uint64_t data_offset = 0;
};

struct GraphData {
	ParsedHeader header;
	std::vector<std::vector<uint32_t>> adj;
	uint64_t edge_count = 0;
	uint64_t invalid_edge_count = 0;
	uint64_t malformed_nodes = 0;
};

bool read_exact(std::ifstream &in, char *ptr, size_t n) {
	in.read(ptr, static_cast<std::streamsize>(n));
	return static_cast<size_t>(in.gcount()) == n;
}

uint64_t get_file_size(std::ifstream &in) {
	const std::streampos cur = in.tellg();
	in.seekg(0, std::ios::end);
	const std::streampos end = in.tellg();
	in.seekg(cur, std::ios::beg);
	if (end < 0) {
		throw std::runtime_error("failed to get file size");
	}
	return static_cast<uint64_t>(end);
}

std::string resolve_input_path(const std::string &input) {
	std::ifstream direct(input, std::ios::binary);
	if (direct.good()) {
		return input;
	}

	const std::string appended = input + "_disk.index";
	std::ifstream fallback(appended, std::ios::binary);
	if (fallback.good()) {
		return appended;
	}

	throw std::runtime_error("cannot open input file: " + input + " (or " + appended + ")");
}

bool try_parse_graph_header(std::ifstream &in, uint64_t file_size, ParsedHeader &h) {
	in.clear();
	in.seekg(0, std::ios::beg);

	uint64_t expected_file_size = 0;
	uint32_t width = 0;
	uint32_t ep = 0;
	uint64_t frozen = 0;

	if (!read_exact(in, reinterpret_cast<char *>(&expected_file_size), sizeof(uint64_t)) ||
			!read_exact(in, reinterpret_cast<char *>(&width), sizeof(uint32_t)) ||
			!read_exact(in, reinterpret_cast<char *>(&ep), sizeof(uint32_t)) ||
			!read_exact(in, reinterpret_cast<char *>(&frozen), sizeof(uint64_t))) {
		return false;
	}

	if (expected_file_size < 24 || expected_file_size > file_size) {
		return false;
	}
	if (width == 0 || width > 1000000) {
		return false;
	}
	if (frozen > 1) {
		return false;
	}

	h.format = "graph";
	h.entry_point = ep;
	h.width = width;
	h.frozen_points = frozen;
	h.file_size_field = expected_file_size;
	h.data_offset = 24;
	return true;
}

bool try_parse_disk_header_new(std::ifstream &in, uint64_t file_size, ParsedHeader &h) {
	in.clear();
	in.seekg(0, std::ios::beg);

	uint32_t nr = 0, nc = 0;
	uint64_t nodes = 0, ndims = 0, ep = 0, max_node_len = 0, nnodes_per_sector = 0;

	if (!read_exact(in, reinterpret_cast<char *>(&nr), sizeof(uint32_t)) ||
			!read_exact(in, reinterpret_cast<char *>(&nc), sizeof(uint32_t)) ||
			!read_exact(in, reinterpret_cast<char *>(&nodes), sizeof(uint64_t)) ||
			!read_exact(in, reinterpret_cast<char *>(&ndims), sizeof(uint64_t)) ||
			!read_exact(in, reinterpret_cast<char *>(&ep), sizeof(uint64_t)) ||
			!read_exact(in, reinterpret_cast<char *>(&max_node_len), sizeof(uint64_t)) ||
			!read_exact(in, reinterpret_cast<char *>(&nnodes_per_sector), sizeof(uint64_t))) {
		return false;
	}

	if (nr != 6 || nc != 1) {
		return false;
	}
	if (nodes == 0 || ndims == 0) {
		return false;
	}
	if (max_node_len < ndims * kCoordBytesPerDim + sizeof(uint32_t)) {
		return false;
	}
	if (ep >= nodes) {
		return false;
	}
	if (kSectorLen >= file_size) {
		return false;
	}

	h.format = "disk";
	h.node_count = nodes;
	h.ndims = ndims;
	h.entry_point = ep;
	h.max_node_len = max_node_len;
	h.nnodes_per_sector = nnodes_per_sector;
	h.data_offset = kSectorLen;
	h.file_size_field = file_size;
	return true;
}

bool try_parse_disk_header_legacy(std::ifstream &in, uint64_t file_size, ParsedHeader &h) {
	in.clear();
	in.seekg(0, std::ios::beg);

	int32_t npts = 0, ndims = 0;
	if (!read_exact(in, reinterpret_cast<char *>(&npts), sizeof(int32_t)) ||
			!read_exact(in, reinterpret_cast<char *>(&ndims), sizeof(int32_t))) {
		return false;
	}

	if (ndims != 1 || npts < 8 || npts > 16) {
		return false;
	}

	std::vector<uint64_t> meta(static_cast<size_t>(npts), 0);
	if (!read_exact(in, reinterpret_cast<char *>(meta.data()), sizeof(uint64_t) * meta.size())) {
		return false;
	}

	const uint64_t nodes = meta[0];
	const uint64_t vec_dims = meta[1];
	const uint64_t ep = meta[2];
	const uint64_t max_node_len = meta[3];
	const uint64_t nnodes_per_sector = meta[4];
	const uint64_t frozen_points = meta.size() > 5 ? meta[5] : 0;
	const uint64_t file_size_field = meta.size() > 7 ? meta[7] : file_size;

	if (nodes == 0 || vec_dims == 0) {
		return false;
	}
	if (max_node_len < vec_dims * kCoordBytesPerDim + sizeof(uint32_t)) {
		return false;
	}
	if (ep >= nodes) {
		return false;
	}
	if (kSectorLen >= file_size) {
		return false;
	}

	h.format = "disk";
	h.node_count = nodes;
	h.ndims = vec_dims;
	h.entry_point = ep;
	h.max_node_len = max_node_len;
	h.nnodes_per_sector = nnodes_per_sector;
	h.frozen_points = frozen_points;
	h.file_size_field = file_size_field;
	h.data_offset = kSectorLen;
	return true;
}

GraphData load_graph_format(const std::string &path, const ParsedHeader &header) {
	std::ifstream in(path, std::ios::binary);
	if (!in) {
		throw std::runtime_error("cannot open file: " + path);
	}

	in.seekg(static_cast<std::streamoff>(header.data_offset), std::ios::beg);

	GraphData out;
	out.header = header;
	uint64_t bytes_read = header.data_offset;

	while (bytes_read < header.file_size_field) {
		uint32_t degree = 0;
		if (!read_exact(in, reinterpret_cast<char *>(&degree), sizeof(uint32_t))) {
			throw std::runtime_error("unexpected EOF while reading graph degree");
		}
		bytes_read += sizeof(uint32_t);

		const uint64_t neigh_bytes = static_cast<uint64_t>(degree) * sizeof(uint32_t);
		if (bytes_read + neigh_bytes > header.file_size_field) {
			throw std::runtime_error("corrupted graph file: degree exceeds declared file size");
		}

		std::vector<uint32_t> neighbors(static_cast<size_t>(degree));
		if (degree > 0 && !read_exact(in, reinterpret_cast<char *>(neighbors.data()), static_cast<size_t>(neigh_bytes))) {
			throw std::runtime_error("unexpected EOF while reading graph neighbors");
		}

		out.edge_count += degree;
		out.adj.emplace_back(std::move(neighbors));
		bytes_read += neigh_bytes;
	}

	out.header.node_count = out.adj.size();
	return out;
}

GraphData load_disk_format(const std::string &path, const ParsedHeader &header) {
	std::ifstream in(path, std::ios::binary);
	if (!in) {
		throw std::runtime_error("cannot open file: " + path);
	}

	GraphData out;
	out.header = header;
	out.adj.resize(static_cast<size_t>(header.node_count));

	const uint64_t coord_bytes = header.ndims * kCoordBytesPerDim;
	const uint64_t degree_capacity = (header.max_node_len - coord_bytes) / sizeof(uint32_t);
	if (degree_capacity == 0) {
		throw std::runtime_error("invalid disk header: no space for degree");
	}
	const uint64_t max_degree = degree_capacity - 1;

	auto parse_node = [&](const char *node_buf, uint64_t node_id) {
		uint32_t degree = 0;
		memcpy(&degree, node_buf + coord_bytes, sizeof(uint32_t));

		if (degree > max_degree) {
			out.malformed_nodes++;
			degree = static_cast<uint32_t>(max_degree);
		}

		std::vector<uint32_t> neighbors;
		neighbors.reserve(std::min<uint32_t>(degree, 2048));
		const char *nbr_ptr = node_buf + coord_bytes + sizeof(uint32_t);

		for (uint32_t i = 0; i < degree; ++i) {
			uint32_t nbr = 0;
			memcpy(&nbr, nbr_ptr + static_cast<size_t>(i) * sizeof(uint32_t), sizeof(uint32_t));
			neighbors.push_back(nbr);
			if (nbr >= header.node_count) {
				out.invalid_edge_count++;
			}
		}

		out.edge_count += neighbors.size();
		out.adj[static_cast<size_t>(node_id)] = std::move(neighbors);
	};

	in.seekg(static_cast<std::streamoff>(header.data_offset), std::ios::beg);

	if (header.nnodes_per_sector > 0) {
		const uint64_t n_sectors = (header.node_count + header.nnodes_per_sector - 1) / header.nnodes_per_sector;
		std::vector<char> sector_buf(static_cast<size_t>(kSectorLen), 0);

		for (uint64_t sector_idx = 0; sector_idx < n_sectors; ++sector_idx) {
			if (!read_exact(in, sector_buf.data(), sector_buf.size())) {
				throw std::runtime_error("unexpected EOF while reading sector");
			}

			const uint64_t st = sector_idx * header.nnodes_per_sector;
			const uint64_t ed = std::min(header.node_count, st + header.nnodes_per_sector);
			for (uint64_t id = st; id < ed; ++id) {
				const uint64_t off = (id - st) * header.max_node_len;
				if (off + header.max_node_len > sector_buf.size()) {
					throw std::runtime_error("corrupted sector layout");
				}
				parse_node(sector_buf.data() + off, id);
			}
		}
	} else {
		const uint64_t nsectors_per_node = (header.max_node_len + kSectorLen - 1) / kSectorLen;
		const uint64_t node_bytes = nsectors_per_node * kSectorLen;
		std::vector<char> node_buf(static_cast<size_t>(node_bytes), 0);

		for (uint64_t id = 0; id < header.node_count; ++id) {
			if (!read_exact(in, node_buf.data(), node_buf.size())) {
				throw std::runtime_error("unexpected EOF while reading large node pages");
			}
			parse_node(node_buf.data(), id);
		}
	}

	return out;
}

GraphData load_any_format(const std::string &path) {
	std::ifstream in(path, std::ios::binary);
	if (!in) {
		throw std::runtime_error("cannot open file: " + path);
	}
	const uint64_t file_size = get_file_size(in);

	ParsedHeader header;
	if (try_parse_graph_header(in, file_size, header)) {
		return load_graph_format(path, header);
	}
	if (try_parse_disk_header_new(in, file_size, header)) {
		return load_disk_format(path, header);
	}
	if (try_parse_disk_header_legacy(in, file_size, header)) {
		return load_disk_format(path, header);
	}

	throw std::runtime_error("unrecognized file format (not graph / disk-new / disk-legacy)");
}

void print_usage(const char *prog) {
	std::cerr << "Usage: " << prog << " <graph_or_disk_file_or_prefix>\n"
						<< "Supports graph format and two disk-index formats.\n";
}

}  // namespace

// ./check_disk_index_cnnectivity diskann_output/glove/original_index_500000/glove_R32_disk.index
int main(int argc, char **argv) {
	if (argc != 2) {
		print_usage(argv[0]);
		return 1;
	}

	try {
		const std::string input = argv[1];
		const std::string path = resolve_input_path(input);
		GraphData g = load_any_format(path);

		const uint64_t n = g.adj.size();
		if (n == 0) {
			std::cout << "graph_file: " << path << "\n";
			std::cout << "format: " << g.header.format << "\n";
			std::cout << "nodes: 0\n";
			std::cout << "edges: 0\n";
			std::cout << "entry_point: 0\n";
			std::cout << "unreachable_nodes: 0\n";
			std::cout << "unreachable_ratio: 0\n";
			return 0;
		}

		if (g.header.entry_point >= n) {
			std::cerr << "Error: entry point " << g.header.entry_point << " is out of range [0, " << (n - 1) << "]\n";
			return 2;
		}

		std::vector<uint8_t> visited(static_cast<size_t>(n), 0);
		std::queue<uint32_t> q;
		visited[static_cast<size_t>(g.header.entry_point)] = 1;
		q.push(static_cast<uint32_t>(g.header.entry_point));

		uint64_t reached = 0;
		while (!q.empty()) {
			const uint32_t u = q.front();
			q.pop();
			reached++;

			for (uint32_t v : g.adj[static_cast<size_t>(u)]) {
				if (v >= n) {
					continue;
				}
				if (!visited[static_cast<size_t>(v)]) {
					visited[static_cast<size_t>(v)] = 1;
					q.push(v);
				}
			}
		}

		const uint64_t unreachable = n - reached;
		const double ratio = static_cast<double>(unreachable) / static_cast<double>(n);

		std::cout << "graph_file: " << path << "\n";
		std::cout << "format: " << g.header.format << "\n";
		std::cout << "nodes: " << n << "\n";
		std::cout << "edges: " << g.edge_count << "\n";
		std::cout << "entry_point: " << g.header.entry_point << "\n";

		if (g.header.format == "graph") {
			std::cout << "width: " << g.header.width << "\n";
			std::cout << "frozen_points: " << g.header.frozen_points << "\n";
			std::cout << "file_size: " << g.header.file_size_field << "\n";
		} else {
			std::cout << "disk_ndims: " << g.header.ndims << "\n";
			std::cout << "max_node_len: " << g.header.max_node_len << "\n";
			std::cout << "nnodes_per_sector: " << g.header.nnodes_per_sector << "\n";
			std::cout << "frozen_points: " << g.header.frozen_points << "\n";
			std::cout << "file_size: " << g.header.file_size_field << "\n";
		}

		std::cout << "malformed_nodes: " << g.malformed_nodes << "\n";
		std::cout << "invalid_edges(ignored): " << g.invalid_edge_count << "\n";
		std::cout << "reachable_nodes: " << reached << "\n";
		std::cout << "unreachable_nodes: " << unreachable << "\n";
		std::cout << std::fixed << std::setprecision(8) << "unreachable_ratio: " << ratio << "\n";
	} catch (const std::exception &e) {
		std::cerr << "Error: " << e.what() << "\n";
		return 2;
	}

	return 0;
}
