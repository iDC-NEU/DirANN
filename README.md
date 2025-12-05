# DirANN: Direction-Aware Repair for Graph-Based ANN Indices

### Software Dependencies

For Ubuntu >= 22.04, the command to install them:

```bash
sudo apt install make cmake g++ libaio-dev libgoogle-perftools-dev clang-format libboost-all-dev libmkl-full-dev libjemalloc-dev
```

The `libmkl` could be replaced by other `BLAS` libraries (e.g., OpenBlas).

### Build and Run

Build the repository.
```bash
./build.sh
```

Configure dataset path and index path.
```bash
./scripts/config.sh
```

Build the index.
```bash
./scripts/process.sh
```

Run the code.
```bash
./scripts/run_exp1.sh
```