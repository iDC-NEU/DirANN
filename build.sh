export ADDITIONAL_DEFINITIONS="-DBG_IO_THREAD -DDELTA_PRUNING"

mkdir build
cd build
cmake ..
make -j