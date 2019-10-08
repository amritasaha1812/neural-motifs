#!/usr/bin/env bash

# CUDA_PATH=/usr/local/cuda/

CUDA_PATH=/opt/share/cuda-9.0/x86_64/

cd src/cuda
echo "Compiling stnm kernels by nvcc..."
/opt/share/cuda-9.0/x86_64/bin/nvcc -c -o nms.cu.o nms_kernel.cu -x cu -Xcompiler -fPIC -arch=sm_37

cd ../..
python build.py
