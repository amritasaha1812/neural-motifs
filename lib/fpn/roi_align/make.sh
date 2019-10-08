#!/usr/bin/env bash

CUDA_PATH=/opt/share/cuda-9.0/x86_64/

cd src
echo "Compiling my_lib kernels by nvcc..."
/opt/share/cuda-9.0/x86_64/bin/nvcc -c -o roi_align_kernel.cu.o roi_align_kernel.cu -x cu -Xcompiler -fPIC -arch=sm_37

cd ../
python build.py
