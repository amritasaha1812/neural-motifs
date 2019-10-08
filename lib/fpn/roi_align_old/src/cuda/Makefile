all: roi_align_kernel.cu roi_align_kernel.h
	/opt/share/cuda-9.1/x86_64/bin/nvcc -c -o roi_align.cu.o roi_align_kernel.cu --compiler-options -fPIC -gencode arch=compute_70,code=sm_70
clean:
	rm roi_align.cu.o
