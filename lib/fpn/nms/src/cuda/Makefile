all: nms_kernel.cu nms_kernel.h
	/opt/share/cuda-9.0/x86_64/bin/nvcc -c -o nms.cu.o nms_kernel.cu --compiler-options -fPIC -gencode arch=compute_37,code=sm_37
clean:
	rm nms.cu.o
