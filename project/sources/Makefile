
NVCC=nvcc
INCLUDE= -I./../libgputk
LIBgpuTK= -L./../build-dir -lgputk -lcuda
LIBS= $(LIBgpuTK)

all: naive no_lock bitmap sort_reduce

naive:
	$(NVCC) -std=c++11 baseline.cu $(INCLUDE) $(LIBS) -o BASE
no_lock:
	$(NVCC) -std=c++11 no_lock.cu $(INCLUDE) $(LIBS) -o NO_LOCK

bitmap:
	$(NVCC) -std=c++11 bitmap_w_lock.cu $(INCLUDE) $(LIBS) -o BITMAP

sort_reduce:
	$(NVCC) -std=c++14 sort_reduce.cu $(INCLUDE) $(LIBS) -o sort_reduce


dataset_generator:
	$(NVCC) -std=c++11 dataset_generator.cpp $(INCLUDE) $(LIBS) -o dataset_generator


clean:
	rm -f ListScan_template dataset_generator
