// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ...
// +
// lst[n-1]}

#include <gputk.h>

#define BLOCK_SIZE 512 //@@ You can change this

#define gpuTKCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      gpuTKLog(ERROR, "Failed to run stmt ", #stmt);                         \
      gpuTKLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

__global__ void scan(float *input, float *output, float* intermediate, int len, int phase) {
  //@@ Modify the body of this function to complete the functionality of
  //@@ the scan on the device
  //@@ You may need multiple kernel calls; write your kernels before this
  //@@ function and call them from here
  __shared__ float sh_scan[BLOCK_SIZE << 1];
  unsigned int tid = threadIdx.x;
  unsigned int start = 2 * blockIdx.x * BLOCK_SIZE;
  if(start + tid < len)
    sh_scan[tid] = input[start + tid];
  else
    sh_scan[tid] = 0;
  if(BLOCK_SIZE + start + tid < len)
    sh_scan[BLOCK_SIZE + tid] = input[BLOCK_SIZE + start + tid];
  else
    sh_scan[BLOCK_SIZE + tid] = 0;
  __syncthreads();
  /* Loading Data to Shared memory */
  for(int stride = 1; stride <= BLOCK_SIZE; stride <<= 1){
    int idx = (tid + 1) * stride * 2 - 1;
    if(idx < 2 * BLOCK_SIZE)
      sh_scan[idx] += sh_scan[idx - stride];
    __syncthreads();
  }
  for(int stride = BLOCK_SIZE >> 1; stride; stride >>= 1){
    int idx = (tid + 1) * stride * 2 - 1;
    if(idx + stride < 2 * BLOCK_SIZE)
      sh_scan[idx + stride] += sh_scan[idx];
    __syncthreads();
  }
  /* Reduction */
  if(tid + start < len)
    output[start + tid] = sh_scan[tid];
  if(tid + start + BLOCK_SIZE < len)
    output[start + tid + BLOCK_SIZE] = sh_scan[tid + BLOCK_SIZE];
  if(phase == 1 && tid == 0)
    intermediate[blockIdx.x] = sh_scan[2 * BLOCK_SIZE - 1];
}
__global__ void adjust(float* input, float* intermediate, int len){
  unsigned int tid = threadIdx.x;
  unsigned int start = 2 * blockIdx.x * BLOCK_SIZE;
  if(blockIdx.x){
    if(tid + start < len)
      input[tid + start] += intermediate[blockIdx.x - 1];
    if(start + BLOCK_SIZE + tid < len)
      input[start + BLOCK_SIZE + tid] += intermediate[blockIdx.x - 1];
  }
}

int main(int argc, char **argv) {
  gpuTKArg_t args;
  float *hostInput;  // The input 1D list
  float *hostOutput; // The output list
  float *deviceInput;
  float *deviceOutput;
  int numElements; // number of elements in the list
  float* intermediate, *intermediate_2;
  args = gpuTKArg_read(argc, argv);
  cudaMalloc(&intermediate, (BLOCK_SIZE << 1) * sizeof(float));
  cudaMalloc(&intermediate_2, (BLOCK_SIZE << 1) * sizeof(float));

  gpuTKTime_start(Generic, "Importing data and creating memory on host");
  hostInput = (float *)gpuTKImport(gpuTKArg_getInputFile(args, 0), &numElements);
  hostOutput = (float *)malloc(numElements * sizeof(float));
  gpuTKTime_stop(Generic, "Importing data and creating memory on host");

  gpuTKLog(TRACE, "The number of input elements in the input is ",
        numElements);

  gpuTKTime_start(GPU, "Allocating GPU memory.");
  gpuTKCheck(cudaMalloc((void **)&deviceInput, numElements * sizeof(float)));
  gpuTKCheck(cudaMalloc((void **)&deviceOutput, numElements * sizeof(float)));
  gpuTKTime_stop(GPU, "Allocating GPU memory.");

  gpuTKTime_start(GPU, "Clearing output memory.");
  gpuTKCheck(cudaMemset(deviceOutput, 0, numElements * sizeof(float)));
  gpuTKTime_stop(GPU, "Clearing output memory.");

  gpuTKTime_start(GPU, "Copying input memory to the GPU.");
  gpuTKCheck(cudaMemcpy(deviceInput, hostInput, numElements * sizeof(float),
                     cudaMemcpyHostToDevice));
  gpuTKTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  int numBlocks = ceil((float)numElements / (BLOCK_SIZE<<1));
  dim3 dimGrid(numBlocks, 1, 1);
  dim3 dimBlock(BLOCK_SIZE, 1, 1);
  gpuTKTime_start(Compute, "Performing CUDA computation");
  //@@ Modify this to complete the functionality of the scan
  //@@ on the deivce
  scan<<<dimGrid, dimBlock>>>(deviceInput, deviceOutput, intermediate, numElements,1);
  cudaDeviceSynchronize();
  scan<<<dim3(1,1,1), dimBlock>>>(intermediate, intermediate_2, NULL, BLOCK_SIZE << 1,2);
  cudaDeviceSynchronize();
  adjust<<<dimGrid, dimBlock>>>(deviceOutput, intermediate_2, numElements);  
  cudaDeviceSynchronize();
  gpuTKTime_stop(Compute, "Performing CUDA computation");

  gpuTKTime_start(Copy, "Copying output memory to the CPU");
  gpuTKCheck(cudaMemcpy(hostOutput, deviceOutput, numElements * sizeof(float),
                     cudaMemcpyDeviceToHost));
  gpuTKTime_stop(Copy, "Copying output memory to the CPU");

  gpuTKTime_start(GPU, "Freeing GPU Memory");
  cudaFree(deviceInput);
  cudaFree(deviceOutput);
  cudaFree(intermediate);
  cudaFree(intermediate_2);
  gpuTKTime_stop(GPU, "Freeing GPU Memory");

  gpuTKSolution(args, hostOutput, numElements);

  free(hostInput);
  free(hostOutput);

  return 0;
}
