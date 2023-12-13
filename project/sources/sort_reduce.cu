#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <time.h>
#include <gputk.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#define BLOCK_SIZE 1024
unsigned int* generateRandomUint(unsigned int* arr, int N) {
    for (int i = 0; i < N; i++) {
        arr[i] = rand() % N;
    }
    return arr;
}
__global__ void reduceDuplicates(unsigned int* input, int N, int* zero_flag){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int value = input[tid];
    bool isEqual = false;
    if (tid<N-1){
        if(value == 0){
            zero_flag[0] = 1;
        }
        if (value == input[tid + 1]){
            isEqual = true;
        }
        else{
            isEqual = false;
        }
        __syncthreads();
        if (isEqual){
            input[tid] = 0;
        }
    }
}
int main(){
    int N = 100000000;
    unsigned int* input = (unsigned int*)malloc(N * sizeof(unsigned int));
    input = generateRandomUint(input, N);
    /*for (int i = 0; i < N; i++) {
        printf("%u ", input[i]);
    }*/
    printf("\n");
    unsigned int* d_input;
    cudaMalloc((void**)&d_input, N * sizeof(unsigned int));
    cudaMemcpy(d_input, input, N * sizeof(unsigned int), cudaMemcpyHostToDevice);
    gpuTKTime_start(GPU, "Sorting");
    thrust::sort(thrust::device, d_input, d_input + N);
    gpuTKTime_stop(GPU, "Sorting");
    int zero_flag = 0;
    int* d_zero_flag;
    cudaMalloc((void**)&d_zero_flag, sizeof(int));
    cudaMemcpy(d_zero_flag, &zero_flag, sizeof(int), cudaMemcpyHostToDevice);
    gpuTKTime_start(GPU, "Reduce Duplicates");
    reduceDuplicates<<<N / BLOCK_SIZE + 1, BLOCK_SIZE >>>(d_input, N, d_zero_flag);
    gpuTKTime_stop(GPU, "Reduce Duplicates");
    cudaMemcpy(&zero_flag, d_zero_flag, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(input, d_input, N * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    /*if (zero_flag == 1){
        printf("0 ");
    }
    for (int i = 0; i < N; i++) {
        if(input[i] != 0)
            printf("%u ", input[i]);
    }*/

    printf("\n");
    cudaFree(d_input);
    cudaFree(d_zero_flag);
    free(input);
    return 0;

}