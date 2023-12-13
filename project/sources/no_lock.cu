#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <time.h>
#include <gputk.h>
#define BLOCK_SIZE 1024
__device__ int mutex = 0;
__device__ int count = 0;
unsigned int* generateRandomUint(unsigned int* arr, int N) {
    for (int i = 0; i < N; i++) {
        arr[i] = rand() % N;
    }
    return arr;
}

__global__ void markBitmap(unsigned int* input, char* d_bitmap, int N, int* d_zero_flag){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid < N){
        unsigned int value = input[tid];
        if(value == 0){
            d_zero_flag[0] = 1;
            return;
        }
        d_bitmap[value] = 1;
    }
}
__global__ void countBitmap(unsigned int* output, char* d_bitmap, int N, unsigned int* index){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    unsigned int idx = index[0] + tid;
    unsigned int value;
    if(idx <= UINT_MAX){
        if(d_bitmap[idx] == 1){
            //printf("%d\n", idx);
            value = idx;
        }
        else
            return;
    }
    else
        return;
    bool blocked = true;
    while(blocked){
        
        if(0 == atomicCAS(&mutex, 0, 1)){
            output[count] = value;
            count++;
            atomicExch(&mutex, 0);
            blocked = false;
        }
    }
}
unsigned int h_output[100000000];
int main(){
    int N = 100000000;
    int zero_flag = 0;
    int* d_zero_flag;
    cudaMalloc((void**)&d_zero_flag, sizeof(int));
    cudaMemset(d_zero_flag, 0, sizeof(int));
    unsigned int* h_input = (unsigned int*)malloc(N * sizeof(unsigned int));
    unsigned int* d_input;
    char* d_bitmap;
    cudaMalloc((void**)&d_bitmap, UINT_MAX * sizeof(char));
    cudaMemset(d_bitmap, 0, UINT_MAX * sizeof(char));
    h_input = generateRandomUint(h_input, N);
    h_input[3] = UINT_MAX;
    /*
    for(int i = 0; i < N; i++){
        printf("%u ", h_input[i]);
    }*/
    printf("\n");
    cudaMalloc((void**)&d_input, N * sizeof(unsigned int));
    cudaMemcpy(d_input, h_input, N * sizeof(unsigned int), cudaMemcpyHostToDevice);
    gpuTKTime_start(GPU, "Mark Bitmap");
    markBitmap<<< N / BLOCK_SIZE + 1, BLOCK_SIZE>>>(d_input, d_bitmap, N, d_zero_flag);
    gpuTKTime_stop(GPU, "Mark Bitmap");
    cudaMemcpy(&zero_flag, d_zero_flag, sizeof(int), cudaMemcpyDeviceToHost);



    unsigned int* d_output;
    cudaMalloc((void**)&d_output, N * sizeof(unsigned int));
    bool check = 1;
    unsigned int h_index = 0;
    unsigned int* d_index;
    cudaMalloc((void**)&d_index, sizeof(unsigned int));
    cudaMemset(d_index, 0, sizeof(unsigned int));
    
    memset(h_output, 0, N * sizeof(unsigned int));
    gpuTKTime_start(GPU, "Collect Uint from Bitmap");
    int i=0;
    while(check){
        cudaMemcpy(d_index, &h_index, sizeof(unsigned int), cudaMemcpyHostToDevice);
        countBitmap<<<BLOCK_SIZE * BLOCK_SIZE,BLOCK_SIZE>>>(d_output, d_bitmap, N, d_index);
        cudaDeviceSynchronize();
        h_index += BLOCK_SIZE * BLOCK_SIZE * BLOCK_SIZE;
        if(h_index == 0)
            check = 0;
        i++;
        printf("i: %d\n", i);
    }
    cudaMemcpy(h_output, d_output, N * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    gpuTKTime_stop(GPU, "Collect Uint from Bitmap");
    /*for(int i = 0; i < N; i++){
        if(h_output[i] != 0)
           printf("%u ", h_output[i]);
    }
    if(zero_flag == 1)
        printf("0 ");
    printf("\n");*/

    cudaFree(d_zero_flag);
    cudaFree(d_input);
    cudaFree(d_bitmap);
    cudaFree(d_output);
    free(h_input);

    return 0;
}