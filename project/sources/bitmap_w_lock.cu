#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <time.h>
#include <gputk.h>
#define BLOCK_SIZE 256
#define BITMAP_SIZE 536870912 // UINT_MAX / 8 + 1
#define LOCKSIZE 536870912 // LOCK per char

char bitmap[BITMAP_SIZE];
__device__ int mutex = 0;
__global__ void markBitmap(unsigned* d_array, unsigned int* locks, char* d_bitmap, int N, int* d_zero_flag) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < N) {
        unsigned int val = d_array[tid];
        if (val == 0){
            d_zero_flag[0] = 1;
            return;
        }
        int idx = val / 8;
        int offset = val % 8;
        int mask = 1 << offset;
        bool blocked = true;
        while(blocked){
            if(0 == atomicCAS(&(locks[idx]), 0u,1u)){
                d_bitmap[idx] |= mask;
                atomicExch(&(locks[idx]),0u);
                blocked = false;
            }
        }
    }
}
__global__ void collectUintFromBitmap(char* d_bitmap, unsigned int* result, int N, int* pos) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int arr[8];
    memset(arr, 0, 8 * sizeof(unsigned int));
    int cnt = 0;
    if (tid < BITMAP_SIZE) {
        char val = d_bitmap[tid];
        if (val == 0){
            return;
        }
        for (int i = 0; i < 8; i++) {
            if ((val & (1 << i)) >> i == 1) {
                arr[cnt] = tid * 8 + i;
                cnt++;
            }
        }
    }
    if(cnt == 0){
        return;
    }
    bool blocked = true;
    while(blocked){
        if(0 == atomicCAS(&mutex, 0,1)){
            for(int i = 0; i<cnt;i++){
                result[pos[0] + i] = arr[i];
            }
            pos[0] += cnt;
            atomicExch(&mutex,0);
            blocked = false;
        }
    }
}

unsigned int* generateRandomUint(unsigned int* arr, int N) {
    for (int i = 0; i < N; i++) {
        arr[i] = rand() % N;
    }
    return arr;
}
unsigned int host_array[100000000];
unsigned int host_result[100000000];
int main() {
    
    int N = 100000000;
            
    char* d_bitmap;
    memset(bitmap, 0, BITMAP_SIZE * sizeof(char));
    
    int zero_flag[1];
    zero_flag[0] = 0;   
    int* d_zero_flag;
    generateRandomUint(host_array, N);
    host_array[100] = 0;
    host_array[30] = UINT_MAX;
    for (int i = 0; i < 20; i++) {
        printf("%u ", host_array[i]);
    }
    printf("\n");
    unsigned int* d_array;
    unsigned int* locks;
    cudaMalloc((void**)&d_zero_flag, sizeof(int));
    cudaMalloc((void**)&d_array, N * sizeof(unsigned int));
    cudaMalloc((void**)&d_bitmap, BITMAP_SIZE * sizeof(char));
    cudaMalloc((void**)&locks, LOCKSIZE * sizeof(unsigned int));
    cudaMemset(d_bitmap, 0, BITMAP_SIZE * sizeof(char));
    cudaMemset(d_zero_flag, 0, sizeof(int));
    cudaMemset(locks, 0, LOCKSIZE * sizeof(unsigned int));
    cudaMemcpy(d_array, host_array, N * sizeof(unsigned int), cudaMemcpyHostToDevice);
    gpuTKTime_start(GPU, "Mark Bitmap");
    markBitmap<<<(N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_array, locks, d_bitmap, N, d_zero_flag);
    printf("Complete1\n");
    gpuTKTime_stop(GPU, "Mark Bitmap");
   
    cudaMemcpy(zero_flag, d_zero_flag, sizeof(int), cudaMemcpyDeviceToHost);

    unsigned int* d_result;
    unsigned int* d_count;
    int* d_pos;
    cudaMalloc((void**)&d_result, N * sizeof(unsigned int));
    cudaMalloc((void**)&d_count, sizeof(unsigned int));
    cudaMalloc((void**)&d_pos, sizeof(int));
    cudaMemset(d_result, 0, N * sizeof(unsigned int));
    cudaMemset(d_count, 0, sizeof(unsigned int));
    cudaMemset(d_pos, 0, sizeof(int));

    gpuTKTime_start(GPU, "Collect Uint from Bitmap");
    collectUintFromBitmap<<<(BITMAP_SIZE) / BLOCK_SIZE + 1, BLOCK_SIZE>>>(d_bitmap, d_result, N, d_pos);
    
    cudaDeviceSynchronize();  // 커널이 완료될 때까지 대기
    printf("Complete2\n");
    gpuTKTime_stop(GPU, "Collect Uint from Bitmap");
    
    cudaMemcpy(host_result, d_result, N * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < 20; ++i) {
        if(host_result[i] != 0)
            printf("%u ", host_result[i]);
    }
    if(zero_flag[0] == 1)
        printf("0 ");
    
    cudaFree(d_array);
    cudaFree(d_bitmap);
    cudaFree(locks);
    cudaFree(d_zero_flag);
    cudaFree(d_result);
    cudaFree(d_count);
    cudaFree(d_pos);
    return 0;
}
