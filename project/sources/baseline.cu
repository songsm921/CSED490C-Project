#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <time.h>
#include <vector>
#include <algorithm>
#include <gputk.h>
std::vector<unsigned int> generateRandom(int N){
    std::vector<unsigned int> v(N);
    for(int i=0;i<N;i++){
        v[i]=rand()%N;
    }
    return v;
}
int main(){
    std::vector<unsigned int> v=generateRandom(100000000);
    /*for(int i=0;i<v.size();i++){
        printf("%u ",v[i]);
    }*/
    printf("\n");
    gpuTKTime_start(Compute, "Process");
    std::sort(v.begin(),v.end());
    v.erase(std::unique(v.begin(),v.end()),v.end());
    gpuTKTime_stop(Compute, "Process");
    /*for(int i=0;i<v.size();i++){
        printf("%u ",v[i]);
    }*/
}

