#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

#define TILE_WIDTH 16

__constant__ __half mask_const[4000];

__global__ void conv_forward_kernel(float *output, const float *input, const float *mask, const int B, const int M, const int C, const int H, const int W, const int K,const int S)
{

    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.

    Function paramter definitions:
    output - output
    input - input
    mask - convolution kernel
    B - batch_size (number of images in x)
    M - number of output feature maps
    C - number of input feature maps
    H - input height dimension
    W - input width dimension
    K - kernel height and width (K x K)
    S - stride step length
    */
    int c, p, q, b, m, h, w;
    __half pConv;

    const int H_out = (H - K)/S + 1;
    const int W_out = (W - K)/S + 1;
    const int W_out_blocks = (int)ceil((__half)W_out / (__half)TILE_WIDTH);
    // (void)H_out; // silence declared but never referenced warning. remove this line when you start working
    // (void)W_out; // silence declared but never referenced warning. remove this line when you start working

    // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    // An example use of these macros:
    // float a = in_4d(0,0,0,0)
    // out_4d(0,0,0,0) = a

    #define out_4d(i3, i2, i1, i0) output[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
    #define mask_4d(i3, i2, i1, i0) mask_const[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    // Insert your GPU convolution kernel code here
    b = blockIdx.x; // for each image in the batch
    m = blockIdx.y;// for each output feature map given an image
    // for each element given the output feature map
    h = (blockIdx.z / W_out_blocks) * TILE_WIDTH + threadIdx.y; 
    w = (blockIdx.z % W_out_blocks) * TILE_WIDTH + threadIdx.x;
    // initialize the output to 0
    pConv = 0.0;
    // sum over the entire input feature map
    for(c = 0; c < C; c++){ 
        for(p = 0; p < K; p++){
            for(q = 0; q < K; q++){
                if(h * S + p < H && w * S + q < W){
                    pConv += (__half)in_4d(b, c, h * S + p, w * S + q) * mask_4d(m, c, p, q);
                }
            }
        }
    }
    
    if(h < H_out && w < W_out){
        out_4d(b, m, h, w) = (float) pConv;
    }

    #undef out_4d
    #undef in_4d
    #undef mask_4d
}

	
__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    // Allocate memory and copy over the relevant data structures to the GPU

    // We pass double pointers for you to initialize the relevant device pointers,
    //  which are passed to the other two functions.

    const int H_out = (H - K)/S + 1;
    const int W_out = (W - K)/S + 1;

    const int input_size = sizeof(float) * B * C * H * W;
    const int mask_size = sizeof(float) * M * C * K * K;
    const int output_size = sizeof(float) * B * M * H_out * W_out;

    cudaMalloc(device_input_ptr, input_size);
    // cudaMalloc(device_mask_ptr, mask_size);
    cudaMalloc(device_output_ptr, output_size);

    cudaMemcpy(*device_input_ptr, host_input, input_size, cudaMemcpyHostToDevice);
    // cudaMemcpy(*device_mask_ptr, host_mask, mask_size, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(mask_const, host_mask, mask_size, 0, cudaMemcpyHostToDevice);

    // Useful snippet for error checking
    // cudaError_t error = cudaGetLastError();
    // if(error != cudaSuccess)
    // {
    //     std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
    //     exit(-1);
    // }
   
}


__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    // Set the kernel dimensions and call the kernel
    const int H_out = (H - K)/S + 1;
    const int W_out = (W - K)/S + 1;

    int H_out_blocks = ceil(H_out / (float)TILE_WIDTH);
    int W_out_blocks = ceil(W_out / (float)TILE_WIDTH);
    
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 gridDim(B, M, H_out_blocks * W_out_blocks);

    conv_forward_kernel<<<gridDim, blockDim>>>(device_output, device_input, device_mask, B, M, C, H, W, K, S);
}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    // Copy the output back to host
    const int H_out = (H - K)/S + 1;
    const int W_out = (W - K)/S + 1;

    const int output_size = sizeof(float) * B * M * H_out * W_out;

    cudaMemcpy(host_output, device_output, output_size, cudaMemcpyDeviceToHost);
   
    // Free device memory
    cudaFree(device_output);
    // cudaFree(device_mask);
    cudaFree(device_input);

}


__host__ void GPUInterface::get_device_properties()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for(int dev = 0; dev < deviceCount; dev++)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
        std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
        std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
        std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
        std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
        std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
        std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
        std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
        std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
    }
}