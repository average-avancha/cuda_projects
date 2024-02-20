// MP Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ...
// +
// lst[n-1]}

#include <wb.h>

#define BLOCK_SIZE 1024 //@@ You can change this

enum scanmode { 
  MODE_INITAL_SCAN, 
  MODE_BLOCK_SCAN 
};

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

__global__ void scan(float *input, float *output, float *aux_array, int len, int mode) {
  //@@ Modify the body of this function to complete the functionality of
  //@@ the scan on the device
  //@@ You may need multiple kernel calls; write your kernels before this
  //@@ function and call them from the host

  int index_device = 2*(blockIdx.x * blockDim.x + threadIdx.x);
  int index_shared = 2*threadIdx.x;

  __shared__ float In[2*BLOCK_SIZE];
  if(index_device < len){
    In[index_shared] = input[index_device];
  } else{
    In[index_shared] = 0;
  }

  if(index_device + 1 < len){
    In[index_shared + 1] = input[index_device + 1];
  } else{
    In[index_shared + 1] = 0;
  }
  
  __syncthreads();

  // initial scan
  int stride = 1;
  while(stride < 2*BLOCK_SIZE){
    __syncthreads();
    index_shared = 2 * (threadIdx.x + 1) * stride - 1;
    if(index_shared < 2*BLOCK_SIZE && index_shared - stride >= 0){
      In[index_shared] += In[index_shared - stride];
    }

    stride = 2*stride;
  }
  __syncthreads();

  // post scan
  stride = BLOCK_SIZE/2;
  while(stride > 0){
    __syncthreads();
    index_shared = 2 * (threadIdx.x + 1) * stride - 1;
    if(index_shared + stride < 2*BLOCK_SIZE){
      In[index_shared + stride] += In[index_shared];
    }

    stride = stride/2;
  }
  __syncthreads();

  index_shared = 2*threadIdx.x;
  if(index_device < len){
    output[index_device] = In[index_shared];
  }

  if(index_device + 1 < len){
    output[index_device + 1] = In[index_shared + 1];
  }

  __syncthreads();

  if(threadIdx.x == 0 && mode == MODE_INITAL_SCAN){
    aux_array[blockIdx.x] = In[2*BLOCK_SIZE - 1];
  }
}

__global__ void addOffset(float *input, float* output, float *sum_array, int len) {
  
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  
  if ((blockIdx.x >= 1) && (index < len)){
    output[index] = input[index] + sum_array[blockIdx.x];
  } else if(index < len){
    output[index] = input[index];
  }
  
}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostInput;  // The input 1D list
  float *hostOutput; // The output list
  float *deviceInput;
  float * deviceTemp;
  float *deviceOutput;
  int numElements; // number of elements in the list
  int numBlocks;
  float * deviceAuxArray;
  float * blockSums;

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &numElements);
  hostOutput = (float *)malloc(numElements * sizeof(float));

  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The number of input elements in the input is ",
        numElements);

  numBlocks = ceil(numElements / (2*BLOCK_SIZE));

  float * hostblockSums = (float *)malloc(numBlocks * sizeof(float));
  // float * hostTemp = (float *)malloc(numElements * sizeof(float));

  wbTime_start(GPU, "Allocating GPU memory.");
  wbCheck(cudaMalloc((void **)&deviceInput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceTemp, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceOutput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceAuxArray, numBlocks * sizeof(float)));
  wbCheck(cudaMalloc((void **)&blockSums, numBlocks * sizeof(float)));
  
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Clearing output memory.");
  wbCheck(cudaMemset(deviceOutput, 0, numElements * sizeof(float)));
  wbTime_stop(GPU, "Clearing output memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  wbCheck(cudaMemcpy(deviceInput, hostInput, numElements * sizeof(float),
                     cudaMemcpyHostToDevice));
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  dim3 DimGrid(numBlocks, 1, 1);
  dim3 DimBlock(BLOCK_SIZE, 1, 1);

  dim3 DimGridBlockScan(1, 1, 1);
  dim3 DimBlockBlockScan(numBlocks, 1, 1);

  dim3 DimGridAdd(numBlocks, 1, 1);
  dim3 DimBlockAdd(2*BLOCK_SIZE, 1, 1);

  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Modify this to complete the functionality of the scan
  //@@ on the deivce
  scan<<<DimGrid, DimBlock>>>(deviceInput, deviceTemp, deviceAuxArray, numElements, MODE_INITAL_SCAN);
  scan<<<DimGridBlockScan, DimBlockBlockScan>>>(deviceAuxArray, blockSums, NULL, numBlocks, MODE_BLOCK_SCAN);
  cudaMemcpy(deviceOutput, deviceTemp, numElements * sizeof(float), cudaMemcpyDeviceToDevice);
  addOffset<<<DimGridAdd, DimBlockAdd>>>(deviceTemp, deviceOutput, blockSums, numElements);
  cudaDeviceSynchronize();

  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements * sizeof(float),
                     cudaMemcpyDeviceToHost));

  // wbCheck(cudaMemcpy(hostTemp, deviceTemp, numElements * sizeof(float),
  //                    cudaMemcpyDeviceToHost));

  wbCheck(cudaMemcpy(hostblockSums, blockSums, numBlocks * sizeof(float),
                     cudaMemcpyDeviceToHost));
  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  cudaFree(deviceInput);
  cudaFree(deviceTemp);
  cudaFree(deviceOutput);
  cudaFree(deviceAuxArray);
  cudaFree(blockSums);
  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostOutput, numElements);
  for(int i = 0; i < numElements; i++){
    wbLog(TRACE, "hostOutput[",i,"]:", hostOutput[i]);
  }
  
  // for(int i = 0; i < numElements; i++){
  //   wbLog(TRACE, "hostTemp[",i,"]:", hostTemp[i]);
  // }

  for(int i = 0; i < numBlocks; i++){
    wbLog(TRACE, "hostblockSums[",i,"]:", hostblockSums[i]);
  }


  free(hostInput);
  free(hostOutput);

  free(hostblockSums);
  // free(hostTemp);

  return 0;
}
