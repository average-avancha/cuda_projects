// Histogram Equalization

#include <wb.h>

#define HISTOGRAM_LENGTH 256
#define BLOCK_SIZE 256.0
#define BLOCK_SIZE_int 256


#define clamp(x, start, end) (fminf( fmaxf( (x), (start) ), (end) ))
#define correct_color(cdfval, cdfmin) ((float)clamp( (float)(255.0 * (cdfval - cdfmin)/(1.0 - cdfmin)), (float)0.0, (float)255.0 ))

__global__ void castImagetoUnsignedChar(float *inputImage, unsigned char *outputImage,
										int imageWidth, int imageHeight, int imageChannels) {

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (idx < imageWidth * imageHeight * imageChannels) {
		outputImage[idx] = (unsigned char)(255 * inputImage[idx]);
	}
}


__global__ void castImagetoFloat(unsigned char *inputImage, float *outputImage, 
								int imageWidth, int imageHeight, int imageChannels) {

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < imageWidth * imageHeight * imageChannels) {
		outputImage[idx] = (((float) inputImage[idx])/255.0);
	}
}


__global__ void convertRGBtoGrayScale(unsigned char *rgbImage, unsigned char *grayImage, int imageWidth, int imageHeight, int imageChannels) {
	
	int r, g, b;

	int idx = (blockIdx.x * blockDim.x + threadIdx.x);


	if (idx < imageWidth * imageHeight ) {
		
		r = rgbImage[imageChannels * idx];
		g = rgbImage[imageChannels * idx + 1];
		b = rgbImage[imageChannels * idx + 2];

		grayImage[idx] = (unsigned char)(0.21 * r + 0.71 * g + 0.07 * b);
	}
}


__global__ void computeHistogram(unsigned int* histogram, unsigned char* grayImage, int imageWidth, int imageHeight, int imageChannels){
	int idx = (blockIdx.x * blockDim.x + threadIdx.x);

	if (idx < imageWidth * imageHeight) {
		atomicAdd(&histogram[grayImage[idx]], 1);
	}
}


__global__ void computeCDF(unsigned int *histogram, float *cdf, int imageWidth, int imageHeight) {
  
	int index_device = 2*(blockIdx.x * blockDim.x + threadIdx.x);
	int index_shared = 2*threadIdx.x;
	int len = HISTOGRAM_LENGTH;

	// if(index_device == 0){
	// 	cdf[0] = ((float)histogram[0])/((float)imageWidth * imageHeight);
	// }
	// __syncthreads();

	__shared__ float In[2*BLOCK_SIZE_int];

	if(index_device < len){
		In[index_shared] = histogram[index_device];
	} else{
		In[index_shared] = 0;
	}

	if(index_device + 1 < len){
		In[index_shared + 1] = histogram[index_device + 1];
	} else{
		In[index_shared + 1] = 0;
	}
	__syncthreads();

	// initial scan
	int stride = 1;
	while(stride < 2*BLOCK_SIZE_int){
		__syncthreads();
		index_shared = 2 * (threadIdx.x + 1) * stride - 1;
		if(index_shared < 2*BLOCK_SIZE_int && index_shared - stride >= 0){
			In[index_shared] += In[index_shared - stride];
		}

		stride = 2*stride;
	}
	__syncthreads();

	// post scan
	stride = BLOCK_SIZE_int/2;
	while(stride > 0){
		__syncthreads();
		index_shared = 2 * (threadIdx.x + 1) * stride - 1;
		if(index_shared + stride < 2*BLOCK_SIZE_int){
			In[index_shared + stride] += In[index_shared];
		}

		stride = stride/2;
	}
	__syncthreads();

	index_shared = 2*threadIdx.x;
	if(index_device < len){
		cdf[index_device] = ((float)In[index_shared])/((float)imageWidth * imageHeight);
	}

	if(index_device + 1 < len){
		cdf[index_device + 1] = ((float)In[index_shared + 1])/((float)imageWidth * imageHeight);
	}

	__syncthreads();
}


__global__ void histogramEqualization(unsigned char *inputImage, float *outputImage, float *cdf, int imageWidth, int imageHeight, int imageChannels) {
	
	int idx = (blockIdx.x * blockDim.x + threadIdx.x);

	if (idx < imageWidth * imageHeight * imageChannels) {
		unsigned char val = inputImage[idx];
		outputImage[idx] = correct_color(cdf[val], cdf[0])/255.0;
	}
}


void printImageData_f(float* inputImage, int imageWidth, int imageHeight, int imageChannels) {
	for (int row = 0; row < imageHeight; row++) {
		for (int col = 0; col < imageWidth; col++) {
			for (int channel = 0; channel < imageChannels; channel++) {
				int idx = (row * imageWidth + col) * imageChannels + channel;
				printf("%f ", inputImage[idx]);
			}
		}
		printf("\n");
	}
	printf("\n");
}


void printImageData_uc(unsigned char* inputImage, int imageWidth, int imageHeight, int imageChannels) {
	for (int row = 0; row < imageHeight; row++) {
		printf("row: %d", row);
		for (int col = 0; col < imageWidth; col++) {
			printf("[ ");
			for (int channel = 0; channel < imageChannels; channel++) {
				int idx = (row * imageWidth + col) * imageChannels + channel;
				printf("%u ", inputImage[idx]);
			}
			printf("]");
		}
		printf("\n");
	}
	printf("\n");
}


int main(int argc, char **argv) {
	wbArg_t args;
	int imageWidth;
	int imageHeight;
	int imageChannels;
	wbImage_t inputImage;
	wbImage_t outputImage;
	float *hostInputImageData;
	float *deviceInputImageData;
	unsigned char *deviceCharImageData;
	unsigned char *deviceGrayScaleImageData;
	unsigned int *deviceHistogram;
	float *deviceCDF;
	float *deviceOutputImageData;
	float *hostOutputImageData;

	// DEBUG
	// unsigned char *hostCharImageData;

	const char *inputImageFile;

	args = wbArg_read(argc, argv); /* parse the input arguments */

	inputImageFile = wbArg_getInputFile(args, 0);

	wbTime_start(Generic, "Importing data and creating memory on host");
	inputImage = wbImport(inputImageFile);
	imageWidth = wbImage_getWidth(inputImage);
	imageHeight = wbImage_getHeight(inputImage);
	imageChannels = wbImage_getChannels(inputImage);
	outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);
	hostInputImageData = wbImage_getData(inputImage);
	hostOutputImageData = wbImage_getData(outputImage);
	wbTime_stop(Generic, "Importing data and creating memory on host");

	cudaMalloc((void **) &deviceInputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
	cudaMalloc((void **) &deviceCharImageData, imageWidth * imageHeight * imageChannels * sizeof(unsigned char));
	cudaMalloc((void **) &deviceGrayScaleImageData, imageWidth * imageHeight * sizeof(unsigned char));
	cudaMalloc((void **) &deviceHistogram, HISTOGRAM_LENGTH * sizeof(unsigned int));
	cudaMalloc((void **) &deviceCDF, HISTOGRAM_LENGTH * sizeof(float));
	cudaMalloc((void **) &deviceOutputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));

	cudaMemcpy(deviceInputImageData, hostInputImageData, imageWidth * imageHeight * imageChannels * sizeof(float), cudaMemcpyHostToDevice);

	// DEBUG
	// printf("Input Image Data (float):\n");
	// for(int j = 0; j < imageWidth; j++){
	// 	for(int c = 0; c < imageChannels; c++){
	// 		printf("hostInputImageData[0][%d][%d]: %f\n", j, c, hostInputImageData[j*imageChannels+c]);
	// 	}
	// }

	// Step 1: Cast image from float to unsigned char
	dim3 dimCastBlock(BLOCK_SIZE, 1, 1);
	dim3 dimCastGrid(ceil((imageWidth * imageHeight * imageChannels)/BLOCK_SIZE), 1, 1);
	castImagetoUnsignedChar<<<dimCastGrid, dimCastBlock>>>(deviceInputImageData, deviceCharImageData, imageWidth, imageHeight, imageChannels);

	// Step 1: DEBUG
	// unsigned char *hostCharImageData;
	// hostCharImageData = (unsigned char *) malloc(sizeof(unsigned char) * imageWidth * imageHeight * imageChannels);
	// cudaMemcpy(hostCharImageData, deviceCharImageData, imageWidth * imageHeight * imageChannels * sizeof(unsigned char), cudaMemcpyDeviceToHost);

	// printf("Input Image Data (unsigned char):\n");
	// for(int j = 0; j < imageWidth; j++){
	// 	for(int c = 0; c < imageChannels; c++){
	// 		printf("deviceCharImageData[0][%d][%d]: %u\n", j, c, hostCharImageData[j*imageChannels+c]);
	// 	}
	// }

	// free(hostCharImageData);

	// DEBUG
	// hostCharImageData = (unsigned char*)malloc(imageWidth * imageHeight * imageChannels * sizeof(unsigned char));
	// cudaMemcpy(hostCharImageData, deviceCharImageData, imageWidth * imageHeight * imageChannels * sizeof(unsigned char), cudaMemcpyDeviceToHost);
	// printImageData_uc(hostCharImageData, imageWidth, imageHeight, imageChannels);
	
	// Step 2: Convert image from RGB to GrayScale
	dim3 dimGrayBlock(BLOCK_SIZE, 1, 1);
	dim3 dimGrayGrid(ceil((imageWidth * imageHeight)/BLOCK_SIZE), 1, 1);
	convertRGBtoGrayScale<<<dimGrayGrid, dimGrayBlock>>>(deviceCharImageData, deviceGrayScaleImageData, imageWidth, imageHeight, imageChannels);
	


	// Step 2: DEBUG
	// unsigned char *hostGrayScaleImageData;
	// hostGrayScaleImageData = (unsigned char *) malloc(sizeof(unsigned char) * imageWidth * imageHeight);
	// cudaMemcpy(hostGrayScaleImageData, deviceGrayScaleImageData, imageWidth * imageHeight * sizeof(unsigned char), cudaMemcpyDeviceToHost);
	
	// printf("Grayscale Image Data (unsigned char):\n");
	// for(int j = 0; j < imageWidth; j++){
	// 	printf("deviceGrayScaleImageData[0][%d]: %u\n", j, hostGrayScaleImageData[j]);
	// }

	// free(hostGrayScaleImageData);

	cudaMemset(deviceHistogram, 0, HISTOGRAM_LENGTH * sizeof(unsigned int));

	// Step 3: Compute histogram of grayscale image
	dim3 dimHistBlock(BLOCK_SIZE, 1, 1);
	dim3 dimHistGrid(ceil((imageWidth * imageHeight)/BLOCK_SIZE), 1, 1);
	computeHistogram<<<dimHistGrid, dimHistBlock>>>(deviceHistogram, deviceGrayScaleImageData, imageWidth, imageHeight, imageChannels);

	// Step 3: DEBUG
	// unsigned int *hostHistogram;
	// hostHistogram = (unsigned int *) malloc(sizeof(unsigned int) * HISTOGRAM_LENGTH);
	// cudaMemcpy(hostHistogram, deviceHistogram, HISTOGRAM_LENGTH * sizeof(unsigned int), cudaMemcpyDeviceToHost);

	// printf("Histogram Data (int):\n");
	// for(int j = 0; j < HISTOGRAM_LENGTH; j++){
	// 	printf("deviceHistogram[%d]: %u\n", j, hostHistogram[j]);
	// }

	// free(hostHistogram);

	// Step 4: Compute CDF of histogram
	dim3 dimCDFBlock(BLOCK_SIZE, 1, 1);
	dim3 dimCDFGrid(ceil((imageWidth * imageHeight)/(2 * BLOCK_SIZE)), 1, 1);
	computeCDF<<<dimCDFGrid, dimCDFBlock>>>(deviceHistogram, deviceCDF, imageWidth, imageHeight);

	// Step 4: DEBUG
	// float *hostCDF;
	// hostCDF = (float *) malloc(sizeof(float) * HISTOGRAM_LENGTH);
	// cudaMemcpy(hostCDF, deviceCDF, HISTOGRAM_LENGTH * sizeof(float), cudaMemcpyDeviceToHost);

	// printf("CDF Data (float):\n");
	// for(int j = 0; j < HISTOGRAM_LENGTH; j++){
	// 	printf("deviceCDF[%d]: %f\n", j, hostCDF[j]);
	// }

	// free(hostCDF);

	// Step 5: Apply histogram equalization function
	dim3 dimEqBlock(BLOCK_SIZE, 1, 1);
	dim3 dimEqGrid(ceil((imageWidth * imageHeight * imageChannels)/BLOCK_SIZE), 1, 1);
	histogramEqualization<<<dimEqGrid, dimEqBlock>>>(deviceCharImageData, deviceOutputImageData, deviceCDF, imageWidth, imageHeight, imageChannels);

	// Step 5: DEBUG
	// hostCharImageData = (unsigned char *) malloc(sizeof(unsigned char) * imageWidth * imageHeight * imageChannels);
	
	// cudaMemcpy(hostCharImageData, deviceCharImageData, imageWidth * imageHeight * imageChannels * sizeof(unsigned char), cudaMemcpyDeviceToHost);

	// printf("Equalized Image Data (unsigned char):\n");
	// for(int j = 0; j < imageWidth; j++){
	// 	for(int c = 0; c < imageChannels; c++){
	// 		printf("deviceCharImageData[0][%d][%d]: %u\n", j, c, hostCharImageData[j*imageChannels+c]);
	// 	}
	// }

	// free(hostCharImageData);

	// cudaMemset(deviceOutputImageData, 123, imageWidth * imageHeight * imageChannels * sizeof(float));

	// Step 6: Cast image from unsigned char back to float
	// dim3 dimFloatBlock(BLOCK_SIZE, 1, 1);
	// dim3 dimFloatGrid(ceil((imageWidth * imageHeight * imageChannels)/BLOCK_SIZE), 1, 1);
	// castImagetoFloat<<<dimFloatGrid, dimFloatBlock>>>(deviceCharImageData, deviceOutputImageData, imageWidth, imageHeight, imageChannels);
	cudaMemcpy(hostOutputImageData, deviceOutputImageData, imageWidth * imageHeight * imageChannels * sizeof(float), cudaMemcpyDeviceToHost);
	
	// Step 6: DEBUG

	// printf("Equalized Image Data (float):\n");
	// for(int j = 0; j < imageWidth; j++){
	// 	for(int c = 0; c < imageChannels; c++){
	// 		printf("hostOutputImageData[0][%d][%d]: %f\n", j, c, hostOutputImageData[j*imageChannels+c]);
	// 	}
	// }

	
	// cudaMemcpy(outputImage, deviceOutputImageData, imageWidth * imageHeight * imageChannels * sizeof(float), cudaMemcpyDeviceToHost);

	wbSolution(args, outputImage);

	// FILE *fptr = fopen("outputimg.ppm", "w");
	// const char * file = "outputimgdata.ppm";
	// wbExport(file, hostOutputImageData);
	
	const char * file = "outputimg.ppm";
	wbExport(file, outputImage);
	// fclose(fptr);

  	cudaFree(deviceInputImageData); 
	cudaFree(deviceCharImageData);
	cudaFree(deviceGrayScaleImageData);
	cudaFree(deviceHistogram); 
	cudaFree(deviceCDF); 
	cudaFree(deviceOutputImageData);

  return 0;
}
