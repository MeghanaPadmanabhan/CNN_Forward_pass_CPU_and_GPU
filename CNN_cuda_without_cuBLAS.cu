//Training of the CNN is done using Keras. After training for 10 epochs, the obtained accuracy on the training data set is 99.70 and on the test data set is 99.14. 
//This model implements the following layes in order- 2DConvolution---->Maxpooling---->2D Convolution---->Maxpooling---->Fully_connected layer---->Fully_connected layer.
//The image is a 28*28 greyscale image. The specifications of the layers are as follows:
//Layer_0: Convolution: 32 3*3 kernels with no padding and 1 stride.
//Layer_1: Maxpooling: 2*2 filters with with no padding and 1 stride.
//Layer_2: Convolution: 64 3*3 kernels with no padding and 1 stride.
//Layer_3: Maxpooling: 2*2 filters with with no padding and 1 stride.
//Layer_4: Flattening 
//Layer_5: Fully connected / dense layer with 1024 output units.
//Layer_6: Dropout (done during training only).
//Layer_7: Fully connected / dense layer with 10 output units.

//All arrays and matrices are designed to be row ordered in this implementation.

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include<stdlib.h>
#include<math.h>


//Kernel that does convolution. This convolution is done by each thread identifying that patch or portion of the image that it is responsible for its result and does the multiplication and addition of it's patche's values with the suitable kernel.
//The depth of the output image is the number of kernels.
__global__ void convolution_kernel(int h, int w, int d, double* gpu_in, int k_h, int k_w, int k_d, double* kernel_weights, double* kernel_biases, int num_kernels, int op_h, int op_w, int op_d, double* gpu_out)
{	
	//Identifying threads by their IDs.
	int row = blockDim.y*blockIdx.y + threadIdx.y;
	int col = blockDim.x*blockIdx.x + threadIdx.x;
	int deep = blockDim.z *blockIdx.z + threadIdx.z;
	//Return if thread out of bounds
	if (row >= op_h || col >= op_w || deep >= op_d) return;
	double out=0.0;
	int kernel_pointer = 0;
	//Each thread/each output node identifies the corresponding element in the matrix that it is responsible to multiply-add.
	for (int depth_pointer = 0; depth_pointer < k_d; depth_pointer++) {
		for (int row_pointer = 0; row_pointer < k_h; row_pointer++) {
			for (int column_pointer = 0; column_pointer < k_w; column_pointer++) {
				out += gpu_in[((row*w + col) + row_pointer * w + column_pointer + h * w*depth_pointer)] * kernel_weights[kernel_pointer + deep * k_h*k_w*k_d];
				kernel_pointer++;
			}
		}
	}
	//Bias addition and relu activation. One bias is applied to one output image layer, since one bias is applicable to one kernel.
	//Relu activation : relu(a)=max(0,a). If the value is less than 0 then it becomes 0, else it is retained.
	if (out + kernel_biases[deep] < 0.0)
		gpu_out[row*op_w + col + deep * op_h*op_w] = 0.0l;
	else
		gpu_out[row*op_w + col + deep * op_h*op_w] = out + kernel_biases[deep];

}

//Kernel that does maxpooling.
__global__ void maxpool_kernel(int h, int w, int d, double* gpu_in, int pool_height, int pool_width, int op_h, int op_w, int op_d, double* gpu_out)
{
	int row = blockDim.y*blockIdx.y + threadIdx.y;
	int col = blockDim.x*blockIdx.x + threadIdx.x;
	int deep = blockDim.z *blockIdx.z + threadIdx.z;
	if (row >= op_h || col >= op_w || deep >= op_d) return;
	double out;
	double max;
	int kernel_pointer = 0;
	//The maximum is chosen to be the first element of the pool filter.
	max = gpu_in[(deep*w*h) + (row*pool_height)*w + (col*pool_width)];
	//We follow throgh all the elements within the filter's size and look for the maximum element and the corresponding maximum element becomes the thread's value.
	for (int row_pointer = 0; row_pointer < pool_height; row_pointer++) {
		for (int column_pointer = 0; column_pointer < pool_width; column_pointer++) {
			if (gpu_in[(deep*w*h) + (row*pool_height)*w + (col*pool_width) + (row_pointer*w) + (column_pointer)] > max)
				max = gpu_in[(deep*w*h) + (row*pool_height)*w + (col*pool_width) + (row_pointer*w) + (column_pointer)];
		}
	}
	gpu_out[deep*op_w*op_h + row * op_w + col] = max;
}

//This kernel implements the fully connected layers.
__global__ void dense_kernel(int num_input, int num_output, double* gpu_in, double* weights, double* biases, double* gpu_out, int num_classes)
{	
	int tid = blockDim.x*blockIdx.x + threadIdx.x;
	if (tid >= num_output) return;
	double sum = 0.0l;
	//The weights are extracted from Keras such that all the weights to one output node appears together, followed by weights to the next node and so on.
	//Thus, each output node will be a multiply add of adjacent weight values with the input nodes.
	for (int count = 0; count < num_input; count++) {
		sum += gpu_in[count] * weights[tid*num_input + count];
	}
	sum += biases[tid];

	//Activation: If the layer is the final layer, then don't do anything, otherwise relu activation max(0,value) is taken.
	if ((num_output) != num_classes) {
		if (sum < 0.0) {
			sum = 0.0l;
		}
	}
	gpu_out[tid] = sum;
}


int main()
{

	//-------------------------------Reading all the weights and biases and the original image----------------------//
	//File pointers to all the weights and biases and the image.
	FILE * pFileImg;
	FILE * pFileW0;
	FILE * pFileB0;
	FILE * pFileW2;
	FILE * pFileB2;
	FILE * pFileDW5;
	FILE * pFileDB5;
	FILE * pFileDW7;
	FILE * pFileDB7;

	//Note: The weights are pulled out after training the mnist digit recognition dataset on keras with handwritten digits 0-9. The images are greysvale and hence to start with they have only one channel. 
	//Weights are pulled out and inputted into the respective arrays.
	//Pulling out image values
	double* img_arr = (double *)malloc(28 * 28 * sizeof(double));
	pFileImg = fopen("/home/meghanap/Image_RO.txt", "r");
	if (pFileImg == NULL) { fputs("File error", stderr); exit(1); }
	for (int i = 0; i < 784; i++) {
		fscanf(pFileImg, "%lf", &img_arr[i]);
	}

	//Pulling out kernel weights for first conv layer.
	double* W0_arr = (double *)malloc(288 * sizeof(double));
	pFileW0 = fopen("/home/meghanap/W0_RO.txt", "r");
	if (pFileW0 == NULL) { fputs("File error", stderr); exit(1); }
	for (int i = 0; i < 288; i++) {
		fscanf(pFileW0, "%lf", &W0_arr[i]);
	}

	//Pulling out kernel biases for first conv layer.
	double* B0_arr = (double *)malloc(32 * sizeof(double));
	pFileB0 = fopen("/home/meghanap/B0.txt", "r");
	if (pFileB0 == NULL) { fputs("File error", stderr); exit(1); }
	for (int i = 0; i < 32; i++) {
		fscanf(pFileB0, "%lf", &B0_arr[i]);
	}


	//Pulling out kernel weights for second conv layer.
	double* W2_arr = (double *)malloc(18432 * sizeof(double));
	pFileW2 = fopen("/home/meghanap/W2_RO.txt", "r");
	if (pFileW2 == NULL) { fputs("File error", stderr); exit(1); }
	for (int i = 0; i < 18432; i++) {
		fscanf(pFileW2, "%lf", &W2_arr[i]);
	}

	//Pulling out kernel biases for second conv layer.
	double* B2_arr = (double *)malloc(64 * sizeof(double));
	pFileB2 = fopen("/home/meghanap/B2.txt", "r");
	if (pFileB2 == NULL) { fputs("File error", stderr); exit(1); }
	for (int i = 0; i < 64; i++) {
		fscanf(pFileB2, "%lf", &B2_arr[i]);
	}


	//Pulling out weights for first fully connected layer.
	double* DW5_arr = (double *)malloc(1638400 * sizeof(double));
	pFileDW5 = fopen("/home/meghanap/DW5_RO.txt", "r");
	if (pFileDW5 == NULL) { fputs("File error", stderr); exit(1); }
	for (int i = 0; i < 1638400; i++) {
		fscanf(pFileDW5, "%lf", &DW5_arr[i]);
	}

	//Pulling out biases for first fully connected layer.
	double* DB5_arr = (double *)malloc(1024 * sizeof(double));
	pFileDB5 = fopen("/home/meghanap/DB5.txt", "r");
	if (pFileDB5 == NULL) { fputs("File error", stderr); exit(1); }
	for (int i = 0; i < 1024; i++) {
		fscanf(pFileDB5, "%lf", &DB5_arr[i]);
	}

	//Pulling out weights for second fully connected layer.
	double* DW7_arr = (double *)malloc(10240 * sizeof(double));
	pFileDW7 = fopen("/home/meghanap/DW7_RO.txt", "r");
	if (pFileDW7 == NULL) { fputs("File error", stderr); exit(1); }
	for (int i = 0; i < 10240; i++) {
		fscanf(pFileDW7, "%lf", &DW7_arr[i]);
	}


	//Pulling out biases for second fully connected layer.
	double* DB7_arr = (double *)malloc(10 * sizeof(double));
	pFileDB7 = fopen("/home/meghanap/DB7.txt", "r");
	if (pFileDB7 == NULL) { fputs("File error", stderr); exit(1); }
	for (int i = 0; i < 10; i++) {
		fscanf(pFileDB7, "%lf", &DB7_arr[i]);
	}

	//-------------------------------------Reading done------------------------------------------------//

	int number_of_classes = 10;
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	int max_threads_per_block = prop.maxThreadsPerBlock;

	//Convolution kernel preparation.
	int input_image_height = 28;
	int input_image_width = 28;
	int input_image_depth = 1;
	int kernel_height = 3;
	int kernel_width = 3;
	int kernel_depth = 1;
	int no_of_kernels = 32;
	int output_image_height = input_image_height - kernel_height + 1;
	int output_image_width = input_image_width - kernel_width + 1;
	int output_image_depth = no_of_kernels;
	//Defined 3 D blocks with z_threads=no_of_kernels and x_threads*y_threads*z_threads=max_threads_per_block. So, if x_threads=y_threads, then x_threads=sqrt(max_threads_per_block/z_threads). 
	//Defined 2 D grids.
	int z_threads = no_of_kernels;
	int x_threads = sqrt(max_threads_per_block / z_threads);
	int y_threads = x_threads;
	dim3 blockdim0(x_threads, y_threads, z_threads);
	dim3 griddim0(output_image_width / x_threads, output_image_height / y_threads, 1);
	//Copying the image into GPU.
	double *gpu_img0;
	cudaMalloc((void **)&gpu_img0, 28 * 28 * sizeof(double));
	cudaMemcpy(gpu_img0, img_arr, 28 * 28 * sizeof(double), cudaMemcpyHostToDevice);
	//Copying the kernel weights into GPU.
	double *kernel_weights0;
	cudaMalloc((void **)&kernel_weights0, 3 * 3 * 32 * sizeof(double));
	cudaMemcpy(kernel_weights0, W0_arr, 3 * 3 * 32 * sizeof(double), cudaMemcpyHostToDevice);
	//Copying kernel biases into GPU.
	double *kernel_biases0;
	cudaMalloc((void **)&kernel_biases0, 32 * sizeof(double));
	cudaMemcpy(kernel_biases0, B0_arr, 32 * sizeof(double), cudaMemcpyHostToDevice);
	//Creating output array inside GPU.
	double *gpu_out0;
	cudaMalloc((void **)&gpu_out0, 26 * 26 * 32 * sizeof(double));
	convolution_kernel << <griddim0, blockdim0 >> > (input_image_height, input_image_width, input_image_depth, gpu_img0, kernel_height, kernel_width, kernel_depth, kernel_weights0, kernel_biases0, no_of_kernels, output_image_height, output_image_width, output_image_depth, gpu_out0);
	double* layer_0 = (double *)malloc(26 * 26 * 32 * sizeof(double));
	cudaMemcpy(layer_0, gpu_out0, 26 * 26 * 32 * sizeof(double), cudaMemcpyDeviceToHost);
	//***layer_0 is the output from the first layer.
	//Free all the unnecessary things from the GPU to make space for the next kernel.
	cudaFree(gpu_img0);
	cudaFree(kernel_weights0);
	cudaFree(kernel_biases0);
	cudaFree(gpu_out0);


	//Maxpooling layer kernel preparation.
	int pool_height = 3;
	int pool_width = 3;
	input_image_height = output_image_height;
	input_image_width = output_image_width;
	input_image_depth = output_image_depth;
	z_threads = input_image_depth;
	x_threads = sqrt(max_threads_per_block / z_threads);
	y_threads = x_threads;
	output_image_height = (input_image_height - (input_image_height % pool_height)) / pool_height;
	output_image_width = (input_image_width - (input_image_width % pool_width)) / pool_width;
	output_image_depth = input_image_depth;
	dim3 blockdim1(x_threads, y_threads, z_threads);
	dim3 griddim1(output_image_width / x_threads, output_image_height / y_threads, 1);
	//Copying the previous output into GPU.
	double *gpu_in1;
	cudaMalloc((void **)&gpu_in1, input_image_height*input_image_width*input_image_depth * sizeof(double));
	cudaMemcpy(gpu_in1, layer_0, input_image_height*input_image_width*input_image_depth * sizeof(double), cudaMemcpyHostToDevice);
	//Creating output array inside GPU.
	double *gpu_out1;
	cudaMalloc((void **)&gpu_out1, output_image_height*output_image_width*output_image_depth * sizeof(double));
	maxpool_kernel << <griddim1, blockdim1 >> > (input_image_height, input_image_width, input_image_depth, gpu_in1, pool_height, pool_width, output_image_height, output_image_width, output_image_depth, gpu_out1);
	double* layer_1 = (double *)malloc(output_image_height*output_image_width*output_image_depth * sizeof(double));
	cudaMemcpy(layer_1, gpu_out1, output_image_height*output_image_width*output_image_depth * sizeof(double), cudaMemcpyDeviceToHost);
	//**layer 1 is the output.
	cudaFree(gpu_in1);
	cudaFree(gpu_out1);

	//Convolution layer preparation.
	input_image_height = output_image_height;
	input_image_width = output_image_width;
	input_image_depth = output_image_depth;
	kernel_height = 3;
	kernel_width = 3;
	kernel_depth = 32;
	no_of_kernels = 64;
	output_image_height = input_image_height - kernel_height + 1;
	output_image_width = input_image_width - kernel_width + 1;
	output_image_depth = no_of_kernels;
	//Defined 3 D blocks with z_threads=no_of_kernels and x_threads*y_threads*z_threads=max_threads_per_block. So, if x_threads=y_threads, then x_threads=sqrt(max_threads_per_block/z_threads). 
	//Defined 2 D grids.
	z_threads = no_of_kernels;
	x_threads = sqrt(max_threads_per_block / z_threads);
	y_threads = x_threads;
	dim3 blockdim2(x_threads, y_threads, z_threads);
	dim3 griddim2(output_image_width / x_threads, output_image_height / y_threads, 1);
	//Copying input into GPU.
	double* gpu_in2;
	cudaMalloc((void**)&gpu_in2, input_image_height*input_image_width *input_image_depth * sizeof(double));
	cudaMemcpy(gpu_in2, layer_1, input_image_height*input_image_width *input_image_depth * sizeof(double), cudaMemcpyHostToDevice);
	//Copying kernels weights into GPU.
	double *kernel_weights2;
	cudaMalloc((void **)&kernel_weights2, kernel_height*kernel_width*kernel_depth*no_of_kernels * sizeof(double));
	cudaMemcpy(kernel_weights2, W2_arr, kernel_height*kernel_width*kernel_depth*no_of_kernels * sizeof(double), cudaMemcpyHostToDevice);
	//Copying kernel biases into GPU.
	double *kernel_biases2;
	cudaMalloc((void **)&kernel_biases2, no_of_kernels * sizeof(double));
	cudaMemcpy(kernel_biases2, B2_arr, no_of_kernels * sizeof(double), cudaMemcpyHostToDevice);
	//Creating output array inside GPU.
	double *gpu_out2;
	cudaMalloc((void **)&gpu_out2, output_image_height*output_image_width*output_image_depth * sizeof(double));
	convolution_kernel << <griddim2, blockdim2 >> > (input_image_height, input_image_width, input_image_depth, gpu_in2, kernel_height, kernel_width, kernel_depth, kernel_weights2, kernel_biases2, no_of_kernels, output_image_height, output_image_width, output_image_depth, gpu_out2);
	double* layer_2 = (double *)malloc(output_image_height*output_image_width*output_image_depth * sizeof(double));
	cudaMemcpy(layer_2, gpu_out2, output_image_height*output_image_width*output_image_depth * sizeof(double), cudaMemcpyDeviceToHost);
	//**Layer 2 is the output.
	cudaFree(gpu_in2);
	cudaFree(gpu_out2);
	cudaFree(kernel_weights2);
	cudaFree(kernel_biases2);

	//Maxpooling layer.
	pool_height = 3;
	pool_width = 3;
	input_image_height = output_image_height;
	input_image_width = output_image_width;
	input_image_depth = output_image_depth;
	z_threads = input_image_depth;
	x_threads = sqrt(max_threads_per_block / z_threads);
	y_threads = x_threads;
	int excess_w = input_image_height % pool_height;
	int excess_h = input_image_height % pool_height;
	output_image_height = (input_image_height) / pool_height;
	output_image_width = (input_image_width) / pool_width;
	output_image_depth = input_image_depth;
	dim3 blockdim3(x_threads, y_threads, z_threads);
	dim3 griddim3(output_image_width / x_threads, output_image_height / y_threads, 1);
	//Copying the previous output into GPU.
	double *gpu_in3;
	cudaMalloc((void **)&gpu_in3, input_image_height*input_image_width*input_image_depth * sizeof(double));
	cudaMemcpy(gpu_in3, layer_2, input_image_height*input_image_width*input_image_depth * sizeof(double), cudaMemcpyHostToDevice);
	//Creating output array inside GPU.
	double *gpu_out3;
	cudaMalloc((void **)&gpu_out3, output_image_height*output_image_width*output_image_depth * sizeof(double));
	maxpool_kernel << <griddim3, blockdim3 >> > (input_image_height, input_image_width, input_image_depth, gpu_in3, pool_height, pool_width, output_image_height, output_image_width, output_image_depth, gpu_out3);
	double* layer_3 = (double *)malloc(output_image_height*output_image_width*output_image_depth * sizeof(double));
	cudaMemcpy(layer_3, gpu_out3, output_image_height*output_image_width*output_image_depth * sizeof(double), cudaMemcpyDeviceToHost);
	//**layer 1 is the output.
	cudaFree(gpu_in3);
	cudaFree(gpu_out3);

	//Flattening in the CPU itself.
	//The idea is to apply the same kind of C major flattening that keras does to the elements coming in from the second pooling layer.
	//The array coming in consists of rows of each sheet arranged side by side followed by the rows of the next sheet and so on. Jumbling up that order to stick with keras type flattening which is the C-major ordering consisting of z-axis changing fastest, follwed by column and then row changing.
	int in_h = output_image_height;
	int in_w = output_image_width;
	int in_d = output_image_depth;
	int image_pointer;
	int channel_pointer;
	int k = 0;
	double* flattened = (double *)malloc(in_h*in_w*in_d * sizeof(double));
	for (image_pointer = 0; image_pointer < in_h*in_w; image_pointer++) {
		for (channel_pointer = 0; channel_pointer < in_d; channel_pointer++) {
			flattened[k] = layer_3[image_pointer + channel_pointer * in_h*in_w];
			k++;
		}
	}

	//Fully connected/Dense layer.
	int input_layer_nodes = output_image_height * output_image_width*output_image_depth;
	int output_layer_nodes = 1024;
	double *gpu_in5;
	cudaMalloc((void **)&gpu_in5, input_layer_nodes * sizeof(double));
	cudaMemcpy(gpu_in5, flattened, input_layer_nodes * sizeof(double), cudaMemcpyHostToDevice);
	double *FC_weights5;
	cudaMalloc((void **)&FC_weights5, input_layer_nodes *output_layer_nodes * sizeof(double));
	cudaMemcpy(FC_weights5, DW5_arr, input_layer_nodes *output_layer_nodes * sizeof(double), cudaMemcpyHostToDevice);
	double *FC_biases5;
	cudaMalloc((void **)&FC_biases5, output_layer_nodes * sizeof(double));
	cudaMemcpy(FC_biases5, DB5_arr, output_layer_nodes * sizeof(double), cudaMemcpyHostToDevice);
	double *gpu_out5;
	cudaMalloc((void **)&gpu_out5, output_layer_nodes * sizeof(double));
	dim3 blocksize5(max_threads_per_block, 1, 1);
	dim3 gridsize5(output_layer_nodes / max_threads_per_block, 1, 1);
	dense_kernel << <gridsize5, blocksize5 >> > (input_layer_nodes, output_layer_nodes, gpu_in5, FC_weights5, FC_biases5, gpu_out5, number_of_classes);
	//**layer5 is the output.
	double* layer_5 = (double *)malloc(output_layer_nodes * sizeof(double));
	cudaMemcpy(layer_5, gpu_out5, output_layer_nodes * sizeof(double), cudaMemcpyDeviceToHost);
	cudaFree(gpu_in5);
	cudaFree(gpu_out5);

	//Fully connected/dense layer.
	input_layer_nodes = output_layer_nodes;
	output_layer_nodes = number_of_classes;
	double *gpu_in7;
	cudaMalloc((void **)&gpu_in7, input_layer_nodes * sizeof(double));
	cudaMemcpy(gpu_in7, layer_5, input_layer_nodes * sizeof(double), cudaMemcpyHostToDevice);
	double *FC_weights7;
	cudaMalloc((void **)&FC_weights7, input_layer_nodes *output_layer_nodes * sizeof(double));
	cudaMemcpy(FC_weights7, DW7_arr, input_layer_nodes *output_layer_nodes * sizeof(double), cudaMemcpyHostToDevice);
	double *FC_biases7;
	cudaMalloc((void **)&FC_biases7, output_layer_nodes * sizeof(double));
	cudaMemcpy(FC_biases7, DB7_arr, output_layer_nodes * sizeof(double), cudaMemcpyHostToDevice);
	double *gpu_out7;
	cudaMalloc((void **)&gpu_out7, output_layer_nodes * sizeof(double));
	dim3 blocksize7(max_threads_per_block, 1, 1);
	dim3 gridsize7(output_layer_nodes / max_threads_per_block, 1, 1);
	dense_kernel << <gridsize7, blocksize7 >> > (input_layer_nodes, output_layer_nodes, gpu_in7, FC_weights7, FC_biases7, gpu_out7, number_of_classes);
	double* layer_7 = (double *)malloc(output_layer_nodes * sizeof(double));
	cudaMemcpy(layer_7, gpu_out7, output_layer_nodes * sizeof(double), cudaMemcpyDeviceToHost);
	//**layer7 is the output.
	cudaFree(gpu_in7);
	cudaFree(gpu_out7);

	//Softmax of the output layer.
	int op_layer_size = number_of_classes;
	int i;
	double sum = 0.0;
	for (i = 0; i < op_layer_size; i++) {
		sum += exp(layer_7[i]);
	}

	double max = layer_7[0] / sum;
	int max_no = 0;
	for (i = 0; i < op_layer_size; i++) {
		if ((layer_7[i] / sum) > max) {
			max_no = i;
		}
	}
	printf("\n The written predicted digit is %d\n", max_no);
}
