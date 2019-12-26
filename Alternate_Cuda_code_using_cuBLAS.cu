In this attempt, we plan on using the cudasgemm approach to do the convolution. Problem cublasSgemm does only float multiplication of the matrices and we are looking for double precision. How will it affect my accuracy
Training of the CNN is done using Keras. After training for 10 epochs, the obtained accuracy on the training data set is 99.70 and on the test data set is 99.14. 
This model implements the following layes in order- 2DConvolution----Maxpooling----2D Convolution----Maxpooling----Fully_connected layer----Fully_connected layer.
The image is a 2828 greyscale image. The specifications of the layers are as follows
Layer_0 Convolution 32 33 kernels with no padding and 1 stride.
Layer_1 Maxpooling 22 filters with with no padding and 1 stride.
Layer_2 Convolution 64 33 kernels with no padding and 1 stride.
Layer_3 Maxpooling 22 filters with with no padding and 1 stride.
Layer_4 Flattening 
Layer_5 Fully connected  dense layer with 1204 output units.
Layer_6 Dropout (done during training only).
Layer_7 Fully connected  dense layer with 10 output units.

#include cuda_runtime.h
#include device_launch_parameters.h
#include stdio.h
#includestdlib.h
#includemath.h
#include thrusthost_vector.h
#include thrustdevice_vector.h
#include cublas_v2.h

Kernel that does bias addition to cublasSgemm output and Relu activation.
__global__ void bias_and_relu_kernel(float gpu_in,double kernel_biases,int op_h, int op_w, int op_d, double gpu_out)
{
	int row = blockDim.yblockIdx.y + threadIdx.y;
	int col = blockDim.xblockIdx.x + threadIdx.x;
	int deep = blockDim.z blockIdx.z + threadIdx.z;
	if (row = op_h  col = op_w  deep = op_d) return;
	float c=gpu_in[deepop_wop_h + row  op_w + col];
	Add the bias, each sheet along z represents the output from one kernel and hence all elements in that sheet (or deep) needs to be added with the corresponding bias element.
	c += kernel_biases[deep];			Float and double are added. Result is therefore double.
	Relu activation  relu(a)=max(0,a).
	if (c  0.0) { c = 0.0; }
	gpu_out[deepop_wop_h + row  op_w + col]= c;
}


Kernel that does maxpooling.
__global__ void maxpool_kernel(int h, int w, int d, double gpu_in, int pool_height, int pool_width, int op_h, int op_w, int op_d, double gpu_out)
{
	int row = blockDim.yblockIdx.y + threadIdx.y;
	int col = blockDim.xblockIdx.x + threadIdx.x;
	int deep = blockDim.z blockIdx.z + threadIdx.z;
	if (row = op_h  col = op_w  deep = op_d) return;
	double max;
	max = gpu_in[(deepwh) + (rowpool_height)w + (colpool_width)];
	for (int row_pointer = 0; row_pointer  pool_height; row_pointer++) {
		for (int column_pointer = 0; column_pointer  pool_width; column_pointer++) {
			if (gpu_in[(deepwh) + (rowpool_height)w + (colpool_width) + (row_pointerw) + (column_pointer)]  max)
			{
				max = gpu_in[(deepwh) + (rowpool_height)w + (colpool_width) + (row_pointerw) + (column_pointer)];
			}
		}
	}
	gpu_out[deepop_wop_h + row  op_w + col] = max;
}

__global__ void dense_kernel(int num_input, int num_output, double gpu_in, double weights, double biases, double gpu_out, int num_classes)
{
	int tid = blockDim.xblockIdx.x + threadIdx.x;
	if (tid = num_output) return;
	double sum = 0.0l;
	for (int count = 0; count  num_input; count++) {
		sum += gpu_in[count]  weights[tidnum_input + count];
	}
	sum += biases[tid];

	Activation If the layer is the final layer, then don't do anything (we do softmax in the CPU), otherwise relu activation max(0,value) is taken.
	if ((num_output) != num_classes) {
		if (sum  0.0) {
			sum = 0.0l;
		}
	}
	gpu_out[tid] = sum;
}

__host__ double data_patch_preparation(double in, int h, int w, int d,int k_h,int k_w,int k_d) {
		Kernels' order is perfectly fine, they are already column ordered(one kernel after the other).
		Input data needs change. It is required that the patches that are used for convolution (element wise multiplication be grouped together).
		Thus, we need to prepare the data such that there is patch after patch, where each patch is the group of elements that are used in one particular convolution.
		int op_h = h - k_h + 1;
		int op_w = w - k_w + 1;
		int k = 0;
		patches will contain all the patches in row ordered fashion. There will definitely be a repeat of the elements from the original matrix because the patches overlap.
		double patches = (double )malloc((w - k_w + 1)(h - k_h + 1)(k_hk_wk_d)  sizeof(double));
		for (int r = 0; r  op_h; r++) {
			for (int c = 0; c  op_w; c++) {
				for (int sheet_pointer = 0; sheet_pointer  k_d; sheet_pointer++) {
					for (int row_pointer = r; row_pointer  r+k_h; row_pointer++) {
						for (int column_pointer = c; column_pointer  c+k_w; column_pointer++) {
							patches[k]=in[sheet_pointerwh + row_pointer  w + column_pointer];
							k++;
						}
					}
				}
			}
		}
		Now from these row ordered patches, we convert them into column ordered fashion so that they can be passed into the cublasSgemm function.
		Size of each patch is k_hk_wk_d.
		double co_patches = (double )malloc((w - k_w + 1)(h - k_h + 1)(k_hk_wk_d)  sizeof(double)); Stands for column oredered patches
		int patch_size = k_h  k_wk_d;
		int num_patches = op_h  op_w;
		k = 0;
		for (int i = 0; i  patch_size; i++) {
			for (int j = 0; j  num_patches; j++) {
				co_patches[k] = patches[i + j  patch_size];
				k++;
			}
		}
		return co_patches;
}




int main()
{

	-------------------------------Reading all the weights and biases and the original image----------------------
	File pointers to all the weights and biases and the image.
	FILE  pFileImg;
	FILE  pFileW0;
	FILE  pFileB0;
	FILE  pFileW2;
	FILE  pFileB2;
	FILE  pFileDW5;
	FILE  pFileDB5;
	FILE  pFileDW7;
	FILE  pFileDB7;

	Note The weights are pulled out after training the mnist digit recognition dataset on keras with handwritten digits 0-9. The images are greysvale and hence to start with they have only one channel. 
	Weights are pulled out and inputted into the respective arrays.
	Pulling out image values
	double img_arr = (double )malloc(28  28  sizeof(double));
	pFileImg = fopen(CUsersmeghaDownloadsFinal_GPU_weightsImage_RO.txt, r);
	if (pFileImg == NULL) { fputs(File error, stderr); exit(1); }
	for (int i = 0; i  784; i++) {
		fscanf(pFileImg, %lf, &img_arr[i]);
	}

	Pulling out kernel weights for first conv layer.
	double W0_arr = (double )malloc(288  sizeof(double));
	pFileW0 = fopen(CUsersmeghaDownloadsFinal_GPU_weightsW0_RO.txt, r);
	if (pFileW0 == NULL) { fputs(File error, stderr); exit(1); }
	for (int i = 0; i  288; i++) {
		fscanf(pFileW0, %lf, &W0_arr[i]);
	}

	Pulling out kernel biases for first conv layer.
	double B0_arr = (double )malloc(32  sizeof(double));
	pFileB0 = fopen(CUsersmeghaDownloadsFinal_GPU_weightsB0.txt, r);
	if (pFileB0 == NULL) { fputs(File error, stderr); exit(1); }
	for (int i = 0; i  32; i++) {
		fscanf(pFileB0, %lf, &B0_arr[i]);
	}


	Pulling out kernel weights for second conv layer.
	double W2_arr = (double )malloc(18432  sizeof(double));
	pFileW2 = fopen(CUsersmeghaDownloadsFinal_GPU_weightsW2_RO.txt, r);
	if (pFileW2 == NULL) { fputs(File error, stderr); exit(1); }
	for (int i = 0; i  18432; i++) {
		fscanf(pFileW2, %lf, &W2_arr[i]);
	}

	Pulling out kernel biases for second conv layer.
	double B2_arr = (double )malloc(64  sizeof(double));
	pFileB2 = fopen(CUsersmeghaDownloadsFinal_GPU_weightsB2.txt, r);
	if (pFileB2 == NULL) { fputs(File error, stderr); exit(1); }
	for (int i = 0; i  64; i++) {
		fscanf(pFileB2, %lf, &B2_arr[i]);
	}


	Pulling out weights for first fully connected layer.
	double DW5_arr = (double )malloc(1638400  sizeof(double));
	pFileDW5 = fopen(CUsersmeghaDownloadsFinal_GPU_weightsDW5_RO.txt, r);
	if (pFileDW5 == NULL) { fputs(File error, stderr); exit(1); }
	for (int i = 0; i  1638400; i++) {
		fscanf(pFileDW5, %lf, &DW5_arr[i]);
	}

	Pulling out biases for first fully connected layer.
	double DB5_arr = (double )malloc(1024  sizeof(double));
	pFileDB5 = fopen(CUsersmeghaDownloadsFinal_GPU_weightsDB5.txt, r);
	if (pFileDB5 == NULL) { fputs(File error, stderr); exit(1); }
	for (int i = 0; i  1024; i++) {
		fscanf(pFileDB5, %lf, &DB5_arr[i]);
	}

	Pulling out weights for second fully connected layer.
	double DW7_arr = (double )malloc(10240  sizeof(double));
	pFileDW7 = fopen(CUsersmeghaDownloadsFinal_GPU_weightsDW7_RO.txt, r);
	if (pFileDW7 == NULL) { fputs(File error, stderr); exit(1); }
	for (int i = 0; i  10240; i++) {
		fscanf(pFileDW7, %lf, &DW7_arr[i]);
	}


	Pulling out biases for second fully connected layer.
	double DB7_arr = (double )malloc(10  sizeof(double));
	pFileDB7 = fopen(CUsersmeghaDownloadsFinal_GPU_weightsDB7.txt, r);
	if (pFileDB7 == NULL) { fputs(File error, stderr); exit(1); }
	for (int i = 0; i  10; i++) {
		fscanf(pFileDB7, %lf, &DB7_arr[i]);
	}
	-------------------------------------Reading done------------------------------------------------

	int number_of_classes = 10;
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	int max_threads_per_block = prop.maxThreadsPerBlock;

	--------------------------Layer_0Convolution--------------------------------------------------
	Convolution is done using the cublasSgemm function. Details on how the kernel weights and inputs are organised to perform this multipliocation is on the report.
	Convolution parameters defined (parameters are self explantory from their names).
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
	
	Block and grid dimensions definitions.
	Defined 3 D blocks with z_threads=no_of_kernels and x_threadsy_threadsz_threads=max_threads_per_block. So, if x_threads=y_threads, then x_threads=sqrt(max_threads_per_blockz_threads). 
	Defined 2 D grids.
	int z_threads = no_of_kernels;
	int x_threads = sqrt(max_threads_per_block  z_threads);
	int y_threads = x_threads;
	dim3 blockdim0(x_threads, y_threads, z_threads);
	dim3 griddim0((output_image_width  x_threads)+1, (output_image_height  y_threads)+1, 1);

	Arranging the input image and the kernel in proper order (column major) to send off to cudaSgemm for multiplication.
	double co_patches_0;
	co_patches_0=data_patch_preparation(img_arr, input_image_height, input_image_width, input_image_depth, kernel_height, kernel_width, kernel_depth); co_patches contain the pathes in column order.
	
	Note It is observed that cublasSgemm supports only float multiplication, thus the double-precision values are explicitly converted to float.
	const float A_0 = (float)co_patches_0;
	const float B_0 = (float)W0_arr;
	float C_0 = (float )malloc(output_image_heightoutput_image_widthoutput_image_depthsizeof(float));
	int m = output_image_height  output_image_width;Number of rows in A.
	int n = no_of_kernels; Number of kernels.
	int k = kernel_height  kernel_widthkernel_depth; Patch size.
	int lda = m, ldb = k, ldc = m;
	const float alf = 1;
	const float bet = 0;
	const float alpha = &alf;
	const float beta = &bet;
	 Create a handle for CUBLAS
	cublasHandle_t handle;
	cublasCreate(&handle);
	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, A_0, lda, B_0, ldb, beta, C_0, ldc);
	cublasDestroy(handle);
	The resulting column ordered matrix is the column ordered matrix, so it should have elements in the order
	(First_patch_1st kernel,second_patch_1st_kernel,third_patch_1st_kernel,.....last_patch_1st_kernel,  and so on for all the patches and kernels).
	To get the corresponding value on the output,divide C into (num_kernels) parts of (num_patches) each.
	Copy co_patches into GPU.
	float gpu_C_0;
	cudaMalloc((void )&gpu_C_0, output_image_heightoutput_image_widthkernel_heightkernel_widthkernel_depth  sizeof(float));
	cudaMemcpy(gpu_C_0, C_0, output_image_heightoutput_image_widthkernel_heightkernel_widthkernel_depth  sizeof(float), cudaMemcpyHostToDevice);
	We can do the bias addition and relu activation by writing GPU kernels for the same.
	Copying kernel biases into GPU.
	double kernel_biases_0;
	cudaMalloc((void )&kernel_biases_0, no_of_kernels  sizeof(double));
	cudaMemcpy(kernel_biases_0, B0_arr, no_of_kernels  sizeof(double), cudaMemcpyHostToDevice);
	Creating output array inside GPU.
	double gpu_out_0;
	cudaMalloc((void )&gpu_out_0, output_image_heightoutput_image_widthno_of_kernels  sizeof(double));
	bias_and_relu_kernel griddim0, blockdim0 (gpu_C_0, kernel_biases_0,output_image_height, output_image_width, output_image_depth, gpu_out_0);
	double layer_0 = (double )malloc(output_image_heightoutput_image_widthno_of_kernels sizeof(double));
	cudaMemcpy(layer_0, gpu_out_0, output_image_heightoutput_image_widthno_of_kernels  sizeof(double), cudaMemcpyDeviceToHost);
	layer_0 is the output from the first layer.
	Free all the unnecessary things from the GPU to make space for the next kernel.
	cudaFree(gpu_C_0);
	cudaFree(kernel_biases_0);
	cudaFree(gpu_out_0);
	free(co_patches_0);
	
	--------------------------Layer 0 done------------------------------------------------------------

	-------------------------------Layer 1 Maxpooling-------------------------------------------------
	Maxpooling layer kernel preparation.
	int pool_height = 3;
	int pool_width = 3;
	input_image_height = output_image_height;
	input_image_width = output_image_width;
	input_image_depth = output_image_depth;
	z_threads = input_image_depth;
	x_threads = sqrt(max_threads_per_block  z_threads);
	y_threads = x_threads;
	When faced with image dimensions not perfectly devisible by the pool dimension, Keras removes the excess indivisible rows and columns before pooling. Doing the same thing here.
	output_image_height = (input_image_height - input_image_height % pool_height)  pool_height;
	output_image_width = (input_image_width - input_image_width % pool_width)  pool_width;
	output_image_depth = input_image_depth;
	dim3 blockdim1(x_threads, y_threads, z_threads);
	dim3 griddim1((output_image_width  x_threads)+1, (output_image_height  y_threads)+1, 1);
	Copying the previous output into GPU.
	double gpu_in_1;
	cudaMalloc((void )&gpu_in_1, input_image_heightinput_image_widthinput_image_depth  sizeof(double));
	cudaMemcpy(gpu_in_1, layer_0, input_image_heightinput_image_widthinput_image_depth  sizeof(double), cudaMemcpyHostToDevice);
	Creating output array inside GPU.
	double gpu_out_1;
	cudaMalloc((void )&gpu_out_1, output_image_heightoutput_image_widthoutput_image_depth  sizeof(double));
	maxpool_kernel  griddim1, blockdim1   (input_image_height, input_image_width, input_image_depth, gpu_in_1, pool_height, pool_width, output_image_height, output_image_width, output_image_depth, gpu_out_1);
	double layer_1 = (double )malloc(output_image_heightoutput_image_widthoutput_image_depth  sizeof(double));
	cudaMemcpy(layer_1, gpu_out_1, output_image_heightoutput_image_widthoutput_image_depth  sizeof(double), cudaMemcpyDeviceToHost);
	layer 1 is the output.
	cudaFree(gpu_in_1);
	cudaFree(gpu_out_1);
	---------------------------------------Layer 1 done-----------------------------------------------
	
	--------------------------------------Layer 2 Convolution----------------------------------------
	Convolution layer preparation.
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
	Defined 3 D blocks with z_threads=no_of_kernels and x_threadsy_threadsz_threads=max_threads_per_block. So, if x_threads=y_threads, then x_threads=sqrt(max_threads_per_blockz_threads). 
	Defined 2 D grids.
	z_threads = no_of_kernels;
	x_threads = sqrt(max_threads_per_block  z_threads);
	y_threads = x_threads;
	dim3 blockdim2(x_threads, y_threads, z_threads);
	dim3 griddim2((output_image_width  x_threads) + 1, (output_image_height  y_threads) + 1, 1);
	
	Arranging the input image and the kernel in proper order (column major) to send off to cudaSgemm for multiplication.
	double co_patches_2;
	co_patches_2 = data_patch_preparation(layer_1, input_image_height, input_image_width, input_image_depth, kernel_height, kernel_width, kernel_depth); co_patches contain the pathes in column order.
	Note It is observed that cublasSgemm supports only float multiplication, thus the double-precision values are explicitly converted to float.
	const float A_2 = (float)co_patches_2;
	const float B_2 = (float)W2_arr;
	float C_2 = (float )malloc(output_image_heightoutput_image_widthoutput_image_depth  sizeof(float));
	m = output_image_height  output_image_width;Number of rows in A.
	n = no_of_kernels; Number of kernels.
	k = kernel_height  kernel_widthkernel_depth; Patch size.
	lda = m, ldb = k, ldc = m;
	 Create a handle for CUBLAS
	cublasHandle_t handle2;
	cublasCreate(&handle2);
	cublasSgemm(handle2, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, A_2, lda, B_2, ldb, beta, C_2, ldc);
	cublasDestroy(handle2);
	The resulting column ordered matrix is the column ordered matrix, so it should have elements in the order
	(First_patch_1st kernel,second_patch_1st_kernel,third_patch_1st_kernel,.....last_patch_1st_kernel,  and so on for all the patches and kernels).
	To get the corresponding value on the output,divide C into (num_kernels) parts of (num_patches) each.
	Getting co_patches into GPU.
	float gpu_C_2;
	cudaMalloc((void )&gpu_C_2, output_image_heightoutput_image_widthkernel_heightkernel_widthkernel_depth  sizeof(float));
	cudaMemcpy(gpu_C_2, C_2, output_image_heightoutput_image_widthkernel_heightkernel_widthkernel_depth  sizeof(float), cudaMemcpyHostToDevice);
	We can do the bias addition and relu activation by writing GPU kernels for the same.
	Copying kernel biases into GPU.
	double kernel_biases_2;
	cudaMalloc((void )&kernel_biases_2, no_of_kernels  sizeof(double));
	cudaMemcpy(kernel_biases_2, B2_arr, no_of_kernels  sizeof(double), cudaMemcpyHostToDevice);
	Creating output array inside GPU.
	double gpu_out_2;
	cudaMalloc((void )&gpu_out_2, output_image_heightoutput_image_widthno_of_kernels  sizeof(double));
	bias_and_relu_kernel griddim2, blockdim2  (gpu_C_2, kernel_biases_2, output_image_height, output_image_width, output_image_depth, gpu_out_2);
	double layer_2 = (double )malloc(output_image_heightoutput_image_widthno_of_kernels  sizeof(double));
	cudaMemcpy(layer_2, gpu_out_2, output_image_heightoutput_image_widthno_of_kernels  sizeof(double), cudaMemcpyDeviceToHost);
	layer_2 is the output from the second layer.
	Free all the unnecessary things from the GPU to make space for the next kernel.
	cudaFree(gpu_C_2);
	cudaFree(kernel_biases_2);
	cudaFree(gpu_out_2);
	free(co_patches_2);
	---------------------------------Layer 2 done--------------------------------------------------------

	----------------------------------Layer 3 Maxpooling------------------------------------------------------
	Maxpooling layer.
	pool_height = 3;
	pool_width = 3;
	input_image_height = output_image_height;
	input_image_width = output_image_width;
	input_image_depth = output_image_depth;
	z_threads = input_image_depth;
	x_threads = sqrt(max_threads_per_block  z_threads);
	y_threads = x_threads;
	output_image_height = (input_image_height - input_image_height % pool_height)  pool_height;
	output_image_width = (input_image_width - input_image_width % pool_width)  pool_width;
	output_image_depth = input_image_depth;
	dim3 blockdim3(x_threads, y_threads, z_threads);
	dim3 griddim3(output_image_width  x_threads, output_image_height  y_threads, 1);
	Copying the previous output into GPU.
	double gpu_in_3;
	cudaMalloc((void )&gpu_in_3, input_image_heightinput_image_widthinput_image_depth  sizeof(double));
	cudaMemcpy(gpu_in_3, layer_2, input_image_heightinput_image_widthinput_image_depth  sizeof(double), cudaMemcpyHostToDevice);
	Creating output array inside GPU.
	double gpu_out_3;
	cudaMalloc((void )&gpu_out_3, output_image_heightoutput_image_widthoutput_image_depth  sizeof(double));
	maxpool_kernel griddim3, blockdim3  (input_image_height, input_image_width, input_image_depth, gpu_in_3, pool_height, pool_width, output_image_height, output_image_width, output_image_depth, gpu_out_3);
	double layer_3 = (double )malloc(output_image_heightoutput_image_widthoutput_image_depth  sizeof(double));
	cudaMemcpy(layer_3, gpu_out_3, output_image_heightoutput_image_widthoutput_image_depth  sizeof(double), cudaMemcpyDeviceToHost);
	layer 1 is the output.
	cudaFree(gpu_in_3);
	cudaFree(gpu_out_3);
	---------------------------------------Layer 3 done--------------------------------------------------
	
	--------------------------------------Layer 4  Flattening----------------------------------------	
	Flattening in the CPU itself.
	The idea is to apply the same kind of C major flattening that keras does to the elements coming in from the second pooling layer.
	The array coming in consists of rows of each sheet arranged side by side followed by the rows of the next sheet and so on. Jumbling up that order to stick with keras type flattening which is the C-major ordering consisting of z-axis changing fastest, followed by column and then row changing.
	int in_h = output_image_height;
	int in_w = output_image_width;
	int in_d = output_image_depth;
	int image_pointer;
	int channel_pointer;
	k = 0;
	double flattened = (double )malloc(in_hin_win_d  sizeof(double));
	for (image_pointer = 0; image_pointer  in_hin_w; image_pointer++) {
		for (channel_pointer = 0; channel_pointer  in_d; channel_pointer++) {
			flattened[k] = layer_3[image_pointer + channel_pointer  in_hin_w];
			k++;
		}
	}
	----------------------------------------Layer 4 done-----------------------------------------------

	----------------------------------------Layer 5 Fully connecteddense layer--------------------------
	int input_layer_nodes = output_image_height  output_image_widthoutput_image_depth;
	int output_layer_nodes = 1024;					This layer has 1024 output nodes.
	double gpu_in_5;
	cudaMalloc((void )&gpu_in_5, input_layer_nodes  sizeof(double));
	cudaMemcpy(gpu_in_5, flattened, input_layer_nodes  sizeof(double), cudaMemcpyHostToDevice);
	double FC_weights_5;
	cudaMalloc((void )&FC_weights_5, input_layer_nodes output_layer_nodes  sizeof(double));
	cudaMemcpy(FC_weights_5, DW5_arr, input_layer_nodes output_layer_nodes  sizeof(double), cudaMemcpyHostToDevice);
	double FC_biases_5;
	cudaMalloc((void )&FC_biases_5, output_layer_nodes  sizeof(double));
	cudaMemcpy(FC_biases_5, DB5_arr, output_layer_nodes  sizeof(double), cudaMemcpyHostToDevice);
	double gpu_out_5;
	cudaMalloc((void )&gpu_out_5, output_layer_nodes  sizeof(double));
	dim3 blocksize5(max_threads_per_block, 1, 1);
	dim3 gridsize5(output_layer_nodes  max_threads_per_block, 1, 1);
	dense_kernel  gridsize5, blocksize5   (input_layer_nodes, output_layer_nodes, gpu_in_5, FC_weights_5, FC_biases_5, gpu_out_5, number_of_classes);
	layer5 is the output.
	double layer_5 = (double )malloc(output_layer_nodes  sizeof(double));
	cudaMemcpy(layer_5, gpu_out_5, output_layer_nodes  sizeof(double), cudaMemcpyDeviceToHost);
	cudaFree(gpu_in_5);
	cudaFree(gpu_out_5);
	cudaFree(FC_biases_5);
	cudaFree(FC_weights_5);
	---------------------------------------Layer 5 done--------------------------------------------

	------------------------------------Layer 6 Fully connecteddense layer--------------------------
	
	input_layer_nodes = output_layer_nodes;
	output_layer_nodes = number_of_classes;
	double gpu_in_6;
	cudaMalloc((void )&gpu_in_6, input_layer_nodes  sizeof(double));
	cudaMemcpy(gpu_in_6, layer_5, input_layer_nodes  sizeof(double), cudaMemcpyHostToDevice);
	double FC_weights_6;
	cudaMalloc((void )&FC_weights_6, input_layer_nodes output_layer_nodes  sizeof(double));
	cudaMemcpy(FC_weights_6, DW7_arr, input_layer_nodes output_layer_nodes  sizeof(double), cudaMemcpyHostToDevice);
	double FC_biases_6;
	cudaMalloc((void )&FC_biases_6, output_layer_nodes  sizeof(double));
	cudaMemcpy(FC_biases_6, DB7_arr, output_layer_nodes  sizeof(double), cudaMemcpyHostToDevice);
	double gpu_out_6;
	cudaMalloc((void )&gpu_out_6, output_layer_nodes  sizeof(double));
	dim3 blocksize6(max_threads_per_block, 1, 1);
	dim3 gridsize6(output_layer_nodes  max_threads_per_block, 1, 1);
	dense_kernel  gridsize6, blocksize6   (input_layer_nodes, output_layer_nodes, gpu_in_6, FC_weights_6, FC_biases_6, gpu_out_6,number_of_classes);
	double layer_7 = (double )malloc(output_layer_nodes  sizeof(double));
	cudaMemcpy(layer_7, gpu_out_6, output_layer_nodes  sizeof(double), cudaMemcpyDeviceToHost);
	layer7 is the output.
	cudaFree(gpu_in_6);
	cudaFree(gpu_out_6);
	cudaFree(FC_biases_6);
	cudaFree(FC_weights_6);
	-----------------------------------Layer 7 done--------------------------------------------------------

	Softmax of the output layer.
	int op_layer_size = number_of_classes;
	int i;
	double sum = 0.0;
	for (i = 0; i  op_layer_size; i++) {
		sum += exp(layer_7[i]);
	}

	printf(n%fn, exp(3));
	double max = layer_7[0]  sum;
	int max_no = 0;
	for (i = 0; i  op_layer_size; i++) {
		printf(%lfn, layer_7[i]);
		if ((layer_7[i]  sum)  max) {
			max_no = i;
		}
	}
}
