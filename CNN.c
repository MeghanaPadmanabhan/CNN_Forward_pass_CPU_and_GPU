#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<float.h>
//-------------------------------------------------Convolution layer---------------------------------------------------------------------//
	//Input_height, input_width, input_depth,input_values, kernel_height, kernel_width, kernel_depth, number_of_kernels, kernel_weights, kernel_biases. 
	//Note: Strictly speking the stride and padding needs to be included in the following calculations but here the stride is always 1 and the padding is always valid, so not included.
	double* convolution_layer(int h,int w,int d,double* input_arr,int k_h,int k_w,int k_d,int num_kernels,double* kernel_weights,double* kernel_biases){		
	double* res_arr=(double *)malloc((h-k_h+1)*(w-k_w+1)*num_kernels*sizeof(double));
	int i;
	int k;
	int op_h=h-k_h+1;
	int op_w=w-k_w+1;
	int column_pointer;
	int element_pointer;
	int row_pointer;
	int kernel_count;
	int z_pointer;
	
	//Creating a 3 D output layer consisting of one kernel output per sheet of the output.
	//Doing the convolution for one kernel at a time.
	for (kernel_count=0;kernel_count<num_kernels;kernel_count++){
		for(row_pointer=0;row_pointer<(h-k_h)+1;row_pointer++){
		for (column_pointer=0;column_pointer<(w-k_w)+1;column_pointer++)
		{	
			res_arr[kernel_count*op_h*op_w+row_pointer*op_w+column_pointer]=0.0;
			k=0;
			i=0;
			for (z_pointer=0;z_pointer<d;z_pointer++){
			for(i=0;i<k_h;i++){
				for(element_pointer=0;element_pointer<k_w;element_pointer++){
					res_arr[kernel_count*op_h*op_w+row_pointer*op_w+column_pointer]+=input_arr[((row_pointer*w)+(z_pointer*h*w)+(column_pointer+i*w)+element_pointer)]*kernel_weights[kernel_count*k_h*k_w*k_d+k]; 
					k++;					
				}
			}
		}
		res_arr[kernel_count*op_h*op_w+row_pointer*op_w+column_pointer]+=kernel_biases[kernel_count];		
	}	
}
}


//Performing relu activation on all the res_arr units that is max(0,res_arr).
for (i=0;i<op_w*op_h*num_kernels;i++){
	
	if (res_arr[i]<0.0)
		res_arr[i]=0.0;
}

return res_arr;
}
//Now res_arr is ready.
//-----------------------------------------------Convolution layer done--------------------------------------------------------//	

//-----------------------------------------------Max pooling layer-------------------------------------------------------------//

//Take as input height, input width, input depth, pool width, pool height.
double* maxpooling_layer(int inp_h,int inp_w,int inp_d,double* inp_arr,int pool_width,int pool_height){
	
int column_pointer;
int row_pointer;
int z_pointer;
double max;
int k=0;
int i;
//Condition considered by Keras: If the width or height of the image is odd, then discard the last column or row when pooling.
int row_extent; //Used to define usable portion along the row.
int column_extent; //Used to define usable portion along column.
int element_pointer;
row_extent=inp_w-(inp_w%pool_width);
column_extent=inp_h-(inp_h%pool_height);
double *max_pool_op_arr=(double *)malloc((row_extent/pool_width)*(column_extent/pool_height)*inp_d*sizeof(double));
for (z_pointer=0;z_pointer<inp_d;z_pointer++){
for (row_pointer=0;row_pointer<column_extent-pool_height+1;row_pointer+=pool_height){
for (column_pointer=0;column_pointer<row_extent-pool_width+1;column_pointer+=pool_width){
	max_pool_op_arr[k]=0.0l;
	max=inp_arr[row_pointer*(row_extent+(inp_w%pool_width))+column_pointer+z_pointer*inp_h*inp_w];
	for (i=0;i<pool_height;i++){
		for(element_pointer=0;element_pointer<pool_width;element_pointer++){	
		if(inp_arr[z_pointer*inp_h*inp_w+row_pointer*(row_extent+(inp_w%pool_width))+column_pointer+i*(row_extent+(inp_w%pool_width))+element_pointer]>max)
		max=inp_arr[z_pointer*inp_h*inp_w+row_pointer*(row_extent+(inp_w%pool_width))+column_pointer+i*(row_extent+(inp_w%pool_width))+element_pointer];
	}
	}
	max_pool_op_arr[k]=max;
	k++;
}
}
}

return max_pool_op_arr;
}
//------------------------------------------Max pooling layer done------------------------------------------------------------//

//------------------------------------------------Dense layer---------------------------------------------------------------------------//
//Dense layer implements the fully connected layer where the input is the flattened and the output is den. The weights are an array with the first n elements representing the inputs to the first output node. Bias is added to each output unit.
double* dense_layer(int ip_size,int den_size,double* weights,double* biases,double* flattened){
int unit_pointer;
int ip_pointer;
int k;
double* den=(double *)malloc(den_size*sizeof(double));
for (k=0;k<den_size;k++){
	den[k]=0.0;
	for (ip_pointer=0;ip_pointer<ip_size;ip_pointer++){
		den[k]+=flattened[ip_pointer]*weights[ip_pointer+k*ip_size];
	}
	den[k]+=biases[k];
}
if(den_size!=10){
//Relu activation of each output unit, ie max(0,value).
for (k=0;k<den_size;k++){
	if (den[k]<0.0){
	den[k]=0.0;
	}
}
}
return den;
}

//--------------------------------------------------Dense layer done-------------------------------------------------------------------------//


int main(){
	
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
	int i;
	//Note: The weights are pulled out after training the mnist digit recognition dataset on keras with handwritten digits 0-9. The images are greysvale and hence to start with they have only one channel. 
	//Weights are pulled out and inputted into the respective arrays.
	//Pulling out image values
	double* img_arr=(double *)malloc(28*28*sizeof(double));
	pFileImg = fopen("/home/meghanap/Image_RO.txt", "r");
	if (pFileImg == NULL) { fputs("File error", stderr); exit(1);}
	for(i=0;i<784;i++){
	fscanf(pFileImg,"%lf", &img_arr[i]);
	}
	
	//Pulling out kernel weights for first conv layer.
	double* W0_arr=(double *)malloc(288*sizeof(double));
	pFileW0 = fopen("/home/meghanap/W0_RO.txt", "r");
	if (pFileW0 == NULL) { fputs("File error", stderr); exit(1);}
	for(i=0;i<288;i++){
	fscanf(pFileW0,"%lf", &W0_arr[i]);
	}
	
	//Pulling out kernel biases for first conv layer.
	double* B0_arr=(double *)malloc(32*sizeof(double));
	pFileB0 = fopen("/home/meghanap/B0.txt", "r");
	if (pFileB0 == NULL) { fputs("File error", stderr); exit(1);}
	for(i=0;i<32;i++){
	fscanf(pFileB0,"%lf", &B0_arr[i]);
	}	

	
	//Pulling out kernel weights for second conv layer.
	double* W2_arr=(double *)malloc(18432*sizeof(double));
	pFileW2 = fopen("/home/meghanap/W2_RO.txt", "r");
	if (pFileW2 == NULL) { fputs("File error", stderr); exit(1);}
	for(i=0;i<18432;i++){
	fscanf(pFileW2,"%lf", &W2_arr[i]);
	}
	
	//Pulling out kernel biases for second conv layer.
	double* B2_arr=(double *)malloc(64*sizeof(double));
	pFileB2 = fopen("/home/meghanap/B2.txt", "r");
	if (pFileB2 == NULL) { fputs("File error", stderr); exit(1);}
	for(i=0;i<64;i++){
	fscanf(pFileB2,"%lf", &B2_arr[i]);
	}
	
	
	//Pulling out weights for first fully connected layer.
	double* DW5_arr=(double *)malloc(1638400*sizeof(double));
	pFileDW5 = fopen("/home/meghanap/DW5_RO.txt", "r");
	if (pFileDW5 == NULL) { fputs("File error", stderr); exit(1);}
	for(i=0;i<1638400;i++){
	fscanf(pFileDW5,"%lf", &DW5_arr[i]);
	}
	
	//Pulling out biases for first fully connected layer.
	double* DB5_arr=(double *)malloc(1024*sizeof(double));
	pFileDB5 = fopen("/home/meghanap/DB5.txt", "r");
	if (pFileDB5 == NULL) { fputs("File error", stderr); exit(1);}
	for(i=0;i<1024;i++){
	fscanf(pFileDB5,"%lf", &DB5_arr[i]);
	}	
		
	//Pulling out weights for second fully connected layer.
	double* DW7_arr=(double *)malloc(10240*sizeof(double));
	pFileDW7 = fopen("/home/meghanap/DW7_RO.txt", "r");
	if (pFileDW7 == NULL) { fputs("File error", stderr); exit(1);}
	for(i=0;i<10240;i++){
	fscanf(pFileDW7,"%lf", &DW7_arr[i]);
	}
	
	
	//Pulling out biases for second fully connected layer.
	double* DB7_arr=(double *)malloc(10*sizeof(double));
	pFileDB7 = fopen("/home/meghanap/DB7.txt", "r");
	if (pFileDB7 == NULL) { fputs("File error", stderr); exit(1);}
	for(i=0;i<10;i++){
	fscanf(pFileDB7,"%lf", &DB7_arr[i]);
	}
	
	
//The layers are Convolution(with relu activation)-->Max_Pooling-->Convolution(with relu activation)-->Max_pooling-->Flatten-->Dense layer(with relu activation)-->Dense layer(with relu activation)-->Softmax.

//Call and pass this function first convolution layer.
double* layer_0=(double *)malloc(26*26*32*sizeof(double));
layer_0=convolution_layer(28,28,1,img_arr,3,3,1,32,W0_arr,B0_arr);

//Call and pass the first max_pool function.
double* layer_1=(double *)malloc(13*13*32*sizeof(double));
layer_1=maxpooling_layer(26,26,32,layer_0,2,2);

//Call and pass the second convolution layer function.
double* layer_2=(double *)malloc(11*11*64*sizeof(double));
layer_2=convolution_layer(13,13,32,layer_1,3,3,32,64,W2_arr,B2_arr);

//Call and pass the second max pooling function.
double* layer_3=(double *)malloc(5*5*64*sizeof(double));
layer_3=maxpooling_layer(5,5,64,layer_2,2,2); 

//------------------------------------------Flattening layer (Not a function bacause happening only once in any program)------------------------------------------------------------//
//The idea is to apply the same kind of C major flattening that keras does to the elements coming in from the second pooling layer.
//The array coming in consists of rows of each sheet arranged side by side followed by the rows of the next sheet and so on. Jumbling up that order to stick with keras type flattening which is the C-major ordering consisting of z-axis changing fastest, follwed by column and then row changing.
int in_h=5;
int in_w=5;
int in_d=64;
int image_pointer;
int channel_pointer;
int k=0;
double* flattened=(double *)malloc(in_h*in_w*in_d*sizeof(double));
for(image_pointer=0;image_pointer<in_h*in_w;image_pointer++){
	for (channel_pointer=0;channel_pointer<in_d;channel_pointer++){
		flattened[k]=layer_3[image_pointer+channel_pointer*in_h*in_w];
		k++;
	}
}

//------------------------------------------------Flattening of array done--------------------------------------------------------------//

	

//Calling the first dense_layer function.
double* layer_5=(double *)malloc(1024*sizeof(double));
layer_5=dense_layer(1600,1024,DW5_arr,DB5_arr,flattened);


//Calling the second dense layer function.
double* layer_7=(double *)malloc(10*sizeof(double));
layer_7=dense_layer(1024,10,DW7_arr,DB7_arr,layer_5);


//-----------------------------------------Softmax activation and printing the result(not written as a function because done only once)------------------------------------------------------------------------//
int op_layer_size=10;
double sum=0.0;
for (i=0;i<op_layer_size;i++){
	sum+=exp(layer_7[i]);
}

double max=exp(layer_7[0])/sum;
int max_no=0;
for(i=0;i<op_layer_size;i++){
	if((exp(layer_7[i])/sum)>max){
	max_no=i;
	}
}

printf("\n The written predicted digit is %d\n",max_no);

//------------------------------------------------Softmax done result printed----------------------------------------------------------------//
}














	
	
