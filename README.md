# CNN_Forward_pass_CPU_and_GPU
<b>Forward pass on CNN C and Cuda C code</b> <br>
<b>CPU C code instructions</b><br>
Final_CNN_weights contains the weights (in row order of kernels weights, biases and fully connected layers) obtained after training the CNN (Convolutional Neural network).<br>
In order to run this code, it is necessary to change the file paths of the weights, biases and input images. <br>
Note that the three randomly selected input images are given by the file names New_images(4), New_images(0) and New_images(9) and represent the handwritten digit they are recognizing and the one that the C code should also represent.<br>
The three input images can be run in succession.<br>
Use gcc CNN.c -lm to compile, since we are the exp() function in the code.<br>
