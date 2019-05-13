//Poisson Blending

/* Background
   ==========

   The goal is to take one image (the source) and
   paste it into another image (the destination) attempting to match the
   two images so that the pasting is non-obvious. This is
   known as a "seamless clone".

   The basic ideas are as follows:

   1) Figure out the interior and border of the source image
   2) Use the values of the border pixels in the destination image 
      as boundary conditions for solving a Poisson equation that tells
      us how to blend the images.
   
      No pixels from the destination except pixels on the border
      are used to compute the match.

   Solving the Poisson Equation
   ============================

   There are multiple ways to solve this equation - we choose an iterative
   method - specifically the Jacobi method. Iterative methods start with
   a guess of the solution and then iterate to try and improve the guess
   until it stops changing.  If the problem was well-suited for the method
   then it will stop and where it stops will be the solution.

   The Jacobi method is the simplest iterative method and converges slowly - 
   that is we need a lot of iterations to get to the answer, but it is the
   easiest method to write.

   Jacobi Iterations
   =================

   Our initial guess is going to be the source image itself.  This is a pretty
   good guess for what the blended image will look like and it means that
   we won't have to do as many iterations compared to if we had started far
   from the final solution.

   ImageGuess_prev (Floating point)
   ImageGuess_next (Floating point)

   DestinationImg
   SourceImg

   Follow these steps to implement one iteration:

   1) For every pixel p in the interior, compute two sums over the four neighboring pixels:
      Sum1: If the neighbor is in the interior then += ImageGuess_prev[neighbor]
             else if the neighbor in on the border then += DestinationImg[neighbor]

      Sum2: += SourceImg[p] - SourceImg[neighbor]   (for all four neighbors)

   2) Calculate the new pixel value:
      float newVal= (Sum1 + Sum2) / 4.f  <------ Notice that the result is FLOATING POINT
      ImageGuess_next[p] = min(255, max(0, newVal)); //clamp to [0, 255]


    In this assignment we will do 800 iterations.
   */



#include "utils.h"
#include <thrust/host_vector.h>
#include <stdio.h>
#include <vector>

__global__ void compute_mask (const uchar4* const inputImage, unsigned char * mask, int numRows,int numCols) {

	if(threadIdx.x > numCols | blockIdx.x > numRows) 
		return;
	int tid = (threadIdx.x ) + (blockIdx.x*numCols);
	uchar4 pixelImage = inputImage[tid];
        mask[tid] = (pixelImage.x + pixelImage.y + pixelImage.z) < 255*3;	
}	

__device__ int isboundaryInDest(int X, int Y,int numRows, int numCols){
	return ((X==0)| (X==numCols) | (Y==0) | (Y==numRows));

}
__global__ void computeBorder(unsigned char * mask, unsigned char * border, unsigned char * interior, int* X_coordinate,int* Y_coordinate,int numRows, int numCols) {
	if(threadIdx.x > numCols | blockIdx.x > numRows) 
		return;
	int X = threadIdx.x;
	int Y = blockIdx.x;
	//Taking into account the boundary conditions
	if(isboundaryInDest(X,Y,numRows,numCols)){
		if(mask[X+(Y*numCols)])
			border[X+(Y*numCols)] = 1;
			X_coordinate[X+(Y*numCols)]=X;
			Y_coordinate[X+(Y*numCols)]=Y;
		return;
	}
	if(mask[(X-1)+((Y)*numCols)] && mask[(X)+((Y-1)*numCols)] &&  mask[(X)+((Y+1)*numCols)] && mask[(X+1)+((Y)*numCols)]){
		interior[X + (Y*numCols)]= 1;
		return;
	}
	border[X+(Y*numCols)] = 1;
	X_coordinate[X+(Y*numCols)]=X;
	Y_coordinate[X+(Y*numCols)]=Y;	
}

__device__ int minimumof(int X, int Y) {
	return ((X<Y)?X:Y);
}

__device__ int maximumof(int X, int Y) {
	return ((X>Y)?X:Y);
}

__global__ void reduce_threads(int* d_in, int* d_out, bool  ismin,int numRows, int numCols) {
	int index = threadIdx.x + (blockIdx.x*blockDim.x);
	int tid   = threadIdx.x;
	if(threadIdx.x > numCols | blockIdx.x > numRows) 
		return;

	for(int s=blockDim.x/2; s>0; s=s>>1){
		if(tid < s) {
		d_in[index] = ismin? minimumof(d_in[index], d_in[index+s]): maximumof(d_in[index],d_in[index+s]);
	}
	__syncthreads();
	}
	if(tid==0){
		d_out[blockIdx.x] = d_in[index];
	}

	return;
}

int reduce_all(int * d_in, bool ismin, int numRows, int numCols) {
	
	int BLOCKS = numRows;
	int THREADS = numCols;
	int total_threads = BLOCKS*THREADS;
	const float  max_thread_per_block = 256;
	
	int* d_out;
	cudaMalloc(&d_out,BLOCKS*sizeof(int));			

	reduce_threads<<<BLOCKS,THREADS>>>(d_in,d_out,ismin,numRows,numCols);

	while (BLOCKS>1){
		total_threads = BLOCKS;
		int BLOCKS = ceil(1.0f*total_threads/max_thread_per_block);
		int THREADS = max_thread_per_block;
		d_in = d_out;
		reduce_threads<<<BLOCKS,THREADS>>>(d_in,d_out,ismin,numRows,numCols);
	}
	return d_out[0];
}




__global__ void jacobiIteration(unsigned char* const channel_source,unsigned char* const channel_dest,unsigned char* buffer_in, unsigned char* buffer_out, unsigned char* interior,unsigned char* border,int minXcoordinate,int minYcoordinate,int numCols,int numRows) {
	int X = minXcoordinate+threadIdx.x;
	int Y = minYcoordinate+blockIdx.x;
	float sum1,sum2;
	sum1 = 0;
	sum2 = channel_source[(X+(Y*numCols))];
	//compute sums considering each of its 4 neighbors independently
	int pos[] = {X+(Y-1)*numCols,X+(Y+1)*numCols,(X+1)+(Y*numCols),(X-1)+(Y*numCols),(X+1)+(Y*numCols) };
	for(int i=0;i<4;i++)	{
		if(interior[pos[i]]){
			sum1 += buffer_in[pos[i]];
		}
		else if(border[pos[i]]) {
			sum1 += channel_dest[pos[i]];
		}
		sum2 -= channel_source[pos[i]];
	}

}
__global__ void separateChannels(const uchar4* const inputImageRGBA,
                      int numRows,
                      int numCols,
                      unsigned char* const redChannel,
                      unsigned char* const greenChannel,
                      unsigned char* const blueChannel)
{
  // TODO
  //
  // NOTE: Be careful not to try to access memory that is outside the bounds of
  // the image. You'll want code that performs the following check before accessing
  // GPU memory:
  //
   if ( threadIdx.x >= numCols ||  blockIdx.x >= numRows )
   {
       return;
   }
	int idx  = threadIdx.x+(numCols*blockIdx.x);
	uchar4 pixelImage = inputImageRGBA[idx];
	redChannel[idx]   = pixelImage.x;
	greenChannel[idx] = pixelImage.y;
        blueChannel[idx]  = pixelImage.z;

}

__global__ void recombineChannels(const uchar4* const d_destImg,const unsigned char* const redChannel,const unsigned char* const greenChannel,const unsigned char* const blueChannel,uchar4* const outputImageRGBA,int numRows,int numCols,unsigned char* interior)
{
	if (threadIdx.x >= numCols || blockIdx.x >= numRows)
		return;
	int thread_1D_pos = threadIdx.x + (blockIdx.x*numCols);
	if (interior[thread_1D_pos] != 1) {
		outputImageRGBA[thread_1D_pos] = d_destImg[thread_1D_pos];
		return;
	}
	unsigned char red = (unsigned char)redChannel[thread_1D_pos];
	unsigned char green = (unsigned char)greenChannel[thread_1D_pos];
	unsigned char blue = (unsigned char)blueChannel[thread_1D_pos];
	//Alpha should be 255 for no transparency
	uchar4 outputPixel = make_uchar4(red, green, blue, 255);
	outputImageRGBA[thread_1D_pos] = outputPixel;
}



void your_blend(const uchar4* const h_sourceImg,  //IN
                const size_t numRowsSource, const size_t numColsSource,
                const uchar4* const h_destImg, //IN
                uchar4* const h_blendedImg) //OUT
{

  /* To Recap here are the steps you need to implement
  
     1) Compute a mask of the pixels from the source image to be copied
        The pixels that shouldn't be copied are completely white, they
        have R=255, G=255, B=255.  Any other pixels SHOULD be copied.

     2) Compute the interior and border regions of the mask.  An interior
        pixel has all 4 neighbors also inside the mask.  A border pixel is
        in the mask itself, but has at least one neighbor that isn't.

     3) Separate out the incoming image into three separate channels

     4) Create two float(!) buffers for each color channel that will
        act as our guesses.  Initialize them to the respective color
        channel of the source image since that will act as our intial guess.

     5) For each color channel perform the Jacobi iteration described 
        above 800 times.

     6) Create the output image by replacing all the interior pixels
        in the destination image with the result of the Jacobi iterations.
        Just cast the floating point values to unsigned chars since we have
        already made sure to clamp them to the correct range.

      Since this is final assignment we provide little boilerplate code to
      help you.  Notice that all the input/output pointers are HOST pointers.

      You will have to allocate all of your own GPU memory and perform your own
      memcopies to get data in and out of the GPU memory.

      Remember to wrap all of your calls with checkCudaErrors() to catch any
      thing that might go wrong.  After each kernel call do:

      cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

      to catch any errors that happened while executing the kernel.
  */

	
	size_t srcSize =  numRowsSource *  numColsSource;
	//input array on the host
	
	//Initial compute mask for the source image
//	unsigned char* h_mask = new unsigned char[srcSize];
	
	//declare GPU memory pointers
	unsigned char * mask;
	uchar4 * d_sourceImage;

	//allocate memory on GPU
	cudaMalloc((void **) &mask, sizeof(unsigned char)*srcSize);
	cudaMalloc((void **) &d_sourceImage, sizeof(unsigned char)*srcSize*4);
	cudaMemcpy(d_sourceImage, h_sourceImg, sizeof(unsigned char)*srcSize*4,cudaMemcpyHostToDevice);
	cudaMemcpy(d_sourceImage, h_sourceImg, sizeof(unsigned char)*srcSize*4,cudaMemcpyHostToDevice);
	compute_mask<<<numRowsSource,numColsSource>>>(d_sourceImage,mask,numRowsSource,numColsSource);
//	cudaMemcpy(h_mask, mask, sizeof(unsigned char)*srcSize,cudaMemcpyDeviceToHost);
	
	cudaDeviceSynchronize();checkCudaErrors(cudaGetLastError());

	//compute border and interior pixels from mask
//	unsigned char* h_border = new unsigned char[srcSize];
//	unsigned char* h_interior = new unsigned char[srcSize];
	unsigned char * d_border;
	unsigned char * d_interior;
	cudaMalloc((void **) &d_border, sizeof(unsigned char)*srcSize);
	cudaMalloc((void **) &d_interior, sizeof(unsigned char)*srcSize);
	cudaMemset(d_border, 0, sizeof(unsigned char)*srcSize);
	cudaMemset(d_interior,0,sizeof(unsigned char)*srcSize);

	int* borderX_coordinate;
	int* borderY_coordinate;
	cudaMalloc((void **) &borderX_coordinate, sizeof(int)*srcSize);
	cudaMalloc((void **) &borderY_coordinate, sizeof(int)*srcSize);
	cudaMemset(borderX_coordinate, -1, sizeof(int)*srcSize);
	cudaMemset(borderY_coordinate,-1,sizeof(int)*srcSize);

        cudaDeviceSynchronize();checkCudaErrors(cudaGetLastError());

	computeBorder<<<numRowsSource,numColsSource>>>(mask,d_border,d_interior,borderX_coordinate,borderY_coordinate,numRowsSource,numColsSource);
//	cudaMemcpy(h_border,d_border,sizeof(unsigned char)*srcSize,cudaMemcpyDeviceToHost);
//	cudaMemcpy(h_interior,d_interior,sizeof(unsigned char)*srcSize,cudaMemcpyDeviceToHost);

	cudaDeviceSynchronize();checkCudaErrors(cudaGetLastError());

	//find min and max coordinates for boundary boxes(This allows us to limit our jacobi iterations only to interior pixels)
	int minx = reduce_all(borderX_coordinate, true, numRowsSource, numColsSource);
	int miny = reduce_all(borderY_coordinate, true, numRowsSource, numColsSource);
	int maxx = reduce_all(borderX_coordinate, false, numRowsSource, numColsSource);
	int maxy = reduce_all(borderY_coordinate, false, numRowsSource, numColsSource);

	int x_range = maxx - minx;
	int y_range = maxy - miny;
	//Separate all 3 R-G-B channel for source and destination images
	unsigned char* h_red = new unsigned char[srcSize];
	unsigned char* h_green = new unsigned char[srcSize];
	unsigned char* h_blue = new unsigned char[srcSize];
	unsigned char* redChannel;
	unsigned char* greenChannel;
	unsigned char* blueChannel;
	cudaMalloc((void **) &redChannel, sizeof(unsigned char)*srcSize);
	cudaMalloc((void **) &greenChannel, sizeof(unsigned char)*srcSize);
	cudaMalloc((void **) &blueChannel, sizeof(unsigned char)*srcSize);

	
	cudaMemcpy(d_sourceImage, h_sourceImg, sizeof(unsigned char)*srcSize*4,cudaMemcpyHostToDevice);

	separateChannels<<<numRowsSource,numColsSource>>>(d_sourceImage,numRowsSource,numColsSource,redChannel,greenChannel,blueChannel);
	cudaMemcpy(h_red,redChannel,sizeof(unsigned char)*srcSize,cudaMemcpyDeviceToHost);
	cudaMemcpy(h_green,greenChannel,sizeof(unsigned char)*srcSize,cudaMemcpyDeviceToHost);
	cudaMemcpy(h_blue,blueChannel,sizeof(unsigned char)*srcSize,cudaMemcpyDeviceToHost);
	//for destination images
	unsigned char* red_dest;
	unsigned char* green_dest;
	unsigned char* blue_dest;
	cudaMalloc((void **) &red_dest, sizeof(unsigned char)*srcSize);
	cudaMalloc((void **) &green_dest, sizeof(unsigned char)*srcSize);
	cudaMalloc((void **) &blue_dest, sizeof(unsigned char)*srcSize);

	uchar4 * d_destImage;
	cudaMalloc((void **) &d_destImage, sizeof(unsigned char)*srcSize*4);
	cudaMemcpy(d_destImage, h_destImg, sizeof(unsigned char)*srcSize*4,cudaMemcpyHostToDevice);
	separateChannels<<<numRowsSource,numColsSource>>>(d_destImage,numRowsSource,numColsSource,red_dest,green_dest,blue_dest);
//	cudaMemcpy(h_red,redChannel,sizeof(unsigned char)*srcSize,cudaMemcpyDeviceToHost);
//	cudaMemcpy(h_green,greenChannel,sizeof(unsigned char)*srcSize,cudaMemcpyDeviceToHost);
//	cudaMemcpy(h_blue,blueChannel,sizeof(unsigned char)*srcSize,cudaMemcpyDeviceToHost);

	cudaDeviceSynchronize();checkCudaErrors(cudaGetLastError());


	unsigned char* red_buffer_1;
	unsigned char* green_buffer_1;
	unsigned char* blue_buffer_1;
	cudaMalloc((void **) &red_buffer_1, sizeof(unsigned char)*srcSize);
	cudaMalloc((void **) &green_buffer_1, sizeof(unsigned char)*srcSize);
	cudaMalloc((void **) &blue_buffer_1, sizeof(unsigned char)*srcSize);
	cudaMemcpy(red_buffer_1,h_red , sizeof(unsigned char)*srcSize,cudaMemcpyHostToDevice);
	cudaMemcpy(green_buffer_1,h_green , sizeof(unsigned char)*srcSize,cudaMemcpyHostToDevice);
	cudaMemcpy(blue_buffer_1,h_blue , sizeof(unsigned char)*srcSize,cudaMemcpyHostToDevice);
	unsigned char* red_buffer_2;
	unsigned char* green_buffer_2;
	unsigned char* blue_buffer_2;
	cudaMalloc((void **) &red_buffer_2, sizeof(unsigned char)*srcSize);
	cudaMalloc((void **) &green_buffer_2, sizeof(unsigned char)*srcSize);
	cudaMalloc((void **) &blue_buffer_2, sizeof(unsigned char)*srcSize);
	cudaMemcpy(red_buffer_2,h_red , sizeof(unsigned char)*srcSize,cudaMemcpyHostToDevice);
	cudaMemcpy(green_buffer_2,h_green , sizeof(unsigned char)*srcSize,cudaMemcpyHostToDevice);
	cudaMemcpy(blue_buffer_2,h_blue , sizeof(unsigned char)*srcSize,cudaMemcpyHostToDevice);

	const int numIterations = 800;
	for(int i=0; i<numIterations;i++) {
		if(i%2) {
		jacobiIteration<<<y_range,x_range>>>(redChannel,red_dest,red_buffer_1,red_buffer_2,d_interior,d_border,minx,miny,numColsSource,numRowsSource);
		jacobiIteration<<<y_range,x_range>>>(greenChannel,green_dest,green_buffer_1,green_buffer_2,d_interior,d_border,minx,miny,numColsSource,numRowsSource);
		jacobiIteration<<<y_range,x_range>>>(blueChannel,blue_dest,blue_buffer_1,blue_buffer_2,d_interior,d_border,minx,miny,numColsSource,numRowsSource);
		}
		else {
		jacobiIteration<<<y_range,x_range>>>(redChannel,red_dest,red_buffer_2,red_buffer_1,d_interior,d_border,minx,miny,numColsSource,numRowsSource);
		jacobiIteration<<<y_range,x_range>>>(greenChannel,green_dest,green_buffer_2,green_buffer_1,d_interior,d_border,minx,miny,numColsSource,numRowsSource);
		jacobiIteration<<<y_range,x_range>>>(blueChannel,blue_dest,blue_buffer_2,blue_buffer_1,d_interior,d_border,minx,miny,numColsSource,numRowsSource);
		}
	}


	uchar4* d_blendedImg;
	checkCudaErrors(cudaMalloc(&d_blendedImg, numRowsSource*numColsSource * sizeof(uchar4)));
	recombineChannels <<<numRowsSource,numColsSource >>> (d_destImage,red_buffer_1,green_buffer_1,blue_buffer_1, d_blendedImg, numRowsSource, numColsSource,d_interior);
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaMemcpy(h_blendedImg, d_blendedImg, numRowsSource*numColsSource * sizeof(uchar4), cudaMemcpyDeviceToHost));

	cudaFree(mask);
	cudaFree(d_sourceImage);
	cudaFree(d_destImage);
	cudaFree(d_border);
	cudaFree(d_interior);
	cudaFree(redChannel);
	cudaFree(greenChannel);
  cudaFree(blueChannel);
  cudaFree(borderX_coordinate);
  cudaFree(borderY_coordinate);
  cudaFree(red_dest);
  cudaFree(green_dest);
  cudaFree(blue_dest);
  cudaFree(red_buffer_1);
  cudaFree(green_buffer_1);
  cudaFree(blue_buffer_1);
  cudaFree(red_buffer_2);
  cudaFree(green_buffer_2);
  cudaFree(blue_buffer_2);  
  cudaFree(d_blendedImg);
}
