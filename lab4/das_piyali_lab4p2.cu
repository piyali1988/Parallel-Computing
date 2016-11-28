#include<iostream>
#include<time.h>
#include<sys/time.h>
#include<cstdio>
#include<stdlib.h>
#include<string>
#include<math.h>
#include<stack>
#include<fstream>
#include<sstream>
#include<vector>
#include<functional>
#include <ctime>
#include <numeric>

#include "read_bmp.h"

using namespace std;

double rtclock(void)
{
  struct timezone Tzp;
  struct timeval Tp;
  int stat;
  stat = gettimeofday (&Tp, &Tzp);
  if (stat != 0) printf("Error return from gettimeofday: %d", stat);
  return(Tp.tv_sec + Tp.tv_usec*1.0e-6);
}

__global__ void Sobel(uint8_t *bmp_data, uint8_t *new_bmp_data, uint32_t wd, uint32_t ht, int threshold )
{
	int i = blockDim.x*blockIdx.x+threadIdx.x;
	int j = blockDim.y*blockIdx.y+threadIdx.y;

	if(i == 0 || i == ht-1 || j ==0 || j == wd-1)
		return;
	int Gx = bmp_data[ (i-1)*wd + (j+1) ] - bmp_data[ (i-1)*wd + (j-1) ] + 2*bmp_data[ (i)*wd + (j+1) ] - 2*bmp_data[ (i)*wd + (j-1) ] + bmp_data[ (i+1)*wd + (j+1) ] - bmp_data[ (i+1)*wd + (j-1) ];

	int Gy = bmp_data[ (i-1)*wd + (j-1) ] + 2*bmp_data[ (i-1)*wd + (j) ] + bmp_data[ (i-1)*wd + (j+1) ] - bmp_data[ (i+1)*wd + (j-1) ] - 2*bmp_data[ (i+1)*wd + (j) ] - bmp_data[ (i+1)*wd + (j+1) ]; 
	int mag = sqrt((float)Gx * Gx + Gy * Gy);
	if(mag > threshold){
		new_bmp_data[ i*wd + j] = 255;
	}else{
		new_bmp_data[ i*wd + j] = 0;
	}
}

int main(int argc, char **argv)
{

	FILE *input_file, *serial_output_file, *cuda_output_file;
	int black_cell_count = 0;
	input_file = fopen(argv[1],"rb");
	serial_output_file = fopen(argv[2],"wb");
	//cout<<"Input file"<<input_file<<endl;
	bmp_image img1;
	uint8_t *bmp_data, *new_bmp_img;
	bmp_data = (uint8_t *)img1.read_bmp_file(input_file);

	//Allocate new output buffer of same size
	new_bmp_img = (uint8_t *)malloc(img1.num_pixel);
	int wd = img1.image_width;
	int ht = img1.image_height;
	//cout<<"Width: "<<wd<<" height "<<ht<<endl;
	int Gx, Gy, mag =0;

	//Convergence loop
	int threshold = 0;

	// Measure the time for the computation
	double start_time, end_time;

	//Convergence loop for serial program
	// Start time for compute on CPU
	start_time = rtclock();

	while(black_cell_count < (75*wd*ht/100))
	{
		black_cell_count = 0;
		threshold += 1;
		for(int i=1; i < (ht-1); i++){
			for(int j=1; j < (wd-1); j++){
				Gx = bmp_data[ (i-1)*wd + (j+1) ] - bmp_data[ (i-1)*wd + (j-1) ] + 2*bmp_data[ (i)*wd + (j+1) ] - 2*bmp_data[ (i)*wd + (j-1) ] + bmp_data[ (i+1)*wd + (j+1) ] - bmp_data[ (i+1)*wd + (j-1) ];

				Gy = bmp_data[ (i-1)*wd + (j-1) ] + 2*bmp_data[ (i-1)*wd + (j) ] + bmp_data[ (i-1)*wd + (j+1) ] - bmp_data[ (i+1)*wd + (j-1) ] - 2*bmp_data[ (i+1)*wd + (j) ] - bmp_data[ (i+1)*wd + (j+1) ];
			
		
				mag = sqrt(Gx * Gx + Gy * Gy);
				if(mag > threshold){
					new_bmp_img[ i*wd + j] = 255;
				}else{
					new_bmp_img[ i*wd + j] = 0;
					black_cell_count++;
				}
			}
		}
	}
	// End time for compute on CPU
	end_time = rtclock();

	printf("Time taken on CPU (sec) = %.5f, Performance (GFlops/sec) = %.5f\n", end_time - start_time, (100 * (wd-1) * (ht-1))/ (1e9 * (end_time - start_time)));
	cout<<"BLACK CELL "<<black_cell_count<<endl;
	cout<<"CPU Threshold "<<threshold<<endl;
	//Write back the new bmp image into output file
	img1.write_bmp_file(serial_output_file, new_bmp_img);

	free(new_bmp_img);
	free(bmp_data);

	//Parallel cuda implementation

	//Cuda initializations
	bmp_image img2;
	uint8_t *new_bmp_data, *d_bmp_data, *d_new_bmp_data;
	bmp_data = (uint8_t *)img2.read_bmp_file(input_file);
	new_bmp_data = (uint8_t *)malloc(img2.num_pixel);
	cuda_output_file = fopen(argv[3],"wb");

	wd = img2.image_width;
	ht = img2.image_height;

    	//size_t memSize = wd * ht * sizeof(int);

	// Define thread hierarchy   make sure ht % 16 == 0, wd % 16 == 0
	int thread_per_block_x= 16, thread_per_block_y= 16;

	while(ht % thread_per_block_x != 0)
		thread_per_block_x--;
	while(wd % thread_per_block_y != 0)
		thread_per_block_y--;
	
	// Allocate device memory
	cudaMalloc( (void**) &d_bmp_data, img2.num_pixel);
	cudaMalloc( (void**) &d_new_bmp_data, img2.num_pixel);

	// Initialize device memory
	cudaMemcpy(d_bmp_data, bmp_data, img2.num_pixel, cudaMemcpyHostToDevice);
	
	// Launch kernel
	dim3 threadsPerBlock(thread_per_block_x, thread_per_block_y);
	dim3 numBlocks(ht/thread_per_block_x, wd/thread_per_block_y);

	black_cell_count =0, threshold =0;
	// Start time for compute on GPU
	start_time = rtclock();

	while(black_cell_count < (75*wd*ht/100))
	{
		threshold++;
		Sobel<<< numBlocks,threadsPerBlock >>>(d_bmp_data, d_new_bmp_data, wd, ht, threshold);
		cudaMemcpy(new_bmp_data, d_new_bmp_data, img2.num_pixel, cudaMemcpyDeviceToHost);
		black_cell_count = 0;
		for(int i =0; i<wd*ht; i++)
		{
			if(new_bmp_data[i] == 0)
				black_cell_count++;
		}
	}
	// End time for compute on GPU
	end_time = rtclock();

	printf("Time taken on GPU (sec) = %.5f, Performance (GFlops/sec) = %.5f\n", end_time - start_time, (100 * (wd-1) * (ht-1))/ (1e9 * (end_time - start_time)));
	cout<<"GPU Threshold "<<threshold<<endl;

	//Write back the new bmp image into output file
	img2.write_bmp_file(cuda_output_file, new_bmp_data);

	// Free memory
	cudaFree(d_bmp_data);
	cudaFree(d_new_bmp_data);
	free(new_bmp_data);
	free(bmp_data);

return 0;
}
