/**
* CSE5441: Lab 4 Problem 1
**/

#include<time.h>
#include<stdio.h>
#include<stdlib.h>
#include<sys/time.h>
#define THRESHOLD 1e-4

// Get the the epoch time in seconds. We don't care about the timezone because
// we will be subtracting the two time values to measure the time spend by the program. 
double rtclock(void)
{
  struct timezone Tzp;
  struct timeval Tp;
  int stat;
  stat = gettimeofday (&Tp, &Tzp);
  if (stat != 0) printf("Error return from gettimeofday: %d", stat);
  return(Tp.tv_sec + Tp.tv_usec*1.0e-6);
}

// Perform computation for the problem on GPU device.
__global__ void ComputeDevice(double *A, int *B) {
	#pragma unroll 100

	int x = blockIdx.x * TILE_DIM + threadIdx.x;
  	int y = blockIdx.y * TILE_DIM + threadIdx.y;
  	int width = gridDim.x * TILE_DIM;

	for (int j = 0; j < TILE_DIM; j+= BLOCK_ROWS)
	    B[x*width + (y+j)] = A[(y+j)*width + x];


}

// Validate the results from the GPU computation with those from the CPU computation. 
// Return 1 if the results match otherwise returns 0.
int ComputeHostValidate(double *A, double *C, int dim)
{
	for (int i = 0; i < dim; i++) 
		for (int j = 0; j < dim; j++) {
			double diff = C[i * dim + j] - A[i * dim + j];

			if(diff < 0) diff *= -1;
			
			// Since the GPU and CPU may differ somewhat in their floating computation,
			// if the difference between the values computes on the GPU and CPU is more
			// than a very small threshold, we will treat that as a correct computation.
			if(diff > THRESHOLD) {
				return 0;
			}
		}

	return 1;
}

// Intialize the matrix for the problem with random values from 1.0 to 2.0.
void initMatrix(double *A, int dim) { 
	for (int i= 0; i< dim; i++)
		for (int j = 0; j < dim; j++)
			A[i* dim + j] = ((double)rand() / RAND_MAX) + 1.0;
}

// Perform the computation for the problem on the CPU.
void ComputeMatrix(double *A, double *B, int dim) {
	for(int k = 0; k < 100; k++) 
		for (int i = 1; i < dim; i++) 
			for (int j = 0; j < dim - 1; j++) 
				B[i+(j-1)*dim] = A[(i-1)*dim+j];
		 		 	
}

int main(void) {
	double *A, *B, *C;

	// Since there wasn't much optimization possible for the problem. To limit the run time 
	// of the problem, I am using a matrix of size 65 * 65
	int dim = 5;

	double *d_A, *d_B;
	int *d_matrixSize;

	size_t memSize = dim * dim * sizeof(double);

	// Allocate memory for the matrices
	A = (double *) malloc(memSize);
	B = (double *) malloc(memSize);
	C = (double *) malloc(memSize);

	// Initialize A
	initMatrix(A, dim);

	// Define thread hierarchy
	int nblocks= 1;
	int num_th_per_blk = 25;
	//int tpb= 1;

	// Allocate device memory
	cudaMalloc( (void**) &d_A, memSize);
	cudaMalloc( (void**) &d_matrixSize, sizeof(int));

	cudaMalloc( (void**) &d_B, memSize);
	//cudaMalloc( (void**) &d_matrixSize, sizeof(int));

	// Initialize device memory
	cudaMemcpy(d_A, A, memSize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_matrixSize, &dim, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, B, memSize, cudaMemcpyHostToDevice);
	
	// Measure the time for the computation
	double start_time, end_time;
	
	// Start time for compute on CPU
	start_time = rtclock();

	// Compute answer on the host 
	ComputeMatrix(A, B, dim);
	
	// End time for compute on CPU
	end_time = rtclock();

	// Print stats for the CPU
	printf("Time taken for compute on CPU (sec) = %.5f, Performance (GFlops/sec) = %.5f\n", end_time - start_time, (100 * (dim-1) * (dim-1))/ (1e9 * (end_time - start_time)));
	
	// launch kernel
	dim3 dimGrid(nblocks);
	dim3 dimBlock(num_th_per_blk);
	
	// Start time for compute on GPU
	start_time = rtclock();

	// Perform the computation on the GPU
	ComputeDevice<<<dimGrid, dimBlock>>>(d_A, d_B, d_matrixSize);

	// Do a cuda synchronize to ensure that the GPU execution finishes for timing it
	cudaThreadSynchronize();	
		
	// End time for compute on GPU
	end_time = rtclock();

	// Print stats for the GPU
	printf("Time taken for compute on GPU (sec) = %.5f, Performance (GFlops/sec) = %.5f\n", end_time - start_time, (100 * (dim-1) * (dim-1))/ (1e9 * (end_time - start_time)));
	
	// Retrieve results from the GPU
	cudaMemcpy(C, d_A, memSize, cudaMemcpyDeviceToHost);
	
	// Verfiy results between the CPU and GPU
	if(!ComputeHostValidate(A, C, dim))
		fprintf(stderr, "Wrong results for compute on GPU\n");

	// Free memory
	cudaFree(d_A);
	cudaFree(d_matrixSize);
	free(A);
	free(C);
}
