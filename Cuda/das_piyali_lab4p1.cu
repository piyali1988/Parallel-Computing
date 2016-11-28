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

// Perform matrix transpose multiply on GPU device.
__global__ void MatrixTransposeMultiplyDevice(double *A, double *C, int *matrixSize) {
	int tidy = blockDim.y * blockIdx.y + threadIdx.y;
	int tidx = blockDim.x * blockIdx.x + threadIdx.x;

	#pragma unroll 64
	for(int k = 0; k < *matrixSize; k++)
		C[tidx * (*matrixSize) + tidy] += A[k * (*matrixSize) + tidx] * A[k * (*matrixSize) + tidy];
}

// Perform matrix transpose multiply on CPU.
void MatrixTransposeMultiplyHost(double *A, double *C, int dim) {
	for (int i = 0; i < dim; i++) 
		for (int j = 0; j < dim; j++) {
		 	double sum = 0;

			for(int k = 0; k < dim; k++)
				sum += A[k * dim + i] * A[k * dim +j]; 

			C[i * dim + j] = sum;
		}
}

// Validate the results from the GPU computation with those from the CPU computation. 
// Return 1 if the results match otherwise returns 0.
int MatrixTransposeMultiplyHostValidate(double *A, double *C, int dim)
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

int main(void) {
	double *A, *C;

	// Since the CPU run time for the problem was too high, to limit the run time 
	// of the problem, I am using a matrix of size 256 * 256.
	int dim = 2048;
	double *d_A, *d_C;
	int *d_matrixSize;

    size_t memSize = dim * dim * sizeof(double);

	// Allocate memory for the matrices.
	A = (double *) malloc(memSize);
	C = (double *) calloc(dim * dim, sizeof(double));

	// Load A.
	initMatrix(A, dim);

	// Define thread hierarchy
	int nblocks= dim/16;
	int tpb= 16;

	// Allocate device memory
	cudaMalloc( (void**) &d_A, memSize);
	cudaMalloc( (void**) &d_C, memSize);
	cudaMalloc( (void**) &d_matrixSize, sizeof(int));

	// Initialize device memory
	cudaMemcpy(d_A, A, memSize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_C, C, memSize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_matrixSize, &dim, sizeof(int), cudaMemcpyHostToDevice);
	
	// Measure the time for the computation
	double start_time, end_time;
	
	// Start time for matrix transpose multiply on CPU
	start_time = rtclock();

	// Compute matrix transpose multiply on the host 
	MatrixTransposeMultiplyHost(A, C, dim);

	// End time for matrix transpose multiply on CPU
	end_time = rtclock();

	// Print stats for the CPU
	printf("Time taken for matrix transpose multiply on CPU (sec) = %.5f, Performance (GFlops/sec) = %.5f\n", end_time - start_time, (2L * dim * dim * dim)/ (1e9 * (end_time - start_time)));
	
	// Launch kernel
	dim3 dimGrid(nblocks, nblocks);
	dim3 dimBlock(tpb, tpb);

	// Start time for matrix transpose multiply on GPU
	start_time = rtclock();

	// Perform matrix transpose multiply on GPU device
	MatrixTransposeMultiplyDevice<<< dimGrid, dimBlock>>>(d_A, d_C, d_matrixSize);

	// Do a cuda synchronize to ensure that the GPU execution finishes for timing it
	cudaThreadSynchronize();

	// End time for matrix transpose multiply on GPU
	end_time = rtclock();

	// Print stats for the GPU
	printf("Time taken for matrix transpose multiply on GPU (sec) = %.5f, Performance (GFlops/sec) = %.5f\n", end_time - start_time, (2L * dim * dim * dim)/ (1e9 * (end_time - start_time)));
	
	// Retrieve results
	cudaMemcpy(A, d_C, memSize, cudaMemcpyDeviceToHost);

	// Verfiy results between the CPU and GPU
	if(!MatrixTransposeMultiplyHostValidate(C, A, dim))
		fprintf(stderr, "Wrong results for matrix transpose multiply on GPU\n");

	// Free memory
	cudaFree(d_A);
	cudaFree(d_C);
	cudaFree(d_matrixSize);
	free(A);
	free(C);
}
