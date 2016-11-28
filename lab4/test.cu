#include <stdio.h>
#include <cstdlib>
#include <time.h>


__global__ void reverse(int *A, int n) 
{
	int tid;
	tid = blockIdx.x*blockDim.x+threadIdx.x;
	printf("from GPU: %d\n", A[tid]);

	//swap
	if(tid<n){
		int temp = A[tid];
		printf("from GPU: %d\n", A[tid]);
		A[tid] = A[n-tid-1];
		A[n-tid-1] = temp;
	}
}

int main()
{

	int num_blocks = 1;
	int num_th_per_blk = 64;

	int num_elem = 128;
	int array_size = num_elem*sizeof(int);	
	int *a = (int*) malloc(array_size);  //pointer to host version of the array	
	for(int i = 0;i<num_elem;i++)
		a[i] = i;


	int *ad;			     //pointer to device version of the array
	cudaMalloc( (void**)&ad, array_size);
	cudaMemcpy( ad, a, array_size, cudaMemcpyHostToDevice ); 
	
	// lunch
	dim3 dimBlock(num_blocks);
	dim3 dimGrid(num_th_per_blk);
	//reverse<<<dimGrid, dimBlock>>>(ad, num_elem);
	reverse<<<1,128>>>(ad, num_elem);

	// copy back and verify
	int *b = (int*) malloc(array_size);
	cudaMemcpy( b, ad, array_size, cudaMemcpyDeviceToHost ); 
	printf("a[i] %d\n", a[2]);	
	printf("b[i] %d\n", b[2]);	
	//cudaFree( ad );
	for(int i = 0;i<(num_elem/2);i++)
		if(a[i]!=b[num_elem-i-1])
		{
			printf("Failure!\n");
			return -1;
		}
	printf("Success!\n");
	return 0;
}

