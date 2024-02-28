#include <builtin_types.h>
#include <cuda.h>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <assert.h>
#include <stdlib.h>
#include <sys/time.h>
#include <iostream>
#include <unistd.h>
using namespace std;

#define cudaErrorCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
	   if (code != cudaSuccess) 
		      {
			            fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
				          if (abort) exit(code);
					     }
}

//__device__ __managed__ int poll=0;

__global__ void child_launch(float *in, float *out, int n ) 
{
	while(1){
//		printf("inside child!!\n");
		for (int i = 0; i < n; i++)
		{
			out[i] = 12; 
		}	
	}
}

//__global__ void parent_launch(float *in, float *out, int n, int *poll ) 
__global__ void parent_launch(float *in, float *out, int n, int *devPoll) 
{
	if (threadIdx.x == 0) 
	{
		child_launch<<< 16, 1 >>>(in, out, n);
		//cudaDeviceSynchronize();
	}

	while (1)
	{
//		printf("%d\n", *devPoll);
		if (*devPoll == 1)
			asm("trap;"); 
	}

}


int main ()
{
	struct timeval kill_st, kill_end;
	int N = 10;
	float *d_in, *d_out, *h_in, *h_out;
	int *poll;
	int *devPoll;
	cudaError_t err;
	err = cudaSetDevice(3);	
	if (err != cudaSuccess)
	{
		cerr<<"failed to se device"<<endl;
	}

	h_in = (float*)malloc(N*sizeof(float));
	h_out = (float*)malloc(N*sizeof(float));

	for (int i = 0; i<N; i++)
	{
		h_in[i] = i;
		h_out[i] = 0;	
	}
	cudaErrorCheck(cudaMalloc(&d_in, N * sizeof(float)));
	cudaErrorCheck(cudaMalloc(&d_out, N * sizeof(float)));
	cudaErrorCheck(cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice));

	cudaErrorCheck(cudaMallocHost((void**)&poll, sizeof(int)));
	cudaErrorCheck(cudaMalloc((void**)&devPoll, sizeof(int)));
	//cudaErrorCheck(cudaMemset((void**)&devPoll, 0, sizeof(int)));
	
	cout<<"start parent kernel"<<endl;

	gettimeofday(&kill_st,NULL);
	double t1 = kill_st.tv_sec  * 1000000 +  kill_st.tv_usec;

	//parent_launch<<< 1, 1 >>>(d_in, d_out, sizeof(float), poll);
	parent_launch<<< 1, 1 >>>(d_in, d_out, sizeof(float), devPoll);
	//sleep(1);		

	cout<<"Poll == "<<*poll<<endl;
	*poll = 1;
	cout<<"Poll == "<<*poll<<endl;
	
	cudaErrorCheck(cudaMemcpy(devPoll, &poll, sizeof(int), cudaMemcpyHostToDevice));

//	cudaDeviceSynchronize();	
	gettimeofday(&kill_end,NULL);
	double t2 = kill_end.tv_sec  * 1000000 +  kill_end.tv_usec;

	long double duration = (t2 - t1)/1000 ;
	cout<<"Duration of kernel ( "<< poll<< "): "<<duration<<" ms"<<endl;

	//	cout<<"2. Poll : "<<poll<<endl;
	cudaErrorCheck(cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost));
	/*	
		for (int i = 0; i< N ;i++)
		cout <<" OUT: "<<h_out[i]<<" ,";
	 */
	cout<<endl;
	cerr<<"Done !!!"<<endl;
//	while(1);
}
