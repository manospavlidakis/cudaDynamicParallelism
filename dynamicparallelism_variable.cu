#include <builtin_types.h>
#include <cuda.h>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <assert.h>
#include <stdlib.h>
#include <sys/time.h>
#include <iostream>
#include <unistd.h>

#define iterations 1000000
#define POLL
#define HOSTALLOC
using namespace std;

__device__ __managed__ volatile int kill = 0;

__global__ void child_launch(float *out, int n) 
{
	//	printf("inside child!!!\n");
	for (int iter=0; iter<iterations; iter++)
	{
		for (int i = 0; i < n; i++)
		{
			out[i] = 12; 
		}
	}
	//in case that there is no need to kill
	//wait for completion and then make kile
	//flag tn 1 in order to exit from parent
	//also!!
	kill = 1;
}

__global__ void parent_launch(float *out, int n) 
{
	if (threadIdx.x == 0) 
	{
		child_launch<<< 16, 1 >>>(out, n);
		//	printf("parent poll %d: \n",poll);
	}
	while (1)
	{
		if (kill == 1)
		{
			//printf("KILL %d: \n",kill);
			//poll = 0;
			asm("exit;"); 
			//assert(0);

			//return ;
			//asm("exit;");
		}
	}
}


void kernel_before_kill()
{
	int N=100;
	cudaError_t err;
	struct timeval kill_st, kill_end;
	float *d_out;
	float h_in[N], h_out[N];

	//float *h_in, *h_out;
	//h_in = (float*)malloc(N*sizeof(float));
	//h_out = (float*)malloc(N*sizeof(float));
	for (int i = 0; i<N; i++)
	{
		h_in[i] = i;
		h_out[i] = 0;
	}

	//Start timer
	gettimeofday(&kill_st,NULL);
	double t1 = kill_st.tv_sec  * 1000000 +  kill_st.tv_usec;
	{
		err = cudaMalloc((void **)&d_out, N * sizeof(float));
		if (err != cudaSuccess) cerr<<"Malloc: "<<cudaGetErrorString(err)<<endl;

		err = cudaMemcpy(d_out, h_in, N * sizeof(float), cudaMemcpyHostToDevice);
		if (err != cudaSuccess) cerr<<"Memcpy (H2D): "<<cudaGetErrorString(err)<<endl;
		child_launch<<< 1, 1 >>>(d_out, N*sizeof(float));
		err=cudaGetLastError();
		if (err != cudaSuccess) cerr<<"Krnl: "<<cudaGetErrorString(err)<<endl;

		err = cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost);
		if (err != cudaSuccess) cerr<<"Mecpy (D2H): "<<cudaGetErrorString(err)<<endl;

		err = cudaFree(d_out);
		if (err != cudaSuccess) cerr<<"Free: "<<cudaGetErrorString(err)<<endl;
	}
	gettimeofday(&kill_end,NULL);
	double t2 = kill_end.tv_sec  * 1000000 +  kill_end.tv_usec;
	long double duration = (t2 - t1)/1000 ;
	cout<<"Duration of kernel (without kill): "<<duration<<" ms"<<endl;

	//free(h_in);
	//free(h_out);
}

void kernel_killed()
{
	cerr<<"Kernel kill!!!"<<endl;
	int N=100;
	cudaError_t err;
	struct timeval kill_st, kill_end;
	float *d_out;
	float *h_in, *h_out;

	h_in = (float*)malloc(N*sizeof(float));
	h_out = (float*)malloc(N*sizeof(float));
	for (int i = 0; i<N; i++)
	{
		h_in[i] = i;
		h_out[i] = 0;
	}

	//Start timer
	gettimeofday(&kill_st,NULL);
	double t1 = kill_st.tv_sec  * 1000000 +  kill_st.tv_usec;
	{

		err = cudaMalloc((void **)&d_out, N * sizeof(float));
		if (err != cudaSuccess) cerr<<"Malloc: "<<cudaGetErrorString(err)<<endl;

		err = cudaMemcpy(d_out, h_in, N * sizeof(float), cudaMemcpyHostToDevice);
		if (err != cudaSuccess) cerr<<"Memcpy (H2D): "<<cudaGetErrorString(err)<<endl;


		//set kill (kernel is going to be killed)
		parent_launch<<< 1, 1 >>>(d_out, N*sizeof(float));
		kill = 1;
		err=cudaGetLastError();
		if (err != cudaSuccess) cerr<<"Krnl: "<<cudaGetErrorString(err)<<endl;

		err = cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost);
		if (err != cudaSuccess) cerr<<"Mecpy (D2H): "<<cudaGetErrorString(err)<<endl;

		err = cudaFree(d_out);
		if (err != cudaSuccess) cerr<<__LINE__<<" Free: "<<cudaGetErrorString(err)<<endl;
		//unset kill (next kernel is not going to be killed)
		kill = 0;

	}
	gettimeofday(&kill_end,NULL);
	double t2 = kill_end.tv_sec  * 1000000 +  kill_end.tv_usec;
	long double duration = (t2 - t1)/1000 ;
	cout<<"Duration of kernel (killed): "<<duration<<" ms"<<endl;

	free(h_in);
	free(h_out);
}

void kernel_after_kill()
{
	int N=100;
	cudaError_t err;
	struct timeval kill_st, kill_end;
	float *dev_out;
	float *host_in, *host_out;

	host_in = (float*)malloc(N*sizeof(float));
	host_out = (float*)malloc(N*sizeof(float));
	for (int i = 0; i<N; i++)
	{
		host_in[i] = i;
		host_out[i] = 0;
	}

	//Start timer
	gettimeofday(&kill_st,NULL);
	double t1 = kill_st.tv_sec  * 1000000 +  kill_st.tv_usec;
	{
		err = cudaMalloc((void **)&dev_out, N * sizeof(float));
		if (err != cudaSuccess) cerr<<"Malloc: "<<cudaGetErrorString(err)<<endl;

		err = cudaMemcpy(dev_out, host_in, N * sizeof(float), cudaMemcpyHostToDevice);
		if (err != cudaSuccess) cerr<<"Memcpy (H2D): "<<cudaGetErrorString(err)<<endl;
		child_launch<<< 1, 1 >>>(dev_out, N*sizeof(float));
		err=cudaGetLastError();
		if (err != cudaSuccess) cerr<<"Krnl: "<<cudaGetErrorString(err)<<endl;

		err = cudaMemcpy(host_out, dev_out, N * sizeof(float), cudaMemcpyDeviceToHost);
		if (err != cudaSuccess) cerr<<"Mecpy (D2H): "<<cudaGetErrorString(err)<<endl;

		err = cudaFree(dev_out);
		if (err != cudaSuccess) cerr<<"Free: "<<cudaGetErrorString(err)<<endl;
	}
	gettimeofday(&kill_end,NULL);
	double t2 = kill_end.tv_sec  * 1000000 +  kill_end.tv_usec;
	long double duration = (t2 - t1)/1000 ;
	cout<<"Duration of kernel (after kill): "<<duration<<" ms"<<endl;	

	free(host_in);
	free(host_out);
}


int main (int argc, char * argv[])
{
	cudaError_t err;
	err = cudaSetDevice(3);	
	if (err != cudaSuccess)
	{
		cerr<<"failed to se device"<<endl;
	}

	cerr<<"->>>>>>>> kernel_before_kill: "<<kill<<endl;
	kernel_before_kill();
	cerr<<"============================================="<<endl;
	sleep(1);	

	cerr<<"->>>>>>>> kernel_killed: "<<kill<<endl;
	kernel_killed();
	cerr<<"============================================="<<endl;
	sleep(1);	

	//	cerr<<"Reset device before next krnl!!!"<<endl;
	//	err = cudaDeviceReset();
	//	err = cudaDeviceSynchronize();
	//	if (err != cudaSuccess) cerr<<"Reset: "<<cudaGetErrorString(err)<<endl;

	cerr<<"->>>>>>>> kernel_after_kill: "<<kill<<endl;
	kernel_after_kill();
	cerr<<"============================================="<<endl;

	cerr<<"Done !!! Press Ctrl+c to exit..."<<endl;
	while(1);

}
