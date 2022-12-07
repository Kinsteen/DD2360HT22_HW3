
#include <stdio.h>
#include <sys/time.h>
#include <random>

#define DataType double

__global__ void vecAdd(DataType *in1, DataType *in2, DataType *out, int len)
{
  //@@ Insert code to implement vector addition here
  int id = threadIdx.x + blockDim.x * blockIdx.x;

  if (id < len)
  {
    out[id] = in1[id] + in2[id];
  }
}

//@@ Insert code to implement timer start
double startTimer()
{
  struct timeval tp;
  gettimeofday(&tp, NULL);
  return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

//@@ Insert code to implement timer stop
void stopTimer(double start, const char* title)
{
  printf("Timer for %s: %lf\n", title, startTimer() - start);
}

int main(int argc, char **argv)
{
  int inputLength;
  DataType *hostInput1;
  DataType *hostInput2;
  DataType *hostOutput;
  DataType *resultRef;
  DataType *deviceInput1;
  DataType *deviceInput2;
  DataType *deviceOutput;

  //@@ Insert code below to read in inputLength from args
  if (argc > 1)
  {
    inputLength = std::stoi(argv[1]);
  }

  printf("The input length is %d\n", inputLength);

  //@@ Insert code below to allocate Host memory for input and output
  hostInput1 = (DataType *)malloc(sizeof(DataType) * inputLength);
  hostInput2 = (DataType *)malloc(sizeof(DataType) * inputLength);
  hostOutput = (DataType *)malloc(sizeof(DataType) * inputLength);
  resultRef = (DataType *)malloc(sizeof(DataType) * inputLength);

  //@@ Insert code below to initialize hostInput1 and hostInput2 to random numbers, and create reference result in CPU
  double lower_bound = 0;
  double upper_bound = 10;
  std::uniform_real_distribution<double> unif(lower_bound, upper_bound);
  std::default_random_engine re;
  double s = startTimer();
  for (int i = 0; i < inputLength; i++)
  {
    DataType d1 = unif(re);
    DataType d2 = unif(re);

    hostInput1[i] = d1;
    hostInput2[i] = d2;

    resultRef[i] = d1 + d2;
  }
  stopTimer(s, "CPU result");

  //@@ Insert code below to allocate GPU memory here
  cudaMalloc(&deviceInput1, sizeof(DataType) * inputLength);
  cudaMalloc(&deviceInput2, sizeof(DataType) * inputLength);
  cudaMalloc(&deviceOutput, sizeof(DataType) * inputLength);

  //@@ Insert code to below to Copy memory to the GPU here
  s = startTimer();
  cudaMemcpy(deviceInput1, hostInput1, sizeof(DataType) * inputLength, cudaMemcpyHostToDevice);
  cudaMemcpy(deviceInput2, hostInput2, sizeof(DataType) * inputLength, cudaMemcpyHostToDevice);
  stopTimer(s, "CUDA memcpy host -> device");

  //@@ Initialize the 1D grid and block dimensions here
  int threads_per_block = 512;
  int blocks = std::ceil(inputLength / (float)threads_per_block);
  printf("TPB: %d\n", threads_per_block);
  printf("Blocks: %d\n", blocks);

  //@@ Launch the GPU Kernel here
  s = startTimer();
  vecAdd<<<blocks, threads_per_block>>>(deviceInput1, deviceInput2, deviceOutput, inputLength);
  cudaDeviceSynchronize();
  stopTimer(s, "Kernel execution");

  //@@ Copy the GPU memory back to the CPU here
  s = startTimer();
  cudaMemcpy(hostOutput, deviceOutput, sizeof(DataType) * inputLength, cudaMemcpyDeviceToHost);
  stopTimer(s, "CUDA memcpy device -> host");

  //@@ Insert code below to compare the output with the reference
  bool areVecEqual = true;

  for (int i = 0; i < inputLength; i++)
  {
    if (hostOutput[i] != resultRef[i])
    {
      areVecEqual = false;
    }
  }

  if (areVecEqual)
  {
    printf("Result are equal!\n");
  }
  else
  {
    printf("Results are different\n");
  }

  //@@ Free the GPU memory here
  cudaFree(deviceInput1);
  cudaFree(deviceInput2);
  cudaFree(deviceOutput);

  //@@ Free the CPU memory here
  free(hostInput1);
  free(hostInput2);
  free(hostOutput);

  cudaDeviceReset();

  return 0;
}
