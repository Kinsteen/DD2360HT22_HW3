
#include <stdio.h>
#include <sys/time.h>
#include <random>
#include <fstream>
#include <iostream>

#define NUM_BINS 4096

__global__ void histogram_kernel_global(unsigned int *input, unsigned int *bins,
                                        unsigned int num_elements,
                                        unsigned int num_bins)
{

  //@@ Insert code below to compute histogram of input using shared memory and atomics
  int id = threadIdx.x + blockDim.x * blockIdx.x;

  if (id < num_elements)
  {
    // __shared__ int shared[NUM_BINS];

    // atomicAdd(shared + input[id], 1);

    // __syncthreads();

    atomicAdd(&bins[input[id]], 1);
  }
}

__global__ void histogram_kernel(unsigned int *input, unsigned int *bins,
                                 unsigned int num_elements,
                                 unsigned int num_bins)
{

  //@@ Insert code below to compute histogram of input using shared memory and atomics
  int id = threadIdx.x + blockDim.x * blockIdx.x;
  int blockId = threadIdx.x;
  __shared__ int shared[NUM_BINS];

  for (int i = blockId; i < num_bins; i += blockDim.x)
  {
    shared[i] = 0;
  }
  __syncthreads();

  if (id < num_elements)
  {
    atomicAdd(&shared[input[id]], 1);
  }
  __syncthreads();

  for (int i = blockId; i < num_bins; i += blockDim.x)
  {
    atomicAdd(&bins[i], shared[i]);
  }
}

__global__ void convert_kernel(unsigned int *bins, unsigned int num_bins)
{
  //@@ Insert code below to clean up bins that saturate at 127
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if (i < num_bins && bins[i] > 127)
    bins[i] = 127;
}

//@@ Insert code to implement timer start
double startTimer()
{
  struct timeval tp;
  gettimeofday(&tp, NULL);
  return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

//@@ Insert code to implement timer stop
void stopTimer(double start, const char *title)
{
  printf("Timer for %s: %lf\n", title, startTimer() - start);
}

int main(int argc, char **argv)
{

  int inputLength;
  unsigned int *hostInput;
  unsigned int *hostBins;
  unsigned int *resultRef;
  unsigned int *deviceInput;
  unsigned int *deviceBins;

  //@@ Insert code below to read in inputLength from args
  if (argc > 1)
  {
    inputLength = std::stoi(argv[1]);
  }

  printf("The input length is %d\n", inputLength);

  //@@ Insert code below to allocate Host memory for input and output
  hostInput = (unsigned int *)malloc(sizeof(unsigned int) * inputLength);
  hostBins = (unsigned int *)malloc(sizeof(unsigned int) * NUM_BINS);
  resultRef = (unsigned int *)malloc(sizeof(unsigned int) * NUM_BINS);

  //@@ Insert code below to initialize hostInput to random numbers whose values range from 0 to (NUM_BINS - 1)
  unsigned int lower_bound = 0;
  unsigned int upper_bound = NUM_BINS - 1;
  std::uniform_int_distribution<unsigned int> unif(lower_bound, upper_bound);
  std::binomial_distribution<unsigned int> normal(lower_bound, upper_bound);
  // std::default_random_engine re(std::random_device{}());
  std::default_random_engine re;
  for (int i = 0; i < inputLength; i++)
  {
    hostInput[i] = unif(re);
  }

  //@@ Insert code below to create reference result in CPU
  double iStart = startTimer();
  for (int i = 0; i < inputLength; i++)
  {
    if (resultRef[hostInput[i]] < 127)
      resultRef[hostInput[i]] += 1;
  }
  stopTimer(iStart, "CPU execution");

  //@@ Insert code below to allocate GPU memory here
  cudaMalloc(&deviceInput, sizeof(unsigned int) * inputLength);
  cudaMalloc(&deviceBins, sizeof(unsigned int) * NUM_BINS);

  //@@ Insert code to Copy memory to the GPU here
  cudaMemcpy(deviceInput, hostInput, sizeof(unsigned int) * inputLength, cudaMemcpyHostToDevice);

  //@@ Insert code to initialize GPU results
  cudaMemset(deviceBins, 0, sizeof(unsigned int) * NUM_BINS);

  //@@ Initialize the grid and block dimensions here
  int threads_per_block = 768;
  int blocks = ceil(inputLength / (float)threads_per_block);
  printf("%d blocks launched\n", blocks);

  //@@ Launch the GPU Kernel here
  iStart = startTimer();
  histogram_kernel<<<blocks, threads_per_block>>>(deviceInput, deviceBins, inputLength, NUM_BINS);
  cudaDeviceSynchronize();
  stopTimer(iStart, "Kernel exec");

  //@@ Initialize the second grid and block dimensions here
  blocks = ceil(NUM_BINS / (float)threads_per_block);
  printf("%d blocks launched\n", blocks);

  //@@ Launch the second GPU Kernel here
  convert_kernel<<<blocks, threads_per_block>>>(deviceBins, NUM_BINS);

  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(hostBins, deviceBins, sizeof(unsigned int) * NUM_BINS, cudaMemcpyDeviceToHost);

  //@@ Insert code below to compare the output with the reference
  std::ofstream outfile;

  outfile.open("result.txt", std::ios_base::out); // append instead of overwrite

  bool isEqual = true;

  for (int i = 0; i < NUM_BINS; i++)
  {
    // printf("dev bin: %d %d - ref bin: %d %d\n", i, hostBins[i], i, resultRef[i]);
    outfile << hostBins[i] << std::endl;
    if (hostBins[i] != resultRef[i])
    {
      isEqual = false;
      printf("Error at %d: %d != %d\n", i, resultRef[i], hostBins[i]);
    }
  }

  if (isEqual)
    printf("is equal %d\n", isEqual);

  //@@ Free the GPU memory here
  cudaFree(deviceInput);
  cudaFree(deviceBins);
  cudaDeviceReset();

  //@@ Free the CPU memory here
  free(hostInput);
  free(hostBins);
  free(resultRef);

  return 0;
}
