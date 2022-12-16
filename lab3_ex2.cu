
#include <cstdio>
#include <sys/time.h>
#include <random>
#include <locale.h>

#define DataType double

// Compute C = A * B
__global__ void gemm(DataType *A, DataType *B, DataType *C, int numARows,
                     int numAColumns, int numBRows, int numBColumns)
{
  //@@ Insert code to implement matrix multiplication here
  int x = threadIdx.x + blockDim.x * blockIdx.x;
  int y = threadIdx.y + blockDim.y * blockIdx.y;

  if (x < numBColumns && y < numARows)
  {
    for (int i = 0; i < numAColumns; i++)
    {
      C[x + y * numBColumns] += A[i + y * numAColumns] * B[x + i * numBColumns];
    }
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

void printMatrix(DataType *m, int row, int col)
{
  for (int i = 0; i < row; i++)
  {
    for (int j = 0; j < col; j++)
    {
      printf("%lf ", m[j + i * col]);
    }
    printf("\n");
  }
  printf("\n");
}

int main(int argc, char **argv)
{
  setlocale(LC_ALL, "");
  DataType *hostA;     // The A matrix
  DataType *hostB;     // The B matrix
  DataType *hostC;     // The output C matrix
  DataType *resultRef; // The reference result
  DataType *deviceA;
  DataType *deviceB;
  DataType *deviceC;
  int numARows;    // number of rows in the matrix A
  int numAColumns; // number of columns in the matrix A
  int numBRows;    // number of rows in the matrix B
  int numBColumns; // number of columns in the matrix B
  int numCRows;
  int numCColumns;

  //@@ Insert code below to read in numARows, numAColumns, numBColumns from args
  if (argc == 4)
  {
    numARows = std::stoi(argv[1]);
    numAColumns = std::stoi(argv[2]);
    numBRows = numAColumns;
    numBColumns = std::stoi(argv[3]);

    numCRows = numARows;
    numCColumns = numBColumns;
  }
  else
  {
    fprintf(stderr, "Wrong number of args\n");
    exit(1);
  }

  printf("Input matrix dim (%d x %d) (%d x %d) (%d x %d)\n", numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);

  //@@ Insert code below to allocate Host memory for input and output
  hostA = (DataType *)malloc(numARows * numAColumns * sizeof(DataType));
  hostB = (DataType *)malloc(numBRows * numBColumns * sizeof(DataType));
  hostC = (DataType *)malloc(numCRows * numCColumns * sizeof(DataType));
  resultRef = (DataType *)malloc(numCRows * numCColumns * sizeof(DataType));

  //@@ Insert code below to initialize hostA and hostB to random numbers, and create reference result in CPU
  double lower_bound = 0;
  double upper_bound = 10;
  std::uniform_real_distribution<double> unif(lower_bound, upper_bound);
  std::default_random_engine re(std::random_device{}());
  for (int i = 0; i < numARows * numAColumns; i++)
  {
    hostA[i] = unif(re);
  }

  for (int i = 0; i < numBRows * numBColumns; i++)
  {
    hostB[i] = unif(re);
  }

  for (int i = 0; i < numCRows * numCColumns; i++)
  {
    int row = i % numCColumns;
    int col = i / numCColumns;

    resultRef[i] = 0;

    for (int j = 0; j < numAColumns; j++)
    {
      resultRef[i] += hostA[j + col * numAColumns] * hostB[row + j * numBColumns];
    }
  }

  //printMatrix(hostA, numARows, numAColumns);
  //printMatrix(hostB, numBRows, numBColumns);
  //printMatrix(resultRef, numCRows, numCColumns);

  //@@ Insert code below to allocate GPU memory here
  cudaMalloc(&deviceA, numARows * numAColumns * sizeof(DataType));
  cudaMalloc(&deviceB, numBRows * numBColumns * sizeof(DataType));
  cudaMalloc(&deviceC, numCRows * numCColumns * sizeof(DataType));

  //@@ Insert code to below to Copy memory to the GPU here
  double iStart = startTimer();
  cudaMemcpy(deviceA, hostA, numARows * numAColumns * sizeof(DataType), cudaMemcpyHostToDevice);
  cudaMemcpy(deviceB, hostB, numBRows * numBColumns * sizeof(DataType), cudaMemcpyHostToDevice);
  stopTimer(iStart, "Copy host->device");

  //@@ Initialize the grid and block dimensions here
  int threads_per_block_x = 32;
  int threads_per_block_y = 24;
  int num_blocks_x = ceil(numCColumns / (float) threads_per_block_x);
  int num_blocks_y = ceil(numCRows / (float) threads_per_block_y);
  printf("Blocks: %d x %d\n", num_blocks_x, num_blocks_y);

  //@@ Launch the GPU Kernel here
  iStart = startTimer();
  gemm<<<dim3(num_blocks_x, num_blocks_y), dim3(32, 24)>>>(deviceA, deviceB, deviceC, numARows, numAColumns, numBRows, numBColumns);
  cudaDeviceSynchronize();
  stopTimer(iStart, "Kernel execution");

  //@@ Copy the GPU memory back to the CPU here
  iStart = startTimer();
  cudaMemcpy(hostC, deviceC, numCRows * numCColumns * sizeof(DataType), cudaMemcpyDeviceToHost);
  stopTimer(iStart, "Copy device->host");

  //@@ Insert code below to compare the output with the reference
  //printMatrix(hostC, numCRows, numCColumns);
  bool isEqual = true;
  for (int i = 0; i < numCRows * numCColumns; i++) {
    if (std::abs(hostC[i] - resultRef[i]) >= 0.00000001) {
      isEqual = false;
      printf("/!\\ %f != %f on id %d\n", hostC[i], resultRef[i], i);
    }
  }

  if (isEqual) {
    printf("Matrices are equal!\n");
  } else {
    printf("Matrices are not equal.\n");
  }

  //@@ Free the GPU memory here
  cudaFree(deviceA);
  cudaFree(deviceB);
  cudaFree(deviceC);
  cudaDeviceReset();

  //@@ Free the CPU memory here
  free(hostA);
  free(hostB);
  free(hostC);
  free(resultRef);

  return 0;
}
