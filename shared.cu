#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <sys/time.h>
#include <float.h>

float k = 20;  // Factor to scale tolerance, compensating for numerical errors
#define NUM_OF_POINTS 600
#define DIMENSIONS 2
#define BLOCK_SIZE 256  // Adjust based on the maximum threads per block your GPU supports

//for compile : nvcc shared.cu -o shared -O3 -lm 
//for run: ./shared 0.5

// CUDA kernel to calculate Euclidean distance between two points
__device__ float distanceFunction(float point1[DIMENSIONS], float point2[DIMENSIONS]) {
    float distance = 0;
    for (int i = 0; i < DIMENSIONS; i++) {
        distance += (point1[i] - point2[i]) * (point1[i] - point2[i]);
    }
    return sqrt(distance);
}

// CUDA kernel to calculate the weight using Gaussian kernel function
__device__ float kernelFunction(float pointYk[DIMENSIONS], float arrayXi[DIMENSIONS]) {
    float s = 1;
    float distance = distanceFunction(pointYk, arrayXi);
    return exp(-((distance * distance) / (2 * s * s)));
}

// CUDA kernel to calculate the magnitude of movement vector
__device__ float movedDistance(float moved[DIMENSIONS]) {
    float distance = 0;
    for (int i = 0; i < DIMENSIONS; i++) {
        distance += moved[i] * moved[i];
    }
    return sqrt(distance);
}

// Main kernel function that performs the MeanShift algorithm on given data points using shared memory
__global__ void shiftingFunction(float *Ykplus1, float *Yk, float *X, float e) {
    float s = 1;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= NUM_OF_POINTS) return;

    __shared__ float sharedX[NUM_OF_POINTS * DIMENSIONS];

    // Load data into shared memory: each thread loads its part
    int numThreads = blockDim.x * gridDim.x;
    int loadIdx = threadIdx.x + blockIdx.x * blockDim.x;
    while (loadIdx < NUM_OF_POINTS) {
        for (int dim = 0; dim < DIMENSIONS; dim++) {
            sharedX[loadIdx * DIMENSIONS + dim] = X[loadIdx * DIMENSIONS + dim];
        }
        loadIdx += numThreads;
    }
    __syncthreads(); // Ensure all data is loaded

    float Ypoint[DIMENSIONS], Xpoint[DIMENSIONS], moved[DIMENSIONS];
    float numerator[DIMENSIONS] = {0}, denominator = 0;

    // Initialize moved to force at least one iteration
    for (int dim = 0; dim < DIMENSIONS; dim++) {
        moved[dim] = FLT_MAX;  
    }
    float weightFromGaussian = 0;

    while (movedDistance(moved) >= e) {
        // Load current point coordinates
        for (int i = 0; i < DIMENSIONS; i++) {
            Ypoint[i] = Yk[index * DIMENSIONS + i];
        }
        
        for (int i = 0; i < DIMENSIONS; i++) {
            numerator[i] = 0;
        }
        denominator = 0;

        // Evaluate each point in X
        for (int i = 0; i < NUM_OF_POINTS; i++) {
            for (int j = 0; j < DIMENSIONS; j++) {
                Xpoint[j] = sharedX[i * DIMENSIONS + j];
            }
            float check = distanceFunction(Ypoint, Xpoint);
            if (check <= s * s && check > 0) {
                weightFromGaussian = kernelFunction(Ypoint, Xpoint);
                for (int j = 0; j < DIMENSIONS; j++) {
                    numerator[j] += weightFromGaussian * sharedX[i * DIMENSIONS + j];
                }
                denominator += weightFromGaussian;
            }
        }

        // Update the position of the current point using numerator, denominator
        for (int j = 0; j < DIMENSIONS; j++) {
            Ykplus1[index * DIMENSIONS + j] = numerator[j] / denominator;
        }

        // Calculate the shift distance and update Yk for the next iteration
        for (int j = 0; j < DIMENSIONS; j++) {
            moved[j] = Ykplus1[index * DIMENSIONS + j] - Yk[index * DIMENSIONS + j];
            Yk[index * DIMENSIONS + j] = Ykplus1[index * DIMENSIONS + j];
        }
    }
}

// Function to verify results
int verifyResults(float *results, const char *refFileName, float tolerance) {
    FILE *fRef = fopen(refFileName, "r");
    if (fRef == NULL) {
        fprintf(stderr, "Failed to open %s for validation.\n", refFileName);
        return -1; // File opening error
    }

    float refValue;
    int errors = 0;
    for (int i = 0; i < NUM_OF_POINTS; i++) {
        for (int j = 0; j < DIMENSIONS; j++) {
            if (fscanf(fRef, "%f%*c", &refValue) != 1) {
                fprintf(stderr, "Error reading validation file at point %d, dimension %d.\n", i, j);
                fclose(fRef);
                return -2; // Reading error
            }

            if (fabs(refValue - results[i * DIMENSIONS + j]) > tolerance) {
                errors++;
            }
        }
    }
    fclose(fRef);
    return errors;
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "Parameter needed\nExample: ./shared 0.5\n");
        return 1;
    }

    float e = atof(argv[1]); // Convergence threshold from command line
    float *arrayStatic, *arrayYk, *arrayYkplus1;
    float *deviceArrayStatic, *deviceArrayYk, *deviceArrayYkplus1;

    // Allocate memory for host arrays
    size_t nBytes = NUM_OF_POINTS * DIMENSIONS * sizeof(float);
    arrayStatic = (float *)malloc(nBytes);
    arrayYk = (float *)malloc(nBytes);
    arrayYkplus1 = (float *)malloc(nBytes);

    // Allocate memory for device arrays
    cudaMalloc((void **)&deviceArrayStatic, nBytes);
    cudaMalloc((void **)&deviceArrayYk, nBytes);
    cudaMalloc((void **)&deviceArrayYkplus1, nBytes);

    // Open input file
    FILE *myFile = fopen("input.txt", "r");
    if (myFile == NULL) {
        fprintf(stderr, "Failed to open input.txt.\n");
        return 1;
    }

    // Read data from file
    for (int i = 0; i < NUM_OF_POINTS; i++) {
        for (int j = 0; j < DIMENSIONS; j++) {
            if (fscanf(myFile, "%f%*c", &arrayStatic[i * DIMENSIONS + j]) != 1) {
                fprintf(stderr, "Error reading file at point %d, dimension %d.\n", i, j);
                fclose(myFile);
                return 1;
            }
            arrayYk[i * DIMENSIONS + j] = arrayStatic[i * DIMENSIONS + j];
        }
    }
    fclose(myFile);

    // Transfer data from host to device
    cudaMemcpy(deviceArrayStatic, arrayStatic, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceArrayYk, arrayYk, nBytes, cudaMemcpyHostToDevice);
    
    // Define block and grid sizes
    dim3 blockSize(BLOCK_SIZE);
    dim3 gridSize((NUM_OF_POINTS + BLOCK_SIZE - 1) / BLOCK_SIZE);

    struct timeval startwtime, endwtime;
    gettimeofday(&startwtime, NULL);

    // Launch kernel
    shiftingFunction<<<gridSize, blockSize>>>(deviceArrayYkplus1, deviceArrayYk, deviceArrayStatic, e);
    cudaDeviceSynchronize();

    gettimeofday(&endwtime, NULL);
    float seq_time = (float)((endwtime.tv_usec - startwtime.tv_usec)/1.0e6 + endwtime.tv_sec - startwtime.tv_sec);
    printf("Wall clock time (using cuda - shared memory utilization) = %f ms\n", 1000 * seq_time);

    // Copy results back to host
    cudaMemcpy(arrayYk, deviceArrayYkplus1, nBytes, cudaMemcpyDeviceToHost);

    // Write results to output.txt
    FILE *f = fopen("output.txt", "w");
    if (f == NULL) {
        fprintf(stderr, "Error opening output file!\n");
        return 1;
    }
    for (int i = 0; i < NUM_OF_POINTS; i++) {
        for (int j = 0; j < DIMENSIONS; j++) {
            fprintf(f, "%f ", arrayYk[i * DIMENSIONS + j]);
        }
        fprintf(f, "\n");
    }
    fclose(f);

    // Validate the results by comparing with a reference file
    int errors = verifyResults(arrayYk, "output_reference.txt", k * e);
    printf("The number of errors is %d\n", errors);

    free(arrayStatic);
    free(arrayYk);
    free(arrayYkplus1);
    cudaFree(deviceArrayStatic);
    cudaFree(deviceArrayYk);
    cudaFree(deviceArrayYkplus1);

    return 0;
}
