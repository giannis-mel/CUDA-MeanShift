/*
 ----------------------------------------------------------
 | Title:   Mean Shift Clustering with CUDA (Non-shared)  |
 | Author:  Giannis Meleziadis                            |
 ----------------------------------------------------------
 | Description:                                           |
 |   CUDA implementation of Mean Shift clustering         |
 |   on 600 2D points without using shared memory.        |
 |   Reads input, clusters using GPU, writes output,      |
 |   and verifies against a reference.                    |
 ----------------------------------------------------------
 | Compilation:                                           |
 |   nvcc nonshared.cu -o nonshared -O3 -lm               |
 ----------------------------------------------------------
 | Execution:                                             |
 |   ./nonshared 0.5                                      |
 ----------------------------------------------------------
 | Files:                                                 |
 |   input.txt - Input points                             |
 |   output.txt - Clustered output                        |
 |   output_reference.txt - Reference for validation      |
 ----------------------------------------------------------
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <sys/time.h>
#include <float.h>
#include <assert.h>

float k = 20;  // Factor to scale tolerance, compensating for numerical errors
#define NUM_OF_POINTS 600
#define DIMENSIONS 2
#define BLOCK_SIZE 512 

inline cudaError_t checkCuda(cudaError_t result)
{
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
        assert(result == cudaSuccess);
    }
    return result;
}

// CUDA kernel to calculate Euclidean distance between two points
__device__ float distanceFunction(const float *point1, const float *point2) {
    float distance = 0;
    for (int i = 0; i < DIMENSIONS; i++) {
        distance += (point1[i] - point2[i]) * (point1[i] - point2[i]);
    }
    return sqrt(distance);
}

// CUDA kernel to calculate the weight using Gaussian kernel function
__device__ float kernelFunction(const float *pointYk, const float *arrayXi) {
    float s = 1;
    float distance = distanceFunction(pointYk, arrayXi);
    distance *= distance;
    return exp(-((distance) / (2 * (s * s))));
}

// CUDA kernel to calculate the magnitude of movement vector
__device__ float movedDistance(const float *moved) {
    float distance = 0;
    for (int i = 0; i < DIMENSIONS; i++) {
        distance += (moved[i]) * (moved[i]);
    }
    return sqrt(distance);
}

// Main kernel function that performs the MeanShift algorithm on given data points
__global__ void shiftFunction(float *Ykplus1, float *Yk, const float *X, float e) {
    float s = 1;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= NUM_OF_POINTS) return;  // Ensure thread index is within bounds

    float numerator[DIMENSIONS] = {0};
    float denominator = 0;
    float weightFromGaussian = 0;
    float Ypoint[DIMENSIONS], Xpoint[DIMENSIONS], moved[DIMENSIONS]; 

    // Initialize moved to force at least one iteration
    for (int dim = 0; dim < DIMENSIONS; dim++) {
        moved[dim] = FLT_MAX;  
    }

    // Iteratively shift point until the movement is below the threshold 'e'
    while (movedDistance(moved) >= e) {
        // Load current point coordinates
        for (int i = 0; i < DIMENSIONS; i++) {
            Ypoint[i] = Yk[index * DIMENSIONS + i];
        }

        // Reset sums for the new iteration
        denominator = 0;
        for (int i = 0; i < DIMENSIONS; i++) {
            numerator[i] = 0;
        }

        // Evaluate each point in X
        for (int i = 0; i < NUM_OF_POINTS; i++) {
            for (int j = 0; j < DIMENSIONS; j++) {
                Xpoint[j] = X[i * DIMENSIONS + j];
            }
            float check = distanceFunction(Ypoint, Xpoint);
            if (check <= s * s && check > 0) {
                weightFromGaussian = kernelFunction(Ypoint, Xpoint);
                for (int j = 0; j < DIMENSIONS; j++) {
                    numerator[j] += weightFromGaussian * X[i * DIMENSIONS + j];
                }
                denominator += weightFromGaussian;
            }
        }

        // Update the position of the current point using numerator, denominator
        // and calculate the shift distance for the next iteration
        for (int j = 0; j < DIMENSIONS; j++) {
            float newY = numerator[j] / denominator;
            moved[j] = newY - Yk[index * DIMENSIONS + j];
            Yk[index * DIMENSIONS + j] = newY;
            Ykplus1[index * DIMENSIONS + j] = newY;
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
        fprintf(stderr, "Parameter needed\nExample: ./nonshared 0.5\n");
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
    checkCuda(cudaMalloc((void **)&deviceArrayStatic, nBytes));
    checkCuda(cudaMalloc((void **)&deviceArrayYk, nBytes));
    checkCuda(cudaMalloc((void **)&deviceArrayYkplus1, nBytes));

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
    checkCuda(cudaMemcpy(deviceArrayStatic, arrayStatic, nBytes, cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(deviceArrayYk, arrayYk, nBytes, cudaMemcpyHostToDevice));

    // GPU information
    int deviceId;
    cudaGetDevice(&deviceId);
    cudaDeviceProp props;
    checkCuda(cudaGetDeviceProperties(&props, deviceId));
    int multiProcessorCount = props.multiProcessorCount;
    
    int threadsPerBlock = BLOCK_SIZE;
    int numberOfBlocks = 32 * multiProcessorCount;

    struct timeval startwtime, endwtime;
    gettimeofday(&startwtime, NULL);

    // Launch kernel
    shiftFunction<<<numberOfBlocks, threadsPerBlock>>>(deviceArrayYkplus1, deviceArrayYk, deviceArrayStatic, e);
    checkCuda(cudaDeviceSynchronize());

    gettimeofday(&endwtime, NULL);
    float seq_time = (float)((endwtime.tv_usec - startwtime.tv_usec)/1.0e6 + endwtime.tv_sec - startwtime.tv_sec);
    printf("Wall clock time (using cuda) = %f ms\n", 1000 * seq_time);

    // Copy results back to host
    checkCuda(cudaMemcpy(arrayYk, deviceArrayYk, nBytes, cudaMemcpyDeviceToHost));

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
    checkCuda(cudaFree(deviceArrayStatic));
    checkCuda(cudaFree(deviceArrayYk));
    checkCuda(cudaFree(deviceArrayYkplus1));

    return 0;
}
