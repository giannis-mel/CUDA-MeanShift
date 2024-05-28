/*
 ----------------------------------------------------------
 | Title:      Mean Shift Clustering (Serial)             |
 | Author:     Giannis Meleziadis                         |
 ----------------------------------------------------------
 | Description:                                           |
 |   Serial implementation of Mean Shift clustering       |
 |   on 600 2D points. Runs on the CPU, reads input,      |
 |   performs clustering, writes output, and verifies     |
 |   against a reference.                                 |
 ----------------------------------------------------------
 | Compilation:                                           |
 |   gcc serial.c -o serial -O3 -lm                       |
 ----------------------------------------------------------
 | Execution:                                             |
 |   ./serial 0.5                                         |
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
#include <sys/time.h>
#include <float.h>

// Compile with: gcc serial.c -o serial -lm 
// Run with: ./serial 0.5

float s = 1;
float k = 20;  // Factor to scale tolerance, compensating for numerical errors
#define NUM_OF_POINTS 600
#define DIMENSIONS 2

// FUNCTIONS //

// Function to calculate the distance between two points
float distanceFunction(float *point1, float *point2) {
    float distance = 0;
    for (int i = 0; i < DIMENSIONS; i++) {
        distance += (point1[i] - point2[i]) * (point1[i] - point2[i]);
    }
    return sqrt(distance);
}


// Function to calculate the kernel weight
float kernelFunction(float *pointYk, float *arrayXi) {
    float distance = distanceFunction(pointYk, arrayXi);
    distance *= distance;
    return exp(-((distance) / (2 * (s * s))));
}


// Function to calculate the distance moved
float movedDistance(float *moved) {
    float distance = 0;
    for (int i = 0; i < DIMENSIONS; i++) {
        distance += (moved[i]) * (moved[i]);
    }
    return sqrt(distance);
}

// Shift function to update the position of a point
void shiftFunction(float *Ykplus1, float *Yk, float *X, float e, int index) {
    float numerator[DIMENSIONS] = {0};
    float denominator = 0;
    float weightFromGaussian = 0;
    float Ypoint[DIMENSIONS], Xpoint[DIMENSIONS], moved[DIMENSIONS];

    // Initialize moved to force at least one iteration
    for (int dim = 0; dim < DIMENSIONS; dim++) {
        moved[dim] = FLT_MAX;  
    }

    // Keep shifting the index point until convergence criterion is met
    while (movedDistance(moved) >= e) {
        
        // Point specified by index
        for (int i = 0; i < DIMENSIONS; i++) {
            Ypoint[i] = Yk[index * DIMENSIONS + i];
        }

        // New shift, everything zero
        denominator = 0;
        weightFromGaussian = 0;
        for (int i = 0; i < DIMENSIONS; i++) {
            numerator[i] = 0;
        }

        // Evaluate each point in X
        for (int i = 0; i < NUM_OF_POINTS; i++) {

            // Check each point of X if they will be counted for Ykplus1
            Xpoint[0] = X[i * DIMENSIONS];
            Xpoint[1] = X[i * DIMENSIONS + 1];

            float check = distanceFunction(Ypoint, Xpoint);
            if (check <= s * s && check > 0) {

                // Calculation of the weight for the point that falls within the bandwidth (check < s^2),
                // and update the numerator and denominator accordingly.
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
        printf("Parameter needed\nExample: ./serial 0.5\n");
        return 1;
    }

    float e = atof(argv[1]); // Convergence parameter

    // Open the input file
    FILE *myFile = fopen("input.txt", "r");
    if (myFile == NULL) {
        fprintf(stderr, "Failed to open input.txt.\n");
        return 1;
    }

    int nBytes = NUM_OF_POINTS * DIMENSIONS * sizeof(float*);

    // Allocate memory for data arrays
    float *arrayStatic = (float *)malloc(nBytes); // Data 
    float *arrayYk = (float *)malloc(nBytes); // MeanShift applies here y(k)
    float *arrayYkplus1 = (float *)malloc(nBytes); // Returned y(k+1)

    // Read points from file
    for (int i = 0; i < NUM_OF_POINTS; i++) {
        for (int j = 0; j < DIMENSIONS; j++) {

            // Does not read char after float (',' or '\n') and fill arrayStatic
            if (fscanf(myFile, "%f%*c", &arrayStatic[i * DIMENSIONS + j]) != 1) {
                printf("Error reading file at point %d, dimension %d.\n", i, j);
                return 1;
            }

            // Initialize arrayYk == arrayStatic
            arrayYk[i * DIMENSIONS + j] = arrayStatic[i * DIMENSIONS + j];
        }
    }

    printf("Finished reading from file\n");
    fclose(myFile);

    // Start the timer
    struct timeval startwtime, endwtime;
    gettimeofday(&startwtime, NULL);

    // Process each point
    for (int i = 0; i < NUM_OF_POINTS; i++) {
        shiftFunction(arrayYkplus1, arrayYk, arrayStatic, e, i);
    }

    // Stop the timer and calculate elapsed time
    gettimeofday(&endwtime, NULL);
    float seq_time = (float)((endwtime.tv_usec - startwtime.tv_usec) / 1.0e6 + endwtime.tv_sec - startwtime.tv_sec);
    printf("Wall clock time (serial) = %f ms\n", 1000 * seq_time);

    // Write the results to output.txt
    FILE *f = fopen("output.txt", "w");
    if (f == NULL) {
        printf("Error opening output file!\n");
        exit(1);
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

    return 0;
}
