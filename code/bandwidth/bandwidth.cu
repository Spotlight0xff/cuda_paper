/*
 * bandwidth-test, used for plotting
 * can use pageable or pinned memory
 */
#include <stdio.h>
#include <getopt.h>
#include <cuda_runtime.h>

#define YELLOW "\e[1;33m"
#define RED    "\e[1;31m"
#define WHITE  "\e[1;00m"



typedef struct {
    float mseconds_h2d;
    float mseconds_d2h;
    float bandwidth_h2d;
    float bandwidth_d2h;
    bool err;
} Profiling;

typedef enum {
    PLOT,
    HUMAN
} Format;

typedef enum {
    PINNED,
    PAGEABLE
} Mode;



struct option long_options[] = {
    {"num"      , required_argument, 0, 0},
    {"mode"     , required_argument, 0, 0},
    {"format"   , required_argument, 0, 0},
    {"help"     , 0                , 0, 0},
    {0          , 0                , 0, 0}
};

float calcBandwidth(size_t size, float mseconds) {
    return (size/((float)1024*1024*1024)) / (mseconds / 1000.0f);
}

/*! perform a profiled memory transfer
 *
 *  with pinned or pageable memory using size bytes
 *
 */
Profiling profileTransfer(size_t size, Mode mode, Format format) {
    cudaEvent_t start, stop;
    cudaError_t err;
    void *devPtr, *hostPtr;
    Profiling profile = {0};
    profile.err = true;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    if (mode == PINNED) {
        err = cudaMallocHost(&hostPtr, size);
        if (err != cudaSuccess) {
            fprintf(stderr, "! host pinned memory allocation failed.\n! %s\n", 
                            cudaGetErrorString(err));
            return profile;
        }
    }else if (mode == PAGEABLE) {
        hostPtr = malloc(size);
        if (hostPtr == NULL) {
            printf("! host memory allocation failed!\n");
            return profile; // lets continue, though it is most likely not working
        }
    }else {
        return profile;
    }

    if (format == HUMAN) {
        printf(" > allocating %s%d%s bytes in device memory\n", YELLOW, size, WHITE);
    }
    cudaEventRecord(start);
    err = cudaMalloc(&devPtr, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "! cuda error: %s\n", cudaGetErrorString(err));
        return profile;
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&profile.mseconds_h2d, start, stop);
    if (format == HUMAN) {
        printf(" > allocated in %.2f milliseconds\n");
    }

    //Host --> Device
    cudaEventRecord(start);
    cudaMemcpy(devPtr, hostPtr, size, cudaMemcpyDefault);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&profile.mseconds_h2d, start, stop);
    profile.bandwidth_h2d = calcBandwidth(size, profile.mseconds_h2d);
    if (format == HUMAN) {
        printf("  > copied host->device in %.2f milliseconds\n", profile.mseconds_h2d);
        printf("  > this equals a bandwidth of %s%.2f%s GB/s\n", RED, profile.bandwidth_h2d, WHITE);
    }
    
    // Device --> Host
    cudaEventRecord(start);
    cudaMemcpy(hostPtr, devPtr, size, cudaMemcpyDefault);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&profile.mseconds_d2h, start, stop);
    profile.bandwidth_d2h = calcBandwidth(size, profile.mseconds_d2h);
    if (format == HUMAN) {
        printf("  > copied device->host in %.2f milliseconds\n", profile.mseconds_d2h);
        printf("  > this equals a bandwidth of %.2f GB/s\n", profile.bandwidth_d2h);
    }

    if (mode == PINNED) {
        cudaFreeHost(hostPtr);
    }else if (mode == PAGEABLE) {
        free(hostPtr);
    }
    cudaFree(devPtr);

    profile.err = false;
    return profile;
}


void showUsage(void) {
    printf("""usage: bandwidth [-p] [-o]\n""");
}



int main(int argc, char** argv)
{
    int num_samples = 0; // number of samples
    Mode mode = PAGEABLE; // Transfer-mode, pinned or pageable
    Format format = HUMAN; // Format: human-readable or plotable
    bool verbose = false;
    char *output = NULL;
    FILE *fd = NULL;

    while(1) {
        int option_index = 0;
        int c = getopt_long(argc, argv, "n:m:f:ho:", long_options, &option_index);
        if (c == -1) {
            break;
        }
        switch(c) {
            case 'm':
                if (optarg) {
                    if (strcmp(optarg, "pinned") == 0) {
                        mode = PINNED;
                    } else if ( strcmp(optarg, "pageable") == 0) {
                        mode = PAGEABLE;
                    }
                }
            break;

            case 'f':
                if (optarg) {
                    if (strcmp(optarg, "plot") == 0) {
                        format = PLOT;
                    } else if (strcmp(optarg, "human") == 0 ||
                               strcmp(optarg, "plain") == 0) {
                        format = HUMAN;
                    } 
                }
                break;

            case 'o': //output file
                if (optarg) {
                    size_t len = strnlen(optarg, 255);
                    output = (char*) malloc(sizeof(char) * len + 1);
                    strncpy(output, optarg, len);
                    output[len] = 0; // null-terminate
                }
                break;
            case 'h':
                showUsage();
                exit(0);

            case 'n': // --num
                if (optarg) {
                    num_samples = atoi(optarg);
                }
                break;

            case 'v': // verbose
                verbose = true;
                break;


            default:
                printf("error\n");
                exit(-1);
        }
    }
    if (num_samples != 0 && format == HUMAN) {
        fprintf(stderr, "!! can't provide amount of samples for human format.\n");
        exit(1);
    }
    cudaDeviceReset();

    size_t *samples = NULL;
    if (format == HUMAN) {
        num_samples = 0;
        size_t samples_tmp[] = {1, 64, 512, 1024, 1024*1024, 1024*1024*10, 1024*1024*10, 1024*1024*200, 1024*1024*500, 1024*1024*900};
        num_samples = sizeof(samples_tmp) / sizeof(samples_tmp[0]);
        printf("num: %d\n", num_samples);
        samples =(size_t*) malloc(sizeof(size_t) * num_samples);
        for(int i=0; i < num_samples; i ++) {
            *(samples+i) = samples_tmp[i]; // i know, i know...
        } // could've used C++ new op...
    }else if (format == PLOT) { // use more data points for plotting
        num_samples = num_samples ? num_samples :  128;
        samples = (size_t*)malloc(sizeof(size_t) * num_samples);
        size_t max_size = 1024*1024*700;
        int i=0;
        for (size_t size = 1; size < max_size; size += max_size/num_samples, i++) {
            samples[i] = size;
        }
    }


    Profiling profile = {0};
    if (output != NULL) {
        fd = fopen(output, "a+");
        if (fd == NULL) {
            fprintf(stderr, "!! error opening output-file.\n");
            exit(1);
        }
    }else {
        fd = stdout;
    }
    for (int i=0; i < num_samples; i++) {
        size_t size = samples[i];
        if (format == PLOT) {
            if (i % (num_samples/10) == 0 && i != 0) {
                fprintf(stderr, "%d[%d]\n", i, size);
            }else {
                fprintf(stderr, ".");
            }
        }

        if (format == HUMAN) {
            printf("* test #%d: %d bytes ( %.2f MB)", i, size, size / (float)(1024*1024));
            printf("> allocating %d bytes in host memory\n", size);
        }
        profile = profileTransfer(size, mode, format);
        fprintf(fd, "%d\t%f\t%f\t%f\n", size, profile.bandwidth_h2d, profile.bandwidth_d2h,
                                        (profile.bandwidth_h2d + profile.bandwidth_d2h) / 2.0);
    }
    if (fd != NULL) {
        fclose(fd);
    }
    free(samples);

    return 0;
}

