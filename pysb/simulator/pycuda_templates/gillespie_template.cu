#include <curand_kernel.h>
#include <stdio.h>
#define num_species {n_species}
#define NPARAM {n_params}
#define NREACT {n_reactions}

__constant__ int stoch_matrix[]={{
{stoch}
}};

__device__ void propensities(int *y, float *h, int tid, float *param_arry)
{{
{hazards}
}}

__device__ float single_propensities(int *y, float *h, int tid, float *param_arry, int rxn_index)
{{
{rxn_dependency}
//if (tid == 0)
//    for(int i=0; i<NREACT;i++)
//        printf("h[%i] = %f\n", i, h[i]);
}}

extern "C"{{
__device__ void stoichiometry(int *y, int r, int *stoch_matrix2){{
//__device__ void stoichiometry(int *y, int r){{
    for(int i=0; i<num_species; i++){{
//        y[i]+=stoch_matrix[r*num_species+ i];
        y[i]+=stoch_matrix2[r*num_species+ i];
    }}
}}


__device__ int sample(int nr, float* a, float u){{
    int i = 0;
    for(;i < nr - 1 && u > a[i]; i++){{
        u -= a[i];
        }}
    return i;
}}
///*
__global__ void Gillespie_all_steps(int* species_matrix, int* result, float* time, int NRESULTS,
                                    const float* param_values, int *stoch_matrix2){{

    int tid = blockDim.x * blockIdx.x + threadIdx.x;
//    curandState randState;
    curandStateMRG32k3a randState;
    curand_init(clock64(), tid, 0, &randState);
    float t = time[0] ;
    float r1 = 0.0f;
    float r2 = 0.0f;
    float a0 = 0.0f;
    int k = 0;
    int y[num_species];
    float A[NREACT];
    float param_arry[NPARAM];

    for(int i=0; i<NPARAM; i++){{
        param_arry[i] = param_values[tid*NPARAM + i];
//        if (tid == 0)
//            printf("param_array[%i] = %f\n", i, param_arry[i]);
        }}

    // start first step
    for(int i=0; i<NREACT; i++){{
        A[i] = 0;
        }}


    for(int i=0; i<num_species; i++){{
        y[i] = species_matrix[tid*num_species + i];
//        if (tid == 0)
//            printf("i = %i, y[i] = %l\n", i, y[i]);
        }}

    propensities( y, A , tid, param_arry);

    for(int i=0; i<NREACT; i++){{
//        if (tid == 0)
//            printf("i = %i, a0 = %f\n", i, A[i]);
        a0 += A[i];
        }}
//    if (tid == 0)
//        printf("ts = %f, a0 = %f\n", t, a0);

    // beginning of loop
    for(int i=0; i<NRESULTS;){{
//        __syncthreads();
        if(t>=time[i]){{
            for(int j=0; j<num_species; j++){{
                result[tid*NRESULTS*num_species + i*num_species + j] = y[j];
                }}
            i++;
//            if (tid == 0)
//                printf("ts = %f, a0 = %f\n", t, a0);
            }}
        else{{
            r1 =  curand_uniform(&randState);
            r2 =  curand_uniform(&randState);
            k = sample(NREACT, A, a0*r1);
//            stoichiometry( y ,k );
            stoichiometry( y ,k, stoch_matrix2 );
            propensities( y, A, tid , param_arry);
            a0 = 0;
            // summing up propensities
            for(int j=0; j<NREACT; j++)
                a0 += A[j];


            t += -__logf(r2)/a0;
            }}
        }}
    return;
    }}
//*/
/*
__global__ void Gillespie_one_step(int* species_matrix, int* result, float* start_time, float end_time,
                                   float* result_time, int stride, const float* param_values, ){{

    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    //curandState randState;
    curandStateMRG32k3a randState;
    curand_init(clock64(), tid, 0, &randState);
    float start_t = start_time[tid] ;

    float r1 = 0.0f;
    float r2 = 0.0f;
    float dt = 0.0f;
    float a0 = 0.0f;
    int k = 0;
    int y[num_species];
    int ylast[num_species];

    float param_arry[NPARAM];

    for(int i=0; i<NPARAM; i++){{
        param_arry[i] = param_values[tid*NPARAM + i];
        }}
    float A[NREACT];

    // start first step
    for(int i=0; i<NREACT; i++){{
        A[i] = 0;
        }}

    for(int i=0; i<num_species; i++){{
        y[i] = ((int*)( (char*) species_matrix + tid * stride))[i];
        }}

    propensities( y, A, tid , param_arry);

    for(int i=0; i<NREACT; i++){{
        a0 += A[i];
        }}

    // beginning of loop
    while (start_t < end_time){{
//        __syncthreads();
        r1 =  curand_uniform(&randState);
        r2 =  curand_uniform(&randState);
        k = sample(NREACT, A, a0*r1);
        stoichiometry( y ,k );
        propensities( y, A, tid , param_arry);
        a0 = 0;
        // summing up propensities
        for(int j=0; j<NREACT; j++){{
            a0 += A[j];
            }}
        if (tid == 0)
            printf("ts = %d, a0 = %d", start_t, a0);


        dt = -__logf(r2)/a0;
        start_t += dt;

        }}

    for(int j=0; j<num_species; j++){{
        result[tid*num_species +  j] = y[j];
        }}
    result_time[tid] = start_t;
    return;
    }}
*/
}}
