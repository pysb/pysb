#include <curand_kernel.h>
#include <stdio.h>
#define num_species {n_species}
#define NPARAM {n_params}
#define NREACT {n_reactions}
#define NREACT_MIN_ONE NREACT-1

__constant__ int stoch_matrix[]={{
{stoch}
}};

{stoch2}

__device__ void propensities(int *y, float *h, float *param_arry)
{{
{hazards}
}}

extern "C"{{
__device__ void stoichiometry(int *y, int r){{
    for(int i=0; i<num_species; i++){{
        y[i]+=stoch_matrix[r*num_species+ i];
    }}
}}

{update}


__device__ int sample(float* a, float u){{
    int i = 0;
    for(;i < NREACT_MIN_ONE && u > a[i]; i++){{
        u -= a[i];
        }}
    return i;
}}
///*
__global__ void Gillespie_all_steps(int* species_matrix, int* result, float* time, int NRESULTS,
                                    const float* param_values){{

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
    float A[NREACT] = {{0.0}};
    float param_arry[NPARAM];

    for(int i=0; i<NPARAM; i++){{
        param_arry[i] = param_values[tid*NPARAM + i];
        }}

    // init species array per thread
    for(int i=0; i<num_species; i++){{
        y[i] = species_matrix[tid*num_species + i];
        }}

    // calculate initial propensities
    propensities(y, A, param_arry);
    for(int j=0; j<NREACT; j++){{
        a0 += A[j];
    }}
    // beginning of loop
    for(int i=0; i<NRESULTS;){{
        if (t<time[i]){{
            r1 =  curand_uniform(&randState);
            r2 =  curand_uniform(&randState);
            k = sample(A, a0*r1);
//            stoichiometry(y, k);
            stoichiometry2(y, k);
            propensities(y, A, param_arry);
            a0 = 0;
            // summing up propensities
            for(int j=0; j<NREACT; j++)
                a0 += A[j];
            t += -__logf(r2)/a0;
            }}
       else {{
            for(int j=0; j<num_species; j++){{
                result[tid*NRESULTS*num_species + i*num_species + j] = y[j];
                }}
            i++;
            }}
        }}

    return;
    }}
//*/
///*
__global__ void Gillespie_one_step(int* species_matrix, int* result, float* start_time, float end_time,
                                   float* result_time, const float* param_values){{

    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    curandState randState;
//    curandStateMRG32k3a randState;
    curand_init(clock64(), tid, 0, &randState);
    float start_t = start_time[tid] ;

    float r1 = 0.0f;
    float r2 = 0.0f;
    float dt = 0.0f;
    float a0 = 0;
    int k = 0;
    int y[num_species];

    float param_arry[NPARAM];

    for(int i=0; i<NPARAM; i++){{
        param_arry[i] = param_values[tid*NPARAM + i];
        }}
    float A[NREACT] = {{0.0}};

    // init species
    for(int i=0; i<num_species; i++){{
        y[i] = species_matrix[tid*num_species + i];
        }}
    // calculate initial propensities
    propensities(y, A, param_arry);
    for(int j=0; j<NREACT; j++){{
        a0 += A[j];
    }}
    // beginning of loop
    while (start_t < end_time){{
//        __syncthreads();
        r1 =  curand_uniform(&randState);
        r2 =  curand_uniform(&randState);
        k = sample(A, a0*r1);
        stoichiometry2( y ,k );
        propensities( y, A, param_arry);
        a0 = 0;
        // summing up propensities
        for(int j=0; j<NREACT; j++){{
            a0 += A[j];
            }}

        dt = -__logf(r2)/a0;
        start_t += dt;
        }}

    for(int j=0; j<num_species; j++){{
        result[tid*num_species +  j] = y[j];
        }}
    result_time[tid] = start_t;
    return;
    }}

}}