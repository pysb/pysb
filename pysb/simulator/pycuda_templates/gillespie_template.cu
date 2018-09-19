#include <curand_kernel.h>
#include <stdio.h>

#define num_species {n_species}
#define NPARAM {n_params}
#define NREACT {n_reactions}
#define NREACT_MIN_ONE NREACT-1




extern "C" {{

__constant__ int stoch_matrix[]={{
{stoch}
}};

__device__ float sum_propensities(float *a){{
    float a0 = 0;
    for(auto j=0; j<NREACT; j++){{
        a0 += a[j];
    }}
    return a0;
}}

__device__ float propensities(int *y, float *h, float *param_arry)
{{
{hazards}
return sum_propensities(h);
}}


__device__ void stoichiometry(int *y, int r){{
    auto step = r*num_species;
    for(auto i=0; i<num_species; i++){{
        y[i]+=stoch_matrix[step + i];
    }}
}}



__device__ int sample(float* a, float u){{
    int i = 0;
    for(;i < NREACT_MIN_ONE && u > a[i]; i++){{
        u -= a[i];
        }}
    return i;
}}


__device__ void update_results(int* result, int *y,  int step, int time_index){{
    for(auto j=0; j<num_species; j++){{
        result[step + j + (time_index * num_species)] = y[j];
    }}
}}

__global__ void Gillespie_all_steps(int* species_matrix, int* result,
                                    float* time, int NRESULTS,
                                    const float* param_values){{

    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    curandState randState;
//    curandStateMRG32k3a randState;
    curand_init(clock64(), tid, 0, &randState);
    auto t = time[0] ;

    int y[num_species];
    int prev[num_species];
    float A[NREACT] = {{0.0}};
    float param_arry[NPARAM] =  {{0.0}};
    const int result_stepping = tid*NRESULTS*num_species;

    // init parameters for thread
    for(auto i=0; i<NPARAM; i++){{
        param_arry[i] = param_values[tid*NPARAM + i];
        }}

    // init species counter for thread
    for(auto i=0; i<num_species; i++){{
        y[i] = species_matrix[tid*num_species + i];
        prev[i] = y[i];
        }}

    int time_index = 0;
    // beginning of loop
    while (time_index < NRESULTS){{
        while (t < time[time_index]){{

            // calculate propensities
            auto a0 = propensities(y, A, param_arry);

            // calculate two random numbers
            auto r1 =  curand_uniform(&randState);
            auto r2 =  curand_uniform(&randState);

            auto tau = -__log10f(r1)/a0;  // find time of next reaction
            auto k = sample(A, a0*r2);  // find next reaction

            t += tau; // update time

            for(auto j=0; j<num_species; j++){{
                prev[j] = y[j];
            }}

            stoichiometry(y, k); // update species matrix
            }}

        update_results(result, prev, result_stepping, time_index);
        time_index++;
        }}

    }}

}} // extern c close
