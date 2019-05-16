#include <curand_kernel.h>
#include <stdio.h>
#include <time.h>

#define num_species {n_species}
#define NPARAM {n_params}
#define NREACT {n_reactions}
#define NREACT_MIN_ONE NREACT-1




extern "C" {{

__device__ int stoch_matrix[]={{
{stoch}
}};

__device__ double sum_propensities(double *a){{
    double a0 = 0;
    #pragma unroll
    for(int j=0; j<NREACT; j++){{
        a0 += a[j];
    }}
    return a0;
}}

__device__ double propensities(int *y, double *h, double *param_vec)
{{
{hazards}
return sum_propensities(h);
}}


__device__ void stoichiometry(int *y, int r){{
    int step = r*num_species;
    for(int i=0; i<num_species; i++){{
        y[i]+=stoch_matrix[step + i];
    }}
}}



__device__ int sample(const double* a, double u){{
    int i = 0;
    #pragma unroll
    for(;i < NREACT_MIN_ONE && u > a[i]; i++){{
        u -= a[i];
        }}
    return i;
}}


__device__ void update_results(int* result, int *y,  int step, int time_index){{

    for(int j=0; j<num_species; j++){{
        result[step + j + (time_index * num_species)] = y[j];
    }}
}}

__global__ void Gillespie_all_steps(const int* species_matrix,  int* result,
                                    const double* time, const int NRESULTS,
                                    const double* param_values){{

    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    curandState randState;
//    curandStateMRG32k3a randState;
    curand_init(clock64(), tid, 0, &randState);


    int y[num_species];
    int prev[num_species];
    double A[NREACT] = {{0.0}};
    double param_vec[NPARAM] =  {{0.0}};
    const int result_stepping = tid*NRESULTS*num_species;

    // init parameters for thread
    #pragma unroll
    for(int i=0; i<NPARAM; i++){{
        param_vec[i] = param_values[tid*NPARAM + i];
        }}

    // init species counter for thread
    #pragma unroll
    for(int i=0; i<num_species; i++){{
        y[i] = species_matrix[tid*num_species + i];
        prev[i] = y[i];
        }}

    double t = time[0] ;
    int time_index = 0;
    // beginning of loop
    while (time_index < NRESULTS){{
        while (t < time[time_index]){{
            // calculate propensities
            double a0 = propensities(y, A, param_vec);
            if (a0 <= 0.0){{
                t = time[NRESULTS-1];
                continue;
            }}

            // store last state to be saved to output
            #pragma unroll
            for(int j=0; j<num_species; j++){{
                prev[j] = y[j];
                }}

            // calculate two random numbers
            double r1 =  curand_uniform(&randState);
            double r2 =  curand_uniform(&randState);

            // find time of next reaction and update time
            double tau = -__logf(r1)/a0;
            t += tau;

            // find next reaction and update species matrix
            double k = sample(A, a0*r2);
            stoichiometry(y, k);
            }}

        update_results(result, prev, result_stepping, time_index);
        time_index++;
        }}

    }}

}} // extern c close
