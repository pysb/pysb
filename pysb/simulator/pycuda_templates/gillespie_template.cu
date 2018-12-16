#include <curand_kernel.h>
#include <stdio.h>

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



__device__ int sample(double* a, double u){{
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
//    curand_init(0, 0, 0, &randState);


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

            #pragma unroll
            for(int j=0; j<num_species; j++){{
                prev[j] = y[j];
                }}

            t +=  -__logf(curand_uniform(&randState))/a0;  // update time
            stoichiometry(y, sample(A, a0*curand_uniform(&randState))); // update species matrix


            // calculate two random numbers
//            double r1 =  curand_uniform(&randState);
//            double tau = -__logf(r1)/a0;  // find time of next reaction
//            t += tau;  // update time

//            double r2 =  curand_uniform(&randState);
//            double k = sample(A, a0*r2);  // find next reaction
//            stoichiometry(y, k); // update species matrix
            }}

        update_results(result, prev, result_stepping, time_index);
        time_index++;
        }}

    }}

}} // extern c close
// 0.221723079681s, each its own
// 0.222158908844s when not keeping r1
// 0.219128847122s when not keeping r1 or tau
