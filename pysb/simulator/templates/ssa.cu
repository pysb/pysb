#include <curand_kernel.h>
#include <stdio.h>
#include <time.h>

#define num_species {n_species}
#define num_params {n_params}
#define num_reaction {n_reactions}
#define n_reaction_min_1 num_reaction-1
//#define VERBOSE false



extern "C" {{

__device__ int stoch_matrix[]={{
{stoch}
}};

__device__ double sum_propensities(double *a){{
    double a0 = 0;
    #pragma unroll
    for(int j=0; j<num_reaction; j++){{
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
    for(;i < n_reaction_min_1 && u > a[i]; i++){{
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
    double A[num_reaction] = {{0.0}};
    double param_vec[num_params] =  {{0.0}};
    const int result_stepping = tid*NRESULTS*num_species;

    // init parameters for thread
    #pragma unroll
    for(int i=0; i<num_params; i++){{
        param_vec[i] = param_values[tid*num_params + i];
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
            // store last state to be saved to output
            #pragma unroll
            for(int j=0; j<num_species; j++){{
                prev[j] = y[j];
            }}

            // calculate propensities
            double a0 = propensities(y, A, param_vec);
            if (a0 > (double)0.0){{

                // calculate two random numbers
                double r1 =  curand_uniform(&randState);
                double r2 =  curand_uniform(&randState);

                // find time of next reaction and update time
                double tau = -__logf(r1)/a0;
                t += tau;

                // find next reaction and update species matrix
                double rxn_k = sample(A, a0*r2);
                stoichiometry(y, rxn_k);
#ifdef VERBOSE
                if (tid == 0){{ printf(" %d %f %f %f %f %f %d\n ", tid, a0, r1, r2, tau, t, rxn_k); }}
#endif
#ifdef VERBOSE_MAX
                printf(" %d %.17g %.17g %.17g %.17g %.17g %d\n ", tid, a0, r1, r2, tau, t, rxn_k);
#endif

            }}
            else{{
                t = time[NRESULTS-1];
                }}
            }}

        update_results(result, prev, result_stepping, time_index);
        time_index++;
        }}

    }}

}} // extern c close
