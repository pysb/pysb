#include <curand_kernel.h>
#include <stdio.h>
#include <time.h>

#define num_species {n_species}
#define NPARAM {n_params}
#define NREACT {n_reactions}
#define NREACT_MIN_ONE NREACT-1
#define SPECIES_DTYPE int
#define USE_LOCAL
// used as place holder. WIll be uncommented if simulator created with verbose>0
//#define VERBOSE false



extern "C" {{

__device__ const SPECIES_DTYPE stoch_matrix[]={{
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

__device__ double propensities(SPECIES_DTYPE *y, double *h, double *param_vec)
{{
{propensities}
return sum_propensities(h);
}}


__device__ void stoichiometry2(SPECIES_DTYPE *y, int r, int *stoch_matrix2){{
    int step = r*num_species;
    #pragma unroll
    for(int i=0; i<num_species; i++){{
        y[i]+=stoch_matrix2[step + i];
    }}
}}


__device__ SPECIES_DTYPE sample(const double* a, double u){{
    SPECIES_DTYPE i = 0;
    #pragma unroll
    for(;i < NREACT_MIN_ONE && u > a[i]; i++){{
        u -= a[i];
        }}
    return i;
}}


__device__ void update_results(SPECIES_DTYPE* result, SPECIES_DTYPE *y,  int step, int time_index){{
    #pragma unroll
    for(int j=0; j<num_species; j++){{
        result[step + j + (time_index * num_species)] = y[j];
    }}
}}

__global__ void Gillespie_all_steps(
                 const SPECIES_DTYPE* species_matrix,
                 SPECIES_DTYPE* result,
                 const double* time,
                 const int NRESULTS,
                 const double* param_values
                 ){{

    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    curandStateMRG32k3a_t  randState;
//    curand_init(clock64(), tid, 0, &randState);
    curand_init(0, tid, 0, &randState);


    SPECIES_DTYPE y[num_species];
    SPECIES_DTYPE prev[num_species];
    double A[NREACT] = {{0.0}};
    double param_vec[NPARAM] =  {{0.0}};

    // spacing from global
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

#ifdef USE_LOCAL
    __shared__ int stoch_matrix2[NREACT*num_species];
#endif
#ifdef USE_PRIVATE
    int stoch_matrix2[NREACT*num_species];
#endif
    for (int i=0; i<NREACT*num_species; i++){{
            stoch_matrix2[i] = stoch_matrix[i];
        }}
    __syncthreads();
    double t = time[0] ;
    int time_index = 0;
    bool not_reach_time_point = true;
    bool cont_simulation = time_index < NRESULTS;
    // beginning of loop
    while (cont_simulation){{
        while (not_reach_time_point){{
            // store last state to be saved to output
//            #pragma unroll
            for(int j=0; j<num_species; j++){{
                prev[j] = y[j];
            }}

            // calculate propensities
            double a0 = propensities(y, A, param_vec);
            bool valid = a0 >= (double) 0.0;
            if (valid){{

                // calculate two random numbers
                double r1 =  curand_uniform_double(&randState);
                double r2 =  curand_uniform_double(&randState);

                // find time of next reaction and update time
                double tau = -__logf(r1)/a0;
                t += tau;

                // find next reaction and update species matrix
                double k = sample(A, a0*r2);
                stoichiometry2(y, k, stoch_matrix2);

#ifdef VERBOSE
                    if (tid == 0){{ printf(" %d %f %f %f %f %f %d\n ", tid, a0, r1, r2, tau, t, rxn_k); }}
#endif
#ifdef VERBOSE_MAX
                    printf(" %d %.17g %.17g %.17g %.17g %.17g %d\n ", tid, a0, r1, r2, tau, t, rxn_k);
#endif
                }}
            else{{ t = time[NRESULTS-1];}}

            not_reach_time_point = t < time[time_index];
        }}

        update_results(result, prev, result_stepping, time_index);
        time_index++;
        not_reach_time_point = t < time[time_index];
        cont_simulation = time_index < NRESULTS;
        }} // while(time_index < NRESULTS) close
}} // function close
}} // extern c close
