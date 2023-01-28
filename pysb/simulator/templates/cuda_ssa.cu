#include <curand_kernel.h>
#include <stdio.h>
#include <time.h>

#define num_species ${n_species}
#define num_params ${n_params}
#define num_reaction ${n_reactions}
#define n_reaction_min_1 num_reaction-1
#define SPECIES_DTYPE  ${spc_type}

${prec}
${verbose}

#ifdef USE_DOUBLE_PRECISION
    typedef double PREC;
    #define rng_update curand_uniform_double
#else
    typedef float PREC;
    #define rng_update curand_uniform
#endif

extern "C" {

__device__ const char stoch_matrix[]={
${stoch}
};

__device__ PREC sum_propensities(PREC *a){
    PREC a0 = 0;
    #pragma unroll
    for(int j=0; j<num_reaction; j++){
        a0 += a[j];
    }
    return a0;
}

__device__ PREC propensities(SPECIES_DTYPE *y, PREC *h, PREC *param_vec)
{
${propensities}
return sum_propensities(h);
}


__device__ void stoichiometry(SPECIES_DTYPE *y, int r,
                              const int *local_stoich_matrix){
    int step = r*num_species;
    #pragma unroll
    for(int i=0; i<num_species; i++){
        y[i] += local_stoich_matrix[step + i];
    }
}


__device__ int find_next_reaction(PREC* a, PREC u){
    SPECIES_DTYPE i = 0;
    #pragma unroll
    for(;i < n_reaction_min_1 && u > a[i]; i++){
        u -= a[i];
        }
    return i;
}


__device__ void update_results(SPECIES_DTYPE* result, SPECIES_DTYPE *y,
                               const int step, int time_index){
    #pragma unroll
    for(int j=0; j<num_species; j++){
        result[step + j + (time_index * num_species)] = y[j];
    }
}

__global__ void Gillespie_all_steps(
                 const SPECIES_DTYPE* species_matrix,
                 SPECIES_DTYPE* result,
                 const PREC* time,
                 const int NRESULTS,
                 const PREC* param_values,
                 const unsigned int* SEED
                 ){

    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    unsigned int local_seed = SEED[tid];
    curandStateMRG32k3a_t  randState;
    curand_init(local_seed, tid, 0, &randState);
//    curand_init(0, 0, 0, &randState);


    SPECIES_DTYPE y[num_species];
    SPECIES_DTYPE prev[num_species];
    PREC A[num_reaction] = {0.0};
    PREC param_vec[num_params] =  {0.0};

    // spacing from global
    const int result_stepping = tid*NRESULTS*num_species;

    // init parameters for thread
    #pragma unroll
    for(int i=0; i<num_params; i++){
        param_vec[i] = param_values[tid*num_params + i];
        }

    // init species counter for thread
    #pragma unroll
    for(int i=0; i<num_species; i++){
        y[i] = species_matrix[tid*num_species + i];
        prev[i] = y[i];
        }

    // create copy of stoichiometry matrix in shared, faster than global memory
    __shared__ int local_stoich_matrix[num_reaction*num_species];

    for (int i=0; i<num_reaction*num_species; i++){
            local_stoich_matrix[i] = stoch_matrix[i];
        }
    __syncthreads();
    double t = (double)time[0] ;
    int time_index = 0;
    bool not_reach_time_point = true;
    bool cont_simulation = time_index < NRESULTS;

    // beginning of loop
    while (cont_simulation){
        while (not_reach_time_point){
            // store last state to be saved to output
            #pragma unroll
            for(int j=0; j<num_species; j++){
                prev[j] = y[j];
            }

            // calculate propensities
            PREC a0 = propensities(y, A, param_vec);
            bool valid = a0 >= (PREC) 0.0;
            if (valid){

                // calculate two random numbers
                PREC r1 =  rng_update(&randState);
                PREC r2 =  rng_update(&randState);

                // find time of next reaction and update time
                double inverse_a = (double)1/a0;
                double tau = -log(r1)*inverse_a;
                t += tau;

                // find next reaction and update species matrix
                int k = find_next_reaction(A, a0*r2);
                stoichiometry(y, k, local_stoich_matrix);

#ifdef VERBOSE
                if (tid == 0){
                     printf(" %d %.17g %.17g %.17g %.17g %.17g %d\n ", tid, a0, r1, r2, tau, t, k);
                 }
#endif
#ifdef VERBOSE_MAX
                    printf(" %d %.17g %.17g %.17g %.17g %.17g %d\n ", tid, a0, r1, r2, tau, t, k);
#endif
                }
            else {
            // if no positive propensity, move to last time point. No reactions
            // left to fire.
                t = time[NRESULTS-1];
            }

            not_reach_time_point = t < time[time_index];
        }

        update_results(result, prev, result_stepping, time_index);
        time_index++;
        not_reach_time_point = t < time[time_index];
        cont_simulation = time_index < NRESULTS;
        } // while(time_index < NRESULTS) close
} // function close
} // extern c close
