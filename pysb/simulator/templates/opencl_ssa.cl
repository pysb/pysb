#ifdef cl_amd_fp64
    #pragma OPENCL EXTENSION cl_amd_fp64 : enable
#elif defined(cl_khr_fp64)
    #pragma OPENCL EXTENSION cl_khr_fp64 : enable
#else
    #error "Double precision floating point not supported by OpenCL implementation."
#endif

#include <tyche_i.h>

// Mako variables to be set by simulator class
#define num_species ${n_species}
#define num_params ${n_params}
#define num_reaction ${n_reactions}
#define n_reaction_min_1 num_reaction-1
#define SPECIES_DTYPE  ${spc_type}
${prec}
${verbose}


#ifdef USE_DOUBLE_PRECISION
    typedef double PREC;
    #define rng_update tyche_i_double
#else
    typedef float PREC;
    #define rng_update tyche_i_float
#endif




PREC sum_propensities(PREC *a){
    PREC a0 = (PREC) 0.0;
    #pragma unroll num_reaction
    for(int j=0; j<num_reaction; j++){
        a0 += a[j];
    }
    return a0;
}

__constant char stoich_matrix[]={
${stoch}
};

PREC calc_propensities(SPECIES_DTYPE *y, PREC *h, PREC *param_vec)
{
${propensities}
return sum_propensities(h);
}


void update_state(SPECIES_DTYPE *y, int r, const __local  int *local_stoich){
    int step = r*num_species;
    #pragma unroll num_species
    for(int i=0; i<num_species; i++){
        y[i] += local_stoich[step + i];
    }
}


int find_next_reaction(PREC* a, PREC u){
    int i = 0;
    for(;i < n_reaction_min_1 && u > a[i]; i++){
        u -= a[i];
        }
    return i;
}



__kernel  void Gillespie_all_steps(
         __global const SPECIES_DTYPE* species_matrix,  // starting intial conditions
         __global SPECIES_DTYPE* result,                // storage for results
         __global const PREC* time,                     // time points to save
         __global const PREC* param_values,             // parameter values
         __global const unsigned int* random_seed,     // seeds for RNG
         const int n_timepoints,               // number of time points
         const int max_sim                     // total number of simulations
         ){

    const int tid = get_global_id(0) * get_global_size(1) + get_global_id(1);


#ifdef VERBOSE
    if (tid==0){
        printf("get_work_dim(0) %d\n", get_work_dim());
        printf("get_num_groups(0) %d\n", get_num_groups(0));
        printf("get_num_groups(1) %d\n", get_num_groups(1));
        printf("get_global_size(0) %d\n", get_global_size(0));
        printf("get_global_size(1) %d\n", get_global_size(1));
        printf("get_local_size(0) %d\n", get_local_size(0));
        printf("get_local_size(1) %d\n", get_local_size(1));
    };
#endif

    PREC propensities[num_reaction] = {{0.0}};
    PREC param_vec[num_params] =  {{0.0}};

    // init parameters for thread
    int param_stride = tid*num_params;

    #pragma unroll num_params
    for(int i=0; i<num_params; i++){
        param_vec[i] = param_values[param_stride + i];
        }
    barrier(CLK_GLOBAL_MEM_FENCE);

    // init species state per thread
    SPECIES_DTYPE species_state[num_species];
    SPECIES_DTYPE prev_state[num_species];

    // spacing from global array for copying to and from global memory
    int species_stride = tid*num_species;

    // store current state and previous state
    // previous state is used to copy trajectories once time progressed past
    // current time
    #pragma unroll num_species
    for(int i=0; i<num_species; i++){
        species_state[i] = species_matrix[species_stride + i];
        prev_state[i] = species_state[i];
        }
    // memory barrier to keep wave fronts synced.
    barrier(CLK_GLOBAL_MEM_FENCE);

    // create copy of stoichiometry matrix in local, faster than global memory
    __local  int local_stoich[num_reaction*num_species];
    for (int i=0; i<num_reaction*num_species; i++){
        local_stoich[i] = stoich_matrix[i];
    }

    barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);

    int index = tid * n_timepoints * num_species;

    // start time and time index for checking steps
    double t = (double)time[0] ;
    int time_index = 0;

    tyche_i_state state;
    tyche_i_seed(&state, random_seed[tid]);
//    tyche_i_seed(&state, 0);


    bool not_reach_time_point = t < time[time_index];
    bool cont_simulation = time_index < n_timepoints;

    // beginning of loop
    while (cont_simulation){
        while (not_reach_time_point){
            //  create backup
            #pragma unroll num_species
            for(int j=0; j<num_species; j++){
                prev_state[j] = species_state[j];
            }

            // calculate propensities
            PREC a0 = calc_propensities(species_state, propensities, param_vec);

            bool valid = a0 > (PREC) 0.0;

            if (valid){
                PREC r1 = rng_update(state);
                PREC r2 = rng_update(state);
                double inverse_a = (double)1/a0;
                double tau = -log(r1)*inverse_a; // find time of next reaction
                t += tau;  // update time

                PREC u = a0*r2;
                // find next reaction
                int rxn_k = find_next_reaction(propensities, u);
                // update species matrix
                update_state(species_state, rxn_k, local_stoich);

#ifdef VERBOSE
                if (tid == 0){
                    printf(" %d %f %f %f %f %f %d\n ", tid, a0, r1, r2, tau, t, rxn_k);
                }
#endif
#ifdef VERBOSE_MAX
                printf(" %d %.10g %.10g %.10g %.10g %.10g %d\n ", tid, a0, r1, r2, tau, t, rxn_k);
#endif
            }
            else{
                // if no positive propensity, move to last time point. No reactions
                // left to fire.
                t = time[n_timepoints-1];
            }
            not_reach_time_point = t < time[time_index];
        }

        // add an entire species time point stride
        int current_index = index + time_index * num_species;
        // iterates through each species
        #pragma unroll num_species
        for(int j=0; j<num_species; j++){
            result[current_index + j] = prev_state[j];
         }
        ++time_index;
        not_reach_time_point = t < time[time_index];
        cont_simulation = time_index < n_timepoints;
        }
}
