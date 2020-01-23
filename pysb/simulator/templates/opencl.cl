#ifdef cl_khr_fp64
    #pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)
    #pragma OPENCL EXTENSION cl_amd_fp64 : enable
#else
    #error "Double precision floating point not supported by OpenCL implementation."
#endif
#include <pyopencl-random123/philox.cl>

#define num_species {n_species}
#define num_params {n_params}
#define num_reaction {n_reactions}
#define n_reaction_min_1 num_reaction-1

// used as place holder. WIll be uncommented if simulator created with verbose>0
//#define VERBOSE false
#define USE_DP

typedef double4 output_vec_t;
typedef philox4x32_ctr_t ctr_t;
typedef philox4x32_key_t key_t;
/*

Unlike cuda, there is not an easy to use library (curand) that allows one to
keep track of RNG across threads.

The code to generate random numbers is taken from pyopencl and based on
Random123 library.
gen_bits



http://www.thesalmons.org/john/random123/releases/latest/docs/index.html
https://software.intel.com/en-us/mkl-vsnotes-philox4x32-10

*/

// Max integers based on single or double precision
__constant double RAND_MAX_INT = (double) 1./(double) UINT_MAX;
__constant double RAND_MAX_LONG = (double) 1./(double) ULONG_MAX;


// convert int to double or float (0-1] (depending on precision level)
#ifdef USE_DP
    #define GET_RANDOM_NUM(gen) (double) 1 * (RAND_MAX_INT * convert_double4(gen) + RAND_MAX_LONG * convert_double4(gen));
#else
    #define GET_RANDOM_NUM(gen) (double) 1 * (RAND_MAX_INT * convert_double4(gen));
#endif


// gen_bits implemented in pyopencl
// https://github.com/inducer/pyopencl/blob/master/pyopencl/clrandom.py
uint4 gen_bits(key_t *key, ctr_t *ctr)
{{
    union {{
        ctr_t ctr_el;
        uint4 vec_el;
    }} u;

    u.ctr_el = philox4x32(*ctr, *key);
    if (++ctr->v[0] == 0)
        if (++ctr->v[1] == 0)
            ++ctr->v[2];
    return u.vec_el;
}}


double sum_propensities(double *a){{
    double a0 = (double) 0.0;
    #pragma unroll num_reaction
    for(unsigned int j=0; j<num_reaction; j++){{
        a0 += (double) a[j];
    }}
    return a0;
}}

__constant int stoch_matrix[]={{
{stoch}
}};

double calc_propensities(int *y, double *h, double *param_vec)
{{
{propensities}
return sum_propensities(h);
}}


void stoichiometry(int *y, unsigned int r){{
    unsigned int step = r*num_species;
    #pragma unroll num_species
    for(unsigned int i=0; i<num_species; i++){{
        y[i] += stoch_matrix[step + i];
    }}

}}


unsigned int sample(double* a, double u){{
    unsigned int i = 0;
    for(;i < n_reaction_min_1 && u > a[i]; i++){{
        u -= a[i];
        }}
    return i;
}}



__kernel  void Gillespie_all_steps(
         __global const unsigned int* species_matrix,   // starting intial conditions
         __global unsigned int* result,                 // storage for results
         __global const double* time,                   // time points to save
         __global const double* param_values,           // parameter values
         __global const unsigned int* random_seed,              // seeds for RNG
         const unsigned int n_timepoints,               // number of time points
         const unsigned int max_sim                     // total number of simulations
         ){{

    const int tid = get_global_id(0) * get_global_size(1) + get_global_id(1);
    if (tid > max_sim){{
        return;
    }}

#ifdef VERBOSE
    if (tid==0){{
        printf("get_work_dim(0) %d\n", get_work_dim());
        printf("get_num_groups(0) %d\n", get_num_groups(0));
        printf("get_num_groups(1) %d\n", get_num_groups(1));
        printf("get_global_size(0) %d\n", get_global_size(0));
        printf("get_global_size(1) %d\n", get_global_size(1));
        printf("get_local_size(0) %d\n", get_local_size(0));
        printf("get_local_size(1) %d\n", get_local_size(1));
    }};
#endif

    double propensities[num_reaction] = {{0.0}};
    double param_vec[num_params] =  {{0.0}};

    // init parameters for thread
    unsigned int param_stride = tid*num_params;
    #pragma unroll num_params
    for(unsigned int i=0; i<num_params; i++){{
        param_vec[i] = param_values[param_stride + i];
        }}

    // init species state per thread
    int species_state[num_species];
    int prev_state[num_species];

    // spacing from global
    unsigned int species_stride = tid*num_species;

    #pragma unroll num_species
    for(unsigned int i=0; i<num_species; i++){{
        species_state[i] = species_matrix[species_stride + i];
        prev_state[i] = species_state[i];
        }}


    unsigned int index = tid * n_timepoints * num_species;

    // start time and time index for checking steps
    double t = time[0] ;
    unsigned int time_index = 0;

    // Arbitrary starting values from
    // https://stackoverflow.com/questions/11268023/random-numbers-with-opencl-using-random123
    key_t k = {{{{random_seed[tid], 0}}}};
    ctr_t c = {{{{0, 0}}}};
    output_vec_t rand_ns;

    // beginning of loop
    while (time_index < n_timepoints){{
        while (t < time[time_index]){{

            // create backup
            #pragma unroll num_species
            for(unsigned int j=0; j<num_species; j++){{
                prev_state[j] = species_state[j];
            }}

            // calculate propensities
            double a0 = calc_propensities(species_state, propensities, param_vec);

            if (a0 > (double)0.0){{
                rand_ns = GET_RANDOM_NUM(gen_bits(&k, &c));
                double r1 = rand_ns.s0;
                double r2 = rand_ns.s1;
                double tau = -log(r1)/a0; // find time of next reaction
                t += tau;  // update time

                double u = a0*r2;
                unsigned int rxn_k = sample(propensities, u);  // find next reaction
                stoichiometry(species_state, rxn_k); // update species matrix

#ifdef VERBOSE
                if (tid == 0){{ printf(" %d %f %f %f %f %f %d\n ", tid, a0, r1, r2, tau, t, rxn_k); }}
#endif
#ifdef VERBOSE_MAX
                printf(" %d %.17g %.17g %.17g %.17g %.17g %d\n ", tid, a0, r1, r2, tau, t, rxn_k);
#endif

            }}
            else{{
                t = time[n_timepoints-1];
                }}
            }}

        // add an entire species timepoint stride
        unsigned int current_index = index + time_index * num_species;

        // iterates through each species
        #pragma unroll num_species
        for(unsigned int j=0; j<num_species; j++){{
            result[current_index + j] = prev_state[j];
         }}
        time_index += 1;
        }}
}}

