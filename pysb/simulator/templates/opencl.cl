#include <pyopencl-random123/philox.cl>
#define num_species {n_species}
#define NPARAM {n_params}
#define NREACT {n_reactions}
#define NREACT_MIN_ONE NREACT-1

typedef double output_t;
typedef double4 output_vec_t;
typedef philox2x32_ctr_t ctr_t;
typedef philox2x32_key_t key_t;
/*
The code to generate random numbers is taken from pyopencl and based on
Random123 library.

Unlike cuda, there is not an easy to use library (curand) that allows one to
keep track of RNG across threads.

gen_bits

https://github.com/inducer/pyopencl/blob/master/pyopencl/clrandom.py

http://www.thesalmons.org/john/random123/releases/latest/docs/index.html
*/
#define GET_RANDOM_NUM(gen) ( ((double) 2.3283064365386963e-10) * convert_double4(gen) + ((double) 5.421010862427522e-20) * convert_double4(gen))

uint4 gen_bits(key_t *key, ctr_t *ctr)
{{
    union {{
        ctr_t ctr_el;
        uint4 vec_el;
    }} u;

    u.ctr_el = philox2x32(*ctr, *key);
    if (++ctr->v[0] == 0)
        if (++ctr->v[1] == 0)
            ++ctr->v[2];

    return u.vec_el;
}}


int constant stoch_matrix[]={{
{stoch}
}};

double sum_propensities(double *a){{
    double a0 = 0;
    for(unsigned int j=0; j<NREACT; j++){{
        a0 += a[j];
    }}
    return a0;
}}

double propensities(unsigned int *y, double *h, double *param_vec)
{{
{hazards}
return sum_propensities(h);
}}


void stoichiometry(unsigned int *y, unsigned int r){{
    unsigned int step = r*num_species;

    for(unsigned int i=0; i<num_species; i++){{
        y[i]+=stoch_matrix[step + i];
    }}

}}


long sample(double* a, double u){{
    unsigned int i = 0;
    for(;i < NREACT_MIN_ONE && u > a[i]; i++){{
        u -= a[i];
        }}
    return i;
}}



__kernel  void Gillespie_all_steps(
        __global const unsigned int* species_matrix,
         __global unsigned int* result,
         __global const double* time,
         __global const double* param_values,
         __global const long* random_seed,
         const long n_timepoints){{

    bool verbose = false;
    const int tid = get_global_id(0);
    // Arbitrary starting values from
    // https://stackoverflow.com/questions/11268023/random-numbers-with-opencl-using-random123
    key_t k = {{{{ random_seed[tid], 0xdecafbad}}}};
    ctr_t c = {{{{0, 0xdecafbad}}}};

    unsigned int y[num_species];
    unsigned int prev[num_species];
    double A[NREACT] = {{0.0}};
    double param_vec[NPARAM] =  {{0.0}};

    // init parameters for thread
    int param_stride = tid*NPARAM;

    for(unsigned int i=0; i<NPARAM; i++){{
        param_vec[i] = param_values[param_stride + i];
        }}

    // init species counter for thread
    unsigned int species_stride = tid*num_species;
    for(unsigned int i=0; i<num_species; i++){{
        y[i] = species_matrix[species_stride + i];
        prev[i] = y[i];
        }}

    double t = time[0] ;
    unsigned int time_index = 0;
    double a0;
    // beginning of loop
    while (time_index < n_timepoints){{
        while (t < time[time_index]){{
                    // calculate propensities
            a0 = propensities(y, A, param_vec);
            if (a0 <= 0.0){{
                t = time[n_timepoints-1];
                continue;
            }}
            for(unsigned int j=0; j<num_species; j++){{
                prev[j] = y[j];
                }}

            output_vec_t ran = GET_RANDOM_NUM(gen_bits(&k, &c));
            double r1 = ran.s0;
	        double r2 = ran.s1;
            double tau = -log(r1)/a0;  // find time of next reaction
            t += tau;  // update time

            unsigned int k = sample(A, a0*r2);  // find next reaction
            stoichiometry(y, k); // update species matrix
            }}

        // resets to correct start
        unsigned int index = tid * n_timepoints * num_species;
        // add an entire species timepoint stride
        index += time_index * num_species;
//        if (tid==1){{ printf("\n Time %f\t%i\n", t, index); }}
        // iterates through each species
        for(unsigned int j=0; j<num_species; j++){{
            result[index + j] = prev[j];
         }}
        time_index+=1;
        }}

    }}

