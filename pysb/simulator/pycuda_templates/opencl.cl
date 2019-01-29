#include <pyopencl-random123/philox.cl>
#define num_species {n_species}
#define NPARAM {n_params}
#define NREACT {n_reactions}
#define NREACT_MIN_ONE NREACT-1

typedef double output_t;
typedef double4 output_vec_t;
typedef philox4x32_ctr_t ctr_t;
typedef philox4x32_key_t key_t;

#define GET_RANDOM_NUM(gen) ( ((double) 2.3283064365386963e-10) * convert_double4(gen)+ ((double) 5.421010862427522e-20) * convert_double4(gen))

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


int constant stoch_matrix[]={{
{stoch}
}};

double sum_propensities(double *a){{
    double a0 = 0;
    for(int j=0; j<NREACT; j++){{
        a0 += a[j];
    }}
    return a0;
}}

double propensities(unsigned int *y, double *h, double *param_vec)
{{
{hazards}
return sum_propensities(h);
}}


void stoichiometry(unsigned int *y, int r){{
    int step = r*num_species;

    for(int i=0; i<num_species; i++){{
        y[i]+=stoch_matrix[step + i];
    }}

}}


int sample(double* a, double u){{
    int i = 0;
    for(;i < NREACT_MIN_ONE && u > a[i]; i++){{
        u -= a[i];
        }}
    return i;
}}



__kernel  void Gillespie_all_steps(
        __global int* species_matrix,
         __global int* result,
         __global double* time,
         __global double* param_values,
         int n_timepoints){{


    int tid = get_global_id(0);
//    printf("%i", get_work_dim());
//    return;
//    return;
//    key_t k = {{{{tid, 0xdecafbad, 0xfacebead, 0x12345678}}}};
    key_t k = {{{{ tid, 0xdecafbad, 0xfacebead, 0x12345678}}}};
    ctr_t c = {{{{0, 0xdecafbad, 0xfacebead, 0x12345678}}}};

    int y[num_species];
    int prev[num_species];
    double A[NREACT] = {{0.0}};
    double param_vec[NPARAM] =  {{0.0}};

    int result_stepping = tid*n_timepoints*num_species;

    printf("Result stepping : %i %i\n", tid, result_stepping);
    // init parameters for thread
    int param_stride = tid*NPARAM;

    for(int i=0; i<NPARAM; i++){{
        param_vec[i] = param_values[param_stride + i];
        }}

    // init species counter for thread
    int species_stride = tid*num_species;
    for(int i=0; i<num_species; i++){{
        y[i] = species_matrix[species_stride + i];
        prev[i] = y[i];
        }}

    double t = time[0] ;
    int time_index = 0;
    // beginning of loop
    while (time_index < n_timepoints){{
        while (t < time[time_index]){{
                    // calculate propensities
            double a0 = propensities(y, A, param_vec);
//            if (a0 <= 0.0){{
//                t = time[NRESULTS-1];
//                continue;
//            }}
            if (tid==0){{
                printf("%i \t %f\t%f %f\n", time_index, t, time[time_index], a0);
            }}
            for(int j=0; j<num_species; j++){{
                prev[j] = y[j];
//                if (tid==0){{printf("%i\t", y[j]); }}
                }}
//              if (tid==0){{ printf("\n"); }}
//            output_vec_t ran = GET_RANDOM_NUM(gen_bits(&k, &c));
//            t +=  -log(ran.x)/a0;  // update time
//            stoichiometry(y, sample(A, a0*ran.y)); // update species matrix

            output_vec_t ran = GET_RANDOM_NUM(gen_bits(&k, &c));
            double r1 = ran.s0;
	        double r2 = ran.s1;


            double tau = -log(r1)/a0;  // find time of next reaction
            t += tau;  // update time

            int k = sample(A, a0*r2);  // find next reaction
//            if (tid==0){{ printf("reaction %i\ta0=%f\n", k, a0); }}
//            if (tid==0){{ printf("\t %f\t%f\t%f\t%i\n ", r1, r2, tau, k);}}
            stoichiometry(y, k); // update species matrix
//            for (int j=0; j<num_species; j++){{
//                if(prev[j] != y[j]+1 || prev[j] != y[j]-1 || prev[j]==y[j]){{
//                printf("%i %i %i\n", k, prev[j], y[j]);
//                return;
//                }}
//            }}
//            return;
            }}


        int index = tid * n_timepoints * num_species;
        for(int j=0; j<num_species; j++){{
            result[index + j + (time_index * num_species)] = prev[j];
         }}
        time_index+=1;
        }}

    }}

