/**


Implements tyche-i RNG.

S. Neves, F. Araujo, Fast and small nonlinear pseudorandom number generators
for computer simulation, in: International Conference on Parallel Processing
and Applied Mathematics, Springer, 2011, pp. 92–101.

Obtained from https://github.com/bstatcomp/RandomCL

BSD 3-Clause License

Copyright (c) 2018, Tadej Ciglarič, Erik Štrumbelj, Rok Češnovar.
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
 are permitted provided that the following conditions are met:

Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.

Redistributions in binary form must reproduce the above copyright notice,
 this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

Neither the name of the copyright holder nor the names of its contributors may
be used to endorse or promote products derived from this software without
specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
#pragma once

#define TYCHE_I_FLOAT_MULTI 5.4210108624275221700372640e-20f
#define TYCHE_I_DOUBLE_MULTI 5.4210108624275221700372640e-20

/**
State of tyche_i RNG.
*/
typedef union{
	struct{
		uint a,b,c,d;
	};
	ulong res;
} tyche_i_state;

#define TYCHE_I_ROT(a,b) (((a) >> (b)) | ((a) << (32 - (b))))
/**
Generates a random 64-bit unsigned integer using tyche_i RNG.

This is alternative, macro implementation of tyche_i RNG.

@param state State of the RNG to use.
*/
#define tyche_i_macro_ulong(state) (tyche_i_macro_advance(state), state.res)
#define tyche_i_macro_advance(state) ( \
	state.b = TYCHE_I_ROT(state.b, 7) ^ state.c, \
	state.c -= state.d, \
	state.d = TYCHE_I_ROT(state.d, 8) ^ state.a,\
	state.a -= state.b, \
	state.b = TYCHE_I_ROT(state.b, 12) ^ state.c, \
	state.c -= state.d, \
	state.d = TYCHE_I_ROT(state.d, 16) ^ state.a, \
	state.a -= state.b \
)

/**
Generates a random 64-bit unsigned integer using tyche_i RNG.

@param state State of the RNG to use.
*/
#define tyche_i_ulong(state) (tyche_i_advance(&state), state.res)
void tyche_i_advance(tyche_i_state* state){
	state->b = TYCHE_I_ROT(state->b, 7) ^ state->c;
	state->c -= state->d;
	state->d = TYCHE_I_ROT(state->d, 8) ^ state->a;
	state->a -= state->b;
	state->b = TYCHE_I_ROT(state->b, 12) ^ state->c;
	state->c -= state->d;
	state->d = TYCHE_I_ROT(state->d, 16) ^ state->a;
	state->a -= state->b;
}

/**
Seeds tyche_i RNG.

@param state Variable, that holds state of the generator to be seeded.
@param seed Value used for seeding. Should be randomly generated for each instance of generator (thread).
*/
void tyche_i_seed(tyche_i_state* state, ulong seed){
	state->a = seed >> 32;
	state->b = seed;
	state->c = 2654435769;
	state->d = 1367130551 ^ (get_global_id(0) + get_global_size(0) * (get_global_id(1) + get_global_size(1) * get_global_id(2)));
	for(uint i=0;i<20;i++){
		tyche_i_advance(state);
	}
}

/**
Generates a random 32-bit unsigned integer using tyche_i RNG.

@param state State of the RNG to use.
*/
#define tyche_i_uint(state) ((uint)tyche_i_ulong(state))

/**
Generates a random float using tyche_i RNG.

@param state State of the RNG to use.
*/
#define tyche_i_float(state) (tyche_i_ulong(state)*TYCHE_I_FLOAT_MULTI)

/**
Generates a random double using tyche_i RNG.

@param state State of the RNG to use.
*/
#define tyche_i_double(state) (tyche_i_ulong(state)*TYCHE_I_DOUBLE_MULTI)

/**
Generates a random double using tyche_i RNG. Since tyche_i returns 64-bit numbers this is equivalent to tyche_i_double.

@param state State of the RNG to use.
*/
#define tyche_i_double2(state) tyche_i_double(state)