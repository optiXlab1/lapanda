#ifndef MATRIX_OPERATIONS_H
#define MATRIX_OPERATIONS_H

#include "../globals/globals.h"
#include "stddef.h"
#include "math.h"
 
#define sq(x) ((x)*(x))

void vector_copy(const real_t* vector1,real_t* vector2,const int size_vector);

void vector_real_mul(real_t* vector,const int size_vector,const real_t real);

real_t vector_norm2(const real_t* vector,const int vector_size);

real_t vector_norm1(const real_t* vector,const int vector_size);

real_t vector_norm_inf(const real_t* vector,const int vector_size);

real_t inner_product(const real_t* vector1,const real_t* vector2,const int size_vector);

void vector_add_ntimes(real_t* vector1,const real_t* vector2,const int size_vector,const real_t n);

void vector_add_2_vectors_a_times(const real_t* vector1,const real_t* vector2,const real_t* vector3,const int size_vector,
    const real_t a_vector2,const real_t a_vector3,real_t* result);

#endif