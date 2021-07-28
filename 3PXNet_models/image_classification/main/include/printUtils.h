#pragma once
#ifndef PRINT_UTILS_GUARD
#define PRINT_UTILS_GUARD

#include <stdio.h>

#include "dl_lib_matrix3d.h"
#include "dl_lib_matrix3dq.h"

void printMatrix(dl_matrix3d_t* mat);

void printMatrixQu(dl_matrix3dq_t* mat);

#endif