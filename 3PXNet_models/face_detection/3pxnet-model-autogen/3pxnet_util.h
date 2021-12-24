#include <stdio.h>
#include "datatypes.h"
#include "utils.h"
#include "xnor_base.h"

int pckDtype_to_int(pckDtype* input, int* output, int n, int pckWidth, int btwdt);
int pckDtype_to_int_var_bits(pckDtype* input, int* output, int n, int pckWidth, int btwdt, int bits);
int int_to_pckDtype(int* input, pckDtype* output, int n, int pckWidth, int btwdt);
int print_int_array(int* input, int n);

